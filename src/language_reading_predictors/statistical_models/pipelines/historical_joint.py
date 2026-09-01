# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Byrne joint correlated-growth orchestration (``kind="historical_joint"``, #338).

``fit_rlm_joint_growth`` fits a small measure set jointly and reports the
between-child cross-measure correlation matrix of the stable child levels. The
within-child companion also reports the correlation matrix of wave-specific
departures from those levels and their matched contrast. Per-measure fitted
cells and common-window growth use the shared historical summaries. LOO is not
computed because no out-of-sample prediction target has been defined and
implemented for this family - not because several likelihood nodes preclude one;
they share an observation coordinate and could be summed per child-wave row
(2026-08-23 joint audit, finding 8). ``plan.loo_reason`` is the statement of
record. Descriptive throughout; the cohort is observational.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from language_reading_predictors.models._reporting import (
    print_table,
    ranked_dataframe_table,
    section_header,
)
from language_reading_predictors.statistical_models import (
    datasets as _datasets,
    diagnostics as _diag,
    factories as _factories,
    historical as _historical,
    reporting as _report,
)
from language_reading_predictors.statistical_models.artifacts import save_table
from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
    make_context,
)
from language_reading_predictors.statistical_models.historical_joint import (
    resolve_historical_joint_run_plan,
)
from language_reading_predictors.statistical_models.new_child_kfold import (
    subset_panel_children,
    write_child_kfold,
)
from language_reading_predictors.statistical_models.preprocessing import (
    load_longitudinal_panel,
)
from language_reading_predictors.statistical_models.prior_artifacts import (
    growth_contrast_pushforward_rows,
    write_prior_pushforward,
)
from language_reading_predictors.statistical_models.publication import (
    print_header,
    render_model_graph,
)
from language_reading_predictors.statistical_models.runtime import (
    attach_built,
    finalize_report,
    require_spec,
    shared_stages,
    write_run_metadata,
)
from language_reading_predictors.statistical_models.stages import PrimaryFitPlan


# Correlation is not interpretable when either latent residual variance collapses
# to zero. A 0.05-logit SD moves a probability by at most 1.25 percentage points
# (at p=0.5), so this is a deliberately small practical-identifiability threshold,
# not a minimum scientifically important coupling. Require 95% posterior support
# above it for both measures before headlining their correlation.
_MIN_RESOLVABLE_WITHIN_SD = 0.05
_MIN_RESOLVABLE_PROB = 0.95


def fit_rlm_joint_growth(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Byrne joint correlated growth fits (#338 Phase B; #409 C2(ii)).

    Fits :func:`factories.build_rlm_joint_growth_model` over a small measure set
    and reports the between-child cross-measure correlation matrix of the
    stable child levels (the headline), plus per-measure fitted cells and
    common-window growth via the shared historical summaries. LOO is not computed
    because this family has no defined and implemented prediction target; see
    ``plan.loo_reason``, which the report renders verbatim.
    """
    require_spec(spec, "historical_joint")

    # Resolve every family setting before ``make_context`` starts an output
    # transaction or the panel loader reads study data (#394 pillar 4).
    plan = resolve_historical_joint_run_plan(spec)
    ctx = make_context(spec, config)
    ctx.resolved_plan = plan
    _report.write_model_recipe(ctx)

    study_id = plan.study_id
    measure_syms = plan.measures

    section_header("Prepare data")
    dataset, measures = _datasets.resolve_dataset(study_id)
    panel = load_longitudinal_panel(
        dataset,
        [measures[m] for m in measure_syms],
        **plan.prepare_kwargs(),
    )
    ctx.prepared = panel
    print_header(ctx)

    section_header("Build model")
    built = _factories.build_rlm_joint_growth_model(
        panel,
        **plan.factory_kwargs(),
    )
    attach_built(ctx, built)
    render_model_graph(ctx)

    # One likelihood node per measure, so emit one check per measure rather than a
    # pooled overlay: these scales have different maxima and pooling their counts has
    # no interpretable predictive distribution (same reasoning as the joint family's
    # symbol-suffixed checks).
    diag_vars = plan.diagnostic_vars()
    # Power-scaling prior sensitivity on the reported parameters (#381). This family
    # is ``compute_loo=False`` — not because several likelihood nodes make a
    # pointwise unit undefined (they share an observation coordinate and could be
    # summed per row) but because no prediction target has been defined and
    # implemented; see ``plan.loo_reason`` (2026-08-23 joint audit, finding 8). The
    # groups psense needs are therefore not attached by the sampling stage and have
    # to be requested here. ``strict=False`` because psense is a secondary diagnostic and must not
    # crash a fit. An earlier comment here recorded both groups as refused by the
    # ``measure_corr_chol_cholesky`` naming seam in ``get_untransformed_name``
    # (notes/202607261700-psense-coverage-backfill.md); that is stale —
    # ``psense_summary.csv`` is written and populated for both registered fits,
    # including the ``measure_corr_pairs`` / ``within_corr_pairs`` headline rows
    # (2026-08-21 historical-families review, finding 9). ``strict=False`` stays as
    # the guard it always was, not as a declaration that psense is unavailable.
    def _plot_prior_predictive(c: StatisticalFitContext) -> None:
        for symbol, node in zip(measure_syms, plan.observation_nodes, strict=True):
            _diag.save_prior_predictive_plot(
                c,
                symbol,
                node=node,
                filename_stem=f"prior_predictive_check_{symbol.lower()}",
            )

    def _validate_new_child(c: StatisticalFitContext) -> None:
        """Grouped child-level K-fold against the declared new-child target (#626).

        Each fold rebuilds the panel with its training children only and refits
        through this family's own factory, so the refit is the same model. The
        held-out children are then scored with their latent departures drawn from
        the population, which is what an unseen child means here.
        """
        write_child_kfold(
            c,
            plan.new_child_plan(),
            plan.kfold_plan(),
            # This family's builder reads the panel as given, so a fold can simply
            # drop the held-out children's rows.
            lambda training, _held_out: _factories.build_rlm_joint_growth_model(
                subset_panel_children(panel, training),
                **plan.factory_kwargs(),
            ),
        )

    shared_stages().run_primary_fit(
        ctx,
        PrimaryFitPlan(
            diagnostic_vars=tuple(diag_vars),
            ppc_var_names=plan.observation_nodes,
            after_trace_audit=_validate_new_child,
            plot_prior_predictive=_plot_prior_predictive,
            prepare_psense=lambda c: _diag.compute_log_likelihood_and_prior(
                c, strict=False
            ),
            compute_loo=plan.compute_loo,
            # LOO-PIT is a pointwise PSIS-LOO quantity, and this family does not
            # compute PSIS-LOO: its declared target is a new child, whose
            # leave-one-child-out importance ratios are far too heavy-tailed to
            # smooth (#626). The log-likelihood group exists only as a side effect
            # of preparing power scaling, so without this the report used to
            # publish a LOO-PIT calibration figure — with its reading guidance —
            # for a model whose own results section says there is no PSIS-LOO, and
            # with no Pareto-k companion to say whether the importance weights
            # behind it were reliable (2026-08-21 review, finding 9). The
            # calibration diagnostic the family *does* publish now comes from the
            # K-fold refits above, where there are no importance weights at all.
            include_loo_pit=False,
        ),
    )
    _diag.save_prior_posterior_plot(ctx, var_names=diag_vars)

    hdi = ctx.reporting.ci_prob
    lo_q = (1.0 - hdi) / 2.0
    post = ctx.trace.posterior

    # --- Cross-measure correlation of stable child levels (the headline) ----
    section_header("Cross-measure correlation")
    corr_draws = post["measure_corr"]
    mnames = [str(m) for m in post["measure"].values]
    corr_df = pd.DataFrame(
        corr_draws.mean(dim=("chain", "draw")).values, index=mnames, columns=mnames
    )
    save_table(ctx, "measure_correlation", corr_df, index=True)
    corr_stacked = corr_draws.stack(sample=("chain", "draw"))
    labels = {
        m: str(measures[m].label) if m in measures else m for m in mnames
    }
    corr_rows = []
    for i, mi in enumerate(mnames):
        for j, mj in enumerate(mnames):
            if j <= i:
                continue
            pair = np.asarray(
                corr_stacked.isel(measure=i, measure_b=j).values
            ).reshape(-1)
            corr_rows.append(
                {
                    "measure_i": mi,
                    "measure_j": mj,
                    "label_i": labels[mi],
                    "label_j": labels[mj],
                    "median": float(np.median(pair)),
                    "mean": float(np.mean(pair)),
                    "lo": float(np.quantile(pair, lo_q)),
                    "hi": float(np.quantile(pair, 1 - lo_q)),
                    "lo50": float(np.quantile(pair, 0.25)),
                    "hi50": float(np.quantile(pair, 0.75)),
                    "prob_pos": float(np.mean(pair > 0)),
                }
            )
    corr_summary_df = pd.DataFrame(corr_rows)
    save_table(ctx, "measure_correlation_summary", corr_summary_df)
    print_table(
        ranked_dataframe_table(
            corr_summary_df,
            title=f"Between-child cross-measure correlations - {int(hdi * 100)}% CI",
            # Median, matching the house convention and the within-child table
            # printed beside it. ``mean`` is kept in the CSV only for the printed
            # correlation *matrix*, where entrywise averaging preserves positive
            # semidefiniteness and entrywise medians would not (2026-08-24
            # historical-joint review).
            columns=["label_i", "label_j", "median", "lo", "hi", "prob_pos"],
            rank_column=False,
            precision=3,
        )
    )

    if plan.within_correlation:
        # --- Wave-specific correlation after stable child levels (jc-002) ---
        section_header("Within-child cross-measure correlation")
        within_draws = post["within_corr"]
        within_df = pd.DataFrame(
            within_draws.mean(dim=("chain", "draw")).values,
            index=mnames,
            columns=mnames,
        )
        save_table(ctx, "within_measure_correlation", within_df, index=True)
        within_stacked = within_draws.stack(sample=("chain", "draw"))
        scale_stacked = post["sigma_within"].stack(sample=("chain", "draw"))
        # ``sigma_within`` is the scale of the latent deviation BEFORE the double
        # sum-to-zero sweep, so it exceeds the spread of the departures the linear
        # predictor actually carries — by the projection factor
        # sqrt(1 - (n_subjects + G*T - G) / (n_subjects*T)), about 0.8 on the
        # balanced three-wave panel. Publish the realised SD beside it rather than
        # letting the fitted parameter stand in for it (2026-08-21 review,
        # finding 6). Measured from the fit's own draws, not derived from a
        # formula, so a different panel shape cannot make it wrong.
        realised = post.get("within_offset")
        realised_sd = (
            realised.std(dim="obs").stack(sample=("chain", "draw"))
            if realised is not None
            else None
        )
        scale_rows = []
        scale_resolvable: dict[str, bool] = {}
        for i, measure in enumerate(mnames):
            values = np.asarray(scale_stacked.isel(measure=i).values).reshape(-1)
            prob_gt = float(np.mean(values > _MIN_RESOLVABLE_WITHIN_SD))
            resolved = prob_gt >= _MIN_RESOLVABLE_PROB
            scale_resolvable[measure] = resolved
            row = {
                "measure": measure,
                "label": labels[measure],
                "median": float(np.median(values)),
                "mean": float(np.mean(values)),
                "lo": float(np.quantile(values, lo_q)),
                "hi": float(np.quantile(values, 1 - lo_q)),
                "lo50": float(np.quantile(values, 0.25)),
                "hi50": float(np.quantile(values, 0.75)),
                "minimum_resolvable_sd": _MIN_RESOLVABLE_WITHIN_SD,
                "prob_above_minimum": prob_gt,
                "resolvable": resolved,
            }
            if realised_sd is not None:
                observed = np.asarray(
                    realised_sd.isel(measure=i).values
                ).reshape(-1)
                # The rule is applied to ``sigma_within``, but its justification is
                # about the departures the linear predictor actually carries, which
                # the double sum-to-zero sweep makes smaller. Publish the same
                # probability on that scale so the rule's leniency is measured
                # rather than left for a reader to infer (2026-08-24
                # historical-joint review). The classification itself is unchanged:
                # it stays on the latent scale the correlation belongs to.
                row["realised_prob_above_minimum"] = float(
                    np.mean(observed > _MIN_RESOLVABLE_WITHIN_SD)
                )
                row["realised_departure_sd_median"] = float(np.median(observed))
                row["realised_departure_sd_lo"] = float(
                    np.quantile(observed, lo_q)
                )
                row["realised_departure_sd_hi"] = float(
                    np.quantile(observed, 1 - lo_q)
                )
            scale_rows.append(row)
        scale_summary_df = pd.DataFrame(scale_rows)
        save_table(ctx, "within_scale_summary", scale_summary_df)
        within_rows = []
        comparison_rows = []
        for i, mi in enumerate(mnames):
            for j, mj in enumerate(mnames):
                if j <= i:
                    continue
                between_pair = np.asarray(
                    corr_stacked.isel(measure=i, measure_b=j).values
                ).reshape(-1)
                within_pair = np.asarray(
                    within_stacked.isel(measure=i, measure_b=j).values
                ).reshape(-1)
                difference = within_pair - between_pair
                within_rows.append(
                    {
                        "measure_i": mi,
                        "measure_j": mj,
                        "label_i": labels[mi],
                        "label_j": labels[mj],
                        "median": float(np.median(within_pair)),
                        "mean": float(np.mean(within_pair)),
                        "lo": float(np.quantile(within_pair, lo_q)),
                        "hi": float(np.quantile(within_pair, 1 - lo_q)),
                        "lo50": float(np.quantile(within_pair, 0.25)),
                        "hi50": float(np.quantile(within_pair, 0.75)),
                        "prob_pos": float(np.mean(within_pair > 0)),
                        "scale_i_resolvable": scale_resolvable[mi],
                        "scale_j_resolvable": scale_resolvable[mj],
                        "pair_resolvable": (
                            scale_resolvable[mi] and scale_resolvable[mj]
                        ),
                    }
                )
                comparison_rows.append(
                    {
                        "measure_i": mi,
                        "measure_j": mj,
                        "label_i": labels[mi],
                        "label_j": labels[mj],
                        "between_median": float(np.median(between_pair)),
                        "between_lo": float(np.quantile(between_pair, lo_q)),
                        "between_hi": float(
                            np.quantile(between_pair, 1 - lo_q)
                        ),
                        "between_lo50": float(
                            np.quantile(between_pair, 0.25)
                        ),
                        "between_hi50": float(
                            np.quantile(between_pair, 0.75)
                        ),
                        "within_median": float(np.median(within_pair)),
                        "within_lo": float(np.quantile(within_pair, lo_q)),
                        "within_hi": float(np.quantile(within_pair, 1 - lo_q)),
                        "within_lo50": float(np.quantile(within_pair, 0.25)),
                        "within_hi50": float(np.quantile(within_pair, 0.75)),
                        "within_minus_between_median": float(
                            np.median(difference)
                        ),
                        "within_minus_between_lo": float(
                            np.quantile(difference, lo_q)
                        ),
                        "within_minus_between_hi": float(
                            np.quantile(difference, 1 - lo_q)
                        ),
                        "within_minus_between_lo50": float(
                            np.quantile(difference, 0.25)
                        ),
                        "within_minus_between_hi50": float(
                            np.quantile(difference, 0.75)
                        ),
                        "prob_within_gt_between": float(
                            np.mean(difference > 0)
                        ),
                        "pair_resolvable": (
                            scale_resolvable[mi] and scale_resolvable[mj]
                        ),
                    }
                )
        within_summary_df = pd.DataFrame(within_rows)
        comparison_df = pd.DataFrame(comparison_rows)
        save_table(
            ctx, "within_measure_correlation_summary", within_summary_df
        )
        save_table(
            ctx, "between_within_correlation_comparison", comparison_df
        )
        print_table(
            ranked_dataframe_table(
                scale_summary_df,
                title="Within-child residual-scale identifiability",
                columns=[
                    "label",
                    "median",
                    "lo",
                    "hi",
                    "prob_above_minimum",
                    "resolvable",
                ],
                rank_column=False,
                precision=3,
            )
        )
        print_table(
            ranked_dataframe_table(
                within_summary_df,
                title=(
                    "Within-child cross-measure correlations - "
                    f"{int(hdi * 100)}% CI"
                ),
                columns=[
                    "label_i",
                    "label_j",
                    "median",
                    "lo",
                    "hi",
                    "prob_pos",
                ],
                rank_column=False,
                precision=3,
            )
        )

    # --- Per-measure fitted cells + growth (shared historical summaries) ----
    section_header("Per-measure growth summaries")
    # One estimand-scale prior row per measure per contrast (#381), accumulated
    # across the loop so the joint fit writes a single table rather than each
    # measure overwriting the last.
    _pf_rows: list[dict[str, object]] = []
    for m in measure_syms:
        label = measures[m].label
        baseline = _historical.observed_baseline(panel, m, label)
        save_table(
            ctx, f"observed_complete_case_baseline_{m}", baseline, register=False
        )
        cells = _historical.cell_summary(
            ctx.trace,
            panel,
            m,
            label,
            baseline,
            mean_var=f"mean_items_{m}",
            fitted_var=f"fitted_mean_items_obs_{m}",
        )
        save_table(ctx, f"posterior_cell_summary_{m}", cells, register=False)
        growth = _historical.growth_summary(
            ctx.trace, panel, m, fitted_var=f"fitted_mean_items_obs_{m}"
        )
        save_table(ctx, f"posterior_growth_summary_{m}", growth)
        _pf_rows.extend(
            growth_contrast_pushforward_rows(
                ctx,
                panel,
                m,
                fitted_var=f"fitted_mean_items_obs_{m}",
                prefix=f"{m}:",
            )
        )
    write_prior_pushforward(ctx, _pf_rows)

    write_run_metadata(
        ctx,
        extra={
            "study_id": study_id,
            "measures": list(measure_syms),
            "measure_labels": {m: measures[m].label for m in measure_syms},
            "waves": list(plan.waves),
            "extension_waves": list(plan.extension_waves),
            "within_correlation": plan.within_correlation,
            "n_subjects": panel.n_subjects,
            "loo_elpd": None,
        },
    )
    return finalize_report(ctx)
