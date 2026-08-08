# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Block-exposure orchestration (LRP-RLI-BX).

``fit_block_exposure`` relates outcome level to cumulative teaching-block
exposure, splitting confounders by measurement timing so pre-exposure covariates
and wave-varying covariates enter on the right footing.
"""

from __future__ import annotations

import pandas as pd

from language_reading_predictors.models._reporting import (
    metrics_table,
    print_table,
    ranked_dataframe_table,
    section_header,
)
from language_reading_predictors.statistical_models import (
    diagnostics as _diag,
    factories as _factories,
    reporting as _report,
)
from language_reading_predictors.statistical_models.artifacts import save_table
from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
    make_context,
)
from language_reading_predictors.statistical_models.factories import default_of
from language_reading_predictors.statistical_models.figure_artifacts import (
    save_association_forest,
    save_forest_plot,
)
from language_reading_predictors.statistical_models.preprocessing import (
    load_and_prepare,
    split_confounders_by_timing,
    split_covariates_by_wave,
)
from language_reading_predictors.statistical_models.prior_artifacts import (
    marginal_pushforward_rows,
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
    run_ppc,
    run_sampling_and_loo,
    write_run_metadata,
)


def _bx_coef_names(
    spec: ModelSpec, adjust_for: tuple[str, ...] | None = None
) -> list[str]:
    """Interpretable block-exposure coefficients (alpha/alpha_time/kappa/sigma_child excluded)."""
    extra = spec.extra
    adj = extra.get("adjust_for", ()) if adjust_for is None else adjust_for
    names = ["delta", "gamma_A"]
    if extra.get("ability_covariate"):
        names.append("gamma_ability")
    names += [f"gamma_{c}" for c in adj]
    return names


def _bx_diag_vars(
    spec: ModelSpec, adjust_for: tuple[str, ...] | None = None
) -> list[str]:
    off_floor = spec.extra.get("likelihood") == "bernoulli_offfloor"
    tail: list[str] = [] if off_floor else ["kappa"]
    if spec.extra.get("use_child_re", True):
        tail.append("sigma_child")
    return ["alpha", "alpha_time", *_bx_coef_names(spec, adjust_for), *tail]


def fit_block_exposure(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Block-2 taught-vocabulary staggered block-active exposure fit (LRPBX, #228 item 5).

    Data-load like ``fit_level_factors`` (per-timepoint levels frame + the revised-DAG
    adjuster wiring); effect readout like ``fit_did`` (the focal ``delta`` + its
    items-scale AME). ``delta`` is an association (parallel-trends), so the factor
    summary flags no causal term.
    """
    require_spec(spec, "block_exposure", outcome=True)
    ctx = make_context(spec, config)
    extra = spec.extra

    section_header("Prepare data")
    sym = spec.outcome_symbol
    ability_covariate = extra.get("ability_covariate")
    likelihood = extra.get("likelihood", "beta_binomial")
    off_floor = likelihood == "bernoulli_offfloor"
    obs_node = "y_offfloor" if off_floor else "y_post"
    baseline_covariates = (ability_covariate,) if ability_covariate else ()
    # Revised-DAG raw-covariate confounders, split by timing exactly as the
    # level-factor path (#247, A1 2026-07-13): language-proximal SP/RW (deapp_c/erbto)
    # read at the pre-randomisation baseline, hearing (hs) contemporaneous. Re-filter
    # after loading so a constant ``_missing`` indicator the loader drops is not gated.
    adjust_for = tuple(extra.get("adjust_for", ()))
    pre_adj, post_adj = split_covariates_by_wave(adjust_for)
    baseline_adj, post_adj = split_confounders_by_timing(post_adj)
    prepared = load_and_prepare(
        phase_mode="levels",
        outcomes=(sym,),
        baseline_covariates=(*baseline_covariates, *baseline_adj),
        covariates=pre_adj,
        post_covariates=post_adj,
        drop_ceiling_violations=tuple(extra.get("drop_ceiling_violations", ())),
    )
    adjust_for = tuple(c for c in adjust_for if c in prepared.covariates)
    ctx.prepared = prepared
    print_header(ctx)

    section_header("Build model")
    built = _factories.build_block_exposure_model(
        prepared,
        outcome_symbol=sym,
        ability_covariate=ability_covariate,
        adjust_for=adjust_for,
        use_child_re=bool(extra.get("use_child_re", True)),
        likelihood=likelihood,
        delta_prior_sigma=extra.get(
            "delta_prior_sigma",
            default_of(_factories.build_block_exposure_model, "delta_prior_sigma"),
        ),
    )
    attach_built(ctx, built)

    render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    _diag.save_prior_predictive_plot(ctx, sym, node=obs_node)

    run_sampling_and_loo(ctx)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=_bx_diag_vars(spec, adjust_for))

    run_ppc(ctx, var_names=[obs_node])

    section_header("Extended diagnostics")
    # ``delta`` is the focal (association) effect — gets the prior-sensitivity +
    # forest evidence, exactly as the level-factor group term does.
    _diag.write_diagnostics_summary(ctx, var_names=_bx_diag_vars(spec, adjust_for))
    _diag.run_extended_diagnostics(ctx, causal_term="delta")
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=_bx_diag_vars(spec, adjust_for))
    save_forest_plot(
        ctx, ["delta"], name="delta_forest.png",
        title="Block-active exposure effect (forest, reference line at 0)",
    )
    _diag.run_psense(ctx, var_names=["delta"])

    section_header("Factor summary")
    # No randomised contrast: block-active exposure is an association (parallel trends),
    # so no term is flagged causal.
    fs = _report.factor_summary(
        ctx.trace, _bx_coef_names(spec, adjust_for), ci_prob=ctx.reporting.ci_prob,
        causal_terms=(),
    )
    save_table(ctx, "factor_summary", fs)
    save_association_forest(ctx, _bx_coef_names(spec, adjust_for), ())
    print_table(
        ranked_dataframe_table(
            fs,
            title=f"Factor summary ({sym}) - {int(ctx.reporting.ci_prob * 100)}% CrI",
            columns=["term", "role", "median", "lo", "hi", "prob_positive"],
            rank_column=False,
            precision=3,
        )
    )

    section_header("Block-2 exposure effect summary")
    from language_reading_predictors.statistical_models.measures import MEASURES

    bx_s = _report.block_exposure_summary(
        ctx.trace,
        ci_prob=ctx.reporting.ci_prob,
        n_trials=1 if off_floor else MEASURES[sym].n_trials,
    )
    bx_df = pd.DataFrame([bx_s])
    save_table(ctx, "block_exposure_summary", bx_df)
    print_table(
        metrics_table(
            [{"metric": k, "value": v} for k, v in bx_s.items()],
            title=(
                f"block-2 exposure effect ({sym}) - "
                f"{int(ctx.reporting.ci_prob * 100)}% CI (equal-tailed); "
                "association (parallel-trends), positive = more taught-word learning"
            ),
            columns=["metric", "value"],
        )
    )
    # Estimand-scale prior check (#381), through the same transform
    # ``block_exposure_summary`` uses for ``delta_items_*``: a forward shift from
    # the un-exposed linear predictor ``eta_base``, not the ITT net-out core.
    write_prior_pushforward(
        ctx,
        marginal_pushforward_rows(
            ctx,
            [("delta", "the block-active exposure association")],
            n_trials=1 if off_floor else MEASURES[sym].n_trials,
            convention="forward",
            eta_name="eta_base",
        ),
    )

    write_run_metadata(
        ctx,
        extra={"loo_elpd": float(ctx.loo.elpd), "block_exposure_summary": bx_s},
    )
    return finalize_report(ctx)
