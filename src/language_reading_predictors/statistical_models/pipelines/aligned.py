# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Onset-aligned per-protocol orchestration (LRP-RLI-AL).

``fit_aligned`` aligns both arms by intervention onset — immediate t1→t3,
wait-list t2→t4 — into one cross-sectional Beta-Binomial ANCOVA per child, with
no random intercept. The cohort contrast is *not* randomised (it is confounded by
age-at-onset and cohort timing), so no term here is flagged causal; every
coefficient is an association, and dose, a collider, enters only the sensitivity
variant.
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
from language_reading_predictors.statistical_models.aligned import resolve_aligned_run_plan
from language_reading_predictors.statistical_models.artifacts import save_table
from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
    make_context,
)
from language_reading_predictors.statistical_models.fitted_payloads import (
    AlignedPayload,
)
from language_reading_predictors.statistical_models.figure_artifacts import (
    save_association_forest,
)
from language_reading_predictors.statistical_models.preprocessing import (
    load_and_prepare_aligned,
)
from language_reading_predictors.statistical_models.prior_artifacts import (
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


def fit_aligned(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    require_spec(spec, "aligned", outcome=True)
    # Resolve and validate the family contract before the context resets an output
    # directory or the loader reads any data (#394 pillar 4). One plan drives
    # preparation, factory arguments, the teaching recipe and config.json.
    plan = resolve_aligned_run_plan(spec)
    ctx = make_context(spec, config)
    ctx.resolved_plan = plan
    _report.write_model_recipe(ctx)

    off_floor = plan.off_floor
    obs_node = plan.obs_node

    section_header("Prepare data")
    prepared = load_and_prepare_aligned(**plan.prepare_kwargs())
    ctx.prepared = prepared
    print_header(ctx)

    section_header("Build model")
    built = _factories.build_aligned_model(prepared, **plan.factory_kwargs())
    attach_built(ctx, built)
    # The score-mean link the factory BUILT, not the one the module declared, so the
    # cohort marginal and its prior pushforward cannot drift from the likelihood
    # (#619).
    link = built.require_payload(
        AlignedPayload, family="aligned"
    ).score_mean_link

    render_model_graph(ctx)

    # Deterministic for a given spec — compute once and reuse across the diagnostics,
    # power-scaling, gate and prior/posterior overlay (PR #408 review).
    _al_vars = plan.diagnostic_vars()
    _al_coef_names = plan.coefficient_names()
    shared_stages().run_primary_fit(
        ctx,
        PrimaryFitPlan(
            diagnostic_vars=tuple(_al_vars),
            ppc_var_names=(obs_node,),
            plot_prior_predictive=lambda c: _diag.save_prior_predictive_plot(
                c, spec.outcome_symbol, node=obs_node
            ),
        ),
    )
    _diag.save_prior_posterior_plot(ctx, var_names=_al_vars)

    section_header("Factor summary")
    # Per-protocol design: NOTHING is a clean randomised effect, so no term is
    # flagged causal -- every coefficient (cohort included) is an association.
    fs = _report.factor_summary(
        ctx.trace, _al_coef_names, ci_prob=ctx.reporting.ci_prob, causal_terms=()
    )
    save_table(ctx, "factor_summary", fs)
    # Per-protocol: every term is an association, so the forest shows them all.
    save_association_forest(ctx, _al_coef_names, ())
    print_table(
        ranked_dataframe_table(
            fs,
            title=f"Factor summary ({spec.outcome_symbol}) - {int(ctx.reporting.ci_prob * 100)}% CrI",
            columns=["term", "role", "median", "lo", "hi", "prob_positive"],
            rank_column=False,
            precision=3,
        )
    )

    meta_extra = {"loo_elpd": float(ctx.loo.elpd)}
    # Items-scale cohort contrast (immediate vs wait-list at aligned endpoints).
    # This is a PER-PROTOCOL association, NOT a randomised treatment effect --
    # confounded by age-at-onset and cohort/timing (see the LRPAL design note).
    if plan.use_cohort:
        cohort = built.prepared.G.astype(float)
        n_marg = 1 if off_floor else built.prepared.n_trials[spec.outcome_symbol]
        cme = _report.treatment_marginal_effect(
            ctx.trace, trt=cohort, n_trials=n_marg, term="beta_cohort",
            ci_prob=ctx.reporting.ci_prob, score_mean_link=link,
        )
        save_table(ctx, "cohort_marginal", pd.DataFrame([cme]))
        meta_extra["cohort_marginal"] = cme
        print_table(
            metrics_table(
                [{"metric": k, "value": v} for k, v in cme.items()],
                title="Per-protocol cohort marginal (NOT randomised)",
                columns=["metric", "value"],
            )
        )
        # Estimand-scale prior check on the same contrast (#381). The cohort term
        # is binary, so this uses the toggle-everyone AME core rather than the
        # one-unit marginal — the same transform ``treatment_marginal_effect``
        # above runs on the posterior.
        try:
            pf = _report.prior_pushforward(
                ctx.prior_samples, G=cohort, n_trials=n_marg,
                term="beta_cohort", varying_term="", ci_prob=ctx.reporting.ci_prob,
                score_mean_link=link,
            )
            rows = [
                _report.labelled_pushforward(
                    pf,
                    estimand="beta_cohort",
                    estimand_label=(
                        "the per-protocol cohort contrast (an association, not a "
                        "randomised effect)"
                    ),
                    role="association",
                )
            ]
        except Exception as exc:  # noqa: BLE001 - absence must stay legible
            rows = [
                _report.unavailable_pushforward(
                    estimand="beta_cohort",
                    estimand_label="the per-protocol cohort contrast",
                    role="association",
                    reason=str(exc),
                )
            ]
        write_prior_pushforward(ctx, rows)
    else:
        write_prior_pushforward(
            ctx,
            [
                _report.unavailable_pushforward(
                    estimand="beta_cohort",
                    estimand_label="the per-protocol cohort contrast",
                    role="association",
                    reason=(
                        "this aligned variant fits a single pooled cohort, so it "
                        "carries no cohort contrast to push a prior through"
                    ),
                )
            ],
        )

    write_run_metadata(ctx, extra=meta_extra)
    return finalize_report(ctx)
