# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regularised-horseshoe predictor-ranking orchestration (``kind="horseshoe"``).

``fit_horseshoe`` refits the construct predictor set under a regularised horseshoe
prior and ranks predictors by posterior ``P(|beta| > delta)``, as an independent
Bayesian cross-check on the gradient-boosting ranking. ``fit_rlm_horseshoe`` is the
Byrne (RLM) port; there is no gradient-boosting layer for that cohort, so its
cross-check partner is the Byrne adjusted fit rather than a GB comparison table.
Both ports read the RLM span frame through :func:`.adjusted.rlm_nuisance_names`.

A horseshoe ranking is an association ranking. It says which predictors carry
signal once the others are in the model, not which of them would change the
outcome if intervened on.
"""

from __future__ import annotations

from language_reading_predictors.models._reporting import (
    print_table,
    ranked_dataframe_table,
    section_header,
)
from language_reading_predictors.statistical_models import (
    diagnostics as _diag,
    factories as _factories,
    horseshoe as _horseshoe,
    reporting as _report,
)
from language_reading_predictors.statistical_models.artifacts import save_table
from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
    make_context,
)
from language_reading_predictors.statistical_models.pipelines.adjusted import (
    rlm_nuisance_names,
)
from language_reading_predictors.statistical_models.preprocessing import (
    load_and_prepare,
)
from language_reading_predictors.statistical_models.prior_artifacts import (
    horseshoe_pushforward_rows,
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


def fit_horseshoe(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Regularized-horseshoe predictor-ranking fit (LRPHS, #116 Phase E).

    An independent Bayesian sensitivity cross-check on the gradient-boosting
    predictor ranking: one horseshoe regression (gain or level, per the resolved plan)
    over the full construct predictor set, ranked by posterior
    ``P(|beta| > delta)``. Writes ``predictor_ranking.csv`` alongside the standard
    trace / diagnostics / LOO / PPC artefacts. Not causal — a which-predictors
    -carry-signal read to compare against the GB cluster ranking.
    """
    require_spec(spec, "horseshoe")
    plan = _horseshoe.resolve_horseshoe_run_plan(spec)
    if plan.port != "rli":
        raise ValueError(f"{spec.model_id}: fit_horseshoe requires the RLI port")

    # 94% intervals, matching the LRP65 adjusted-model convention.
    ctx = make_context(spec, config, ci_prob=0.89)
    ctx.resolved_plan = plan
    _report.write_model_recipe(ctx)
    # The horseshoe has a funnel geometry (global-local scales); lift target_accept
    # above the tier default so the sampler takes smaller steps near the neck.

    section_header("Prepare data")
    prepared = load_and_prepare(**plan.rli_prepare_kwargs())
    ctx.prepared = prepared
    print_header(ctx)

    section_header("Build model")
    built = _factories.build_horseshoe_model(
        prepared,
        **plan.rli_factory_kwargs(),
    )
    attach_built(ctx, built)
    render_model_graph(ctx)

    # Coupling term present in the model: gamma_own (gain) or the fixed age slope
    # gamma_A (level) — but the level model suppresses gamma_A when age is itself a
    # horseshoe-ranked predictor (build_horseshoe_model), so only list it then.
    diag_vars = plan.diagnostic_vars()
    shared_stages().run_primary_fit(
        ctx,
        PrimaryFitPlan(
            diagnostic_vars=tuple(diag_vars),
            ppc_var_names=(plan.observation_node,),
            plot_prior_predictive=lambda c: _diag.save_prior_predictive_plot(
                c, plan.outcome_symbol, node=plan.observation_node
            ),
            extended_term=plan.focal_term,
            compute_loo=plan.compute_loo,
        ),
    )
    _diag.save_prior_posterior_plot(ctx, var_names=diag_vars)

    section_header("Predictor ranking")
    ranking = _report.horseshoe_ranking(ctx.trace, delta=plan.delta)
    save_table(ctx, "predictor_ranking", ranking)
    print_table(ranked_dataframe_table(ranking.head(10), title="Horseshoe predictor ranking (top 10)"))
    write_prior_pushforward(
        ctx,
        horseshoe_pushforward_rows(
            ctx,
            list(plan.predictors),
            plan.outcome_symbol,
        ),
    )

    meta_extra = {
        "framing": "gain" if plan.gain else "level",
        "phase_mode": plan.phase_mode,
        "predictors": list(plan.predictors),
        "covariates": list(plan.covariates),
        "delta": plan.delta,
        "tau0": plan.tau0,
        "slab_scale": plan.slab_scale,
        "slab_df": plan.slab_df,
        "gb_reference": plan.gb_reference,
        "ranking_top": ranking.head(3)[["predictor", "p_abs_gt_delta"]].to_dict(
            "records"
        ),
    }
    write_run_metadata(ctx, extra=meta_extra)

    return finalize_report(ctx)


def fit_rlm_horseshoe(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Byrne horseshoe predictor-ranking fit (#338 Phase D, ``lrp-rlm-hs-001``).

    The RLI gain-framing ``fit_horseshoe`` on the Byrne span frame: one
    regularised-horseshoe regression over the wave-1 predictor set (age
    included), ranked by posterior ``P(|beta| > delta)``. Writes
    ``predictor_ranking.csv`` so the shared ``horseshoe`` partial and
    key-findings builder apply unchanged. There is no gradient-boosting layer
    for the Byrne cohort, so no ``horseshoe_vs_gb.csv`` comparison is written -
    the cross-check partner here is ``lrp-rlm-adj-001``.
    """
    from language_reading_predictors.statistical_models.preprocessing import (
        load_rlm_span_frame,
    )

    require_spec(spec, "horseshoe")
    plan = _horseshoe.resolve_horseshoe_run_plan(spec)
    if plan.port != "rlm":
        raise ValueError(f"{spec.model_id}: fit_rlm_horseshoe requires the RLM port")

    ctx = make_context(spec, config, ci_prob=0.89)
    ctx.resolved_plan = plan
    _report.write_model_recipe(ctx)

    section_header("Prepare data")
    frame = load_rlm_span_frame(**plan.rlm_prepare_kwargs())
    ctx.prepared = frame
    predictors = list(frame.predictors)
    print_header(ctx)

    section_header("Build model")
    built = _factories.build_rlm_horseshoe_model(
        frame,
        **plan.rlm_factory_kwargs(predictors=predictors),
    )
    attach_built(ctx, built)
    render_model_graph(ctx)

    nuisance = rlm_nuisance_names(frame)
    diag_vars = plan.diagnostic_vars(nuisance=tuple(nuisance))
    shared_stages().run_primary_fit(
        ctx,
        PrimaryFitPlan(
            diagnostic_vars=tuple(diag_vars),
            ppc_var_names=(plan.observation_node,),
            plot_prior_predictive=lambda c: _diag.save_prior_predictive_plot(
                c, plan.outcome_symbol, node=plan.observation_node
            ),
            extended_term=plan.focal_term,
            compute_loo=plan.compute_loo,
        ),
    )
    _diag.save_prior_posterior_plot(ctx, var_names=diag_vars)

    section_header("Predictor ranking")
    ranking = _report.horseshoe_ranking(ctx.trace, delta=plan.delta)
    save_table(ctx, "predictor_ranking", ranking)
    print_table(
        ranked_dataframe_table(ranking, title="Horseshoe predictor ranking")
    )
    write_prior_pushforward(
        ctx,
        horseshoe_pushforward_rows(ctx, predictors, plan.outcome_symbol),
    )

    write_run_metadata(
        ctx,
        extra={
            "study_id": "rlm",
            "framing": "gain",
            "outcome": plan.outcome_symbol,
            "pre_wave": plan.pre_wave,
            "post_wave": plan.post_wave,
            "predictors": predictors,
            "group_nuisance_terms": nuisance,
            "delta": plan.delta,
            "tau0": plan.tau0,
            "slab_scale": plan.slab_scale,
            "slab_df": plan.slab_df,
            "ranking_top": ranking.head(3)[["predictor", "p_abs_gt_delta"]].to_dict(
                "records"
            ),
        },
    )
    return finalize_report(ctx)
