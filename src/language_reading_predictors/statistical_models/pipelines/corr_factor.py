# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Correlated-domain-factor measurement orchestration (``kind="corr_factor"``, #134).

``fit_correlated_factor`` fits a reflective CFA with correlated vocabulary, code
and grammar factors over the standardised T1 skill indicators, plus a structural
Beta-Binomial leg for the reading-gain outcome, and reports the loadings and
communalities, the factor correlation matrix and the measurement-error-corrected
factor→gain slopes. ``fit_rlm_corr_factor`` is the Byrne port: measurement-only
by the 2026-07-16 sign-off, no structural leg, publishing the same three CSVs so
the shared partial and key-findings builder apply unchanged.

A triangulation / measurement family: every factor→gain slope is a
latent-ability-confounded adjusted association, never causal (#115 ID-2). LOO is
not computed for either fit.
"""

from __future__ import annotations

import pandas as pd
from rich import print as rprint

from language_reading_predictors.models._reporting import (
    print_table,
    ranked_dataframe_table,
    section_header,
)
from language_reading_predictors.statistical_models import (
    corr_factor as _corr_factor,
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
from language_reading_predictors.statistical_models.preprocessing import (
    load_and_prepare,
)
from language_reading_predictors.statistical_models.prior_artifacts import (
    write_indicator_prior_check,
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


def fit_correlated_factor(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Correlated-domain-factor measurement model (LRPMM01, #134).

    Fits a reflective CFA with correlated vocabulary / code / grammar factors over
    the standardised T1 skill indicators, plus a structural Beta-Binomial leg for
    the reading-gain outcome, and reports the loadings / communalities, the factor
    correlation matrix, and the measurement-error-corrected factor->gain slopes.
    A triangulation / measurement model, not causal (every factor->gain slope is a
    latent-ability-confounded adjusted association; #115 ID-2).
    """
    require_spec(spec, "corr_factor")
    plan = _corr_factor.resolve_corr_factor_run_plan(spec)
    ctx = make_context(spec, config)
    ctx.resolved_plan = plan
    _report.write_model_recipe(ctx)
    # The correlated-factor CFA is a small-n latent model; even with the factor
    # scores marginalised out of the measurement likelihood a few boundary
    # divergences survive at the tier-default target_accept, so lift it via the spec
    # (the strict gate requires zero), as the horseshoe fit does for its funnel.

    section_header("Prepare data")
    domains = plan.domain_mapping()
    outcome = plan.outcome_symbol
    assert outcome is not None
    structural_covs = plan.structural_covariates
    prepared = load_and_prepare(**plan.rli_prepare_kwargs())
    ctx.prepared = prepared
    # A structural covariate can go constant on the fitted span rows — e.g. an
    # ``erbto_missing`` indicator that is all-zero because phonological memory is
    # observed for every fitted child at t1 — so the loader drops it. Re-filter to the
    # covariates actually present, mirroring the mechanism/mediation pipelines'
    # #247/#258 re-filter, so the factory is not asked for a coefficient on a dropped
    # covariate (it raises KeyError otherwise) and the effective set is honest.
    _dropped_structural = tuple(c for c in structural_covs if c not in prepared.covariates)
    if _dropped_structural:
        structural_covs = tuple(c for c in structural_covs if c in prepared.covariates)
        plan = plan.with_active_structural_covariates(structural_covs)
        ctx.resolved_plan = plan
        _report.write_model_recipe(ctx)
        rprint(
            "[yellow]fit_correlated_factor: dropped constant structural covariate(s) "
            f"{list(_dropped_structural)} (not in prepared.covariates on the fitted "
            "rows).[/yellow]"
        )
    print_header(ctx)

    section_header("Build model")
    built = _factories.build_correlated_factor_model(
        prepared,
        **plan.rli_factory_kwargs(),
    )
    attach_built(ctx, built)
    render_model_graph(ctx)

    summary_vars = plan.diagnostic_vars()

    section_header("Prior predictive")
    # Draw the full prior, not just the two observed nodes (#381). Restricting
    # ``var_names`` to ``["Z_obs", "y_post"]`` left the persisted ``prior`` group
    # completely empty, so ``save_prior_posterior_plot`` below had nothing to
    # overlay and these three fits shipped with no prior-vs-posterior figure at
    # all — the one measurement family the prior-analysis review most wanted to
    # see. The default (all free RVs + deterministics + observed nodes) is what
    # every other family uses, and ``run_prior_predictive`` falls back to the
    # minimal set on failure.
    _diag.run_prior_predictive(ctx, draws=1000)
    _diag.save_prior_predictive_plot(ctx, outcome, node="y_post")

    # Two observed nodes (the indicator matrix Z_obs + the structural y_post) make
    # a single-target PSIS-LOO ambiguous, so LOO is skipped here as in the
    # mediation family; this is a measurement / triangulation model, not a
    # predictive one, and #134 turns on the loadings / communalities, not on LOO.
    run_sampling_and_loo(ctx, compute_loo=plan.compute_loo)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=summary_vars)
    # Power-scaling prior sensitivity (#381): this family does not compute LOO,
    # so add the log groups explicitly, then power-scale the reported parameters.
    _diag.compute_log_likelihood_and_prior(ctx, strict=False)
    _diag.run_psense(ctx, var_names=summary_vars)

    # Sample both observed nodes (the indicator matrix + the structural outcome).
    # These are two SEPARATE checks, not a joint predictive draw: the factor scores
    # condition on the observed indicator data (``Z_d``), not on the replicated
    # ``Z_obs``, so a replicated indicator is independent of the replicated factor
    # it loads on. ``Z_obs`` is a marginal check of the measurement covariance;
    # ``y_post`` is a check of the structural leg *conditional on the observed
    # indicators*. Together they do not certify the joint model. See the
    # predictive-simulation caveat in ``build_correlated_factor_model``.
    run_ppc(ctx, var_names=list(plan.observation_nodes))

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=summary_vars)
    _diag.run_extended_diagnostics(ctx, causal_term=plan.focal_term)
    _diag.save_trace(ctx)
    # Indicator-scale prior check (#381). Only ``Z_obs`` is the indicator matrix;
    # ``y_post`` is the structural outcome and is covered by the ordinary
    # prior-predictive plot above. AFTER save_trace, which is what attaches the
    # prior/prior_predictive groups to ctx.trace on a fresh fit: called earlier,
    # the check found no prior_predictive group and skipped silently (#383).
    write_indicator_prior_check(ctx, ["Z_obs"])
    _diag.save_prior_posterior_plot(ctx, var_names=summary_vars)

    post = ctx.trace.posterior
    hdi = ctx.reporting.ci_prob
    lo_q = (1.0 - hdi) / 2.0

    # --- Loadings + communalities (the measurement headline) ---
    section_header("Loadings + communalities")
    from language_reading_predictors.statistical_models import (
        corr_factor_summaries as _cf_summaries,
    )

    # The RLI factor loadings live under ``lambda_load`` (the Byrne model uses
    # ``loading``); the residual sigma is free, so lambda is a coefficient on the
    # unit-variance factor, not in general a correlation — the standardised loading /
    # indicator-factor correlation reported alongside is sqrt(communality).
    load_df = _cf_summaries.loadings_communalities_table(
        post, domains, lo_q=lo_q, loading_var="lambda_load"
    )
    save_table(ctx, "loadings_summary", load_df)
    print_table(
        ranked_dataframe_table(
            load_df,
            title=f"Loadings, correlations + communalities - {int(hdi * 100)}% CI (equal-tailed)",
            columns=[
                "indicator", "domain", "loading_mean", "correlation_mean",
                "communality_mean", "communality_lo", "communality_hi",
            ],
            rank_column=False,
            precision=3,
        )
    )

    # --- Factor correlation matrix ---
    section_header("Factor correlation")
    corr_df = _cf_summaries.factor_correlation_matrix(post)
    # Domain names are also used by the structural leg below (beta_factor dims).
    dnames = [str(d) for d in post["domain"].values]
    save_table(ctx, "factor_correlation", corr_df, index=True)
    # The bare mean matrix above is kept for the heatmap, but the house rule is
    # "never a bare point estimate": persist each unique off-diagonal pair with a
    # posterior mean, equal-tailed interval and tail probability alongside it.
    corr_summary_df = _cf_summaries.factor_correlation_pairs(post, lo_q=lo_q)
    save_table(ctx, "factor_correlation_summary", corr_summary_df)

    # --- Structural slopes: factor -> reading gain (adjusted associations) ---
    section_header("Structural slopes (factor -> gain)")
    # The structural leg regresses on all domain factors (beta_factor dims "domain")
    # unless structural_factors isolated a subset (dims "struct_domain", #228 item 14).
    struct_names = (
        list(plan.structural_factors)
        if plan.structural_factors is not None
        else dnames
    )
    _bf_dim = "struct_domain" if plan.structural_factors is not None else "domain"
    struct_rows = [
        _report.coef_row(
            f"beta_{d}", post["beta_factor"].isel({_bf_dim: k}).values, hdi
        )
        for k, d in enumerate(struct_names)
    ]
    extra_terms = (
        (["beta_G"] if plan.use_group else [])
        + (["beta_age"] if plan.use_age else [])
        + [f"beta_{c}" for c in structural_covs]
    )
    struct_rows += [_report.coef_row(t, post[t].values, hdi) for t in extra_terms]
    struct_df = pd.DataFrame(struct_rows)
    save_table(ctx, "structural_summary", struct_df)
    print_table(
        ranked_dataframe_table(
            struct_df,
            title=(
                f"Structural slopes (factor -> gain; adjusted associations) - "
                f"{int(hdi * 100)}% CI"
            ),
            columns=["coefficient", "mean", "lo", "hi", "prob_pos"],
            rank_column=False,
            precision=3,
        )
    )

    write_run_metadata(
        ctx,
        extra={
            "domains": {k: list(v) for k, v in domains.items()},
            "loadings_summary": load_df.to_dict("records"),
            "factor_correlation": corr_df.to_dict(),
            "structural_summary": struct_df.to_dict("records"),
        },
    )

    return finalize_report(ctx)


def fit_rlm_corr_factor(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Byrne correlated-domain-factor measurement fit (#338 Phase B, mm-001).

    Measurement-only (per the 2026-07-16 sign-off): loadings/communalities and
    the domain-factor correlation matrix over the wave-3 nine-measure battery,
    no structural leg. Writes ``loadings_summary.csv``,
    ``factor_correlation.csv`` and ``factor_correlation_summary.csv`` in the
    RLI ``corr_factor`` schema so the shared partial and key-findings builder
    apply unchanged. LOO is not computed, matching the RLI corr-factor family.
    """
    from language_reading_predictors.statistical_models.preprocessing import (
        load_rlm_wave_battery,
    )

    require_spec(spec, "corr_factor")
    plan = _corr_factor.resolve_corr_factor_run_plan(spec)
    ctx = make_context(spec, config)
    ctx.resolved_plan = plan
    _report.write_model_recipe(ctx)
    hdi = ctx.reporting.ci_prob
    lo_q = (1.0 - hdi) / 2.0

    section_header("Prepare data")
    domains = plan.domain_mapping()
    battery = load_rlm_wave_battery(**plan.rlm_prepare_kwargs())
    ctx.prepared = battery
    print_header(ctx)

    section_header("Build model")
    built = _factories.build_rlm_corr_factor_model(
        battery,
        **plan.rlm_factory_kwargs(),
    )
    attach_built(ctx, built)
    render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    _diag.save_prior_predictive_dist_overlay(ctx)

    run_sampling_and_loo(ctx, compute_loo=plan.compute_loo)

    diag_vars = plan.diagnostic_vars()
    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=diag_vars)
    # Power-scaling prior sensitivity (#381), as in the RLI ``corr_factor`` family:
    # LOO is skipped here, so the log groups have to be added explicitly before the
    # reported loadings, residual scales and factor correlations can be power-scaled.
    # #381 exempted this model on the grounds that its posterior had not converged;
    # since the #383 ``LKJCorr`` fix it does (0 divergences, max R-hat 1.0004), so the
    # exemption no longer applies and a latent-factor model is exactly where an
    # unmeasured prior dependence would matter most.
    _diag.compute_log_likelihood_and_prior(ctx, strict=False)
    _diag.run_psense(ctx, var_names=diag_vars)

    run_ppc(ctx, var_names=list(plan.observation_nodes))

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=diag_vars)
    _diag.run_extended_diagnostics(ctx, causal_term=plan.focal_term)
    _diag.save_trace(ctx)
    # Indicator-scale prior check (#381) — the measurement families' stand-in for
    # the estimand pushforward the outcome families get. AFTER save_trace, which
    # is what attaches the prior/prior_predictive groups to ctx.trace on a fresh
    # fit: called earlier, the check found no prior_predictive group and skipped
    # silently (#383) — the re-emitted reporting artefacts never showed it
    # because a reused trace arrives with its groups already on disk.
    write_indicator_prior_check(ctx, ["Z_obs"])
    _diag.save_prior_posterior_plot(ctx, var_names=diag_vars)

    post = ctx.trace.posterior

    # --- Loadings + communalities (the measurement headline) ----------------
    section_header("Loadings + communalities")
    from language_reading_predictors.statistical_models import (
        rlm_corr_factor_summaries as _rlm_summaries,
    )

    load_df = _rlm_summaries.loadings_communalities_table(post, domains, lo_q=lo_q)
    save_table(ctx, "loadings_summary", load_df)
    print_table(
        ranked_dataframe_table(
            load_df,
            title=f"Loadings, correlations + communalities - {int(hdi * 100)}% CI",
            columns=[
                "indicator", "domain", "loading_mean", "correlation_mean",
                "communality_mean", "communality_lo", "communality_hi",
            ],
            rank_column=False,
            precision=3,
        )
    )

    # --- Factor correlation matrix + per-pair summary ------------------------
    section_header("Factor correlation")
    corr_df = _rlm_summaries.factor_correlation_matrix(post)
    save_table(ctx, "factor_correlation", corr_df, index=True)
    corr_summary_df = _rlm_summaries.factor_correlation_pairs(post, lo_q=lo_q)
    save_table(ctx, "factor_correlation_summary", corr_summary_df)
    print_table(
        ranked_dataframe_table(
            corr_summary_df,
            title=f"Domain-factor correlations - {int(hdi * 100)}% CI",
            columns=["domain_i", "domain_j", "mean", "lo", "hi", "prob_pos"],
            rank_column=False,
            precision=3,
        )
    )

    write_run_metadata(
        ctx,
        extra={
            "study_id": "rlm",
            "wave": plan.wave,
            "domains": {k: list(v) for k, v in domains.items()},
            "single_indicator_reliability": plan.single_indicator_reliability,
            "n_children": battery.n_children,
            "structural_leg": False,
        },
    )
    return finalize_report(ctx)
