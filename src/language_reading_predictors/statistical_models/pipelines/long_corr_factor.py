# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Longitudinal correlated-domain-factor orchestration (``kind="long_corr_factor"``).

``fit_longitudinal_corr_factor`` fits the four-wave extension of the
``corr_factor`` CFA over the child×wave panel: correlated vocabulary, code and
grammar factors at every timepoint with a trait/state across-wave structure and
the factor scores marginalised out. It reports the per-wave latent skill
correlation matrices, the conditional latent slopes, the loadings and
communalities, and a descriptive comparison against the observed same-wave
correlations. Those quantities differ in aggregation and conditioning, so no
magnitude ordering between them is required.

The substantive numerical algorithms it depends on are **not** here: the exact
child-level log-likelihood and the constrained-scale log-prior recovery live in
:mod:`lcf_inference`, and the descriptive comparison summaries in
:mod:`lcf_summaries`, so both are testable without a report, an output directory
or a terminal (#394 design point 7). What stays with the family is the LOO
stitching that binds the recovered likelihood back onto the trace.

A measurement / triangulation model — every quantity is a descriptive
association, never causal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rich import print as rprint

from language_reading_predictors.models._reporting import (
    print_table,
    ranked_dataframe_table,
    section_header,
)
from language_reading_predictors.statistical_models import (
    diagnostics as _diag,
    factories as _factories,
    lcf_inference as _lcf_inference,
    lcf_summaries as _lcf_summaries,
    long_corr_factor as _long_corr_factor,
    reporting as _report,
)
from language_reading_predictors.statistical_models.artifacts import save_table
from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
    make_context,
)
from language_reading_predictors.statistical_models.preprocessing import (
    load_wave_panel,
)
from language_reading_predictors.statistical_models.prior_artifacts import (
    write_indicator_prior_check,
)
from language_reading_predictors.statistical_models.publication import (
    print_header,
    print_loo_row,
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

# The LCF exact child-level log-likelihood and constrained-scale log-prior
# recovery are substantive numerical algorithms, isolated in ``lcf_inference``
# so they are testable independently of report publication (#394, pillar 7).
# Re-exported here under their historical private names for existing callers.
_lcf_child_log_likelihood = _lcf_inference.child_log_likelihood
_lcf_log_prior = _lcf_inference.log_prior


def _lcf_stitch_loo(ctx: StatisticalFitContext, built) -> None:
    """Pointwise PSIS-LOO for the longitudinal CFA (custom, per-child stitch).

    The masked-cell likelihood is one ``MvNormal`` per observed-cell pattern, so
    there is no single observed node ``az.loo`` can key on. Compute each pattern
    node's per-child log-likelihood, stitch them into one ``(chain, draw, child)``
    array over all children, and run pointwise LOO on that — the honest per-child
    predictive score an invariance comparison (#313) would use. Attach the exact
    constrained-scale prior terms through the companion workaround above. As in the
    other LOO-enabled families, a failure is fatal: a reporting run must not silently
    complete without the likelihood, prior and predictive diagnostics its output
    contract requires.
    """
    import arviz as az
    import xarray as xr

    stitched = _lcf_child_log_likelihood(ctx.trace, built)
    if "log_likelihood" not in ctx.trace.children:
        ctx.trace["log_likelihood"] = xr.Dataset()
    ctx.trace.log_likelihood["lcf_child"] = stitched
    ctx.trace["log_prior"] = _lcf_log_prior(ctx.trace, ctx.model)
    ctx.loo = az.loo(ctx.trace, var_name="lcf_child", pointwise=True)
    _report.write_loo_summary(ctx)
    print_loo_row(ctx)


# LCF descriptive/comparison summaries live in ``lcf_summaries`` (#394 pillar 6);
# re-exported here under their historical private names for existing callers.
_lcf_observed_domain_corr = _lcf_summaries.observed_domain_corr
_lcf_items_scale = _lcf_summaries.items_scale
_lcf_observed_conditional_slope = _lcf_summaries.observed_conditional_slope
_lcf_concurrent_comparison = _lcf_summaries.concurrent_comparison


def fit_longitudinal_corr_factor(
    spec: ModelSpec, config: str = "dev"
) -> StatisticalFitContext:
    """Longitudinal correlated-domain-factor model (LRP-RLI-LCF-001, #313).

    Fits the four-wave extension of the ``corr_factor`` CFA over the child×wave
    panel: correlated vocabulary / code / grammar factors at every timepoint, with a
    trait/state across-wave structure and the factor scores marginalised out. Reports
    the per-wave latent skill correlation matrices, the conditional (partial) latent
    slopes, the loadings / communalities, and a descriptive comparison against the
    observed same-wave correlations (the #312 anchor). The quantities differ in their
    aggregation and conditioning, so no magnitude ordering is required. A measurement
    / triangulation model — every quantity is a descriptive association, never causal.
    """
    require_spec(spec, "long_corr_factor")
    plan = _long_corr_factor.resolve_long_corr_factor_run_plan(spec)
    ctx = make_context(spec, config)
    ctx.resolved_plan = plan
    _report.write_model_recipe(ctx)
    # A small-n latent model; even fully marginalised a few boundary divergences can
    # survive at the tier default, so lift target_accept via the spec (as mm-001 does).

    section_header("Prepare data")
    domains = plan.domain_mapping()
    panel = load_wave_panel(**plan.prepare_kwargs())
    ctx.prepared = panel
    print_header(ctx)

    section_header("Build model")
    built = _factories.build_longitudinal_corr_factor_model(
        panel,
        **plan.factory_kwargs(),
    )
    attach_built(ctx, built)
    render_model_graph(ctx)

    z_nodes = built.extras["z_nodes"]
    summary_vars = plan.diagnostic_vars()

    # Dedupe: ``communality`` is itself a free RV under the default
    # parameterisation, so appending it unconditionally would double it. The
    # derived lambda_load / sigma_indicator / within_share are named explicitly —
    # as Deterministics they are no longer covered by the free-RV listing, and the
    # prior-vs-posterior overlay below needs their prior draws.
    prior_vars = plan.prior_vars(
        free_rv_names=[rv.name for rv in built.model.free_RVs],
        observation_nodes=z_nodes,
    )

    def _stitch_child_loo(c: StatisticalFitContext) -> None:
        # Automatic single-target LOO is ambiguous with per-pattern observed
        # nodes, so the plan skips ordinary LOO and stitches the exact per-child
        # likelihood immediately after sampling.
        section_header("LOO-PSIS (per-child stitch)")
        _lcf_stitch_loo(c, built)

    # ``communality`` joins the power-scaled set (#383 follow-up): under the
    # pooled-budget parameterisation it is the free measurement parameter behind
    # the reported loadings table, exactly the place a prior dependence would live.
    shared_stages().run_primary_fit(
        ctx,
        PrimaryFitPlan(
            diagnostic_vars=tuple(summary_vars),
            ppc_var_names=tuple(z_nodes),
            prior_predictive_var_names=tuple(prior_vars),
            plot_prior_predictive=_diag.save_prior_predictive_dist_overlay,
            post_sampling_audit=_stitch_child_loo,
            psense_vars=tuple(plan.psense_vars()),
            psense_timing="before_trace",
            extended_term=plan.focal_term,
            compute_loo=plan.compute_loo,
        ),
    )
    # Indicator-scale prior check (#381), pooled across the missingness-pattern
    # blocks: each block is its own observed node but the indicators are shared.
    # AFTER save_trace, which is what attaches the prior/prior_predictive groups
    # to ctx.trace on a fresh fit: called earlier, the check found no
    # prior_predictive group and skipped silently (#383).
    write_indicator_prior_check(ctx, z_nodes)
    _diag.save_prior_posterior_plot(ctx, var_names=summary_vars)

    post = ctx.trace.posterior
    hdi = ctx.reporting.ci_prob
    lo_q = (1 - hdi) / 2

    # --- Loadings + communalities (the measurement layer) ---
    section_header("Loadings + communalities")
    dom_of = built.extras["domain_of"]
    load_rows = []
    for j, name in enumerate(str(s) for s in post["indicator"].values):
        lam_d = post["lambda_load"].isel(indicator=j).values.reshape(-1)
        com_d = post["communality"].isel(indicator=j).values.reshape(-1)
        corr_d = np.sqrt(com_d)
        load_rows.append(
            {
                "indicator": name,
                "domain": dom_of.get(name, "?"),
                "loading_mean": float(np.mean(lam_d)),
                "loading_lo": float(np.quantile(lam_d, lo_q)),
                "loading_hi": float(np.quantile(lam_d, 1 - lo_q)),
                "correlation_mean": float(np.mean(corr_d)),
                "communality_mean": float(np.mean(com_d)),
                "communality_lo": float(np.quantile(com_d, lo_q)),
                "communality_hi": float(np.quantile(com_d, 1 - lo_q)),
            }
        )
    load_df = pd.DataFrame(load_rows)
    save_table(ctx, "loadings_summary", load_df)
    print_table(
        ranked_dataframe_table(
            load_df,
            title=f"Loadings + communalities - {int(hdi * 100)}% CI (equal-tailed)",
            columns=[
                "indicator", "domain", "loading_mean", "correlation_mean",
                "communality_mean", "communality_lo", "communality_hi",
            ],
            rank_column=False,
            precision=3,
        )
    )

    # --- Per-wave latent factor correlations (the headline) ---
    section_header("Per-wave latent factor correlations")
    corr_df = _report.longitudinal_factor_correlations(ctx.trace, ci_prob=hdi)
    save_table(ctx, "factor_correlation_by_wave", corr_df)
    print_table(
        ranked_dataframe_table(
            corr_df,
            title=f"Per-wave latent factor correlations - {int(hdi * 100)}% ETI",
            columns=["wave", "domain_i", "domain_j", "mean", "lo", "hi", "prob_pos"],
            rank_column=False,
            precision=3,
        )
    )

    # --- Conditional (partial) latent slopes ---
    section_header("Conditional latent slopes")
    slope_df = _report.longitudinal_conditional_slopes(ctx.trace, ci_prob=hdi)
    save_table(ctx, "latent_conditional_slopes", slope_df)

    # --- Trait / state (across-wave) structure ---
    section_header("Trait / state structure")
    ts_rows = []
    for j, d in enumerate(str(s) for s in post["domain"].values):
        pi_d = post["trait_share"].isel(domain=j).values.reshape(-1)
        ts_rows.append(
            {
                "domain": d,
                "trait_share_mean": float(np.mean(pi_d)),
                "trait_share_lo": float(np.quantile(pi_d, lo_q)),
                "trait_share_hi": float(np.quantile(pi_d, 1 - lo_q)),
            }
        )
    ts_df = pd.DataFrame(ts_rows)
    save_table(ctx, "trait_state_summary", ts_df)

    # --- Latent-versus-observed comparison (#312 triangulation anchor) --------
    section_header("Latent-versus-observed correlation comparison")
    obs_df = _lcf_observed_domain_corr(built)
    xcheck_df = _report.disattenuation_crosscheck(corr_df, obs_df)
    save_table(ctx, "disattenuation_crosscheck", xcheck_df)
    n_latent_below = int((~xcheck_df["latent_ge_observed"]).sum())
    n_latent_at_or_above = len(xcheck_df) - n_latent_below
    rprint(
        "[cyan]Latent-versus-observed comparison: "
        f"{n_latent_below} wave/pair(s) are below and {n_latent_at_or_above} are at "
        "or above the mean observed indicator-pair magnitude. This is a descriptive "
        "gap direction between different estimands, not a pass/fail ordering.[/cyan]"
    )

    # --- Items-scale translation for the headline pairs ---
    section_header("Items-scale translation (selected pairs)")
    items_df = _lcf_items_scale(ctx, built)
    save_table(ctx, "latent_items_slopes", items_df)

    # --- Directed comparison with matching concurrent associations (#312) ---
    section_header("Directed LCF-versus-concurrent comparison")
    concurrent_df = _lcf_concurrent_comparison(ctx, built)
    save_table(ctx, "lcf_concurrent_comparison", concurrent_df)
    n_ca_available = int(concurrent_df["ca_available"].sum())
    if n_ca_available < len(concurrent_df):
        rprint(
            "[yellow]Directed #312 comparison is incomplete: "
            f"{n_ca_available}/{len(concurrent_df)} matching concurrent rows were "
            "found under this output root/config. Fit CA002--006 at the same tier "
            "to populate the missing side.[/yellow]"
        )

    write_run_metadata(
        ctx,
        extra={
            "loo_elpd": float(ctx.loo.elpd) if ctx.loo is not None else None,
            "domains": {k: list(v) for k, v in domains.items()},
            "invariance": built.extras["invariance"],
            "n_used_children": built.extras["n_used_children"],
            "loadings_summary": load_df.to_dict("records"),
            "factor_correlation_by_wave": corr_df.to_dict("records"),
            "trait_state_summary": ts_df.to_dict("records"),
            # Keep the legacy key for output compatibility; neither count is a gate.
            "disattenuation_reversals": n_latent_below,
            "latent_below_observed_count": n_latent_below,
            "latent_observed_comparison": "descriptive; no required ordering",
            "lcf_concurrent_comparison_rows": int(len(concurrent_df)),
            "lcf_concurrent_comparison_available": n_ca_available,
        },
    )

    return finalize_report(ctx)
