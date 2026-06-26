# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
End-to-end fit pipeline for LRP52-LRP60.

``fit_itt(spec, config)`` is the entry point for the ITT models.
``fit_joint(spec, config)`` is the entry point for LRP55.
``fit_mechanism(spec, config)`` is the entry point for LRP56/57/58.

Each pipeline:

1. Loads data via :func:`preprocessing.load_and_prepare`.
2. Builds the PyMC model via the appropriate factory.
3. Writes prior-panel plots.
4. Runs prior predictive, posterior sampling (nutpie), LOO, posterior
   predictive.
5. Saves ``trace.nc``, ``config.json``, ``metrics.json`` and the standard
   diagnostic plots to ``output/statistical_models/models/{model_id}-{config}/``.
6. Copies ``docs/models/{model_id}/index.qmd`` alongside the artefacts so
   the Quarto report can be rendered in-place.
"""

from __future__ import annotations

import os
import shutil

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from rich import print as rprint

from language_reading_predictors.models._reporting import (
    metrics_table,
    print_panel,
    print_table,
    ranked_dataframe_table,
    run_summary_panel,
    section_header,
    stat_model_header_panel,
)
from language_reading_predictors.statistical_models import (
    diagnostics as _diag,
    factories as _factories,
    priors as _priors,
    reporting as _report,
)
from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
    make_context,
)
from language_reading_predictors.statistical_models.environment import DOCS_DIR
from language_reading_predictors.statistical_models.preprocessing import (
    load_and_prepare,
    load_and_prepare_lagged_outcome,
    load_wave_panel,
)


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


def _print_header(ctx: StatisticalFitContext) -> None:
    """Print the start-of-fit banner panel."""
    spec = ctx.spec
    prepared = ctx.prepared
    rprint()
    print_panel(
        stat_model_header_panel(
            model_id=spec.model_id,
            title=spec.title,
            kind=spec.kind,
            config_name=ctx.reporting.config_name,
            outcome_symbol=spec.outcome_symbol,
            mechanism_symbol=spec.mechanism_symbol,
            adjustment=spec.adjustment or None,
            n_obs=prepared.n_obs if prepared else None,
            n_children=prepared.n_children if prepared else None,
            n_phases=prepared.n_phases if prepared else None,
        )
    )


def _print_footer(ctx: StatisticalFitContext) -> None:
    """Print the end-of-fit banner panel."""
    rprint()
    print_panel(run_summary_panel(output_dir=ctx.output_dir))


def _print_loo_row(ctx: StatisticalFitContext) -> None:
    """Render the LOO ELPD / p / se summary as a small table.

    arviz 1.x ``ELPDData`` exposes ``elpd`` / ``se`` / ``p`` (the 0.x
    ``elpd_loo`` / ``p_loo`` / ``looic`` attributes were removed).
    """
    if ctx.loo is None:
        return
    rows = [
        {"metric": "elpd_loo", "value": float(ctx.loo.elpd)},
        {"metric": "se", "value": float(ctx.loo.se)},
        {"metric": "p_loo", "value": float(ctx.loo.p)},
    ]
    print_table(
        metrics_table(
            rows,
            title="LOO-PSIS",
            columns=["metric", "value"],
        )
    )


def _copy_report_template(context: StatisticalFitContext) -> None:
    src = os.path.join(DOCS_DIR, "models", context.spec.model_id, "index.qmd")
    dst = os.path.join(context.output_dir, "index.qmd")
    if os.path.exists(src):
        shutil.copy(src, dst)
        rprint(f"  Report template copied to {dst}")
    else:
        rprint(f"  [yellow]No report template found at {src}[/yellow]")


def _render_model_graph(context: StatisticalFitContext) -> None:
    try:
        g = _graphviz(context.model)
        g.render(
            filename=os.path.join(context.output_dir, "model_graph"),
            format="svg",
            cleanup=True,
        )
    except Exception as exc:  # pragma: no cover
        rprint(f"[yellow]Graphviz render failed: {exc}[/yellow]")


def _graphviz(model):
    import pymc as pm

    g = pm.model_to_graphviz(model)
    g.graph_attr["fontname"] = "Helvetica"
    g.node_attr["fontname"] = "Helvetica"
    g.edge_attr["fontname"] = "Helvetica"
    return g


def _save_ppc(context: StatisticalFitContext) -> None:
    # arviz 1.x removed az.plot_ppc; the equivalent is arviz_plots.plot_ppc_dist
    # (returns a PlotCollection with .savefig). Guarded — a PPC plot failure must
    # not abort the fit.
    try:
        import arviz_plots as azp

        pc = azp.plot_ppc_dist(context.trace)
        pc.savefig(
            os.path.join(context.output_dir, "posterior_predictive_check.png"),
            dpi=300,
        )
        plt.close("all")
    except Exception as exc:  # pragma: no cover
        rprint(f"[yellow]PPC plot failed: {exc}[/yellow]")


# ---------------------------------------------------------------------------
# ITT pipeline (LRP52 / LRP53 / LRP54 / LRP60)
# ---------------------------------------------------------------------------


def fit_itt(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    assert spec.kind == "itt"
    assert spec.outcome_symbol is not None

    ctx = make_context(spec, config)

    section_header("Prepare data")
    adjust_for = tuple(spec.extra.get("adjust_for", ()))
    # Optionally restrict to the complete-case subset of some columns *without*
    # adjusting for them (matched comparator, e.g. LRP60a vs LRP60).
    restrict_complete = tuple(spec.extra.get("restrict_complete", ()))
    # An ITT model whose outcome is outside ``ITT_OUTCOMES`` (e.g. the taught-
    # vocabulary block measures, LRP74/LRP75) overrides the prepared outcome set
    # so only the outcome and its chosen cross-baselines are loaded - this keeps
    # the complete-case mask from dropping rows for the eight standardised
    # outcomes the model never uses. ``cross_symbols`` selects the cross-baseline
    # set (default = every other ITT outcome).
    extra_outcomes = spec.extra.get("outcomes")
    cross_symbols = spec.extra.get("cross_symbols")
    if extra_outcomes is not None:
        extra_outcomes = tuple(extra_outcomes)
        if spec.outcome_symbol not in extra_outcomes:
            raise ValueError(
                f"outcome_symbol {spec.outcome_symbol!r} must be included in "
                f"outcomes={extra_outcomes!r}"
            )
        prepared = load_and_prepare(
            phase_mode="itt",
            outcomes=extra_outcomes,
            covariates=adjust_for,
            restrict_complete=restrict_complete,
        )
    else:
        prepared = load_and_prepare(
            phase_mode="itt", covariates=adjust_for, restrict_complete=restrict_complete
        )
    ctx.prepared = prepared

    _print_header(ctx)

    section_header("Build model")
    _priors.save_shared_prior_panel(ctx.output_dir)

    built = _factories.build_itt_model(
        prepared,
        outcome_symbol=spec.outcome_symbol,
        use_age_gp=spec.extra.get("use_age_gp", False),
        use_own_baseline_gp=spec.extra.get("use_own_baseline_gp", False),
        use_varying_tau=spec.extra.get("use_varying_tau", False),
        adjust_for=adjust_for,
        cross_symbols=cross_symbols,
    )
    ctx.model = built.model
    ctx.model_vars = built.variables
    ctx.prepared = built.prepared

    _render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000, var_names=["y_post", "eta"])

    section_header("Sampling posterior (nutpie)")
    _diag.sample_posterior(ctx)

    section_header("LOO-PSIS")
    _diag.compute_log_likelihood_and_loo(ctx)
    _report.write_loo_summary(ctx)
    _print_loo_row(ctx)

    section_header("Summary diagnostics")
    diag_vars = ["alpha", "tau", "gamma_own", "kappa"]
    diag_vars.extend(f"gamma_{c}" for c in adjust_for)
    _diag.summary_diagnostics(
        ctx, var_names=diag_vars
    )

    section_header("Posterior predictive")
    _diag.sample_posterior_predictive(ctx, var_names=["y_post"])
    _save_ppc(ctx)

    _diag.save_trace(ctx)

    # Treatment-effect summary on both scales.
    section_header("Treatment-effect summary")
    tau_s = _report.tau_summary_itt(
        ctx.trace,
        hdi_prob=ctx.reporting.hdi,
        # built.prepared is the (possibly row-subset) frame the model was fit
        # on, so G aligns with eta's obs_id axis (finding #2 in issue #78).
        G=built.prepared.G,
    )
    tau_df = pd.DataFrame([tau_s])
    tau_df.to_csv(os.path.join(ctx.output_dir, "tau_summary.csv"), index=False)
    ctx.tables["tau_summary"] = tau_df
    print_table(
        metrics_table(
            [{"metric": k, "value": v} for k, v in tau_s.items()],
            title=f"tau ({spec.outcome_symbol}) - {int(ctx.reporting.hdi * 100)}% CI (equal-tailed)",
            columns=["metric", "value"],
        )
    )

    _report.write_run_metadata(
        ctx,
        extra={
            "loo_elpd": float(ctx.loo.elpd),
            "tau_summary": tau_s,
            "adjust_for": list(adjust_for),
        },
    )

    section_header("Report")
    _copy_report_template(ctx)
    _print_footer(ctx)
    return ctx


# ---------------------------------------------------------------------------
# Joint pipeline (LRP55)
# ---------------------------------------------------------------------------


def fit_joint(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    assert spec.kind == "joint"

    ctx = make_context(spec, config)

    section_header("Prepare data")
    # A joint model may target an explicit outcome set (e.g. the taught-vs-not-
    # taught contrast in LRP76); load exactly those so the complete-case mask is
    # not driven by the eight standardised outcomes. Defaults to ITT_OUTCOMES.
    joint_outcomes = spec.extra.get("outcomes")
    if joint_outcomes is not None:
        prepared = load_and_prepare(phase_mode="itt", outcomes=tuple(joint_outcomes))
    else:
        prepared = load_and_prepare(phase_mode="itt")
    ctx.prepared = prepared

    _print_header(ctx)

    section_header("Build model")
    _priors.save_shared_prior_panel(ctx.output_dir)

    built = _factories.build_joint_model(
        prepared,
        outcomes=spec.extra.get("outcomes"),
        use_age_gp=spec.extra.get("use_age_gp", False),
        partial_pool_age_gp=spec.extra.get("partial_pool_age_gp", True),
        use_residual_correlation=spec.extra.get("use_residual_correlation", False),
    )
    ctx.model = built.model
    ctx.model_vars = built.variables
    ctx.prepared = built.prepared

    _render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000, var_names=["y_post"])

    section_header("Sampling posterior (nutpie)")
    _diag.sample_posterior(ctx)

    section_header("LOO-PSIS")
    _diag.compute_log_likelihood_and_loo(ctx)
    _report.write_loo_summary(ctx)
    _print_loo_row(ctx)

    section_header("Summary diagnostics")
    _joint_vars = ["alpha", "tau", "kappa"]
    if spec.extra.get("use_residual_correlation", False):
        _joint_vars.append("sigma_outcome")
    _diag.summary_diagnostics(ctx, var_names=_joint_vars)

    section_header("Posterior predictive")
    _diag.sample_posterior_predictive(ctx, var_names=["y_post"])
    _save_ppc(ctx)
    _diag.save_trace(ctx)

    section_header("Treatment-effect summary")
    outcomes = list(ctx.trace.posterior["outcome"].values)
    tau_df = _report.tau_summary_joint(ctx.trace, outcomes, hdi_prob=ctx.reporting.hdi)
    tau_df.to_csv(os.path.join(ctx.output_dir, "tau_summary.csv"), index=False)
    ctx.tables["tau_summary"] = tau_df
    print_table(
        ranked_dataframe_table(
            tau_df,
            title=f"tau by outcome - {int(ctx.reporting.hdi * 100)}% CI (equal-tailed)",
            columns=["outcome", "tau_mean", "tau_lo", "tau_hi", "prob_pos"],
            rank_column=False,
        )
    )

    contrast = _report.tau_contrast_matrix(ctx.trace, outcomes)
    contrast.to_csv(os.path.join(ctx.output_dir, "tau_contrast_matrix.csv"))
    ctx.tables["tau_contrast_matrix"] = contrast

    meta_extra: dict = {"loo_elpd": float(ctx.loo.elpd)}

    # Headline difference parameter for a two-outcome contrast (LRP76: taught vs
    # not-taught). ``difference = (a, b)`` reports the posterior of tau[a]-tau[b].
    difference = spec.extra.get("difference")
    if difference is not None:
        pair = tuple(difference)
        section_header("Treatment-effect difference")
        diff_s = _report.tau_difference_summary(
            ctx.trace, outcomes, pair, hdi_prob=ctx.reporting.hdi
        )
        diff_df = pd.DataFrame([diff_s])
        diff_df.to_csv(os.path.join(ctx.output_dir, "tau_difference.csv"), index=False)
        ctx.tables["tau_difference"] = diff_df
        print_table(
            metrics_table(
                [{"metric": k, "value": v} for k, v in diff_s.items()],
                title=(
                    f"tau[{pair[0]}] - tau[{pair[1]}] "
                    f"- {int(ctx.reporting.hdi * 100)}% CI (equal-tailed)"
                ),
                columns=["metric", "value"],
            )
        )
        meta_extra["tau_difference"] = diff_s

    _report.write_run_metadata(ctx, extra=meta_extra)

    section_header("Report")
    _copy_report_template(ctx)
    _print_footer(ctx)
    return ctx


# ---------------------------------------------------------------------------
# Mechanism pipeline (LRP56 / LRP57 / LRP58)
# ---------------------------------------------------------------------------


def fit_mechanism(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    assert spec.kind == "mechanism"
    assert spec.mechanism_symbol is not None

    ctx = make_context(spec, config)

    section_header("Prepare data")
    # A model may restrict the prepared outcomes (e.g. LRP72 uses only L/B/N) so
    # ``drop_missing_pre`` does not discard rows for measures the model ignores.
    extra_outcomes = spec.extra.get("outcomes")
    if extra_outcomes is not None:
        prepared = load_and_prepare(phase_mode="all", outcomes=tuple(extra_outcomes))
    else:
        prepared = load_and_prepare(phase_mode="all")
    ctx.prepared = prepared

    _print_header(ctx)

    section_header("Build model")
    _priors.save_shared_prior_panel(ctx.output_dir)

    moderator_symbol = spec.extra.get("moderator_symbol")
    # Drop the autoregressive baseline (any ``*_pre`` token, e.g. W_pre / N_pre)
    # from the confounder list — it enters via ``adjust_baseline_symbol``.
    confounders = [s for s in spec.adjustment if not s.endswith("_pre")]
    if moderator_symbol is not None:
        # The moderator is carried by its standardised main effect + interaction
        # in the factory, so drop it from the plain confounder loop to avoid a
        # collinear duplicate main effect. The standardised term still adjusts
        # for M, preserving any DAG-required conditioning (e.g. E in LRP71).
        confounders = [s for s in confounders if s != moderator_symbol]

    built = _factories.build_mechanism_model(
        prepared,
        mechanism_symbol=spec.mechanism_symbol,
        outcome_symbol=spec.outcome_symbol or "W",
        adjust_baseline_symbol=spec.extra.get("adjust_baseline_symbol", "W"),
        confounder_symbols=tuple(s for s in confounders if s in ("G", "A") or len(s) == 1),
        use_age_gp=spec.extra.get("use_age_gp", False),
        phase_specific_mechanism=spec.extra.get("phase_specific_mechanism", False),
        use_subject_random_intercept=spec.extra.get(
            "use_subject_random_intercept", True
        ),
        moderator_symbol=moderator_symbol,
        moderator_is_covariate=spec.extra.get("moderator_is_covariate", False),
        include_interaction=spec.extra.get("include_interaction", True),
        linear_mechanism=spec.extra.get("linear_mechanism", False),
    )
    ctx.model = built.model
    ctx.model_vars = built.variables
    ctx.prepared = built.prepared

    _render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000, var_names=["y_post", "eta"])

    section_header("Sampling posterior (nutpie)")
    _diag.sample_posterior(ctx)

    section_header("LOO-PSIS")
    _diag.compute_log_likelihood_and_loo(ctx)
    _report.write_loo_summary(ctx)
    _print_loo_row(ctx)

    section_header("Summary diagnostics")
    _mech_vars = ["alpha", "beta_G", "gamma_own", "kappa"]
    if spec.extra.get("use_subject_random_intercept", True):
        _mech_vars.append("sigma_child")
    if spec.extra.get("linear_mechanism", False):
        _mech_vars.append("beta_mech")
    if moderator_symbol is not None:
        _mech_vars.append("gamma_mod")
        if spec.extra.get("include_interaction", True):
            _mech_vars.append("gamma_int")
    _diag.summary_diagnostics(ctx, var_names=_mech_vars)

    section_header("Posterior predictive")
    _diag.sample_posterior_predictive(ctx, var_names=["y_post"])
    _save_ppc(ctx)

    # Mechanism curve: f_mech vs mech_post_logit grid (logit-contribution scale only).
    section_header("Mechanism curve")
    _write_mechanism_curve(ctx)

    meta_extra = {"loo_elpd": float(ctx.loo.elpd), "adjustment": spec.adjustment}

    # Linear-moderation summary (gamma_int / gamma_mod), when a moderator is set.
    if moderator_symbol is not None:
        section_header("Interaction summary")
        gi = _report.gamma_interaction_summary(ctx.trace, hdi_prob=ctx.reporting.hdi)
        gi_df = pd.DataFrame([gi])
        gi_df.to_csv(
            os.path.join(ctx.output_dir, "interaction_summary.csv"), index=False
        )
        ctx.tables["interaction_summary"] = gi_df
        print_table(
            metrics_table(
                [{"metric": k, "value": v} for k, v in gi.items()],
                title=(
                    f"Linear moderation by {moderator_symbol} "
                    f"- {int(ctx.reporting.hdi * 100)}% CI (equal-tailed)"
                ),
                columns=["metric", "value"],
            )
        )
        meta_extra["moderator_symbol"] = moderator_symbol
        meta_extra["interaction_summary"] = gi

    _diag.save_trace(ctx)
    _report.write_run_metadata(ctx, extra=meta_extra)

    section_header("Report")
    _copy_report_template(ctx)
    _print_footer(ctx)
    return ctx


def _write_mechanism_curve(ctx: StatisticalFitContext) -> None:
    post = ctx.trace.posterior
    if "f_mech" not in post:
        return
    f = post["f_mech"].stack(sample=("chain", "draw")).values  # (n_obs, n_sample)

    from language_reading_predictors.statistical_models.measures import MEASURES
    from language_reading_predictors.statistical_models.preprocessing import logit_safe

    sym = ctx.spec.mechanism_symbol
    N = MEASURES[sym].n_trials
    mech_logit = logit_safe(ctx.prepared.post_counts[sym], N)
    order = np.argsort(mech_logit)
    x = mech_logit[order]
    f_ord = f[order]
    mean = f_ord.mean(axis=1)
    lo = np.quantile(f_ord, 0.025, axis=1)
    hi = np.quantile(f_ord, 0.975, axis=1)
    pd.DataFrame(
        {"mech_logit": x, "f_mean": mean, "f_lo": lo, "f_hi": hi}
    ).to_csv(os.path.join(ctx.output_dir, "mechanism_curve.csv"), index=False)
    plt.figure(figsize=(6, 4))
    plt.plot(x, mean, color="#1f77b4", lw=2)
    plt.fill_between(x, lo, hi, color="#1f77b4", alpha=0.2)
    plt.xlabel(f"logit({ctx.spec.mechanism_symbol}_post)")
    plt.ylabel(r"$f^{mech}$ (logit contribution)")
    plt.title(f"Mechanism curve: {ctx.spec.mechanism_symbol} -> W")
    plt.savefig(
        os.path.join(ctx.output_dir, "mechanism_curve.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# ---------------------------------------------------------------------------
# Mediation pipeline (LRP59)
# ---------------------------------------------------------------------------

_MED_COEF_VARS = [
    "a0", "a_G", "a_L", "a_A", "a_E", "a_R", "kappa_M",
    "b0", "b_G", "b_M", "b_GM", "b_W", "b_A", "b_E", "b_R", "kappa_Y",
]

# LRP62 (gaussian_composite): the mediator leg is Normal (a_comp / sigma_M) and
# the observed mediator node is "M_post" rather than the Beta-Binomial "L_post".
_MED_COEF_VARS_GAUSSIAN = [
    "a0", "a_G", "a_comp", "a_A", "a_E", "a_R", "sigma_M",
    "b0", "b_G", "b_M", "b_GM", "b_W", "b_A", "b_E", "b_R", "kappa_Y",
]


_T3_SENSITIVITY_TIME = 3  # post-RCT wave used for the temporal-ordering check


def _fit_t3_sensitivity(
    ctx: StatisticalFitContext,
    spec: ModelSpec,
    *,
    confounders: tuple[str, ...],
    mediator_kind: str,
    route_symbols: tuple[str, ...],
):
    """Temporal-ordering sensitivity fit for the mediation models (issue #84).

    Refits the *identical* mediation model but with the outcome measured at a
    later wave (t3) while the mediator stays at t2, so the mediator precedes the
    outcome in time. The t2 -> t3 increment is **not randomised** (both arms are
    treated after t2), so this is a triangulation point for the contemporaneous
    measurement caveat, not a cleaner causal estimate. Returns the g-formula
    decomposition DataFrame for the t3-outcome variant.
    """
    import pymc as pm

    from language_reading_predictors.statistical_models import mediation as _med

    outcome_symbol = spec.outcome_symbol or "W"
    prepared_t3 = load_and_prepare_lagged_outcome(
        outcome_symbol, outcome_time=_T3_SENSITIVITY_TIME
    )
    built_t3, med_t3 = _factories.build_mediation_model(
        prepared_t3,
        mediator_symbol=spec.mechanism_symbol or "L",
        outcome_symbol=outcome_symbol,
        confounder_symbols=confounders,
        mediator_kind=mediator_kind,
        route_symbols=route_symbols,
    )
    s = ctx.sampling
    with built_t3.model:
        trace_t3 = pm.sample(
            draws=s.draws,
            tune=s.tune,
            chains=s.chains,
            cores=s.cores,
            target_accept=s.target_accept,
            nuts_sampler="nutpie",
            return_inferencedata=True,
            random_seed=s.random_seed,
            progressbar=False,
        )
    return _med.decompose(
        trace_t3,
        med_t3,
        hdi_prob=ctx.reporting.hdi,
    )
def fit_mediation(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """ITT-phase mediation decomposition (LRP59): how much of G -> W flows via L."""
    assert spec.kind == "mediation"
    from language_reading_predictors.statistical_models import mediation as _med

    ctx = make_context(spec, config)

    section_header("Prepare data")
    # Phase 0 only (t1 -> t2): the single randomised contrast. One row per child.
    prepared = load_and_prepare(phase_mode="itt")
    ctx.prepared = prepared

    _print_header(ctx)

    section_header("Build model")
    _priors.save_shared_prior_panel(ctx.output_dir)

    confounders = tuple(
        s for s in spec.adjustment if s not in ("G", "A", "L_t1", "W_pre")
    )
    mediator_kind = spec.extra.get("mediator_kind", "beta_binomial")
    route_symbols = tuple(spec.extra.get("route_symbols", ()))
    built, med_data = _factories.build_mediation_model(
        prepared,
        mediator_symbol=spec.mechanism_symbol or "L",
        outcome_symbol=spec.outcome_symbol or "W",
        confounder_symbols=confounders,
        mediator_kind=mediator_kind,
        route_symbols=route_symbols,
    )
    ctx.model = built.model
    ctx.model_vars = built.variables
    ctx.prepared = built.prepared

    # The mediator observed node differs by kind: Beta-Binomial "L_post" vs the
    # Gaussian composite "M_post".
    is_gaussian = mediator_kind == "gaussian_composite"
    mediator_node = "M_post" if is_gaussian else "L_post"
    # Diagnose every scalar coefficient the model actually built (deterministics
    # and the observed mediator/outcome nodes are not free RVs), so the list
    # tracks the fitted confounder set instead of a hand-maintained constant.
    coef_vars = sorted(rv.name for rv in built.model.free_RVs if rv.ndim == 0)

    _render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000, var_names=[mediator_node, "y_post"])

    section_header("Sampling posterior (nutpie)")
    _diag.sample_posterior(ctx)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=coef_vars)

    section_header("Posterior predictive")
    _diag.sample_posterior_predictive(ctx, var_names=[mediator_node, "y_post"])
    _save_ppc(ctx)
    _diag.save_trace(ctx)

    section_header("Mediation decomposition (g-formula)")
    med_df = _med.decompose(
        ctx.trace,
        med_data,
        hdi_prob=ctx.reporting.hdi,
    )
    med_df.to_csv(os.path.join(ctx.output_dir, "mediation_summary.csv"), index=False)
    ctx.tables["mediation_summary"] = med_df
    print_table(
        ranked_dataframe_table(
            med_df,
            title=f"Mediation (intervention-helps; words out of {med_data.n_trials_W})",
            columns=["quantity", "words_mean", "words_lo", "words_hi", "prob_pos"],
            rank_column=False,
            precision=3,
        )
    )

    # --- Temporal-ordering sensitivity: outcome at t3, mediator still at t2 ---
    # Triangulation for the contemporaneous-measurement caveat (issue #84): the
    # mediator now precedes the outcome in time. NB the t2 -> t3 increment is not
    # randomised (both arms treated after t2), so read this as triangulation only.
    section_header("Temporal-ordering sensitivity (outcome at t3)")
    med_df_t3 = _fit_t3_sensitivity(
        ctx,
        spec,
        confounders=confounders,
        mediator_kind=mediator_kind,
        route_symbols=route_symbols,
    )
    med_df_t3.to_csv(
        os.path.join(ctx.output_dir, "mediation_summary_t3.csv"), index=False
    )
    ctx.tables["mediation_summary_t3"] = med_df_t3
    print_table(
        ranked_dataframe_table(
            med_df_t3,
            title="Temporal-ordering sensitivity (outcome W at t3; NOT randomised)",
            columns=["quantity", "words_mean", "words_lo", "words_hi", "prob_pos"],
            rank_column=False,
            precision=3,
        )
    )

    _summary = {r["quantity"]: r for r in med_df.to_dict("records")}
    _summary_t3 = {r["quantity"]: r for r in med_df_t3.to_dict("records")}
    _report.write_run_metadata(
        ctx,
        extra={
            "adjustment": spec.adjustment,
            "n_obs": prepared.n_obs,
            "mediation": _summary,
            "mediation_t3_sensitivity": _summary_t3,
        },
    )

    section_header("Report")
    _copy_report_template(ctx)
    _print_footer(ctx)
    return ctx


# ---------------------------------------------------------------------------
# Longitudinal dynamic pipelines (LRP67 LCSM, LRP68 RI-CLPM)
# ---------------------------------------------------------------------------


def _coef_row(label: str, draws, hdi_prob: float) -> dict:
    """Posterior mean, equal-tailed central interval and ``P(coef > 0)``.

    Equal-tailed quantiles at coverage ``hdi_prob`` — the same convention as
    :func:`reporting.tau_summary_itt` (not a highest-density interval).
    """
    d = np.asarray(draws).reshape(-1)
    lo_q = (1 - hdi_prob) / 2
    return {
        "coefficient": label,
        "mean": float(np.mean(d)),
        "lo": float(np.quantile(d, lo_q)),
        "hi": float(np.quantile(d, 1 - lo_q)),
        "prob_pos": float(np.mean(d > 0)),
    }


def fit_lcsm(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Latent change-score model (LRP67).

    Fits the coupled McArdle latent change-score model with process noise and
    reports the reading-change coupling table — the within-trajectory analogue
    of LRP65's between-child predictors of reading gain.
    """
    assert spec.kind == "lcsm"

    ctx = make_context(spec, config)

    section_header("Prepare data")
    outcomes = tuple(spec.extra.get("outcomes", ("W", "L", "E")))
    reading_symbol = spec.outcome_symbol or "W"
    panel = load_wave_panel(outcomes=outcomes)
    ctx.prepared = panel

    _print_header(ctx)

    section_header("Build model")
    built = _factories.build_lcsm_model(
        panel,
        reading_symbol=reading_symbol,
        coupling_prior_sigma=spec.extra.get("coupling_prior_sigma", 0.5),
        use_process_noise=spec.extra.get("use_process_noise", True),
        shared_process_noise=spec.extra.get("shared_process_noise", False),
    )
    ctx.model = built.model
    ctx.model_vars = built.variables
    ctx.prepared = built.prepared

    _render_model_graph(ctx)

    cross = [s for s in outcomes if s != reading_symbol]
    diag_vars = [f"g_{s}" for s in cross]
    diag_vars += ["a_change", "b_self", "d_age", "d_dose", "sigma1", "kappa"]
    if spec.extra.get("use_process_noise", True):
        diag_vars.append("sigma_proc")

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000, var_names=["y_obs"])

    section_header("Sampling posterior (nutpie)")
    _diag.sample_posterior(ctx)

    section_header("LOO-PSIS")
    _diag.compute_log_likelihood_and_loo(ctx)
    _report.write_loo_summary(ctx)
    _print_loo_row(ctx)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=diag_vars)

    section_header("Posterior predictive")
    _diag.sample_posterior_predictive(ctx, var_names=["y_obs"])
    _save_ppc(ctx)
    _diag.save_trace(ctx)

    # Reading-change coupling table — the headline "what predicts reading
    # change" output and the basis for the LRP65 (between-child) sanity-check.
    section_header("Reading-change coupling summary")
    post = ctx.trace.posterior
    rows = [
        _coef_row(
            f"g_{s} (prior {s} -> {reading_symbol} change)",
            post[f"g_{s}"].values,
            ctx.reporting.hdi,
        )
        for s in cross
    ]
    for name, label in (
        ("b_self", f"b_self[{reading_symbol}] (reading self-feedback)"),
        ("a_change", f"a_change[{reading_symbol}] (reading baseline change)"),
        ("d_age", f"d_age[{reading_symbol}] (age -> reading change)"),
        ("d_dose", f"d_dose[{reading_symbol}] (dose -> reading change)"),
    ):
        rows.append(
            _coef_row(label, post[name].sel(outcome=reading_symbol).values, ctx.reporting.hdi)
        )
    coupling_df = pd.DataFrame(rows)
    coupling_df.to_csv(os.path.join(ctx.output_dir, "coupling_summary.csv"), index=False)
    ctx.tables["coupling_summary"] = coupling_df
    print_table(
        ranked_dataframe_table(
            coupling_df,
            title=(
                f"Reading-change couplings - {int(ctx.reporting.hdi * 100)}% CI "
                "(equal-tailed)"
            ),
            columns=["coefficient", "mean", "lo", "hi", "prob_pos"],
            rank_column=False,
            precision=3,
        )
    )

    _report.write_run_metadata(
        ctx,
        extra={
            "loo_elpd": float(ctx.loo.elpd),
            "outcomes": list(outcomes),
            "reading_symbol": reading_symbol,
            "coupling_summary": rows,
        },
    )

    section_header("Report")
    _copy_report_template(ctx)
    _print_footer(ctx)
    return ctx


def fit_riclpm(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Constrained RI-CLPM (LRP68) with the competing-structure LOO comparison.

    Fits the AR-only / L->R / R-driven / reciprocal structures, compares them by
    PSIS-LOO (does any cross-lagged structure beat AR-only?), then runs full
    diagnostics on the headline (reciprocal) model plus a cross-lagged prior
    sensitivity sweep on the L->reading path.
    """
    assert spec.kind == "riclpm"

    ctx = make_context(spec, config)

    section_header("Prepare data")
    outcomes = tuple(spec.extra.get("outcomes", ("W", "L", "E")))
    reading_symbol = spec.outcome_symbol or "W"
    letter_symbol = spec.extra.get("letter_symbol", "L")
    structures = list(
        spec.extra.get("structures", ["ar", "l_to_r", "r_driven", "reciprocal"])
    )
    headline = spec.extra.get("headline_structure", "reciprocal")
    cross_sd = float(spec.extra.get("cross_prior_sigma", 0.5))
    panel = load_wave_panel(outcomes=outcomes)
    ctx.prepared = panel

    _print_header(ctx)

    def _fit(structure: str, cross_prior_sigma: float):
        built = _factories.build_riclpm_model(
            panel,
            structure=structure,
            reading_symbol=reading_symbol,
            letter_symbol=letter_symbol,
            cross_prior_sigma=cross_prior_sigma,
        )
        s = ctx.sampling
        with built.model:
            tr = pm.sample(
                draws=s.draws,
                tune=s.tune,
                chains=s.chains,
                cores=s.cores,
                target_accept=s.target_accept,
                nuts_sampler="nutpie",
                return_inferencedata=True,
                random_seed=s.random_seed,
                progressbar=False,
            )
            tr = pm.compute_log_likelihood(tr, progressbar=False)
        return built, tr

    section_header("Competing structures + LOO-PSIS")
    loos: dict[str, object] = {}
    divergences: dict[str, int] = {}
    headline_built = None
    headline_trace = None
    for st in structures:
        built, tr = _fit(st, cross_sd)
        loos[st] = az.loo(tr)
        divergences[st] = int(tr.sample_stats["diverging"].sum())
        rprint(
            f"  structure [bold]{st}[/bold]: elpd={loos[st].elpd:.1f} "
            f"(p_loo={loos[st].p:.1f}), divergences={divergences[st]}"
        )
        if st == headline:
            headline_built, headline_trace = built, tr

    if headline_built is None:
        headline_built, headline_trace = _fit(headline, cross_sd)
        loos[headline] = az.loo(headline_trace)

    ctx.model = headline_built.model
    ctx.model_vars = headline_built.variables
    ctx.trace = headline_trace
    ctx.loo = loos[headline]
    _render_model_graph(ctx)

    # LOO comparison table + "does any cross-lagged structure beat AR-only?"
    cmp = az.compare(loos)
    cmp.to_csv(os.path.join(ctx.output_dir, "loo_compare.csv"))
    ctx.tables["loo_compare"] = cmp
    cmp_disp = cmp.rename_axis("structure").reset_index()
    print_table(
        ranked_dataframe_table(
            cmp_disp,
            title="RI-CLPM structures by LOO (rank 0 = best predictive)",
            columns=[
                c
                for c in ["structure", "rank", "elpd", "p", "elpd_diff", "dse", "weight"]
                if c in cmp_disp.columns
            ],
            rank_column=False,
            precision=2,
        )
    )
    beats_ar: dict[str, dict] = {}
    if "ar" in loos:
        for st in structures:
            if st == "ar":
                continue
            d = _report.loo_delta(loos[st], loos["ar"])
            beats_ar[st] = d
            verdict = "favours cross-lagged" if d["d_elpd"] > 0 else "no gain over AR"
            rprint(
                f"  {st} vs AR-only: dELPD={d['d_elpd']:+.1f} "
                f"(dSE={d['d_se']:.1f}) - {verdict}"
            )

    section_header(f"Summary diagnostics (headline: {headline})")
    diag_vars = ["A", "mu", "sigma_u", "sigma_w1", "sigma_inn", "d_age", "d_dose", "kappa"]
    _diag.summary_diagnostics(ctx, var_names=diag_vars)

    section_header("Posterior predictive")
    _diag.sample_posterior_predictive(ctx, var_names=["y_obs"])
    _save_ppc(ctx)
    _diag.save_trace(ctx)

    # Cross-lagged path summary (headline) — A[target <- source]; the headline
    # is A[reading <- letter] (letter-sounds -> reading within child).
    section_header("Cross-lagged paths (headline)")
    post = ctx.trace.posterior
    cross_rows = []
    for tgt in outcomes:
        for src in outcomes:
            if tgt == src:
                continue
            label = f"A[{tgt}<-{src}]"
            if tgt == reading_symbol and src == letter_symbol:
                label += " (headline: letter-sounds -> reading)"
            cross_rows.append(
                _coef_row(label, post["A"].sel(outcome=tgt, outcome2=src).values, ctx.reporting.hdi)
            )
    cross_df = pd.DataFrame(cross_rows)
    cross_df.to_csv(os.path.join(ctx.output_dir, "cross_lagged_summary.csv"), index=False)
    ctx.tables["cross_lagged_summary"] = cross_df
    print_table(
        ranked_dataframe_table(
            cross_df,
            title=f"Cross-lagged paths - {int(ctx.reporting.hdi * 100)}% CI (equal-tailed)",
            columns=["coefficient", "mean", "lo", "hi", "prob_pos"],
            rank_column=False,
            precision=3,
        )
    )

    # Prior sensitivity on the cross-lagged prior SD for the headline A[W<-L].
    section_header(
        f"Cross-lagged prior sensitivity (A[{reading_symbol}<-{letter_symbol}])"
    )
    sens_rows = [
        _coef_row(
            f"cross_prior_sigma={cross_sd:g}",
            post["A"].sel(outcome=reading_symbol, outcome2=letter_symbol).values,
            ctx.reporting.hdi,
        )
    ]
    for sd in spec.extra.get("prior_sensitivity_sigmas", [0.3, 0.7]):
        _, tr_s = _fit(headline, float(sd))
        sens_rows.append(
            _coef_row(
                f"cross_prior_sigma={float(sd):g}",
                tr_s.posterior["A"]
                .sel(outcome=reading_symbol, outcome2=letter_symbol)
                .values,
                ctx.reporting.hdi,
            )
        )
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(os.path.join(ctx.output_dir, "prior_sensitivity.csv"), index=False)
    ctx.tables["prior_sensitivity"] = sens_df
    print_table(
        ranked_dataframe_table(
            sens_df,
            title=f"A[{reading_symbol}<-{letter_symbol}] prior sensitivity",
            columns=["coefficient", "mean", "lo", "hi", "prob_pos"],
            rank_column=False,
            precision=3,
        )
    )

    _report.write_run_metadata(
        ctx,
        extra={
            "loo_elpd": float(ctx.loo.elpd),
            "headline_structure": headline,
            "structures": structures,
            "divergences": divergences,
            "beats_ar": beats_ar,
            "outcomes": list(outcomes),
            "cross_lagged_summary": cross_rows,
            "prior_sensitivity": sens_rows,
        },
    )

    section_header("Report")
    _copy_report_template(ctx)
    _print_footer(ctx)
    return ctx
