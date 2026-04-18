# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
End-to-end fit pipeline for LRP52-LRP58.

``fit_itt(spec, config)`` is the entry point for the LRP52/53/54 ITT models.
``fit_joint(spec, config)`` is the entry point for LRP55.
``fit_mechanism(spec, config)`` is the entry point for LRP56/57/58.

Each pipeline:

1. Loads data via :func:`preprocessing.load_and_prepare`.
2. Builds the PyMC model via the appropriate factory.
3. Writes prior-panel plots.
4. Runs prior predictive, posterior sampling (nutpie), LOO, posterior
   predictive.
5. Saves ``trace.nc``, ``config.json``, ``metrics.json`` and the standard
   diagnostic plots to ``output/statistical_models/{model_id}-{config}/``.
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
from rich import print as rprint

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
)


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


def _copy_report_template(context: StatisticalFitContext) -> None:
    src = os.path.join(DOCS_DIR, "models", context.spec.model_id, "index.qmd")
    dst = os.path.join(context.output_dir, "index.qmd")
    if os.path.exists(src):
        shutil.copy(src, dst)
        rprint(f"[green]Report template copied to {dst}[/green]")
    else:
        rprint(f"[yellow]No report template found at {src}[/yellow]")


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
    try:
        total_pp = int(
            context.trace.posterior_predictive.sizes.get("chain", 1)
            * context.trace.posterior_predictive.sizes.get("draw", 1)
        )
        num = max(1, min(100, total_pp))
        az.plot_ppc(context.trace, num_pp_samples=num)
        plt.savefig(
            os.path.join(context.output_dir, "posterior_predictive_check.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    except Exception as exc:  # pragma: no cover
        rprint(f"[yellow]PPC plot failed: {exc}[/yellow]")


# ---------------------------------------------------------------------------
# ITT pipeline (LRP52 / LRP53 / LRP54)
# ---------------------------------------------------------------------------


def fit_itt(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    assert spec.kind == "itt"
    assert spec.outcome_symbol is not None

    ctx = make_context(spec, config)
    rprint(f"\n[bold green]{spec.banner}[/bold green]\n")

    prepared = load_and_prepare(phase_mode="itt")
    ctx.prepared = prepared

    _priors.save_shared_prior_panel(ctx.output_dir)

    built = _factories.build_itt_model(
        prepared,
        outcome_symbol=spec.outcome_symbol,
        use_age_gp=spec.extra.get("use_age_gp", True),
        use_own_baseline_gp=spec.extra.get("use_own_baseline_gp", True),
        use_varying_tau=spec.extra.get("use_varying_tau", False),
    )
    ctx.model = built.model
    ctx.model_vars = built.variables
    ctx.prepared = built.prepared

    _render_model_graph(ctx)

    rprint("[green]Prior predictive...[/green]")
    _diag.run_prior_predictive(ctx, draws=1000, var_names=["y_post", "eta"])

    rprint("[green]Sampling posterior (nutpie)...[/green]")
    _diag.sample_posterior(ctx)

    rprint("[green]LOO-PSIS...[/green]")
    _diag.compute_log_likelihood_and_loo(ctx)
    _report.write_loo_summary(ctx)

    _diag.summary_diagnostics(
        ctx, var_names=["alpha", "tau", "gamma_own", "kappa"]
    )

    rprint("[green]Posterior predictive...[/green]")
    _diag.sample_posterior_predictive(ctx, var_names=["y_post"])
    _save_ppc(ctx)

    _diag.save_trace(ctx)

    # Treatment-effect summary on both scales.
    summary = _extract_alpha_gamma_means(ctx)
    tau_s = _report.tau_summary_itt(
        ctx.trace,
        hdi_prob=ctx.reporting.hdi,
        pre_logit_mean=float(prepared.pre_logit[spec.outcome_symbol].mean()),
        gamma_own_mean=summary["gamma_own"],
        alpha_mean=summary["alpha"],
    )
    pd.DataFrame([tau_s]).to_csv(
        os.path.join(ctx.output_dir, "tau_summary.csv"), index=False
    )
    ctx.tables["tau_summary"] = pd.DataFrame([tau_s])

    _report.write_run_metadata(
        ctx,
        extra={"loo_elpd": float(ctx.loo.elpd_loo), "tau_summary": tau_s},
    )

    _copy_report_template(ctx)
    return ctx


def _extract_alpha_gamma_means(ctx: StatisticalFitContext) -> dict:
    post = ctx.trace.posterior
    return {
        "alpha": float(post["alpha"].mean()),
        "gamma_own": float(post["gamma_own"].mean()),
    }


# ---------------------------------------------------------------------------
# Joint pipeline (LRP55)
# ---------------------------------------------------------------------------


def fit_joint(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    assert spec.kind == "joint"

    ctx = make_context(spec, config)
    rprint(f"\n[bold green]{spec.banner}[/bold green]\n")

    prepared = load_and_prepare(phase_mode="itt")
    ctx.prepared = prepared

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

    rprint("[green]Prior predictive...[/green]")
    _diag.run_prior_predictive(ctx, draws=1000, var_names=["y_post"])

    rprint("[green]Sampling posterior (nutpie)...[/green]")
    _diag.sample_posterior(ctx)

    rprint("[green]LOO-PSIS...[/green]")
    _diag.compute_log_likelihood_and_loo(ctx)
    _report.write_loo_summary(ctx)

    _joint_vars = ["alpha", "tau", "kappa"]
    if spec.extra.get("use_residual_correlation", False):
        _joint_vars.append("sigma_outcome")
    _diag.summary_diagnostics(ctx, var_names=_joint_vars)
    rprint("[green]Posterior predictive...[/green]")
    _diag.sample_posterior_predictive(ctx, var_names=["y_post"])
    _save_ppc(ctx)
    _diag.save_trace(ctx)

    outcomes = list(ctx.trace.posterior["outcome"].values)
    tau_df = _report.tau_summary_joint(ctx.trace, outcomes, hdi_prob=ctx.reporting.hdi)
    tau_df.to_csv(os.path.join(ctx.output_dir, "tau_summary.csv"), index=False)
    ctx.tables["tau_summary"] = tau_df

    contrast = _report.tau_contrast_matrix(ctx.trace, outcomes)
    contrast.to_csv(os.path.join(ctx.output_dir, "tau_contrast_matrix.csv"))
    ctx.tables["tau_contrast_matrix"] = contrast

    _report.write_run_metadata(ctx, extra={"loo_elpd": float(ctx.loo.elpd_loo)})
    _copy_report_template(ctx)
    return ctx


# ---------------------------------------------------------------------------
# Mechanism pipeline (LRP56 / LRP57 / LRP58)
# ---------------------------------------------------------------------------


def fit_mechanism(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    assert spec.kind == "mechanism"
    assert spec.mechanism_symbol is not None

    ctx = make_context(spec, config)
    rprint(f"\n[bold green]{spec.banner}[/bold green]\n")

    prepared = load_and_prepare(phase_mode="all")
    ctx.prepared = prepared

    _priors.save_shared_prior_panel(ctx.output_dir)

    confounders = [s for s in spec.adjustment if s not in ("W_pre",)]

    built = _factories.build_mechanism_model(
        prepared,
        mechanism_symbol=spec.mechanism_symbol,
        outcome_symbol=spec.outcome_symbol or "W",
        adjust_baseline_symbol=spec.extra.get("adjust_baseline_symbol", "W"),
        confounder_symbols=tuple(s for s in confounders if s in ("G", "A") or len(s) == 1),
        use_age_gp=spec.extra.get("use_age_gp", True),
        phase_specific_mechanism=spec.extra.get("phase_specific_mechanism", False),
    )
    ctx.model = built.model
    ctx.model_vars = built.variables
    ctx.prepared = built.prepared

    _render_model_graph(ctx)

    _diag.run_prior_predictive(ctx, draws=1000, var_names=["y_post", "eta"])
    _diag.sample_posterior(ctx)
    _diag.compute_log_likelihood_and_loo(ctx)
    _report.write_loo_summary(ctx)

    _diag.summary_diagnostics(ctx, var_names=["alpha", "beta_G", "gamma_own", "kappa"])
    _diag.sample_posterior_predictive(ctx, var_names=["y_post"])
    _save_ppc(ctx)

    # Mechanism curve: f_mech vs mech_post_logit grid, on both scales.
    _write_mechanism_curve(ctx)

    _diag.save_trace(ctx)
    _report.write_run_metadata(
        ctx,
        extra={"loo_elpd": float(ctx.loo.elpd_loo), "adjustment": spec.adjustment},
    )
    _copy_report_template(ctx)
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
