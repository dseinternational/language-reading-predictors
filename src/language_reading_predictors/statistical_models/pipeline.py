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
    """Render the LOO ELPD / p_loo / looic summary as a small table."""
    if ctx.loo is None:
        return
    rows = [
        {"metric": "elpd_loo", "value": float(ctx.loo.elpd_loo)},
        {"metric": "se", "value": float(ctx.loo.se)},
        {"metric": "p_loo", "value": float(ctx.loo.p_loo)},
    ]
    looic = getattr(ctx.loo, "looic", None)
    if looic is not None:
        rows.append({"metric": "looic", "value": float(looic)})
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

    section_header("Prepare data")
    prepared = load_and_prepare(phase_mode="itt")
    ctx.prepared = prepared

    _print_header(ctx)

    section_header("Build model")
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

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000, var_names=["y_post", "eta"])

    section_header("Sampling posterior (nutpie)")
    _diag.sample_posterior(ctx)

    section_header("LOO-PSIS")
    _diag.compute_log_likelihood_and_loo(ctx)
    _report.write_loo_summary(ctx)
    _print_loo_row(ctx)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(
        ctx, var_names=["alpha", "tau", "gamma_own", "kappa"]
    )

    section_header("Posterior predictive")
    _diag.sample_posterior_predictive(ctx, var_names=["y_post"])
    _save_ppc(ctx)

    _diag.save_trace(ctx)

    # Treatment-effect summary on both scales.
    section_header("Treatment-effect summary")
    summary = _extract_alpha_gamma_means(ctx)
    tau_s = _report.tau_summary_itt(
        ctx.trace,
        hdi_prob=ctx.reporting.hdi,
        pre_logit_mean=float(prepared.pre_logit[spec.outcome_symbol].mean()),
        gamma_own_mean=summary["gamma_own"],
        alpha_mean=summary["alpha"],
    )
    tau_df = pd.DataFrame([tau_s])
    tau_df.to_csv(os.path.join(ctx.output_dir, "tau_summary.csv"), index=False)
    ctx.tables["tau_summary"] = tau_df
    print_table(
        metrics_table(
            [{"metric": k, "value": v} for k, v in tau_s.items()],
            title=f"tau ({spec.outcome_symbol}) - HDI {int(ctx.reporting.hdi * 100)}%",
            columns=["metric", "value"],
        )
    )

    _report.write_run_metadata(
        ctx,
        extra={"loo_elpd": float(ctx.loo.elpd_loo), "tau_summary": tau_s},
    )

    section_header("Report")
    _copy_report_template(ctx)
    _print_footer(ctx)
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

    section_header("Prepare data")
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
            title=f"tau by outcome - HDI {int(ctx.reporting.hdi * 100)}%",
            columns=["outcome", "tau_mean", "tau_lo", "tau_hi", "prob_pos"],
            rank_column=False,
        )
    )

    contrast = _report.tau_contrast_matrix(ctx.trace, outcomes)
    contrast.to_csv(os.path.join(ctx.output_dir, "tau_contrast_matrix.csv"))
    ctx.tables["tau_contrast_matrix"] = contrast

    _report.write_run_metadata(ctx, extra={"loo_elpd": float(ctx.loo.elpd_loo)})

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
    prepared = load_and_prepare(phase_mode="all")
    ctx.prepared = prepared

    _print_header(ctx)

    section_header("Build model")
    _priors.save_shared_prior_panel(ctx.output_dir)

    confounders = [s for s in spec.adjustment if s not in ("W_pre",)]

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
    _diag.summary_diagnostics(ctx, var_names=_mech_vars)

    section_header("Posterior predictive")
    _diag.sample_posterior_predictive(ctx, var_names=["y_post"])
    _save_ppc(ctx)

    # Mechanism curve: f_mech vs mech_post_logit grid, on both scales.
    section_header("Mechanism curve")
    _write_mechanism_curve(ctx)

    _diag.save_trace(ctx)
    _report.write_run_metadata(
        ctx,
        extra={"loo_elpd": float(ctx.loo.elpd_loo), "adjustment": spec.adjustment},
    )

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
