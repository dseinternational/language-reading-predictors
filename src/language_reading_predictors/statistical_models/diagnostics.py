# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared sampling / diagnostics helpers for the statistical models.

Each LRP model runs the same diagnostic suite:

1. Prior predictive check (1000 draws). The prior + prior-predictive groups are
   persisted onto ``trace.nc`` (issue #125 step 0b) so the report can show
   prior-predictive checks and prior-vs-posterior overlays without recomputation.
2. Sampling via NUTS (nutpie backend).
3. Summary diagnostics (R-hat, ESS over the scalar parameters; a separate
   summary for deterministics / HSGP basis weights), trace / energy / posterior
   plots, and a ``diagnostics_summary.json`` pass/fail convergence verdict
   (divergences, BFMI, R-hat, ESS) that the report renders *first*.
4. LOO-PSIS via ArviZ (pointwise, so Pareto-k bands are available) and a
   ``log_prior`` group for power-scaling prior sensitivity.
5. Posterior predictive draws, plus the extended diagnostics (Pareto-k, rank,
   ESS-evolution, LOO-PIT).

Everything is written to ``context.output_dir`` and the trace persisted as
``trace.nc`` (NetCDF, an ``xarray`` DataTree).

ArviZ note: this is the ArviZ 1.x split stack (``arviz`` / ``arviz_plots`` /
``arviz_stats``). Legacy ``az.plot_ppc`` / ``plot_posterior`` do not exist; plots
go through ``arviz_plots`` and return a ``PlotCollection`` saved via ``.savefig``.
Every plot / extra-diagnostic call is guarded so a backend or API hiccup degrades
to a warning rather than aborting the fit — the numeric summaries are the
substantive output.
"""

from __future__ import annotations

import os
import re

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from rich import print as rprint

from dse_research_utils.statistics.diagnostics import (
    BFMI_THRESHOLD,
    ESS_THRESHOLD,
    RHAT_MAX,
    _bfmi_per_chain,
)
from dse_research_utils.statistics.diagnostics import (
    write_diagnostics_summary as _shared_write_diagnostics_summary,
)

from language_reading_predictors.models._reporting import (
    print_table,
    ranked_dataframe_table,
)
from language_reading_predictors.statistical_models.context import (
    StatisticalFitContext,
)

# Convergence-gate thresholds (issue #125 Area 3; Vehtari et al. 2021 for R-hat)
# and the per-chain BFMI helper are now owned by the shared package and
# re-exported here so existing call sites and tests keep their import paths.
__all__ = ["RHAT_MAX", "ESS_THRESHOLD", "BFMI_THRESHOLD", "_bfmi_per_chain"]


def run_prior_predictive(
    context: StatisticalFitContext,
    draws: int = 1000,
    var_names: list[str] | None = None,
) -> None:
    """Draw from the prior + prior predictive into ``context.prior_samples``.

    When ``var_names`` is ``None`` (the default for every family now), the prior
    is sampled over *all* free RVs, deterministics, and observed nodes of the
    model. This makes the persisted ``prior`` group rich enough for prior-vs-
    posterior overlays and the prior pushforward (it carries the effect term and
    ``eta``), and the ``prior_predictive`` group carries the outcome node — at no
    extra cost beyond the draws already taken. Falls back to a minimal
    observed + ``eta`` set if the full draw fails.
    """
    model = context.model
    if var_names is None:
        names: list[str] = []
        names += [rv.name for rv in model.free_RVs]
        names += [d.name for d in model.deterministics]
        names += [rv.name for rv in model.observed_RVs]
        var_names = list(dict.fromkeys(names))  # de-dupe, preserve order
    try:
        with model:
            prior = pm.sample_prior_predictive(
                draws=draws,
                var_names=var_names,
                random_seed=context.sampling.random_seed,
            )
    except Exception as exc:  # pragma: no cover - defensive fallback
        rprint(f"[yellow]Full prior-predictive draw failed ({exc}); retrying minimal set[/yellow]")
        fallback = [rv.name for rv in model.observed_RVs]
        if any(d.name == "eta" for d in model.deterministics):
            fallback.append("eta")
        with model:
            prior = pm.sample_prior_predictive(
                draws=draws,
                var_names=fallback,
                random_seed=context.sampling.random_seed,
            )
    context.prior_samples = prior


def sample_posterior(context: StatisticalFitContext) -> None:
    s = context.sampling
    with context.model:
        trace = pm.sample(
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
    context.trace = trace


def compute_log_likelihood_and_loo(context: StatisticalFitContext) -> None:
    """Add log-likelihood + log-prior groups and compute pointwise LOO.

    The LOO is computed ``pointwise=True`` so the per-observation Pareto-k
    diagnostics survive on ``context.loo`` for the report's Pareto-k bands
    (load-bearing at n ≈ 33–54, where one influential child can drive elpd). The
    ``log_prior`` group is added (guarded) so power-scaling prior sensitivity
    (``arviz_stats.psense``) is reachable from the persisted trace.
    """
    with context.model:
        context.trace = pm.compute_log_likelihood(context.trace)
        try:
            from pymc.stats import compute_log_prior

            context.trace = compute_log_prior(context.trace)
        except Exception as exc:  # pragma: no cover - psense is secondary
            rprint(f"[yellow]log_prior group skipped: {exc}[/yellow]")
    context.loo = az.loo(context.trace, pointwise=True)


def _interval_cols(columns) -> list[str]:
    """Return the credible-interval column names in an ``az.summary`` frame.

    ArviZ 1.x names equal-tailed columns ``eti95_lb`` / ``eti95_ub`` (and HDI
    ``hdi_3%`` style), so match the ``eti``/``hdi`` prefix with an optional
    coverage number before the separator.
    """
    return [c for c in columns if re.match(r"^(eti|hdi)\d*_", str(c))]


def summary_diagnostics(
    context: StatisticalFitContext,
    var_names: list[str] | None = None,
    max_vars_for_pairs: int = 8,
) -> None:
    out = context.output_dir
    os.makedirs(out, exist_ok=True)

    # Narrow to scalar RVs by default so the summary table is readable.
    if var_names is None:
        scalar_vars = []
        for rv in context.model.unobserved_RVs:
            try:
                if int(np.prod(rv.shape.eval())) <= 2:
                    scalar_vars.append(rv.name)
            except Exception:
                continue
        var_names = scalar_vars

    if var_names:
        # Equal-tailed central interval (issue #125 0c / #101): the report cards
        # and prose use equal-tailed quantiles, so the diagnostics table must too
        # — ``ci_kind="hdi"`` here was inconsistent with the rest of the suite.
        summary = az.summary(
            context.trace,
            var_names=var_names,
            round_to=3,
            ci_prob=context.reporting.hdi,
            ci_kind="eti",
        )
        summary.to_csv(os.path.join(out, "diagnostics.csv"))
        context.tables["diagnostics"] = summary

        ci_pct = int(round(context.reporting.hdi * 100))
        interval_cols = _interval_cols(summary.columns)
        wanted = [
            c
            for c in ["mean", "sd", *interval_cols, "ess_bulk", "ess_tail", "r_hat"]
            if c in summary.columns
        ]
        display_df = summary.reset_index().rename(columns={"index": "variable"})
        print_table(
            ranked_dataframe_table(
                display_df,
                title=f"Posterior diagnostics (equal-tailed {ci_pct}%)",
                columns=["variable", *wanted],
                rank_column=False,
                precision=3,
            )
        )

    # Deterministics / HSGP basis-weight summary (issue #125 Area 3): the
    # ``var_names is None`` autodetect above keeps only scalars, so GP-bearing
    # variants never get their basis weights summarised. Emit those to a separate
    # CSV so the scalar table stays readable. Guarded and best-effort.
    _summarise_deterministics(context, scalar_var_names=var_names)

    # Diagnostic plots. arviz 1.x routes plotting through arviz_plots, whose
    # functions return a PlotCollection saved via ``.savefig`` (not the old
    # matplotlib ``plt.savefig``) and drop several 0.x kwargs (``combined``,
    # ``kind``, ``divergences``). Each plot is guarded so a backend/API hiccup
    # degrades to a warning rather than failing the fit — the numeric summary
    # above is the substantive output.
    import arviz_plots as azp

    # Plots use a draw-thinned view (visually identical, but the full reporting
    # trace can be very slow / hang for these routines); diagnostics.csv above
    # used the full trace.
    tr = thin_for_plots(context.trace)

    _save_pc(
        out,
        lambda: azp.plot_trace(tr, var_names=var_names or None),
        "trace_plot.png",
    )
    _save_pc(out, lambda: azp.plot_energy(tr), "energy_plot.png")

    if var_names:
        _save_pc(
            out,
            lambda: azp.plot_dist(
                tr,
                var_names=var_names,
                group="posterior",
                ci_prob=context.reporting.hdi,
            ),
            "posterior_plot.png",
        )
        if len(var_names) <= max_vars_for_pairs:
            _save_pc(
                out,
                lambda: azp.plot_pair(tr, var_names=var_names),
                "pair_plot.png",
            )


def _save_pc(out: str, make, name: str) -> None:
    """Build a PlotCollection and save it, degrading to a warning on failure."""
    try:
        pc = make()
        pc.savefig(os.path.join(out, name), dpi=300)
        plt.close("all")
    except Exception as exc:  # pragma: no cover
        rprint(f"[yellow]{name} skipped: {exc}[/yellow]")


def thin_for_plots(trace, max_draws: int = 1000):
    """Return a draw-thinned view of the trace for plotting at scale.

    Several ``arviz_plots`` routines — ``plot_rank`` in particular — are
    pathologically slow (effectively hang) on a reporting-config trace
    (6000 draws × 6 chains = 36k draws), while running fine at dev scale (~1k).
    The numeric summaries always use the full trace; the diagnostic *plots* are
    visually identical on a thinned view, so thin the draw dimension so that
    chain × draw ≲ ``max_draws`` before plotting. Guarded — returns the original
    trace if anything about the structure is unexpected.
    """
    try:
        post = trace.posterior
        total = int(post.sizes.get("chain", 1)) * int(post.sizes.get("draw", 1))
        if total <= max_draws:
            return trace
        k = int(np.ceil(total / max_draws))
        return trace.isel(draw=slice(None, None, k))
    except Exception:  # pragma: no cover
        return trace


def _summarise_deterministics(
    context: StatisticalFitContext, scalar_var_names: list[str] | None
) -> None:
    """Write ``diagnostics_deterministics.csv`` for vector / GP-weight nodes."""
    try:
        scalar = set(scalar_var_names or [])
        det_names = [d.name for d in context.model.deterministics if d.name not in scalar]
        # Only summarise nodes actually present in the posterior, and skip the
        # per-observation ``eta`` (n_obs rows) which would dominate the table.
        present = [
            n
            for n in det_names
            if n in context.trace.posterior and n not in ("eta",)
        ]
        if not present:
            return
        summary = az.summary(
            context.trace, var_names=present, round_to=3, ci_kind="eti"
        )
        summary.to_csv(os.path.join(context.output_dir, "diagnostics_deterministics.csv"))
        context.tables["diagnostics_deterministics"] = summary
    except Exception as exc:  # pragma: no cover
        rprint(f"[yellow]deterministics summary skipped: {exc}[/yellow]")


def write_diagnostics_summary(
    context: StatisticalFitContext,
    *,
    var_names: list[str] | None = None,
) -> dict:
    """Emit ``diagnostics_summary.json`` — the report's pass/fail convergence gate.

    Thin wrapper over :func:`dse_research_utils.statistics.diagnostics.write_diagnostics_summary`
    so the convergence gate (and its JSON schema) is defined once across DSE
    projects. The shared implementation evaluates the gate on *unrounded* R-hat
    (``round_to=None`` — a borderline 1.01004 would otherwise round to 1.0100 and
    slip through) and treats a non-finite per-chain BFMI as a failure rather than
    letting it pass order-dependently. Written unconditionally for every family
    (incl. mediation, which has no LOO) so the report's banner always renders.
    """
    return _shared_write_diagnostics_summary(
        context.trace,
        context.output_dir,
        var_names=var_names,
        tables=context.tables,
    )


def run_extended_diagnostics(
    context: StatisticalFitContext,
    *,
    causal_term: str | None = None,
) -> None:
    """Pareto-k, rank, ESS-evolution and LOO-PIT plots (issue #125 Area 3).

    Called after posterior-predictive sampling so all groups are present. Pareto-k
    reuses ``context.loo`` (computed ``pointwise=True``); rank focuses on the
    causal term; LOO-PIT needs the posterior-predictive group. All guarded.
    """
    out = context.output_dir
    import arviz_plots as azp

    # Pareto-k reads context.loo (per-observation, not draws) — full trace, fast.
    if context.loo is not None:
        _save_pc(out, lambda: azp.plot_khat(context.loo), "pareto_k.png")

    # Draw-based plots use a thinned view (full trace hangs plot_rank at reporting
    # scale; thinning is visually identical and reproduces the fast dev path).
    tr = thin_for_plots(context.trace)

    if causal_term is not None and causal_term in context.trace.posterior:
        _save_pc(
            out,
            lambda: azp.plot_rank(tr, var_names=[causal_term]),
            "rank_plot.png",
        )

    _save_pc(
        out,
        lambda: azp.plot_ess_evolution(
            tr,
            var_names=[causal_term] if causal_term else None,
            min_ess=ESS_THRESHOLD,
        ),
        "ess_evolution.png",
    )

    try:
        if "posterior_predictive" in context.trace.children:
            _save_pc(out, lambda: azp.plot_loo_pit(tr), "loo_pit.png")
    except Exception as exc:  # pragma: no cover
        rprint(f"[yellow]loo_pit skipped: {exc}[/yellow]")


def save_prior_posterior_plot(
    context: StatisticalFitContext,
    *,
    var_names: list[str] | None = None,
) -> None:
    """Prior-vs-posterior overlay for the key parameters (issue #125 Area 1).

    Needs the ``prior`` group on the trace (attached by :func:`save_trace`), so
    call this *after* ``save_trace``. Shows how far the data moved each parameter
    from its prior. Guarded.
    """
    out = context.output_dir
    import arviz_plots as azp

    tr = thin_for_plots(context.trace)
    _save_pc(
        out,
        lambda: azp.plot_prior_posterior(tr, var_names=var_names),
        "prior_posterior.png",
    )


def run_psense(
    context: StatisticalFitContext,
    *,
    var_names: list[str],
) -> None:
    """Power-scaling prior/likelihood sensitivity (issue #125 Area 1, secondary).

    Writes ``psense_summary.csv`` and ``psense.png`` for the named parameters
    (usually the causal term). Requires the ``log_prior`` and ``log_likelihood``
    groups added by :func:`compute_log_likelihood_and_loo`. Guarded — a missing
    group or an API mismatch degrades to a warning (psense is recommended-but-
    secondary at this n). Kallioinen et al. 2024.
    """
    out = context.output_dir
    tr = thin_for_plots(context.trace)
    try:
        import arviz_stats as azs

        s = azs.psense_summary(tr, var_names=var_names)
        if hasattr(s, "to_dataframe"):
            df = s.to_dataframe()
        else:
            import pandas as pd

            df = pd.DataFrame(s)
        df.to_csv(os.path.join(out, "psense_summary.csv"))
        context.tables["psense_summary"] = df
    except Exception as exc:  # pragma: no cover
        rprint(f"[yellow]psense_summary skipped: {exc}[/yellow]")

    import arviz_plots as azp

    _save_pc(
        out,
        lambda: azp.plot_psense_dist(tr, var_names=var_names),
        "psense.png",
    )


def sample_posterior_predictive(
    context: StatisticalFitContext,
    var_names: list[str] | None = None,
) -> None:
    with context.model:
        context.trace = pm.sample_posterior_predictive(
            context.trace,
            var_names=var_names,
            extend_inferencedata=True,
            random_seed=context.sampling.random_seed,
            progressbar=False,
        )


def _attach_prior_groups(context: StatisticalFitContext) -> None:
    """Graft the prior + prior_predictive groups onto the trace before saving.

    ``run_prior_predictive`` stores 1000 prior draws on ``context.prior_samples``;
    previously they were discarded (issue #125 step 0b). Copy the ``prior`` and
    ``prior_predictive`` subtrees onto ``context.trace`` (an ``xarray`` DataTree)
    so ``trace.nc`` carries them for prior-predictive checks and prior-vs-
    posterior overlays. Guarded — a merge failure must not lose the trace.
    """
    if context.prior_samples is None or context.trace is None:
        return
    for group in ("prior", "prior_predictive"):
        try:
            if group in context.prior_samples.children and group not in context.trace.children:
                context.trace[group] = context.prior_samples[group]
        except Exception as exc:  # pragma: no cover
            rprint(f"[yellow]Could not attach {group} group to trace: {exc}[/yellow]")


def save_trace(context: StatisticalFitContext, filename: str = "trace.nc") -> str:
    _attach_prior_groups(context)
    path = os.path.join(context.output_dir, filename)
    context.trace.to_netcdf(path)
    return path


def save_prior_predictive_plot(
    context: StatisticalFitContext,
    outcome_symbol: str,
    *,
    node: str | None = None,
) -> None:
    """Surface the prior-predictive check in the report (#127 / #125 Area 2).

    Overlays the prior-predictive distribution of the outcome count against the
    observed counts and writes ``prior_predictive_check.png``. ``node`` defaults
    to the model's observed node (so this works for every family, not just GF/LF).
    A rootogram is added when the count outcome makes one meaningful. Guarded — a
    plotting failure must not abort the fit.
    """
    if context.prior_samples is None or context.prepared is None:
        rprint("[yellow]No prior samples to plot[/yellow]")
        return
    if node is None:
        try:
            node = context.model.observed_RVs[0].name
        except Exception:
            node = "y_post"
    try:
        pp = context.prior_samples.prior_predictive[node]
        rep = np.asarray(pp.values, dtype=float).ravel()
        obs = np.asarray(context.prepared.post_counts[outcome_symbol], dtype=float)
        obs = obs[np.isfinite(obs)]
        if node == "y_offfloor":
            obs = (obs > 0).astype(float)  # off-floor indicator, to match the node
        plt.figure(figsize=(6, 4))
        plt.hist(
            rep, bins=40, density=True, color="#1f77b4", alpha=0.55,
            label="prior predictive",
        )
        plt.hist(obs, bins=20, density=True, color="#d62728", alpha=0.55, label="observed")
        plt.xlabel(f"{outcome_symbol} count")
        plt.ylabel("density")
        plt.title("Prior-predictive check")
        plt.legend()
        plt.savefig(
            os.path.join(context.output_dir, "prior_predictive_check.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    except Exception as exc:  # pragma: no cover
        rprint(f"[yellow]Prior-predictive plot failed: {exc}[/yellow]")
