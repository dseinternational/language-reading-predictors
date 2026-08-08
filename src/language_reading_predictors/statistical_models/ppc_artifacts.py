# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Posterior-predictive artefacts: coverage CSV, calibration panel, overlays.

The #318 posterior-predictive suite, shared by every family: it computes the
coverage statistic from the existing ``posterior_predictive`` group (no new
sampling), routes by outcome-node kind (bounded count, binary off-floor,
measurement/latent) and writes the calibration and overlay figures. Split out of
``pipeline.py`` for #394.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dse_research_utils.plot.styles import (
    COLOUR_BLUE,
    COLOUR_RED,
    FIGSIZE_LG,
)
from rich import print as rprint

from language_reading_predictors.statistical_models import reporting as _report
from language_reading_predictors.statistical_models.artifacts import (
    guard_optional,
    save_table,
)
from language_reading_predictors.statistical_models.context import (
    StatisticalFitContext,
)
from language_reading_predictors.statistical_models.measures import MEASURES
from language_reading_predictors.statistical_models.plotting import (
    save_plotcollection,
    save_styled_figure,
)


# Posterior-predictive check suite (issue #318) --------------------------------
# The stock ArviZ overlay pooled every likelihood node onto one unlabelled axis and
# offered no verdict. The redesign emits, from the existing posterior_predictive
# group (no new sampling): a computed coverage statement (ppc_summary.csv), a
# per-observation calibration panel, and — for single-measure count families — a
# relabelled distribution overlay. Floor-rule / binary nodes report off-floor RATE
# coverage by group cell instead (per-observation 0/1 interval coverage is
# degenerate). See notes/202607151942-ppc-coverage-redesign.md.

# Families whose single likelihood node flattens several measures with different
# denominators (6..170), so a shared count axis for the overlay would pool them —
# the meaninglessness #271 item 2 flagged. They still get coverage + calibration.
_PPC_MULTI_OUTCOME_KINDS = {"joint", "lcsm", "growth"}
# Of those, ``joint`` splits its own per-outcome overlays and calibration tables in
# ``fit_joint`` (``save_joint_posterior_predictive_plot``), so the generic
# per-measure dispatch would only be overwritten by them.
_PPC_FAMILY_OWN_OVERLAY_KINDS = {"joint"}
# Binary / event nodes: off-floor rate coverage rather than count intervals.
_PPC_BINARY_NODES = {"y_offfloor", "y_event"}
# Bounded-count outcome nodes that take the count-interval treatment.
_PPC_COUNT_NODES = {"y_post", "y_obs", "score"}


def save_ppc(context: StatisticalFitContext, *, primary_node: str = "y_post") -> None:
    """Write the posterior-predictive coverage CSV + figures for the primary node.

    ``primary_node`` is the outcome leg (the last node in every multi-node family's
    ``var_names``). Routes by node kind; every sub-step is independently guarded so a
    plotting hiccup never aborts the fit or loses the coverage CSV.
    """
    node = primary_node
    symbol = context.spec.outcome_symbol
    if node in _PPC_BINARY_NODES:
        _save_offfloor_ppc(context, node, symbol)
    elif node in _PPC_COUNT_NODES:
        _save_count_ppc(context, node, symbol, context.spec.kind)
    else:
        # Measurement / latent nodes (corr-factor indicators, longitudinal z-patterns,
        # a standalone mediator leg): no single count outcome, so keep the legacy
        # overlay and emit no coverage statistic.
        _save_legacy_ppc_overlay(context)


def _save_count_ppc(
    context: StatisticalFitContext, node: str, symbol: str | None, kind: str
) -> None:
    """Count-interval coverage CSV + calibration panel (+ overlay for single-measure)."""
    with guard_optional(context, "ppc_summary.csv", filename="ppc_summary.csv", kind="table"):
        cov = _report.ppc_interval_coverage(context.trace, node=node)
        save_table(context, "ppc_summary", cov, required=False)
    with guard_optional(
        context, "PPC calibration figure",
        filename="ppc_calibration.png", kind="figure", verb="skipped",
    ):
        cal = _report.ppc_calibration_table(context.trace, node=node, ci_prob=0.9)
        _ppc_calibration_figure(context, symbol, cal)
    if kind in _PPC_FAMILY_OWN_OVERLAY_KINDS:
        rprint(
            f"[dim]PPC distribution overlay left to the '{kind}' family's own "
            "per-outcome writer.[/dim]"
        )
    elif kind in _PPC_MULTI_OUTCOME_KINDS:
        _save_multi_outcome_ppc_overlays(context, node, kind)
    else:
        _ppc_overlay_figure(context, node, symbol)


def _save_multi_outcome_ppc_overlays(
    context: StatisticalFitContext, node: str, kind: str
) -> None:
    """One overlay per measure for a stacked multi-outcome likelihood.

    ``joint`` / ``lcsm`` / ``growth`` flatten every measure into a single likelihood
    node, so a pooled overlay puts scales with different maxima on one axis and has
    no interpretable predictive distribution. This was previously skipped outright
    for those families; it is emitted per measure now that the factories persist a
    cell map (``y_post_cell_outcome`` for the joint family, ``y_obs_cell_outcome``
    for the stacked LCSM / growth likelihoods), which is the same selection the
    prior-predictive checks use. The map is read back, never re-derived — a
    reconstructed index is what silently misaligned those checks before.

    The first measure keeps the unsuffixed filename the report partials expect;
    the rest are suffixed. Falls back to the pooled overlay only when no map is
    present, since that is a single-measure node reaching here by another route.
    """
    cd = getattr(context.trace, "constant_data", None)
    key = next(
        (k for k in (f"{node}_cell_outcome", "y_post_cell_outcome") if cd is not None and k in cd),
        None,
    )
    outcomes = [str(o) for o in (context.spec.extra.get("outcomes") or ())]
    if not outcomes:
        plan = getattr(context, "resolved_plan", None)
        outcomes = [str(o) for o in (getattr(plan, "outcomes", ()) or ())]
    if key is None or not outcomes:
        rprint(
            f"[yellow]PPC per-measure overlay unavailable for '{kind}' "
            f"(cell map={key!r}, outcomes={outcomes or None}); "
            "falling back to the pooled overlay.[/yellow]"
        )
        _ppc_overlay_figure(context, node, context.spec.outcome_symbol)
        return

    idx = np.asarray(cd[key].values).ravel().astype(int)
    for position, sym in enumerate(outcomes):
        if not np.any(idx == position):
            # A declared outcome with no rows in this fit's cell map. Skip rather
            # than emit an empty overlay; ``ppc_summary.csv`` still counts the rest.
            continue
        _ppc_overlay_figure(
            context,
            node,
            sym,
            row_mask=(idx == position),
            filename_stem=(
                "posterior_predictive_check"
                if position == 0
                else f"posterior_predictive_check_{sym.lower()}"
            ),
        )


def _save_offfloor_ppc(
    context: StatisticalFitContext, node: str, symbol: str | None
) -> None:
    """Off-floor RATE coverage CSV + per-cell observed-vs-predicted rate figure."""
    group = _offfloor_group_labels(context)
    with guard_optional(context, "ppc_summary.csv", filename="ppc_summary.csv", kind="table"):
        cov = _report.ppc_offfloor_rate_coverage(context.trace, node=node, group=group)
        save_table(context, "ppc_summary", cov, required=False)
    with guard_optional(
        context, "PPC off-floor figure",
        filename="posterior_predictive_check.png", kind="figure", verb="skipped",
    ):
        cells = _report.ppc_offfloor_cell_table(
            context.trace, node=node, group=group, ci_prob=0.9
        )
        _ppc_offfloor_figure(context, symbol, cells)


def _offfloor_group_labels(context: StatisticalFitContext) -> np.ndarray | None:
    """Arm (× wave, when present) cell labels for off-floor coverage, or None.

    Reads ``prepared.G`` (0=waitlist, 1=immediate) and, when aligned, ``prepared.phase``
    so the off-floor rate is checked by group × wave cell. Returns None when no group
    is available (the coverage helper then uses one overall cell).
    """
    prep = context.prepared
    G = getattr(prep, "G", None)
    if G is None:
        return None
    G = np.asarray(G)
    arm = np.where(G == 1, "immediate", "waitlist")
    phase = getattr(prep, "phase", None)
    if phase is not None and np.asarray(phase).shape[0] == G.shape[0]:
        phase = np.asarray(phase)
        return np.array([f"t{int(p) + 1}·{a}" for p, a in zip(phase, arm, strict=True)])
    return arm


def _ppc_measure_label(symbol: str | None) -> tuple[str, int | None]:
    """Human label + denominator for the PPC axes (falls back gracefully)."""
    measure = MEASURES.get(symbol) if symbol else None
    if measure is not None:
        return measure.label, int(measure.n_trials)
    return (symbol or "outcome"), None


def _ppc_overlay_figure(
    context: StatisticalFitContext,
    node: str,
    symbol: str | None,
    *,
    row_mask: np.ndarray | None = None,
    filename_stem: str = "posterior_predictive_check",
) -> None:
    """Relabelled observed-vs-simulated distribution overlay on a labelled items axis.

    The observed count density (black) against the posterior-predictive band (blue:
    pointwise 5-95% of replicate-dataset densities, plus the median). Each replicate
    dataset is one posterior-predictive draw over all observations. Writes
    ``posterior_predictive_check.png`` (+ a density-band data CSV).
    """
    with guard_optional(
        context,
        "PPC overlay figure",
        filename=f"{filename_stem}.png",
        kind="figure",
        verb="failed",
    ):
        y_rep, y_obs = _report._ppc_node_arrays(context.trace, node)
        if row_mask is not None:
            # One measure's rows out of a stacked likelihood, selected by the
            # factory-persisted cell map (never a re-derived index).
            y_rep, y_obs = y_rep[row_mask], y_obs[row_mask]
        finite = np.isfinite(y_obs)
        y_rep, y_obs = y_rep[finite], y_obs[finite]
        label, n_trials = _ppc_measure_label(symbol)
        hi = int(n_trials) if n_trials else int(max(y_obs.max(), y_rep.max()))
        bins = np.arange(0, hi + 2) - 0.5  # integer-centred bins
        centers = 0.5 * (bins[:-1] + bins[1:])
        obs_dens, _ = np.histogram(y_obs, bins=bins, density=True)
        n_samples = y_rep.shape[1]
        idx = np.unique(np.linspace(0, n_samples - 1, min(n_samples, 200)).astype(int))
        rep_dens = np.stack(
            [np.histogram(y_rep[:, s], bins=bins, density=True)[0] for s in idx]
        )
        lo_band = np.quantile(rep_dens, 0.05, axis=0)
        hi_band = np.quantile(rep_dens, 0.95, axis=0)
        med_band = np.median(rep_dens, axis=0)
        plt.figure(figsize=FIGSIZE_LG)
        plt.fill_between(
            centers, lo_band, hi_band, color=COLOUR_BLUE, alpha=0.3,
            label="posterior-predictive 90% band",
        )
        plt.plot(centers, med_band, color=COLOUR_BLUE, lw=1.2, alpha=0.85,
                 label="posterior-predictive median")
        plt.plot(centers, obs_dens, color="black", lw=2, label="observed")
        axis_lbl = f"{label} — score (0–{hi} items)" if n_trials else f"{label} — score"
        plt.xlabel(axis_lbl)
        plt.ylabel("density")
        plt.title(f"Posterior-predictive check: {label}")
        plt.legend(fontsize=8)
        data = pd.DataFrame(
            {
                "score": centers,
                "observed_density": obs_dens,
                "pp_density_median": med_band,
                "pp_density_lo": lo_band,
                "pp_density_hi": hi_band,
            }
        )
        save_styled_figure(context.output_dir, filename_stem, data=data)


def _ppc_calibration_figure(
    context: StatisticalFitContext, symbol: str | None, cal: pd.DataFrame
) -> None:
    """Per-observation calibration panel: observed vs posterior-predictive median.

    Observed score (x) against the predictive median with a 90% interval (y) and a
    ``y = x`` diagonal; points off the diagonal are directly-readable mis-fits, and
    observations whose observed score falls outside the 90% range are flagged.
    Writes ``ppc_calibration.png`` (+ the per-observation data CSV).
    """
    with guard_optional(
        context, "PPC calibration figure",
        filename="ppc_calibration.png", kind="figure", verb="failed",
    ):
        label, n_trials = _ppc_measure_label(symbol)
        obs = cal["observed"].to_numpy(float)
        med = cal["pp_median"].to_numpy(float)
        lo = cal["pp_lo"].to_numpy(float)
        hi = cal["pp_hi"].to_numpy(float)
        inside = cal["inside"].to_numpy(bool)
        lim_hi = float(n_trials) if n_trials else float(max(obs.max(), hi.max()))
        plt.figure(figsize=(5.5, 5.5))
        plt.plot([0, lim_hi], [0, lim_hi], color="#888", ls="--", lw=1,
                 label="perfect calibration (y = x)")
        plt.errorbar(
            obs, med, yerr=np.vstack((med - lo, hi - med)), fmt="none",
            ecolor=COLOUR_BLUE, alpha=0.35, capsize=0, zorder=1,
        )
        plt.scatter(obs[inside], med[inside], s=18, color=COLOUR_BLUE,
                    label="observed inside 90% range", zorder=2)
        plt.scatter(obs[~inside], med[~inside], s=26, color=COLOUR_RED, marker="x",
                    lw=1.6, label="observed outside 90% range", zorder=3)
        plt.xlabel(f"observed {label} score")
        plt.ylabel("posterior-predictive median (90% range)")
        plt.title(f"Per-observation calibration: {label}")
        plt.legend(fontsize=8)
        save_styled_figure(context.output_dir, "ppc_calibration", data=cal)


def _ppc_offfloor_figure(
    context: StatisticalFitContext, symbol: str | None, cells: pd.DataFrame
) -> None:
    """Floor-rule PPC figure: observed off-floor rate vs its predictive rate by cell.

    Writes ``posterior_predictive_check.png`` (the floor-rule analogue of the count
    overlay: the observed rate should sit inside the model's predictive range for
    each cell) plus the per-cell data CSV.
    """
    with guard_optional(
        context, "PPC off-floor figure",
        filename="posterior_predictive_check.png", kind="figure", verb="failed",
    ):
        label, _ = _ppc_measure_label(symbol)
        x = np.arange(len(cells))
        med = cells["pp_rate_median"].to_numpy(float)
        lo = cells["pp_rate_lo"].to_numpy(float)
        hi = cells["pp_rate_hi"].to_numpy(float)
        obs = cells["observed_rate"].to_numpy(float)
        plt.figure(figsize=(max(5.0, 1.6 * len(cells) + 2.0), 4))
        plt.errorbar(
            x, med, yerr=np.vstack((med - lo, hi - med)), fmt="o", color=COLOUR_BLUE,
            capsize=4, label="posterior-predictive median and 90% range",
        )
        plt.scatter(x, obs, marker="x", s=60, lw=2, color=COLOUR_RED,
                    label="observed", zorder=3)
        plt.xticks(x, cells["cell"].tolist())
        plt.ylabel("off-floor rate")
        plt.ylim(-0.02, 1.02)
        plt.title(f"Off-floor rate posterior-predictive check: {label}")
        plt.legend(fontsize=8)
        save_styled_figure(context.output_dir, "posterior_predictive_check", data=cells)


def _save_legacy_ppc_overlay(context: StatisticalFitContext) -> None:
    # arviz 1.x removed az.plot_ppc; the equivalent is arviz_plots.plot_ppc_dist
    # (returns a PlotCollection with .savefig). Used for measurement / latent nodes
    # that have no single count outcome. Guarded — a PPC plot failure must not abort.
    with guard_optional(
        context, "PPC plot",
        filename="posterior_predictive_check.png", kind="figure", verb="failed",
    ):
        import arviz_plots as azp

        pc = azp.plot_ppc_dist(context.trace)
        save_plotcollection(
            pc,
            context.output_dir,
            "posterior_predictive_check.png",
            suptitle="Posterior-predictive vs observed",
        )
