# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Intervention vs no-intervention posterior-overlap figures for ITT models.

The predicted-scores panel (:mod:`predicted_scores`) answers "what does the
model say about actual test scores"; this module adds the complementary
*overlap* view a reader asked for: two smoothed density curves — intervention
and no-intervention — drawn over one another so the eye can judge how far apart
the arms are. It emits **two independent, single-axis figures** (never a shared
two-panel figure) so each can be reused on its own:

1. ``arm_overlap_mean`` — the posterior of each arm's *population-average*
   expected outcome level (proportion correct for graded outcomes; off-floor
   probability for the floor rule). Narrow curves; their separation is the
   average treatment effect and their overlap is posterior uncertainty about
   *which arm is higher*. This is the "how distinguishable are the arms" read.
2. ``arm_overlap_predictive`` — the posterior-predictive outcome *for a new
   child* under each arm (parameter uncertainty **plus** child-to-child and
   sampling spread). Much wider curves; the overlap is the individual-level
   overlap, and heavy overlap is expected even when the average effect is well
   supported. Graded outcomes only — a single binary off-floor outcome has no
   smooth predictive density, so the floor rule emits only figure 1.

Both reuse :func:`predicted_scores.counterfactual_predictive_contrast`, so the
annotated average marginal effect is the same guard-tested quantity that drives
``rope_summary.csv`` / ``predicted_scores.csv``, and the predictive curves are
drawn from the identical simulated scores when the same random seed is used.

Caveat worth stating in captions: the two arm means share the untreated linear
predictor per posterior draw, so they are positively correlated. Overlapping
*marginal* curves can therefore overstate uncertainty about the *difference* —
the effect/ROPE density remains the authoritative read; these figures are the
intuitive companion.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import gaussian_kde

from dse_research_utils.plot.styles import COLOUR_BLUE, COLOUR_ORANGE, FIGSIZE_LG

from language_reading_predictors.figure_io import save_styled_figure
from language_reading_predictors.statistical_models.predicted_scores import (
    PredictiveContrast,
    counterfactual_predictive_contrast,
)

__all__ = [
    "OverlapCurves",
    "arm_overlap_summary",
    "overlap_curves",
    "save_arm_overlap_mean",
    "save_arm_overlap_predictive",
    "write_arm_overlap_artifacts",
]

#: Arm colours from the shared project palette (``dse_research_utils.plot.styles``):
#: wait-list control orange, immediate intervention blue. The overlap region is
#: not given its own hue — it reads as the natural blend where the two
#: translucent fills cross.
_CONTROL_COLOR = COLOUR_ORANGE
_INTERVENTION_COLOR = COLOUR_BLUE


@dataclass(frozen=True, slots=True)
class OverlapCurves:
    """Two smoothed densities on a shared grid plus their overlapping area."""

    grid: np.ndarray
    density_control: np.ndarray
    density_intervention: np.ndarray
    overlap_coefficient: float

    @property
    def density_overlap(self) -> np.ndarray:
        """Pointwise minimum of the two densities (the shaded overlap band)."""
        return np.minimum(self.density_control, self.density_intervention)


def overlap_curves(
    control: np.ndarray,
    intervention: np.ndarray,
    *,
    clip: tuple[float, float] | None = (0.0, 100.0),
    n_grid: int = 512,
    pad_frac: float = 0.08,
) -> OverlapCurves:
    """KDE both arms on one grid; return curves and the overlapping coefficient.

    The overlapping coefficient is the area under the pointwise minimum of the
    two densities — 1 when the arms are indistinguishable, 0 when disjoint. A
    degenerate (zero-variance) arm falls back to a narrow spike so the figure
    still renders.
    """
    a = np.asarray(control, dtype=float)
    b = np.asarray(intervention, dtype=float)
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    pad = pad_frac * (hi - lo) if hi > lo else max(abs(hi), 1.0) * pad_frac
    grid_lo, grid_hi = lo - pad, hi + pad
    if clip is not None:
        grid_lo = max(clip[0], grid_lo)
        grid_hi = min(clip[1], grid_hi)
    grid = np.linspace(grid_lo, grid_hi, n_grid)

    def _density(x: np.ndarray) -> np.ndarray:
        try:
            return gaussian_kde(x)(grid)
        except (np.linalg.LinAlgError, ValueError):  # pragma: no cover - degenerate
            hist, edges = np.histogram(x, bins=min(60, grid.size), density=True)
            centres = 0.5 * (edges[:-1] + edges[1:])
            return np.interp(grid, centres, hist, left=0.0, right=0.0)

    dc = _density(a)
    di = _density(b)
    overlap = float(np.trapezoid(np.minimum(dc, di), grid))
    return OverlapCurves(grid=grid, density_control=dc, density_intervention=di,
                         overlap_coefficient=overlap)


def _band(draws: np.ndarray, ci_prob: float) -> tuple[float, float, float, float, float]:
    """(median, lo, hi, lo50, hi50) for the given equal-tailed coverage."""
    d = np.asarray(draws, dtype=float)
    lo_q = (1.0 - ci_prob) / 2.0
    return (
        float(np.median(d)),
        float(np.quantile(d, lo_q)),
        float(np.quantile(d, 1.0 - lo_q)),
        float(np.quantile(d, 0.25)),
        float(np.quantile(d, 0.75)),
    )


def arm_overlap_summary(
    *,
    outcome_symbol: str,
    control: np.ndarray,
    intervention: np.ndarray,
    effect: np.ndarray,
    effect_scale: str,
    level_scale: str,
    overlap_coefficient: float,
    ci_prob: float,
    population: str,
    contrast_status: str,
) -> pd.DataFrame:
    """Tabulate the citable overlap quantities as ``<figure>.csv`` (issue #208)."""

    def _row(quantity: str, draws: np.ndarray, scale: str) -> dict:
        median, lo, hi, lo50, hi50 = _band(draws, ci_prob)
        return {
            "outcome": outcome_symbol,
            "quantity": quantity,
            "scale": scale,
            "median": median,
            "lo": lo,
            "hi": hi,
            "lo50": lo50,
            "hi50": hi50,
            "population": population,
            "contrast_status": contrast_status,
        }

    def _scalar(quantity: str, value: float, scale: str) -> dict:
        return {
            "outcome": outcome_symbol,
            "quantity": quantity,
            "scale": scale,
            "median": float(value),
            "lo": float("nan"),
            "hi": float("nan"),
            "lo50": float("nan"),
            "hi50": float("nan"),
            "population": population,
            "contrast_status": contrast_status,
        }

    rows = [
        _row("no_intervention_level", control, level_scale),
        _row("intervention_level", intervention, level_scale),
        _row("average_marginal_effect", effect, effect_scale),
        _scalar("overlap_coefficient", overlap_coefficient, "fraction"),
        _scalar("p_intervention_higher", float(np.mean(np.asarray(effect) > 0)),
                "probability"),
    ]
    return pd.DataFrame(rows)


def _draw_overlap(
    ax: plt.Axes,
    curves: OverlapCurves,
    control: np.ndarray,
    intervention: np.ndarray,
    *,
    control_label: str,
    intervention_label: str,
) -> None:
    """Shared drawing: filled KDE curves and arm-median rules.

    The two arms are drawn as translucent fills; where they cross, the natural
    blend of the two reads as the overlap — no separate overlap hue is added
    (its area is reported numerically in the annotation and the sidecar CSV).
    """
    ax.fill_between(curves.grid, curves.density_control, color=_CONTROL_COLOR,
                    alpha=0.35, label=control_label)
    ax.fill_between(curves.grid, curves.density_intervention,
                    color=_INTERVENTION_COLOR, alpha=0.35, label=intervention_label)
    ax.plot(curves.grid, curves.density_control, color=_CONTROL_COLOR, lw=1.3)
    ax.plot(curves.grid, curves.density_intervention, color=_INTERVENTION_COLOR, lw=1.3)
    ax.axvline(float(np.median(control)), color=_CONTROL_COLOR, lw=1.4, ls="--")
    ax.axvline(float(np.median(intervention)), color=_INTERVENTION_COLOR, lw=1.4, ls="--")
    ax.set_ylabel("posterior density")
    ax.legend(loc="upper right", fontsize=8, frameon=False)


def save_arm_overlap_mean(
    output_dir: str,
    contrast: PredictiveContrast,
    *,
    outcome_symbol: str,
    item_label: str,
    likelihood: str,
    ci_prob: float,
    population: str,
    contrast_status: str,
    event_label: str = "off the floor at follow-up",
    name: str = "arm_overlap_mean",
) -> pd.DataFrame:
    """Figure 1: posterior of each arm's population-average expected outcome.

    Values are shown as percentages: expected proportion correct for graded
    outcomes, or P(event) for the floor rule. Returns the summary table it
    saves as the ``<name>.csv`` sidecar.
    """
    pc = np.asarray(contrast.prob_control, dtype=float) * 100.0
    pt = np.asarray(contrast.prob_intervention, dtype=float) * 100.0
    ame_pp = np.asarray(contrast.ame_prob, dtype=float) * 100.0  # percentage points

    curves = overlap_curves(pc, pt)
    summary = arm_overlap_summary(
        outcome_symbol=outcome_symbol,
        control=pc,
        intervention=pt,
        effect=ame_pp,
        effect_scale="percentage_points",
        level_scale="percent",
        overlap_coefficient=curves.overlap_coefficient,
        ci_prob=ci_prob,
        population=population,
        contrast_status=contrast_status,
    )

    fig, ax = plt.subplots(figsize=FIGSIZE_LG)
    _draw_overlap(ax, curves, pc, pt,
                  control_label="no intervention (wait-list)",
                  intervention_label="intervention (immediate)")
    if likelihood == "bernoulli":
        ax.set_xlabel(f"P({event_label}) — {item_label} (%)")
        ax.set_title(f"Posterior probability {event_label} by arm ({outcome_symbol})",
                     fontsize=10)
    else:
        ax.set_xlabel(f"expected {item_label} score (% correct)")
        ax.set_title(
            f"Posterior of expected outcome by arm ({outcome_symbol}) — "
            "population average",
            fontsize=10,
        )
    lo_e, hi_e = np.quantile(ame_pp, [(1 - ci_prob) / 2, 1 - (1 - ci_prob) / 2])
    ax.text(
        0.02, 0.98,
        (
            f"median: {np.median(pt):.1f}% vs {np.median(pc):.1f}%\n"
            f"average effect: {np.median(ame_pp):+.1f} pp "
            f"[{lo_e:+.1f}, {hi_e:+.1f}]\n"
            f"P(intervention higher) = {np.mean(ame_pp > 0):.2f}\n"
            f"overlap = {curves.overlap_coefficient:.0%}"
        ),
        transform=ax.transAxes, va="top", ha="left", fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    fig.tight_layout()
    save_styled_figure(output_dir, name, fig=fig, data=summary)
    return summary


def save_arm_overlap_predictive(
    output_dir: str,
    contrast: PredictiveContrast,
    *,
    outcome_symbol: str,
    item_label: str,
    ci_prob: float,
    population: str,
    contrast_status: str,
    name: str = "arm_overlap_predictive",
) -> pd.DataFrame | None:
    """Figure 2: predicted outcome for a *new child* under each arm.

    Uses the simulated new-child scores (graded outcomes only), expressed as a
    percentage of the test ceiling so the axis matches figure 1. Returns the
    summary table, or ``None`` when no score simulation is available (the floor
    rule), in which case no figure is written.
    """
    if not contrast.score_control.size:
        return None
    n = float(contrast.n_trials)
    pc = np.asarray(contrast.score_control, dtype=float) / n * 100.0
    pt = np.asarray(contrast.score_intervention, dtype=float) / n * 100.0
    diff_items = np.asarray(contrast.score_difference, dtype=float)  # paired, items

    curves = overlap_curves(pc, pt)
    summary = arm_overlap_summary(
        outcome_symbol=outcome_symbol,
        control=pc,
        intervention=pt,
        effect=diff_items,
        effect_scale="items",
        level_scale="percent",
        overlap_coefficient=curves.overlap_coefficient,
        ci_prob=ci_prob,
        population=population,
        contrast_status=contrast_status,
    )

    fig, ax = plt.subplots(figsize=FIGSIZE_LG)
    _draw_overlap(ax, curves, pc, pt,
                  control_label="no intervention (wait-list)",
                  intervention_label="intervention (immediate)")
    ax.set_xlabel(f"predicted {item_label} score for a new child (% correct)")
    ax.set_title(
        f"Predicted-outcome overlap, new child ({outcome_symbol})", fontsize=10
    )
    ax.text(
        0.02, 0.98,
        (
            f"median: {np.median(pt):.0f}% vs {np.median(pc):.0f}%\n"
            f"paired effect: {np.median(diff_items):+.1f} items\n"
            f"overlap = {curves.overlap_coefficient:.0%}"
        ),
        transform=ax.transAxes, va="top", ha="left", fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    fig.tight_layout()
    save_styled_figure(output_dir, name, fig=fig, data=summary)
    return summary


def write_arm_overlap_artifacts(
    output_dir: str,
    trace: xr.DataTree,
    *,
    outcome_symbol: str,
    item_label: str,
    G: np.ndarray,
    n_trials: int,
    term: str,
    varying_term: str = "tau_i",
    eta_name: str = "eta",
    moderators=None,
    row_mask=None,
    likelihood: str = "beta_binomial",
    kappa_name: str = "kappa",
    child_effect_name: str | None = None,
    child_sd_name: str | None = None,
    child_idx=None,
    ci_prob: float = 0.95,
    population: str = "new typical child; covariates from the fitted reference rows",
    contrast_status: str = "randomised contrast",
    event_label: str = "off the floor at follow-up",
    random_seed: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute the contrast once and write both individual overlap figures.

    Mirrors :func:`predicted_scores.write_predicted_scores_artifacts`'s argument
    surface so the pipeline can wire the two the same way. Returns the written
    summary tables keyed by figure stem (the predictive entry is absent for the
    floor rule).
    """
    rng = np.random.default_rng(random_seed)
    contrast = counterfactual_predictive_contrast(
        trace,
        G=np.asarray(G, dtype=float),
        n_trials=int(n_trials),
        term=term,
        varying_term=varying_term,
        eta_name=eta_name,
        moderators=moderators,
        row_mask=row_mask,
        likelihood=likelihood,
        kappa_name=kappa_name,
        child_effect_name=child_effect_name,
        child_sd_name=child_sd_name,
        child_idx=child_idx,
        rng=rng,
    )

    tables: dict[str, pd.DataFrame] = {}
    tables["arm_overlap_mean"] = save_arm_overlap_mean(
        output_dir,
        contrast,
        outcome_symbol=outcome_symbol,
        item_label=item_label,
        likelihood=likelihood,
        ci_prob=ci_prob,
        population=population,
        contrast_status=contrast_status,
        event_label=event_label,
    )
    predictive = save_arm_overlap_predictive(
        output_dir,
        contrast,
        outcome_symbol=outcome_symbol,
        item_label=item_label,
        ci_prob=ci_prob,
        population=population,
        contrast_status=contrast_status,
    )
    if predictive is not None:
        tables["arm_overlap_predictive"] = predictive
    return tables
