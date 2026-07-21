# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ROPE-anchored effect figures for a randomised treatment effect.

Draws the two views behind the ROPE summary from a vector of items-scale (or
risk-difference-scale) effect draws:

* the posterior effect density with the region of practical equivalence (ROPE)
  shaded; and
* ``P(effect > delta)`` as the minimally-important difference delta rises.

``write_rope_figures`` can lay these out as one combined two-panel figure (the
default, kept for the families that present them together) or, with
``split=True``, as two individual files (``rope_summary`` and
``rope_benefit_curve``) so each is reusable on its own. The drawing is shared, so
the two layouts stay identical bar the arrangement and figure size. The core is
kept free of any fit context so both the fit pipeline and the standalone
regeneration script can call it.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dse_research_utils.plot.styles import FIGSIZE_LG
from scipy.stats import gaussian_kde

from language_reading_predictors.figure_io import save_styled_figure

__all__ = ["benefit_curve_table", "write_rope_figures"]

#: Effect-density and benefit-curve accents (unchanged from the original panel).
_EFFECT_COLOR = "#1b7837"
_BENEFIT_COLOR = "#2166ac"
_ROPE_COLOR = "#bdbdbd"


def _labels(risk_difference: bool) -> tuple[str, str, str]:
    if risk_difference:
        return (
            "treatment effect (risk difference)",
            "minimally-important difference, delta (probability)",
            "risk-difference scale",
        )
    return (
        "treatment effect (extra items correct)",
        "minimally-important difference, delta (items)",
        "items scale",
    )


def benefit_curve_table(items: np.ndarray, delta: float, *, xmax: float) -> pd.DataFrame:
    """The ``P(effect > delta)`` sweep plotted in the benefit-curve panel."""
    items = np.asarray(items, dtype=float)
    dgrid = np.linspace(0.0, max(xmax, delta + 0.5), 200)
    pex = np.array([float((items > d).mean()) for d in dgrid])
    return pd.DataFrame({"delta": dgrid, "prob_effect_gt_delta": pex})


def _draw_effect(ax: plt.Axes, items: np.ndarray, *, delta: float, symbol: str,
                 risk_difference: bool, xmin: float, xmax: float) -> None:
    effect_label, _, scale_title = _labels(risk_difference)
    xs = np.linspace(xmin, xmax, 300)
    kde = gaussian_kde(items)
    med = float(np.median(items))
    ax.axvspan(-delta, delta, color=_ROPE_COLOR, alpha=0.30,
               label=f"ROPE (within ±{delta:g})")
    ax.axvline(0, color="#444444", lw=1.0, ls=":")
    ax.plot(xs, kde(xs), color=_EFFECT_COLOR, lw=2.2)
    ax.fill_between(xs, kde(xs), color=_EFFECT_COLOR, alpha=0.12)
    ax.axvline(med, color=_EFFECT_COLOR, lw=1.2, label=f"median {med:+.1f}")
    ax.set_xlabel(effect_label)
    ax.set_ylabel("posterior density")
    ax.set_title(f"{symbol}: effect on the {scale_title}, with ROPE")
    ax.legend(fontsize=8, frameon=False)


def _draw_benefit(ax: plt.Axes, sweep: pd.DataFrame, *, delta: float,
                  risk_difference: bool) -> None:
    _, delta_label, _ = _labels(risk_difference)
    ax.plot(sweep["delta"], sweep["prob_effect_gt_delta"], color=_BENEFIT_COLOR, lw=2.2)
    ax.axvline(delta, color="#888888", lw=1.0, ls="--", label=f"delta = {delta:g}")
    ax.axhline(0.975, color="#cccccc", lw=1.0, ls=":")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(delta_label)
    ax.set_ylabel("P(effect > delta)")
    ax.set_title("Probability of a meaningful benefit")
    ax.legend(fontsize=8, frameon=False)


def _despine(*axes: plt.Axes) -> None:
    for ax in axes:
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)


def write_rope_figures(
    output_dir: str,
    items: np.ndarray,
    *,
    symbol: str,
    delta: float,
    n_trials: int,
    split: bool = False,
) -> None:
    """Write the ROPE effect figure(s) from a vector of effect draws.

    ``split=False`` writes the combined two-panel ``rope_summary``; ``split=True``
    writes ``rope_summary`` (effect + ROPE) and ``rope_benefit_curve``
    (``P(effect > delta)`` sweep) as individual FIGSIZE_LG files, the second with
    its sweep data as the sidecar CSV.
    """
    items = np.asarray(items, dtype=float)
    risk_difference = n_trials == 1 and delta <= 1
    xmax = float(np.quantile(items, 0.995)) + 0.5
    xmin = min(-delta - 0.5, float(np.quantile(items, 0.005)))
    sweep = benefit_curve_table(items, delta, xmax=xmax)

    if not split:
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.2))
        _draw_effect(ax_l, items, delta=delta, symbol=symbol,
                     risk_difference=risk_difference, xmin=xmin, xmax=xmax)
        _draw_benefit(ax_r, sweep, delta=delta, risk_difference=risk_difference)
        _despine(ax_l, ax_r)
        fig.tight_layout()
        save_styled_figure(output_dir, "rope_summary", fig=fig)
        return

    fig_e, ax_e = plt.subplots(figsize=FIGSIZE_LG)
    _draw_effect(ax_e, items, delta=delta, symbol=symbol,
                 risk_difference=risk_difference, xmin=xmin, xmax=xmax)
    _despine(ax_e)
    fig_e.tight_layout()
    save_styled_figure(output_dir, "rope_summary", fig=fig_e)

    fig_b, ax_b = plt.subplots(figsize=FIGSIZE_LG)
    _draw_benefit(ax_b, sweep, delta=delta, risk_difference=risk_difference)
    _despine(ax_b)
    fig_b.tight_layout()
    save_styled_figure(output_dir, "rope_benefit_curve", fig=fig_b, data=sweep)
