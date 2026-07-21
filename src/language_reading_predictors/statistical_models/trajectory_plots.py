# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Data-space report figures for the longitudinal families (#317).

The repeated-measures families previously rendered no data-space figure at all —
a level-factors report showed coefficient tables but never a score trajectory.
This module adds the two undergraduate-readability figures issue #317 specifies:

1. a **group trajectory** — the model's posterior mean score across waves (items
   scale) with credible ribbons, observed means overlaid, and a crossover marker at
   t2. Per arm for the waitlist-crossover families (``level_factors`` / ``did``); per
   measure for ``growth`` (whose "arm" is a latent tempo factor, not an observed
   randomised arm). The ribbon is **population-level / marginal over children**:
   per posterior draw the fitted child intercept is removed and the child random
   effect is integrated over its ``Normal(0, sigma_child)`` distribution
   (Gauss--Hermite), then the per-row marginal probability is averaged over the
   children observed in each wave x arm cell (g-computation over the cell's own
   children). This is **not** the same-children conditional display of figure 2, and
   captions must say so.

2. **per-child fitted-vs-observed small multiples** — a grid of ~12 children (a
   seeded random draw plus the 2--3 worst-fitting by Pareto-k, flagged), observed
   scores as dots over waves with the posterior-predictive ribbon behind. These are
   **same-children** predictions (the posterior-predictive rows are conditional on
   each child's own fitted intercept), which makes the child random intercept
   concrete. Panels are indexed, never labelled with subject ids.

Floored outcomes (P, N; ``bernoulli_offfloor``) replace the item-score axis with an
off-floor probability axis: ``expit(eta)`` is P(off-floor), the observed overlay is
the off-floor rate / the 0/1 indicator.

The obs_id families (``level_factors``, ``did``, ``gain_factors``, ``mechanism``)
carry everything the figures need in the trace (``posterior/eta``, ``u_child`` /
``sigma_child``, ``posterior_predictive`` + ``observed_data``); the multivariate
masked families (``growth``, ``lcsm``) store a flat masked ``y_obs`` with no
cell->(child, wave, outcome) mapping, so their figures are built at fit time from the
``WavePanel`` and the dense latent grid (``theta`` / ``x_latent``).

Style and guarding mirror ``predicted_scores.py``; ``figure_io.save_styled_figure``
emits the PNG + SVG sibling + data CSV.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.special import expit

from dse_research_utils.plot.styles import (
    COLOUR_BLUE,
    COLOUR_DARK_GREEN,
    COLOUR_ORANGE,
    COLOUR_RED,
)

from language_reading_predictors.figure_io import save_styled_figure

__all__ = [
    "ARM_LABELS",
    "child_predictive_bands",
    "marginal_cell_probabilities",
    "observed_cell_means",
    "select_children",
    "write_child_fit_obsid",
    "write_child_fit_panel",
    "write_group_arm_trajectory",
    "write_outcome_trajectory",
]

#: Arm colours from the shared project palette (``dse_research_utils.plot.styles``),
#: matching ``predicted_scores.py``: wait-list control orange, immediate
#: intervention blue.
_CONTROL_COLOR = COLOUR_ORANGE
_INTERVENTION_COLOR = COLOUR_BLUE
#: Observed-overlay (red) and fit-ribbon (dark green) accents from the same palette.
_OBSERVED_COLOR = COLOUR_RED
_FIT_COLOR = COLOUR_DARK_GREEN
#: Per-arm colour lookup and human labels (dataset arm coding: 1 = immediate, 0 = waitlist).
ARM_COLORS: dict[int, str] = {0: _CONTROL_COLOR, 1: _INTERVENTION_COLOR}
ARM_LABELS: dict[int, str] = {0: "wait-list control", 1: "immediate intervention"}


def _wave_label(index: int) -> str:
    """Timepoint label from a 0-based wave/phase index (``0 -> "t1"``)."""
    return f"t{int(index) + 1}"


# ---------------------------------------------------------------------------
# Child selection for the small multiples
# ---------------------------------------------------------------------------


def select_children(
    pareto_k_by_child: Mapping[int, float] | np.ndarray | None,
    n_children: int,
    *,
    n_total: int = 12,
    n_worst: int = 3,
    seed: int = 47,
) -> tuple[list[int], set[int]]:
    """Choose the children shown in the small multiples, reproducibly.

    Returns ``(ordered_child_ids, worst_set)``. The panel is the union of the
    ``n_worst`` children with the highest per-child Pareto-k (the worst-fitting,
    flagged in ``worst_set``) and a **seeded** random draw filling the grid up to
    ``n_total``; seeding from the run config makes refits comparable (#317). With no
    Pareto-k available the panel is a pure seeded draw. The returned ids are ordered
    worst-first so the flagged panels lead the grid.
    """
    rng = np.random.default_rng(seed)
    n_total = min(int(n_total), int(n_children))

    worst: list[int] = []
    if pareto_k_by_child is not None and n_worst > 0:
        if isinstance(pareto_k_by_child, Mapping):
            items = list(pareto_k_by_child.items())
        else:
            k = np.asarray(pareto_k_by_child, dtype=float)
            items = list(enumerate(k.tolist()))
        # Highest k first; NaNs (undefined k) sort last.
        items = [(c, v) for c, v in items if np.isfinite(v)]
        items.sort(key=lambda cv: cv[1], reverse=True)
        worst = [int(c) for c, _ in items[: min(n_worst, n_total)]]

    worst_set = set(worst)
    remaining = [c for c in range(int(n_children)) if c not in worst_set]
    rng.shuffle(remaining)
    fill = remaining[: max(0, n_total - len(worst))]
    ordered = worst + sorted(fill)
    return ordered, worst_set


# ---------------------------------------------------------------------------
# obs_id families: marginal (population) cell probabilities and observed means
# ---------------------------------------------------------------------------


def _gh_marginal_prob(
    eta0: np.ndarray, sigma: np.ndarray | None, *, n_gh: int
) -> np.ndarray:
    """E_u[expit(eta0 + u)] with ``u ~ Normal(0, sigma)`` by Gauss--Hermite quadrature.

    ``eta0`` is ``(n_rows, S)`` (the linear predictor with the fitted child intercept
    already removed); ``sigma`` is ``(S,)`` the per-draw random-effect SD, or ``None``
    for a model without a child random effect (then ``expit(eta0)`` is returned). The
    node loop keeps the peak footprint at one ``(n_rows, S)`` array.
    """
    if sigma is None:
        return expit(eta0)
    nodes, weights = hermgauss(int(n_gh))
    scale = np.sqrt(2.0) * np.asarray(sigma, dtype=float)[None, :]  # (1, S)
    acc = np.zeros_like(eta0)
    for x_k, w_k in zip(nodes, weights, strict=True):
        acc += w_k * expit(eta0 + scale * x_k)
    return acc / np.sqrt(np.pi)


def marginal_cell_probabilities(
    eta: np.ndarray,
    *,
    arm: np.ndarray,
    wave: np.ndarray,
    u_child_rows: np.ndarray | None = None,
    sigma_child: np.ndarray | None = None,
    n_gh: int = 16,
) -> dict[tuple[int, int], np.ndarray]:
    """Per-draw population mean probability for every (arm, wave) cell.

    ``eta`` is ``(n_obs, S)``; ``arm`` / ``wave`` are ``(n_obs,)`` integer labels;
    ``u_child_rows`` is the fitted child intercept already gathered to each row
    (``(n_obs, S)``) and ``sigma_child`` the per-draw intercept SD ``(S,)``. Removing
    ``u_child_rows`` and integrating over ``Normal(0, sigma_child)`` yields the
    **marginal** (new-child) per-row probability; averaging over the rows in each
    (arm, wave) cell standardises over the children actually observed in that cell
    (g-computation). Returns ``{(arm, wave): (S,)}``.
    """
    eta = np.asarray(eta, dtype=float)
    eta0 = eta if u_child_rows is None else eta - np.asarray(u_child_rows, dtype=float)
    p_row = _gh_marginal_prob(eta0, sigma_child, n_gh=n_gh)  # (n_obs, S)
    arm = np.asarray(arm)
    wave = np.asarray(wave)
    out: dict[tuple[int, int], np.ndarray] = {}
    for g in np.unique(arm):
        for w in np.unique(wave):
            mask = (arm == g) & (wave == w)
            if mask.any():
                out[(int(g), int(w))] = p_row[mask].mean(axis=0)
    return out


def observed_cell_means(
    observed: np.ndarray, *, arm: np.ndarray, wave: np.ndarray
) -> dict[tuple[int, int], float]:
    """Observed mean of ``observed`` per (arm, wave) cell (counts, or 0/1 off-floor)."""
    observed = np.asarray(observed, dtype=float)
    arm = np.asarray(arm)
    wave = np.asarray(wave)
    out: dict[tuple[int, int], float] = {}
    for g in np.unique(arm):
        for w in np.unique(wave):
            mask = (arm == g) & (wave == w)
            if mask.any():
                out[(int(g), int(w))] = float(observed[mask].mean())
    return out


# ---------------------------------------------------------------------------
# Small multiples: per-child posterior-predictive bands
# ---------------------------------------------------------------------------


def child_predictive_bands(
    ppc_rows: np.ndarray, *, ci_prob: float
) -> dict[str, np.ndarray]:
    """Median and inner-50 / ``ci_prob`` bands of a child's predictive draws by wave.

    ``ppc_rows`` is ``(n_waves, S)`` posterior-predictive draws for one child's rows
    in wave order. Returns median / lo / hi / lo50 / hi50 arrays of length ``n_waves``.
    """
    ppc_rows = np.asarray(ppc_rows, dtype=float)
    lo_q = (1 - ci_prob) / 2
    return {
        "median": np.median(ppc_rows, axis=1),
        "lo": np.quantile(ppc_rows, lo_q, axis=1),
        "hi": np.quantile(ppc_rows, 1 - lo_q, axis=1),
        "lo50": np.quantile(ppc_rows, 0.25, axis=1),
        "hi50": np.quantile(ppc_rows, 0.75, axis=1),
    }


# ---------------------------------------------------------------------------
# Trace helpers
# ---------------------------------------------------------------------------


def _stack_last(da) -> np.ndarray:
    """Stack ``(chain, draw)`` into a trailing ``sample`` axis and return an array.

    A scalar parameter becomes ``(S,)``; ``(…, dim)`` becomes ``(dim, S)``.
    """
    stacked = da.stack(sample=("chain", "draw"))
    other = [d for d in stacked.dims if d != "sample"]
    return stacked.transpose(*other, "sample").values


def _thin_columns(n_samples: int, max_draws: int) -> np.ndarray:
    """Even stride over ``n_samples`` posterior samples, capped at ``max_draws``."""
    if max_draws is None or n_samples <= max_draws:
        return np.arange(n_samples)
    step = int(np.ceil(n_samples / max_draws))
    return np.arange(0, n_samples, step)


def _quantile_band(draws: np.ndarray, ci_prob: float) -> dict[str, float]:
    """Median + inner-50 % + ``ci_prob`` (89 %) equal-tailed summary of a draw vector."""
    lo_q = (1 - ci_prob) / 2
    d = np.asarray(draws, dtype=float)
    return {
        "median": float(np.median(d)),
        "lo": float(np.quantile(d, lo_q)),
        "hi": float(np.quantile(d, 1 - lo_q)),
        "lo50": float(np.quantile(d, 0.25)),
        "hi50": float(np.quantile(d, 0.75)),
    }


# ---------------------------------------------------------------------------
# obs_id group trajectory (level_factors, did)
# ---------------------------------------------------------------------------


def write_group_arm_trajectory(
    output_dir: str,
    trace,
    *,
    arm: np.ndarray,
    wave: np.ndarray,
    child_idx: np.ndarray,
    n_trials: int,
    outcome_symbol: str,
    item_label: str,
    off_floor: bool = False,
    ci_prob: float = 0.95,
    crossover_wave: int = 1,
    eta_name: str = "eta",
    obs_node: str = "y_post",
    child_effect_name: str | None = "u_child",
    child_sd_name: str | None = "sigma_child",
    n_gh: int = 16,
    max_draws: int = 4000,
    name: str = "group_trajectory",
) -> pd.DataFrame:
    """Population per-arm score trajectory with ribbons + observed means (#317 fig 1).

    Marginal over children (the ribbon integrates the child random effect and
    averages over each cell's observed children); the observed dots are the raw
    per-arm cell means. Writes ``{name}.png`` / ``.svg`` and ``{name}.csv``; returns
    the summary table.
    """
    post = trace.posterior
    eta = _stack_last(post[eta_name])  # (n_obs, S)
    cols = _thin_columns(eta.shape[1], max_draws)
    eta = eta[:, cols]

    u_child_rows = None
    sigma = None
    if child_effect_name is not None and child_effect_name in post:
        u_child = _stack_last(post[child_effect_name])[:, cols]  # (n_children, S)
        u_child_rows = u_child[np.asarray(child_idx, dtype=int)]  # (n_obs, S)
        if child_sd_name is not None and child_sd_name in post:
            sigma = _stack_last(post[child_sd_name])[cols]  # (S,)

    cell_draws = marginal_cell_probabilities(
        eta, arm=arm, wave=wave, u_child_rows=u_child_rows, sigma_child=sigma, n_gh=n_gh
    )
    observed = np.asarray(trace.observed_data[obs_node].values, dtype=float)
    if off_floor:
        observed = (observed > 0).astype(float)
    obs_means = observed_cell_means(observed, arm=arm, wave=wave)
    arm_arr = np.asarray(arm)
    wave_arr = np.asarray(wave)
    cell_n = {
        key: int(np.sum((arm_arr == key[0]) & (wave_arr == key[1])))
        for key in cell_draws
    }

    scale_mult = 1.0 if off_floor else float(n_trials)
    waves = sorted({w for (_, w) in cell_draws})
    arms = sorted({g for (g, _) in cell_draws})

    rows: list[dict] = []
    for g in arms:
        for w in waves:
            key = (g, w)
            if key not in cell_draws:
                continue
            band = _quantile_band(cell_draws[key] * scale_mult, ci_prob)
            rows.append(
                {
                    "arm": g,
                    "arm_label": ARM_LABELS.get(g, str(g)),
                    "wave": w,
                    "timepoint": _wave_label(w),
                    "scale": "off_floor_probability" if off_floor else "items",
                    "observed_mean": obs_means.get(key, np.nan),
                    "predicted_median": band["median"],
                    "predicted_lo": band["lo"],
                    "predicted_hi": band["hi"],
                    "predicted_lo50": band["lo50"],
                    "predicted_hi50": band["hi50"],
                    "n_children_in_cell": cell_n.get(key, 0),
                    "n_trials": 1 if off_floor else int(n_trials),
                    "outcome": outcome_symbol,
                }
            )
    summary = pd.DataFrame(rows)

    _draw_group_trajectory(
        output_dir,
        summary,
        arms=arms,
        waves=waves,
        item_label=item_label,
        outcome_symbol=outcome_symbol,
        off_floor=off_floor,
        n_trials=n_trials,
        crossover_wave=crossover_wave,
        ci_prob=ci_prob,
        name=name,
    )
    return summary


def _draw_group_trajectory(
    output_dir: str,
    summary: pd.DataFrame,
    *,
    arms: Sequence[int],
    waves: Sequence[int],
    item_label: str,
    outcome_symbol: str,
    off_floor: bool,
    n_trials: int,
    crossover_wave: int | None,
    ci_prob: float,
    name: str,
) -> None:
    """Render the per-arm trajectory figure from the summary table."""
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x = np.asarray(waves, dtype=float)
    xticklabels = [_wave_label(w) for w in waves]
    pct = int(round(ci_prob * 100))
    for g in arms:
        sub = summary[summary["arm"] == g].sort_values("wave")
        color = ARM_COLORS.get(g, "#333333")
        ax.fill_between(
            sub["wave"], sub["predicted_lo"], sub["predicted_hi"],
            color=color, alpha=0.18, linewidth=0,
        )
        ax.plot(
            sub["wave"], sub["predicted_median"], color=color, lw=2.0,
            label=f"{ARM_LABELS.get(g, str(g))} — model mean ({pct}% CrI)",
        )
        ax.scatter(
            sub["wave"], sub["observed_mean"], color=color, edgecolor="white",
            s=55, zorder=5, marker="o",
        )
    if crossover_wave is not None and crossover_wave in list(waves):
        ax.axvline(
            crossover_wave, color="#555555", lw=1.0, ls="--",
            label=f"wait-list crossover ({_wave_label(crossover_wave)})",
        )
    ax.set_xticks(x, xticklabels)
    ax.set_xlabel("assessment wave")
    if off_floor:
        ax.set_ylabel(f"P(off the floor) — {item_label}")
        ax.set_ylim(0, 1)
    else:
        ax.set_ylabel(f"predicted score (items, out of {n_trials})")
    ax.set_title(
        f"Group score trajectory ({outcome_symbol}); filled dots = observed arm means"
    )
    ax.legend(fontsize=8, frameon=False, loc="best")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    save_styled_figure(output_dir, name, fig=fig, data=summary)


# ---------------------------------------------------------------------------
# obs_id per-child small multiples (level_factors, did, gain_factors, mechanism)
# ---------------------------------------------------------------------------


def _pareto_k_by_child(
    pareto_k: np.ndarray | None, child_idx: np.ndarray
) -> dict[int, float] | None:
    """Aggregate a per-observation Pareto-k vector to a per-child maximum."""
    if pareto_k is None:
        return None
    k = np.asarray(pareto_k, dtype=float)
    idx = np.asarray(child_idx, dtype=int)
    if k.shape[0] != idx.shape[0]:
        return None
    out: dict[int, float] = {}
    for c in np.unique(idx):
        vals = k[idx == c]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[int(c)] = float(vals.max())
    return out or None


def write_child_fit_obsid(
    output_dir: str,
    trace,
    *,
    wave: np.ndarray,
    child_idx: np.ndarray,
    n_trials: int,
    outcome_symbol: str,
    item_label: str,
    off_floor: bool = False,
    obs_node: str = "y_post",
    eta_name: str = "eta",
    pareto_k: np.ndarray | None = None,
    seed: int = 47,
    ci_prob: float = 0.95,
    x_label: str = "assessment wave",
    n_panels: int = 12,
    name: str = "child_fit_panels",
) -> pd.DataFrame:
    """Per-child fitted-vs-observed small multiples for an obs_id family (#317 fig 2).

    Same-children predictions: the posterior-predictive rows are conditional on each
    child's own fitted intercept. Selects a seeded random draw plus the worst-fitting
    children by Pareto-k (flagged). Writes ``{name}.png`` / ``.svg`` / ``.csv``.
    """
    child_idx = np.asarray(child_idx, dtype=int)
    wave = np.asarray(wave, dtype=int)
    n_children = int(child_idx.max()) + 1 if child_idx.size else 0
    k_by_child = _pareto_k_by_child(pareto_k, child_idx)
    chosen, worst_set = select_children(
        k_by_child, n_children, n_total=n_panels, seed=seed
    )

    observed = np.asarray(trace.observed_data[obs_node].values, dtype=float)
    if off_floor:
        observed = (observed > 0).astype(float)
        band_source = expit(_stack_last(trace.posterior[eta_name]))  # (n_obs, S) prob
    else:
        band_source = _stack_last(trace.posterior_predictive[obs_node])  # (n_obs, S)

    panels: list[dict] = []
    rows: list[dict] = []
    for pos, c in enumerate(chosen):
        sel = np.nonzero(child_idx == c)[0]
        order = np.argsort(wave[sel])
        sel = sel[order]
        waves_c = wave[sel]
        bands = child_predictive_bands(band_source[sel], ci_prob=ci_prob)
        obs_c = observed[sel]
        k_val = k_by_child.get(int(c)) if k_by_child else None
        panels.append(
            {
                "ordinal": pos + 1,
                "worst": c in worst_set,
                "pareto_k": k_val,
                "waves": waves_c,
                "observed": obs_c,
                "median": bands["median"],
                "lo": bands["lo"],
                "hi": bands["hi"],
            }
        )
        for j, w in enumerate(waves_c):
            rows.append(
                {
                    "panel": pos + 1,
                    "child_index": int(c),
                    "worst_fitting": bool(c in worst_set),
                    "pareto_k": k_val,
                    "wave": int(w),
                    "timepoint": _wave_label(w),
                    "observed": float(obs_c[j]),
                    "predicted_median": float(bands["median"][j]),
                    "predicted_lo": float(bands["lo"][j]),
                    "predicted_hi": float(bands["hi"][j]),
                    "scale": "off_floor_probability" if off_floor else "items",
                    "outcome": outcome_symbol,
                }
            )
    summary = pd.DataFrame(rows)
    _draw_small_multiples(
        output_dir,
        panels,
        summary=summary,
        item_label=item_label,
        outcome_symbol=outcome_symbol,
        off_floor=off_floor,
        n_trials=n_trials,
        x_label=x_label,
        ci_prob=ci_prob,
        name=name,
    )
    return summary


def _draw_small_multiples(
    output_dir: str,
    panels: Sequence[Mapping],
    *,
    summary: pd.DataFrame,
    item_label: str,
    outcome_symbol: str,
    off_floor: bool,
    n_trials: int,
    x_label: str,
    ci_prob: float,
    name: str,
    ncols: int = 4,
) -> None:
    """Render the ~12-panel fitted-vs-observed grid; worst-k panels are flagged."""
    n = len(panels)
    if n == 0:  # pragma: no cover - guarded upstream
        return
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    pct = int(round(ci_prob * 100))
    ymax = 1.0 if off_floor else float(n_trials)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.0 * ncols, 2.4 * nrows), sharex=True, sharey=True,
        squeeze=False,
    )
    flat = axes.ravel()
    for ax, p in zip(flat[:n], panels, strict=False):
        xs = np.asarray(p["waves"], dtype=float)
        ax.fill_between(xs, p["lo"], p["hi"], color=_FIT_COLOR, alpha=0.20, linewidth=0)
        ax.plot(xs, p["median"], color=_FIT_COLOR, lw=1.5)
        ax.scatter(
            xs, p["observed"], color=_OBSERVED_COLOR, s=30, zorder=5,
            edgecolor="white", linewidth=0.5,
        )
        ax.set_xticks(xs, [_wave_label(w) for w in p["waves"]], fontsize=7)
        ax.set_ylim(0, ymax)
        flag = f"\nhigh Pareto-k = {p['pareto_k']:.2f}" if p["worst"] and p["pareto_k"] is not None else ""
        ax.set_title(f"child #{p['ordinal']}{flag}", fontsize=8,
                     color=_OBSERVED_COLOR if p["worst"] else "black")
        ax.tick_params(labelsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in flat[n:]:
        ax.axis("off")
    ylabel = "P(off the floor)" if off_floor else f"score (out of {n_trials})"
    fig.supylabel(ylabel, fontsize=9)
    fig.supxlabel(x_label, fontsize=9)
    # The graded path plots posterior-predictive score draws (with observation
    # noise); the floored path plots the posterior credible interval for P(off the
    # floor) from expit(eta) — the per-row Bernoulli predictive is degenerate, so
    # the band label must not claim "posterior-predictive" there.
    band_desc = (
        f"posterior {pct}% band for P(off the floor)"
        if off_floor
        else f"posterior-predictive {pct}% band"
    )
    fig.suptitle(
        f"Per-child fit ({outcome_symbol}, {item_label}) — observed dots, "
        f"{band_desc}; same-children",
        fontsize=10,
    )
    fig.tight_layout()
    save_styled_figure(output_dir, name, fig=fig, data=summary)


# ---------------------------------------------------------------------------
# Multivariate masked families (growth, lcsm): latent-grid figures built from
# the WavePanel (the flat posterior-predictive vector carries no cell mapping).
# ---------------------------------------------------------------------------


def _panel_child_index(panel) -> np.ndarray:
    """Rebuild the flattened observed-cell -> child map (``np.nonzero(mask)`` order).

    Mirrors the growth / LCSM factories' likelihood construction
    (``mask = stack(obs_mask, axis=2); idx_i, _, _ = np.nonzero(mask)``) so a
    per-observation Pareto-k vector aligns to children the same way the fit stacked
    the observations.
    """
    mask = np.stack([panel.obs_mask[s] for s in panel.outcomes], axis=2)  # (N, T, K)
    idx_i, _, _ = np.nonzero(mask)
    return idx_i


def write_outcome_trajectory(
    output_dir: str,
    trace,
    panel,
    *,
    latent_name: str,
    ci_prob: float = 0.95,
    max_draws: int = 4000,
    name: str = "group_trajectory",
) -> pd.DataFrame:
    """Per-measure population growth trajectory (no arm) for the panel families (#317).

    For each outcome the ribbon is the posterior of the mean expected score across
    the cohort at each wave (``mean_over_children(sigmoid(latent)) * n_trials``);
    observed dots are the per-wave observed means. Writes ``{name}.png`` / ``.svg`` /
    ``.csv``.
    """
    from language_reading_predictors.statistical_models.measures import MEASURES

    latent = _stack_last(trace.posterior[latent_name])  # (child, wave, outcome, S)
    cols = _thin_columns(latent.shape[-1], max_draws)
    latent = latent[..., cols]
    waves = list(panel.waves)
    rows: list[dict] = []
    facet_data: dict[str, dict] = {}
    for k, sym in enumerate(panel.outcomes):
        n_tr = int(panel.n_trials[sym])
        counts = panel.counts[sym]  # (child, wave), NaN where missing
        prob = expit(latent[:, :, k, :])  # (child, wave, S)
        pop_mean = prob.mean(axis=0) * n_tr  # (wave, S)
        obs_mean = np.nanmean(counts, axis=0)  # (wave,)
        band = [_quantile_band(pop_mean[t], ci_prob) for t in range(pop_mean.shape[0])]
        facet_data[sym] = {
            "n_trials": n_tr,
            "label": MEASURES[sym].label if sym in MEASURES else sym,
            "observed": obs_mean,
            "median": np.array([b["median"] for b in band]),
            "lo": np.array([b["lo"] for b in band]),
            "hi": np.array([b["hi"] for b in band]),
        }
        for t, w in enumerate(waves):
            rows.append(
                {
                    "outcome": sym,
                    "wave": int(w),
                    "timepoint": f"t{int(w)}",
                    "observed_mean": float(obs_mean[t]),
                    "predicted_median": band[t]["median"],
                    "predicted_lo": band[t]["lo"],
                    "predicted_hi": band[t]["hi"],
                    "predicted_lo50": band[t]["lo50"],
                    "predicted_hi50": band[t]["hi50"],
                    "n_trials": n_tr,
                }
            )
    summary = pd.DataFrame(rows)
    _draw_outcome_trajectory(
        output_dir, facet_data, waves=waves, summary=summary, ci_prob=ci_prob, name=name
    )
    return summary


def _draw_outcome_trajectory(
    output_dir: str,
    facet_data: Mapping[str, Mapping],
    *,
    waves: Sequence[int],
    summary: pd.DataFrame,
    ci_prob: float,
    name: str,
    ncols: int = 3,
) -> None:
    """Faceted per-measure trajectory figure (one panel per outcome, no arm split)."""
    syms = list(facet_data)
    n = len(syms)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    pct = int(round(ci_prob * 100))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.4 * ncols, 2.8 * nrows), squeeze=False
    )
    flat = axes.ravel()
    xs = np.arange(len(waves), dtype=float)
    for ax, sym in zip(flat[:n], syms, strict=False):
        d = facet_data[sym]
        ax.fill_between(xs, d["lo"], d["hi"], color=_INTERVENTION_COLOR, alpha=0.18, linewidth=0)
        ax.plot(xs, d["median"], color=_INTERVENTION_COLOR, lw=1.8)
        ax.scatter(xs, d["observed"], color=_OBSERVED_COLOR, s=32, zorder=5, edgecolor="white")
        ax.set_xticks(xs, [f"t{int(w)}" for w in waves], fontsize=7)
        ax.set_ylim(0, d["n_trials"])
        ax.set_title(f"{sym}: {d['label']}", fontsize=8)
        ax.tick_params(labelsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in flat[n:]:
        ax.axis("off")
    fig.supylabel("predicted score (items)", fontsize=9)
    fig.supxlabel("assessment wave", fontsize=9)
    fig.suptitle(
        f"Cohort score trajectory by measure — model mean ({pct}% CrI), "
        "observed dots (marginal over children; no arm split)",
        fontsize=10,
    )
    fig.tight_layout()
    save_styled_figure(output_dir, name, fig=fig, data=summary)


def write_child_fit_panel(
    output_dir: str,
    trace,
    panel,
    *,
    latent_name: str,
    focal_symbol: str,
    kappa_name: str = "kappa",
    pareto_k: np.ndarray | None = None,
    seed: int = 47,
    ci_prob: float = 0.95,
    n_panels: int = 12,
    max_draws: int = 4000,
    name: str = "child_fit_panels",
) -> pd.DataFrame:
    """Per-child fitted-vs-observed small multiples for a masked panel family (#317).

    One focal outcome (``focal_symbol``) to keep a single figure; the caption states
    the choice. Predictive ribbon = Beta-Binomial draws through ``sigmoid(latent)``
    (same-children — the latent trajectory carries each child's fitted intercept).
    """
    from language_reading_predictors.statistical_models.measures import MEASURES

    k_index = list(panel.outcomes).index(focal_symbol)
    n_tr = int(panel.n_trials[focal_symbol])
    latent = _stack_last(trace.posterior[latent_name])[:, :, k_index, :]  # (child, wave, S)
    cols = _thin_columns(latent.shape[-1], max_draws)
    latent = latent[..., cols]
    kappa = _stack_last(trace.posterior[kappa_name])  # (outcome, S) or (S,)
    kappa_s = kappa[k_index, cols] if kappa.ndim == 2 else kappa[cols]  # (S,)

    idx_i = _panel_child_index(panel)
    k_by_child = _pareto_k_by_child(pareto_k, idx_i)
    chosen, worst_set = select_children(
        k_by_child, panel.n_children, n_total=n_panels, seed=seed
    )

    rng = np.random.default_rng(seed)
    counts = panel.counts[focal_symbol]  # (child, wave)
    mask = panel.obs_mask[focal_symbol]  # (child, wave)
    waves = list(panel.waves)
    lo_q = (1 - ci_prob) / 2

    panels: list[dict] = []
    rows: list[dict] = []
    for pos, c in enumerate(chosen):
        mu = np.clip(expit(latent[c]), 1e-9, 1 - 1e-9)  # (wave, S)
        p = rng.beta(mu * kappa_s[None, :], (1 - mu) * kappa_s[None, :])
        y = rng.binomial(n_tr, p).astype(float)  # (wave, S)
        med = np.median(y, axis=1)
        lo = np.quantile(y, lo_q, axis=1)
        hi = np.quantile(y, 1 - lo_q, axis=1)
        obs_c = np.where(mask[c], counts[c], np.nan)
        k_val = k_by_child.get(int(c)) if k_by_child else None
        panels.append(
            {
                "ordinal": pos + 1,
                "worst": c in worst_set,
                "pareto_k": k_val,
                "waves": np.arange(len(waves)),
                "observed": obs_c,
                "median": med,
                "lo": lo,
                "hi": hi,
                "_wave_labels": [f"t{int(w)}" for w in waves],
            }
        )
        for t, w in enumerate(waves):
            rows.append(
                {
                    "panel": pos + 1,
                    "child_index": int(c),
                    "worst_fitting": bool(c in worst_set),
                    "pareto_k": k_val,
                    "wave": int(w),
                    "timepoint": f"t{int(w)}",
                    "observed": float(obs_c[t]) if np.isfinite(obs_c[t]) else np.nan,
                    "predicted_median": float(med[t]),
                    "predicted_lo": float(lo[t]),
                    "predicted_hi": float(hi[t]),
                    "scale": "items",
                    "outcome": focal_symbol,
                }
            )
    summary = pd.DataFrame(rows)
    label = MEASURES[focal_symbol].label if focal_symbol in MEASURES else focal_symbol
    _draw_small_multiples_panel(
        output_dir, panels, summary=summary, item_label=label,
        outcome_symbol=focal_symbol, n_trials=n_tr, ci_prob=ci_prob, name=name,
    )
    return summary


def _draw_small_multiples_panel(
    output_dir: str,
    panels: Sequence[Mapping],
    *,
    summary: pd.DataFrame,
    item_label: str,
    outcome_symbol: str,
    n_trials: int,
    ci_prob: float,
    name: str,
    ncols: int = 4,
) -> None:
    """Small-multiples grid for the panel families (explicit wave-label ticks)."""
    n = len(panels)
    if n == 0:  # pragma: no cover
        return
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    pct = int(round(ci_prob * 100))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.0 * ncols, 2.4 * nrows), sharex=True, sharey=True,
        squeeze=False,
    )
    flat = axes.ravel()
    for ax, p in zip(flat[:n], panels, strict=False):
        xs = np.asarray(p["waves"], dtype=float)
        ax.fill_between(xs, p["lo"], p["hi"], color=_FIT_COLOR, alpha=0.20, linewidth=0)
        ax.plot(xs, p["median"], color=_FIT_COLOR, lw=1.5)
        ax.scatter(
            xs, p["observed"], color=_OBSERVED_COLOR, s=30, zorder=5,
            edgecolor="white", linewidth=0.5,
        )
        ax.set_xticks(xs, p["_wave_labels"], fontsize=7)
        ax.set_ylim(0, n_trials)
        flag = f"\nhigh Pareto-k = {p['pareto_k']:.2f}" if p["worst"] and p["pareto_k"] is not None else ""
        ax.set_title(f"child #{p['ordinal']}{flag}", fontsize=8,
                     color=_OBSERVED_COLOR if p["worst"] else "black")
        ax.tick_params(labelsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in flat[n:]:
        ax.axis("off")
    fig.supylabel(f"score (out of {n_trials})", fontsize=9)
    fig.supxlabel("assessment wave", fontsize=9)
    fig.suptitle(
        f"Per-child fit ({outcome_symbol}, {item_label}) — observed dots, "
        f"posterior-predictive {pct}% band; same-children",
        fontsize=10,
    )
    fig.tight_layout()
    save_styled_figure(output_dir, name, fig=fig, data=summary)
