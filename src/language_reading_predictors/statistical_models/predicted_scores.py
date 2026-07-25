# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Predicted-scores contrast figures for the randomised-contrast families (#316).

The model reports previously showed the treatment effect only as a logit-scale
forest entry; nothing displayed what the model says about *actual test scores*.
This module adds the undergraduate-readability artefacts:

1. a **predicted-scores contrast panel** — the posterior-predictive distribution
   of the test score for a *new typical child* under the treated and untreated
   counterfactuals, medians marked and the items-scale difference annotated;
2. an **items-scale effect density** with the ROPE band shaded and the three
   probabilities (benefit >= delta / negligible / harm >= delta) printed on the
   plot — the numbers already in ``rope_summary.csv``, drawn rather than
   tabulated;
3. a **100-dot icon array** coloured by the same ROPE triple ("in 93 of 100
   plausible worlds the benefit exceeds one item"); and
4. ``predicted_scores.csv`` carrying the plotted quantities so report prose can
   cite them.

Two prediction populations, kept distinct (#391 finding 4). Covariate profiles
always come from the designated reference rows (for the gain family, the period-1
randomised transition — the same rows as ``treatment_marginal.csv``). They differ
in how the child random intercept is handled:

- **observed-child, sample-conditional** — the reference rows keep their *fitted*
  intercepts. This is the ``average_marginal_effect`` (and per-arm event
  probabilities), matching ``treatment_marginal.csv`` / ``rope_summary.csv``.
- **new-child, population-average** — the fitted intercept is removed and
  integrated over the population ``Normal(0, sigma_child)`` distribution. The
  simulated score distributions draw a fresh ``u`` per simulated child; the
  ``*_new_child_population`` rows integrate it out by Gauss–Hermite quadrature.

Because inverse-logit is nonlinear the two are not equal, so they are labelled and
tabulated separately rather than merged under one "new typical child" caption. A
model with no child random intercept has only one population (the two coincide).

Floor-rule outcomes (P, N) replace the score distributions with a paired
off-floor probability display (two bars with credible intervals) plus a
risk-difference density, mirroring the floor rule's binary estimand.

The counterfactual arithmetic here deliberately mirrors
``reporting._itt_ame_draws`` (eta0 = eta - delta*G, delta = term + interaction
contributions); ``tests/statistical_models/test_predicted_scores.py`` guards the
two against drift by asserting identical average-marginal-effect draws.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import expit
from scipy.stats import gaussian_kde

from dse_research_utils.plot.styles import (
    COLOUR_BLUE,
    COLOUR_GREEN,
    COLOUR_ORANGE,
    COLOUR_RED,
    FIGSIZE_LG,
)

from language_reading_predictors.figure_io import save_styled_figure

__all__ = [
    "PredictiveContrast",
    "counterfactual_predictive_contrast",
    "icon_array_counts",
    "predicted_scores_table",
    "save_icon_array",
    "save_predicted_distribution",
    "save_predicted_effect",
    "save_predicted_scores_panel",
    "write_predicted_scores_artifacts",
]

#: Arm colours from the shared project palette (``dse_research_utils.plot.styles``):
#: wait-list control orange and immediate intervention blue.
_CONTROL_COLOR = COLOUR_ORANGE
_INTERVENTION_COLOR = COLOUR_BLUE
#: Icon-array / ROPE-triple colours: benefit (green), negligible (neutral grey),
#: harm (red) — the semantic colours from the same palette, with a neutral for
#: the "no meaningful difference" band.
_BENEFIT_COLOR = COLOUR_GREEN
_ROPE_COLOR = "#c7c7c7"
_HARM_COLOR = COLOUR_RED


@dataclass
class PredictiveContrast:
    """Draws behind one predicted-scores panel.

    ``ame_prob`` (and its ``ame_items`` rescale) reproduce
    ``reporting._itt_ame_draws`` exactly — the guard test relies on this. It is
    the **observed-child, sample-conditional** effect: the reference rows carry
    their *fitted* child random intercepts, so ``ame_prob`` is an average over the
    fitted sample, not over a new child (#391 finding 4).

    ``ame_prob_new_child`` (and its ``ame_items_new_child`` rescale) is the
    companion **new-child, population-average** effect: the fitted intercept is
    removed and integrated over the population ``Normal(0, sigma_child)`` intercept
    distribution (Gauss–Hermite), so it answers "a *new* typical child" rather than
    "the fitted children on average". Because inverse-logit is nonlinear the two
    generally differ. Populated (with ``prob_*_new_child``) only when the model has
    a child random intercept; empty otherwise (then new-child == conditional).

    ``score_*`` hold simulated new-child test scores (empty for the binary
    off-floor path); ``prob_*`` hold the per-draw observed-child marginal
    event/score probabilities per arm.
    """

    ame_prob: np.ndarray
    n_trials: int
    prob_control: np.ndarray
    prob_intervention: np.ndarray
    score_control: np.ndarray = field(default_factory=lambda: np.empty(0))
    score_intervention: np.ndarray = field(default_factory=lambda: np.empty(0))
    ame_prob_new_child: np.ndarray = field(default_factory=lambda: np.empty(0))
    prob_control_new_child: np.ndarray = field(default_factory=lambda: np.empty(0))
    prob_intervention_new_child: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def ame_items(self) -> np.ndarray:
        return self.ame_prob * float(self.n_trials)

    @property
    def ame_items_new_child(self) -> np.ndarray:
        return self.ame_prob_new_child * float(self.n_trials)

    @property
    def score_difference(self) -> np.ndarray:
        """Paired treated-minus-untreated score per simulated child (CRN)."""
        return self.score_intervention - self.score_control


def _counterfactual_components(
    trace: xr.DataTree,
    *,
    G: np.ndarray,
    term: str,
    varying_term: str = "tau_i",
    eta_name: str = "eta",
    moderators: Sequence[tuple[str, np.ndarray]] | None = None,
    row_mask: np.ndarray | None = None,
    group: str = "posterior",
) -> tuple[np.ndarray, np.ndarray]:
    """Untreated linear predictor ``eta0`` and treatment contribution ``delta``.

    Mirrors ``reporting._itt_ame_draws`` (same eta0/delta construction, same
    moderator handling, same row_mask semantics) but returns the per-row
    components so the predictive simulation can push them through the
    likelihood. Both arrays are restricted to the ``row_mask`` rows and have
    shape ``(n_rows, S)``. Drift between the two implementations is caught by
    the guard test, which asserts identical AME draws.
    """
    posterior = getattr(trace, group)
    term_draws = posterior[term].stack(sample=("chain", "draw")).values.ravel()  # (S,)
    eta = (
        posterior[eta_name]
        .stack(sample=("chain", "draw"))
        .transpose("obs_id", "sample")
        .values
    )  # (n_obs, S)
    G = np.asarray(G, dtype=float)
    if G.shape[0] != eta.shape[0]:
        raise ValueError(
            f"G has {G.shape[0]} rows but eta has {eta.shape[0]} observations; "
            "pass built.prepared.G (aligned with the fitted subset)."
        )
    if varying_term and varying_term in posterior:
        delta = (
            posterior[varying_term]
            .stack(sample=("chain", "draw"))
            .transpose("obs_id", "sample")
            .values
        )  # (n_obs, S)
    else:
        delta = np.broadcast_to(term_draws[None, :], eta.shape)
    for coef_name, mod_vec in moderators or ():
        if coef_name not in posterior:
            raise KeyError(
                f"moderator coefficient {coef_name!r} not in the {group} group; "
                "the model must register it for the interaction-aware contrast."
            )
        coef_draws = posterior[coef_name].stack(sample=("chain", "draw")).values.ravel()
        mod = np.asarray(mod_vec, dtype=float)
        if mod.shape[0] != eta.shape[0]:
            raise ValueError(
                f"moderator {coef_name!r} has {mod.shape[0]} rows but eta has "
                f"{eta.shape[0]} observations; pass the fitted-subset vector."
            )
        delta = delta + np.outer(mod, coef_draws)
    eta0 = eta - delta * G[:, None]

    if row_mask is not None:
        m = np.asarray(row_mask)
        if m.ndim != 1:
            raise ValueError(f"row_mask must be 1-D, got a {m.ndim}-D array.")
        if m.dtype == bool and m.shape[0] != eta.shape[0]:
            raise ValueError(
                f"boolean row_mask has {m.shape[0]} entries but eta has "
                f"{eta.shape[0]} observations."
            )
        eta0 = eta0[m]
        delta = delta[m]
        if eta0.shape[0] == 0:
            raise ValueError("row_mask selects no observations for the contrast.")
    return np.ascontiguousarray(eta0), np.ascontiguousarray(delta)


def _new_child_adjustment(
    trace: xr.DataTree,
    eta0: np.ndarray,
    *,
    child_effect_name: str | None,
    child_sd_name: str | None,
    child_idx: np.ndarray | None,
    row_mask: np.ndarray | None,
    group: str = "posterior",
) -> tuple[np.ndarray, np.ndarray]:
    """Swap fitted child intercepts for the population intercept distribution.

    Returns ``(eta0_new_child, sigma_child)`` where ``eta0_new_child`` has the
    fitted ``u_child[child_idx]`` contribution removed and ``sigma_child`` is
    the per-posterior-sample intercept SD (shape ``(S,)``). The simulation
    draws a fresh ``Normal(0, sigma_child[s_j])`` intercept **per simulated
    child** from it, so children simulated from the same posterior draw still
    receive independent intercepts.
    """
    if child_effect_name is None:
        return eta0, np.zeros(eta0.shape[1])
    if child_sd_name is None or child_idx is None:
        raise ValueError("child_sd_name and child_idx are required with child_effect_name")
    posterior = getattr(trace, group)
    u_child = (
        posterior[child_effect_name]
        .stack(sample=("chain", "draw"))
        .transpose("child", "sample")
        .values
    )  # (n_children, S)
    idx = np.asarray(child_idx, dtype=int)
    if row_mask is not None:
        idx = idx[np.asarray(row_mask)]
    if idx.shape[0] != eta0.shape[0]:
        raise ValueError(
            f"child_idx selects {idx.shape[0]} rows but eta0 has {eta0.shape[0]}."
        )
    sigma = posterior[child_sd_name].stack(sample=("chain", "draw")).values.ravel()  # (S,)
    return eta0 - u_child[idx], sigma


# Gauss–Hermite degree for the new-child intercept integral. 32 nodes make the
# smooth 1-D Normal integral of expit effectively exact at negligible cost.
_GH_DEG = 32


def _new_child_prob(eta: np.ndarray, sigma_child: np.ndarray) -> np.ndarray:
    """Population-average event probability integrating the intercept out.

    Returns ``E_{u ~ Normal(0, sigma_child[s])}[expit(eta[r, s] + u)]`` per
    ``(row r, draw s)`` by Gauss–Hermite quadrature. With ``sigma_child == 0`` the
    nodes collapse and this returns ``expit(eta)`` exactly, so a model with no
    child intercept reduces to the conditional probability (new-child == observed).
    """
    nodes, weights = np.polynomial.hermite.hermgauss(_GH_DEG)
    norm = float(np.sqrt(np.pi))
    acc = np.zeros_like(eta, dtype=float)
    # E_{u~N(0,s)}[g(u)] = (1/sqrt(pi)) * sum_k w_k g(sqrt(2) s x_k).
    shift = np.sqrt(2.0) * sigma_child[None, :]  # (1, S)
    for x_k, w_k in zip(nodes, weights, strict=True):
        acc += w_k * expit(eta + shift * x_k)
    return acc / norm


def counterfactual_predictive_contrast(
    trace: xr.DataTree,
    *,
    G: np.ndarray,
    n_trials: int,
    term: str,
    varying_term: str = "tau_i",
    eta_name: str = "eta",
    moderators: Sequence[tuple[str, np.ndarray]] | None = None,
    row_mask: np.ndarray | None = None,
    likelihood: str = "beta_binomial",
    kappa_name: str = "kappa",
    child_effect_name: str | None = None,
    child_sd_name: str | None = None,
    child_idx: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    max_score_sims: int = 20_000,
) -> PredictiveContrast:
    """Posterior-predictive treated/untreated contrast for a new typical child.

    For every posterior draw the untreated linear predictor ``eta0`` and the
    per-row treatment contribution ``delta`` are recovered exactly as in
    ``reporting._itt_ame_draws``. The per-draw marginal probabilities average
    ``expit(eta0)`` / ``expit(eta0 + delta)`` over the reference rows, and
    ``ame_prob`` is their difference (the guard-tested AME).

    With ``likelihood="beta_binomial"`` the contrast additionally simulates
    integer test scores: each simulation picks one posterior draw and one
    reference row (common random numbers across the two arms), swaps any fitted
    child intercept for a fresh ``Normal(0, sigma_child)`` population draw, then
    samples ``p ~ Beta(mu*kappa, (1-mu)*kappa)`` and ``y ~ Binomial(n_trials,
    p)`` per arm. With ``likelihood="bernoulli"`` (the floor rule) no score
    simulation is run — the display quantity is the pair of marginal event
    probabilities.
    """
    if likelihood not in ("beta_binomial", "bernoulli"):
        raise ValueError(f"likelihood must be 'beta_binomial' or 'bernoulli', got {likelihood!r}")
    rng = np.random.default_rng(0) if rng is None else rng

    eta0, delta = _counterfactual_components(
        trace,
        G=G,
        term=term,
        varying_term=varying_term,
        eta_name=eta_name,
        moderators=moderators,
        row_mask=row_mask,
    )
    # AME on the fitted intercepts — identical to _itt_ame_draws / rope_summary,
    # so the CSV row and printed triple match the existing artefacts exactly.
    ame_prob = (expit(eta0 + delta) - expit(eta0)).mean(axis=0)
    prob_control = expit(eta0).mean(axis=0)
    prob_intervention = expit(eta0 + delta).mean(axis=0)

    contrast = PredictiveContrast(
        ame_prob=ame_prob,
        n_trials=int(n_trials),
        prob_control=prob_control,
        prob_intervention=prob_intervention,
    )

    # New-child, population-average effect: strip the fitted child intercept and
    # integrate over the population Normal(0, sigma_child) intercept distribution.
    # Computed for BOTH likelihoods whenever the model has a child intercept — the
    # floor-rule Bernoulli path used to return before any new-child step (#391
    # finding 4), so its reports could only show the observed-child conditional
    # effect. Reused below for the graded score simulation.
    eta0_new, sigma_child = _new_child_adjustment(
        trace,
        eta0,
        child_effect_name=child_effect_name,
        child_sd_name=child_sd_name,
        child_idx=child_idx,
        row_mask=row_mask,
    )
    if child_effect_name is not None:
        pc_new = _new_child_prob(eta0_new, sigma_child).mean(axis=0)
        pi_new = _new_child_prob(eta0_new + delta, sigma_child).mean(axis=0)
        contrast.prob_control_new_child = pc_new
        contrast.prob_intervention_new_child = pi_new
        contrast.ame_prob_new_child = pi_new - pc_new

    if likelihood == "bernoulli":
        return contrast

    posterior = trace.posterior
    if kappa_name not in posterior:
        raise KeyError(f"{kappa_name!r} not in the posterior; needed for score simulation")
    kappa = posterior[kappa_name].stack(sample=("chain", "draw")).values.ravel()  # (S,)

    n_rows, n_samples = eta0_new.shape
    n_sims = int(min(max_score_sims, max(n_samples, 4_000)))
    sample_j = (
        rng.choice(n_samples, size=n_sims, replace=n_sims > n_samples)
        if n_sims != n_samples
        else np.arange(n_samples)
    )
    row_j = rng.integers(0, n_rows, size=n_sims)

    # Fresh population intercept per *simulated child*: simulations that reuse
    # a posterior draw (n_sims > n_samples) must still get independent
    # intercepts, or child-to-child spread is understated.
    u_j = rng.normal(0.0, 1.0, size=n_sims) * sigma_child[sample_j]
    eta_c = eta0_new[row_j, sample_j] + u_j
    eta_t = eta_c + delta[row_j, sample_j]
    kappa_j = kappa[sample_j]

    def _scores(eta_arm: np.ndarray) -> np.ndarray:
        mu = np.clip(expit(eta_arm), 1e-9, 1 - 1e-9)
        p = rng.beta(mu * kappa_j, (1.0 - mu) * kappa_j)
        return rng.binomial(int(n_trials), p).astype(np.int64)

    contrast.score_control = _scores(eta_c)
    contrast.score_intervention = _scores(eta_t)
    return contrast


def predicted_scores_table(
    contrast: PredictiveContrast,
    *,
    outcome_symbol: str,
    ci_prob: float,
    population: str,
    contrast_status: str,
) -> pd.DataFrame:
    """Tabulate the plotted quantities for ``predicted_scores.csv``.

    One row per quantity, each tagged with the population its target is defined
    over (#391 finding 4). The ``average_marginal_effect`` row is the
    **observed-child, sample-conditional** effect whose median must agree with
    ``treatment_marginal.csv`` / ``rope_summary.csv`` (guard test). For graded
    outcomes the per-arm predictive score distributions and their paired difference
    (common random numbers → the treated-minus-untreated score for the *same*
    simulated child) are **new-child, population-average** quantities. For the
    binary floor rule: the per-arm off-floor probabilities (observed-child) and
    their risk difference.

    When the model has a child random intercept, a companion
    ``average_marginal_effect_new_child_population`` row (and, for the floor rule,
    ``event_probability_*_new_child_population``) is appended — the same effect with
    the intercept integrated out — so the two targets are never conflated under one
    population label. Absent a child intercept the two coincide and only the single
    labelled set is written.
    """
    lo_q = (1 - ci_prob) / 2
    hi_q = 1 - lo_q
    has_new = contrast.ame_prob_new_child.size > 0
    # `population` carries the covariate/subject provenance (shared by both targets
    # and used verbatim in report captions); the orthogonal `intercept_basis` column
    # records which random-intercept target the row is defined over.
    _OBSERVED = "observed-child; random intercept at fitted values"
    _NEWCHILD = "new-child; random intercept integrated over its population"
    _SINGLE = "single population; no child random intercept"
    observed_basis = _OBSERVED if has_new else _SINGLE
    newchild_basis = _NEWCHILD if has_new else _SINGLE

    def _row(quantity: str, draws: np.ndarray, scale: str, basis: str) -> dict:
        d = np.asarray(draws, dtype=float)
        return {
            "outcome": outcome_symbol,
            "quantity": quantity,
            "scale": scale,
            "median": float(np.median(d)),
            "lo": float(np.quantile(d, lo_q)),
            "hi": float(np.quantile(d, hi_q)),
            "lo50": float(np.quantile(d, 0.25)),
            "hi50": float(np.quantile(d, 0.75)),
            "n_trials": contrast.n_trials,
            "population": population,
            "intercept_basis": basis,
            "contrast_status": contrast_status,
        }

    rows: list[dict] = []
    if contrast.score_control.size:
        rows.append(_row("predicted_score_control", contrast.score_control, "items", newchild_basis))
        rows.append(
            _row("predicted_score_intervention", contrast.score_intervention, "items", newchild_basis)
        )
        rows.append(
            _row("predicted_score_paired_difference", contrast.score_difference, "items", newchild_basis)
        )
        rows.append(_row("average_marginal_effect", contrast.ame_items, "items", observed_basis))
        if has_new:
            rows.append(
                _row(
                    "average_marginal_effect_new_child_population",
                    contrast.ame_items_new_child,
                    "items",
                    _NEWCHILD,
                )
            )
    else:
        rows.append(_row("event_probability_control", contrast.prob_control, "probability", observed_basis))
        rows.append(
            _row(
                "event_probability_intervention",
                contrast.prob_intervention,
                "probability",
                observed_basis,
            )
        )
        rows.append(_row("average_marginal_effect", contrast.ame_prob, "risk_difference", observed_basis))
        if has_new:
            rows.append(
                _row(
                    "event_probability_control_new_child_population",
                    contrast.prob_control_new_child,
                    "probability",
                    _NEWCHILD,
                )
            )
            rows.append(
                _row(
                    "event_probability_intervention_new_child_population",
                    contrast.prob_intervention_new_child,
                    "probability",
                    _NEWCHILD,
                )
            )
            rows.append(
                _row(
                    "average_marginal_effect_new_child_population",
                    contrast.ame_prob_new_child,
                    "risk_difference",
                    _NEWCHILD,
                )
            )
    return pd.DataFrame(rows)


def _rope_triple(effect: np.ndarray, delta: float) -> tuple[float, float, float]:
    """(P(benefit >= delta), P(|effect| <= delta), P(harm >= delta)) — matching rope_card."""
    effect = np.asarray(effect, dtype=float)
    return (
        float(np.mean(effect >= delta)),
        float(np.mean(np.abs(effect) <= delta)),
        float(np.mean(effect <= -delta)),
    )


def _effect_density_axis(
    ax: plt.Axes,
    effect: np.ndarray,
    *,
    delta: float | None,
    unit_label: str,
    delta_unit: str = "",
) -> None:
    """Items/risk-difference effect density with ROPE band and the triple printed.

    ``effect`` and ``delta`` must share one display scale (items, or percentage
    points for the floor rule — the caller rescales); ``delta_unit`` is the
    short unit suffix printed after δ (e.g. ``" items"``, ``" pp"``).
    """
    effect = np.asarray(effect, dtype=float)
    grid = np.linspace(np.quantile(effect, 0.001), np.quantile(effect, 0.999), 512)
    try:
        density = gaussian_kde(effect)(grid)
    except Exception:  # pragma: no cover - degenerate posterior fallback
        ax.hist(effect, bins=60, density=True, color=_INTERVENTION_COLOR, alpha=0.5)
        grid = density = None
    if grid is not None:
        ax.fill_between(grid, density, color=_INTERVENTION_COLOR, alpha=0.35)
        ax.plot(grid, density, color=_INTERVENTION_COLOR, lw=1.0)
    if delta is not None:
        ax.axvspan(
            -delta, delta, color=_ROPE_COLOR, alpha=0.45,
            label=f"ROPE (±{delta:g}{delta_unit})",
        )
        p_benefit, p_rope, p_harm = _rope_triple(effect, delta)
        ax.text(
            0.02,
            0.98,
            (
                f"P(benefit ≥ {delta:g}{delta_unit}) = {p_benefit:.2f}\n"
                f"P(negligible, within ±{delta:g}{delta_unit}) = {p_rope:.2f}\n"
                f"P(harm ≥ {delta:g}{delta_unit}) = {p_harm:.2f}"
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )
    ax.axvline(0.0, color="k", lw=0.75, ls="--")
    ax.axvline(float(np.median(effect)), color=_INTERVENTION_COLOR, lw=1.25)
    ax.set_xlabel(unit_label)
    ax.set_ylabel("posterior density")
    if delta is not None:
        ax.legend(loc="upper right", fontsize=8)


def _draw_distribution_axis(
    ax: plt.Axes,
    contrast: PredictiveContrast,
    *,
    outcome_symbol: str,
    item_label: str,
    event_label: str,
) -> None:
    """Left-panel body: per-arm predicted-score distribution (graded) or off-floor
    probability bars (floor rule). Shared by the combined panel and the split file."""
    if contrast.score_control.size:
        n = contrast.n_trials
        bins = np.arange(-0.5, n + 1.5) if n <= 60 else 60
        ax.hist(contrast.score_control, bins=bins, density=True,
                color=_CONTROL_COLOR, alpha=0.55, label="wait-list control")
        ax.hist(contrast.score_intervention, bins=bins, density=True,
                color=_INTERVENTION_COLOR, alpha=0.55, label="immediate intervention")
        med_c = float(np.median(contrast.score_control))
        med_t = float(np.median(contrast.score_intervention))
        ame_med = float(np.median(contrast.ame_items))
        ax.axvline(med_c, color=_CONTROL_COLOR, lw=1.5)
        ax.axvline(med_t, color=_INTERVENTION_COLOR, lw=1.5)
        ax.set_xlabel(f"{item_label} — score out of {contrast.n_trials}")
        ax.set_ylabel("predictive density")
        ax.set_title(f"Predicted score, new typical child ({outcome_symbol})", fontsize=10)
        ax.legend(fontsize=8)
        ax.text(
            0.98, 0.98,
            f"medians: {med_t:.0f} vs {med_c:.0f}\naverage effect ≈ {ame_med:+.1f} items",
            transform=ax.transAxes, va="top", ha="right", fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )
    else:
        arms = ("wait-list control", "immediate intervention")
        draws = (contrast.prob_control, contrast.prob_intervention)
        colors = (_CONTROL_COLOR, _INTERVENTION_COLOR)
        for x, (label, d, color) in enumerate(zip(arms, draws, colors, strict=True)):
            med = float(np.median(d))
            lo, hi = float(np.quantile(d, 0.055)), float(np.quantile(d, 0.945))
            ax.bar(x, med, color=color, alpha=0.75, width=0.6, label=label)
            ax.errorbar(x, med, yerr=[[med - lo], [hi - med]], fmt="none", ecolor="k", capsize=4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(arms, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"P({event_label})")
        ax.set_title(f"Probability {event_label} by arm ({outcome_symbol})", fontsize=10)


def _draw_effect_axis(
    ax: plt.Axes, contrast: PredictiveContrast, *, delta: float | None
) -> None:
    """Right-panel body: items-scale (graded) or percentage-point (floor) effect
    density with the ROPE band and the three ROPE probabilities printed."""
    if contrast.score_control.size:
        _effect_density_axis(ax, contrast.ame_items, delta=delta,
                             unit_label="treatment effect (items)", delta_unit=" items")
        ax.set_title("Items-scale effect with ROPE", fontsize=10)
    else:
        # Percentage-point scale so the ROPE band / triple read "±10 pp".
        _effect_density_axis(
            ax, contrast.ame_prob * 100.0,
            delta=None if delta is None else float(delta) * 100.0,
            unit_label="risk difference (percentage points)", delta_unit=" pp",
        )
        ax.set_title("Risk difference with ROPE", fontsize=10)


def save_predicted_scores_panel(
    output_dir: str,
    contrast: PredictiveContrast,
    *,
    outcome_symbol: str,
    item_label: str,
    delta: float | None,
    summary: pd.DataFrame,
    event_label: str = "off the floor at follow-up",
    name: str = "predicted_scores",
) -> None:
    """Combined two-panel predicted-scores figure (#316 items 1–2).

    Retained for the families that present the two panels together; the ITT
    reports instead emit the two panels as individual files (see
    :func:`save_predicted_distribution` and :func:`save_predicted_effect`).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    _draw_distribution_axis(ax1, contrast, outcome_symbol=outcome_symbol,
                            item_label=item_label, event_label=event_label)
    _draw_effect_axis(ax2, contrast, delta=delta)
    fig.tight_layout()
    save_styled_figure(output_dir, name, fig=fig, data=summary)


def save_predicted_distribution(
    output_dir: str,
    contrast: PredictiveContrast,
    *,
    outcome_symbol: str,
    item_label: str,
    summary: pd.DataFrame,
    event_label: str = "off the floor at follow-up",
    name: str = "predicted_scores",
) -> None:
    """Individual predicted-distribution figure (the combined panel's left half)."""
    fig, ax = plt.subplots(figsize=FIGSIZE_LG)
    _draw_distribution_axis(ax, contrast, outcome_symbol=outcome_symbol,
                            item_label=item_label, event_label=event_label)
    fig.tight_layout()
    save_styled_figure(output_dir, name, fig=fig, data=summary)


def save_predicted_effect(
    output_dir: str,
    contrast: PredictiveContrast,
    *,
    delta: float | None,
    summary: pd.DataFrame,
    name: str = "predicted_effect",
) -> None:
    """Individual treatment-effect-with-ROPE figure (the combined panel's right half)."""
    fig, ax = plt.subplots(figsize=FIGSIZE_LG)
    _draw_effect_axis(ax, contrast, delta=delta)
    fig.tight_layout()
    save_styled_figure(output_dir, name, fig=fig, data=summary)


def icon_array_counts(
    p_benefit: float, p_rope: float, p_harm: float, *, total: int = 100
) -> tuple[int, int, int]:
    """Integer dot counts for the icon array, summing exactly to ``total``.

    Largest-remainder (Hamilton) rounding over the *four*-way split (benefit /
    negligible / harm / the small-effect remainder between them). The three ROPE
    probabilities may sum to less than 1 — effects between delta and the ROPE
    edge in either direction fall outside all three — so the remainder is folded
    into the negligible band for display, which the caption states. A sum
    materially above 1 has no coherent icon-array reading and is rejected;
    boundary ties from ``rope_card``'s inclusive comparisons (a draw exactly at
    delta counts as both benefit and negligible) may push the sum a hair over 1
    and are tolerated.
    """
    probs = np.asarray([p_benefit, p_rope, p_harm], dtype=float)
    if np.any(probs < 0) or np.any(probs > 1):
        raise ValueError("ROPE probabilities must lie in [0, 1]")
    if float(probs.sum()) > 1.0 + 1e-6:
        raise ValueError(
            f"ROPE probabilities sum to {float(probs.sum()):.6f} > 1; the three "
            "bands must be (near-)disjoint for an icon array"
        )
    remainder = max(0.0, 1.0 - float(probs.sum()))
    quotas = np.append(probs, remainder) * total
    counts = np.floor(quotas).astype(int)
    short = total - int(counts.sum())
    if short > 0:
        order = np.argsort(-(quotas - counts))
        counts[order[:short]] += 1
    elif short < 0:  # tolerated boundary-tie overshoot: trim the largest band
        counts[int(np.argmax(counts))] += short
    benefit, rope, harm, leftover = (int(c) for c in counts)
    return benefit, rope + leftover, harm


def save_icon_array(
    output_dir: str,
    *,
    p_benefit: float,
    p_rope: float,
    p_harm: float,
    delta_label: str,
    outcome_symbol: str,
    name: str = "icon_array",
) -> None:
    """100-dot icon array coloured by the ROPE triple (#316 item 3)."""
    n_benefit, n_rope, n_harm = icon_array_counts(p_benefit, p_rope, p_harm)
    colors = (
        [_BENEFIT_COLOR] * n_benefit + [_ROPE_COLOR] * n_rope + [_HARM_COLOR] * n_harm
    )
    fig, ax = plt.subplots(figsize=(5.4, 5.6))
    for i, color in enumerate(colors):
        # Fill row-by-row from the top-left so the benefit block reads first.
        row, col = divmod(i, 10)
        ax.add_patch(
            plt.Circle((col + 0.5, 9 - row + 0.5), 0.38, color=color, ec="white", lw=0.5)
        )
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"In {n_benefit} of 100 plausible worlds the benefit is at least "
        f"{delta_label} ({outcome_symbol})",
        fontsize=10,
    )
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=_BENEFIT_COLOR, label=f"benefit ≥ {delta_label} ({n_benefit})"),
        plt.Line2D([], [], marker="o", ls="", color=_ROPE_COLOR, label=f"smaller either way ({n_rope})"),
        plt.Line2D([], [], marker="o", ls="", color=_HARM_COLOR, label=f"harm ≥ {delta_label} ({n_harm})"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=8, frameon=False)
    data = pd.DataFrame(
        [
            {
                "outcome": outcome_symbol,
                "delta_label": delta_label,
                "p_benefit_ge_delta": float(p_benefit),
                "p_in_rope": float(p_rope),
                "p_harm_ge_delta": float(p_harm),
                "dots_benefit": n_benefit,
                "dots_negligible_or_small": n_rope,
                "dots_harm": n_harm,
            }
        ]
    )
    save_styled_figure(output_dir, name, fig=fig, data=data)


def write_predicted_scores_artifacts(
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
    moderators: Sequence[tuple[str, np.ndarray]] | None = None,
    row_mask: np.ndarray | None = None,
    likelihood: str = "beta_binomial",
    kappa_name: str = "kappa",
    child_effect_name: str | None = None,
    child_sd_name: str | None = None,
    child_idx: np.ndarray | None = None,
    delta: float | None = None,
    ci_prob: float = 0.95,
    population: str = "new typical child; covariates from the fitted reference rows",
    contrast_status: str = "randomised contrast",
    event_label: str = "off the floor at follow-up",
    random_seed: int | None = None,
    split: bool = False,
) -> pd.DataFrame:
    """Compute and save every #316 artefact; returns the summary table.

    With ``split=False`` writes the combined two-panel ``predicted_scores.png``.
    With ``split=True`` (the ITT reports) writes the two panels as individual
    files instead — ``predicted_scores.png`` (the distribution) and
    ``predicted_effect.png`` (the effect with ROPE) — each with its own sidecar
    CSV. ``predicted_scores.csv`` is always written, and when ``delta`` is
    available so is ``icon_array.png``/``.svg``/``.csv``.
    """
    rng = np.random.default_rng(random_seed)
    contrast = counterfactual_predictive_contrast(
        trace,
        G=G,
        n_trials=n_trials,
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
    summary = predicted_scores_table(
        contrast,
        outcome_symbol=outcome_symbol,
        ci_prob=ci_prob,
        population=population,
        contrast_status=contrast_status,
    )
    # Write the citable numbers first so the CSV survives even if a plotting
    # backend fails; the panel then attaches the same table as its #208 sidecar.
    summary.to_csv(os.path.join(output_dir, "predicted_scores.csv"), index=False)
    if split:
        save_predicted_distribution(
            output_dir, contrast, outcome_symbol=outcome_symbol,
            item_label=item_label, summary=summary, event_label=event_label,
        )
        save_predicted_effect(output_dir, contrast, delta=delta, summary=summary)
    else:
        save_predicted_scores_panel(
            output_dir,
            contrast,
            outcome_symbol=outcome_symbol,
            item_label=item_label,
            delta=delta,
            summary=summary,
            event_label=event_label,
        )
    if delta is not None:
        effect = contrast.ame_items if likelihood == "beta_binomial" else contrast.ame_prob
        p_benefit, p_rope, p_harm = _rope_triple(np.asarray(effect), float(delta))
        delta_label = (
            f"{delta:g} item{'s' if float(delta) != 1 else ''}"
            if likelihood == "beta_binomial"
            else f"{100 * float(delta):g} percentage points"
        )
        save_icon_array(
            output_dir,
            p_benefit=p_benefit,
            p_rope=p_rope,
            p_harm=p_harm,
            delta_label=delta_label,
            outcome_symbol=outcome_symbol,
        )
    return summary
