# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The mechanism family's declared natural-scale estimand (#319, #602).

The mechanism reports plot the HSGP (or linear) dose-response on the *logit
contribution* scale (``mechanism_curve.csv``: ``f_mech`` vs the exposure logit),
which is unreadable for the undergraduate-readability audience and hides that the
items-scale exchange rate varies along the curve. This module renders the same
fitted relation on the **items scale** — exposure items on the x-axis (e.g. "Letter
sounds known, out of 32"), predicted outcome items on the y-axis (e.g. "Words
read, out of 79") — with its credible ribbon, and computes the family's headline
contrast on it.

## One declared estimand (#602)

Until #602 the family published two different natural-scale answers to the same
question in the same report: ``mechanism_summary.csv`` contrasted the **observed
minimum and maximum** exposure and averaged fitted probabilities across rows, while
the items curve's worked example contrasted the **interquartile** exposure for a
constructed "typical child" — ``expit`` of a row-averaged linear predictor with the
child intercept removed. Neither was wrong; they were different estimands over
different intervals, and averaging probabilities is not applying the link to an
average, so the two would have disagreed even on one interval.

The family now declares a single headline estimand, and everything here computes
it:

**Reference population.** Standardise over the **fitted rows**: hold each row's own
phase, covariates, autoregressive baseline and *fitted child random intercept* at
their values, set only the exposure, and average the resulting predicted items. For
posterior draw ``s`` and exposure value ``x``,

    y(x, s) = N_outcome * mean_i expit( eta_base[i, s] + f_i(x, s) )

where ``eta_base[i, s] = eta[i, s] - f_i(x_i, s) - moderator[i, s]`` is that row's
linear predictor with the fitted mechanism and moderator contributions removed, and
``f_i(x, s)`` is the mechanism contribution row ``i`` would carry at exposure ``x``.
Because ``eta`` is registered as a deterministic, ``eta_base`` is recovered from it
rather than re-derived, so it cannot silently drift from the factory's term set. The
child intercept is **retained at its fitted value**, so the answer is "averaged over
the children actually analysed" rather than "for a constructed typical child"; any
moderator is held at its standardised mean (its main effect and interaction both
vanish).

**Exposure interval.** The **interquartile range of the fitted exposure**
(``items_ref_quantiles``, per-model configurable, ``(0.25, 0.75)`` by default). The
observed extremes are order statistics of a 156-row sample: they move with a single
child entering or leaving, and they are exactly where an HSGP curve is least
constrained. The full observed range is still reported, as an explicitly labelled
**secondary** contrast.

The plotted curve is the same standardised quantity evaluated across the observed
exposure grid, so the annotated worked-example points lie exactly on it, and the
headline number in ``mechanism_summary.csv``, ``key_findings.json`` and the figure
caption is one number computed once.

## Row-dependent mechanism terms

``f_i(x, s)`` depends on the row for two of the family's designs, which is why the
contribution is resolved as a function rather than a vector:

* **HSGP curve** and **pooled linear slope** — the contribution depends on the
  exposure only, so it is the same for every row at a given ``x``.
* **Between/within split** (#603) — ``f_i(x) = beta_between * mbar_i +
  beta_within * (z(x) - mbar_i)``: moving a wave's exposure holds that child's
  study average fixed, which is exactly the within-child reading. The between term
  cancels from any contrast, so the headline contrast is the *within-child* one.
* **Per-period slopes** (#604) — ``f_i(x) = beta_mech_phase[phase_i] * z(x)``, so
  the standardised contrast averages the per-period slopes over the fitted rows'
  period composition.

Everything here is an **adjusted association** under the DAG, never a causal
skill-to-skill effect — the one figure a student is most likely to over-read — so
the flag is drawn on the figure itself and stated in the caption.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from dse_research_utils.plot.styles import COLOUR_BLUE, COLOUR_RED
from scipy.special import expit

from language_reading_predictors.figure_io import save_styled_figure

__all__ = [
    "HEADLINE_ESTIMAND",
    "SECONDARY_ESTIMAND",
    "MechanismTerms",
    "mechanism_items_curve",
    "resolve_mechanism_terms",
    "save_mechanism_items_figure",
    "standardised_items_by_row",
    "write_mechanism_items_artifacts",
]

#: Machine-readable identifiers for the family's two natural-scale contrasts. They
#: are written into ``mechanism_summary.csv``, folded into ``config.json`` and read
#: back by the report partial and the key-findings box, so a reader (or a test) can
#: tell which quantity a number is without parsing prose.
HEADLINE_ESTIMAND = "mechanism_items_contrast_interquartile_fitted_rows"
SECONDARY_ESTIMAND = "mechanism_items_contrast_observed_range_fitted_rows"

#: Curve / ribbon and worked-example accents from the shared project palette
#: (``dse_research_utils.plot.styles``), matching ``_write_mechanism_curve``.
_CURVE_COLOR = COLOUR_BLUE
_WORKED_COLOR = COLOUR_RED


def _stack_obs(da: xr.DataArray) -> np.ndarray:
    """Return an ``(n_obs, n_sample)`` array from a posterior DataArray.

    The HSGP ``f_mech`` carries an auto-named obs dimension (e.g. ``f_mech_dim_0``)
    rather than ``obs_id``, so take whichever non-sample dim it has — the same
    handling as ``reporting.readiness_threshold``.
    """
    stacked = da.stack(sample=("chain", "draw"))
    obs_dim = next(d for d in stacked.dims if d != "sample")
    return stacked.transpose(obs_dim, "sample").values


def _constant_1d(trace: xr.DataTree, name: str) -> np.ndarray:
    return np.asarray(trace.constant_data[name].values).reshape(-1)


def _draws(post, name: str) -> np.ndarray:
    """Flat ``(n_sample,)`` draws of a scalar variable."""
    return post[name].stack(sample=("chain", "draw")).values.ravel()


def _interp_draws(xs: np.ndarray, fs: np.ndarray, x_ref: float) -> np.ndarray:
    """Linear interpolation of ``fs`` (``(U, S)``, rows keyed by sorted ``xs``) at ``x_ref``.

    Vectorised over draws: locate the bracketing pair once and blend. ``x_ref`` is
    clamped to the observed range, so the worked example never extrapolates.
    """
    if xs.size == 1:
        return fs[0]
    j = int(np.clip(np.searchsorted(xs, x_ref), 1, xs.size - 1))
    x0, x1 = xs[j - 1], xs[j]
    w = 0.0 if x1 == x0 else (float(np.clip(x_ref, xs[0], xs[-1])) - x0) / (x1 - x0)
    return fs[j - 1] * (1.0 - w) + fs[j] * w


def _exposure_to_z(
    x_exposure: np.ndarray, z_obs: np.ndarray, n_trials: int | None
) -> Callable[[float], float]:
    """Map an exposure value to the fitted standardised regressor ``z``.

    The factory builds ``z`` as ``(t(x) - mean) / sd`` with ``t`` the logit-safe
    transform for a bounded-count exposure and the identity for a standardised
    covariate, so ``z`` is an exact affine function of ``t(x)``. Recovering that
    affine map from the stored pairs — and checking it reproduces every fitted value
    — is both exact at arbitrary reference points and a row-identity guard: a
    re-loaded frame whose rows differ from the fitted ones will not fit.

    Falls back to interpolating ``z`` over the observed grid when the affine
    recovery is not exact (an unexpected transform), which is accurate at every
    observed exposure value and never extrapolates.
    """
    x = np.asarray(x_exposure, dtype=float)
    z = np.asarray(z_obs, dtype=float)

    def transform(values: np.ndarray) -> np.ndarray:
        if n_trials is None:
            return np.asarray(values, dtype=float)
        from language_reading_predictors.statistical_models.preprocessing import (
            logit_safe,
        )

        return np.asarray(logit_safe(np.asarray(values, dtype=float), n_trials), dtype=float)

    t = transform(x)
    if np.ptp(t) > 0:
        a, b = np.polyfit(t, z, 1)
        if float(np.max(np.abs(a * t + b - z))) <= 1e-6:
            return lambda value: float(a * transform(np.array([value]))[0] + b)

    order = np.argsort(x)
    xs, zs = x[order], z[order]
    return lambda value: float(np.interp(value, xs, zs))


@dataclass(frozen=True)
class MechanismTerms:
    """The fitted mechanism contribution, and what it would be at any exposure.

    ``fitted`` is the ``(n_obs, n_sample)`` contribution each row actually carries,
    which is what has to be subtracted from ``eta`` to build the reference linear
    predictor. ``contribution_at(x)`` returns the contribution those same rows would
    carry if the exposure were set to ``x``, as an array **broadcastable against**
    ``(n_obs, n_sample)``: ``(1, n_sample)`` for the designs whose term depends on
    the exposure alone (the HSGP curve and the pooled linear slope), and genuinely
    ``(n_obs, n_sample)`` for the between/within split and the per-period slopes.
    Returning the minimal shape matters at reporting tier, where one materialised
    ``(156, 36000)`` array is 45 MB and the curve evaluates ~30 of them.
    """

    kind: str
    fitted: np.ndarray
    contribution_at: Callable[[float], np.ndarray]


def resolve_mechanism_terms(
    trace: xr.DataTree,
    *,
    x_exposure: np.ndarray,
    exposure_n_trials: int | None = None,
    group: str = "posterior",
) -> MechanismTerms:
    """Resolve the fitted mechanism term for any of the family's exposure designs.

    Dispatch order matches the factory's own branches, and an unrecognised posterior
    raises rather than silently reporting a model without its exposure term.
    """
    post = getattr(trace, group)
    x_exposure = np.asarray(x_exposure, dtype=float).reshape(-1)

    if "f_mech" in post:
        f = _stack_obs(post["f_mech"])
        xs, first_idx = np.unique(x_exposure, return_index=True)
        fs = f[first_idx]

        def at_gp(value: float, _xs=xs, _fs=fs) -> np.ndarray:
            return _interp_draws(_xs, _fs, value)[None, :]

        return MechanismTerms(kind="GP", fitted=f, contribution_at=at_gp)

    if "z_mech_logit" not in getattr(trace, "constant_data", {}):
        raise KeyError(
            "trace has neither 'f_mech' nor the standardised exposure "
            "'z_mech_logit'; the mechanism contribution cannot be reconstructed"
        )
    z_obs = _constant_1d(trace, "z_mech_logit")
    z_of = _exposure_to_z(x_exposure, z_obs, exposure_n_trials)

    if "beta_between" in post:
        # Mundlak split (#603): moving a wave's exposure holds the child's own study
        # average fixed, so the counterfactual deviation is ``z(x) - mbar_i``.
        between = _draws(post, "beta_between")
        mbar = _constant_1d(trace, "mech_child_mean")
        dev = _constant_1d(trace, "mech_within_dev")
        within_slope, kind = _within_slope(post, trace, "linear_between_within")
        fitted = mbar[:, None] * between[None, :] + within_slope * dev[:, None]

        def at_split(value: float) -> np.ndarray:
            return mbar[:, None] * between[None, :] + within_slope * (
                z_of(value) - mbar[:, None]
            )

        return MechanismTerms(kind=kind, fitted=fitted, contribution_at=at_split)

    slope, kind = _within_slope(post, trace, "linear")
    fitted = slope * z_obs[:, None]

    def at_linear(value: float) -> np.ndarray:
        return slope * z_of(value)

    return MechanismTerms(kind=kind, fitted=fitted, contribution_at=at_linear)


def _within_slope(post, trace: xr.DataTree, base_kind: str) -> tuple[np.ndarray, str]:
    """Per-row slope draws ``(n_obs, n_sample)`` (or ``(1, n_sample)``) and the kind.

    ``beta_mech_phase`` (#604) gives each fitted row its own period's slope;
    ``beta_within`` and ``beta_mech`` are scalar and broadcast.
    """
    if "beta_mech_phase" in post:
        phase = _constant_1d(trace, "phase_idx").astype(int)
        per_phase = _stack_obs(post["beta_mech_phase"])  # (n_phases, S)
        return per_phase[phase], f"{base_kind}_phase_varying"
    if "beta_within" in post:
        return _draws(post, "beta_within")[None, :], base_kind
    if "beta_mech" in post:
        return _draws(post, "beta_mech")[None, :], base_kind
    raise KeyError(
        "posterior has none of 'f_mech', 'beta_mech', 'beta_within' or "
        "'beta_mech_phase'; the mechanism contribution cannot be reconstructed"
    )


def _moderator_contribution(
    trace: xr.DataTree, n_obs: int, group: str = "posterior"
) -> np.ndarray:
    """Per-observation moderator contribution ``gamma_mod*z_M + gamma_int*z_L*z_M``.

    Zero (broadcast) when the fit has no moderator. Read from the registered
    ``z_moderator`` / ``z_mech_logit`` constant-data nodes and the ``gamma_mod`` /
    ``gamma_int`` posteriors so the reference population holds the moderator at its
    standardised mean (z_M has sample-mean 0, so the main effect vanishes; the
    interaction is removed explicitly because ``mean_i(z_L*z_M)`` need not be 0).
    """
    post = getattr(trace, group)
    if "gamma_mod" not in post:
        return np.zeros((n_obs, 1))
    z_M = _constant_1d(trace, "z_moderator")  # (n_obs,)
    contrib = z_M[:, None] * _draws(post, "gamma_mod")[None, :]
    if "gamma_int" in post:
        z_L = _constant_1d(trace, "z_mech_logit")
        contrib = contrib + (z_L * z_M)[:, None] * _draws(post, "gamma_int")[None, :]
    return contrib


def _reference_linear_predictor(
    trace: xr.DataTree,
    terms: MechanismTerms,
    *,
    eta_name: str = "eta",
    group: str = "posterior",
) -> np.ndarray:
    """Per-row reference linear predictor ``eta_base`` (see the module docstring).

    Every non-mechanism term stays at its fitted value for that row — including the
    child random intercept, which is what makes the standardisation "over the
    children actually analysed". Only the mechanism contribution and any moderator
    terms are removed.
    """
    eta = _stack_obs(getattr(trace, group)[eta_name])  # (n_obs, S)
    return eta - terms.fitted - _moderator_contribution(trace, eta.shape[0], group)


def mechanism_items_curve(
    trace: xr.DataTree,
    *,
    x_exposure: np.ndarray,
    n_trials_outcome: int,
    exposure_n_trials: int | None = None,
    ci_prob: float = 0.95,
    ref_quantiles: tuple[float, float] = (0.25, 0.75),
    round_exposure: bool = True,
    outcome_off_floor: bool = False,
    eta_name: str = "eta",
    group: str = "posterior",
) -> tuple[pd.DataFrame, dict]:
    """Standardised items-scale mechanism curve and the family's declared contrasts.

    ``x_exposure`` is the observed exposure value per fitted row — the raw item
    count for a bounded-count measure exposure, or the raw covariate score for a
    covariate exposure (``mechanism_is_covariate``). Its ordering must match the
    model's observation order (i.e. ``prepared.post_counts[sym]`` on the fitted
    subset). ``n_trials_outcome`` is the outcome's item ceiling;
    ``exposure_n_trials`` is the *exposure's* ceiling, or ``None`` for a covariate
    exposure, and is used only to recover the exact exposure→``z`` map on the linear
    designs.

    Returns ``(curve_df, worked)``. ``curve_df`` has one row per distinct exposure
    value (the plotted curve as numbers): ``exposure`` and the predicted-outcome
    ``mean`` / central-interval / 50% columns, standardised over the fitted rows.
    When ``outcome_off_floor`` is True (future floored mechanism outcomes) the y
    quantity is the off-floor *probability* rather than an item count. ``worked``
    records the **headline** interquartile contrast — reference points, predicted
    outcomes, the difference with its credible interval and its tail probability —
    plus the **secondary** observed-range contrast under ``secondary``, both tagged
    with their machine-readable estimand ids.

    ``group`` selects the inference group: ``"posterior"`` for the reported curve,
    or ``"prior"`` to push the prior through this same transform for the
    estimand-scale prior check (#381). Running the prior through the identical
    reference-population and interpolation path is the point — a separate
    approximation would not be a check *of the reported quantity*.
    """
    x_exposure = np.asarray(x_exposure, dtype=float).reshape(-1)
    # Guard the row alignment *before* resolving the term: a mismatched vector would
    # otherwise index the fitted curve out of range with an opaque IndexError.
    n_fitted = int(_stack_obs(getattr(trace, group)[eta_name]).shape[0])
    if n_fitted != x_exposure.shape[0]:
        raise ValueError(
            f"x_exposure has {x_exposure.shape[0]} rows but the fit has {n_fitted}; "
            "pass the fitted-subset exposure vector."
        )
    terms = resolve_mechanism_terms(
        trace,
        x_exposure=x_exposure,
        exposure_n_trials=exposure_n_trials,
        group=group,
    )
    eta_base = _reference_linear_predictor(
        trace, terms, eta_name=eta_name, group=group
    )

    lo_q = (1 - ci_prob) / 2
    hi_q = 1 - lo_q
    scale = 1.0 if outcome_off_floor else float(n_trials_outcome)

    def standardised(value: float) -> np.ndarray:
        """``(n_sample,)`` predicted outcome, averaged over the fitted rows."""
        return scale * expit(eta_base + terms.contribution_at(value)).mean(axis=0)

    def logit_shift(value: float) -> np.ndarray:
        """``(n_sample,)`` mechanism contribution at ``value``, averaged over rows."""
        return terms.contribution_at(value).mean(axis=0)

    xs = np.unique(x_exposure)
    y = np.stack([standardised(float(v)) for v in xs])  # (U, S)
    curve_df = pd.DataFrame(
        {
            "exposure": xs,
            "outcome_mean": y.mean(axis=1),
            "outcome_lo": np.quantile(y, lo_q, axis=1),
            "outcome_hi": np.quantile(y, hi_q, axis=1),
            "outcome_lo50": np.quantile(y, 0.25, axis=1),
            "outcome_hi50": np.quantile(y, 0.75, axis=1),
        }
    )

    def contrast(x_lo: float, x_hi: float) -> dict:
        y_lo, y_hi = standardised(x_lo), standardised(x_hi)
        diff = y_hi - y_lo
        logit_diff = logit_shift(x_hi) - logit_shift(x_lo)
        return {
            "exposure_ref_low": float(x_lo),
            "exposure_ref_high": float(x_hi),
            "predicted_low_median": float(np.median(y_lo)),
            "predicted_high_median": float(np.median(y_hi)),
            "outcome_difference_median": float(np.median(diff)),
            "outcome_difference_lo": float(np.quantile(diff, lo_q)),
            "outcome_difference_hi": float(np.quantile(diff, hi_q)),
            "outcome_difference_lo50": float(np.quantile(diff, 0.25)),
            "outcome_difference_hi50": float(np.quantile(diff, 0.75)),
            "prob_pos": float(np.mean(diff > 0)),
            "logit_difference_median": float(np.median(logit_diff)),
            "logit_difference_lo": float(np.quantile(logit_diff, lo_q)),
            "logit_difference_hi": float(np.quantile(logit_diff, hi_q)),
        }

    q_lo, q_hi = float(ref_quantiles[0]), float(ref_quantiles[1])
    x_lo = float(np.quantile(x_exposure, q_lo))
    x_hi = float(np.quantile(x_exposure, q_hi))
    if round_exposure:
        x_lo, x_hi = float(round(x_lo)), float(round(x_hi))

    worked = {
        "curve_kind": terms.kind,
        "estimand": HEADLINE_ESTIMAND,
        "contrast": "headline_interquartile",
        "reference_population": "fitted_rows",
        "child_intercept": "retained_at_fitted_value",
        "n_trials_outcome": int(n_trials_outcome),
        "outcome_off_floor": bool(outcome_off_floor),
        "ref_quantile_low": q_lo,
        "ref_quantile_high": q_hi,
        "n_obs": int(x_exposure.size),
        "ci_prob": float(ci_prob),
        **contrast(x_lo, x_hi),
    }
    # The former headline, retained under an explicit label so nothing is lost and
    # nobody has to guess which interval a published number came from (#602).
    worked["secondary"] = {
        "estimand": SECONDARY_ESTIMAND,
        "contrast": "secondary_observed_range",
        "reference_population": "fitted_rows",
        "child_intercept": "retained_at_fitted_value",
        **contrast(float(xs[0]), float(xs[-1])),
    }
    return curve_df, worked


def standardised_items_by_row(
    trace: xr.DataTree,
    *,
    x_exposure: np.ndarray,
    n_trials_outcome: int,
    exposure_n_trials: int | None = None,
    outcome_off_floor: bool = False,
    eta_name: str = "eta",
    group: str = "posterior",
) -> np.ndarray:
    """``(n_obs, n_sample)`` standardised predicted outcome at each row's exposure.

    Row ``i`` carries ``y(x_i)`` — the declared reference population's predicted
    outcome *at that row's exposure value*, not that row's own fitted prediction. It
    is therefore the family's headline curve sampled at the observed exposure
    density, which is what the items-scale steepest-interval statistic needs:
    binning it and taking between-bin secants gives ``d E[y] / dx`` under the same
    population the headline contrast uses, rather than a relabelling of the
    latent-logit derivative (#602).

    Evaluated once per *distinct* exposure value and expanded back to rows, so the
    cost matches the curve rather than the row count.
    """
    x_exposure = np.asarray(x_exposure, dtype=float).reshape(-1)
    terms = resolve_mechanism_terms(
        trace,
        x_exposure=x_exposure,
        exposure_n_trials=exposure_n_trials,
        group=group,
    )
    eta_base = _reference_linear_predictor(
        trace, terms, eta_name=eta_name, group=group
    )
    scale = 1.0 if outcome_off_floor else float(n_trials_outcome)
    xs, inverse = np.unique(x_exposure, return_inverse=True)
    y_unique = np.stack(
        [
            scale * expit(eta_base + terms.contribution_at(float(v))).mean(axis=0)
            for v in xs
        ]
    )  # (U, S)
    return y_unique[inverse]


def _worked_sentence(
    worked: dict,
    *,
    exposure_noun: str,
    outcome_noun: str,
) -> str:
    """Plain-language, computed worked-example sentence for caption / prose."""
    unit = "" if worked["outcome_off_floor"] else " items"
    if worked["outcome_off_floor"]:
        # Probability difference reads in percentage points.
        m = 100.0 * worked["outcome_difference_median"]
        lo = 100.0 * worked["outcome_difference_lo"]
        hi = 100.0 * worked["outcome_difference_hi"]
        change = f"{m:+.0f} percentage points"
        crange = f"{lo:+.0f} to {hi:+.0f} pp"
    else:
        m = worked["outcome_difference_median"]
        lo = worked["outcome_difference_lo"]
        hi = worked["outcome_difference_hi"]
        change = f"{m:+.1f}{unit}"
        crange = f"{lo:+.1f} to {hi:+.1f}"
    return (
        f"A child at {worked['exposure_ref_high']:g} rather than "
        f"{worked['exposure_ref_low']:g} {exposure_noun} "
        f"(the {int(round(100 * worked['ref_quantile_high']))}th vs "
        f"{int(round(100 * worked['ref_quantile_low']))}th percentile of the "
        f"observed range) is predicted to differ by ≈ {change} on "
        f"{outcome_noun} "
        f"({int(round(100 * worked['ci_prob']))}% CrI {crange}), averaged over the "
        "children analysed with everything else held at their own values."
    )


def save_mechanism_items_figure(
    output_dir: str,
    curve_df: pd.DataFrame,
    worked: dict,
    *,
    x_label: str,
    y_label: str,
    exposure_noun: str,
    outcome_noun: str,
    title: str,
    name: str = "mechanism_curve_items",
) -> None:
    """Items-scale curve with credible ribbon and the worked-example annotation."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    x = curve_df["exposure"].to_numpy()
    # The ribbon is the ``outcome_lo``/``outcome_hi`` band, computed at the fit's
    # ``ci_prob`` — so the legend coverage must track it rather than hard-code 95%.
    _cov = int(round(100 * float(worked.get("ci_prob", 0.95))))
    ax.fill_between(
        x,
        curve_df["outcome_lo"],
        curve_df["outcome_hi"],
        color=_CURVE_COLOR,
        alpha=0.2,
        label=f"{_cov}% credible interval",
    )
    ax.plot(x, curve_df["outcome_mean"], color=_CURVE_COLOR, lw=2, label="posterior mean")

    # Worked-example points + connecting drop lines.
    x_lo, x_hi = worked["exposure_ref_low"], worked["exposure_ref_high"]
    y_lo, y_hi = worked["predicted_low_median"], worked["predicted_high_median"]
    ax.scatter([x_lo, x_hi], [y_lo, y_hi], color=_WORKED_COLOR, zorder=5, s=28)
    for xr_, yr_ in ((x_lo, y_lo), (x_hi, y_hi)):
        ax.plot([xr_, xr_], [ax.get_ylim()[0], yr_], color=_WORKED_COLOR, lw=0.8, ls=":")
    sentence = _worked_sentence(
        worked, exposure_noun=exposure_noun, outcome_noun=outcome_noun
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=10)
    # The worked example and the association flag go on the figure itself so the
    # curve cannot be read in isolation.
    ax.text(
        0.5,
        -0.30,
        sentence + "\nAdjusted association under the DAG — not a causal effect.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        wrap=True,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    save_styled_figure(output_dir, name, fig=fig, data=curve_df)


def mechanism_summary_table(worked: dict, *, exposure_unit: str) -> pd.DataFrame:
    """``mechanism_summary.csv``: the headline contrast first, the secondary second.

    Both rows describe the same estimand family — the standardised items-scale
    exposure contrast over the fitted rows — and differ only in the exposure
    interval, which is named in ``contrast`` and ``estimand`` on every row. The
    legacy column names (``items_*``, ``exposure_low``/``high``, ``prob_pos``) are
    kept so existing readers keep working; ``_kf_csv_row`` takes the first row,
    which is the headline by construction.
    """
    common = {
        "exposure_unit": exposure_unit,
        "curve_kind": worked["curve_kind"],
        "n_trials_outcome": worked["n_trials_outcome"],
        "n_obs": worked.get("n_obs"),
        "ci_prob": worked["ci_prob"],
    }

    def row(source: dict, *, quantiles: tuple[float, float] | None) -> dict:
        return {
            "contrast": source["contrast"],
            "estimand": source["estimand"],
            "reference_population": source["reference_population"],
            "child_intercept": source["child_intercept"],
            "exposure_quantile_low": None if quantiles is None else quantiles[0],
            "exposure_quantile_high": None if quantiles is None else quantiles[1],
            "exposure_low": source["exposure_ref_low"],
            "exposure_high": source["exposure_ref_high"],
            "predicted_low_median": source["predicted_low_median"],
            "predicted_high_median": source["predicted_high_median"],
            "items_median": source["outcome_difference_median"],
            "items_lo": source["outcome_difference_lo"],
            "items_hi": source["outcome_difference_hi"],
            "items_lo50": source["outcome_difference_lo50"],
            "items_hi50": source["outcome_difference_hi50"],
            "logit_median": source["logit_difference_median"],
            "logit_lo": source["logit_difference_lo"],
            "logit_hi": source["logit_difference_hi"],
            "prob_pos": source["prob_pos"],
            **common,
        }

    rows = [
        row(worked, quantiles=(worked["ref_quantile_low"], worked["ref_quantile_high"]))
    ]
    if worked.get("secondary"):
        rows.append(row(worked["secondary"], quantiles=None))
    return pd.DataFrame(rows)


def write_mechanism_items_artifacts(
    output_dir: str,
    trace: xr.DataTree,
    *,
    x_exposure: np.ndarray,
    outcome_symbol: str,
    outcome_label: str,
    n_trials_outcome: int,
    exposure_label: str,
    exposure_is_covariate: bool,
    exposure_n_trials: int | None = None,
    ci_prob: float = 0.95,
    ref_quantiles: tuple[float, float] = (0.25, 0.75),
    outcome_off_floor: bool = False,
    save_figure: bool = True,
) -> dict:
    """Compute and save ``mechanism_curve_items.csv`` + the items-scale figure.

    Returns the ``worked`` dict (augmented with the labels and axis text) so the
    caller can write ``mechanism_summary.csv`` from the same numbers and fold the
    reference points into ``config.json``. The numbers are written before the
    figure, so a plotting-backend failure cannot cost the deliverable; pass
    ``save_figure=False`` to skip the figure entirely (a backfill over a stored fit
    that only needs the tables).
    """
    round_exposure = not exposure_is_covariate  # item counts round; raw scores don't
    curve_df, worked = mechanism_items_curve(
        trace,
        x_exposure=x_exposure,
        n_trials_outcome=n_trials_outcome,
        exposure_n_trials=None if exposure_is_covariate else exposure_n_trials,
        ci_prob=ci_prob,
        ref_quantiles=ref_quantiles,
        round_exposure=round_exposure,
        outcome_off_floor=outcome_off_floor,
    )

    exposure_noun = (
        f"on {exposure_label} (raw score)"
        if exposure_is_covariate
        else f"on {exposure_label}"
    )
    x_label = (
        f"{exposure_label} — raw score"
        if exposure_is_covariate or exposure_n_trials is None
        else f"{exposure_label} — score out of {exposure_n_trials}"
    )
    if outcome_off_floor:
        y_label = f"P({outcome_label} off the floor at follow-up)"
        outcome_noun = f"their chance of being off the {outcome_label} floor"
    else:
        y_label = f"Predicted {outcome_label} — out of {n_trials_outcome}"
        outcome_noun = outcome_label
    title = f"Items-scale mechanism curve: {exposure_label} → {outcome_label}"

    worked.update(
        {
            "outcome_symbol": outcome_symbol,
            "outcome_label": outcome_label,
            "exposure_label": exposure_label,
            "exposure_is_covariate": bool(exposure_is_covariate),
            "exposure_noun": exposure_noun,
            "outcome_noun": outcome_noun,
            "x_label": x_label,
            "y_label": y_label,
            "caption": _worked_sentence(
                worked, exposure_noun=exposure_noun, outcome_noun=outcome_noun
            ),
        }
    )
    # Write the numbers first so the CSV survives even if the plotting backend
    # fails; the figure then re-attaches the same table as its #208 sidecar.
    curve_df.to_csv(
        os.path.join(output_dir, "mechanism_curve_items.csv"), index=False
    )
    if save_figure:
        save_mechanism_items_figure(
            output_dir,
            curve_df,
            worked,
            x_label=x_label,
            y_label=y_label,
            exposure_noun=exposure_noun,
            outcome_noun=outcome_noun,
            title=title,
        )
    return worked
