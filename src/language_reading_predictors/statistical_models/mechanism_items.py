# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Items-scale mechanism dose-response curve + worked example (#319).

The mechanism reports plot the HSGP (or linear) dose-response on the *logit
contribution* scale (``mechanism_curve.csv``: ``f_mech`` vs the exposure logit),
which is unreadable for the undergraduate-readability audience and hides that the
items-scale exchange rate varies along the curve. This module renders the same
fitted curve on the **items scale** — exposure items on the x-axis (e.g. "Letter
sounds known, out of 32"), predicted outcome items on the y-axis (e.g. "Words
read, out of 79") — with its credible ribbon, and annotates one data-driven
worked-example contrast on it.

The conversion holds the rest of the linear predictor at reference values and is
population-level over the child intercepts. Concretely, for posterior draw ``s``
the plotted outcome is

    y(x, s) = N_outcome * expit( C[s] + f_curve(x, s) )

where ``f_curve`` is the fitted mechanism contribution (the HSGP ``f_mech`` or the
linear ``beta_mech * z(exposure)``) and ``C[s]`` is the per-draw reference
constant — the linear predictor with the mechanism term, the child random
intercept and any moderator terms removed, averaged over the fitted rows. Because
``eta`` is registered as a deterministic, ``C[s]`` is recovered as

    C[s] = mean_i( eta[i,s] - f_curve[i,s] - u_child[i,s] - moderator[i,s] )

i.e. every non-mechanism covariate sits at its fitted-sample mean, the child
intercept at its population mean (0), and the moderator at its mean (so its main
effect and interaction both vanish). This avoids re-deriving the factory's term
set and cannot silently drift from it.

Everything here is an **adjusted association** under the DAG, never a causal
skill-to-skill effect — the one figure a student is most likely to over-read — so
the flag is drawn on the figure itself and stated in the caption.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import expit

from language_reading_predictors.figure_io import save_styled_figure

__all__ = [
    "mechanism_items_curve",
    "save_mechanism_items_figure",
    "write_mechanism_items_artifacts",
]

#: Curve / ribbon colour, matching ``_write_mechanism_curve``'s logit-scale plot.
_CURVE_COLOR = "#1f77b4"
#: Worked-example annotation colour (a warm contrast to the cool curve).
_WORKED_COLOR = "#d62728"


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


def _mechanism_contribution(trace: xr.DataTree) -> tuple[np.ndarray, str]:
    """Per-observation mechanism contribution ``f_curve`` and its kind.

    ``f_mech`` (the HSGP / phase-specific curve) when present, else the linear
    ``beta_mech * z(exposure)`` reconstructed from the registered ``z_mech_logit``
    constant-data node — mirroring ``pipeline._write_mechanism_curve``.
    """
    post = trace.posterior
    if "f_mech" in post:
        return _stack_obs(post["f_mech"]), "GP"
    if "beta_mech" in post:
        b = post["beta_mech"].stack(sample=("chain", "draw")).values.ravel()  # (S,)
        z_L = _constant_1d(trace, "z_mech_logit")  # (n_obs,)
        return z_L[:, None] * b[None, :], "linear"
    raise KeyError(
        "trace has neither 'f_mech' nor 'beta_mech' in the posterior; the "
        "items-scale mechanism curve needs one of them (phase-specific fits "
        "register the per-obs curve as 'f_mech')."
    )


def _moderator_contribution(
    trace: xr.DataTree, n_obs: int
) -> np.ndarray:
    """Per-observation moderator contribution ``gamma_mod*z_M + gamma_int*z_L*z_M``.

    Zero (broadcast) when the fit has no moderator. Read from the registered
    ``z_moderator`` / ``z_mech_logit`` constant-data nodes and the ``gamma_mod`` /
    ``gamma_int`` posteriors so the reference constant holds the moderator at its
    mean (z_M has sample-mean 0, so the main effect vanishes; the interaction is
    removed explicitly because ``mean_i(z_L*z_M)`` need not be 0).
    """
    post = trace.posterior
    if "gamma_mod" not in post:
        return np.zeros((n_obs, 1))
    z_M = _constant_1d(trace, "z_moderator")  # (n_obs,)
    gamma_mod = post["gamma_mod"].stack(sample=("chain", "draw")).values.ravel()  # (S,)
    contrib = z_M[:, None] * gamma_mod[None, :]
    if "gamma_int" in post:
        z_L = _constant_1d(trace, "z_mech_logit")
        gamma_int = (
            post["gamma_int"].stack(sample=("chain", "draw")).values.ravel()
        )
        contrib = contrib + (z_L * z_M)[:, None] * gamma_int[None, :]
    return contrib


def _reference_constant(
    trace: xr.DataTree,
    f_curve: np.ndarray,
    *,
    eta_name: str = "eta",
    child_effect_name: str = "u_child",
) -> np.ndarray:
    """Per-draw reference constant ``C[s]`` (see the module docstring)."""
    post = trace.posterior
    eta = _stack_obs(post[eta_name])  # (n_obs, S)
    n_obs = eta.shape[0]

    baseline = eta - f_curve
    if child_effect_name in post:
        child_idx = _constant_1d(trace, "child_idx").astype(int)
        u_child = _stack_obs(post[child_effect_name])  # (n_children, S)
        baseline = baseline - u_child[child_idx]
    baseline = baseline - _moderator_contribution(trace, n_obs)
    return baseline.mean(axis=0)  # (S,)


def _interp_draws(
    xs: np.ndarray, fs: np.ndarray, x_ref: float
) -> np.ndarray:
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


def mechanism_items_curve(
    trace: xr.DataTree,
    *,
    x_exposure: np.ndarray,
    n_trials_outcome: int,
    ci_prob: float = 0.95,
    ref_quantiles: tuple[float, float] = (0.25, 0.75),
    round_exposure: bool = True,
    outcome_off_floor: bool = False,
    eta_name: str = "eta",
    child_effect_name: str = "u_child",
) -> tuple[pd.DataFrame, dict]:
    """Items-scale mechanism curve and a data-driven worked-example contrast.

    ``x_exposure`` is the observed exposure value per fitted row — the raw item
    count for a bounded-count measure exposure, or the raw covariate score for a
    covariate exposure (``mechanism_is_covariate``). Its ordering must match the
    model's observation order (i.e. ``prepared.post_counts[sym]`` on the fitted
    subset). ``n_trials_outcome`` is the outcome's item ceiling.

    Returns ``(curve_df, worked)``. ``curve_df`` has one row per distinct exposure
    value (the plotted curve as numbers): ``exposure`` and the predicted-outcome
    ``mean`` / central-interval / 90% columns. When ``outcome_off_floor`` is True
    (future floored mechanism outcomes) the y quantity is the off-floor
    *probability* rather than an item count. ``worked`` records the quantile
    reference points and the predicted difference between them, with its credible
    interval — everything the caption states, computed not hand-written.
    """
    x_exposure = np.asarray(x_exposure, dtype=float).reshape(-1)
    f_curve, kind = _mechanism_contribution(trace)
    if f_curve.shape[0] != x_exposure.shape[0]:
        raise ValueError(
            f"x_exposure has {x_exposure.shape[0]} rows but the fitted mechanism "
            f"curve has {f_curve.shape[0]}; pass the fitted-subset exposure vector."
        )
    C = _reference_constant(
        trace, f_curve, eta_name=eta_name, child_effect_name=child_effect_name
    )  # (S,)

    lo_q = (1 - ci_prob) / 2
    hi_q = 1 - lo_q

    def _to_outcome(eta_items: np.ndarray) -> np.ndarray:
        p = expit(eta_items)
        return p if outcome_off_floor else float(n_trials_outcome) * p

    # Distinct exposure values (the deterministic curve is identical for rows that
    # share an exposure), sorted for a clean plotted line and interpolation.
    xs, first_idx = np.unique(x_exposure, return_index=True)
    fs = f_curve[first_idx]  # (U, S)
    eta_items = C[None, :] + fs  # (U, S)
    y = _to_outcome(eta_items)  # (U, S)
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

    # Worked example: predicted-outcome difference between two fixed quantiles of
    # the observed exposure distribution (data-driven reference points).
    q_lo, q_hi = float(ref_quantiles[0]), float(ref_quantiles[1])
    x_lo = float(np.quantile(x_exposure, q_lo))
    x_hi = float(np.quantile(x_exposure, q_hi))
    if round_exposure:
        x_lo, x_hi = float(round(x_lo)), float(round(x_hi))
    y_lo = _to_outcome(C + _interp_draws(xs, fs, x_lo))  # (S,)
    y_hi = _to_outcome(C + _interp_draws(xs, fs, x_hi))  # (S,)
    diff = y_hi - y_lo

    worked = {
        "curve_kind": kind,
        "n_trials_outcome": int(n_trials_outcome),
        "outcome_off_floor": bool(outcome_off_floor),
        "ref_quantile_low": q_lo,
        "ref_quantile_high": q_hi,
        "exposure_ref_low": x_lo,
        "exposure_ref_high": x_hi,
        "predicted_low_median": float(np.median(y_lo)),
        "predicted_high_median": float(np.median(y_hi)),
        "outcome_difference_median": float(np.median(diff)),
        "outcome_difference_lo": float(np.quantile(diff, lo_q)),
        "outcome_difference_hi": float(np.quantile(diff, hi_q)),
        "outcome_difference_lo50": float(np.quantile(diff, 0.25)),
        "outcome_difference_hi50": float(np.quantile(diff, 0.75)),
        "ci_prob": float(ci_prob),
    }
    return curve_df, worked


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
        f"({int(round(100 * worked['ci_prob']))}% CrI {crange}), all else equal."
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
) -> dict:
    """Compute and save ``mechanism_curve_items.csv`` + the items-scale figure.

    Returns the ``worked`` dict (augmented with the labels and axis text) so the
    caller can fold the quantile reference points into ``config.json`` and the
    report partial can render the computed caption. Never raises through to the
    fit: a failure logs and returns ``{}`` (the logit curve remains the analyst's
    object and is written independently).
    """
    round_exposure = not exposure_is_covariate  # item counts round; raw scores don't
    curve_df, worked = mechanism_items_curve(
        trace,
        x_exposure=x_exposure,
        n_trials_outcome=n_trials_outcome,
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
