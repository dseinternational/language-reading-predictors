# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Readiness calculations and summaries."""

from __future__ import annotations

from typing import Any

from language_reading_predictors.statistical_models.posteriors import REPORTING_CI_PROB

import warnings
import numpy as np
import xarray as xr
from language_reading_predictors.statistical_models.posteriors import (
    derived_mc_diagnostics,
)


#: Qualification thresholds for calling a steepest latent-logit interval a "knee"
#: (#586 finding 1). ``_KNEE_MIN_INCREASING`` is the pre-existing net-rise share;
#: ``_KNEE_MIN_CURVATURE`` is the shared evidence ladder's "moderate" rung (10:1
#: odds, ``dse_research_utils.statistics.evidence``), applied to the local slope
#: contrast so a straight line — which sits near 0.5 whatever its net rise — can
#: never qualify.
_KNEE_MIN_INCREASING: float = 0.9


_KNEE_MIN_CURVATURE: float = 0.91


def _readiness_knee(
    f: np.ndarray,
    ell: np.ndarray | None,
    *,
    n_trials: int | None = None,
    count_values: np.ndarray | None = None,
    ci_prob: float = REPORTING_CI_PROB,
    n_bins: int = 6,
    n_chains: int | None = None,
    n_draws: int | None = None,
) -> dict[str, Any]:
    """Locate the steepest latent-logit interval of a per-observation ``f_mech`` posterior.

    Pure-numpy core of :func:`readiness_threshold` (split out so the logic is
    unit-testable without a trace, #293 review). ``f`` is ``(n_obs, n_draws)`` HSGP
    curve draws; ``ell`` is the ``(n_obs,)`` Haldane-corrected mechanism logit.

    **What the statistic is.** The located quantity is the between-bin interval with
    the largest derivative of ``f_mech`` — the *steepest latent-logit interval*. The
    derivative is taken on the outcome-**logit** scale, because ``f_mech`` is a logit
    contribution; the expected-items derivative carries an extra ``p * (1 - p)``
    inverse-link factor and can peak at a different exposure value (#586 finding 1).
    ``scale`` records this so no downstream renderer can silently call it an items
    result. ``half_rise_count_*`` is a complementary mid-rise summary (where the
    curve first reaches the midpoint of its binned range). Both are summarised over
    the *increasing* draws only (net end-to-end rise on the binned curve; the share
    is ``increasing_frac``) — on a flat or falling draw the estimands are undefined.

    **When it may be called a knee.** A net rise is not a threshold: a perfectly
    linear increasing curve has ``increasing_frac == 1`` and still yields an
    ``argmax``, and a curve that accelerates all the way to the edge of its support
    pins that ``argmax`` on the last interval, where it is censored by the data
    rather than located by them. Both failure modes were live — the letter-sound
    fits put 73% of draws in the top interval with the knee median equal to its own
    upper credible limit (#586 finding 1). ``knee_well_defined`` is therefore a
    conjunction of three checks, each also reported on its own so a reader can see
    which one failed:

    - ``increasing_frac`` > ``_KNEE_MIN_INCREASING`` — the curve rises at all;
    - ``not boundary_pinned`` — the modal steepest interval is interior, so the
      location is identified rather than censored by the end of the observed range;
    - ``prob_slope_above_gt_below`` >= ``_KNEE_MIN_CURVATURE`` — the mean slope above
      the located interval genuinely exceeds the mean slope below it. For a straight
      line this probability sits near 0.5 whatever ``increasing_frac`` says, which is
      what separates a bend from a constant rise.

    ``steepest_interval_share`` is the share of increasing draws whose ``argmax``
    falls in the modal interval — selection stability, low when the intervals are
    effectively tied.
    """
    if count_values is not None:
        # Continuous-covariate exposure (e.g. intervention sessions, LRP92): the knee
        # is located in the exposure's own raw units directly — there is no bounded
        # count and no logit -> count back-transform. ``knee_count_*`` /
        # ``half_rise_count_*`` / ``obs_count_*`` then read in those raw units.
        L = np.asarray(count_values, dtype=float).reshape(-1)
    else:
        # Inverse Haldane-corrected logit -> approximate predictor count, clipped to range.
        # ell = log((y+0.5)/(n-y+0.5)) => expit(ell) = (y+0.5)/(n+1), so y = (n+1)*expit(ell) - 0.5
        # (the denominator is n+1, not n; #293 review).
        if ell is None or n_trials is None:
            raise ValueError("_readiness_knee needs ell + n_trials unless count_values is given.")
        L = np.clip((n_trials + 1.0) / (1.0 + np.exp(-ell)) - 0.5, 0.0, float(n_trials))

    edges = np.unique(np.quantile(L, np.linspace(0.0, 1.0, n_bins + 1)))
    nb = len(edges) - 1
    if nb < 2:
        raise ValueError("Too few distinct predictor bins to locate a knee.")
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(L, edges[1:-1]), 0, nb - 1)
    binmean = np.full((nb, f.shape[1]), np.nan)
    for b in range(nb):
        m = idx == b
        if m.any():
            binmean[b] = f[m].mean(axis=0)
    slope = np.diff(binmean, axis=0) / np.diff(centers)[:, None]  # (nb-1, S)
    knee_bin = np.nanargmax(slope, axis=0)  # steepest-rise interval per draw
    knee_L = 0.5 * (centers[knee_bin] + centers[knee_bin + 1])  # (S,)

    # Net end-to-end rise per draw; the estimand summaries pool these draws only.
    increasing = binmean[-1] > binmean[0]  # (S,) — NaN endpoints compare False

    # Per-draw half-rise: where the binned curve first reaches the midpoint of its
    # range, linearly interpolated between the straddling bin centres.
    lo_f = np.nanmin(binmean, axis=0)  # (S,)
    hi_f = np.nanmax(binmean, axis=0)
    target = 0.5 * (lo_f + hi_f)
    first = np.argmax(binmean >= target[None, :], axis=0)  # first bin at/above midpoint
    half_L = np.full(f.shape[1], centers[0])  # first==0: starts at/above the midpoint
    interior = first > 0
    if interior.any():
        s = np.flatnonzero(interior)
        j = first[s]
        f_lo, f_hi = binmean[j - 1, s], binmean[j, s]
        with np.errstate(invalid="ignore", divide="ignore"):
            t = np.where(f_hi > f_lo, (target[s] - f_lo) / (f_hi - f_lo), 0.0)
        half_L[s] = centers[j - 1] + t * (centers[j] - centers[j - 1])

    lo, hi = (1 - ci_prob) / 2, 1 - (1 - ci_prob) / 2

    def _q(a: np.ndarray) -> tuple[float, float, float]:
        a = a[np.isfinite(a)]
        if not a.size:
            return (float("nan"),) * 3
        return (
            float(np.median(a)),
            float(np.quantile(a, lo)),
            float(np.quantile(a, hi)),
        )

    kmed, k_lo, k_hi = _q(knee_L[increasing])
    hmed, h_lo, h_hi = _q(half_L[increasing])

    # Classify each between-bin interval by its midpoint relative to the median knee, so
    # the knee interval itself counts as "above" and the "above" set is never empty when
    # the steepest rise is the top interval (a late-accelerating curve).
    if np.isfinite(kmed):
        interval_mid = 0.5 * (centers[:-1] + centers[1:])
        below = interval_mid < kmed
        # An all-NaN "below" set is a real outcome, not an error: it means the
        # steepest interval is the lowest one, so there is nothing below it to
        # average. It stays NaN and the renderer must say so rather than print it.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            slope_below = np.nanmean(np.where(below[:, None], slope, np.nan), axis=0)
            slope_above = np.nanmean(np.where(~below[:, None], slope, np.nan), axis=0)
    else:  # no increasing draws — the below/above split is undefined
        slope_below = slope_above = np.full(f.shape[1], np.nan)

    def _med(a: np.ndarray) -> float:
        a = a[np.isfinite(a) & increasing]
        return float(np.median(a)) if a.size else float("nan")

    # --- qualification diagnostics (#586 finding 1) -------------------------
    # Selection stability and boundary censoring are properties of the *argmax*
    # over intervals, so they are computed on the increasing draws' winning bins.
    n_intervals = nb - 1
    kb_increasing = knee_bin[increasing]
    if kb_increasing.size:
        counts = np.bincount(kb_increasing, minlength=n_intervals)
        modal_interval = int(np.argmax(counts))
        interval_share = float(counts[modal_interval] / kb_increasing.size)
        # An argmax on the first or last interval is censored by the end of the
        # observed range: the curve may go on steepening where there are no data,
        # so the location is a bound, not an estimate.
        boundary_pinned = modal_interval in (0, n_intervals - 1)
    else:
        modal_interval, interval_share, boundary_pinned = -1, float("nan"), True

    # Local slope contrast: does the curve genuinely bend? For a straight line the
    # above/below means coincide and this sits near 0.5 however strongly the curve
    # rises, which is precisely the case ``increasing_frac`` cannot detect.
    contrast = slope_above - slope_below
    contrast = contrast[np.isfinite(contrast) & increasing]
    prob_curvature = float(np.mean(contrast > 0)) if contrast.size else float("nan")

    increasing_frac = float(np.mean(increasing))
    well_defined = bool(
        increasing_frac > _KNEE_MIN_INCREASING
        and not boundary_pinned
        and np.isfinite(prob_curvature)
        and prob_curvature >= _KNEE_MIN_CURVATURE
    )

    result = {
        "knee_count_median": kmed,
        "knee_count_ci_low": k_lo,
        "knee_count_ci_high": k_hi,
        "half_rise_count_median": hmed,
        "half_rise_count_ci_low": h_lo,
        "half_rise_count_ci_high": h_hi,
        "slope_below_knee_median": _med(slope_below),
        "slope_above_knee_median": _med(slope_above),
        "increasing_frac": increasing_frac,
        # The derivative is a logit-scale contribution, never expected items: the
        # items-scale maximum carries an extra p*(1-p) factor and can sit elsewhere.
        "scale": "latent_logit",
        "steepest_interval_index": modal_interval,
        "steepest_interval_share": interval_share,
        "boundary_pinned": boundary_pinned,
        "prob_slope_above_gt_below": prob_curvature,
        "knee_well_defined": well_defined,
        "obs_count_min": float(L.min()),
        "obs_count_max": float(L.max()),
        "ci_prob": float(ci_prob),
        "n_draws": int(f.shape[1]),
        "n_obs": int(f.shape[0]),
        "n_bins": int(nb),
    }
    # Monte-Carlo precision of the derived knee location (a non-smooth argmax over
    # binned draws, so it can mix worse than its parent GP weights). ESS is computed
    # over all draws to keep the chain layout; the reported median/CI pool the
    # ``increasing`` subset (share ``increasing_frac``).
    if n_chains is not None and n_draws is not None:
        result.update(
            derived_mc_diagnostics(
                knee_L, n_chains=n_chains, n_draws=n_draws, prefix="knee_"
            )
        )
    return result


def readiness_threshold(
    trace: xr.DataTree,
    *,
    n_trials: int | None = None,
    exposure_values: np.ndarray | None = None,
    ci_prob: float = REPORTING_CI_PROB,
    n_bins: int = 6,
    curve: np.ndarray | None = None,
    scale: str = "latent_logit",
) -> dict[str, Any]:
    """Steepest latent-logit interval of a mechanism curve (#230 §2/§5, #586 finding 1).

    Post-processes an HSGP mechanism model's adjusted curve ``f_mech`` to locate the
    interval over which it rises fastest, in the predictor's raw count units. For each
    posterior draw the per-observation ``f_mech`` is binned over the observed predictor
    range (quantile bins) and the steepest between-bin rise is found; the reported
    location is that interval's midpoint, giving a posterior over it. Reports its
    median + equal-tailed CI, a complementary half-rise summary, and the mean marginal
    slope below vs above it.

    The derivative is on the outcome-**logit** scale (``f_mech`` is a logit
    contribution), not the items scale: the expected-items derivative carries an extra
    ``p * (1 - p)`` factor and its maximum can fall at a different exposure value. The
    returned ``scale`` field records this.

    A located interval is **not** by itself a threshold. ``knee_well_defined``
    combines the net-rise share with a boundary check (an ``argmax`` on the first or
    last interval is censored by the end of the observed range) and a local
    slope-contrast probability (near 0.5 for a straight line). Read
    ``increasing_frac``, ``boundary_pinned``, ``steepest_interval_share`` and
    ``prob_slope_above_gt_below`` alongside the location; only call it a knee when
    ``knee_well_defined`` is true.

    Pure post-processing (no re-fit): needs the ``f_mech`` posterior and the
    ``mech_post_logit`` constant-data node of a standard HSGP mechanism fit (e.g.
    ``lrp-rli-mech-058``). ``n_trials`` is the mechanism predictor's item ceiling (letter
    sounds = 32) used to back-transform the logit input to an approximate count.

    For a continuous-covariate exposure (``mechanism_is_covariate`` with the HSGP curve
    on, e.g. ``lrp-rli-mech-191`` sessions -> word reading), pass ``exposure_values``
    (the per-observation raw exposure, in the same order as ``f_mech``'s rows) instead
    of ``n_trials``; the knee/half-rise/``obs_count_*`` fields are then in the
    exposure's own raw units (e.g. sessions) rather than a bounded count.
    """
    post = trace.posterior
    if curve is None:
        if "f_mech" not in post:
            raise KeyError(
                "trace has no 'f_mech' posterior — the readiness threshold needs an "
                "HSGP mechanism fit (not the linear-mechanism or phase-specific "
                "variant)."
            )
        # The HSGP ``f_mech`` carries an auto-named obs dimension (e.g.
        # ``f_mech_dim_0``), not ``obs_id``; take whichever non-sample dim it has. Its
        # rows are in the model's observation order, aligned to the
        # ``mech_post_logit`` constant-data node below.
        f_stacked = post["f_mech"].stack(sample=("chain", "draw"))
        obs_dim = next(d for d in f_stacked.dims if d != "sample")
        f = f_stacked.transpose(obs_dim, "sample").values  # (n_obs, S)
    else:
        # Caller-supplied curve on another scale — the expected-items curve
        # standardised over the fitted rows (#602). The binning, argmax, boundary and
        # curvature logic is scale-free, so it is reused verbatim; only ``scale``
        # changes, and it is recorded so no renderer can confuse the two.
        f = np.asarray(curve, dtype=float)
        if f.ndim != 2:
            raise ValueError("curve must be a (n_obs, n_draws) array")
    n_chains, n_draws = int(post.sizes["chain"]), int(post.sizes["draw"])
    if exposure_values is not None:
        # Continuous-covariate exposure: the knee lives in the exposure's own units.
        # ``exposure_values`` must be in the same observation order as the curve rows.
        result = _readiness_knee(
            f, None,
            count_values=np.asarray(exposure_values, dtype=float).reshape(-1),
            ci_prob=ci_prob, n_bins=n_bins, n_chains=n_chains, n_draws=n_draws,
        )
    else:
        ell = np.asarray(
            trace.constant_data["mech_post_logit"].values
        ).reshape(-1)  # (n_obs,)
        result = _readiness_knee(
            f, ell, n_trials=n_trials, ci_prob=ci_prob, n_bins=n_bins,
            n_chains=n_chains, n_draws=n_draws,
        )
    result["scale"] = scale
    return result
