# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Joint calculations and summaries."""

from __future__ import annotations

from language_reading_predictors.statistical_models.posteriors import REPORTING_CI_PROB

from collections.abc import Mapping, Sequence
import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import expit
from language_reading_predictors.statistical_models.posteriors import (
    band50,
)


def _joint_observed_row_masks(
    trace: xr.DataTree,
    *,
    n_outcomes: int,
    n_obs: int,
) -> np.ndarray:
    """Return the observed-row mask for each flattened joint outcome.

    New traces carry both flattened-cell mappings in ``constant_data``. Older
    traces do not; for those, standardise over every fitted row rather than fail.
    The fallback never mixes outcome counts. It only changes the covariate
    distribution over which an outcome's AME is averaged when that outcome has
    missing post-scores.
    """
    masks: np.ndarray = np.ones((n_outcomes, n_obs), dtype=bool)
    constant = getattr(trace, "constant_data", None)
    if constant is None:
        return masks
    if not {"y_post_cell_row", "y_post_cell_outcome"}.issubset(constant):
        return masks
    rows = np.asarray(constant["y_post_cell_row"].values, dtype=int).ravel()
    cols = np.asarray(constant["y_post_cell_outcome"].values, dtype=int).ravel()
    if rows.size != cols.size:
        raise ValueError("joint flattened-cell row and outcome maps differ in length")
    if rows.size and (rows.min() < 0 or rows.max() >= n_obs or cols.min() < 0 or cols.max() >= n_outcomes):
        raise ValueError("joint flattened-cell map contains an out-of-range index")
    masks[:] = False
    masks[cols, rows] = True
    if np.any(masks.sum(axis=1) == 0):
        raise ValueError("joint flattened-cell map leaves an outcome with no observations")
    return masks


def _joint_ame_draws(
    trace: xr.DataTree,
    outcomes: Sequence[str],
    *,
    G: np.ndarray | None = None,
    group: str = "posterior",
    row_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return logit coefficients and probability-scale AMEs by outcome and draw.

    Both returned arrays have shape ``(outcome, sample)``. For outcome ``k`` and
    draw ``s`` the average marginal effect is the mean, over rows observed for
    that outcome, of ``expit(eta0 + tau_k) - expit(eta0)``. It is therefore a
    common proportion-correct risk-difference scale even when tests have different
    item denominators. This is the multi-outcome analogue of
    :func:`_itt_ame_draws`. ``row_mask`` optionally restricts the averaging
    population and is intersected with each outcome's observed-row mask.

    **Estimand, when the LKJ residual block is on** (2026-08-22 ITT audit,
    finding 3). ``eta`` as stored already contains the fitted per-child residual
    ``u_i``, and only the treatment term is netted out, so the AME is
    *observed-child, conditional on the fitted residuals* — not a new-child
    population marginal integrating ``u_new ~ MVN(0, Sigma)``. That is the
    intended target: a dependence companion exists to be read beside a parent
    that has no random effect at all, and marginalising would introduce
    attenuation the parent does not have, moving the estimand away from the one
    being compared. Measured on the three registered companions, integrating
    fresh residuals instead changes the medians by less than 0.00012, so the
    choice is about which quantity is named rather than about the number. Fits
    without the block have no ``u_i`` and the distinction does not arise.
    """
    posterior = getattr(trace, group)
    outcome_names = [str(o) for o in outcomes]
    tau_da = posterior["tau"]
    eta_da = posterior["eta"]
    if "outcome" not in tau_da.dims or "outcome" not in eta_da.dims:
        raise ValueError("joint tau and eta must carry a labelled outcome dimension")
    available = [str(o) for o in tau_da.coords["outcome"].values]
    missing = [o for o in outcome_names if o not in available]
    if missing:
        raise KeyError(f"joint outcomes absent from posterior: {missing}")
    outcome_indices = [available.index(outcome) for outcome in outcome_names]
    tau = tau_da.sel(outcome=outcome_names).stack(sample=("chain", "draw")).transpose("outcome", "sample").values
    eta = (
        eta_da.sel(outcome=outcome_names)
        .stack(sample=("chain", "draw"))
        .transpose("outcome", "obs_id", "sample")
        .values
    )
    if G is None:
        constant = getattr(trace, "constant_data", None)
        if constant is None or "G" not in constant:
            raise ValueError("G is required when the trace has no constant_data['G']")
        G = np.asarray(constant["G"].values, dtype=float)
    else:
        G = np.asarray(G, dtype=float)
    if G.ndim != 1 or G.size != eta.shape[1]:
        raise ValueError(f"G must have one entry per fitted row ({eta.shape[1]}), got {G.shape}")
    all_masks = _joint_observed_row_masks(trace, n_outcomes=len(available), n_obs=eta.shape[1])
    masks = all_masks[outcome_indices]
    if row_mask is not None:
        selected = np.asarray(row_mask)
        if selected.ndim != 1:
            raise ValueError(f"row_mask must be 1-D, got a {selected.ndim}-D array.")
        if selected.dtype == bool:
            if selected.shape[0] != eta.shape[1]:
                raise ValueError(
                    f"boolean row_mask has {selected.shape[0]} entries but eta has "
                    f"{eta.shape[1]} observations; pass the fitted-subset mask."
                )
        elif np.issubdtype(selected.dtype, np.integer):
            if selected.size and (int(selected.min()) < 0 or int(selected.max()) >= eta.shape[1]):
                raise ValueError(f"integer row_mask has indices outside [0, {eta.shape[1]}).")
            selector = np.zeros(eta.shape[1], dtype=bool)
            selector[selected] = True
            selected = selector
        else:
            raise ValueError(f"row_mask must be a boolean mask or integer index array, got dtype {selected.dtype}.")
        masks = masks & selected[None, :]
        if np.any(masks.sum(axis=1) == 0):
            raise ValueError("row_mask leaves a joint outcome with no observations")
    ame = np.empty_like(tau, dtype=float)
    for k in range(len(outcome_names)):
        eta0 = eta[k] - tau[k][None, :] * G[:, None]
        contribution = expit(eta0 + tau[k][None, :]) - expit(eta0)
        ame[k] = contribution[masks[k]].mean(axis=0)
    return tau, ame


def tau_summary_joint(
    trace: xr.DataTree,
    outcomes: list[str],
    ci_prob: float,
    *,
    G: np.ndarray | None = None,
    row_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """Summarise each outcome on probability and logit scales.

    The headline ``ame_prob_*`` columns are average treatment risk differences
    in proportion correct, a common scale across outcome denominators. The
    ``tau_logit_*`` columns retain the conditional model coefficients as secondary
    summaries. Legacy ``tau_*`` aliases remain for existing comparison scripts
    and explicitly refer to the logit coefficient. ``row_mask`` optionally
    restricts every outcome to a common subset of fitted children, after
    intersection with that outcome's observed-score rows.
    """
    draws, ame = _joint_ame_draws(trace, outcomes, G=G, row_mask=row_mask)
    out = []
    lo_q = (1 - ci_prob) / 2
    hi_q = 1 - lo_q
    for k, s in enumerate(outcomes):
        d = draws[k]
        a = ame[k]
        a50 = band50(a)
        d50 = band50(d)
        out.append(
            {
                "outcome": s,
                "ame_prob_median": float(np.median(a)),
                "ame_prob_mean": float(np.mean(a)),
                "ame_prob_lo": float(np.quantile(a, lo_q)),
                "ame_prob_hi": float(np.quantile(a, hi_q)),
                "ame_prob_lo50": a50[0],
                "ame_prob_hi50": a50[1],
                "prob_ame_pos": float(np.mean(a > 0)),
                "tau_logit_median": float(np.median(d)),
                "tau_logit_lo": float(np.quantile(d, lo_q)),
                "tau_logit_hi": float(np.quantile(d, hi_q)),
                "tau_median": float(np.median(d)),
                "tau_lo": float(np.quantile(d, lo_q)),
                "tau_hi": float(np.quantile(d, hi_q)),
                "tau_lo50": d50[0],
                "tau_hi50": d50[1],
                "prob_pos": float(np.mean(d > 0)),
            }
        )
    return pd.DataFrame(out)


def joint_treatment_marginals(
    trace: xr.DataTree,
    *,
    outcomes: Sequence[str],
    G: np.ndarray,
    n_trials: Mapping[str, int],
    deltas: Mapping[str, float],
    ci_prob: float = REPORTING_CI_PROB,
    row_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """Items-scale treatment marginals for every outcome in a joint ITT fit.

    The joint model stores ``eta`` on ``(obs_id, outcome)`` and one ``tau`` per
    outcome.  This is the items-scale companion to :func:`tau_summary_joint`: it
    takes that function's probability-scale average marginal effect and multiplies
    by each outcome's item denominator, so the two summaries of a single fit report
    the *same* quantity on two scales.

    **Averaging population (#392):** the AME is computed by
    :func:`_joint_ame_draws`, which averages each outcome over the rows where that
    outcome is *observed* (its flattened-cell mask), not over every fitted row.
    Under outcome-specific post-score missingness the observed populations differ
    per outcome — this function reports each outcome on its own observed population,
    matching :func:`tau_summary_joint`. Passing ``row_mask`` (a boolean/int mask over
    fitted rows) restricts every outcome to a *common* subset, intersected with each
    outcome's observed rows, for a common-population cross-outcome comparison. (On the
    current registered joint datasets every outcome is complete, so the mask is all
    rows and the estimates are unchanged.)

    ``deltas`` contains the project-agreed minimally-important item difference
    where one exists.  Rows without an agreed delta retain the items-scale
    estimate but leave the ROPE fields missing.
    """
    _, ame = _joint_ame_draws(trace, outcomes, G=G, row_mask=row_mask)
    lo_q = (1 - ci_prob) / 2
    rows: list[dict[str, float | str]] = []
    for k, outcome in enumerate(outcomes):
        item_draws = ame[k] * float(n_trials[outcome])
        delta = deltas.get(outcome)
        row: dict[str, float | str] = {
            "outcome": outcome,
            "items_median": float(np.median(item_draws)),
            "items_lo": float(np.quantile(item_draws, lo_q)),
            "items_hi": float(np.quantile(item_draws, 1 - lo_q)),
            "items_lo50": float(np.quantile(item_draws, 0.25)),
            "items_hi50": float(np.quantile(item_draws, 0.75)),
            "prob_pos": float(np.mean(item_draws > 0)),
        }
        if delta is not None:
            d = float(delta)
            row.update(
                {
                    "delta_items": d,
                    "prob_benefit_ge_delta": float(np.mean(item_draws >= d)),
                    "prob_in_rope": float(np.mean(np.abs(item_draws) <= d)),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def gamma_interaction_summary(
    trace: xr.DataTree,
    *,
    ci_prob: float,
) -> dict[str, float]:
    """Summarise the linear-moderation coefficients ``gamma_int`` / ``gamma_mod``.

    Reports the posterior mean, equal-tailed central interval at coverage
    ``ci_prob`` (same convention as :func:`tau_summary_itt`), and ``P(coef > 0)``
    for each coefficient present in the trace. ``gamma_int`` is the moderation
    (>0: the standardised mechanism effect strengthens with the moderator);
    ``gamma_mod`` is the moderator main effect at the mean of the mechanism.
    """
    posterior = trace.posterior
    lo_q = (1 - ci_prob) / 2
    hi_q = 1 - lo_q
    out: dict[str, float] = {}
    for name in ("gamma_int", "gamma_mod"):
        if name not in posterior:
            continue
        d = posterior[name].stack(sample=("chain", "draw")).values
        out[f"{name}_median"] = float(np.median(d))  # median-first (#271)
        out[f"{name}_mean"] = float(np.mean(d))
        out[f"{name}_lo"] = float(np.quantile(d, lo_q))
        out[f"{name}_hi"] = float(np.quantile(d, hi_q))
        out[f"{name}_lo50"], out[f"{name}_hi50"] = band50(d)
        out[f"prob_{name}_pos"] = float(np.mean(d > 0))
    return out


def tau_contrast_matrix(
    trace: xr.DataTree,
    outcomes: list[str],
    *,
    G: np.ndarray | None = None,
    scale: str = "probability",
) -> pd.DataFrame:
    """Compute pairwise effect probabilities on the requested scale.

    ``scale='probability'`` (default) compares proportion-correct average
    marginal effects and is the reportable cross-outcome contrast. ``'logit'``
    retains the conditional-coefficient comparison as a secondary diagnostic.
    """
    logit_draws, probability_draws = _joint_ame_draws(trace, outcomes, G=G)
    if scale == "probability":
        draws = probability_draws
    elif scale == "logit":
        draws = logit_draws
    else:
        raise ValueError("scale must be 'probability' or 'logit'")
    K = draws.shape[0]
    M = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i == j:
                M[i, j] = np.nan
            else:
                M[i, j] = float(np.mean(draws[i] > draws[j]))
    return pd.DataFrame(M, index=outcomes, columns=outcomes)


def tau_difference_summary(
    trace: xr.DataTree,
    outcomes: list[str],
    pair: tuple[str, str],
    *,
    ci_prob: float,
    G: np.ndarray | None = None,
    metadata: dict[str, str] | None = None,
    row_mask: np.ndarray | None = None,
) -> dict[str, float | str]:
    """Summarise an outcome-effect difference on probability and logit scales.

    The headline contrast subtracts per-draw proportion-correct average marginal
    effects, giving a common risk-difference scale despite different test
    denominators. The logit-coefficient difference is retained as secondary.
    Both are computed per draw. For registered factorised models those draws do
    not estimate within-child residual covariance, so a paired contrast requires
    the documented dependence sensitivity.

    Human-readable semantics come from ``metadata`` rather than being inferred
    from symbols. This keeps LRPITT16's expressive-versus-receptive contrast
    distinct from LRPITT15/115's taught-versus-untaught contrasts.

    ``row_mask`` restricts the standardisation population exactly as it does in
    :func:`tau_summary_joint`, and is intersected with each outcome's observed
    rows. The influence audit needs it: per-outcome movement is not sufficient to
    determine contrast movement, because both magnitude *and* posterior covariance
    matter, so the declared contrast has to be recomputed over the retained
    children rather than reconstructed from its marginal components (2026-08-23
    joint audit, finding 9).
    """
    a, b = pair
    draws, ame = _joint_ame_draws(trace, outcomes, G=G, row_mask=row_mask)
    ia, ib = outcomes.index(a), outcomes.index(b)
    diff = draws[ia] - draws[ib]
    diff_prob = ame[ia] - ame[ib]
    lo_q = (1 - ci_prob) / 2
    hi_q = 1 - lo_q
    result: dict[str, float | str] = {
        "contrast": f"{a}_minus_{b}",
        "headline_scale": "proportion_correct_risk_difference",
        "diff_prob_median": float(np.median(diff_prob)),
        "diff_prob_mean": float(np.mean(diff_prob)),
        "diff_prob_lo": float(np.quantile(diff_prob, lo_q)),
        "diff_prob_hi": float(np.quantile(diff_prob, hi_q)),
        "diff_prob_lo50": band50(diff_prob)[0],
        "diff_prob_hi50": band50(diff_prob)[1],
        "prob_diff_pos": float(np.mean(diff_prob > 0)),
        "diff_logit_median": float(np.median(diff)),  # median-first (#271)
        "diff_logit_mean": float(np.mean(diff)),
        "diff_logit_lo": float(np.quantile(diff, lo_q)),
        "diff_logit_hi": float(np.quantile(diff, hi_q)),
        "diff_logit_lo50": band50(diff)[0],
        "diff_logit_hi50": band50(diff)[1],
        "prob_diff_logit_pos": float(np.mean(diff > 0)),
    }
    for key in (
        "contrast_kind",
        "contrast_label",
        "positive_interpretation",
        "negative_interpretation",
        "transfer_outcome",
        "transfer_interpretation",
        "dependence_note",
    ):
        if metadata and key in metadata:
            result[key] = str(metadata[key])
    return result
