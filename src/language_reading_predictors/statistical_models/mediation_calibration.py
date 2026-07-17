# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Named-confounder calibration for the mediation NIE sensitivity sweep (#324).

The generic sensitivity sweep in :mod:`mediation` asks how large a shift ``delta``
in the fitted mediator-to-outcome logit coefficient would make the NIE interval
include zero.  This module anchors that abstract shift to intervention sessions
(``IS`` / ``attend``), the treatment-induced common cause that the revised DAG can
name but that a natural-effect adjustment set cannot validly condition on.

The calibration is deliberately a *scenario*, not a repaired causal estimate.  It
uses the familiar single-omitted-variable approximation

``delta_IS ~= |beta(IS -> M_std) * beta(IS -> Y_logit)|``

after converting the dose-response slopes to the mediation fit's one-standard-
deviation session scale and standardised mediator scale.  Treating the whole fitted
``IS -> Y`` association as confounding is conservative in one specific sense: that
association may itself include a genuine ``IS -> M -> Y`` path.  The 90% range is
an endpoint envelope from separate marginal slope intervals, not a joint posterior
or credible interval.  The report says both things explicitly.

For an off-floor outcome, ``delta`` is still a shift on the Bernoulli outcome's
log-odds coefficient.  There is no constant conversion from log odds to a risk
difference.  Instead, the existing sensitivity sweep re-runs the Bernoulli
g-formula at every ``delta``; interpolation on that computed curve gives the NIE
on the off-floor risk-difference scale.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm, t

from language_reading_predictors import paths
from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.factories import MediationData
from language_reading_predictors.statistical_models.preprocessing import (
    PreparedData,
    load_and_prepare,
    logit_safe,
    standardise,
)


@dataclass(frozen=True)
class _CalibrationSources:
    mediator_dose_model: str
    outcome_dose_model: str | None


@dataclass(frozen=True)
class SlopeEstimate:
    """One dose slope and its 90% marginal interval on a documented scale."""

    point: float
    lo: float
    hi: float
    source: str
    dose_sd_sessions: float | None = None


# The direct code-route models requested by #324 plus MED-059, where the same
# L <- IS -> W structure and both fitted dose-response sources already exist.
IS_CALIBRATION_SOURCES: dict[str, _CalibrationSources] = {
    "lrp-rli-med-059": _CalibrationSources(
        mediator_dose_model="lrp-rli-dose-083",
        outcome_dose_model="lrp-rli-dose-077",
    ),
    "lrp-rli-med-086": _CalibrationSources(
        mediator_dose_model="lrp-rli-dose-083",
        # No registered off-floor N dose-response model exists.  The outcome
        # anchor is therefore the observed phase-1 adjusted log-odds association.
        outcome_dose_model=None,
    ),
    "lrp-rli-med-087": _CalibrationSources(
        mediator_dose_model="lrp-rli-dose-083",
        outcome_dose_model="lrp-rli-dose-084",
    ),
}


class _CalibrationUnavailable(RuntimeError):
    """A required fitted source artefact is absent or failed its gate."""


def supports_is_calibration(spec: ModelSpec) -> bool:
    """Whether ``spec`` has the signed-off IS calibration structure."""
    return spec.model_id in IS_CALIBRATION_SOURCES


def _design(dose_std: np.ndarray, *controls: np.ndarray) -> np.ndarray:
    """Intercept + dose (column 1) + finite adjustment columns."""
    cols = [np.ones_like(dose_std), dose_std]
    cols.extend(np.asarray(c, dtype=float) for c in controls)
    X = np.column_stack(cols)
    if not np.isfinite(X).all():
        raise ValueError("IS calibration design contains non-finite values")
    return X


def _linear_slope(y: np.ndarray, X: np.ndarray, *, source: str) -> SlopeEstimate:
    """Adjusted least-squares dose slope with a descriptive 90% t interval."""
    y = np.asarray(y, dtype=float)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    if dof <= 0:
        raise ValueError("IS calibration linear association has no residual degrees of freedom")
    sigma2 = float(resid @ resid / dof)
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = float(np.sqrt(max(cov[1, 1], 0.0)))
    crit = float(t.ppf(0.95, dof))
    point = float(beta[1])
    return SlopeEstimate(point, point - crit * se, point + crit * se, source)


def _logistic_slope(y: np.ndarray, X: np.ndarray, *, source: str) -> SlopeEstimate:
    """Adjusted Bernoulli-logit dose slope with a descriptive 90% Wald interval.

    This is an observed-data calibration association, not a new registered model
    and not a causal estimate.  Iteratively reweighted least squares keeps the
    calculation dependency-light and deterministic for the regenerate path.
    """
    y = np.asarray(y, dtype=float)
    if not set(np.unique(y)).issubset({0.0, 1.0}) or np.unique(y).size < 2:
        raise ValueError("off-floor IS calibration requires both binary outcome levels")
    beta = np.zeros(X.shape[1], dtype=float)
    converged = False
    for _ in range(100):
        eta = X @ beta
        p = expit(eta)
        w = np.clip(p * (1.0 - p), 1e-8, None)
        z = eta + (y - p) / w
        information = X.T @ (w[:, None] * X)
        updated = np.linalg.pinv(information) @ (X.T @ (w * z))
        if not np.isfinite(updated).all() or np.max(np.abs(updated)) > 30.0:
            raise ValueError("off-floor IS calibration logistic association is unstable")
        if np.max(np.abs(updated - beta)) < 1e-9:
            beta = updated
            converged = True
            break
        beta = updated
    if not converged:
        raise ValueError("off-floor IS calibration logistic association did not converge")
    p = expit(X @ beta)
    w = np.clip(p * (1.0 - p), 1e-8, None)
    cov = np.linalg.pinv(X.T @ (w[:, None] * X))
    se = float(np.sqrt(max(cov[1, 1], 0.0)))
    crit = float(norm.ppf(0.95))
    point = float(beta[1])
    return SlopeEstimate(point, point - crit * se, point + crit * se, source)


def _phase1_attendance(subject_ids: np.ndarray, data_path: Path | None) -> np.ndarray:
    """Raw t1 interval-session counts aligned to the fitted mediation children."""
    csv_path = data_path or (paths.DATA_DIR / "rli_data_long.csv")
    data = pd.read_csv(csv_path)
    t1 = data.loc[data[V.TIME] == 1, [V.SUBJECT_ID, V.ATTEND]]
    if t1[V.SUBJECT_ID].duplicated().any():
        raise ValueError("phase-1 attendance data contain duplicate child rows")
    attend = t1.set_index(V.SUBJECT_ID)[V.ATTEND].reindex(subject_ids)
    if attend.isna().any():
        missing = np.asarray(subject_ids)[attend.isna().to_numpy()]
        raise ValueError(f"phase-1 attendance is missing for children {missing.tolist()}")
    return attend.to_numpy(dtype=float)


def _observed_slopes(
    prepared: PreparedData,
    med: MediationData,
    *,
    outcome_symbol: str,
    data_path: Path | None,
) -> tuple[SlopeEstimate, SlopeEstimate, float]:
    """Observed phase-1 ``IS -> L`` and ``IS -> outcome`` associations.

    The adjustment geometry mirrors the dose-response family: arm, age and the
    corresponding outcome's own baseline.  For off-floor N the suite's floor rule
    omits the near-degenerate own baseline, matching MED-086.
    """
    if med.mediator_kind != "beta_binomial" or med.mediator_symbol != "L":
        raise ValueError("IS calibration currently requires the single L mediator")
    attend_raw = _phase1_attendance(prepared.subject_ids, data_path)
    attend_std, attend_scaler = standardise(attend_raw)
    med_logit = logit_safe(med.L2_count, med.n_trials_L)
    med_std = (med_logit - med.med_mean) / med.med_sd
    X_med = _design(attend_std, prepared.G, prepared.A_std, med.L1_logit)
    med_slope = _linear_slope(
        med_std,
        X_med,
        source="observed ITT association (arm, age and L baseline adjusted)",
    )

    if med.off_floor:
        y = (np.asarray(med.W2_count) > 0).astype(float)
        X_out = _design(attend_std, prepared.G, prepared.A_std)
        out_slope = _logistic_slope(
            y,
            X_out,
            source="observed ITT off-floor logit association (arm and age adjusted)",
        )
    else:
        y = logit_safe(med.W2_count, med.n_trials_W)
        X_out = _design(
            attend_std,
            prepared.G,
            prepared.A_std,
            prepared.pre_logit[outcome_symbol],
        )
        out_slope = _linear_slope(
            y,
            X_out,
            source="observed ITT logit association (arm, age and own baseline adjusted)",
        )
    return med_slope, out_slope, float(attend_scaler.sd)


def _dose_sd_from_data(symbol: str, data_path: Path | None) -> float:
    """Reconstruct the dose model's standardiser for pre-#324 fit artefacts."""
    prepared = load_and_prepare(
        path=data_path,
        phase_mode="all",
        outcomes=(symbol,),
        covariates=(V.ATTEND,),
    )
    return float(prepared.covariate_scalers[V.ATTEND].sd)


def _read_dose_slope(
    models_dir: Path,
    *,
    model_id: str,
    config: str,
    outcome_symbol: str,
    data_path: Path | None,
) -> SlopeEstimate:
    """Read a gate-passed period-1 dose slope and its original dose scale."""
    source_dir = models_dir / f"{model_id}-{config}"
    diag_path = source_dir / "diagnostics_summary.json"
    slope_path = source_dir / "dose_slope_summary.csv"
    if not diag_path.exists() or not slope_path.exists():
        raise _CalibrationUnavailable(
            f"{model_id}-{config} has not produced gate and dose-slope artefacts"
        )
    try:
        with diag_path.open() as f:
            diagnostics = json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        raise _CalibrationUnavailable(
            f"{model_id}-{config} diagnostics could not be read"
        ) from exc
    if not diagnostics.get("passed", False):
        raise _CalibrationUnavailable(
            f"{model_id}-{config} failed its convergence gate"
        )
    slopes = pd.read_csv(slope_path)
    row = slopes.loc[slopes["term"] == "dose_period1"]
    if len(row) != 1:
        raise _CalibrationUnavailable(
            f"{model_id}-{config} has no unique dose_period1 slope"
        )
    r = row.iloc[0]
    point_key = "median" if "median" in slopes.columns else "mean"
    values = [r[point_key], r["lo"], r["hi"]]
    if not np.isfinite(np.asarray(values, dtype=float)).all():
        raise _CalibrationUnavailable(
            f"{model_id}-{config} dose_period1 slope is non-finite"
        )
    dose_sd = (
        float(r["dose_sd_sessions"])
        if "dose_sd_sessions" in slopes.columns
        and np.isfinite(float(r["dose_sd_sessions"]))
        else _dose_sd_from_data(outcome_symbol, data_path)
    )
    return SlopeEstimate(
        point=float(r[point_key]),
        lo=float(r["lo"]),
        hi=float(r["hi"]),
        source=f"{model_id}-{config} dose_period1",
        dose_sd_sessions=dose_sd,
    )


def _rescale(est: SlopeEstimate, factor: float, *, source_suffix: str) -> SlopeEstimate:
    values = np.asarray([est.point, est.lo, est.hi], dtype=float) * factor
    lo, hi = sorted(values[1:])
    return SlopeEstimate(
        point=float(values[0]),
        lo=float(lo),
        hi=float(hi),
        source=f"{est.source}; {source_suffix}",
    )


def _abs_product_envelope(a: SlopeEstimate, b: SlopeEstimate) -> tuple[float, float]:
    products = np.asarray(
        [a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi]
    )
    raw_lo, raw_hi = float(products.min()), float(products.max())
    low = 0.0 if raw_lo <= 0.0 <= raw_hi else min(abs(raw_lo), abs(raw_hi))
    return low, max(abs(raw_lo), abs(raw_hi))


def _interp_sweep(sweep: pd.DataFrame, delta: float, column: str) -> float:
    x = sweep["delta"].to_numpy(dtype=float)
    if delta < x.min() or delta > x.max():
        return float("nan")
    return float(np.interp(delta, x, sweep[column].to_numpy(dtype=float)))


def _as_bool(value: object, *, field: str) -> bool:
    """Parse booleans safely after a CSV round trip."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised == "true":
            return True
        if normalised == "false":
            return False
    raise ValueError(f"{field} must be a boolean, got {value!r}")


def calibrate_is_scenario(
    sweep: pd.DataFrame,
    sensitivity_summary: Mapping[str, object],
    *,
    fitted_mediator_slope: SlopeEstimate,
    fitted_outcome_slope: SlopeEstimate,
    observed_mediator_slope: SlopeEstimate,
    observed_outcome_slope: SlopeEstimate,
    n_obs: int,
    off_floor: bool,
    n_trials_outcome: int,
) -> dict[str, object]:
    """Locate the fitted/observed IS scenario on an existing NIE sweep.

    All four slopes must already be on the mediation sample's scale: one standard
    deviation of phase-1 sessions; mediator slopes additionally use one standard
    deviation of the fitted mediator logit.  This pure function contains the
    comparison, response-scale interpolation and sentence generation, so the
    statistical assumptions are unit-testable without file-system fixtures.
    """
    delta_point = abs(fitted_mediator_slope.point * fitted_outcome_slope.point)
    fitted_lo, fitted_hi = _abs_product_envelope(
        fitted_mediator_slope, fitted_outcome_slope
    )
    delta_observed = abs(observed_mediator_slope.point * observed_outcome_slope.point)
    observed_lo, observed_hi = _abs_product_envelope(
        observed_mediator_slope, observed_outcome_slope
    )
    scenario_lo = min(fitted_lo, observed_lo)
    scenario_hi = max(fitted_hi, observed_hi)
    sweep_max = float(sweep["delta"].max())

    nie_median = _interp_sweep(sweep, delta_point, "nie_median")
    nie_lo = _interp_sweep(sweep, delta_point, "nie_lo")
    nie_hi = _interp_sweep(sweep, delta_point, "nie_hi")
    response_scale = "off-floor risk difference" if off_floor else "items"
    response_multiplier = 1 if off_floor else n_trials_outcome
    nie_response = nie_median * response_multiplier
    nie_response_lo = nie_lo * response_multiplier
    nie_response_hi = nie_hi * response_multiplier

    already_null = _as_bool(
        sensitivity_summary["already_null_at_zero"], field="already_null_at_zero"
    )
    robust = _as_bool(
        sensitivity_summary["robust_over_full_sweep"],
        field="robust_over_full_sweep",
    )
    tipping = float(sensitivity_summary["tipping_delta"])
    band = f"{scenario_lo:.2f} to {scenario_hi:.2f}"
    mapped = ""
    if np.isfinite(nie_response):
        mapped = (
            (
                "; the g-formula maps the point scenario to an NIE off-floor "
                f"risk difference of {nie_response:+.2f}"
            )
            if off_floor
            else f"; the g-formula maps the point scenario to an NIE of {nie_response:+.2f} items"
        )
        mapped += f" (95% {nie_response_lo:+.2f} to {nie_response_hi:+.2f})"

    if already_null:
        verdict = "already_null"
        sentence = (
            f"At n = {n_obs}, the IS point calibration is delta about "
            f"{delta_point:.2f} (90% endpoint scenario {band}){mapped}, but the "
            "95% NIE interval already includes zero before any shift, so there is "
            "no credibly non-zero indirect effect for IS to explain away."
        )
    elif np.isfinite(tipping):
        if delta_point >= tipping:
            verdict = "could_account_point"
            sentence = (
                f"At n = {n_obs}, the IS point calibration delta about "
                f"{delta_point:.2f} reaches the NIE tipping point delta* about "
                f"{tipping:.2f} (90% endpoint scenario {band}){mapped}, so "
                "IS-strength confounding could account for the estimated NIE."
            )
        elif scenario_hi >= tipping:
            verdict = "could_account_band"
            sentence = (
                f"At n = {n_obs}, the IS point calibration delta about "
                f"{delta_point:.2f} is below the NIE tipping point delta* about "
                f"{tipping:.2f}, but its wide 90% endpoint scenario ({band}) reaches "
                f"that point{mapped}, so IS-strength confounding could plausibly "
                "account for the estimated NIE."
            )
        else:
            verdict = "survives_band"
            sentence = (
                f"At n = {n_obs}, the IS calibration delta about {delta_point:.2f} "
                f"(90% endpoint scenario {band}) remains below the NIE tipping point "
                f"delta* about {tipping:.2f}{mapped}, so the NIE survives this named-"
                "confounder scenario, although both the dose links and the mediation "
                "estimate remain imprecise."
            )
    elif robust and scenario_hi <= sweep_max:
        verdict = "survives_sweep"
        sentence = (
            f"At n = {n_obs}, the IS calibration delta about {delta_point:.2f} "
            f"(90% endpoint scenario {band}) lies within the tested sweep, across "
            f"which the NIE interval never reaches zero{mapped}, so the NIE survives "
            "this named-confounder scenario, although both source slopes remain "
            "imprecise."
        )
    else:
        verdict = "outside_sweep"
        sentence = (
            f"At n = {n_obs}, the IS calibration delta about {delta_point:.2f} "
            f"(90% endpoint scenario {band}) extends beyond the tested sweep maximum "
            f"of {sweep_max:.2f}, so this named-confounder calibration is inconclusive "
            "until the sensitivity surface is extended."
        )

    return {
        "status": "ok",
        "n_calibration": int(n_obs),
        "off_floor": bool(off_floor),
        "response_scale": response_scale,
        "delta_is_point": float(delta_point),
        "delta_is_observed_point": float(delta_observed),
        "delta_is_scenario_lo": float(scenario_lo),
        "delta_is_scenario_hi": float(scenario_hi),
        "sweep_max_delta": sweep_max,
        "tipping_delta": tipping,
        "already_null_at_zero": already_null,
        "robust_over_full_sweep": robust,
        "verdict": verdict,
        "nie_response_at_is": float(nie_response),
        "nie_response_lo_at_is": float(nie_response_lo),
        "nie_response_hi_at_is": float(nie_response_hi),
        "fitted_mediator_dose_slope": fitted_mediator_slope.point,
        "fitted_mediator_dose_lo": fitted_mediator_slope.lo,
        "fitted_mediator_dose_hi": fitted_mediator_slope.hi,
        "fitted_outcome_dose_slope": fitted_outcome_slope.point,
        "fitted_outcome_dose_lo": fitted_outcome_slope.lo,
        "fitted_outcome_dose_hi": fitted_outcome_slope.hi,
        "observed_mediator_dose_slope": observed_mediator_slope.point,
        "observed_outcome_dose_slope": observed_outcome_slope.point,
        "mediator_source": fitted_mediator_slope.source,
        "outcome_source": fitted_outcome_slope.source,
        "scenario_band_method": (
            "envelope of separate 90% marginal slope endpoints plus the "
            "observed-data cross-check; not a joint credible interval"
        ),
        "scale_conversion": (
            "delta is an outcome-logit coefficient shift per 1-SD mediator; "
            "the fitted g-formula maps delta to the response scale at every sweep point"
        ),
        "sentence": sentence,
    }


def generate_is_calibration(
    spec: ModelSpec,
    *,
    config: str,
    output_dir: str | Path,
    prepared: PreparedData,
    med: MediationData,
    sweep: pd.DataFrame,
    sensitivity_summary: Mapping[str, object],
    data_path: str | Path | None = None,
) -> pd.DataFrame | None:
    """Build the auditable one-row IS calibration artefact for a supported model.

    Missing or gate-failed dose sources return a ``not_available`` row instead of
    aborting a mediation fit.  A full-suite fit discovers dose modules before
    mediation modules, so the normal reporting/refit path has the sources ready;
    the standalone regenerate script covers already-fitted mediation outputs.
    """
    if not supports_is_calibration(spec):
        return None
    source = IS_CALIBRATION_SOURCES[spec.model_id]
    path = Path(data_path) if data_path is not None else None
    out = Path(output_dir)
    base = {
        "model_id": spec.model_id,
        "outcome_symbol": spec.outcome_symbol,
        "mediator_symbol": spec.mechanism_symbol,
    }
    try:
        observed_m, observed_y, calibration_dose_sd = _observed_slopes(
            prepared,
            med,
            outcome_symbol=spec.outcome_symbol or "W",
            data_path=path,
        )
        fitted_m_raw = _read_dose_slope(
            out.parent,
            model_id=source.mediator_dose_model,
            config=config,
            outcome_symbol=spec.mechanism_symbol or "L",
            data_path=path,
        )
        mediator_factor = calibration_dose_sd / (
            float(fitted_m_raw.dose_sd_sessions) * med.med_sd
        )
        fitted_m = _rescale(
            fitted_m_raw,
            mediator_factor,
            source_suffix="rescaled to 1-SD phase-1 IS and 1-SD mediator logit",
        )
        if source.outcome_dose_model is None:
            fitted_y = observed_y
        else:
            fitted_y_raw = _read_dose_slope(
                out.parent,
                model_id=source.outcome_dose_model,
                config=config,
                outcome_symbol=spec.outcome_symbol or "W",
                data_path=path,
            )
            outcome_factor = calibration_dose_sd / float(
                fitted_y_raw.dose_sd_sessions
            )
            fitted_y = _rescale(
                fitted_y_raw,
                outcome_factor,
                source_suffix="rescaled to 1-SD phase-1 IS",
            )
        row = calibrate_is_scenario(
            sweep,
            sensitivity_summary,
            fitted_mediator_slope=fitted_m,
            fitted_outcome_slope=fitted_y,
            observed_mediator_slope=observed_m,
            observed_outcome_slope=observed_y,
            n_obs=prepared.n_obs,
            off_floor=med.off_floor,
            n_trials_outcome=med.n_trials_W,
        )
        row.update(
            {
                **base,
                "calibration_attend_sd_sessions": calibration_dose_sd,
                "mediator_logit_sd": med.med_sd,
            }
        )
    except (_CalibrationUnavailable, KeyError, OSError, ValueError) as exc:
        row = {
            **base,
            "status": "not_available",
            "reason": str(exc),
            "sentence": f"IS calibration is not available: {exc}.",
        }
    return pd.DataFrame([row])
