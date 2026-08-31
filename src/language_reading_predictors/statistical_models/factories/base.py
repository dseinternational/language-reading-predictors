# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared construction helpers every family factory uses.

The pieces more than one family factory needs: the built-model wrapper, the
outcome-tier prior scales, the child random intercept, the phase-zero
broadcasts, the baseline standardiser, the bivariate LKJ residual block and
the adjusted-predictor resolver. Kept separate so a family module never has
to import a sibling (#637 stage 3).
"""

from __future__ import annotations

import inspect

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Iterable, TypeVar

import numpy as np
import pymc as pm
import pytensor.tensor as pt

if TYPE_CHECKING:
    pass


from language_reading_predictors.statistical_models import priors as _priors
from language_reading_predictors.statistical_models.fitted_payloads import (
    FittedPayload,
)
from language_reading_predictors.statistical_models.measures import (
    is_distal,
)
from language_reading_predictors.statistical_models.preprocessing import (
    LongitudinalPanel,
    PreparedData,
    WavePanel,
    standardise,
)

# Basis count for the mechanism-curve HSGP (issue #265). Fewer functions than the
# generic default (20) shrink the parameter space feeding the boundary-geometry
# funnel; at n ~ 157 with a smooth curve, ~12 is more resolution than the data
# support. Scoped to f_mech so other GP-bearing models keep the default.
_MECH_HSGP_M = 10


# ``build_hsgp_1d``'s default boundary factor, named here so a frozen design can
# reproduce the same boundary on a row subset.
_MECH_HSGP_C = 1.5


def _scalar_prior(name: str, prior_ctor) -> pt.TensorVariable:
    return prior_ctor().to_pymc(name)


def _tau_sigma_for(outcome_symbol: str, override: float | None = None) -> float:
    """Treatment-effect prior SD for a single-outcome causal term (issue #141).

    Returns ``override`` when given (prior-sensitivity fits), else the outcome
    tier default: the tighter ``TAU_SIGMA_DISTAL`` for broad standardised-transfer
    outcomes (``measures.DISTAL_OUTCOMES``) and the wider ``TAU_SIGMA_PROXIMAL``
    for the directly-taught / decoding outcomes. Applied to the randomised
    treatment effect only (ITT ``tau``, gain-factors ``beta_trt``, level-factors
    group contrast, DiD ``delta``) — not to adjusted-association group terms
    (mechanism / dose-response ``beta_G``, aligned ``beta_cohort``).
    """
    if override is not None:
        return override
    return (
        _priors.TAU_SIGMA_DISTAL
        if is_distal(outcome_symbol)
        else _priors.TAU_SIGMA_PROXIMAL
    )


def _alpha_sigma_for(outcome_symbol: str, override: float | None = None) -> float:
    """Intercept prior SD for a single-outcome ANCOVA (prior-critical-review 2026-07-07).

    Mirrors :func:`_tau_sigma_for`: returns ``override`` when given (prior-
    sensitivity fits), else the outcome tier — the tighter ``ALPHA_SIGMA_DISTAL``
    (1.0) for the broad high-denominator standardised-transfer outcomes
    (``measures.DISTAL_OUTCOMES``) and the wider ``ALPHA_SIGMA_PROXIMAL`` (1.5)
    otherwise. A no-op for proximal outcomes, so only distal-outcome fits tighten.
    Applies to the free ``alpha`` intercept of the ANCOVA families whose linear
    predictor already carries the outcome level in the ``gamma_own * logit(y_pre)``
    term (so ``alpha``'s mean is a ~0 deviation, tiered by SD — not re-anchored;
    see :func:`priors.alpha_prior`). The growth/LCSM *level* models instead anchor
    the intercept mean and do not use this.
    """
    if override is not None:
        return override
    return (
        _priors.ALPHA_SIGMA_DISTAL
        if is_distal(outcome_symbol)
        else _priors.ALPHA_SIGMA_PROXIMAL
    )


def _add_child_random_intercept(
    eta: pt.TensorVariable,
    child_idx: pt.TensorVariable,
    *,
    sigma_prior_sigma: float = 0.5,
) -> pt.TensorVariable:
    """Add a non-centred subject random intercept to ``eta`` (call inside a model).

    Creates ``sigma_child ~ HalfNormal(sigma_prior_sigma)``,
    ``u_child_raw ~ Normal(0, 1, dims="child")`` and the deterministic
    ``u_child = sigma_child * u_child_raw``, then returns ``eta + u_child[child_idx]``.
    Centralises the block previously copy-pasted across the mechanism,
    dose-response, DiD, gain-factors and level-factors factories so the random-
    intercept parameterisation cannot drift between them.
    """
    sigma_child = pm.HalfNormal("sigma_child", sigma=sigma_prior_sigma)
    u_child_raw = pm.Normal("u_child_raw", mu=0.0, sigma=1.0, dims="child")
    u_child = pm.Deterministic("u_child", sigma_child * u_child_raw, dims="child")
    return eta + u_child[child_idx]


def _broadcast_phase_zero(
    prepared: PreparedData,
    values: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    """Broadcast one phase-zero value per child across that child's rows.

    The DiD redesign uses only genuinely pre-randomisation quantities as
    precision covariates.  In the transition frame phase zero starts at t1; in
    the levels frame phase zero *is* t1.  This helper therefore extracts a
    single t1 value per subject and broadcasts it without ever substituting the
    treatment-affected t2 value used to start P2.
    """
    arr = np.asarray(values, dtype=float)
    if arr.shape != (prepared.n_obs,):
        raise ValueError(
            f"{label} has shape {arr.shape}; expected ({prepared.n_obs},)."
        )
    phase_zero = prepared.phase == 0
    if not np.any(phase_zero):
        raise ValueError(f"Cannot construct {label}: prepared data have no phase-zero rows.")

    by_subject: dict[Any, float] = {}
    for subject, value in zip(
        prepared.subject_ids[phase_zero], arr[phase_zero], strict=True
    ):
        if subject in by_subject and not np.isclose(
            by_subject[subject], value, equal_nan=True
        ):
            raise ValueError(
                f"Cannot construct {label}: subject {subject!r} has conflicting "
                "phase-zero values."
            )
        by_subject[subject] = float(value)

    missing = [s for s in np.unique(prepared.subject_ids) if s not in by_subject]
    if missing:
        raise ValueError(
            f"Cannot construct {label}: {len(missing)} subject(s) lack a phase-zero row."
        )
    return np.asarray([by_subject[s] for s in prepared.subject_ids], dtype=float)


def _broadcast_phase_zero_optional(
    prepared: PreparedData,
    values: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    """:func:`_broadcast_phase_zero`, but NaN for a child with no phase-zero value.

    The dose family's ability adjusters are broadcast *before* the outcome mask, so a
    child can legitimately be absent from phase zero — their period-1 outcome may be
    missing while later transitions are observed. A hard failure there would refuse to
    fit; substituting the child's later, treatment-affected value is exactly the defect
    the broadcast exists to remove (#587 finding 1). Returning NaN lets the caller drop
    those rows under an explicit, attributable mask instead.
    """
    arr = np.asarray(values, dtype=float)
    if arr.shape != (prepared.n_obs,):
        raise ValueError(
            f"{label} has shape {arr.shape}; expected ({prepared.n_obs},)."
        )
    phase_zero = prepared.phase == 0
    by_subject: dict[Any, float] = {}
    for subject, value in zip(
        prepared.subject_ids[phase_zero], arr[phase_zero], strict=True
    ):
        if np.isnan(value):
            continue
        if subject in by_subject and not np.isclose(by_subject[subject], value):
            raise ValueError(
                f"Cannot construct {label}: subject {subject!r} has conflicting "
                "phase-zero values."
            )
        by_subject[subject] = float(value)
    return np.asarray(
        [by_subject.get(s, np.nan) for s in prepared.subject_ids], dtype=float
    )


def _standardise_child_baseline(
    prepared: PreparedData,
    values: np.ndarray,
    *,
    label: str,
) -> tuple[np.ndarray, Any]:
    """Standardise a broadcast child baseline once per child, then rebroadcast."""
    broadcast = _broadcast_phase_zero(prepared, values, label=label)
    phase_zero = prepared.phase == 0
    baseline_z, scaler = standardise(broadcast[phase_zero])
    by_subject = dict(
        zip(prepared.subject_ids[phase_zero], baseline_z, strict=True)
    )
    return (
        np.asarray([by_subject[s] for s in prepared.subject_ids], dtype=float),
        scaler,
    )


PayloadT = TypeVar("PayloadT", bound=FittedPayload, covariant=True)


RequiredPayloadT = TypeVar("RequiredPayloadT", bound=FittedPayload)


@dataclass
class BuiltModel(Generic[PayloadT]):
    model: pm.Model
    prepared: PreparedData | WavePanel | LongitudinalPanel
    """The (possibly row-subset) prepared data that the model was built on.

    Factories may drop rows with missing post-scores or missing confounder
    values; this attribute exposes the actually-used data so the pipeline can
    align posterior indices to input rows.
    """
    payload: PayloadT
    """Typed family payload containing non-RV values realised by the factory."""

    @property
    def prior_descriptors(self) -> dict[str, "_priors.PriorDescriptor"]:
        """What each of this model's priors means, recorded as it was created.

        Carried beside the typed payload (#637 stage 2) rather than reconstructed
        from variable names downstream. Covers every variable built through a named
        constructor; a variable built by a bare ``pm.*`` call, or by the shared
        HSGP builder inside ``dse_research_utils``, is absent until it declares
        itself through :func:`priors.declare`.
        """
        return _priors.descriptors_for(self.model)

    def require_payload(
        self,
        payload_type: type[RequiredPayloadT],
        *,
        family: str,
    ) -> RequiredPayloadT:
        """Return the payload or reject a mismatched factory/family combination."""
        if not isinstance(self.payload, payload_type):
            raise TypeError(
                f"{family} requires {payload_type.__name__}, but the built model "
                f"carries {type(self.payload).__name__}"
            )
        return self.payload


def _bivariate_lkj_residual(
    name: str,
    *,
    n_outcomes: int,
    row_dim: str,
    lkj_eta: float,
    sd_sigma: float,
) -> tuple[pt.TensorVariable, pt.TensorVariable, pt.TensorVariable]:
    """Non-centred multivariate-normal offsets with an LKJ correlation, one row per
    ``row_dim`` entry and one column per outcome.

    Returns ``(u, corr, sigmas)`` where ``u`` has dims ``(row_dim, "outcome")``,
    ``corr`` is the outcome x outcome correlation matrix (dims
    ``("outcome", "outcome2")``) and ``sigmas`` the per-outcome standard deviations
    (dims ``"outcome"``). Both are registered as Deterministics under ``{name}_corr``
    / ``sigma_{name}``.

    ``pm.LKJCholeskyCov`` with ``sd_dist=HalfNormal(sd_sigma)`` bakes the per-outcome
    scales into ``chol`` (Sigma = chol @ chol.T), so there is **no** separate outer
    scale term — multiplying ``chol`` by an independent HalfNormal would double-scale
    Sigma and leave the block unidentified (the bug fixed in ``build_joint_model``).
    Contrast :func:`build_longitudinal_corr_factor_model`, which uses bare
    ``pm.LKJCorr`` precisely because a correlation-only role has no use for the sds;
    here the sds are load-bearing — the conditional-slope deterministics below are
    functions of ``rho * sigma_focal / sigma_held``, not of ``rho`` alone.

    Must be called inside a ``pm.Model`` whose coords declare ``row_dim``,
    ``"outcome"`` and ``"outcome2"``.
    """
    chol, corr, sigmas = pm.LKJCholeskyCov(
        f"{name}_chol",
        n=n_outcomes,
        eta=lkj_eta,
        sd_dist=pm.HalfNormal.dist(sd_sigma),
        compute_corr=True,
    )
    pm.Deterministic(f"{name}_corr", corr, dims=("outcome", "outcome2"))
    pm.Deterministic(f"sigma_{name}", sigmas, dims="outcome")
    z_raw = pm.Normal(f"{name}_z", mu=0.0, sigma=1.0, dims=(row_dim, "outcome"))
    # u_i = chol @ z_i  =>  rowwise U = Z @ chol.T.
    u = pm.Deterministic(f"{name}", pt.dot(z_raw, chol.T), dims=(row_dim, "outcome"))
    return u, corr, sigmas


def _resolve_adjusted_predictor(
    prepared: PreparedData, key: str, language_symbols: tuple[str, ...]
) -> tuple[str, np.ndarray, str]:
    """Map an LRP65 predictor key to ``(coef_name, standardised_vector, label)``.

    Keys: a measure symbol (``"L"``, ``"B"``) -> standardised T1 logit;
    ``"lang"`` -> the language composite; ``"age"`` -> the standardised T1 age;
    a covariate column already on ``prepared.covariates`` (``"blocks"``,
    ``"behav"``, ``"mumedupost16"``) -> that standardised covariate. Every key
    maps to coefficient ``beta_<key>``.

    Every predictor is standardised on the rows the model actually fits. The
    loader standardises covariates and age on *its* row set (every child with any
    requested outcome observed at the post wave), and :func:`build_adjusted_model`
    then drops the children missing the focal outcome's post score — so a
    loader-scale z-score is not mean-0 / SD-1 on the fitted rows and "+1 SD" would
    mean slightly different things for the skills (standardised here) and the
    covariates (2026-08-22 adjusted-family review, finding 11). Re-standardising
    here puts every key on the fitted-row scale, matching the Byrne loaders, which
    standardise on the complete-case rows they return. A covariate that is constant
    on the fitted rows (a ``_missing`` indicator whose only flagged child was
    dropped) cannot be re-scaled and keeps its loader-scale values unchanged — the
    coefficient then carries no information and the prior holds it.
    """
    from language_reading_predictors.statistical_models.measures import MEASURES
    from language_reading_predictors.statistical_models.preprocessing import (
        standardise,
    )

    def _fitted_scale(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        if not np.isfinite(sd) or sd <= 0.0:
            return arr
        z, _ = standardise(arr)
        return z

    coef = f"beta_{key}"
    if key == "lang":
        return coef, _t1_language_composite(prepared, language_symbols), (
            "Language composite (" + "+".join(language_symbols) + ", T1)"
        )
    if key == "age":
        return coef, _fitted_scale(prepared.A_months), "Age (T1)"
    if key in prepared.covariates:
        return coef, _fitted_scale(prepared.covariates[key]), f"{key} (T1)"
    if key in prepared.pre_logit:
        z, _ = standardise(prepared.pre_logit[key])
        label = MEASURES[key].label if key in MEASURES else key
        return coef, z, f"{label} (T1)"
    raise KeyError(f"Unknown LRP65 predictor key {key!r}")


def _interaction_product(term_vecs: dict[str, np.ndarray], a: str, b: str) -> np.ndarray:
    """Elementwise product of two named (already-standardised) factor terms."""
    return np.asarray(term_vecs[a], dtype=float) * np.asarray(term_vecs[b], dtype=float)


_LCF_DOMAINS: dict[str, tuple[str, ...]] = {
    "vocabulary": ("R", "E", "TR", "TE"),
    "code": ("L", "B"),
    "grammar": ("F", "T"),
}


def _rlm_group_nuisance(frame, eta):
    """Add non-interpretable group-nuisance dummies for the Byrne cohort factor.

    ``readgrp`` is observational: with three groups, two ``Normal(0, 1)`` dummy
    slopes (reference = the largest group, average readers) absorb cohort
    composition exactly as ``beta_group_nuisance`` does in the RLI concurrent
    family - flagged non-interpretable, never a group effect estimate.
    """
    codes = sorted(set(np.asarray(frame.group_code, dtype=int)))
    counts = {c: int((frame.group_code == c).sum()) for c in codes}
    reference = max(counts, key=lambda c: (counts[c], -c))
    for code in codes:
        if code == reference:
            continue
        slug = frame.group_labels[code].lower().replace(" ", "_").replace("-", "_")
        d = pm.Data(
            f"grp_{slug}",
            (frame.group_code == code).astype(float),
            dims="obs_id",
        )
        beta_g = pm.Normal(f"beta_group_nuisance_{slug}", mu=0.0, sigma=1.0)
        eta = eta + beta_g * d
    return eta


def _rlm_dispersion_kappa(dispersion_prior_sigma: float):
    """``kappa`` on the dispersion scale, as in the RLM historical factories.

    ``inv_sqrt_kappa ~ HalfNormal(dispersion_prior_sigma)`` with the Deterministic
    ``kappa = 1 / (inv_sqrt_kappa**2 + 1e-6)`` (the nugget keeps kappa finite and
    the gradient smooth as the dispersion goes to zero; at the fitted range it moves
    kappa by under 0.01%). Call inside a model context. Shared by the Byrne
    adjusted span and stacked-transition factories so the two cannot drift.
    """
    inv_sqrt_kappa = _priors.inv_sqrt_kappa_prior(
        sigma=dispersion_prior_sigma
    ).to_pymc("inv_sqrt_kappa")
    return pm.Deterministic("kappa", 1.0 / (inv_sqrt_kappa**2 + 1e-6))


def default_of(fn, param: str) -> float:
    """The default value of keyword ``param`` in factory ``fn``'s signature.

    Makes the factory the single source of truth for a prior-scale default, so a
    a settings fallback in the pipeline cannot silently drift
    from the factory it feeds (the failure Copilot caught on #209: the adjusted
    fallback was re-hardcoded and lagged the reconciled factory default). Prefer
    this over re-typing the number: if ``param`` is ever renamed the lookup raises
    ``KeyError`` loudly at fit time rather than falling back to a stale literal.
    ``test_pipeline_fallback_defaults`` guards that this stays in step.
    """
    return inspect.signature(fn).parameters[param].default


def _t1_language_composite(
    prepared: PreparedData, symbols: Iterable[str]
) -> np.ndarray:
    """Equal-weight standardised-logit language composite at T1.

    Each symbol's Haldane-logit baseline is standardised; the equal-weight mean
    is then standardised again so the composite is a unit-SD predictor. The
    pooled-framing analogue is LRP62's ``_build_route_composite``; that helper
    standardises on the *post* distribution and carries a paired baseline, which
    the T1-only between-child design does not need.
    """
    from language_reading_predictors.statistical_models.preprocessing import (
        standardise,
    )

    cols = []
    for s in symbols:
        if s not in prepared.pre_logit:
            raise KeyError(f"Language-composite symbol {s!r} not in prepared data")
        z, _ = standardise(prepared.pre_logit[s])
        cols.append(z)
    comp = np.mean(np.stack(cols, axis=1), axis=1)
    z_comp, _ = standardise(comp)
    return z_comp
