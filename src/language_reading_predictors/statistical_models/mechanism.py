# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Family-owned data preparation and construction for the ``mechanism`` family.

Extracted from ``pipeline.fit_mechanism`` (#438) so that the *fitted* model and any
*refitted* model — a leave-one-out refit for ``reloo``, or a grouped K-fold fold —
are built from one source rather than two that can drift apart. A spliced exact
elpd value is only meaningful if the refit is the same model as the original, and a
duplicated construction path gives no way to know whether it is. ``fit_mechanism``
now calls :func:`resolve_mechanism_plan` and :func:`build_mechanism_for_plan`, and so
does :mod:`language_reading_predictors.statistical_models.loo_refit`.

This is a **behaviour-preserving relocation** in the #394 idiom: the loader-argument
derivation, the confounder filtering and the factory keyword mapping are moved
verbatim, and ``tests/statistical_models/test_mechanism_plan.py`` asserts the plan
reproduces the construction the stored fits used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from language_reading_predictors.statistical_models import factories as _factories
from language_reading_predictors.statistical_models import priors as _priors
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.measures import MEASURES
from language_reading_predictors.statistical_models.preprocessing import (
    PreparedData,
    load_and_prepare,
    split_covariates_by_wave,
)

__all__ = [
    "MechanismPlan",
    "resolve_mechanism_plan",
    "build_mechanism_for_plan",
    "mechanism_diagnostic_vars",
]


@dataclass(frozen=True)
class MechanismPlan:
    """Everything needed to construct a mechanism model, resolved from a spec.

    ``prepared`` is the full analysis frame the spec implies. ``factory_kwargs`` are
    the keyword arguments for :func:`factories.build_mechanism_model` *other than*
    the prepared data, so a refit can rebuild on a subset by passing a different
    ``PreparedData`` with identical keywords. ``confounders`` and ``adjust_for`` are
    retained because the diagnostics variable list is derived from them.
    """

    spec: ModelSpec
    prepared: PreparedData
    factory_kwargs: dict[str, Any]
    confounders: tuple[str, ...]
    adjust_for: tuple[str, ...]


def _load_covariates(spec: ModelSpec) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Resolve ``(load_covariates, require_observed)`` for a mechanism spec.

    Raw-covariate adjusters (revised-DAG confounders that are not bounded-count
    measures): hearing (hs/hs_missing), speech (deapp_c), phonological memory
    (erbto), sessions (attend). A covariate *exposure* (#311 route (b)) and a
    non-age covariate *moderator* must also be loaded, since the factory reads both
    from ``prepared.covariates`` whereas age alone is intrinsic (``prepared.A_std``).
    Neither may sit in ``adjust_for``: the factory gives them ``beta_mech`` /
    ``gamma_mod``, so a plain ``gamma`` term too would enter them twice.
    """
    adjust_for = tuple(spec.extra.get("adjust_for", ()))
    require_observed = tuple(spec.extra.get("require_observed", ()))
    load_covariates = adjust_for

    if bool(spec.extra.get("mechanism_is_covariate", False)):
        if spec.mechanism_symbol in adjust_for:
            raise ValueError(
                f"{spec.model_id}: covariate exposure {spec.mechanism_symbol!r} "
                "must not also appear in adjust_for (it would enter the linear "
                "predictor twice)."
            )
        extra_load: tuple[str, ...] = (spec.mechanism_symbol,)
        if spec.mechanism_symbol in require_observed:
            extra_load += (f"{spec.mechanism_symbol}_missing",)
        load_covariates = tuple(dict.fromkeys((*adjust_for, *extra_load)))

    moderator_symbol = spec.extra.get("moderator_symbol")
    if spec.extra.get("moderator_is_covariate") and moderator_symbol not in (None, "A"):
        if moderator_symbol in adjust_for:
            raise ValueError(
                f"{spec.model_id}: covariate moderator {moderator_symbol!r} must not "
                "also appear in adjust_for (it would enter the linear predictor twice)."
            )
        mod_load: tuple[str, ...] = (moderator_symbol,)
        # When the spec complete-cases on the moderator, load its ``_missing`` flag
        # too so the loader's ``require_observed`` filter can drop the mean-imputed
        # rows. Moderating by an average-filled effect modifier is not meaningful.
        if moderator_symbol in require_observed:
            mod_load += (f"{moderator_symbol}_missing",)
        load_covariates = tuple(dict.fromkeys((*load_covariates, *mod_load)))

    return load_covariates, require_observed


def resolve_mechanism_plan(spec: ModelSpec) -> MechanismPlan:
    """Load the analysis frame and resolve the factory keywords for ``spec``."""
    load_covariates, require_observed = _load_covariates(spec)
    pre_adj, post_adj = split_covariates_by_wave(load_covariates)
    kw: dict[str, Any] = {
        "covariates": pre_adj,
        "post_covariates": post_adj,
        "require_observed": require_observed,
    }
    # A model may restrict the prepared outcomes (e.g. LRP72 uses only L/B/N) so
    # ``drop_missing_pre`` does not discard rows for measures the model ignores.
    extra_outcomes = spec.extra.get("outcomes")
    if extra_outcomes is not None:
        prepared = load_and_prepare(phase_mode="all", outcomes=tuple(extra_outcomes), **kw)
    else:
        prepared = load_and_prepare(phase_mode="all", **kw)

    # A constant covariate (e.g. an all-zero ``_missing`` indicator on the fitted
    # rows) is dropped by the loader and receives no coefficient, so it must not be
    # built into the model nor reported as adjusted-for.
    adjust_for = tuple(c for c in spec.extra.get("adjust_for", ()) if c in prepared.covariates)
    mechanism_is_covariate = bool(spec.extra.get("mechanism_is_covariate", False))
    if mechanism_is_covariate and spec.mechanism_symbol not in prepared.covariates:
        # The drop-constant policy is fine for an adjuster but fatal for the
        # exposure itself — there is no model without it.
        raise ValueError(
            f"{spec.model_id}: covariate exposure {spec.mechanism_symbol!r} was "
            "dropped by the loader (constant on the fitted rows); cannot fit."
        )

    moderator_symbol = spec.extra.get("moderator_symbol")
    # Drop the autoregressive baseline (any ``*_pre`` token, e.g. W_pre / N_pre)
    # from the confounder list — it enters via ``adjust_baseline_symbol``.
    confounders = [s for s in spec.adjustment if not s.endswith("_pre")]
    if moderator_symbol is not None:
        # The moderator is carried by its standardised main effect + interaction in
        # the factory, so drop it from the plain confounder loop to avoid a collinear
        # duplicate main effect. The standardised term still adjusts for M.
        confounders = [s for s in confounders if s != moderator_symbol]

    factory_kwargs: dict[str, Any] = {
        "mechanism_symbol": spec.mechanism_symbol,
        "outcome_symbol": spec.outcome_symbol or "W",
        "adjust_baseline_symbol": spec.extra.get("adjust_baseline_symbol", "W"),
        "confounder_symbols": tuple(
            s for s in confounders if s in ("G", "A") or s in MEASURES
        ),
        "use_age_gp": spec.extra.get("use_age_gp", False),
        "phase_specific_mechanism": spec.extra.get("phase_specific_mechanism", False),
        "use_subject_random_intercept": spec.extra.get("use_subject_random_intercept", True),
        "moderator_symbol": moderator_symbol,
        "moderator_is_covariate": spec.extra.get("moderator_is_covariate", False),
        "include_interaction": spec.extra.get("include_interaction", True),
        "linear_mechanism": spec.extra.get("linear_mechanism", False),
        "adjust_for": adjust_for,
        "mechanism_is_covariate": mechanism_is_covariate,
        "mechanism_at_pre": spec.extra.get("mechanism_at_pre", False),
        # Thin-support HSGP reparameterisation (#430): a spec may shrink the basis
        # count and/or tighten the lengthscale prior for a mechanism curve whose
        # exposure support is too thin for the shared defaults (e.g. mech-190
        # blending). Both default to None -> the factory keeps _MECH_HSGP_M /
        # ell_prior_mech, so every other mechanism model is byte-identical.
        "mech_hsgp_m": spec.extra.get("mech_hsgp_m"),
        "mech_lengthscale_prior": (
            _priors.ell_prior_mech_tight()
            if spec.extra.get("mech_lengthscale_tight", False)
            else None
        ),
    }

    return MechanismPlan(
        spec=spec,
        prepared=prepared,
        factory_kwargs=factory_kwargs,
        confounders=tuple(confounders),
        adjust_for=adjust_for,
    )


def build_mechanism_for_plan(
    plan: MechanismPlan,
    prepared: PreparedData | None = None,
    *,
    frozen_design=None,
):
    """Build the mechanism model for ``plan``, optionally on a row subset.

    ``prepared`` defaults to the plan's full analysis frame. A refit passes a
    :func:`factories._subset` view so the construction is identical apart from the
    rows. The factory keywords are shared by reference, which is the point: a refit
    cannot silently differ in likelihood, priors or adjustment set.
    """
    return _factories.build_mechanism_model(
        plan.prepared if prepared is None else prepared,
        **plan.factory_kwargs,
        frozen_design=frozen_design,
    )


def mechanism_diagnostic_vars(plan: MechanismPlan) -> list[str]:
    """Curated diagnostic/summary variables for a mechanism fit."""
    spec = plan.spec
    names = ["alpha", "beta_G", "gamma_own", "kappa"]
    names += [f"gamma_{s}" for s in plan.confounders if s in MEASURES]
    names += [f"gamma_{c}" for c in plan.adjust_for]
    if "A" in plan.confounders and not spec.extra.get("use_age_gp", False):
        names.append("gamma_A")
    if spec.extra.get("use_subject_random_intercept", True):
        names.append("sigma_child")
    if spec.extra.get("linear_mechanism", False):
        names.append("beta_mech")
    if spec.extra.get("moderator_symbol") is not None:
        names.append("gamma_mod")
        if spec.extra.get("include_interaction", True):
            names.append("gamma_int")
    return names


def holdout_is_safe(prepared: PreparedData, idx: int) -> tuple[bool, str]:
    """Whether row ``idx`` can be held out without changing the parameter vector.

    ``factories._subset`` re-indexes children densely, so dropping the *only* row
    for a child removes an element of ``u_child_raw`` and shifts every later child's
    index. The refit posterior would then be incompatible with the full model used
    to evaluate the held-out point, and — worse — the shift would silently misalign
    child effects rather than raise. Refuse that case explicitly.
    """
    child = int(prepared.child_idx[idx])
    n_rows_for_child = int((prepared.child_idx == child).sum())
    if n_rows_for_child <= 1:
        return False, (
            f"row {idx} is the only observation for child index {child}; holding it "
            "out changes the child random-effect dimension"
        )
    return True, ""


def holdout_mask(prepared: PreparedData, idx: int) -> np.ndarray:
    """Boolean keep-mask excluding row ``idx``."""
    keep = np.ones(prepared.n_obs, dtype=bool)
    keep[idx] = False
    return keep
