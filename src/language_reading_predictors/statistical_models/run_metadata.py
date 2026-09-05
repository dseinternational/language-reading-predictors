# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The fit's own record of itself: ``config.json``, identities and trace reuse.

Resolves a family's run plan for the metadata writer, digests the fitted rows,
the executable model design and the environment, and both writes and checks the
versioned trace-reuse contract (#637 stages 1 and 3).
"""


from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from language_reading_predictors import paths as _paths
from language_reading_predictors.statistical_models.context import (
    StatisticalFitContext,
)
from language_reading_predictors.statistical_models.aligned import (
    AlignedRunPlan,
    resolve_aligned_run_plan,
)
from language_reading_predictors.statistical_models.adjusted import (
    AdjustedRunPlan,
    resolve_adjusted_run_plan,
)
from language_reading_predictors.statistical_models.block_exposure import (
    BlockExposureRunPlan,
    resolve_block_exposure_run_plan,
)
from language_reading_predictors.statistical_models.concurrent import (
    ConcurrentRunPlan,
    resolve_concurrent_run_plan,
)
from language_reading_predictors.statistical_models.corr_factor import (
    CorrFactorRunPlan,
    resolve_corr_factor_run_plan,
)
from language_reading_predictors.statistical_models.did import (
    DiDRunPlan,
    resolve_did_run_plan,
)
from language_reading_predictors.statistical_models.dose_response import (
    DoseResponseRunPlan,
    resolve_dose_response_run_plan,
)
from language_reading_predictors.statistical_models.gain_factors import (
    GainFactorsRunPlan,
    resolve_gain_factors_run_plan,
)
from language_reading_predictors.statistical_models.growth import (
    GrowthRunPlan,
    resolve_growth_run_plan,
)
from language_reading_predictors.statistical_models.historical_growth import (
    HistoricalGrowthRunPlan,
    resolve_historical_growth_run_plan,
)
from language_reading_predictors.statistical_models.historical_joint import (
    HistoricalJointRunPlan,
    resolve_historical_joint_run_plan,
)
from language_reading_predictors.statistical_models.horseshoe import (
    HorseshoeRunPlan,
    resolve_horseshoe_run_plan,
)
from language_reading_predictors.statistical_models.itt import (
    IttRunPlan,
    declared_settings_dict,
    resolve_itt_run_plan,
)
from language_reading_predictors.statistical_models.joint import (
    JointRunPlan,
    resolve_joint_run_plan,
)
from language_reading_predictors.statistical_models.joint_mechanism import (
    JointMechanismRunPlan,
    resolve_joint_mechanism_run_plan,
)
from language_reading_predictors.statistical_models.level_factors import (
    LevelFactorsRunPlan,
    resolve_level_factors_run_plan,
)
from language_reading_predictors.statistical_models.lcsm import (
    LcsmRunPlan,
    resolve_lcsm_run_plan,
)
from language_reading_predictors.statistical_models.long_corr_factor import (
    LongCorrFactorRunPlan,
    resolve_long_corr_factor_run_plan,
)
from language_reading_predictors.statistical_models.mechanism import (
    MechanismRunPlan,
    resolve_mechanism_run_plan,
    validate_mechanism_run_plan,
)
from language_reading_predictors.statistical_models.mediation_settings import (
    MediationMultiRunPlan,
    MediationRunPlan,
    resolve_mediation_multi_run_plan,
    resolve_mediation_run_plan,
)
from language_reading_predictors.statistical_models.provenance import (
    environment_lock_sha256 as _environment_lock_sha256,
)
from language_reading_predictors.statistical_models.provenance import (
    run_provenance,
    write_environment_lock,
)
from language_reading_predictors.statistical_models.survival import (
    SurvivalRunPlan,
    resolve_survival_run_plan,
)

def _json_safe(value):
    """Return a reconstructable JSON representation of model settings.

    ``ModelSpec.extra`` is intentionally free-form. Most registered settings are
    primitives, tuples or mappings, but a few families use NumPy scalars,
    dataclasses or callables. Serialising those with ``default=str`` alone loses
    structure and can make an old fit impossible to reconstruct.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_safe(item) for item in sorted(value, key=repr)]
    if callable(value):
        module = getattr(value, "__module__", "")
        name = getattr(value, "__qualname__", getattr(value, "__name__", repr(value)))
        return f"{module}.{name}" if module else name
    return str(value)


_FITTED_SUBJECT_IDENTITY_DOMAIN = "dse-lrp-fitted-subject-identity-v1"


def fitted_subject_identity(prepared: Any) -> dict[str, Any] | None:
    """Return a non-identifying fingerprint of the rows used by a primary fit.

    ``write_run_metadata`` runs after the model factory has attached its possibly
    filtered prepared frame to the context. The sequence fingerprint therefore
    identifies the primary fit's actual rows, rather than only the source file or
    the loader's pre-factory rows. It is suitable for checking that a parent and
    companion fit used the same ordered subjects without persisting raw IDs.

    Canonicalisation is deliberately explicit: preserve ``prepared.subject_ids``
    row order (including duplicates), convert each value to text with ``str``,
    encode it as UTF-8, and prefix it with an unsigned 64-bit big-endian byte
    length. SHA-256 receives a versioned UTF-8 domain separator plus NUL before
    those records. The full digest is retained to make accidental collisions
    negligible.

    This is an audit fingerprint, not encryption: it avoids publishing identifiers
    but should not be treated as anonymisation against an attacker who already has
    a small candidate set of subject IDs.
    """

    subject_ids = getattr(prepared, "subject_ids", None)
    if subject_ids is None:
        return None
    values = np.asarray(subject_ids)
    if values.ndim != 1:
        raise ValueError(
            "prepared.subject_ids must be one-dimensional to fingerprint fitted rows"
        )

    hasher = hashlib.sha256()
    hasher.update(_FITTED_SUBJECT_IDENTITY_DOMAIN.encode("utf-8"))
    hasher.update(b"\0")
    encoded_values: list[bytes] = []
    for value in values:
        if isinstance(value, np.generic):
            value = value.item()
        encoded = str(value).encode("utf-8")
        encoded_values.append(encoded)
        hasher.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        hasher.update(encoded)

    return {
        "algorithm": "sha256",
        "domain_separator": _FITTED_SUBJECT_IDENTITY_DOMAIN,
        "encoding": "str(value) UTF-8 with uint64 big-endian byte-length prefix",
        "order": "prepared.subject_ids fitted-row order; unsorted; duplicates retained",
        "n_rows": int(values.size),
        "n_unique_subjects": len(set(encoded_values)),
        "sha256": hasher.hexdigest(),
    }


def _effective_model_settings(context: StatisticalFitContext) -> dict:
    """Resolve the spec and prepared-data choices that actually reached a fit."""

    spec = context.spec
    prepared = context.prepared
    resolved = _resolved_run_plan(context)
    settings = resolved.as_dict() if resolved is not None else {}

    if spec.kind == "itt":
        plan = _itt_run_plan(context)
        settings = plan.as_dict()
        if plan.floor_rule:
            likelihood = "bernoulli_offfloor_exploratory_with_beta_binomial_secondaries"
        else:
            likelihood = plan.headline_likelihood
        settings.update(
            {
                "likelihood": likelihood,
                "floor_rule": plan.floor_rule,
                "outcomes": list(plan.outcomes),
                "baseline_terms": {
                    "use_own_baseline": plan.use_own_baseline,
                    "use_own_baseline_gp": plan.use_own_baseline_gp,
                    "cross_symbols": list(plan.cross_symbols),
                    "pre_required": _json_safe(plan.pre_required),
                },
                "age_effect": plan.age_effect,
                "use_age_gp": plan.use_age_gp,
                "use_age_linear": plan.use_age_linear,
                "use_residual_correlation": False,
            }
        )
    elif spec.kind == "joint":
        plan = _joint_run_plan(context)
        outcomes = list(plan.outcomes)
        use_cross_baselines = plan.use_cross_baselines
        use_age_gp = plan.use_age_gp
        use_age_linear = plan.use_age_linear
        settings = plan.as_dict()
        settings.update(
            {
                "likelihood": plan.likelihood,
                "floor_rule": False,
                "outcomes": outcomes,
                "baseline_terms": {
                    "use_own_baseline": True,
                    "use_cross_baselines": use_cross_baselines,
                    "cross_symbols": outcomes if use_cross_baselines else [],
                },
                "age_effect": (
                    "gp" if use_age_gp else "linear" if use_age_linear else "none"
                ),
                "use_age_gp": use_age_gp,
                "partial_pool_age_gp": plan.partial_pool_age_gp,
                "use_age_linear": use_age_linear,
                "use_residual_correlation": plan.use_residual_correlation,
            }
        )

    post_counts = getattr(prepared, "post_counts", {}) if prepared is not None else {}
    covariates = getattr(prepared, "covariates", {}) if prepared is not None else {}
    effective_adjustment = list(covariates)
    if spec.kind == "itt":
        effective_adjustment = [
            name for name in plan.adjust_for if name in covariates
        ]
    elif spec.kind == "dose_response":
        # The loaded covariate of a dose fit is its **exposure**, not an adjuster, so
        # the generic "everything in prepared.covariates" fallback recorded the
        # adjustment set as ``["attend"]`` — naming the exposure while omitting arm,
        # age and the baselines the model actually conditions on (#587 finding 10).
        # The family writes the real record under ``extra.effective_adjustment``; the
        # exposure is named here as an exposure.
        exposure = {
            settings.get("dose_covariate"),
            settings.get("dose_stage_covariate"),
        }
        effective_adjustment = [
            name for name in covariates if name not in exposure
        ]
    settings.update(
        {
            "prepared_outcomes": list(post_counts),
            "effective_adjustment": effective_adjustment,
            "prepared_covariates": list(covariates),
            "covariate_time": _json_safe(
                getattr(prepared, "covariate_time", {})
                if prepared is not None
                else {}
            ),
            "dropped_covariates": list(
                getattr(prepared, "dropped_covariates", ())
                if prepared is not None
                else ()
            ),
            "phase_mode": getattr(prepared, "phase_mode", None),
        }
    )
    return settings


def _itt_analysis_set_metadata(context: StatisticalFitContext) -> dict:
    """Return arm-specific analysis-set counts for an ITT-family fit."""

    if context.spec.kind not in {"itt", "joint"}:
        return {}
    prepared = context.prepared
    if prepared is None or not hasattr(prepared, "G") or not hasattr(prepared, "post_counts"):
        return {}

    from language_reading_predictors.statistical_models.itt_audit import (
        analysis_set_table,
    )

    if context.spec.kind == "itt":
        symbol = context.spec.outcome_symbol
        return {
            "analysis_set_by_arm": _json_safe(
                analysis_set_table(prepared, outcome_symbol=symbol).to_dict(orient="records")
            )
        }

    records = []
    outcomes = _joint_run_plan(context).outcomes
    for symbol in outcomes:
        table = analysis_set_table(prepared, outcome_symbol=symbol)
        table.insert(0, "outcome", symbol)
        records.extend(table.to_dict(orient="records"))
    return {"analysis_set_by_outcome_and_arm": _json_safe(records)}


def _itt_run_plan(context: StatisticalFitContext) -> IttRunPlan:
    """Return the plan resolved before loading, or reconstruct it for old callers."""
    resolved_plan = getattr(context, "resolved_plan", None)
    if isinstance(resolved_plan, IttRunPlan):
        return resolved_plan
    return resolve_itt_run_plan(context.spec)


def _gain_factors_run_plan(context: StatisticalFitContext) -> GainFactorsRunPlan:
    """Return the gain-factor plan resolved before loading, or reconstruct it."""
    resolved_plan = getattr(context, "resolved_plan", None)
    if isinstance(resolved_plan, GainFactorsRunPlan):
        return resolved_plan
    return resolve_gain_factors_run_plan(context.spec)


def _joint_run_plan(context: StatisticalFitContext) -> JointRunPlan:
    """Return the joint plan resolved before loading, or reconstruct it."""
    resolved_plan = getattr(context, "resolved_plan", None)
    if isinstance(resolved_plan, JointRunPlan):
        return resolved_plan
    return resolve_joint_run_plan(context.spec)


def _joint_mechanism_run_plan(
    context: StatisticalFitContext,
) -> JointMechanismRunPlan:
    """Return the joint-mechanism plan resolved before loading."""
    resolved_plan = getattr(context, "resolved_plan", None)
    if isinstance(resolved_plan, JointMechanismRunPlan):
        return resolved_plan
    return resolve_joint_mechanism_run_plan(context.spec)


def _level_factors_run_plan(context: StatisticalFitContext) -> LevelFactorsRunPlan:
    """Return the level-factor plan resolved before loading, or reconstruct it."""
    resolved_plan = getattr(context, "resolved_plan", None)
    if isinstance(resolved_plan, LevelFactorsRunPlan):
        return resolved_plan
    return resolve_level_factors_run_plan(context.spec)


def _did_run_plan(context: StatisticalFitContext) -> DiDRunPlan:
    """Return the DiD plan resolved before loading, or reconstruct it."""
    resolved_plan = getattr(context, "resolved_plan", None)
    if isinstance(resolved_plan, DiDRunPlan):
        return resolved_plan
    return resolve_did_run_plan(context.spec)


def _pooled_levels_run_plan(context: StatisticalFitContext):
    """Return the pooled-levels plan resolved before loading, or reconstruct it."""
    from language_reading_predictors.statistical_models.pooled_levels import (
        PooledLevelsRunPlan,
        resolve_pooled_levels_run_plan,
    )

    resolved_plan = getattr(context, "resolved_plan", None)
    if isinstance(resolved_plan, PooledLevelsRunPlan):
        return resolved_plan
    return resolve_pooled_levels_run_plan(context.spec)


def _concurrent_run_plan(context: StatisticalFitContext) -> ConcurrentRunPlan:
    """Return the concurrent plan resolved before loading, or reconstruct it."""
    resolved_plan = getattr(context, "resolved_plan", None)
    if isinstance(resolved_plan, ConcurrentRunPlan):
        return resolved_plan
    return resolve_concurrent_run_plan(context.spec)


def _aligned_run_plan(context: StatisticalFitContext) -> AlignedRunPlan:
    """Return the aligned plan resolved before loading, or reconstruct it."""
    resolved_plan = getattr(context, "resolved_plan", None)
    if isinstance(resolved_plan, AlignedRunPlan):
        return resolved_plan
    return resolve_aligned_run_plan(context.spec)


def _adjusted_run_plan(context: StatisticalFitContext) -> AdjustedRunPlan:
    """Return the adjusted-association plan resolved before loading."""
    resolved_plan = getattr(context, "resolved_plan", None)
    if isinstance(resolved_plan, AdjustedRunPlan):
        return resolved_plan
    return resolve_adjusted_run_plan(context.spec)


def _lcsm_run_plan(context: StatisticalFitContext) -> LcsmRunPlan:
    """Return the LCSM plan resolved before loading, or reconstruct it."""
    resolved_plan = getattr(context, "resolved_plan", None)
    if isinstance(resolved_plan, LcsmRunPlan):
        return resolved_plan
    return resolve_lcsm_run_plan(context.spec)


def _growth_run_plan(context: StatisticalFitContext) -> GrowthRunPlan:
    """Return the growth plan resolved before loading, or reconstruct it."""
    resolved_plan = getattr(context, "resolved_plan", None)
    if isinstance(resolved_plan, GrowthRunPlan):
        return resolved_plan
    return resolve_growth_run_plan(context.spec)


def _historical_growth_run_plan(
    context: StatisticalFitContext,
) -> HistoricalGrowthRunPlan | None:
    """Return the historical-growth plan resolved before loading, or reconstruct it."""
    resolved_plan = getattr(context, "resolved_plan", None)
    if isinstance(resolved_plan, HistoricalGrowthRunPlan):
        return resolved_plan
    spec = context.spec
    if (
        spec.study_id == "rli"
        and spec.family is None
        and spec.outcome_symbol is None
        and spec.model_settings is None
        and not spec.extra
    ):
        # Some shared reporting and trace-reuse callers use a minimal ModelSpec
        # solely to identify a generic fit.  That placeholder predates the typed
        # historical-growth family and contains no declaration from which a valid
        # RLM recipe can be reconstructed.  Real family fits attach their resolved
        # plan; substantive legacy declarations still take the strict path below.
        return None
    return resolve_historical_growth_run_plan(spec)


def _historical_joint_run_plan(
    context: StatisticalFitContext,
) -> HistoricalJointRunPlan:
    """Return the historical-joint plan resolved before loading, or reconstruct it."""
    resolved_plan = getattr(context, "resolved_plan", None)
    if isinstance(resolved_plan, HistoricalJointRunPlan):
        return resolved_plan
    return resolve_historical_joint_run_plan(context.spec)


def _mechanism_run_plan(context: StatisticalFitContext) -> MechanismRunPlan:
    """Return the mechanism plan resolved before loading, or reconstruct it."""
    resolved_plan = getattr(context, "resolved_plan", None)
    if isinstance(resolved_plan, MechanismRunPlan):
        return validate_mechanism_run_plan(context.spec, resolved_plan)
    return resolve_mechanism_run_plan(context.spec)


def _survival_run_plan(context: StatisticalFitContext) -> SurvivalRunPlan:
    """Return the survival plan resolved before loading, or reconstruct it."""
    resolved_plan = getattr(context, "resolved_plan", None)
    if isinstance(resolved_plan, SurvivalRunPlan):
        return resolved_plan
    return resolve_survival_run_plan(context.spec)


def _block_exposure_run_plan(
    context: StatisticalFitContext,
) -> BlockExposureRunPlan:
    """Return the block-exposure plan resolved before loading, or reconstruct it."""
    resolved_plan = getattr(context, "resolved_plan", None)
    if isinstance(resolved_plan, BlockExposureRunPlan):
        return resolved_plan
    return resolve_block_exposure_run_plan(context.spec)


def _dose_response_run_plan(
    context: StatisticalFitContext,
) -> DoseResponseRunPlan:
    """Return the dose-response plan resolved before loading, or reconstruct it."""
    resolved_plan = getattr(context, "resolved_plan", None)
    if isinstance(resolved_plan, DoseResponseRunPlan):
        return resolved_plan
    return resolve_dose_response_run_plan(context.spec)


def _horseshoe_run_plan(context: StatisticalFitContext) -> HorseshoeRunPlan:
    """Return the horseshoe plan resolved before loading, or reconstruct it."""
    resolved_plan = getattr(context, "resolved_plan", None)
    if isinstance(resolved_plan, HorseshoeRunPlan):
        return resolved_plan
    return resolve_horseshoe_run_plan(context.spec)


def _long_corr_factor_run_plan(
    context: StatisticalFitContext,
) -> LongCorrFactorRunPlan:
    """Return the longitudinal-factor plan resolved before loading."""
    resolved_plan = getattr(context, "resolved_plan", None)
    if isinstance(resolved_plan, LongCorrFactorRunPlan):
        return resolved_plan
    return resolve_long_corr_factor_run_plan(context.spec)


def _corr_factor_run_plan(context: StatisticalFitContext) -> CorrFactorRunPlan:
    """Return the correlated-factor plan resolved before loading."""
    resolved_plan = getattr(context, "resolved_plan", None)
    if isinstance(resolved_plan, CorrFactorRunPlan):
        return resolved_plan
    return resolve_corr_factor_run_plan(context.spec)


def _mediation_run_plan(
    context: StatisticalFitContext,
) -> MediationRunPlan | MediationMultiRunPlan | None:
    """Return the mediation plan resolved before loading or reconstruct it."""
    resolved_plan = getattr(context, "resolved_plan", None)
    if isinstance(resolved_plan, (MediationRunPlan, MediationMultiRunPlan)):
        return resolved_plan
    spec = context.spec
    if (
        spec.outcome_symbol is None
        and spec.mechanism_symbol is None
        and spec.model_settings is None
        and not spec.extra
    ):
        # Metadata-only audit fixtures predate typed declarations and intentionally
        # carry no model recipe. Real fits attach a plan before context creation.
        return None
    if context.spec.kind == "mediation_multi":
        return resolve_mediation_multi_run_plan(context.spec)
    return resolve_mediation_run_plan(context.spec)


def _resolved_run_plan(context: StatisticalFitContext):
    """The typed run plan for whichever families have been converted, else None.

    Families are added here one at a time by the #394 pillar-4 series; the
    branches below are the authoritative list, so this docstring deliberately
    does not enumerate them."""
    if context.spec.kind == "itt":
        return _itt_run_plan(context)
    if context.spec.kind == "joint":
        return _joint_run_plan(context)
    if context.spec.kind == "joint_mechanism":
        return _joint_mechanism_run_plan(context)
    if context.spec.kind == "gain_factors":
        return _gain_factors_run_plan(context)
    if context.spec.kind == "level_factors":
        return _level_factors_run_plan(context)
    if context.spec.kind == "did":
        return _did_run_plan(context)
    if context.spec.kind == "pooled_levels":
        return _pooled_levels_run_plan(context)
    if context.spec.kind == "concurrent":
        return _concurrent_run_plan(context)
    if context.spec.kind == "aligned":
        return _aligned_run_plan(context)
    if context.spec.kind == "adjusted":
        return _adjusted_run_plan(context)
    if context.spec.kind == "lcsm":
        return _lcsm_run_plan(context)
    if context.spec.kind == "growth":
        return _growth_run_plan(context)
    if context.spec.kind == "historical_growth":
        return _historical_growth_run_plan(context)
    if context.spec.kind == "historical_joint":
        return _historical_joint_run_plan(context)
    if context.spec.kind == "mechanism":
        return _mechanism_run_plan(context)
    if context.spec.kind == "survival":
        return _survival_run_plan(context)
    if context.spec.kind == "block_exposure":
        return _block_exposure_run_plan(context)
    if context.spec.kind == "dose_response":
        return _dose_response_run_plan(context)
    if context.spec.kind == "horseshoe":
        return _horseshoe_run_plan(context)
    if context.spec.kind == "long_corr_factor":
        return _long_corr_factor_run_plan(context)
    if context.spec.kind == "corr_factor":
        return _corr_factor_run_plan(context)
    if context.spec.kind in {"mediation", "mediation_multi"}:
        return _mediation_run_plan(context)
    return None


#: Version of the serialised reuse contract. Bumped when the *set* of bound
#: fields changes, so a stored fit written under an older contract is refused by
#: version rather than silently compared over whichever fields both happen to
#: carry.
_REUSE_CONTRACT_SCHEMA_VERSION = 2


#: ``config.json`` key holding the whole serialised contract. The contract is
#: written and compared as one value (#637 stage 1): the previous arrangement
#: computed the contract, persisted a hand-picked subset of its fields at the top
#: level and then compared the full field list, so ``model_design_identity`` — in
#: the list, never written — made every writer-to-reader round trip fail.
REUSE_CONTRACT_KEY = "reuse_contract"


#: The bound fields, for documentation and for the test that holds this list and
#: :func:`_reuse_compatibility_contract` to the same set. It is deliberately *not*
#: what the comparison iterates: that reads the stored and current contract values
#: themselves, so a field can never be compared without having been persisted.
_REUSE_CONFIG_FIELDS = (
    "model_id",
    "kind",
    "outcome_symbol",
    "mechanism_symbol",
    "adjustment",
    "spec_extra",
    "model_settings",
    "resolved_run_plan",
    "effective_model_settings",
    "config_name",
    "sampling",
    "data_sha256",
    "n_obs",
    "n_children",
    "n_phases",
    "n_waves",
    "dropped_rows",
    "dropped_by_reason",
    "fitted_subject_identity",
    "fitted_data_identity",
    # 2026-08-22 ITT audit, finding 6. A stored fit written before these existed
    # carries neither, so reuse against it is now refused by name rather than
    # silently authorised — the fail-closed reading, since those posteriors were
    # never checked this way.
    "model_design_identity",
    "environment_lock_sha256",
    "model_recipe_file",
)


def _sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fitted_data_identity(context: StatisticalFitContext) -> dict[str, Any]:
    """Strong row-and-observation identity used to authorise trace reuse."""

    from language_reading_predictors.statistical_models.subfits import (
        describe_fitted_data,
    )

    return _json_safe(asdict(describe_fitted_data(context)))


def _model_design_identity(context: StatisticalFitContext) -> dict[str, Any]:
    """Use the same computational-graph identity for primary and secondary fits."""
    from language_reading_predictors.statistical_models.model_identity import (
        model_design_identity,
    )

    return model_design_identity(getattr(context, "model", None))


def _reuse_compatibility_contract(
    context: StatisticalFitContext,
) -> dict[str, Any]:
    """Current scientific-fit contract that a reused posterior must match.

    One serialisable value. :func:`write_run_metadata` persists exactly this
    mapping under :data:`REUSE_CONTRACT_KEY` and
    :func:`require_reuse_compatibility` compares exactly this mapping against it,
    so the writer and the reader cannot disagree about which fields are bound.
    """

    spec = context.spec
    plan = _resolved_run_plan(context)
    resolved_plan = plan.as_dict() if plan is not None else None
    recipe_path = Path(context.output_dir) / "model_recipe.md"
    return {
        "schema_version": _REUSE_CONTRACT_SCHEMA_VERSION,
        "model_id": spec.model_id,
        "kind": spec.kind,
        "outcome_symbol": spec.outcome_symbol,
        "mechanism_symbol": spec.mechanism_symbol,
        "adjustment": _json_safe(spec.adjustment),
        "spec_extra": _json_safe(spec.extra),
        "model_settings": (
            _json_safe(declared_settings_dict(spec))
            if spec.kind == "itt" or spec.model_settings is not None
            else None
        ),
        "resolved_run_plan": _json_safe(resolved_plan),
        "effective_model_settings": _json_safe(_effective_model_settings(context)),
        "config_name": getattr(context.reporting, "config_name", None),
        "sampling": {
            "draws": context.sampling.draws,
            "tune": context.sampling.tune,
            "chains": context.sampling.chains,
            "target_accept": context.sampling.target_accept,
            "random_seed": context.sampling.random_seed,
        },
        "data_sha256": getattr(context.prepared, "data_sha256", None),
        "n_obs": context.prepared.n_obs if context.prepared else None,
        "n_children": context.prepared.n_children if context.prepared else None,
        "n_phases": context.prepared.n_phases if context.prepared else None,
        "n_waves": (
            getattr(context.prepared, "n_waves", None) if context.prepared else None
        ),
        "dropped_rows": context.prepared.dropped_rows if context.prepared else None,
        "dropped_by_reason": (
            dict(getattr(context.prepared, "dropped_by_reason", {}) or {})
            if context.prepared
            else None
        ),
        "fitted_subject_identity": fitted_subject_identity(context.prepared),
        "fitted_data_identity": _fitted_data_identity(context),
        "model_design_identity": _model_design_identity(context),
        # Already computed and written beside every fit; it simply was never
        # compared, so a posterior sampled under a different dependency set could
        # be reused unchallenged.
        "environment_lock_sha256": _environment_lock_sha256(),
        "model_recipe_file": recipe_path.name if recipe_path.is_file() else None,
    }


def require_reuse_compatibility(
    context: StatisticalFitContext, source_dir: str | Path
) -> None:
    """Fail unless a prior publication matches the current scientific contract."""

    source = Path(source_dir)
    config_path = source / "config.json"
    try:
        with open(config_path, encoding="utf-8") as handle:
            previous = json.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"reuse-trace mode requires prior run metadata at {config_path}"
        ) from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"reuse-trace prior run metadata is unreadable: {config_path}"
        ) from exc
    if not isinstance(previous, Mapping):
        raise ValueError("reuse-trace prior config.json is not a JSON object")

    current = _reuse_compatibility_contract(context)
    current_data = current.get("fitted_data_identity") or {}
    if not isinstance(current_data, Mapping) or not current_data.get("digest"):
        raise ValueError(
            "reuse-trace cannot verify the current fitted rows and observations"
        )

    stored = previous.get(REUSE_CONTRACT_KEY)
    if not isinstance(stored, Mapping):
        # Fail closed. A stored fit written before the contract was serialised as
        # one value carries only the historical top-level subset, so the fields it
        # never recorded could not be compared at all — the reading those
        # posteriors were never checked under.
        raise ValueError(
            "reuse-trace compatibility check failed for the prior publication: "
            f"{REUSE_CONTRACT_KEY} (the prior config.json predates the serialised "
            "reuse contract)"
        )
    if stored.get("schema_version") != current.get("schema_version"):
        raise ValueError(
            "reuse-trace compatibility check failed for the prior publication: "
            f"{REUSE_CONTRACT_KEY} schema_version "
            f"({stored.get('schema_version')!r} stored, "
            f"{current.get('schema_version')!r} current)"
        )
    # Compare the *values*, not a separately maintained field list: every key
    # either side carries is bound, so a contract field cannot be checked without
    # having been persisted, nor persisted without being checked.
    mismatched = [
        field
        for field in sorted({*stored, *current} - {"schema_version"})
        if _json_safe(stored.get(field)) != _json_safe(current.get(field))
    ]

    recipe_name = current.get("model_recipe_file")
    if recipe_name is not None:
        previous_recipe = source / str(recipe_name)
        current_recipe = Path(context.output_dir) / str(recipe_name)
        if not previous_recipe.is_file() or not current_recipe.is_file():
            mismatched.append("model_recipe_file")
        elif _sha256_path(previous_recipe) != _sha256_path(current_recipe):
            mismatched.append("model_recipe_sha256")

    trace_path = source / "trace.nc"
    recorded_trace_sha256 = previous.get("trace_sha256")
    if (
        not isinstance(recorded_trace_sha256, str)
        or not trace_path.is_file()
        or _sha256_path(trace_path) != recorded_trace_sha256
    ):
        mismatched.append("trace_sha256")
    if mismatched:
        fields = ", ".join(dict.fromkeys(mismatched))
        raise ValueError(
            "reuse-trace compatibility check failed for the prior publication: "
            + fields
        )


def write_model_recipe(context: StatisticalFitContext, *, plan=None) -> str | None:
    """Write the human-readable recipe generated from a typed run plan, if any.

    ``plan`` overrides the context's stored plan for the prose only: after data
    loading a family may drop a constant covariate and re-describe the ACTIVE
    model here, while ``config.json`` keeps the resolver's own plan so the #623
    currency check compares resolution with resolution (2026-08-26 batch).
    """
    plan = plan if plan is not None else _resolved_run_plan(context)
    if plan is None:
        return None
    os.makedirs(context.output_dir, exist_ok=True)
    path = os.path.join(context.output_dir, "model_recipe.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(plan.recipe_markdown(title=context.spec.title))
    return path


def _publication_input_contract(context: StatisticalFitContext) -> dict | None:
    """Snapshot non-RLI input validity for the fit-level release decision."""

    spec = context.spec
    if spec.study_id == "rli":
        return None

    from language_reading_predictors.statistical_models.datasets import (
        publication_input_contract,
        resolve_dataset,
    )

    try:
        _dataset, catalogue = resolve_dataset(spec.study_id)
    except KeyError as exc:
        return {
            "schema_version": 1,
            "study_id": spec.study_id,
            "publication_ready": False,
            "dataset": {},
            "measures": {},
            "blockers": [str(exc)],
        }

    candidates: list[str] = []

    def add(value: object) -> None:
        if isinstance(value, str):
            candidates.append(value)
        elif isinstance(value, Mapping):
            for nested in value.values():
                add(nested)
        elif isinstance(value, Sequence):
            for nested in value:
                add(nested)

    add(spec.outcome_symbol)
    plan = _resolved_run_plan(context)
    plan_settings = plan.as_dict() if plan is not None else {}
    for key in (
        "measure",
        "measures",
        "outcomes",
        "predictor_measures",
        "baseline_covariate",
    ):
        add(plan_settings.get(key))
    add(plan_settings.get("domains"))

    prepared = context.prepared
    for name in ("measures", "outcomes", "outcome"):
        add(getattr(prepared, name, None))
    for name in (
        "counts",
        "indicators",
        "n_trials",
        "post_counts",
        "pre_logit",
        "predictors",
        "baseline",
    ):
        values = getattr(prepared, name, None)
        if isinstance(values, Mapping):
            add(tuple(values))

    selected = tuple(
        dict.fromkeys(symbol for symbol in candidates if symbol in catalogue)
    )
    return publication_input_contract(spec.study_id, selected)


def write_run_metadata(context: StatisticalFitContext, extra: dict | None = None) -> None:
    """Persist a reconstructable ``config.json`` and basic report metrics."""
    out = context.output_dir
    os.makedirs(out, exist_ok=True)
    environment_path, environment_sha256 = write_environment_lock(out)
    spec = context.spec
    recipe_path = write_model_recipe(context)
    _plan = _resolved_run_plan(context)
    resolved_plan = _plan.as_dict() if _plan is not None else None
    reuse_contract = _reuse_compatibility_contract(context)
    trace_path = Path(out) / "trace.nc"
    cfg = {
        "model_id": spec.model_id,
        # Canonical model-ID scheme (#168 Phase 1); legacy id stays primary.
        "canonical_model_id": spec.canonical_model_id,
        "legacy_model_id": spec.legacy_model_id,
        "family_code": spec.family_code,
        "study_code": spec.study_code,
        "variant_role": spec.variant_role,
        "parent_model_id": spec.parent_model_id,
        "kind": spec.kind,
        "title": spec.title,
        "outcome_symbol": spec.outcome_symbol,
        "mechanism_symbol": spec.mechanism_symbol,
        "adjustment": spec.adjustment,
        # Dataset / estimand metadata (#165) - default to the RLI intervention
        # study for the existing models; historical/cross-study models set them.
        "study_id": spec.study_id,
        "family": spec.family,
        "design": spec.design,
        "estimand_type": spec.estimand_type,
        "causal_status": spec.causal_status,
        "dataset_ref": spec.dataset_ref,
        "audit_baseline": spec.audit_baseline,
        # Fit-time snapshot of unresolved dataset lineage, bounded-count
        # denominators and instrument identities.  The release evaluator reads this
        # stored contract rather than silently inheriting later catalogue changes.
        "publication_input_contract": _publication_input_contract(context),
        # Preserve both what the module requested and what preprocessing/factory
        # resolution actually used. This is deliberately separate from ``extra``
        # below, which contains post-fit summaries supplied by the pipeline.
        "spec_extra": _json_safe(spec.extra),
        "model_settings": (
            _json_safe(declared_settings_dict(spec))
            if spec.kind == "itt" or spec.model_settings is not None
            else None
        ),
        "resolved_run_plan": _json_safe(resolved_plan),
        "model_recipe_file": os.path.basename(recipe_path) if recipe_path else None,
        "effective_model_settings": _effective_model_settings(context),
        "n_obs": context.prepared.n_obs if context.prepared else None,
        "n_children": context.prepared.n_children if context.prepared else None,
        "n_phases": context.prepared.n_phases if context.prepared else None,
        "n_waves": getattr(context.prepared, "n_waves", None) if context.prepared else None,
        "dropped_rows": context.prepared.dropped_rows if context.prepared else None,
        "dropped_by_reason": (
            dict(getattr(context.prepared, "dropped_by_reason", {}) or {})
            if context.prepared
            else None
        ),
        # Privacy-preserving fitted-row identity. Unlike ``data_sha256`` and row
        # counts, this lets companion fits prove that their primary analysis rows
        # match without writing raw subject identifiers to the artefact bundle.
        "fitted_subject_identity": fitted_subject_identity(context.prepared),
        "fitted_data_identity": reuse_contract["fitted_data_identity"],
        # The whole trace-reuse contract, serialised once as the value
        # ``require_reuse_compatibility`` compares (#637 stage 1). The top-level
        # fields above are the published consumer surface — the release evaluator
        # and the blending-pair gates read ``fitted_data_identity`` from there —
        # and are deliberately left alone; this block is what binds reuse.
        REUSE_CONTRACT_KEY: _json_safe(reuse_contract),
        "ci_prob": context.reporting.ci_prob,
        # Persist the named sampling preset independently of its numeric settings.
        # ``dev`` and ``test`` can occasionally clear the numerical convergence
        # thresholds by chance, but they remain diagnostic presets; the publication
        # gate needs the name to distinguish them from ``rep-lite`` / ``reporting``.
        # ``getattr`` preserves lightweight historical test/sweep contexts that predate
        # ``ReportingConfiguration.config_name``; stored legacy fits are resolved from
        # their ``<model-id>-<preset>`` directory suffix by the release evaluator.
        "config_name": getattr(context.reporting, "config_name", None),
        "sampling": {
            "draws": context.sampling.draws,
            "tune": context.sampling.tune,
            "chains": context.sampling.chains,
            "target_accept": context.sampling.target_accept,
            "random_seed": context.sampling.random_seed,
        },
        "output_root": str(_paths.output_root()),
        "data_path": getattr(context.prepared, "data_path", None),
        "data_sha256": getattr(context.prepared, "data_sha256", None),
        "trace_sha256": _sha256_path(trace_path) if trace_path.is_file() else None,
        "provenance": run_provenance(),
        "environment_lock_file": os.path.basename(environment_path),
        "environment_lock_sha256": environment_sha256,
        "extra": _json_safe(extra or {}),
        **_itt_analysis_set_metadata(context),
    }
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def write_loo_summary(context: StatisticalFitContext) -> None:
    if context.loo is None:
        return
    out = context.output_dir
    path = os.path.join(out, "loo.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(context.loo))
