# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""One description of each model family (#637 stage 4).

A ``ModelSpec.kind`` is answered in four places, and until now each kept its own
catalogue of the twenty-three families:

* which typed settings class it declares (``family_settings.SETTINGS_CLASSES``);
* which function resolves its run plan (an if-chain in ``run_metadata``, and a
  seven-entry subset in ``blending_sensitivity._PLAN_RESOLVERS``);
* which module under ``pipelines/`` owns it and what its entry points are (a
  table maintained in the boundary tests);
* which builder writes its key findings (``key_findings._KF_BUILDERS``).

Four catalogues of the same fact drift. The seven-entry subset is the clearest
case: it exists because the currency check needs a resolver per family, and adding
a family to it was a separate act of memory from adding the family.

:data:`FAMILIES` is the one description. Each catalogue is derived from it, and
the tests parameterise over it rather than over a copy — which is what #637 means
by "derive family test parameters from the descriptor instead of maintaining
parallel catalogues".

Deliberately **declarative**: names of modules and functions, resolved on demand,
so importing this module does not pull in twenty-three families' worth of PyMC.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable

from language_reading_predictors.statistical_models.definitions import KINDS

__all__ = [
    "FamilyDescriptor",
    "FAMILIES",
    "descriptor_for",
    "resolve_run_plan",
    "settings_class_for_kind",
]

SM = "language_reading_predictors.statistical_models"


@dataclass(frozen=True, slots=True)
class FamilyDescriptor:
    """Everything the shared layer needs to know about one ``ModelSpec.kind``."""

    #: The ``ModelSpec.kind`` this describes.
    kind: str
    #: Module owning the family's settings, run plan and resolver.
    settings_module: str
    #: The typed settings dataclass every registered spec of this kind declares.
    settings_class_name: str
    #: The function that turns a spec into a validated run plan.
    resolver_name: str
    #: Module under ``pipelines/`` that orchestrates a fit of this kind.
    pipeline_module: str
    #: Entry points that module exposes. Most families have one; mediation has
    #: three fit functions, and the families with a Byrne/RLM cohort port have two.
    entry_points: tuple[str, ...]
    #: Builder writing this family's key-findings box, or ``None`` for a family
    #: that has none and falls back to the generic sentence.
    key_findings_builder: str | None = None

    def settings_class(self) -> type:
        """Import and return the settings dataclass."""
        module = importlib.import_module(f"{SM}.{self.settings_module}")
        return getattr(module, self.settings_class_name)

    def resolver(self) -> Callable[..., Any]:
        """Import and return the run-plan resolver."""
        module = importlib.import_module(f"{SM}.{self.settings_module}")
        return getattr(module, self.resolver_name)

    def pipeline(self) -> Any:
        """Import and return the family's orchestration module."""
        return importlib.import_module(f"{SM}.pipelines.{self.pipeline_module}")


def _family(
    kind: str,
    settings_module: str,
    settings_class_name: str,
    resolver_name: str,
    *,
    pipeline_module: str | None = None,
    entry_points: tuple[str, ...],
    key_findings_builder: str | None = None,
) -> FamilyDescriptor:
    return FamilyDescriptor(
        kind=kind,
        settings_module=settings_module,
        settings_class_name=settings_class_name,
        resolver_name=resolver_name,
        pipeline_module=pipeline_module or kind,
        entry_points=entry_points,
        key_findings_builder=key_findings_builder,
    )


#: Every registered family, keyed by ``ModelSpec.kind``. Complete over
#: :data:`definitions.KINDS`; a new family fails the registry test until it has an
#: entry here, which is the point.
FAMILIES: dict[str, FamilyDescriptor] = {
    descriptor.kind: descriptor
    for descriptor in (
        _family("adjusted", "adjusted", "AdjustedModelSettings",
                "resolve_adjusted_run_plan",
                entry_points=("fit_adjusted", "fit_rlm_adjusted"),
                key_findings_builder="_kf_build_adjusted"),
        _family("aligned", "aligned", "AlignedModelSettings",
                "resolve_aligned_run_plan", entry_points=("fit_aligned",),
                key_findings_builder="_kf_build_aligned"),
        _family("block_exposure", "block_exposure", "BlockExposureModelSettings",
                "resolve_block_exposure_run_plan",
                entry_points=("fit_block_exposure",),
                key_findings_builder="_kf_build_block_exposure"),
        _family("concurrent", "concurrent", "ConcurrentModelSettings",
                "resolve_concurrent_run_plan", entry_points=("fit_concurrent",),
                key_findings_builder="_kf_build_concurrent"),
        _family("corr_factor", "corr_factor", "CorrFactorModelSettings",
                "resolve_corr_factor_run_plan",
                entry_points=("fit_correlated_factor", "fit_rlm_corr_factor"),
                key_findings_builder="_kf_build_corr_factor"),
        _family("did", "did", "DiDModelSettings", "resolve_did_run_plan",
                entry_points=("fit_did",), key_findings_builder="_kf_build_did"),
        _family("dose_response", "dose_response", "DoseResponseModelSettings",
                "resolve_dose_response_run_plan",
                entry_points=("fit_dose_response",),
                key_findings_builder="_kf_build_dose_response"),
        _family("gain_factors", "gain_factors", "GainFactorsModelSettings",
                "resolve_gain_factors_run_plan", entry_points=("fit_gain_factors",),
                key_findings_builder="_kf_build_gain_factors"),
        _family("growth", "growth", "GrowthModelSettings", "resolve_growth_run_plan",
                entry_points=("fit_growth",),
                key_findings_builder="_kf_build_growth"),
        _family("historical_growth", "historical_growth",
                "HistoricalGrowthModelSettings",
                "resolve_historical_growth_run_plan",
                entry_points=("fit_historical_growth",),
                key_findings_builder="_kf_build_historical_growth"),
        _family("historical_joint", "historical_joint", "HistoricalJointModelSettings",
                "resolve_historical_joint_run_plan",
                entry_points=("fit_rlm_joint_growth",),
                key_findings_builder="_kf_build_historical_joint"),
        _family("horseshoe", "horseshoe", "HorseshoeModelSettings",
                "resolve_horseshoe_run_plan",
                entry_points=("fit_horseshoe", "fit_rlm_horseshoe"),
                key_findings_builder="_kf_build_horseshoe"),
        _family("itt", "itt", "IttModelSettings", "resolve_itt_run_plan",
                entry_points=("fit_itt",), key_findings_builder="_kf_build_itt"),
        _family("joint", "joint", "JointModelSettings", "resolve_joint_run_plan",
                entry_points=("fit_joint",), key_findings_builder="_kf_build_joint"),
        _family("joint_mechanism", "joint_mechanism", "JointMechanismModelSettings",
                "resolve_joint_mechanism_run_plan",
                entry_points=("fit_joint_mechanism",),
                key_findings_builder="_kf_build_joint_mechanism"),
        _family("lcsm", "lcsm", "LcsmModelSettings", "resolve_lcsm_run_plan",
                entry_points=("fit_lcsm",), key_findings_builder="_kf_build_lcsm"),
        _family("level_factors", "level_factors", "LevelFactorsModelSettings",
                "resolve_level_factors_run_plan", entry_points=("fit_level_factors",),
                key_findings_builder="_kf_build_level_factors"),
        _family("long_corr_factor", "long_corr_factor", "LongCorrFactorModelSettings",
                "resolve_long_corr_factor_run_plan",
                entry_points=("fit_longitudinal_corr_factor",),
                key_findings_builder="_kf_build_long_corr_factor"),
        _family("mechanism", "mechanism", "MechanismModelSettings",
                "resolve_mechanism_run_plan", entry_points=("fit_mechanism",),
                key_findings_builder="_kf_build_mechanism"),
        # ``mediation`` and ``mediation_multi`` are distinct kinds sharing one
        # pipeline module: the two-mediator decomposition reuses the g-formula
        # machinery. ``prepare_mediation_data`` is a maintenance-script helper
        # rather than a fit entry point, so it is not listed.
        _family("mediation", "mediation_settings", "MediationModelSettings",
                "resolve_mediation_run_plan", pipeline_module="mediation",
                entry_points=("fit_mediation", "fit_mediation_period_stacked"),
                key_findings_builder="_kf_build_mediation"),
        _family("mediation_multi", "mediation_settings", "MediationMultiModelSettings",
                "resolve_mediation_multi_run_plan", pipeline_module="mediation",
                entry_points=("fit_mediation_multi",),
                key_findings_builder="_kf_build_mediation"),
        _family("pooled_levels", "pooled_levels", "PooledLevelsModelSettings",
                "resolve_pooled_levels_run_plan", entry_points=("fit_pooled_levels",),
                key_findings_builder="_kf_build_pooled_levels"),
        _family("survival", "survival", "SurvivalModelSettings",
                "resolve_survival_run_plan", entry_points=("fit_survival",),
                key_findings_builder="_kf_build_survival"),
    )
}


def descriptor_for(kind: str) -> FamilyDescriptor:
    """The descriptor for ``kind``, or ``KeyError`` naming the gap."""
    try:
        return FAMILIES[kind]
    except KeyError:
        raise KeyError(
            f"{kind!r} has no family descriptor; add it to family_registry.FAMILIES "
            "when the family is introduced"
        ) from None


def settings_class_for_kind(kind: str) -> type:
    """The typed settings class ``kind`` requires."""
    return descriptor_for(kind).settings_class()


def resolve_run_plan(spec: Any) -> Any:
    """Resolve ``spec``'s validated run plan through its family's own resolver."""
    return descriptor_for(spec.kind).resolver()(spec)


def missing_descriptors() -> tuple[str, ...]:
    """Registered kinds with no descriptor — empty when the map is complete."""
    return tuple(sorted(KINDS - set(FAMILIES)))
