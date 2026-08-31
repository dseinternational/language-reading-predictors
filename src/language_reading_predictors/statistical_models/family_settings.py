# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Which typed settings class each model family declares (#637 stage 2).

The typed-settings migration is complete: every registered ``SPEC`` declares its
family's settings dataclass through :attr:`ModelSpec.model_settings`, and none
carries a scientific key in ``ModelSpec.extra``. This module is what keeps it that
way — one map from ``ModelSpec.kind`` to the class that kind requires, and one
validator over the whole registry.

The map is deliberately narrow. It answers "which settings class does this family
take", not "how is this family fitted": the family entry points live in
``pipelines/``, the resolvers in each family module, and the ``FamilyDescriptor``
that #637 stage 4 proposes would draw those together. Keeping this to settings
means the registry check below cannot be blocked on that larger design.

``extra`` is not retired. The ``from_legacy_extra`` adapters remain, because a
stored ``config.json`` written before its family migrated records its declaration
that way and must stay readable; they simply have no registered caller.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from language_reading_predictors.statistical_models.adjusted import (
    AdjustedModelSettings,
)
from language_reading_predictors.statistical_models.aligned import AlignedModelSettings
from language_reading_predictors.statistical_models.block_exposure import (
    BlockExposureModelSettings,
)
from language_reading_predictors.statistical_models.concurrent import (
    ConcurrentModelSettings,
)
from language_reading_predictors.statistical_models.corr_factor import (
    CorrFactorModelSettings,
)
from language_reading_predictors.statistical_models.did import DiDModelSettings
from language_reading_predictors.statistical_models.dose_response import (
    DoseResponseModelSettings,
)
from language_reading_predictors.statistical_models.gain_factors import (
    GainFactorsModelSettings,
)
from language_reading_predictors.statistical_models.growth import GrowthModelSettings
from language_reading_predictors.statistical_models.historical_growth import (
    HistoricalGrowthModelSettings,
)
from language_reading_predictors.statistical_models.historical_joint import (
    HistoricalJointModelSettings,
)
from language_reading_predictors.statistical_models.horseshoe import (
    HorseshoeModelSettings,
)
from language_reading_predictors.statistical_models.itt import IttModelSettings
from language_reading_predictors.statistical_models.joint import JointModelSettings
from language_reading_predictors.statistical_models.joint_mechanism import (
    JointMechanismModelSettings,
)
from language_reading_predictors.statistical_models.lcsm import LcsmModelSettings
from language_reading_predictors.statistical_models.level_factors import (
    LevelFactorsModelSettings,
)
from language_reading_predictors.statistical_models.long_corr_factor import (
    LongCorrFactorModelSettings,
)
from language_reading_predictors.statistical_models.mechanism import (
    MechanismModelSettings,
)
from language_reading_predictors.statistical_models.mediation_settings import (
    MediationModelSettings,
    MediationMultiModelSettings,
)
from language_reading_predictors.statistical_models.pooled_levels import (
    PooledLevelsModelSettings,
)
from language_reading_predictors.statistical_models.survival import (
    SurvivalModelSettings,
)

if TYPE_CHECKING:
    from language_reading_predictors.statistical_models.context import ModelSpec

__all__ = [
    "SETTINGS_CLASSES",
    "settings_class_for",
    "registered_settings_failures",
]

#: ``ModelSpec.kind`` -> the typed settings dataclass that kind must declare.
#: Complete over :data:`definitions.KINDS`; the test below fails if a new family
#: is registered without an entry here.
SETTINGS_CLASSES: dict[str, type] = {
    "adjusted": AdjustedModelSettings,
    "aligned": AlignedModelSettings,
    "block_exposure": BlockExposureModelSettings,
    "concurrent": ConcurrentModelSettings,
    "corr_factor": CorrFactorModelSettings,
    "did": DiDModelSettings,
    "dose_response": DoseResponseModelSettings,
    "gain_factors": GainFactorsModelSettings,
    "growth": GrowthModelSettings,
    "historical_growth": HistoricalGrowthModelSettings,
    "historical_joint": HistoricalJointModelSettings,
    "horseshoe": HorseshoeModelSettings,
    "itt": IttModelSettings,
    "joint": JointModelSettings,
    "joint_mechanism": JointMechanismModelSettings,
    "lcsm": LcsmModelSettings,
    "level_factors": LevelFactorsModelSettings,
    "long_corr_factor": LongCorrFactorModelSettings,
    "mechanism": MechanismModelSettings,
    "mediation": MediationModelSettings,
    "mediation_multi": MediationMultiModelSettings,
    "pooled_levels": PooledLevelsModelSettings,
    "survival": SurvivalModelSettings,
}


def settings_class_for(kind: str) -> type:
    """The settings class ``kind`` must declare, or ``KeyError`` naming the gap."""

    try:
        return SETTINGS_CLASSES[kind]
    except KeyError:
        raise KeyError(
            f"{kind!r} has no registered settings class; add it to "
            "family_settings.SETTINGS_CLASSES when the family is introduced"
        ) from None


def registered_settings_failures(spec: ModelSpec) -> list[str]:
    """Why ``spec`` does not meet the typed-settings contract; empty if it does.

    Three requirements, each of which a registered model failed before this stage:

    1. ``model_settings`` is declared. 108 specs across six families declared
       their settings as a free-form ``extra`` dict instead.
    2. It is an instance of the family's settings class — not a dataclass from a
       neighbouring family, and not the class object itself.
    3. ``extra`` is empty. It held ``target_accept`` on 18 typed specs, which is
       a sampler knob rather than a scientific setting and now has its own
       first-class :attr:`ModelSpec.target_accept` field.
    """

    failures: list[str] = []
    try:
        expected = settings_class_for(spec.kind)
    except KeyError as exc:
        return [f"{spec.model_id}: {exc.args[0]}"]

    settings = spec.model_settings
    if settings is None:
        failures.append(
            f"{spec.model_id}: declares no model_settings; kind={spec.kind!r} "
            f"requires {expected.__name__}"
        )
    elif isinstance(settings, type) or not dataclasses.is_dataclass(settings):
        failures.append(
            f"{spec.model_id}: model_settings must be a settings *instance*, got "
            f"{settings!r}"
        )
    elif not isinstance(settings, expected):
        failures.append(
            f"{spec.model_id}: kind={spec.kind!r} requires {expected.__name__}, "
            f"got {type(settings).__name__}"
        )

    if spec.extra:
        failures.append(
            f"{spec.model_id}: declares spec.extra keys "
            f"{sorted(spec.extra)}; scientific settings belong in "
            f"{expected.__name__} and target_accept in ModelSpec.target_accept"
        )
    return failures
