# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Registry-wide typed-settings contract (#637 stage 2).

108 registered specs across six families — 41 mechanism, 33 gain-factor, 18 DiD,
10 aligned, 3 growth and 3 pooled-level — declared their family settings as a
free-form ``ModelSpec.extra`` dict, translated at resolution time by
``from_legacy_extra``. A key misspelt there was rejected by the family's allow
list, but a key that existed and a key that mattered were indistinguishable in
``config.json``, which recorded ``model_settings: null`` and the raw dict.

Every registered model now declares its family's settings class, and ``extra`` is
empty everywhere. These tests hold that over the whole registry rather than model
by model, so a new model cannot reintroduce the legacy style.
"""

from __future__ import annotations

import dataclasses

import pytest

from language_reading_predictors.statistical_models import definitions
from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    spec_target_accept,
)
from language_reading_predictors.statistical_models.family_settings import (
    SETTINGS_CLASSES,
    registered_settings_failures,
    settings_class_for,
)
from language_reading_predictors.statistical_models.registry import discover_models


def _registered_specs() -> list[ModelSpec]:
    specs = []
    for model_id, lazy in discover_models().items():
        spec = getattr(lazy.load(), "SPEC", None)
        if spec is not None:
            specs.append(spec)
    return specs


REGISTERED = _registered_specs()


def test_the_registry_is_worth_checking():
    """Guard the sweep: an empty registry must not pass these tests silently."""
    assert len(REGISTERED) > 250


def test_every_kind_has_a_settings_class():
    assert set(SETTINGS_CLASSES) == set(definitions.KINDS)
    for kind, cls in SETTINGS_CLASSES.items():
        assert dataclasses.is_dataclass(cls), kind
        assert cls.__name__.endswith("ModelSettings"), kind


def test_an_unknown_kind_names_the_gap_rather_than_raising_bare_keyerror():
    with pytest.raises(KeyError, match="has no registered settings class"):
        settings_class_for("not_a_family")


def test_every_registered_model_declares_its_family_settings_class():
    """The whole point of the migration, over the whole registry."""
    failures = [
        failure for spec in REGISTERED for failure in registered_settings_failures(spec)
    ]
    assert failures == [], failures


def test_no_registered_model_keeps_scientific_keys_in_extra():
    """``extra`` is empty everywhere — including the sampler knob.

    18 typed specs in four families kept ``extra={"target_accept": ...}``, which
    their resolvers tolerated through a ``_GLOBAL_KEYS`` exemption. That made
    ``ModelSpec.target_accept``'s own docstring false: the legacy route was
    reachable from a typed module after all.
    """
    offenders = {spec.model_id: sorted(spec.extra) for spec in REGISTERED if spec.extra}
    assert offenders == {}, offenders


def test_the_sampler_knob_survived_the_move_to_its_own_field():
    """Moving ``target_accept`` out of ``extra`` must not lose a declaration."""
    declared = {
        spec.model_id: spec_target_accept(spec)
        for spec in REGISTERED
        if spec_target_accept(spec) is not None
    }
    assert len(declared) == 38
    assert all(0.0 < value < 1.0 for value in declared.values())
    # Every one of them now reads from the first-class field.
    assert all(
        next(s for s in REGISTERED if s.model_id == model_id).target_accept == value
        for model_id, value in declared.items()
    )


def test_a_spec_declaring_another_family_s_settings_is_rejected():
    from language_reading_predictors.statistical_models.aligned import (
        AlignedModelSettings,
    )

    spec = ModelSpec(
        model_id="lrp-test-001",
        kind="mechanism",
        title="wrong family settings",
        model_settings=AlignedModelSettings(),
    )
    failures = registered_settings_failures(spec)
    assert any("requires MechanismModelSettings" in failure for failure in failures)


def test_a_spec_declaring_the_class_rather_than_an_instance_is_rejected():
    from language_reading_predictors.statistical_models.mechanism import (
        MechanismModelSettings,
    )

    spec = ModelSpec(
        model_id="lrp-test-001",
        kind="mechanism",
        title="class not instance",
        model_settings=MechanismModelSettings,
    )
    assert any(
        "must be a settings *instance*" in failure
        for failure in registered_settings_failures(spec)
    )


def test_a_legacy_extra_declaration_is_rejected():
    spec = ModelSpec(
        model_id="lrp-test-001",
        kind="mechanism",
        title="legacy declaration",
        extra={"linear_mechanism": True},
    )
    failures = registered_settings_failures(spec)
    assert any("declares no model_settings" in failure for failure in failures)
    assert any("declares spec.extra keys" in failure for failure in failures)


def test_the_legacy_adapters_remain_for_archived_configurations():
    """Retained deliberately: a stored ``config.json`` predates its family's migration.

    They simply have no registered caller any more — which is what
    ``test_every_registered_model_declares_its_family_settings_class`` asserts.
    """
    for cls in SETTINGS_CLASSES.values():
        adapter = getattr(cls, "from_legacy_extra", None) or getattr(
            cls, "from_extra", None
        )
        assert adapter is not None, cls.__name__
        assert callable(adapter)
