# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Strict Boolean settings across every family (#637 stage 1).

Each ``*ModelSettings`` class used to carry its own list of Boolean fields to
type-check, and two lists had drifted from the fields their class declares:
``MechanismModelSettings`` omitted ``exposure_positive_only`` and
``PooledLevelsModelSettings`` checked only ``mechanism_is_covariate``. A value
such as ``include_group="false"`` was accepted, stayed a string and — being
truthy — turned **on** the design it names as off.

These tests are reflective on purpose. They discover the settings classes and
their Boolean fields from the package itself, so a new family, or a new flag on
an existing one, is covered the day it is declared rather than the day someone
remembers to extend a list.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import pathlib

import pytest

from language_reading_predictors.statistical_models import definitions
from language_reading_predictors.statistical_models.settings_validation import (
    boolean_fields,
    require_declared_booleans,
)

PACKAGE = pathlib.Path(definitions.__file__).parent
MODULE_ROOT = "language_reading_predictors.statistical_models"


def _settings_classes() -> list[type]:
    """Every ``*ModelSettings`` dataclass declared by a family module."""

    found: dict[str, type] = {}
    for path in sorted(PACKAGE.glob("*.py")):
        if path.stem.startswith(("lrp_", "__")):
            continue
        module = importlib.import_module(f"{MODULE_ROOT}.{path.stem}")
        for name, obj in vars(module).items():
            if (
                name.endswith("ModelSettings")
                and inspect.isclass(obj)
                and dataclasses.is_dataclass(obj)
                and obj.__module__ == module.__name__
            ):
                found[f"{path.stem}.{name}"] = obj
    return [found[key] for key in sorted(found)]


SETTINGS_CLASSES = _settings_classes()

#: Values that must never be accepted for a declared Boolean. ``0`` and ``1``
#: are included because ``isinstance(1, bool)`` is False while ``bool(1)`` is
#: True: a silently coerced integer is the same defect as a silently coerced
#: string, one keystroke further from being noticed.
NON_BOOLEANS = ("false", "true", "", 0, 1, 2, 0.0, [], {})


def _bool_field_cases() -> list[tuple[type, str, bool]]:
    return [
        (cls, name, optional)
        for cls in SETTINGS_CLASSES
        for name, optional in boolean_fields(cls)
    ]


BOOL_FIELD_CASES = _bool_field_cases()


def _adapter(cls: type):
    """The family's legacy ``spec.extra`` adapter, with its required arguments."""

    adapter = getattr(cls, "from_legacy_extra", None) or getattr(cls, "from_extra", None)
    if adapter is None:
        return None, {}
    kwargs = {}
    for name, parameter in list(inspect.signature(adapter).parameters.items())[1:]:
        if parameter.default is inspect.Parameter.empty:
            kwargs[name] = "lrp-test-001" if "id" in name else None
    return adapter, kwargs


def test_every_family_declares_a_settings_class():
    """A guard on the reflection itself: an empty sweep must not pass silently."""

    assert len(SETTINGS_CLASSES) >= len(definitions.KINDS)
    assert BOOL_FIELD_CASES


@pytest.mark.parametrize(
    ("cls", "field", "optional"),
    BOOL_FIELD_CASES,
    ids=[f"{c.__name__}.{f}" for c, f, _ in BOOL_FIELD_CASES],
)
def test_typed_construction_rejects_non_booleans(cls, field, optional):
    for value in NON_BOOLEANS:
        with pytest.raises(TypeError, match=field):
            cls(**{field: value})
    if not optional:
        with pytest.raises(TypeError, match=field):
            cls(**{field: None})


@pytest.mark.parametrize(
    ("cls", "field", "optional"),
    BOOL_FIELD_CASES,
    ids=[f"{c.__name__}.{f}" for c, f, _ in BOOL_FIELD_CASES],
)
def test_the_legacy_extra_adapter_rejects_non_booleans(cls, field, optional):
    """A declaration reaching the family through ``spec.extra`` binds identically.

    108 registered modules still declare their settings that way, so an adapter
    that coerced would leave the strict typed path checking nothing that matters.
    """
    adapter, kwargs = _adapter(cls)
    if adapter is None:
        pytest.skip(f"{cls.__name__} has no legacy extra adapter")
    for value in ("false", 1):
        with pytest.raises((TypeError, ValueError)):
            adapter({field: value}, **kwargs)


@pytest.mark.parametrize(
    ("cls", "field", "optional"),
    BOOL_FIELD_CASES,
    ids=[f"{c.__name__}.{f}" for c, f, _ in BOOL_FIELD_CASES],
)
def test_booleans_themselves_are_still_accepted(cls, field, optional):
    """The validator must reject only non-Booleans, not the flag's own values.

    Some flags are rejected by a *cross-field* rule at their non-default value —
    that is the family's own design constraint and stays a ``ValueError``. What
    must never happen is a ``TypeError`` naming the field's type.
    """
    for value in (True, False):
        try:
            cls(**{field: value})
        except ValueError:
            pass
        except TypeError as exc:  # pragma: no cover - regression guard
            pytest.fail(f"{cls.__name__}.{field}={value!r} rejected as a type: {exc}")
    if optional:
        try:
            cls(**{field: None})
        except ValueError:
            pass


def test_the_validator_reads_annotations_not_a_maintained_list():
    """The defect this replaces: a Boolean field missing from a hand list.

    ``exposure_positive_only`` was declared ``bool`` and omitted from the
    mechanism family's check for as long as it existed.
    """
    from language_reading_predictors.statistical_models.mechanism import (
        MechanismModelSettings,
    )
    from language_reading_predictors.statistical_models.pooled_levels import (
        PooledLevelsModelSettings,
    )

    mechanism = dict(boolean_fields(MechanismModelSettings))
    assert mechanism["exposure_positive_only"] is False
    pooled = dict(boolean_fields(PooledLevelsModelSettings))
    assert set(pooled) == {
        "use_wave_intercepts",
        "decompose_between_within",
        "use_subject_random_intercept",
        "include_group",
        "mechanism_is_covariate",
    }


def test_an_optional_boolean_still_accepts_none():
    """``bool | None`` is the one declared type for which ``None`` is a value."""

    @dataclasses.dataclass(frozen=True)
    class _Settings:
        flag: bool = True
        maybe: bool | None = None

        def __post_init__(self) -> None:
            require_declared_booleans(self)

    assert _Settings(maybe=None).maybe is None
    with pytest.raises(TypeError, match="flag must be a boolean, got None"):
        _Settings(flag=None)
    with pytest.raises(TypeError, match="maybe must be a boolean or None, got 1"):
        _Settings(maybe=1)


def test_a_union_that_is_not_optional_bool_is_not_treated_as_boolean():
    """``bool`` mixed with anything else belongs to the class that declared it."""

    @dataclasses.dataclass(frozen=True)
    class _Settings:
        mixed: bool | str = "auto"

    assert boolean_fields(_Settings) == ()


def test_typed_and_legacy_paths_reject_the_same_pooled_levels_declaration():
    """The reported case, end to end: ``include_group="false"`` enabled the group."""
    from language_reading_predictors.statistical_models.pooled_levels import (
        PooledLevelsModelSettings,
    )

    with pytest.raises(TypeError, match="include_group must be a boolean"):
        PooledLevelsModelSettings(include_group="false")
    with pytest.raises(TypeError, match="include_group must be a boolean"):
        PooledLevelsModelSettings.from_extra(
            {"include_group": "false"}, model_id="lrp-rli-pl-001"
        )
