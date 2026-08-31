# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""One description of each family, and the catalogues derived from it (#637 stage 4).

A ``ModelSpec.kind`` used to be answered in four places — its settings class, its
run-plan resolver, its pipeline entry points and its key-findings builder — each
with its own catalogue of the twenty-three families. The clearest symptom was
``blending_sensitivity._PLAN_RESOLVERS``, a seven-entry subset of a fact the
package already knew: adding a family to it was a separate act of memory from
adding the family, and a gated family missing from it made the currency check
raise rather than run.

These tests parameterise over :data:`FAMILIES` itself, so a new family is covered
the moment it is described rather than when someone remembers to extend a list.
"""

from __future__ import annotations

import dataclasses

import pytest

from language_reading_predictors.statistical_models import definitions
from language_reading_predictors.statistical_models.family_registry import (
    FAMILIES,
    FamilyDescriptor,
    descriptor_for,
    missing_descriptors,
    resolve_run_plan,
)

KINDS = sorted(FAMILIES)


def test_every_registered_kind_is_described():
    assert missing_descriptors() == ()
    assert set(FAMILIES) == set(definitions.KINDS)


def test_an_undescribed_kind_names_the_gap():
    with pytest.raises(KeyError, match="has no family descriptor"):
        descriptor_for("not_a_family")


@pytest.mark.parametrize("kind", KINDS)
def test_each_descriptor_resolves_what_it_names(kind):
    """Names, resolved on demand — so a typo is a test failure, not a fit failure."""
    descriptor = descriptor_for(kind)
    settings = descriptor.settings_class()
    assert dataclasses.is_dataclass(settings)
    assert settings.__name__.endswith("ModelSettings")
    assert callable(descriptor.resolver())
    pipeline = descriptor.pipeline()
    for entry in descriptor.entry_points:
        assert callable(getattr(pipeline, entry)), f"{kind}: {entry}"


@pytest.mark.parametrize("kind", KINDS)
def test_each_family_declares_at_least_one_entry_point(kind):
    assert descriptor_for(kind).entry_points


def test_the_settings_catalogue_is_derived_not_restated():
    from language_reading_predictors.statistical_models.family_settings import (
        SETTINGS_CLASSES,
    )

    assert SETTINGS_CLASSES == {
        kind: descriptor.settings_class() for kind, descriptor in FAMILIES.items()
    }


def test_the_blending_currency_check_covers_every_family_not_seven():
    """The seven-entry subset is gone.

    A family gated by the blending pair policy but absent from that subset made
    ``_stale_plan_fields`` raise "has no registered run-plan resolver" — which
    fails closed, correctly, but for a bookkeeping reason rather than a scientific
    one.
    """
    from language_reading_predictors.statistical_models.blending_sensitivity import (
        _PLAN_RESOLVERS,
    )

    assert set(_PLAN_RESOLVERS) == set(FAMILIES)
    for kind, (module, function) in _PLAN_RESOLVERS.items():
        descriptor = FAMILIES[kind]
        assert (module, function) == (
            descriptor.settings_module,
            descriptor.resolver_name,
        )


def test_the_key_findings_builders_match_the_descriptors():
    """Every family that names a builder has one, and every builder is named."""
    from language_reading_predictors.statistical_models import key_findings

    described = {
        kind: descriptor.key_findings_builder
        for kind, descriptor in FAMILIES.items()
        if descriptor.key_findings_builder is not None
    }
    for kind, builder in described.items():
        assert hasattr(key_findings, builder), f"{kind}: {builder}"
        assert key_findings._KF_BUILDERS[kind] is getattr(key_findings, builder)
    assert set(key_findings._KF_BUILDERS) == set(described)


def test_the_boundary_tests_derive_their_entry_points_from_the_descriptors():
    from .test_pipeline_boundaries import FAMILY_ENTRY_POINTS

    for descriptor in FAMILIES.values():
        declared = set(FAMILY_ENTRY_POINTS[descriptor.pipeline_module])
        assert set(descriptor.entry_points) <= declared, descriptor.kind


def test_resolve_run_plan_dispatches_through_the_descriptor():
    from language_reading_predictors.statistical_models.lrp_rli_itt_010 import SPEC

    plan = resolve_run_plan(SPEC)
    assert plan.as_dict()["model_id"] == SPEC.model_id


def test_the_descriptor_is_immutable():
    """A registry a caller can mutate is not a single description of anything."""
    descriptor = descriptor_for("itt")
    assert isinstance(descriptor, FamilyDescriptor)
    with pytest.raises(dataclasses.FrozenInstanceError):
        descriptor.kind = "joint"  # type: ignore[misc]
