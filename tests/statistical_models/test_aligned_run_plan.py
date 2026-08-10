# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the typed onset-aligned settings and run plan (#394 pillar 4)."""

from __future__ import annotations

import glob
import importlib
import os

import pytest

from language_reading_predictors.statistical_models.aligned import (
    AlignedModelSettings,
    AlignedRunPlan,
    resolve_aligned_run_plan,
)
from language_reading_predictors.statistical_models.context import ModelSpec

_META_FIELDS = (
    "design",
    "estimand",
    "causal_status",
    "analysis_population",
    "missing_data_assumption",
)


def _aligned_specs() -> list[ModelSpec]:
    """Every registered onset-aligned model's SPEC."""
    root = os.path.dirname(
        importlib.import_module(
            "language_reading_predictors.statistical_models.aligned"
        ).__file__
    )
    specs: list[ModelSpec] = []
    for path in sorted(glob.glob(os.path.join(root, "lrp_rli_al_*.py"))):
        mod = importlib.import_module(
            "language_reading_predictors.statistical_models." + os.path.basename(path)[:-3]
        )
        spec = getattr(mod, "SPEC", None)
        if spec is not None and spec.kind == "aligned":
            specs.append(spec)
    return specs


# --- settings validation ------------------------------------------------------


def test_settings_reject_unknown_likelihood():
    with pytest.raises(ValueError, match="likelihood"):
        AlignedModelSettings(likelihood="poisson")


def test_settings_reject_non_bool_use_cohort():
    with pytest.raises(TypeError, match="use_cohort"):
        AlignedModelSettings(use_cohort=1)  # type: ignore[arg-type]


def test_settings_reject_empty_ability_covariate():
    with pytest.raises(TypeError, match="ability_covariate"):
        AlignedModelSettings(ability_covariate="")


def test_from_legacy_extra_rejects_unknown_key():
    with pytest.raises(ValueError, match="unknown aligned setting"):
        AlignedModelSettings.from_legacy_extra(
            {"use_dose": True, "use_doze": True},  # typo
            model_id="lrp-rli-al-999",
        )


def test_from_legacy_extra_round_trips_known_keys():
    settings = AlignedModelSettings.from_legacy_extra(
        {
            "ability_covariate": "blocks",
            "use_cohort": False,
            "use_dose": True,
            "likelihood": "bernoulli_offfloor",
        },
        model_id="lrp-rli-al-999",
    )
    assert settings.ability_covariate == "blocks"
    assert settings.use_cohort is False
    assert settings.use_dose is True
    assert settings.likelihood == "bernoulli_offfloor"


# --- resolve ------------------------------------------------------------------


def _spec(**extra) -> ModelSpec:
    return ModelSpec(
        model_id="lrp-rli-al-000",
        kind="aligned",
        title="test",
        outcome_symbol="W",
        extra=extra,
    )


def test_resolve_rejects_wrong_kind():
    spec = ModelSpec(model_id="x", kind="itt", title="t", outcome_symbol="W")
    with pytest.raises(ValueError, match="expected kind 'aligned'"):
        resolve_aligned_run_plan(spec)


def test_resolve_primary_is_per_protocol_association():
    plan = resolve_aligned_run_plan(_spec(ability_covariate="blocks"))
    assert plan.settings_source == "legacy_extra"
    assert not plan.off_floor and plan.obs_node == "y_post"
    prep = plan.prepare_kwargs()
    assert prep == {
        "outcomes": ("W",),
        "ability_covariate": "blocks",
        "include_dose": False,
    }
    assert plan.factory_kwargs()["use_cohort"] is True
    assert "per-protocol" in plan.causal_status.lower()
    assert "not an available-case modified itt estimate" in plan.estimand.lower()


def test_resolve_dose_variant_requests_include_dose():
    plan = resolve_aligned_run_plan(_spec(use_dose=True))
    assert plan.prepare_kwargs()["include_dose"] is True
    assert plan.factory_kwargs()["use_dose"] is True


def test_resolve_off_floor_sets_bernoulli_node():
    plan = resolve_aligned_run_plan(_spec(likelihood="bernoulli_offfloor"))
    assert plan.off_floor
    assert plan.obs_node == "y_offfloor"
    assert plan.factory_kwargs()["likelihood"] == "bernoulli_offfloor"


def test_typed_settings_are_accepted_and_sourced():
    spec = ModelSpec(
        model_id="lrp-rli-al-000",
        kind="aligned",
        title="test",
        outcome_symbol="W",
        model_settings=AlignedModelSettings(use_dose=True),
    )
    plan = resolve_aligned_run_plan(spec)
    assert plan.settings_source == "typed"
    assert plan.use_dose is True


def test_split_settings_between_typed_and_extra_is_rejected():
    spec = ModelSpec(
        model_id="lrp-rli-al-000",
        kind="aligned",
        title="test",
        outcome_symbol="W",
        model_settings=AlignedModelSettings(),
        extra={"use_dose": True},
    )
    with pytest.raises(ValueError, match="cannot be split"):
        resolve_aligned_run_plan(spec)


# --- registered-specification coverage (acceptance criterion) -----------------


def test_every_registered_aligned_model_resolves_with_metadata():
    """Every registered onset-aligned model resolves to a validated plan that records
    the design, estimand, causal status, analysis population and missing-data
    assumption (#394 pillar 4)."""
    specs = _aligned_specs()
    assert len(specs) >= 9, f"expected the full aligned suite, found {len(specs)}"
    for spec in specs:
        plan = resolve_aligned_run_plan(spec)
        assert isinstance(plan, AlignedRunPlan)
        recorded = plan.as_dict()
        for field in _META_FIELDS:
            assert isinstance(recorded[field], str) and recorded[field], (
                f"{spec.model_id}: {field} not recorded"
            )
        assert plan.prepare_kwargs()["outcomes"] == (spec.outcome_symbol,)
