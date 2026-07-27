# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the typed latent growth-curve settings and run plan (#394 pillar 4)."""

from __future__ import annotations

import glob
import importlib
import os

import pytest

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.growth import (
    GrowthModelSettings,
    GrowthRunPlan,
    resolve_growth_run_plan,
)

_META_FIELDS = (
    "design",
    "estimand",
    "causal_status",
    "analysis_population",
    "missing_data_assumption",
)


def _growth_specs() -> list[ModelSpec]:
    """Every registered latent growth-curve model's SPEC."""
    root = os.path.dirname(
        importlib.import_module(
            "language_reading_predictors.statistical_models.growth"
        ).__file__
    )
    specs: list[ModelSpec] = []
    for path in sorted(glob.glob(os.path.join(root, "lrp_rli_gc_*.py"))):
        mod = importlib.import_module(
            "language_reading_predictors.statistical_models." + os.path.basename(path)[:-3]
        )
        spec = getattr(mod, "SPEC", None)
        if spec is not None and spec.kind == "growth":
            specs.append(spec)
    return specs


# --- settings validation ------------------------------------------------------


def test_settings_reject_string_outcomes():
    with pytest.raises(TypeError, match="outcomes"):
        GrowthModelSettings(outcomes="W")  # type: ignore[arg-type]


def test_settings_reject_empty_outcomes():
    with pytest.raises(ValueError, match="at least one measure"):
        GrowthModelSettings(outcomes=())


def test_settings_reject_empty_baseline_covariate():
    with pytest.raises(TypeError, match="baseline_covariate"):
        GrowthModelSettings(baseline_covariate="")


def test_settings_reject_non_bool_use_shared_factor():
    with pytest.raises(TypeError, match="use_shared_factor"):
        GrowthModelSettings(use_shared_factor=1)  # type: ignore[arg-type]


def test_from_legacy_extra_rejects_unknown_key():
    with pytest.raises(ValueError, match="unknown growth setting"):
        GrowthModelSettings.from_legacy_extra(
            {"use_shared_factor": True, "use_shared_factr": False},  # typo
            model_id="lrp-rli-gc-999",
        )


def test_from_legacy_extra_round_trips_known_keys():
    settings = GrowthModelSettings.from_legacy_extra(
        {
            "outcomes": ("W", "L"),
            "baseline_covariate": "blocks2",
            "use_shared_factor": True,
            "age_ability_interaction": True,
        },
        model_id="lrp-rli-gc-999",
    )
    assert settings.outcomes == ("W", "L")
    assert settings.baseline_covariate == "blocks2"
    assert settings.use_shared_factor is True
    assert settings.age_ability_interaction is True


# --- resolve ------------------------------------------------------------------


def _spec(**extra) -> ModelSpec:
    # Growth is a multi-outcome family with no single outcome_symbol.
    return ModelSpec(model_id="lrp-rli-gc-000", kind="growth", title="test", extra=extra)


def test_resolve_rejects_wrong_kind():
    spec = ModelSpec(model_id="x", kind="itt", title="t", outcome_symbol="W")
    with pytest.raises(ValueError, match="expected kind 'growth'"):
        resolve_growth_run_plan(spec)


def test_resolve_defaults_five_measures_on_blocks():
    plan = resolve_growth_run_plan(_spec())
    assert plan.settings_source == "legacy_extra"
    prep = plan.prepare_kwargs()
    assert prep["outcomes"] == ("R", "E", "T", "W", "L")
    assert prep["baseline_covariates"] == ("blocks",)
    fac = plan.factory_kwargs()
    assert fac == {
        "baseline_covariate": "blocks",
        "use_shared_factor": False,
        "age_ability_interaction": False,
    }
    assert "associational" in plan.causal_status.lower()
    assert "growth rate" in plan.estimand.lower()


def test_resolve_keeps_shared_factor_and_interaction():
    plan = resolve_growth_run_plan(
        _spec(use_shared_factor=True, age_ability_interaction=True)
    )
    assert plan.factory_kwargs()["use_shared_factor"] is True
    assert plan.factory_kwargs()["age_ability_interaction"] is True


def test_typed_settings_are_accepted_and_sourced():
    spec = ModelSpec(
        model_id="lrp-rli-gc-000",
        kind="growth",
        title="test",
        model_settings=GrowthModelSettings(use_shared_factor=True),
    )
    plan = resolve_growth_run_plan(spec)
    assert plan.settings_source == "typed"
    assert plan.use_shared_factor is True


def test_split_settings_between_typed_and_extra_is_rejected():
    spec = ModelSpec(
        model_id="lrp-rli-gc-000",
        kind="growth",
        title="test",
        model_settings=GrowthModelSettings(),
        extra={"use_shared_factor": True},
    )
    with pytest.raises(ValueError, match="cannot be split"):
        resolve_growth_run_plan(spec)


# --- registered-specification coverage (acceptance criterion) -----------------


def test_every_registered_growth_model_resolves_with_metadata():
    """Every registered latent growth-curve model resolves to a validated plan that
    records the design, estimand, causal status, analysis population and missing-data
    assumption (#394 pillar 4)."""
    specs = _growth_specs()
    assert len(specs) >= 3, f"expected the full growth suite, found {len(specs)}"
    saw_factor = saw_interaction = False
    for spec in specs:
        plan = resolve_growth_run_plan(spec)
        assert isinstance(plan, GrowthRunPlan)
        recorded = plan.as_dict()
        for field in _META_FIELDS:
            assert isinstance(recorded[field], str) and recorded[field], (
                f"{spec.model_id}: {field} not recorded"
            )
        assert plan.prepare_kwargs()["baseline_covariates"] == (plan.baseline_covariate,)
        saw_factor |= plan.use_shared_factor
        saw_interaction |= plan.age_ability_interaction
    assert saw_factor, "no shared-factor growth model found (gc-070)"
    assert saw_interaction, "no age x ability growth model found (gc-085)"
