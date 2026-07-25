# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the typed gain-factor settings and resolved run plan (#391 finding 6)."""

from __future__ import annotations

import glob
import importlib
import os

import pytest

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.gain_factors import (
    GainFactorsModelSettings,
    GainFactorsRunPlan,
    resolve_gain_factors_run_plan,
)

_META_FIELDS = (
    "design",
    "estimand",
    "causal_status",
    "analysis_population",
    "missing_data_assumption",
)


def _gain_factor_specs() -> list[ModelSpec]:
    """Every registered gain-factor model's SPEC (primary and treated-only)."""
    root = os.path.dirname(
        importlib.import_module(
            "language_reading_predictors.statistical_models.gain_factors"
        ).__file__
    )
    specs: list[ModelSpec] = []
    for path in sorted(glob.glob(os.path.join(root, "lrp_rli_gf_*.py"))):
        mod = importlib.import_module(
            "language_reading_predictors.statistical_models." + os.path.basename(path)[:-3]
        )
        spec = getattr(mod, "SPEC", None)
        if spec is not None and spec.kind == "gain_factors":
            specs.append(spec)
    return specs


# --- settings validation ------------------------------------------------------


def test_settings_reject_unknown_likelihood():
    with pytest.raises(ValueError, match="likelihood"):
        GainFactorsModelSettings(likelihood="poisson")


def test_settings_reject_non_bool_treated_only():
    with pytest.raises(TypeError, match="treated_only"):
        GainFactorsModelSettings(treated_only=1)  # type: ignore[arg-type]


def test_settings_reject_bad_interaction_pairs():
    with pytest.raises(TypeError, match="interactions"):
        GainFactorsModelSettings(interactions=(("trt",),))  # not a 2-tuple


def test_settings_reject_string_skill_symbols():
    # A bare string is a common mistake for a sequence-of-strings field.
    with pytest.raises(TypeError, match="skill_symbols"):
        GainFactorsModelSettings(skill_symbols="TR")  # type: ignore[arg-type]


def test_from_legacy_extra_rejects_unknown_key():
    with pytest.raises(ValueError, match="unknown gain-factor setting"):
        GainFactorsModelSettings.from_legacy_extra(
            {"skill_symbols": ("R",), "skil_symbols": ("E",)},  # typo
            model_id="lrp-rli-gf-999",
        )


def test_from_legacy_extra_round_trips_known_keys():
    settings = GainFactorsModelSettings.from_legacy_extra(
        {
            "skill_symbols": ("R", "E"),
            "ability_covariate": "blocks",
            "interactions": (("trt", "own"),),
            "treated_only": True,
            "likelihood": "bernoulli_offfloor",
        },
        model_id="lrp-rli-gf-999",
    )
    assert settings.skill_symbols == ("R", "E")
    assert settings.ability_covariate == "blocks"
    assert settings.interactions == (("trt", "own"),)
    assert settings.treated_only is True
    assert settings.likelihood == "bernoulli_offfloor"


# --- resolve ------------------------------------------------------------------


def _spec(**extra) -> ModelSpec:
    return ModelSpec(
        model_id="lrp-rli-gf-000",
        kind="gain_factors",
        title="test",
        outcome_symbol="W",
        extra=extra,
    )


def test_resolve_rejects_outcome_as_own_skill_baseline():
    with pytest.raises(ValueError, match="cannot also be an upstream skill"):
        resolve_gain_factors_run_plan(_spec(skill_symbols=("W", "R")))


def test_resolve_primary_records_randomised_causal_status():
    plan = resolve_gain_factors_run_plan(_spec(skill_symbols=("R",)))
    assert plan.settings_source == "legacy_extra"
    assert not plan.off_floor
    assert plan.obs_node == "y_post"
    assert "randomised" in plan.causal_status
    assert "average marginal effect" in plan.estimand
    # prepare/factory kwargs are shaped for the loader and the factory.
    assert plan.prepare_kwargs()["outcomes"] == ("W", "R")
    assert plan.prepare_kwargs()["phase_mode"] == "all"
    assert plan.factory_kwargs()["skill_symbols"] == ("R",)


def test_resolve_treated_only_is_associational():
    plan = resolve_gain_factors_run_plan(_spec(treated_only=True))
    assert plan.treated_only is True
    assert plan.causal_status.startswith("Associational")
    assert "no randomised" in plan.estimand.lower()


def test_resolve_off_floor_sets_bernoulli_node_and_risk_difference():
    plan = resolve_gain_factors_run_plan(_spec(likelihood="bernoulli_offfloor"))
    assert plan.off_floor
    assert plan.obs_node == "y_offfloor"
    assert "risk difference" in plan.estimand


def test_resolve_splits_adjust_for_by_wave():
    # deapp_c (speech) is a language-proximal confounder → baseline (t1) timing;
    # hs (hearing) is exogenous → contemporaneous (post). Mirrors #247 timing.
    plan = resolve_gain_factors_run_plan(
        _spec(ability_covariate="blocks", adjust_for=("hs", "deapp_c"))
    )
    assert "blocks" in plan.baseline_covariates
    assert "deapp_c" in plan.baseline_covariates
    assert "hs" in plan.post_covariates


def test_factory_kwargs_apply_effective_adjustment():
    plan = resolve_gain_factors_run_plan(_spec(adjust_for=("hs", "deapp_c")))
    kw = plan.factory_kwargs(effective_adjustment=("hs",))
    assert kw["adjust_for"] == ("hs",)


# --- registered-specification coverage (acceptance criterion) -----------------


def test_every_registered_gain_factor_model_resolves_with_metadata():
    """Every registered gain-factor model — primary and treated-only — resolves to
    a validated plan that records the design, estimand, causal status, analysis
    population and missing-data assumption (#391 finding 6 acceptance criterion)."""
    specs = _gain_factor_specs()
    assert len(specs) >= 15, f"expected the full gain-factor suite, found {len(specs)}"
    saw_primary = saw_treated_only = False
    for spec in specs:
        plan = resolve_gain_factors_run_plan(spec)
        assert isinstance(plan, GainFactorsRunPlan)
        recorded = plan.as_dict()
        for field in _META_FIELDS:
            assert isinstance(recorded[field], str) and recorded[field], (
                f"{spec.model_id}: {field} not recorded"
            )
        # The outcome is always loaded as its own first outcome.
        assert plan.prepare_kwargs()["outcomes"][0] == spec.outcome_symbol
        saw_primary |= not plan.treated_only
        saw_treated_only |= plan.treated_only
    assert saw_primary, "no primary gain-factor model found"
    assert saw_treated_only, "no treated-only gain-factor model found"
