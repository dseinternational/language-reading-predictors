# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the typed level-factor settings and resolved run plan (#389 finding 6)."""

from __future__ import annotations

import glob
import importlib
import os

import pytest

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.level_factors import (
    LevelFactorsModelSettings,
    LevelFactorsRunPlan,
    resolve_level_factors_run_plan,
)

_META_FIELDS = (
    "design",
    "estimand",
    "causal_status",
    "analysis_population",
    "missing_data_assumption",
)


def _level_factor_specs() -> list[ModelSpec]:
    """Every registered level-factor model's SPEC."""
    root = os.path.dirname(
        importlib.import_module(
            "language_reading_predictors.statistical_models.level_factors"
        ).__file__
    )
    specs: list[ModelSpec] = []
    for path in sorted(glob.glob(os.path.join(root, "lrp_rli_lf_*.py"))):
        mod = importlib.import_module(
            "language_reading_predictors.statistical_models." + os.path.basename(path)[:-3]
        )
        spec = getattr(mod, "SPEC", None)
        if spec is not None and spec.kind == "level_factors":
            specs.append(spec)
    return specs


# --- settings validation ------------------------------------------------------


def test_settings_reject_unknown_likelihood():
    with pytest.raises(ValueError, match="likelihood"):
        LevelFactorsModelSettings(likelihood="poisson")


def test_settings_reject_non_bool_group_by_time():
    with pytest.raises(TypeError, match="group_by_time"):
        LevelFactorsModelSettings(group_by_time=1)  # type: ignore[arg-type]


def test_settings_reject_non_bool_group_ability():
    with pytest.raises(TypeError, match="group_ability"):
        LevelFactorsModelSettings(group_ability=0)  # type: ignore[arg-type]


def test_settings_reject_string_adjust_for():
    # A bare string is a common mistake for a sequence-of-strings field.
    with pytest.raises(TypeError, match="adjust_for"):
        LevelFactorsModelSettings(adjust_for="hs")  # type: ignore[arg-type]


def test_from_legacy_extra_rejects_unknown_key():
    with pytest.raises(ValueError, match="unknown level-factor setting"):
        LevelFactorsModelSettings.from_legacy_extra(
            {"group_by_time": True, "group_by_tyme": False},  # typo
            model_id="lrp-rli-lf-999",
        )


def test_from_legacy_extra_round_trips_known_keys():
    settings = LevelFactorsModelSettings.from_legacy_extra(
        {
            "ability_covariate": "blocks",
            "adjust_for": ("hs", "deapp_c"),
            "group_by_time": False,
            "ability_by_time": False,
            "group_ability": False,
            "likelihood": "bernoulli_offfloor",
        },
        model_id="lrp-rli-lf-999",
    )
    assert settings.ability_covariate == "blocks"
    assert settings.adjust_for == ("hs", "deapp_c")
    assert settings.group_by_time is False
    assert settings.ability_by_time is False
    assert settings.group_ability is False
    assert settings.likelihood == "bernoulli_offfloor"


def test_from_legacy_extra_defaults_flags_true():
    # The three structural flags default True (the shipped per-timepoint design).
    settings = LevelFactorsModelSettings.from_legacy_extra({}, model_id="lrp-rli-lf-999")
    assert settings.group_by_time is True
    assert settings.ability_by_time is True
    assert settings.group_ability is True
    assert settings.likelihood == "beta_binomial"


# --- resolve ------------------------------------------------------------------


def _spec(**extra) -> ModelSpec:
    return ModelSpec(
        model_id="lrp-rli-lf-000",
        kind="level_factors",
        title="test",
        outcome_symbol="W",
        extra=extra,
    )


def test_resolve_rejects_wrong_kind():
    spec = ModelSpec(model_id="x", kind="gain_factors", title="t", outcome_symbol="W")
    with pytest.raises(ValueError, match="expected kind 'level_factors'"):
        resolve_level_factors_run_plan(spec)


def test_resolve_primary_records_t2_randomised_estimand():
    plan = resolve_level_factors_run_plan(_spec(ability_covariate="blocks"))
    assert plan.settings_source == "legacy_extra"
    assert not plan.off_floor
    assert plan.obs_node == "y_post"
    assert "randomised" in plan.causal_status
    assert "t2" in plan.estimand
    # prepare/factory kwargs are shaped for the loader and the factory.
    assert plan.prepare_kwargs()["outcomes"] == ("W",)
    assert plan.prepare_kwargs()["phase_mode"] == "levels"
    assert plan.factory_kwargs()["group_by_time"] is True


def test_resolve_off_floor_sets_bernoulli_node_and_risk_difference():
    plan = resolve_level_factors_run_plan(
        _spec(ability_covariate="blocks", likelihood="bernoulli_offfloor")
    )
    assert plan.off_floor
    assert plan.obs_node == "y_offfloor"
    assert "risk difference" in plan.estimand


def test_resolve_rejects_group_ability_without_an_ability_covariate():
    # build_level_factors_model raises this, but only after the output directory has
    # been reset and the panel loaded; the plan has to catch it first.
    with pytest.raises(ValueError, match="group_ability requires an ability_covariate"):
        resolve_level_factors_run_plan(_spec(group_ability=True))


def test_resolve_splits_adjust_for_by_wave():
    # deapp_c (speech) is a language-proximal confounder → baseline (t1) timing so
    # the t2 contrast is not conditioned on a treatment-affected descendant; hs
    # (hearing) is exogenous → contemporaneous (post). Mirrors #247 timing.
    plan = resolve_level_factors_run_plan(
        _spec(ability_covariate="blocks", adjust_for=("hs", "deapp_c"))
    )
    assert "blocks" in plan.baseline_covariates
    assert "deapp_c" in plan.baseline_covariates
    assert "hs" in plan.post_covariates


def test_factory_kwargs_apply_effective_adjustment():
    plan = resolve_level_factors_run_plan(
        _spec(ability_covariate="blocks", adjust_for=("hs", "deapp_c"))
    )
    kw = plan.factory_kwargs(effective_adjustment=("hs",))
    assert kw["adjust_for"] == ("hs",)


def test_typed_settings_are_accepted_and_sourced():
    spec = ModelSpec(
        model_id="lrp-rli-lf-000",
        kind="level_factors",
        title="test",
        outcome_symbol="W",
        model_settings=LevelFactorsModelSettings(ability_covariate="blocks"),
    )
    plan = resolve_level_factors_run_plan(spec)
    assert plan.settings_source == "typed"
    assert plan.ability_covariate == "blocks"


def test_split_settings_between_typed_and_extra_is_rejected():
    spec = ModelSpec(
        model_id="lrp-rli-lf-000",
        kind="level_factors",
        title="test",
        outcome_symbol="W",
        model_settings=LevelFactorsModelSettings(),
        extra={"ability_covariate": "blocks"},
    )
    with pytest.raises(ValueError, match="cannot be split"):
        resolve_level_factors_run_plan(spec)


# --- registered-specification coverage (acceptance criterion) -----------------


def test_every_registered_level_factor_model_resolves_with_metadata():
    """Every registered level-factor model resolves to a validated plan that records
    the design, estimand, causal status, analysis population and missing-data
    assumption (#389 findings 4 & 6 acceptance criteria)."""
    specs = _level_factor_specs()
    assert len(specs) >= 11, f"expected the full level-factor suite, found {len(specs)}"
    for spec in specs:
        plan = resolve_level_factors_run_plan(spec)
        assert isinstance(plan, LevelFactorsRunPlan)
        recorded = plan.as_dict()
        for field in _META_FIELDS:
            assert isinstance(recorded[field], str) and recorded[field], (
                f"{spec.model_id}: {field} not recorded"
            )
        # The outcome is always loaded as its own (only) outcome.
        assert plan.prepare_kwargs()["outcomes"] == (spec.outcome_symbol,)


# --- plan-owned names, roles and data guards (#389 criteria 10-11) -------------


def _primary_spec(**extra_overrides) -> ModelSpec:
    extra = {"ability_covariate": "blocks", "adjust_for": ("hs", "erbto")}
    extra.update(extra_overrides)
    return ModelSpec(
        model_id="lrp-test-lf-plan", kind="level_factors", title="t",
        outcome_symbol="W", extra=extra,
    )


def test_plan_owns_coefficient_names_and_diag_vars():
    """The run plan is the single source of truth for coefficient and diagnostic
    names (#389 finding 6): the former pipeline reconstruction is reproduced,
    with the anchored-intercept nodes (#389 finding 2) in the gated set."""
    plan = resolve_level_factors_run_plan(_primary_spec())
    assert plan.coefficient_names() == [
        "b_grp_time", "gamma_A", "gamma_ability_time", "gamma_grp_ability",
        "gamma_hs", "gamma_erbto",
    ]
    # A loader-dropped constant covariate shrinks the reported set to match.
    assert plan.coefficient_names(effective_adjustment=("erbto",)) == [
        "b_grp_time", "gamma_A", "gamma_ability_time", "gamma_grp_ability",
        "gamma_erbto",
    ]
    assert plan.diag_vars(effective_adjustment=()) == [
        "alpha", "alpha_offset", "alpha_time",
        "b_grp_time", "gamma_A", "gamma_ability_time", "gamma_grp_ability",
        "kappa", "sigma_child",
    ]
    assert plan.causal_vector == "b_grp_time"
    assert plan.causal_terms == ("b_grp_time[1]",)


def test_plan_offfloor_diag_vars_drop_kappa_and_pooled_group_unflagged():
    off = resolve_level_factors_run_plan(
        _primary_spec(adjust_for=(), likelihood="bernoulli_offfloor")
    )
    assert "kappa" not in off.diag_vars()
    pooled = resolve_level_factors_run_plan(_primary_spec(adjust_for=(), group_by_time=False))
    assert pooled.causal_vector == "beta_grp"
    # A pooled group coefficient mixes post-crossover waves: never flagged causal.
    assert pooled.causal_terms == ()


def _toy_prepared(*, drop_arm_at_t2: bool = False, nan_ability: bool = False):
    """Minimal stand-in exposing the attributes validate_prepared reads."""
    import numpy as np
    from types import SimpleNamespace

    n_children, n_phases = 6, 4
    child = np.repeat(np.arange(n_children), n_phases)
    phase = np.tile(np.arange(n_phases), n_children)
    G = (child < 3).astype(float)
    post = np.full(child.shape, 5.0)
    if drop_arm_at_t2:
        post[(phase == 1) & (G == 1.0)] = np.nan  # no immediate-arm outcomes at t2
    blocks = np.linspace(-1.0, 1.0, child.size)
    if nan_ability:
        blocks = blocks.copy()
        blocks[3] = np.nan
    return SimpleNamespace(
        post_counts={"W": post}, phase=phase, G=G, covariates={"blocks": blocks},
    )


def test_validate_prepared_accepts_identifiable_panel():
    plan = resolve_level_factors_run_plan(_primary_spec(adjust_for=()))
    plan.validate_prepared(_toy_prepared())  # must not raise


def test_validate_prepared_rejects_t2_missing_an_arm():
    """#389 acceptance criterion: fail before fitting if t2 lacks either arm."""
    plan = resolve_level_factors_run_plan(_primary_spec(adjust_for=()))
    with pytest.raises(ValueError, match="both randomised arms"):
        plan.validate_prepared(_toy_prepared(drop_arm_at_t2=True))


def test_validate_prepared_rejects_non_finite_ability():
    """#389 acceptance criterion: fail before fitting on a non-finite ability."""
    plan = resolve_level_factors_run_plan(_primary_spec(adjust_for=()))
    with pytest.raises(ValueError, match="non-finite 'blocks'"):
        plan.validate_prepared(_toy_prepared(nan_ability=True))
