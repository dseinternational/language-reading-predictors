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
    resolve_active_interactions,
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


def test_settings_reject_an_unknown_interaction_term():
    # #455: build_gain_factors_model rejects this, but only after an output directory
    # has been reset and the panel loaded. A typo is the realistic case.
    with pytest.raises(ValueError, match="interaction term 'abilty' not available"):
        GainFactorsModelSettings(
            interactions=(("trt", "abilty"),), ability_covariate="blocks"
        )


def test_settings_reject_ability_interaction_without_an_ability_covariate():
    with pytest.raises(ValueError, match="interaction term 'ability' not available"):
        GainFactorsModelSettings(interactions=(("trt", "ability"),))


def test_settings_reject_an_interaction_on_an_undeclared_skill():
    with pytest.raises(ValueError, match="interaction term 'TR' not available"):
        GainFactorsModelSettings(interactions=(("trt", "TR"),))


def test_settings_accept_every_term_the_factory_builds():
    # trt / age / own are always available; declared skills and (with a covariate)
    # "ability" join them. treated_only keeps "trt" — the factory allows it too.
    GainFactorsModelSettings(
        skill_symbols=("TR", "R"),
        ability_covariate="blocks",
        interactions=(("trt", "own"), ("age", "ability"), ("trt", "TR"), ("own", "R")),
    )
    GainFactorsModelSettings(interactions=(("trt", "own"),), treated_only=True)


def test_treated_only_plan_separates_declared_from_fitted_interactions():
    # The b companions declare their parent's interactions on purpose, so the pair is
    # a one-line diff; the factory then drops the trt ones because the treatment
    # indicator is constant. The plan must record both, or config.json would claim
    # coefficients the posterior does not contain.
    spec = _spec(
        skill_symbols=("TR",),
        ability_covariate="blocks",
        interactions=(("trt", "ability"), ("trt", "own"), ("age", "ability")),
        treated_only=True,
    )
    plan = resolve_gain_factors_run_plan(spec)

    assert plan.interactions == (("trt", "ability"), ("trt", "own"), ("age", "ability"))
    assert plan.active_interactions == (("age", "ability"),)
    assert plan.factory_kwargs()["interactions"] == (("age", "ability"),)

    recorded = plan.as_dict()
    assert recorded["interactions"] == [
        ["trt", "ability"], ["trt", "own"], ["age", "ability"],
    ]
    assert recorded["active_interactions"] == [["age", "ability"]]

    recipe = plan.recipe_markdown(title="t")
    assert "Interactions: age x ability" in recipe
    assert "declared but not fitted" in recipe


def test_untreated_plan_leaves_interactions_alone():
    plan = resolve_gain_factors_run_plan(
        _spec(
            ability_covariate="blocks",
            interactions=(("trt", "ability"), ("age", "ability")),
        )
    )
    assert plan.active_interactions == plan.interactions
    assert plan.as_dict()["active_interactions"] == plan.as_dict()["interactions"]
    assert "declared but not fitted" not in plan.recipe_markdown(title="t")


def test_active_interactions_matches_the_factory_filter():
    """The shared helper and build_gain_factors_model must not drift apart.

    The factory keeps its own copy because it takes raw keyword arguments from any
    caller, so pin its filter line and check the helper agrees on both branches.
    """
    import inspect

    from language_reading_predictors.statistical_models import factories

    src = inspect.getsource(factories.build_gain_factors_model)
    assert 'pair for pair in interactions if include_trt or "trt" not in pair' in src
    assert "include_trt = not treated_only" in src

    declared = (("trt", "ability"), ("trt", "own"), ("age", "ability"), ("own", "TR"))
    for treated_only in (False, True):
        include_trt = not treated_only
        expected = tuple(
            p for p in declared if include_trt or "trt" not in p
        )
        assert (
            resolve_active_interactions(declared, treated_only=treated_only) == expected
        )


def test_interaction_vocabulary_matches_the_factory_term_set():
    """The settings check and build_gain_factors_model must not drift apart.

    Both are asserted against the factory's own source so a change there fails here
    rather than silently making one of the two stricter than the other.
    """
    import inspect

    from language_reading_predictors.statistical_models import factories

    src = inspect.getsource(factories.build_gain_factors_model)
    assert 'valid_terms = {"trt", "age", "own", *skill_symbols}' in src
    assert 'valid_terms.add("ability")' in src

    for skills in [(), ("TR",), ("R", "E"), ("TR", "TE", "L", "B")]:
        for ability in (None, "blocks"):
            expected = {"trt", "age", "own", *skills}
            if ability is not None:
                expected.add("ability")
            settings = GainFactorsModelSettings(
                skill_symbols=skills, ability_covariate=ability
            )
            assert set(settings.interaction_vocabulary()) == expected


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
