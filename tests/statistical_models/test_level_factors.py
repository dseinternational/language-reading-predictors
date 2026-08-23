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
    # The three structural flags default True (the shipped per-timepoint design),
    # and the arm-by-time vector is centred on the t1 gap by default (#552).
    settings = LevelFactorsModelSettings.from_legacy_extra({}, model_id="lrp-rli-lf-999")
    assert settings.group_by_time is True
    assert settings.ability_by_time is True
    assert settings.group_ability is True
    assert settings.likelihood == "beta_binomial"
    assert settings.arm_gap_reference == "t1"


def test_settings_reject_unknown_arm_gap_reference():
    with pytest.raises(ValueError, match="arm_gap_reference"):
        LevelFactorsModelSettings(arm_gap_reference="t2")


def test_from_legacy_extra_round_trips_arm_gap_reference():
    settings = LevelFactorsModelSettings.from_legacy_extra(
        {"ability_covariate": "blocks", "arm_gap_reference": "free"},
        model_id="lrp-rli-lf-999",
    )
    assert settings.arm_gap_reference == "free"



def test_settings_reject_duplicate_adjusters():
    """#584 lower-severity 4: a repeated adjuster used to reach PyMC, where the
    duplicate ``gamma_<c>`` name failed after the output directory had been reset."""
    with pytest.raises(ValueError, match="repeats hs"):
        LevelFactorsModelSettings(
            ability_covariate="blocks", adjust_for=("hs", "hs_missing", "hs")
        )


def test_settings_reject_a_missing_indicator_without_its_base_term():
    """#584 lower-severity 4: an indicator with no covariate to flag is not the
    two-term missing-indicator idiom the reports describe."""
    with pytest.raises(ValueError, match="without the covariate they flag"):
        LevelFactorsModelSettings(
            ability_covariate="blocks", adjust_for=("hs_missing",)
        )


def test_settings_accept_a_paired_missing_indicator():
    settings = LevelFactorsModelSettings(
        ability_covariate="blocks", adjust_for=("hs", "hs_missing")
    )
    assert settings.adjust_for == ("hs", "hs_missing")


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
    assert plan.factory_kwargs()["arm_gap_reference"] == "t1"
    # #552: the default parameterisation names the t2 *change* as the focal term
    # and records the parameterisation in the persisted plan.
    assert plan.arm_gap_reference == "t1"
    assert plan.t1_referenced
    assert plan.focal_vector == "d_grp_time"
    assert plan.focal_index == 0
    assert plan.focal_term == "d_grp_time[t2]"
    assert "difference-in-differences" in plan.estimand
    assert "balance" in plan.causal_status
    recorded = plan.as_dict()
    assert recorded["arm_gap_reference"] == "t1"
    assert recorded["focal_term"] == "d_grp_time[t2]"
    assert "Arm-by-time parameterisation" in plan.recipe_markdown(title="t")
    assert "d_grp_time[t2]" in plan.recipe_markdown(title="t")


def test_resolve_free_comparator_keeps_the_raw_t2_gap_focal():
    """``arm_gap_reference="free"`` is the pre-#552 comparator: one free
    coefficient per timepoint, ``b_grp_time[1]`` focal, no balance term."""
    plan = resolve_level_factors_run_plan(
        _spec(ability_covariate="blocks", arm_gap_reference="free")
    )
    assert plan.arm_gap_reference == "free"
    assert not plan.t1_referenced
    assert plan.focal_vector == "b_grp_time"
    assert plan.focal_index == 1
    assert plan.focal_term == "b_grp_time[1]"
    assert plan.causal_terms == ("b_grp_time[1]",)
    assert plan.balance_terms == ()
    assert plan.levels_view_terms == ()
    # 2026-08-20 review finding 9: the free vector's t1 element is the
    # pre-randomisation arm gap — a balance quantity, not an adjusted
    # association — labelled by element so the summary table fences it off.
    # balance_terms stays empty because its other consumers (forest / psense
    # variable lists) need whole variable names.
    assert plan.factor_summary_roles() == {"b_grp_time[0]": "balance"}
    assert plan.coefficient_names()[0] == "b_grp_time"
    assert "b_grp_time[1]" in plan.estimand
    assert "comparator" in plan.recipe_markdown(title="t")


def test_resolve_rejects_t1_reference_on_a_pooled_group_term():
    """#552 acceptance criterion: an incoherent declaration (centring a pooled
    group coefficient on the t1 gap) fails at resolution, before any output
    directory is reset or data are loaded."""
    with pytest.raises(ValueError, match="arm_gap_reference='t1' requires group_by_time"):
        resolve_level_factors_run_plan(
            _spec(ability_covariate="blocks", group_by_time=False)
        )


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


def test_resolve_rejects_ability_by_time_without_an_ability_covariate():
    """#584 lower-severity 4: ``ability_by_time`` silently did nothing without a
    covariate to vary, so a declaration could claim a per-wave ability vector the
    fit never built."""
    spec = ModelSpec(
        model_id="lrp-test-lf-noability",
        kind="level_factors",
        title="t",
        outcome_symbol="W",
        model_settings=LevelFactorsModelSettings(
            ability_covariate=None,
            group_ability=False,
            group_by_time=False,
            arm_gap_reference="free",
        ),
    )
    with pytest.raises(ValueError, match="ability_by_time requires"):
        resolve_level_factors_run_plan(spec)


def test_resolve_pooled_plan_prose_claims_no_randomised_contrast():
    """#584 lower-severity 6: a pooled plan has no focal term, so the generated
    estimand and causal-status prose must not name a t2 randomised contrast."""
    plan = resolve_level_factors_run_plan(
        _primary_spec(adjust_for=(), group_by_time=False, arm_gap_reference="free")
    )
    assert plan.focal_term is None
    assert plan.estimand.startswith("No randomised contrast")
    assert "The t2 randomised group contrast" not in plan.estimand
    assert plan.causal_status.startswith("No coefficient in this fit is causal")
    # The per-timepoint plans keep the t2 wording.
    primary = resolve_level_factors_run_plan(_primary_spec(adjust_for=()))
    assert primary.estimand.startswith("The t2 randomised group contrast")


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


#: The exact registered level-factor suite (#584 lower-severity 3 & 5). The
#: dynamic coverage test above only asserts "at least eleven, each internally
#: consistent"; this table pins the contract a reader of the reports relies on —
#: which outcome each model fits, on which likelihood, and exactly which
#: background terms enter its linear predictor — so an accidental change to an
#: adjustment set, a likelihood or the arm-gap parameterisation, or an unintended
#: extra registration, fails here rather than silently changing a published
#: adjusted association.
_REGISTERED_CONTRACT: dict[str, dict[str, object]] = {
    "lrp-rli-lf-001": {"outcome": "W", "likelihood": "beta_binomial", "adjust_for": ()},
    "lrp-rli-lf-002": {
        "outcome": "R",
        "likelihood": "beta_binomial",
        "adjust_for": ("hs", "hs_missing", "erbto", "erbto_missing"),
    },
    "lrp-rli-lf-003": {
        "outcome": "E",
        "likelihood": "beta_binomial",
        "adjust_for": (
            "hs", "hs_missing", "deapp_c", "deapp_c_missing", "erbto", "erbto_missing",
        ),
    },
    "lrp-rli-lf-004": {
        "outcome": "L",
        "likelihood": "beta_binomial",
        "adjust_for": ("hs", "hs_missing", "deapp_c", "deapp_c_missing"),
    },
    "lrp-rli-lf-005": {
        "outcome": "P",
        "likelihood": "bernoulli_offfloor",
        "adjust_for": ("erbto", "erbto_missing"),
    },
    "lrp-rli-lf-006": {
        "outcome": "B",
        "likelihood": "beta_binomial",
        "adjust_for": (
            "hs", "hs_missing", "deapp_c", "deapp_c_missing", "erbto", "erbto_missing",
        ),
    },
    "lrp-rli-lf-007": {"outcome": "F", "likelihood": "beta_binomial", "adjust_for": ()},
    "lrp-rli-lf-008": {"outcome": "T", "likelihood": "beta_binomial", "adjust_for": ()},
    "lrp-rli-lf-009": {
        "outcome": "TR",
        "likelihood": "beta_binomial",
        "adjust_for": ("hs", "hs_missing", "erbto", "erbto_missing"),
    },
    "lrp-rli-lf-010": {
        "outcome": "TE",
        "likelihood": "beta_binomial",
        "adjust_for": (
            "hs", "hs_missing", "deapp_c", "deapp_c_missing", "erbto", "erbto_missing",
        ),
    },
    "lrp-rli-lf-011": {
        "outcome": "N",
        "likelihood": "bernoulli_offfloor",
        "adjust_for": ("deapp_c", "deapp_c_missing", "erbto", "erbto_missing"),
    },
    # The registered phoneme-blending response-link companion (#584 decision 2):
    # identical to lf-006 in every respect except the score mean.
    "lrp-rli-lf-106": {
        "outcome": "B",
        "likelihood": "beta_binomial",
        "adjust_for": (
            "hs", "hs_missing", "deapp_c", "deapp_c_missing", "erbto", "erbto_missing",
        ),
        "score_mean_link": "three_choice_guessing_floor",
    },
}


# The randomised-window comparators (#584 decision 3) are their primaries in every
# respect except the analysis window, so the table is derived rather than restated:
# a change to a primary's adjustment set must move its comparator too.
_REGISTERED_CONTRACT.update(
    {
        f"lrp-rli-lf-2{int(model_id[-3:]):02d}": {**expected, "waves": ("t1", "t2")}
        for model_id, expected in list(_REGISTERED_CONTRACT.items())
        if model_id != "lrp-rli-lf-106"
    }
)


def test_registered_suite_matches_the_declared_contract():
    """Pin the exact registered ID / outcome / likelihood / adjustment map."""
    specs = {spec.model_id: spec for spec in _level_factor_specs()}
    assert set(specs) == set(_REGISTERED_CONTRACT)
    for model_id, expected in _REGISTERED_CONTRACT.items():
        plan = resolve_level_factors_run_plan(specs[model_id])
        assert plan.outcome_symbol == expected["outcome"], model_id
        assert plan.likelihood == expected["likelihood"], model_id
        assert plan.adjust_for == expected["adjust_for"], model_id
        # The score-mean link is part of the published contract: a silent flip
        # between the two blending links would change the reported effect size.
        assert plan.score_mean_link == expected.get("score_mean_link", "logit"), model_id
        # The analysis window is part of the contract: silently widening a
        # comparator would make it a duplicate of the model of record.
        assert plan.waves == expected.get(
            "waves", ("t1", "t2", "t3", "t4")
        ), model_id
        # Every registered model is a per-wave t1-centred fit on baseline block
        # design, with the group x ability term and the randomised t2 focal change.
        assert plan.ability_covariate == "blocks", model_id
        assert (plan.group_by_time, plan.ability_by_time, plan.group_ability) == (
            True, True, True,
        ), model_id
        assert plan.arm_gap_reference == "t1", model_id
        assert plan.focal_term == "d_grp_time[t2]", model_id


def test_every_registered_model_declares_typed_settings():
    """#584 lower-severity 3: METHODS and docs/models/README say converted
    families declare immutable typed settings; the eleven level modules used to
    declare legacy mutable ``extra`` dictionaries instead."""
    for spec in _level_factor_specs():
        assert isinstance(spec.model_settings, LevelFactorsModelSettings), spec.model_id
        assert not spec.extra, spec.model_id
        assert resolve_level_factors_run_plan(spec).settings_source == "typed"


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
    names (#389 finding 6): under the t1-referenced parameterisation (#552) the
    balance term, the arm-gap changes and the derived levels view lead the report
    order, with the anchored-intercept nodes (#389 finding 2) in the gated set."""
    plan = resolve_level_factors_run_plan(_primary_spec())
    assert plan.coefficient_names() == [
        "arm_gap_t1", "d_grp_time", "b_grp_time",
        "gamma_A", "gamma_ability_time", "gamma_grp_ability",
        "gamma_hs", "gamma_erbto",
    ]
    # A loader-dropped constant covariate shrinks the reported set to match.
    assert plan.coefficient_names(effective_adjustment=("erbto",)) == [
        "arm_gap_t1", "d_grp_time", "b_grp_time",
        "gamma_A", "gamma_ability_time", "gamma_grp_ability",
        "gamma_erbto",
    ]
    assert plan.diag_vars(effective_adjustment=()) == [
        "alpha", "alpha_offset", "alpha_time",
        "arm_gap_t1", "d_grp_time", "b_grp_time",
        "gamma_A", "gamma_ability_time", "gamma_grp_ability",
        # Both the sampled dispersion parameter and the kappa Deterministic the
        # reports quote (#584 decision 4).
        "inv_sqrt_kappa", "kappa", "sigma_child",
    ]
    assert plan.causal_vector == "d_grp_time"
    assert plan.causal_terms == ("d_grp_time[t2]",)
    assert plan.balance_terms == ("arm_gap_t1",)
    assert plan.levels_view_terms == ("b_grp_time",)
    assert plan.factor_summary_roles() == {
        "arm_gap_t1": "balance",
        "b_grp_time": "levels_view",
    }


def test_plan_free_comparator_coefficient_names():
    """The free comparator reproduces the pre-#552 report order."""
    plan = resolve_level_factors_run_plan(_primary_spec(arm_gap_reference="free"))
    assert plan.coefficient_names() == [
        "b_grp_time", "gamma_A", "gamma_ability_time", "gamma_grp_ability",
        "gamma_hs", "gamma_erbto",
    ]
    assert plan.causal_vector == "b_grp_time"


def test_plan_offfloor_diag_vars_drop_kappa_and_pooled_group_unflagged():
    off = resolve_level_factors_run_plan(
        _primary_spec(adjust_for=(), likelihood="bernoulli_offfloor")
    )
    assert "kappa" not in off.diag_vars()
    pooled = resolve_level_factors_run_plan(
        _primary_spec(adjust_for=(), group_by_time=False, arm_gap_reference="free")
    )
    assert pooled.causal_vector == "beta_grp"
    # A pooled group coefficient mixes post-crossover waves: never flagged causal,
    # and it has no focal element for the release gate to read.
    assert pooled.causal_terms == ()
    assert pooled.focal_term is None
    assert pooled.coefficient_names()[0] == "beta_grp"


def _toy_prepared(
    *,
    drop_arm_at_t2: bool = False,
    nan_ability: bool = False,
    drop_arm_at_wave: int | None = None,
    empty_wave: int | None = None,
):
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
    if drop_arm_at_wave is not None:
        post[(phase == drop_arm_at_wave) & (G == 1.0)] = np.nan
    if empty_wave is not None:
        post[phase == empty_wave] = np.nan
    blocks = np.linspace(-1.0, 1.0, child.size)
    if nan_ability:
        blocks = blocks.copy()
        blocks[3] = np.nan
    return SimpleNamespace(
        post_counts={"W": post},
        phase=phase,
        G=G,
        n_phases=n_phases,
        covariates={"blocks": blocks},
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


def test_validate_prepared_rejects_t1_missing_an_arm():
    """#584 finding 8: the t1 balance term the changes are measured from needs
    both arms at t1 too — a one-arm baseline leaves it prior- and later-wave
    driven while the report still calls the t2 change a t1-to-t2 randomised
    difference-in-differences."""
    plan = resolve_level_factors_run_plan(_primary_spec(adjust_for=()))
    with pytest.raises(ValueError, match="t1 rows.*both randomised arms"):
        plan.validate_prepared(_toy_prepared(drop_arm_at_wave=0))


def test_validate_prepared_rejects_a_post_crossover_wave_missing_an_arm():
    """Every wave carries a published arm coefficient, so t3 needs both arms."""
    plan = resolve_level_factors_run_plan(_primary_spec(adjust_for=()))
    with pytest.raises(ValueError, match="t3 rows.*both randomised arms"):
        plan.validate_prepared(_toy_prepared(drop_arm_at_wave=2))


def test_validate_prepared_rejects_an_unsupported_wave():
    """An interior wave with no fitted rows leaves a prior-only published
    coefficient (#584 finding 8)."""
    plan = resolve_level_factors_run_plan(_primary_spec(adjust_for=()))
    with pytest.raises(ValueError, match=r"t4 rows.*both randomised arms.*\[\]"):
        plan.validate_prepared(_toy_prepared(empty_wave=3))


# --- the published natural-scale target (#584 decision 1) ---------------------


def test_plan_records_the_arm_free_standardised_estimand():
    """The stored plan must state which natural-scale quantity the card is, with
    its population, random-effect and moderation conventions, rather than leaving
    a reader to infer them from whichever reporting code is current."""
    plan = resolve_level_factors_run_plan(_primary_spec(adjust_for=()))
    assert plan.standardisation_balance_term == "arm_gap_t1"
    recorded = plan.as_dict()["natural_scale_estimand"]
    assert recorded.startswith("Arm-free standardised items-scale average marginal")
    assert "d_grp_time[t2]" in recorded
    assert "fitted t2 children" in recorded
    assert "each child's own posterior intercept" in recorded
    assert "centred ability" in recorded
    assert plan.as_dict()["standardisation_balance_term"] == "arm_gap_t1"


def test_free_comparator_has_no_separate_balance_term_to_standardise():
    """Under the free parameterisation the focal `b_grp_time[1]` IS the whole t2
    arm gap, so there is nothing else to net out and the decision does not move
    that comparator's card."""
    plan = resolve_level_factors_run_plan(
        _primary_spec(adjust_for=(), arm_gap_reference="free")
    )
    assert plan.standardisation_balance_term is None
    assert plan.as_dict()["standardisation_balance_term"] is None


def test_offfloor_plan_states_the_risk_difference_target():
    plan = resolve_level_factors_run_plan(
        _primary_spec(adjust_for=(), likelihood="bernoulli_offfloor")
    )
    assert plan.natural_scale_estimand.startswith(
        "Arm-free standardised off-floor risk difference"
    )


def test_pooled_plan_publishes_no_natural_scale_target():
    plan = resolve_level_factors_run_plan(
        _primary_spec(adjust_for=(), group_by_time=False, arm_gap_reference="free")
    )
    assert plan.natural_scale_estimand.startswith("none:")


def test_the_registered_blending_link_pair_is_paired_both_ways():
    """#584 decision 2: lf-006 and lf-106 fit the same data under the two response
    links, each naming the other, and neither may release alone."""
    specs = {spec.model_id: spec for spec in _level_factor_specs()}
    primary = resolve_level_factors_run_plan(specs["lrp-rli-lf-006"])
    companion = resolve_level_factors_run_plan(specs["lrp-rli-lf-106"])
    assert primary.score_mean_link == "logit"
    assert companion.score_mean_link == "three_choice_guessing_floor"
    assert primary.required_link_companion_model_id == "lrp-rli-lf-106"
    assert companion.required_link_companion_model_id == "lrp-rli-lf-006"
    assert primary.link_sensitivity_required_for_release
    assert companion.link_sensitivity_required_for_release
    # Same analysis, one difference: the pair is only comparable if everything
    # else about the two fits agrees.
    for field in (
        "outcome_symbol", "adjust_for", "ability_covariate", "group_by_time",
        "ability_by_time", "group_ability", "likelihood", "arm_gap_reference",
        "focal_term", "baseline_covariates", "pre_covariates", "post_covariates",
    ):
        assert getattr(primary, field) == getattr(companion, field), field


def test_only_graded_blending_fits_require_the_link_pair():
    """A non-B outcome has no chance floor to respect, and the off-floor branch
    models a binary indicator rather than a score mean."""
    for kwargs in (
        {},
        {"likelihood": "bernoulli_offfloor"},
    ):
        plan = resolve_level_factors_run_plan(_primary_spec(adjust_for=(), **kwargs))
        assert plan.outcome_symbol == "W"
        assert not plan.link_sensitivity_required_for_release
        assert plan.required_link_companion_model_id is None


# --- the randomised-window comparator (#584 decision 3) -----------------------


def test_settings_reject_a_non_prefix_analysis_window():
    """The t1-centred parameterisation measures every change from t1, so a window
    without t1 — or with a hole in it — would leave a d_grp_time coordinate with no
    data behind it."""
    for bad in (("t1",), ("t2", "t3"), ("t1", "t3"), ("t1", "t2", "t4")):
        with pytest.raises(ValueError, match="contiguous prefix"):
            LevelFactorsModelSettings(ability_covariate="blocks", waves=bad)


def test_two_wave_plan_carries_one_post_phase_coordinate():
    plan = resolve_level_factors_run_plan(
        _primary_spec(adjust_for=(), waves=("t1", "t2"))
    )
    assert plan.two_wave_window
    assert plan.post_phase_labels == ("t2",)
    assert plan.focal_term == "d_grp_time[t2]"
    assert plan.factory_kwargs()["post_phase_labels"] == ("t2",)
    # The model of record is unchanged.
    full = resolve_level_factors_run_plan(_primary_spec(adjust_for=()))
    assert not full.two_wave_window
    assert full.post_phase_labels == ("t2", "t3", "t4")


def test_two_wave_plan_prose_says_it_is_a_comparator():
    plan = resolve_level_factors_run_plan(
        _primary_spec(adjust_for=(), waves=("t1", "t2"))
    )
    assert "the four-wave fit is the model of record" in plan.design
    assert "no post-crossover observation informs the contrast" in plan.causal_status


def _real_toy_prepared():
    """A minimal real ``PreparedData`` — ``restrict_to_declared_waves`` goes through
    the shared row subsetter, which rebuilds every row-indexed field."""
    import numpy as np
    from language_reading_predictors.statistical_models.preprocessing import PreparedData

    n_children, n_phases = 6, 4
    child = np.repeat(np.arange(n_children), n_phases)
    phase = np.tile(np.arange(n_phases), n_children)
    return PreparedData(
        subject_ids=np.array([f"c{i}" for i in child]),
        child_idx=child.astype(np.int64),
        phase=phase.astype(np.int64),
        G=(child < 3).astype(float),
        A_months=np.linspace(60.0, 90.0, child.size),
        A_std=np.linspace(-1.0, 1.0, child.size),
        age_scaler=(75.0, 10.0),
        pre_logit={},
        post_counts={"W": np.full(child.shape, 5.0)},
        n_trials={"W": 79},
        n_obs=int(child.size),
        n_children=n_children,
        n_phases=n_phases,
        dropped_rows=0,
        phase_mode="levels",
        covariates={"blocks": np.linspace(-1.0, 1.0, child.size)},
    )


def test_restricting_to_the_window_drops_the_later_waves_attributably():
    plan = resolve_level_factors_run_plan(
        _primary_spec(adjust_for=(), waves=("t1", "t2"))
    )
    restricted = plan.restrict_to_declared_waves(_real_toy_prepared())
    assert restricted.n_phases == 2
    assert set(restricted.phase.tolist()) == {0, 1}
    assert restricted.n_obs == 12  # 6 children x 2 waves
    # Excluded by design, so counted under its own reason rather than as missing data.
    assert restricted.dropped_by_reason["outside_declared_analysis_window"] == 12
    plan.validate_prepared(restricted)  # must not raise


def test_the_model_of_record_window_is_left_untouched():
    plan = resolve_level_factors_run_plan(_primary_spec(adjust_for=()))
    prepared = _real_toy_prepared()
    assert plan.restrict_to_declared_waves(prepared) is prepared


def test_a_window_comparator_is_exempt_from_the_blending_link_pairing():
    """#584 decisions 2 + 3 interact: the link pairing governs the fit whose B card
    is the headline — the four-wave model of record — so requiring it of a two-wave
    comparator would demand a two-wave floor-link twin that does not exist."""
    comparator = ModelSpec(
        model_id="lrp-rli-lf-206",
        kind="level_factors",
        title="t",
        outcome_symbol="B",
        model_settings=LevelFactorsModelSettings(
            ability_covariate="blocks", waves=("t1", "t2")
        ),
    )
    plan = resolve_level_factors_run_plan(comparator)
    assert not plan.link_sensitivity_required_for_release
    assert plan.required_link_companion_model_id is None
    # ... and it says where the headline lives instead of going silent.
    assert "lrp-rli-lf-006 + lrp-rli-lf-106" in plan.design


def test_every_primary_has_a_registered_window_comparator():
    specs = {spec.model_id: spec for spec in _level_factor_specs()}
    for n in range(1, 12):
        primary, comparator = f"lrp-rli-lf-{n:03d}", f"lrp-rli-lf-2{n:02d}"
        assert comparator in specs, comparator
        parent = resolve_level_factors_run_plan(specs[primary])
        window = resolve_level_factors_run_plan(specs[comparator])
        assert window.waves == ("t1", "t2")
        assert parent.waves == ("t1", "t2", "t3", "t4")
        # A comparator that differed in anything else would not be comparable.
        for field in (
            "outcome_symbol", "adjust_for", "ability_covariate", "group_by_time",
            "ability_by_time", "group_ability", "likelihood", "arm_gap_reference",
            "score_mean_link", "focal_term",
        ):
            assert getattr(parent, field) == getattr(window, field), (comparator, field)


# --- dispersion and child-heterogeneity priors (#584 decision 4) ---------------


def test_the_family_defaults_to_the_dispersion_parameterisation():
    plan = resolve_level_factors_run_plan(_primary_spec(adjust_for=()))
    assert plan.kappa_prior_family == "halfnormal_inverse_sqrt"
    assert plan.sigma_child_prior_sigma == 1.0
    # ``kappa`` becomes a Deterministic, so anything naming a FREE random variable
    # -- power scaling, the all-free-RV gate -- must ask for inv_sqrt_kappa.
    assert plan.dispersion_free_term == "inv_sqrt_kappa"
    assert plan.nuisance_terms == ("inv_sqrt_kappa", "sigma_child")
    # ... while the reports keep speaking in kappa, which is the documented unit.
    assert "kappa" in plan.diag_vars()
    assert "inv_sqrt_kappa" in plan.diag_vars()


def test_the_pre_decision_priors_remain_expressible_as_a_comparator():
    plan = resolve_level_factors_run_plan(
        _primary_spec(
            adjust_for=(),
            kappa_prior_family="halfnormal_concentration",
            sigma_child_prior_sigma=0.5,
        )
    )
    assert plan.dispersion_free_term == "kappa"
    assert plan.nuisance_terms == ("kappa", "sigma_child")
    assert plan.factory_kwargs()["kappa_prior_family"] == "halfnormal_concentration"
    assert plan.factory_kwargs()["sigma_child_prior_sigma"] == 0.5


def test_an_off_floor_fit_records_no_dispersion_prior_at_all():
    """A Bernoulli off-floor fit has no score mean, so it has no concentration —
    recorded as absent rather than as a setting that does nothing."""
    plan = resolve_level_factors_run_plan(
        _primary_spec(adjust_for=(), likelihood="bernoulli_offfloor")
    )
    assert plan.kappa_prior_family is None
    assert plan.kappa_prior_sigma is None
    assert plan.dispersion_free_term is None
    assert plan.nuisance_terms == ("sigma_child",)
    kwargs = plan.factory_kwargs()
    assert "kappa_prior_family" not in kwargs
    assert "kappa_prior_sigma" not in kwargs


def test_prior_scales_must_be_positive_numbers():
    with pytest.raises(ValueError, match="sigma_child_prior_sigma must be positive"):
        LevelFactorsModelSettings(ability_covariate="blocks", sigma_child_prior_sigma=0)
    with pytest.raises(ValueError, match="kappa_prior_sigma must be positive"):
        LevelFactorsModelSettings(ability_covariate="blocks", kappa_prior_sigma=-1.0)
    with pytest.raises(ValueError, match="kappa_prior_family must be one of"):
        LevelFactorsModelSettings(
            ability_covariate="blocks", kappa_prior_family="halfcauchy"
        )


def test_every_registered_model_uses_the_decided_priors():
    """The decision is family-wide, so no registered fit may quietly keep the
    pre-decision scales."""
    for spec in _level_factor_specs():
        plan = resolve_level_factors_run_plan(spec)
        assert plan.sigma_child_prior_sigma == 1.0, plan.model_id
        assert plan.kappa_prior_sigma is None, plan.model_id
        expected = None if plan.off_floor else "halfnormal_inverse_sqrt"
        assert plan.kappa_prior_family == expected, plan.model_id

