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


def _spec(*, outcome_symbol: str = "W", **extra) -> ModelSpec:
    return ModelSpec(
        model_id="lrp-rli-al-000",
        kind="aligned",
        title="test",
        outcome_symbol=outcome_symbol,
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
    # The off-floor variant swaps the graded onset-logit coupling for the binary
    # off-floor-at-onset contrast (#391 finding 2; 2026-08-21 aligned review,
    # finding 2) and its recorded design/recipe must describe the Bernoulli
    # likelihood, not the Beta-Binomial (finding 3).
    assert "gamma_own_offfloor" in plan.coefficient_names()
    assert "gamma_own" not in plan.coefficient_names()
    assert "kappa" not in plan.diagnostic_vars()
    assert "Bernoulli" in plan.design
    assert "Beta-Binomial" not in plan.design
    assert "risk difference" in plan.estimand
    recipe = plan.recipe_markdown(title="t")
    assert "bernoulli_offfloor" in recipe
    assert "off-floor-at-onset" in recipe


def test_resolve_beta_binomial_design_and_recipe_name_the_likelihood():
    plan = resolve_aligned_run_plan(_spec())
    assert "gamma_own" in plan.coefficient_names()
    assert "gamma_own_offfloor" not in plan.coefficient_names()
    assert "Beta-Binomial" in plan.design
    assert "beta_binomial" in plan.recipe_markdown(title="t")


def test_resolve_rejects_unknown_outcome_symbol_before_io():
    with pytest.raises(ValueError, match="unknown aligned outcome_symbol"):
        resolve_aligned_run_plan(_spec(outcome_symbol="ZZ"))


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


# --- the phoneme-blending response-link pair (#619) ---------------------------


def test_the_registered_blending_link_pair_is_paired_both_ways():
    """#619: al-006 and al-306 fit the same analysis under the two response links,
    each naming the other, and neither may release alone."""
    specs = {spec.model_id: spec for spec in _aligned_specs()}
    primary = resolve_aligned_run_plan(specs["lrp-rli-al-006"])
    companion = resolve_aligned_run_plan(specs["lrp-rli-al-306"])
    assert primary.score_mean_link == "logit"
    assert companion.score_mean_link == "three_choice_guessing_floor"
    assert primary.required_link_companion_model_id == "lrp-rli-al-306"
    assert companion.required_link_companion_model_id == "lrp-rli-al-006"
    assert primary.link_sensitivity_required_for_release
    assert companion.link_sensitivity_required_for_release
    # Same analysis, one difference.
    for field in (
        "outcome_symbol", "ability_covariate", "use_cohort", "use_dose",
        "likelihood", "off_floor", "estimand", "causal_status",
        "analysis_population",
    ):
        assert getattr(primary, field) == getattr(companion, field), field


def test_only_graded_blending_headlines_require_the_link_pair():
    """A non-B outcome has no chance floor, and the off-floor branch models a binary
    indicator rather than a score mean."""
    for kwargs in ({}, {"likelihood": "bernoulli_offfloor"}):
        plan = resolve_aligned_run_plan(_spec(**kwargs))
        assert plan.outcome_symbol == "W"
        assert not plan.link_sensitivity_required_for_release
        assert plan.required_link_companion_model_id is None


def test_the_dose_sensitivity_variant_is_exempt_from_the_pairing():
    """The pairing governs the fit whose B card is published. The cumulative-session
    dose variant conditions on a collider and is a sensitivity reported beside the
    headline, so requiring a floor twin of it would demand a fit that does not
    exist — the boundary the level window comparator and the gain variants draw."""
    plan = resolve_aligned_run_plan(_spec(outcome_symbol="B", use_dose=True))
    assert not plan.link_sensitivity_required_for_release
    assert plan.required_link_companion_model_id is None
    # ... and it says where the headline lives instead of going silent.
    assert "lrp-rli-al-006 + lrp-rli-al-306" in plan.design


def test_the_guessing_floor_link_is_rejected_for_a_non_blending_outcome():
    with pytest.raises(ValueError, match="only valid for phoneme blending"):
        resolve_aligned_run_plan(
            _spec(score_mean_link="three_choice_guessing_floor")
        )


def test_settings_reject_the_guessing_floor_on_the_off_floor_branch():
    with pytest.raises(ValueError, match="no score mean to map"):
        AlignedModelSettings(
            likelihood="bernoulli_offfloor",
            score_mean_link="three_choice_guessing_floor",
        )


def test_settings_reject_an_unknown_score_mean_link():
    with pytest.raises(ValueError, match="score_mean_link must be one of"):
        AlignedModelSettings(score_mean_link="probit")


def test_the_link_reaches_the_factory_and_the_recipe():
    specs = {spec.model_id: spec for spec in _aligned_specs()}
    companion = resolve_aligned_run_plan(specs["lrp-rli-al-306"])
    assert (
        companion.factory_kwargs()["score_mean_link"]
        == "three_choice_guessing_floor"
    )
    recipe = companion.recipe_markdown(title="t")
    assert "Score-mean link: three_choice_guessing_floor" in recipe
    assert "lrp-rli-al-006" in recipe
