# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the typed difference-in-differences settings and run plan (#394 pillar 4)."""

from __future__ import annotations

import glob
import importlib
import os

import pytest

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.did import (
    DiDModelSettings,
    DiDRunPlan,
    resolve_did_run_plan,
)

_META_FIELDS = (
    "design",
    "estimand",
    "causal_status",
    "analysis_population",
    "missing_data_assumption",
)


def _did_specs() -> list[ModelSpec]:
    """Every registered difference-in-differences model's SPEC."""
    root = os.path.dirname(
        importlib.import_module(
            "language_reading_predictors.statistical_models.did"
        ).__file__
    )
    specs: list[ModelSpec] = []
    for path in sorted(glob.glob(os.path.join(root, "lrp_rli_did_*.py"))):
        mod = importlib.import_module(
            "language_reading_predictors.statistical_models." + os.path.basename(path)[:-3]
        )
        spec = getattr(mod, "SPEC", None)
        if spec is not None and spec.kind == "did":
            specs.append(spec)
    return specs


# --- settings validation ------------------------------------------------------


def test_settings_reject_unknown_likelihood():
    with pytest.raises(ValueError, match="likelihood"):
        DiDModelSettings(likelihood="poisson")


def test_settings_reject_non_bool_dose():
    with pytest.raises(TypeError, match="dose"):
        DiDModelSettings(dose=1)  # type: ignore[arg-type]


def test_settings_reject_string_outcomes():
    with pytest.raises(TypeError, match="outcomes"):
        DiDModelSettings(outcomes="W")  # type: ignore[arg-type]


def test_settings_reject_non_int_waves():
    with pytest.raises(TypeError, match="waves"):
        DiDModelSettings(waves=(0, "1"))  # type: ignore[list-item]


def test_settings_reject_period_varying_without_dose():
    with pytest.raises(ValueError, match="period_varying_dose requires dose"):
        DiDModelSettings(period_varying_dose=True)  # dose defaults False


def test_from_legacy_extra_rejects_unknown_key():
    with pytest.raises(ValueError, match="unknown DiD setting"):
        DiDModelSettings.from_legacy_extra(
            {"dose": True, "doze": True},  # typo
            model_id="lrp-rli-did-999",
        )


def test_from_legacy_extra_round_trips_the_dose_keys():
    # Split from the binary case below because dose excludes bernoulli_offfloor and
    # use_varying_delta (#455); one settings object cannot carry every non-default at
    # once and still describe a model the factory would build.
    settings = DiDModelSettings.from_legacy_extra(
        {
            "dose": True,
            "period_varying_dose": True,
            "likelihood": "beta_binomial",
            "outcomes": ("W",),
            "waves": (0, 1),
            "periods": (0, 1),
            "use_child_re": True,
            "use_age": False,
        },
        model_id="lrp-rli-did-999",
    )
    assert settings.dose is True
    assert settings.period_varying_dose is True
    assert settings.outcomes == ("W",)
    assert settings.waves == (0, 1)
    assert settings.periods == (0, 1)
    assert settings.use_age is False


def test_from_legacy_extra_round_trips_the_binary_keys():
    settings = DiDModelSettings.from_legacy_extra(
        {
            "dose": False,
            "likelihood": "bernoulli_offfloor",
            "outcomes": ("W", "L"),
            "waves": (0, 1, 2),
            "use_child_re": True,
            "use_varying_delta": True,
        },
        model_id="lrp-rli-did-999",
    )
    assert settings.dose is False
    assert settings.likelihood == "bernoulli_offfloor"
    assert settings.outcomes == ("W", "L")
    assert settings.waves == (0, 1, 2)
    assert settings.use_child_re is True
    assert settings.use_varying_delta is True


def test_settings_reject_off_floor_with_dose():
    with pytest.raises(ValueError, match="bernoulli_offfloor is the binary prevalence"):
        DiDModelSettings(dose=True, likelihood="bernoulli_offfloor")


def test_settings_reject_varying_delta_with_dose():
    with pytest.raises(ValueError, match="use_varying_delta is unavailable for dose"):
        DiDModelSettings(dose=True, use_varying_delta=True)


def test_settings_reject_varying_delta_without_a_child_random_intercept():
    with pytest.raises(ValueError, match="use_varying_delta=True requires use_child_re"):
        DiDModelSettings(use_varying_delta=True, use_child_re=False)


def test_settings_reject_dose_with_non_transition_periods():
    with pytest.raises(ValueError, match=r"dose variants require periods=\(0, 1\)"):
        DiDModelSettings(dose=True, periods=(0, 1, 2))


# --- resolve ------------------------------------------------------------------


def _spec(**extra) -> ModelSpec:
    return ModelSpec(
        model_id="lrp-rli-did-000",
        kind="did",
        title="test",
        outcome_symbol="W",
        extra=extra,
    )


def test_resolve_rejects_wrong_kind():
    spec = ModelSpec(model_id="x", kind="itt", title="t", outcome_symbol="W")
    with pytest.raises(ValueError, match="expected kind 'did'"):
        resolve_did_run_plan(spec)


def test_resolve_binary_is_levels_frame_and_tau_t2_estimand():
    plan = resolve_did_run_plan(_spec())
    assert plan.settings_source == "legacy_extra"
    assert not plan.dose and not plan.period_varying and not plan.off_floor
    assert plan.effect_term == "tau_t2"
    assert plan.obs_node == "y_post"
    assert "randomised" in plan.causal_status
    assert "tau_t2" in plan.estimand
    prep = plan.prepare_kwargs()
    assert prep["phase_mode"] == "levels"
    assert prep["outcomes"] == ("W",)  # defaults to the outcome symbol
    assert prep["require_any_post"] is False
    assert "covariates" not in prep  # binary loads no session covariate


def test_resolve_dose_is_transition_frame_with_attend_and_associational():
    plan = resolve_did_run_plan(_spec(dose=True))
    assert plan.dose and not plan.period_varying
    assert plan.effect_term == "beta_dose"
    prep = plan.prepare_kwargs()
    assert prep["phase_mode"] == "all"
    assert prep["covariates"] == ("attend",)
    assert prep["pre_required"] == ()
    assert plan.causal_status.startswith("Associational")
    assert "observational" in plan.estimand.lower()


def test_resolve_period_varying_dose_focal_term():
    plan = resolve_did_run_plan(_spec(dose=True, period_varying_dose=True))
    assert plan.period_varying
    assert plan.effect_term == "mu_dose"
    # The factory receives the *resolved* period_varying under period_varying_dose.
    assert plan.factory_kwargs()["period_varying_dose"] is True


def test_resolve_off_floor_sets_bernoulli_node():
    plan = resolve_did_run_plan(_spec(likelihood="bernoulli_offfloor"))
    assert plan.off_floor
    assert plan.obs_node == "y_offfloor"
    assert plan.factory_kwargs()["likelihood"] == "bernoulli_offfloor"


def test_resolve_explicit_outcomes_are_kept():
    plan = resolve_did_run_plan(_spec(outcomes=("W", "L")))
    assert plan.prepare_kwargs()["outcomes"] == ("W", "L")


def test_typed_settings_are_accepted_and_sourced():
    spec = ModelSpec(
        model_id="lrp-rli-did-000",
        kind="did",
        title="test",
        outcome_symbol="W",
        model_settings=DiDModelSettings(dose=True),
    )
    plan = resolve_did_run_plan(spec)
    assert plan.settings_source == "typed"
    assert plan.dose is True


def test_split_settings_between_typed_and_extra_is_rejected():
    spec = ModelSpec(
        model_id="lrp-rli-did-000",
        kind="did",
        title="test",
        outcome_symbol="W",
        model_settings=DiDModelSettings(),
        extra={"dose": True},
    )
    with pytest.raises(ValueError, match="cannot be split"):
        resolve_did_run_plan(spec)


# --- registered-specification coverage (acceptance criterion) -----------------


def test_every_registered_did_model_resolves_with_metadata():
    """Every registered DiD model resolves to a validated plan that records the
    design, estimand, causal status, analysis population and missing-data
    assumption (#394 pillar 4)."""
    specs = _did_specs()
    assert len(specs) >= 14, f"expected the full DiD suite, found {len(specs)}"
    saw_dose = saw_binary = False
    for spec in specs:
        plan = resolve_did_run_plan(spec)
        assert isinstance(plan, DiDRunPlan)
        recorded = plan.as_dict()
        for field in _META_FIELDS:
            assert isinstance(recorded[field], str) and recorded[field], (
                f"{spec.model_id}: {field} not recorded"
            )
        assert plan.prepare_kwargs()["outcomes"][0] == spec.outcome_symbol
        saw_dose |= plan.dose
        saw_binary |= not plan.dose
    assert saw_dose, "no dose DiD model found"
    assert saw_binary, "no binary DiD model found"
