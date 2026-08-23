# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the typed dose-response settings and run plan (#394 pillar 4)."""

from __future__ import annotations

import glob
import importlib
import inspect
import os
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from language_reading_predictors.statistical_models import dose_response as D
from language_reading_predictors.statistical_models import reporting as R
from language_reading_predictors.statistical_models.context import ModelSpec

_META_FIELDS = (
    "design",
    "estimand",
    "causal_status",
    "analysis_population",
    "missing_data_assumption",
)


def _spec(
    *,
    settings=None,
    outcome_symbol: str | None = "W",
    study_id: str = "rli",
    **extra,
) -> ModelSpec:
    return ModelSpec(
        model_id="lrp-rli-dose-999",
        kind="dose_response",
        title="test dose response",
        outcome_symbol=outcome_symbol,
        study_id=study_id,
        family="dose_response",
        design="period-resolved conditional change",
        estimand_type="association",
        causal_status="none",
        model_settings=settings,
        extra=extra,
    )


def _registered_specs() -> list[ModelSpec]:
    root = os.path.dirname(D.__file__)
    specs: list[ModelSpec] = []
    for path in sorted(glob.glob(os.path.join(root, "lrp_rli_dose_*.py"))):
        module = importlib.import_module(
            "language_reading_predictors.statistical_models."
            + os.path.basename(path)[:-3]
        )
        spec = getattr(module, "SPEC", None)
        if spec is not None and spec.kind == "dose_response":
            specs.append(spec)
    return specs


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"adjust_baseline_symbol": ""}, TypeError, "non-empty string"),
        ({"dose_covariate": None}, TypeError, "non-empty string"),
        ({"dose_stage_covariate": ""}, TypeError, "non-empty string"),
        ({"period_varying_dose": 1}, TypeError, "must be a boolean"),
        (
            {"use_subject_random_intercept": 1},
            TypeError,
            "must be a boolean",
        ),
        ({"ability_adjust_symbols": "L"}, TypeError, "sequence of strings"),
        (
            {"ability_adjust_symbols": ("L", "L")},
            ValueError,
            "duplicate",
        ),
        ({"outcomes": ("W", "")}, TypeError, "non-empty string"),
        ({"adjust_group": 1}, TypeError, "must be a boolean"),
        ({"adjust_age": 1}, TypeError, "must be a boolean"),
    ],
)
def test_settings_reject_invalid_values(kwargs, error, message):
    with pytest.raises(error, match=message):
        D.DoseResponseModelSettings(**kwargs)


def test_settings_reject_unknown_legacy_key():
    with pytest.raises(ValueError, match="unknown dose-response setting.*dose_covraite"):
        D.DoseResponseModelSettings.from_legacy_extra(
            {"dose_covraite": "attend"},
            model_id="lrp-rli-dose-999",
        )


def test_settings_accept_global_target_accept_without_owning_it():
    settings = D.DoseResponseModelSettings.from_legacy_extra(
        {"target_accept": 0.99, "period_varying_dose": False},
        model_id="lrp-rli-dose-999",
    )
    assert settings.period_varying_dose is False
    assert "target_accept" not in settings.__dataclass_fields__


def test_typed_settings_allow_only_global_extra_keys():
    plan = D.resolve_dose_response_run_plan(
        _spec(settings=D.DoseResponseModelSettings(), target_accept=0.99)
    )
    assert plan.settings_source == "typed"

    with pytest.raises(ValueError, match="cannot be split.*dose_covariate"):
        D.resolve_dose_response_run_plan(
            _spec(
                settings=D.DoseResponseModelSettings(),
                dose_covariate="sessions",
            )
        )


def test_resolve_rejects_wrong_kind_study_and_missing_outcome():
    wrong = ModelSpec(model_id="x", kind="itt", title="x", outcome_symbol="W")
    with pytest.raises(ValueError, match="expected kind 'dose_response'"):
        D.resolve_dose_response_run_plan(wrong)

    with pytest.raises(ValueError, match="requires study_id='rli'"):
        D.resolve_dose_response_run_plan(_spec(study_id="rlm"))

    with pytest.raises(ValueError, match="outcome_symbol is required"):
        D.resolve_dose_response_run_plan(_spec(outcome_symbol=None))


def test_default_legacy_plan_preserves_execution_contract():
    plan = D.resolve_dose_response_run_plan(_spec())

    assert plan.settings_source == "legacy_extra"
    assert plan.prepare_kwargs() == {
        "phase_mode": "all",
        "outcomes": ("W",),
        "covariates": ("attend",),
    }
    assert plan.factory_kwargs() == {
        "outcome_symbol": "W",
        "adjust_baseline_symbol": "W",
        "dose_covariate": "attend",
        "dose_stage_covariate": None,
        "period_varying_dose": True,
        "use_subject_random_intercept": True,
        "adjust_group": True,
        "adjust_age": True,
        "ability_adjust_symbols": (),
        "ability_baseline_wave": "t1",
        "decompose_between_within": True,
    }
    assert plan.diagnostic_vars() == [
        "alpha",
        "gamma_own",
        "kappa",
        "theta_treated",
        "sigma_child",
        "beta_arm_late",
        "gamma_A",
        "beta_dose_between",
        "mu_dose",
        "sigma_dose",
        "beta_dose_phase",
    ]
    assert plan.observation_node == "y_post"
    assert plan.compute_loo is True
    # Whole-child, not row-level: a row's own baseline is the previous row's
    # fitted outcome, so a row-level score is not out-of-sample (#587 finding 4).
    assert plan.loo_unit == "child"
    assert plan.focal_term == "mu_dose"


def test_plan_owns_pooled_and_ability_adjusted_branches():
    plan = D.resolve_dose_response_run_plan(
        _spec(
            settings=D.DoseResponseModelSettings(
                period_varying_dose=False,
                use_subject_random_intercept=False,
                adjust_group=False,
                adjust_age=False,
                ability_adjust_symbols=("L", "E", "B"),
                outcomes=("W", "L", "E", "B"),
                dose_stage_covariate="attend_cumul",
            )
        )
    )

    assert plan.loader_covariates == ("attend", "attend_cumul")
    assert plan.focal_term == "beta_dose"
    assert plan.diagnostic_vars() == [
        "alpha",
        "gamma_own",
        "kappa",
        "theta_treated",
        "beta_dose_between",
        "beta_dose",
        "gamma_dose_stage",
        "gamma_L_pre",
        "gamma_E_pre",
        "gamma_B_pre",
    ]


def test_resolve_rejects_cross_field_contradictions():
    with pytest.raises(ValueError, match="outcomes must load every"):
        D.resolve_dose_response_run_plan(
            _spec(
                settings=D.DoseResponseModelSettings(
                    adjust_baseline_symbol="L",
                    outcomes=("W",),
                )
            )
        )

    with pytest.raises(ValueError, match="must differ from dose_covariate"):
        D.resolve_dose_response_run_plan(
            _spec(
                settings=D.DoseResponseModelSettings(
                    dose_stage_covariate="attend",
                )
            )
        )


def test_wrong_typed_settings_class_is_rejected():
    from language_reading_predictors.statistical_models.survival import (
        SurvivalModelSettings,
    )

    with pytest.raises(TypeError, match="requires DoseResponseModelSettings"):
        D.resolve_dose_response_run_plan(_spec(settings=SurvivalModelSettings()))


def test_invalid_setting_fails_before_context_reset_or_data_loading(monkeypatch):
    from language_reading_predictors.statistical_models.pipelines import (
        dose_response as P,
    )

    touched = {"context": False, "data": False}

    def _context(*args, **kwargs):
        touched["context"] = True
        raise AssertionError("make_context must not run")

    def _data(*args, **kwargs):
        touched["data"] = True
        raise AssertionError("load_and_prepare must not run")

    monkeypatch.setattr(P, "make_context", _context)
    monkeypatch.setattr(P, "load_and_prepare", _data)

    with pytest.raises(ValueError, match="unknown dose-response setting"):
        P.fit_dose_response(_spec(dose_covraite="attend"))
    assert touched == {"context": False, "data": False}


def test_reporting_dispatch_and_recipe_use_the_attached_plan(tmp_path):
    spec = _spec(
        settings=D.DoseResponseModelSettings(
            ability_adjust_symbols=("L",),
            outcomes=("W", "L"),
        )
    )
    plan = D.resolve_dose_response_run_plan(spec)
    ctx = SimpleNamespace(spec=spec, resolved_plan=plan, output_dir=str(tmp_path))

    assert R._resolved_run_plan(ctx) is plan
    path = R.write_model_recipe(ctx)
    assert path is not None
    text = (tmp_path / "model_recipe.md").read_text(encoding="utf-8")
    assert "validated dose-response run plan" in text
    assert "Observational association, not a randomised treatment effect" in text
    assert "Ability adjustments: L" in text
    assert "zero-divergence convergence gate" in text


def test_pipeline_has_no_direct_dose_response_setting_reads():
    from language_reading_predictors.statistical_models.pipelines import (
        dose_response as P,
    )

    source = inspect.getsource(P.fit_dose_response)
    assert "spec.extra" not in source
    assert "ctx.spec.extra" not in source


def test_registered_models_are_typed_and_preserve_the_legacy_contract():
    specs = _registered_specs()
    assert len(specs) == 5
    assert {spec.outcome_symbol for spec in specs} == {"W", "L", "B"}

    for registered in specs:
        settings = registered.model_settings
        assert isinstance(settings, D.DoseResponseModelSettings)
        assert set(registered.extra) <= {"target_accept"}
        typed = D.resolve_dose_response_run_plan(registered)
        legacy = D.resolve_dose_response_run_plan(
            ModelSpec(
                model_id=registered.model_id,
                kind="dose_response",
                title=registered.title,
                outcome_symbol=registered.outcome_symbol,
                study_id="rli",
                extra={**asdict(settings), **registered.extra},
            )
        )
        typed_contract = typed.as_dict()
        legacy_contract = legacy.as_dict()
        typed_contract.pop("settings_source")
        legacy_contract.pop("settings_source")
        assert typed_contract == legacy_contract
        assert typed.settings_source == "typed"
        assert legacy.settings_source == "legacy_extra"
        for field in _META_FIELDS:
            assert isinstance(typed_contract[field], str) and typed_contract[field]
