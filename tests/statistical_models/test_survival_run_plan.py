# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the typed survival settings and run plan (#394 pillar 4)."""

from __future__ import annotations

import glob
import importlib
import inspect
import os
from types import SimpleNamespace

import pytest

from language_reading_predictors.statistical_models import reporting as R
from language_reading_predictors.statistical_models import survival as S
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
    outcome_symbol: str | None = "P",
    study_id: str = "rli",
    **extra,
) -> ModelSpec:
    return ModelSpec(
        model_id="lrp-rli-surv-999",
        kind="survival",
        title="test off-floor survival",
        outcome_symbol=outcome_symbol,
        study_id=study_id,
        family="survival",
        design="discrete-time off-floor hazard (person-period)",
        estimand_type="descriptive",
        causal_status="none",
        model_settings=settings,
        extra=extra,
    )


def _registered_specs() -> list[ModelSpec]:
    root = os.path.dirname(S.__file__)
    specs: list[ModelSpec] = []
    for path in sorted(glob.glob(os.path.join(root, "lrp_rli_surv_*.py"))):
        module = importlib.import_module(
            "language_reading_predictors.statistical_models."
            + os.path.basename(path)[:-3]
        )
        spec = getattr(module, "SPEC", None)
        if spec is not None and spec.kind == "survival":
            specs.append(spec)
    return specs


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"hazard_link": "probit"}, ValueError, "cloglog.*logit"),
        ({"use_treatment": 1}, TypeError, "must be a boolean"),
    ],
)
def test_settings_reject_invalid_values(kwargs, error, message):
    with pytest.raises(error, match=message):
        S.SurvivalModelSettings(**kwargs)


def test_settings_reject_unknown_legacy_key():
    with pytest.raises(ValueError, match="unknown survival setting.*hazard_lnik"):
        S.SurvivalModelSettings.from_legacy_extra(
            {"hazard_lnik": "logit"},
            model_id="lrp-rli-surv-999",
        )


def test_settings_accept_global_target_accept_without_owning_it():
    settings = S.SurvivalModelSettings.from_legacy_extra(
        {"target_accept": 0.99, "hazard_link": "logit"},
        model_id="lrp-rli-surv-999",
    )
    assert settings.hazard_link == "logit"
    assert "target_accept" not in settings.__dataclass_fields__


def test_resolve_rejects_wrong_kind_study_and_non_floored_outcome():
    wrong = ModelSpec(model_id="x", kind="itt", title="x", outcome_symbol="P")
    with pytest.raises(ValueError, match="expected kind 'survival'"):
        S.resolve_survival_run_plan(wrong)

    with pytest.raises(ValueError, match="requires study_id='rli'"):
        S.resolve_survival_run_plan(_spec(study_id="rlm"))

    for outcome in (None, "W"):
        with pytest.raises(ValueError, match="outcome_symbol must be one of"):
            S.resolve_survival_run_plan(_spec(outcome_symbol=outcome))


def test_default_legacy_plan_preserves_execution_contract():
    plan = S.resolve_survival_run_plan(_spec())

    assert plan.settings_source == "legacy_extra"
    assert plan.study_id == "rli"
    assert plan.outcome_symbol == "P"
    assert plan.prepare_kwargs() == {"symbol": "P"}
    assert plan.factory_kwargs() == {
        "hazard_link": "cloglog",
        "use_treatment": True,
    }
    assert plan.diagnostic_vars(("L0", "W0", "A0")) == (
        "alpha",
        "beta_L0",
        "beta_W0",
        "beta_A0",
        "tau",
    )
    assert plan.observation_node == "y_event"
    assert plan.compute_loo is True
    assert plan.loo_unit == "person_period_row"
    assert plan.focal_term == "tau"


def test_logit_no_treatment_plan_removes_tau_from_factory_and_diagnostics():
    plan = S.resolve_survival_run_plan(
        _spec(hazard_link="logit", use_treatment=False)
    )

    assert plan.factory_kwargs() == {
        "hazard_link": "logit",
        "use_treatment": False,
    }
    assert plan.diagnostic_vars(("L0",)) == ("alpha", "beta_L0")
    assert plan.focal_term is None
    assert "without an intervention-aligned treatment" in plan.estimand


def test_split_settings_between_typed_and_extra_is_rejected():
    with pytest.raises(ValueError, match="cannot be split"):
        S.resolve_survival_run_plan(
            _spec(
                settings=S.SurvivalModelSettings(),
                hazard_link="logit",
            )
        )


def test_wrong_typed_settings_class_is_rejected():
    from language_reading_predictors.statistical_models.growth import (
        GrowthModelSettings,
    )

    with pytest.raises(TypeError, match="requires SurvivalModelSettings"):
        S.resolve_survival_run_plan(_spec(settings=GrowthModelSettings()))


def test_invalid_setting_fails_before_context_reset_or_data_loading(monkeypatch):
    from language_reading_predictors.statistical_models.pipelines import survival as P

    touched = {"context": False, "data": False}

    def _context(*args, **kwargs):
        touched["context"] = True
        raise AssertionError("make_context must not run")

    def _data(*args, **kwargs):
        touched["data"] = True
        raise AssertionError("prepare_survival must not run")

    monkeypatch.setattr(P, "make_context", _context)
    monkeypatch.setattr(P._survival, "prepare_survival", _data)

    with pytest.raises(ValueError, match="unknown survival setting"):
        P.fit_survival(_spec(hazard_lnik="logit"))
    assert touched == {"context": False, "data": False}


def test_reporting_dispatch_and_recipe_use_the_attached_plan(tmp_path):
    spec = _spec(
        settings=S.SurvivalModelSettings(hazard_link="logit")
    )
    plan = S.resolve_survival_run_plan(spec)
    ctx = SimpleNamespace(spec=spec, resolved_plan=plan, output_dir=str(tmp_path))

    assert R._resolved_run_plan(ctx) is plan
    path = R.write_model_recipe(ctx)
    assert path is not None
    text = (tmp_path / "model_recipe.md").read_text(encoding="utf-8")
    assert "validated survival run plan" in text
    assert "interval-specific probability of first moving above the floor" in text
    assert "Hazard link: `logit`" in text
    assert "person_period_row" in text
    assert "zero-divergence convergence gate" in text


def test_pipeline_has_no_direct_survival_setting_reads():
    from language_reading_predictors.statistical_models.pipelines import survival as P

    source = inspect.getsource(P)
    assert "spec.extra" not in source
    assert "ctx.spec.extra" not in source


def test_registered_models_are_typed_and_preserve_the_legacy_contract():
    specs = _registered_specs()
    assert len(specs) == 2
    assert {spec.outcome_symbol for spec in specs} == {"P", "N"}

    for registered in specs:
        assert isinstance(registered.model_settings, S.SurvivalModelSettings)
        assert registered.extra == {}
        typed = S.resolve_survival_run_plan(registered)
        legacy = S.resolve_survival_run_plan(
            ModelSpec(
                model_id=registered.model_id,
                kind="survival",
                title=registered.title,
                outcome_symbol=registered.outcome_symbol,
                study_id="rli",
                extra={
                    "hazard_link": "cloglog",
                    "use_treatment": True,
                },
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
