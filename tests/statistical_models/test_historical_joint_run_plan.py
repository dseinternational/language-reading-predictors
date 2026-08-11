# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the typed historical-joint settings and run plan (#394)."""

from __future__ import annotations

import importlib
import inspect
from types import SimpleNamespace

import pytest

from language_reading_predictors.statistical_models import historical_joint as HJ
from language_reading_predictors.statistical_models import reporting as R
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.factories import (
    build_rlm_joint_growth_model,
    default_of,
)


_LEGACY_REGISTERED_EXTRA = {
    "study_id": "rlm",
    "measures": ("basread", "bpvs", "basdig"),
    "waves": (1, 2, 3),
    "extension_waves": (4, 5),
    "eta_prior_sigma": 1.5,
    "sigma_subject_prior_sigma": 1.0,
    "kappa_prior_sigma": 50.0,
    "lkj_eta": 2.0,
}
_META_FIELDS = (
    "design",
    "estimand",
    "causal_status",
    "analysis_population",
    "missing_data_assumption",
)


def _spec(*, settings=None, spec_study_id="rlm", **extra) -> ModelSpec:
    return ModelSpec(
        model_id="lrp-rlm-jc-999",
        kind="historical_joint",
        title="test historical joint",
        study_id=spec_study_id,
        family="historical_joint",
        design="historical_cohort",
        estimand_type="descriptive",
        causal_status="none",
        model_settings=settings,
        extra=extra,
    )


def _registered_spec() -> ModelSpec:
    return importlib.import_module("language_reading_predictors.statistical_models.lrp_rlm_jc_001").SPEC


def test_settings_reject_unknown_legacy_key():
    with pytest.raises(ValueError, match="unknown historical_joint setting.*lkj_etta"):
        HJ.HistoricalJointModelSettings.from_legacy_extra(
            {"lkj_etta": 2.0},
            model_id="lrp-rlm-jc-999",
            spec_study_id="rlm",
        )


def test_settings_accept_global_target_accept_without_owning_it():
    settings = HJ.HistoricalJointModelSettings.from_legacy_extra(
        {"target_accept": 0.99, "lkj_eta": 3.0},
        model_id="lrp-rlm-jc-999",
        spec_study_id="rlm",
    )
    assert settings.lkj_eta == 3.0
    assert "target_accept" not in settings.__dataclass_fields__


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"measures": ("basread",)}, "at least two"),
        ({"measures": ("basread", "basread")}, "duplicate"),
        ({"waves": (1,)}, "at least two"),
        ({"waves": (1, 3, 2)}, "strictly increasing"),
        ({"waves": (1, 2), "extension_waves": (2, 3)}, "overlap"),
        ({"eta_prior_sigma": 0.0}, "positive finite"),
        ({"sigma_subject_prior_sigma": True}, "positive finite"),
        ({"kappa_prior_sigma": float("inf")}, "positive finite"),
        ({"lkj_eta": -1.0}, "positive finite"),
    ],
)
def test_settings_reject_misshaped_or_incoherent_values(kwargs, message):
    with pytest.raises((TypeError, ValueError), match=message):
        HJ.HistoricalJointModelSettings(**kwargs)


def test_resolve_rejects_wrong_kind_unknown_measure_and_study_conflict():
    wrong = ModelSpec(model_id="x", kind="joint", title="x", study_id="rlm")
    with pytest.raises(ValueError, match="expected kind 'historical_joint'"):
        HJ.resolve_historical_joint_run_plan(wrong)

    with pytest.raises(ValueError, match="unregistered 'rlm' measure"):
        HJ.resolve_historical_joint_run_plan(_spec(measures=("basread", "not_a_measure")))

    with pytest.raises(ValueError, match="contradicts ModelSpec.study_id"):
        HJ.resolve_historical_joint_run_plan(_spec(study_id="other"))


def test_default_legacy_plan_preserves_loader_factory_and_diagnostic_contract():
    plan = HJ.resolve_historical_joint_run_plan(_spec())
    assert plan.settings_source == "legacy_extra"
    assert plan.study_id == "rlm"
    assert plan.prepare_kwargs() == {
        "waves": (1, 2, 3),
        "complete_case": True,
        "extension_waves": (),
    }
    assert plan.factory_kwargs() == {
        "measures": ("basread", "bpvs", "basdig"),
        "eta_prior_sigma": default_of(build_rlm_joint_growth_model, "eta_prior_sigma"),
        "sigma_subject_prior_sigma": default_of(build_rlm_joint_growth_model, "sigma_subject_prior_sigma"),
        "kappa_prior_sigma": default_of(build_rlm_joint_growth_model, "kappa_prior_sigma"),
        "lkj_eta": default_of(build_rlm_joint_growth_model, "lkj_eta"),
    }
    assert plan.diagnostic_vars() == [
        "eta_cell",
        "sigma_subject",
        "kappa",
        "measure_corr_pairs",
    ]
    assert plan.compute_loo is False
    assert plan.observation_nodes == (
        "score_basread",
        "score_bpvs",
        "score_basdig",
    )


def test_split_settings_between_typed_and_extra_is_rejected():
    spec = _spec(
        settings=HJ.HistoricalJointModelSettings(),
        lkj_eta=3.0,
    )
    with pytest.raises(ValueError, match="cannot be split"):
        HJ.resolve_historical_joint_run_plan(spec)


def test_invalid_setting_fails_before_context_reset_or_data_loading(monkeypatch):
    from language_reading_predictors.statistical_models.pipelines import (
        historical_joint as P,
    )

    touched = {"context": False, "data": False}

    def _context(*args, **kwargs):
        touched["context"] = True
        raise AssertionError("make_context must not run")

    def _data(*args, **kwargs):
        touched["data"] = True
        raise AssertionError("load_longitudinal_panel must not run")

    monkeypatch.setattr(P, "make_context", _context)
    monkeypatch.setattr(P, "load_longitudinal_panel", _data)
    with pytest.raises(ValueError, match="unknown historical_joint setting"):
        P.fit_rlm_joint_growth(_spec(lkj_etta=2.0))
    assert touched == {"context": False, "data": False}


def test_reporting_dispatch_and_recipe_use_the_attached_plan(tmp_path):
    spec = _spec(settings=HJ.HistoricalJointModelSettings(extension_waves=(4, 5)))
    plan = HJ.resolve_historical_joint_run_plan(spec)
    ctx = SimpleNamespace(spec=spec, resolved_plan=plan, output_dir=str(tmp_path))
    assert R._resolved_run_plan(ctx) is plan
    path = R.write_model_recipe(ctx)
    assert path is not None
    text = (tmp_path / "model_recipe.md").read_text(encoding="utf-8")
    assert "validated historical-joint run plan" in text
    assert "between-child correlation matrix" in text
    assert "Available-case extension waves: 4, 5" in text
    assert "PSIS-LOO is not computed" in text


def test_pipeline_has_no_direct_historical_joint_setting_reads():
    from language_reading_predictors.statistical_models.pipelines import (
        historical_joint as P,
    )

    source = inspect.getsource(P)
    assert "spec.extra" not in source
    assert "ctx.spec.extra" not in source


def test_registered_model_is_typed_and_preserves_the_legacy_contract():
    registered = _registered_spec()
    assert isinstance(registered.model_settings, HJ.HistoricalJointModelSettings)
    assert registered.extra == {}

    typed = HJ.resolve_historical_joint_run_plan(registered)
    legacy = HJ.resolve_historical_joint_run_plan(_spec(**_LEGACY_REGISTERED_EXTRA))
    typed_contract = typed.as_dict()
    legacy_contract = legacy.as_dict()
    typed_contract.pop("model_id")
    legacy_contract.pop("model_id")
    typed_contract.pop("settings_source")
    legacy_contract.pop("settings_source")
    assert typed_contract == legacy_contract
    assert typed.settings_source == "typed"
    assert legacy.settings_source == "legacy_extra"
    assert typed.prepare_kwargs() == {
        "waves": (1, 2, 3),
        "complete_case": True,
        "extension_waves": (4, 5),
    }
    assert typed.factory_kwargs() == {
        "measures": ("basread", "bpvs", "basdig"),
        "eta_prior_sigma": 1.5,
        "sigma_subject_prior_sigma": 1.0,
        "kappa_prior_sigma": 50.0,
        "lkj_eta": 2.0,
    }
    for field in _META_FIELDS:
        assert isinstance(typed.as_dict()[field], str) and typed.as_dict()[field]
