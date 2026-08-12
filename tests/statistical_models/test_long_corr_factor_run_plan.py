# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the longitudinal correlated-factor typed run plan (#394)."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from language_reading_predictors.statistical_models import long_corr_factor as L
from language_reading_predictors.statistical_models import reporting as R
from language_reading_predictors.statistical_models.context import ModelSpec


def _spec(*, settings=None, study_id: str = "rli", **extra) -> ModelSpec:
    return ModelSpec(
        model_id="lrp-rli-lcf-999",
        kind="long_corr_factor",
        title="test longitudinal factor",
        outcome_symbol=None,
        study_id=study_id,
        family="long_corr_factor",
        design="measurement model",
        estimand_type="association",
        causal_status="none",
        model_settings=settings,
        extra=extra,
    )


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"domains": "vocabulary"}, TypeError, "mapping or sequence"),
        ({"domains": (("vocabulary", ("R",)),)}, ValueError, "at least two"),
        (
            {"domains": (("vocabulary", ("R", "R")), ("code", ("L",)))},
            ValueError,
            "duplicate indicators",
        ),
        (
            {"domains": (("vocabulary", ("R",)), ("code", ("R",)))},
            ValueError,
            "only one domain",
        ),
        ({"loading_prior": "pooled"}, ValueError, "communality.*free"),
        ({"comm_alpha": True}, TypeError, "must be a number"),
        ({"comm_beta": 0}, ValueError, "positive and finite"),
        ({"loading_sigma": float("inf")}, ValueError, "positive and finite"),
        ({"lkj_eta": -1}, ValueError, "positive and finite"),
        ({"trait_share_a": 0}, ValueError, "positive and finite"),
    ],
)
def test_settings_reject_invalid_values(kwargs, error, message):
    with pytest.raises(error, match=message):
        L.LongCorrFactorModelSettings(**kwargs)


def test_settings_reject_unknown_legacy_key():
    with pytest.raises(ValueError, match="unknown longitudinal-factor setting.*domians"):
        L.LongCorrFactorModelSettings.from_legacy_extra(
            {"domians": {}},
            model_id="lrp-rli-lcf-999",
        )


def test_typed_settings_allow_only_global_extra_keys():
    plan = L.resolve_long_corr_factor_run_plan(
        _spec(settings=L.LongCorrFactorModelSettings(), target_accept=0.999)
    )
    assert plan.settings_source == "typed"

    with pytest.raises(ValueError, match="cannot be split.*lkj_eta"):
        L.resolve_long_corr_factor_run_plan(
            _spec(settings=L.LongCorrFactorModelSettings(), lkj_eta=3)
        )


def test_resolve_rejects_wrong_kind_study_and_outcome():
    wrong = ModelSpec(model_id="x", kind="itt", title="x", outcome_symbol="W")
    with pytest.raises(ValueError, match="expected kind 'long_corr_factor'"):
        L.resolve_long_corr_factor_run_plan(wrong)

    with pytest.raises(ValueError, match="requires study_id='rli'"):
        L.resolve_long_corr_factor_run_plan(_spec(study_id="rlm"))

    spec = _spec()
    spec.outcome_symbol = "W"
    with pytest.raises(ValueError, match="requires outcome_symbol=None"):
        L.resolve_long_corr_factor_run_plan(spec)


def test_default_legacy_plan_preserves_execution_contract():
    plan = L.resolve_long_corr_factor_run_plan(_spec())

    assert plan.settings_source == "legacy_extra"
    assert plan.domain_mapping() == {
        "vocabulary": ("R", "E", "TR", "TE"),
        "code": ("L", "B"),
        "grammar": ("F", "T"),
    }
    assert plan.prepare_kwargs() == {
        "outcomes": ("R", "E", "TR", "TE", "L", "B", "F", "T")
    }
    assert plan.factory_kwargs() == {
        "domains": plan.domain_mapping(),
        "loading_prior": "communality",
        "comm_alpha": 2.0,
        "comm_beta": 2.0,
        "loading_sigma": 1.0,
        "residual_sigma": 1.0,
        "lkj_eta": 2.0,
        "factor_mean_sigma": 1.0,
        "trait_share_a": 1.5,
        "trait_share_b": 1.5,
    }
    assert plan.compute_loo is False
    assert plan.custom_loo is True
    assert plan.loo_unit == "child"
    assert plan.focal_term is None


@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize(
    ("loading_prior", "knob", "message"),
    [
        ("communality", "loading_sigma", "only apply to loading_prior='free'"),
        ("free", "comm_alpha", "only apply to loading_prior='communality'"),
    ],
)
def test_resolve_rejects_inactive_loading_knobs(
    typed, loading_prior, knob, message
):
    values = {"loading_prior": loading_prior, knob: 0.5}
    spec = (
        _spec(settings=L.LongCorrFactorModelSettings(**values))
        if typed
        else _spec(**values)
    )
    with pytest.raises(ValueError, match=message):
        L.resolve_long_corr_factor_run_plan(spec)


def test_free_loading_plan_resolves_legacy_defaults():
    plan = L.resolve_long_corr_factor_run_plan(
        _spec(settings=L.LongCorrFactorModelSettings(loading_prior="free"))
    )
    assert plan.loading_sigma == 1.0
    assert plan.residual_sigma == 1.0
    assert plan.comm_alpha == 2.0
    assert plan.comm_beta == 2.0


def test_wrong_typed_settings_class_is_rejected():
    from language_reading_predictors.statistical_models.survival import (
        SurvivalModelSettings,
    )

    with pytest.raises(TypeError, match="requires LongCorrFactorModelSettings"):
        L.resolve_long_corr_factor_run_plan(
            _spec(settings=SurvivalModelSettings())
        )


def test_invalid_setting_fails_before_context_reset_or_data_loading(monkeypatch):
    from language_reading_predictors.statistical_models.pipelines import (
        long_corr_factor as P,
    )

    touched = {"context": False, "data": False}

    def _context(*args, **kwargs):
        touched["context"] = True
        raise AssertionError("make_context must not run")

    def _data(*args, **kwargs):
        touched["data"] = True
        raise AssertionError("load_wave_panel must not run")

    monkeypatch.setattr(P, "make_context", _context)
    monkeypatch.setattr(P, "load_wave_panel", _data)

    with pytest.raises(ValueError, match="unknown longitudinal-factor setting"):
        P.fit_longitudinal_corr_factor(_spec(domians={}))
    assert touched == {"context": False, "data": False}


def test_reporting_dispatch_and_recipe_use_attached_plan(tmp_path):
    spec = _spec(settings=L.LongCorrFactorModelSettings())
    plan = L.resolve_long_corr_factor_run_plan(spec)
    ctx = SimpleNamespace(spec=spec, resolved_plan=plan, output_dir=str(tmp_path))

    assert R._resolved_run_plan(ctx) is plan
    path = R.write_model_recipe(ctx)
    assert path is not None
    text = (tmp_path / "model_recipe.md").read_text(encoding="utf-8")
    assert "validated longitudinal-factor run plan" in text
    assert "Descriptive measurement associations only" in text
    assert "vocabulary: R, E, TR, TE" in text


def test_pipeline_does_not_read_family_settings_from_extra():
    from language_reading_predictors.statistical_models.pipelines import (
        long_corr_factor as P,
    )

    source = inspect.getsource(P.fit_longitudinal_corr_factor)
    assert "spec.extra" not in source


def test_registered_spec_is_typed_and_matches_legacy_contract():
    from language_reading_predictors.statistical_models.lrp_rli_lcf_001 import SPEC

    assert isinstance(SPEC.model_settings, L.LongCorrFactorModelSettings)
    assert set(SPEC.extra) == {"target_accept"}
    typed = L.resolve_long_corr_factor_run_plan(SPEC)
    legacy = L.resolve_long_corr_factor_run_plan(
        ModelSpec(
            model_id=SPEC.model_id,
            kind=SPEC.kind,
            title=SPEC.title,
            outcome_symbol=None,
            study_id=SPEC.study_id,
            extra={
                "domains": typed.domain_mapping(),
                "target_accept": SPEC.extra["target_accept"],
            },
        )
    )
    assert typed.settings_source == "typed"
    assert legacy.settings_source == "legacy_extra"
    assert typed.prepare_kwargs() == legacy.prepare_kwargs()
    assert typed.factory_kwargs() == legacy.factory_kwargs()
    assert typed.diagnostic_vars() == legacy.diagnostic_vars()
