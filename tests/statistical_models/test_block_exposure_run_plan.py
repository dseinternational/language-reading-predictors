# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the typed block-exposure settings and run plan (#394 pillar 4)."""

from __future__ import annotations

import glob
import importlib
import inspect
import os
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from language_reading_predictors.statistical_models import block_exposure as B
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
    outcome_symbol: str | None = "TE2",
    study_id: str = "rli",
    **extra,
) -> ModelSpec:
    return ModelSpec(
        model_id="lrp-rli-bx-999",
        kind="block_exposure",
        title="test block exposure",
        outcome_symbol=outcome_symbol,
        study_id=study_id,
        family="block_exposure",
        design="staggered block-active exposure",
        estimand_type="association",
        causal_status="none",
        model_settings=settings,
        extra=extra,
    )


def _registered_specs() -> list[ModelSpec]:
    root = os.path.dirname(B.__file__)
    specs: list[ModelSpec] = []
    for path in sorted(glob.glob(os.path.join(root, "lrp_rli_bx_*.py"))):
        module = importlib.import_module(
            "language_reading_predictors.statistical_models."
            + os.path.basename(path)[:-3]
        )
        spec = getattr(module, "SPEC", None)
        if spec is not None and spec.kind == "block_exposure":
            specs.append(spec)
    return specs


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"ability_covariate": ""}, TypeError, "non-empty string"),
        ({"adjust_for": "hs"}, TypeError, "sequence of strings"),
        ({"adjust_for": ("hs", "hs")}, ValueError, "duplicate"),
        ({"use_child_re": 1}, TypeError, "must be a boolean"),
        ({"likelihood": "binomial"}, ValueError, "beta_binomial"),
        (
            {"drop_ceiling_violations": "UR2"},
            TypeError,
            "sequence of strings",
        ),
        ({"delta_prior_sigma": True}, TypeError, "number or None"),
        ({"delta_prior_sigma": 0}, ValueError, "positive and finite"),
        ({"delta_prior_sigma": float("inf")}, ValueError, "positive and finite"),
    ],
)
def test_settings_reject_invalid_values(kwargs, error, message):
    with pytest.raises(error, match=message):
        B.BlockExposureModelSettings(**kwargs)


def test_settings_reject_unknown_legacy_key():
    with pytest.raises(ValueError, match="unknown block-exposure setting.*adjust_fro"):
        B.BlockExposureModelSettings.from_legacy_extra(
            {"adjust_fro": ("hs",)},
            model_id="lrp-rli-bx-999",
        )


def test_settings_accept_global_target_accept_without_owning_it():
    settings = B.BlockExposureModelSettings.from_legacy_extra(
        {"target_accept": 0.99, "delta_prior_sigma": 0.5},
        model_id="lrp-rli-bx-999",
    )
    assert settings.delta_prior_sigma == 0.5
    assert "target_accept" not in settings.__dataclass_fields__


def test_resolve_rejects_wrong_kind_study_and_outcome():
    wrong = ModelSpec(model_id="x", kind="itt", title="x", outcome_symbol="TE2")
    with pytest.raises(ValueError, match="expected kind 'block_exposure'"):
        B.resolve_block_exposure_run_plan(wrong)

    with pytest.raises(ValueError, match="requires study_id='rli'"):
        B.resolve_block_exposure_run_plan(_spec(study_id="rlm"))

    for outcome in (None, "W"):
        with pytest.raises(ValueError, match="outcome_symbol must be one of"):
            B.resolve_block_exposure_run_plan(_spec(outcome_symbol=outcome))


def test_default_legacy_plan_preserves_execution_contract():
    plan = B.resolve_block_exposure_run_plan(_spec())

    assert plan.settings_source == "legacy_extra"
    assert plan.prepare_kwargs() == {
        "phase_mode": "levels",
        "outcomes": ("TE2",),
        "baseline_covariates": (),
        "covariates": (),
        "post_covariates": (),
        "drop_ceiling_violations": (),
    }
    assert plan.factory_kwargs() == {
        "outcome_symbol": "TE2",
        "ability_covariate": None,
        "adjust_for": (),
        "use_child_re": True,
        "likelihood": "beta_binomial",
        "delta_prior_sigma": None,
    }
    assert plan.coefficient_names() == ["delta", "gamma_A"]
    assert plan.diagnostic_vars() == [
        "alpha",
        "alpha_time",
        "delta",
        "gamma_A",
        "kappa",
        "sigma_child",
    ]
    assert plan.observation_node == "y_post"
    assert plan.compute_loo is True
    assert plan.loo_unit == "observation_row"
    assert plan.focal_term == "delta"


def test_plan_owns_covariate_timing_and_effective_adjustment_names():
    plan = B.resolve_block_exposure_run_plan(
        _spec(
            settings=B.BlockExposureModelSettings(
                ability_covariate="blocks",
                adjust_for=(
                    "hs",
                    "hs_missing",
                    "deapp_c",
                    "deapp_c_missing",
                    "erbto",
                    "erbto_missing",
                ),
                delta_prior_sigma=0.5,
            )
        )
    )

    assert plan.baseline_covariates == (
        "blocks",
        "deapp_c",
        "deapp_c_missing",
        "erbto",
        "erbto_missing",
    )
    assert plan.pre_covariates == ()
    assert plan.post_covariates == ("hs", "hs_missing")
    effective = ("hs", "deapp_c", "erbto")
    assert plan.factory_kwargs(effective_adjustment=effective)["adjust_for"] == effective
    assert plan.coefficient_names(effective_adjustment=effective) == [
        "delta",
        "gamma_A",
        "gamma_ability",
        "gamma_hs",
        "gamma_deapp_c",
        "gamma_erbto",
    ]


def test_off_floor_without_child_intercept_changes_nodes_and_diagnostics():
    plan = B.resolve_block_exposure_run_plan(
        _spec(
            settings=B.BlockExposureModelSettings(
                likelihood="bernoulli_offfloor",
                use_child_re=False,
            )
        )
    )

    assert plan.off_floor is True
    assert plan.observation_node == "y_offfloor"
    assert "kappa" not in plan.diagnostic_vars()
    assert "sigma_child" not in plan.diagnostic_vars()


def test_resolve_rejects_cross_field_contradictions():
    with pytest.raises(ValueError, match="must not also appear in adjust_for"):
        B.resolve_block_exposure_run_plan(
            _spec(
                settings=B.BlockExposureModelSettings(
                    ability_covariate="blocks",
                    adjust_for=("blocks",),
                )
            )
        )

    with pytest.raises(ValueError, match="may only name the fitted outcome"):
        B.resolve_block_exposure_run_plan(
            _spec(
                settings=B.BlockExposureModelSettings(
                    drop_ceiling_violations=("UR2",),
                )
            )
        )


def test_split_settings_between_typed_and_extra_is_rejected():
    with pytest.raises(ValueError, match="cannot be split"):
        B.resolve_block_exposure_run_plan(
            _spec(
                settings=B.BlockExposureModelSettings(),
                use_child_re=False,
            )
        )


def test_wrong_typed_settings_class_is_rejected():
    from language_reading_predictors.statistical_models.survival import (
        SurvivalModelSettings,
    )

    with pytest.raises(TypeError, match="requires BlockExposureModelSettings"):
        B.resolve_block_exposure_run_plan(_spec(settings=SurvivalModelSettings()))


def test_invalid_setting_fails_before_context_reset_or_data_loading(monkeypatch):
    from language_reading_predictors.statistical_models.pipelines import (
        block_exposure as P,
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

    with pytest.raises(ValueError, match="unknown block-exposure setting"):
        P.fit_block_exposure(_spec(adjust_fro=("hs",)))
    assert touched == {"context": False, "data": False}


def test_reporting_dispatch_and_recipe_use_the_attached_plan(tmp_path):
    spec = _spec(
        settings=B.BlockExposureModelSettings(
            ability_covariate="blocks",
            delta_prior_sigma=0.5,
        )
    )
    plan = B.resolve_block_exposure_run_plan(spec)
    ctx = SimpleNamespace(spec=spec, resolved_plan=plan, output_dir=str(tmp_path))

    assert R._resolved_run_plan(ctx) is plan
    path = R.write_model_recipe(ctx)
    assert path is not None
    text = (tmp_path / "model_recipe.md").read_text(encoding="utf-8")
    assert "validated block-exposure run plan" in text
    assert "Adjusted association, not a randomised treatment effect" in text
    assert "Ability covariate: blocks" in text
    assert "Focal delta prior sigma: 0.5" in text
    assert "zero-divergence convergence gate" in text


def test_pipeline_has_no_direct_block_exposure_setting_reads():
    from language_reading_predictors.statistical_models.pipelines import (
        block_exposure as P,
    )

    source = inspect.getsource(P)
    assert "spec.extra" not in source
    assert "ctx.spec.extra" not in source


def test_registered_models_are_typed_and_preserve_the_legacy_contract():
    specs = _registered_specs()
    assert len(specs) == 5
    assert {spec.outcome_symbol for spec in specs} == {"TE2", "TR2", "UE2", "UR2"}

    for registered in specs:
        settings = registered.model_settings
        assert isinstance(settings, B.BlockExposureModelSettings)
        assert registered.extra == {}
        typed = B.resolve_block_exposure_run_plan(registered)
        legacy = B.resolve_block_exposure_run_plan(
            ModelSpec(
                model_id=registered.model_id,
                kind="block_exposure",
                title=registered.title,
                outcome_symbol=registered.outcome_symbol,
                study_id="rli",
                extra=asdict(settings),
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
