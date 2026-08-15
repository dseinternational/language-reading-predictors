# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the typed horseshoe settings and run plan (#394 pillar 4)."""

from __future__ import annotations

import glob
import importlib
import inspect
import os
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from language_reading_predictors.statistical_models import horseshoe as H
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
        model_id=f"lrp-{study_id}-hs-999",
        kind="horseshoe",
        title="test horseshoe",
        outcome_symbol=outcome_symbol,
        study_id=study_id,
        family="horseshoe",
        design="predictor ranking",
        estimand_type="association",
        causal_status="none",
        model_settings=settings,
        extra=extra,
    )


def _registered_specs() -> list[ModelSpec]:
    root = os.path.dirname(H.__file__)
    specs: list[ModelSpec] = []
    for path in sorted(glob.glob(os.path.join(root, "lrp_*_hs_*.py"))):
        module = importlib.import_module(
            "language_reading_predictors.statistical_models."
            + os.path.basename(path)[:-3]
        )
        spec = getattr(module, "SPEC", None)
        if spec is not None and spec.kind == "horseshoe":
            specs.append(spec)
    return specs


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"gain": 1}, TypeError, "boolean or None"),
        ({"predictors": "L"}, TypeError, "sequence of strings"),
        ({"predictors": ("L", "L")}, ValueError, "duplicate"),
        ({"language_composite_symbols": ("R", "")}, TypeError, "non-empty"),
        ({"delta": True}, TypeError, "must be a number"),
        ({"tau0": 0}, ValueError, "positive and finite"),
        ({"slab_scale": float("inf")}, ValueError, "positive and finite"),
        ({"slab_df": -1}, ValueError, "positive and finite"),
        ({"post_time": 0}, ValueError, "at least 1"),
        ({"phase_mode": ""}, TypeError, "non-empty string or None"),
        ({"use_age_predictor": 1}, TypeError, "boolean or None"),
        ({"pre_wave": 1.5}, TypeError, "integer or None"),
        ({"require_confirmed_inputs": 1}, TypeError, "must be a boolean"),
    ],
)
def test_settings_reject_invalid_values(kwargs, error, message):
    with pytest.raises(error, match=message):
        H.HorseshoeModelSettings(**kwargs)


def test_settings_reject_unknown_legacy_key():
    with pytest.raises(ValueError, match="unknown horseshoe setting.*predictores"):
        H.HorseshoeModelSettings.from_legacy_extra(
            {"predictores": ("L",)},
            model_id="lrp-rli-hs-999",
        )


def test_typed_settings_allow_only_global_extra_keys():
    plan = H.resolve_horseshoe_run_plan(
        _spec(
            settings=H.HorseshoeModelSettings(predictors=("L", "age")),
            target_accept=0.99,
        )
    )
    assert plan.settings_source == "typed"

    with pytest.raises(ValueError, match="cannot be split.*gain"):
        H.resolve_horseshoe_run_plan(
            _spec(
                settings=H.HorseshoeModelSettings(predictors=("L",)),
                gain=False,
            )
        )


def test_resolve_rejects_wrong_kind_study_and_missing_outcome():
    wrong = ModelSpec(model_id="x", kind="itt", title="x", outcome_symbol="W")
    with pytest.raises(ValueError, match="expected kind 'horseshoe'"):
        H.resolve_horseshoe_run_plan(wrong)

    with pytest.raises(ValueError, match="study_id must be 'rli' or 'rlm'"):
        H.resolve_horseshoe_run_plan(_spec(study_id="other"))

    with pytest.raises(ValueError, match="outcome_symbol is required"):
        H.resolve_horseshoe_run_plan(_spec(outcome_symbol=None))


def test_rli_plan_preserves_gain_execution_contract():
    plan = H.resolve_horseshoe_run_plan(
        _spec(
            settings=H.HorseshoeModelSettings(
                gain=True,
                predictors=("L", "lang", "age", "blocks"),
                covariates=("blocks",),
                gb_reference="lrp-rli-gbg-012",
            )
        )
    )

    assert plan.port == "rli"
    assert plan.phase_mode == "span"
    assert plan.post_time == 4
    assert plan.measure_symbols == ("W", "L", "R", "E", "F")
    assert plan.rli_prepare_kwargs() == {
        "phase_mode": "span",
        "post_time": 4,
        "outcomes": ("W", "L", "R", "E", "F"),
        "covariates": ("blocks",),
    }
    assert plan.diagnostic_vars() == [
        "alpha",
        "gamma_own",
        "kappa",
        "hs_tau",
        "hs_c2",
        "beta",
    ]
    assert plan.observation_node == "y_post"
    assert plan.compute_loo is True
    assert plan.focal_term is None


def test_rli_level_plan_owns_age_coupling_rule():
    ranked_age = H.resolve_horseshoe_run_plan(
        _spec(
            settings=H.HorseshoeModelSettings(
                gain=False,
                predictors=("L", "age"),
            )
        )
    )
    fixed_age = H.resolve_horseshoe_run_plan(
        _spec(
            settings=H.HorseshoeModelSettings(
                gain=False,
                predictors=("L",),
            )
        )
    )
    assert ranked_age.phase_mode == "levels"
    assert "gamma_A" not in ranked_age.diagnostic_vars()
    assert "gamma_A" in fixed_age.diagnostic_vars()


def test_rlm_plan_preserves_historical_port_contract():
    plan = H.resolve_horseshoe_run_plan(
        _spec(
            study_id="rlm",
            outcome_symbol="basread",
            settings=H.HorseshoeModelSettings(
                predictor_measures=("bpvs", "trog"),
                use_age_predictor=True,
                pre_wave=1,
                post_wave=3,
            ),
        )
    )

    assert plan.port == "rlm"
    assert plan.rlm_prepare_kwargs() == {
        "outcome": "basread",
        "predictor_measures": ("bpvs", "trog"),
        "include_age": True,
        "pre_wave": 1,
        "post_wave": 3,
    }
    assert plan.diagnostic_vars(nuisance=("beta_group",)) == [
        "alpha",
        "gamma_own",
        "kappa",
        "hs_tau",
        "hs_c2",
        "beta",
        "beta_group",
    ]


def test_rlm_confirmed_input_contract_rejects_provisional_measures():
    with pytest.raises(ValueError, match="requires confirmed.*basnum"):
        H.resolve_horseshoe_run_plan(
            _spec(
                study_id="rlm",
                outcome_symbol="bpvs",
                settings=H.HorseshoeModelSettings(
                    predictor_measures=("basnum",),
                    require_confirmed_inputs=True,
                ),
            )
        )


def test_resolve_rejects_cross_port_and_wave_contradictions():
    with pytest.raises(ValueError, match="RLI horseshoe predictors cannot be empty"):
        H.resolve_horseshoe_run_plan(_spec(settings=H.HorseshoeModelSettings()))

    with pytest.raises(ValueError, match="RLM-only settings.*predictor_measures"):
        H.resolve_horseshoe_run_plan(
            _spec(
                settings=H.HorseshoeModelSettings(
                    predictors=("L",),
                    predictor_measures=("bpvs",),
                )
            )
        )

    with pytest.raises(ValueError, match="RLI-only settings.*predictors"):
        H.resolve_horseshoe_run_plan(
            _spec(
                study_id="rlm",
                outcome_symbol="basread",
                settings=H.HorseshoeModelSettings(predictors=("L",)),
            )
        )

    with pytest.raises(ValueError, match="post_wave must be greater"):
        H.resolve_horseshoe_run_plan(
            _spec(
                study_id="rlm",
                outcome_symbol="basread",
                settings=H.HorseshoeModelSettings(pre_wave=3, post_wave=1),
            )
        )

    conflict = _spec(study_id="rlm", outcome_symbol="basread")
    conflict.extra["study_id"] = "rli"
    with pytest.raises(ValueError, match="legacy study_id.*conflicts"):
        H.resolve_horseshoe_run_plan(conflict)


def test_wrong_typed_settings_class_is_rejected():
    from language_reading_predictors.statistical_models.survival import (
        SurvivalModelSettings,
    )

    with pytest.raises(TypeError, match="requires HorseshoeModelSettings"):
        H.resolve_horseshoe_run_plan(_spec(settings=SurvivalModelSettings()))


def test_invalid_setting_fails_before_context_reset_or_data_loading(monkeypatch):
    from language_reading_predictors.statistical_models.pipelines import horseshoe as P

    touched = {"context": False, "data": False}

    def _context(*args, **kwargs):
        touched["context"] = True
        raise AssertionError("make_context must not run")

    def _data(*args, **kwargs):
        touched["data"] = True
        raise AssertionError("load_and_prepare must not run")

    monkeypatch.setattr(P, "make_context", _context)
    monkeypatch.setattr(P, "load_and_prepare", _data)

    with pytest.raises(ValueError, match="unknown horseshoe setting"):
        P.fit_horseshoe(_spec(predictores=("L",)))
    assert touched == {"context": False, "data": False}


def test_reporting_dispatch_and_recipe_use_the_attached_plan(tmp_path):
    spec = _spec(
        settings=H.HorseshoeModelSettings(
            gain=True,
            predictors=("L", "age"),
        )
    )
    plan = H.resolve_horseshoe_run_plan(spec)
    ctx = SimpleNamespace(spec=spec, resolved_plan=plan, output_dir=str(tmp_path))

    assert R._resolved_run_plan(ctx) is plan
    path = R.write_model_recipe(ctx)
    assert path is not None
    text = (tmp_path / "model_recipe.md").read_text(encoding="utf-8")
    assert "validated horseshoe run plan" in text
    assert "Associational ranking only" in text
    assert "Ranked predictors: L, age" in text
    assert "zero-divergence fit" in text


def test_pipeline_has_no_direct_horseshoe_setting_reads():
    from language_reading_predictors.statistical_models.pipelines import horseshoe as P

    for fit in (P.fit_horseshoe, P.fit_rlm_horseshoe):
        source = inspect.getsource(fit)
        assert "spec.extra" not in source
        assert "ctx.spec.extra" not in source


def test_registered_models_are_typed_and_preserve_the_legacy_contract():
    specs = _registered_specs()
    assert len(specs) == 6
    assert {spec.study_id for spec in specs} == {"rli", "rlm"}

    for registered in specs:
        settings = registered.model_settings
        assert isinstance(settings, H.HorseshoeModelSettings)
        assert set(registered.extra) <= {"target_accept"}
        typed = H.resolve_horseshoe_run_plan(registered)
        legacy = H.resolve_horseshoe_run_plan(
            ModelSpec(
                model_id=registered.model_id,
                kind="horseshoe",
                title=registered.title,
                outcome_symbol=registered.outcome_symbol,
                study_id=registered.study_id,
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
