# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the typed LCSM settings and run plan (#394 pillar 4)."""

from __future__ import annotations

import glob
import importlib
import inspect
import os
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from language_reading_predictors.statistical_models import lcsm as L
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
        model_id="lrp-rli-lcsm-999",
        kind="lcsm",
        title="test latent change-score model",
        outcome_symbol=outcome_symbol,
        study_id=study_id,
        family="lcsm",
        estimand_type="association",
        causal_status="none",
        model_settings=settings,
        extra=extra,
    )


def _registered_specs() -> list[ModelSpec]:
    root = os.path.dirname(L.__file__)
    specs: list[ModelSpec] = []
    for path in sorted(glob.glob(os.path.join(root, "lrp_rli_lcsm_*.py"))):
        module = importlib.import_module(
            "language_reading_predictors.statistical_models."
            + os.path.basename(path)[:-3]
        )
        spec = getattr(module, "SPEC", None)
        if spec is not None and spec.kind == "lcsm":
            specs.append(spec)
    return specs


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"outcomes": "W"}, TypeError, "sequence of strings"),
        ({"outcomes": ("W", "W")}, ValueError, "duplicate"),
        ({"couplings": (("W", "L"),)}, TypeError, "sequence of strings"),
        (
            {"couplings": (("W", ("L",)), ("W", ("E",)))},
            ValueError,
            "duplicate targets",
        ),
        ({"lagged_change_couplings": "W"}, TypeError, "target-to-sources"),
        ({"covariate_block": ("hs", "hs")}, ValueError, "duplicate"),
        ({"dominance_pair": ("W",)}, ValueError, "exactly two"),
        ({"coupling_prior_sigma": 0}, ValueError, "positive finite"),
        ({"use_process_noise": 1}, TypeError, "must be a boolean"),
        ({"arm_window_intercepts": 1}, TypeError, "must be a boolean"),
    ],
)
def test_settings_reject_invalid_values(kwargs, error, message):
    with pytest.raises(error, match=message):
        L.LcsmModelSettings(**kwargs)


def test_settings_reject_unknown_legacy_key():
    with pytest.raises(ValueError, match="unknown LCSM setting.*coupling_prior_sgima"):
        L.LcsmModelSettings.from_legacy_extra(
            {"coupling_prior_sgima": 0.3},
            model_id="lrp-rli-lcsm-999",
        )


def test_typed_settings_allow_only_global_extra_keys():
    plan = L.resolve_lcsm_run_plan(
        _spec(settings=L.LcsmModelSettings(), target_accept=0.99)
    )
    assert plan.settings_source == "typed"

    with pytest.raises(ValueError, match="cannot be split.*outcomes"):
        L.resolve_lcsm_run_plan(
            _spec(settings=L.LcsmModelSettings(), outcomes=("W", "L"))
        )


def test_resolve_rejects_wrong_kind_study_and_missing_outcome():
    wrong = ModelSpec(model_id="x", kind="itt", title="x", outcome_symbol="W")
    with pytest.raises(ValueError, match="expected kind 'lcsm'"):
        L.resolve_lcsm_run_plan(wrong)

    with pytest.raises(ValueError, match="requires study_id='rli'"):
        L.resolve_lcsm_run_plan(_spec(study_id="rlm"))

    with pytest.raises(ValueError, match="requires outcome_symbol"):
        L.resolve_lcsm_run_plan(_spec(outcome_symbol=None))


def test_default_legacy_plan_preserves_execution_contract():
    plan = L.resolve_lcsm_run_plan(_spec())

    assert plan.settings_source == "legacy_extra"
    assert plan.prepare_kwargs() == {
        "outcomes": ("W", "L", "E"),
        "wave_covariates": (),
        "include_hearing": False,
    }
    assert plan.factory_kwargs() == {
        "reading_symbol": "W",
        "couplings": {"W": ("L", "E")},
        "lagged_change_couplings": None,
        "arm_window_intercepts": False,
        "covariate_block": (),
        "covariate_targets": (),
        "coupling_prior_sigma": 0.3,
        "use_process_noise": True,
        "shared_process_noise": False,
    }
    assert plan.coupling_names() == {("L", "W"): "g_L", ("E", "W"): "g_E"}
    assert plan.lagged_names() == {}
    assert plan.diagnostic_vars() == [
        "g_L",
        "g_E",
        "a_change",
        "b_self",
        "d_age",
        "sigma1",
        "kappa",
        "sigma_proc",
    ]
    assert plan.observation_node == "y_obs"
    assert plan.compute_loo is True


def test_plan_owns_crossover_covariate_and_lagged_names():
    plan = L.resolve_lcsm_run_plan(
        _spec(
            settings=L.LcsmModelSettings(
                outcomes=("TE", "TR", "W"),
                couplings=(("TE", ("W", "TR")), ("TR", ("W",))),
                lagged_change_couplings=(("TE", ("W",)),),
                arm_window_intercepts=True,
                covariate_block=(
                    "hs",
                    "hs_missing",
                    "erbto",
                    "erbto_missing",
                ),
                covariate_targets=("TE", "TR"),
            ),
            outcome_symbol="TE",
        )
    )

    assert plan.prepare_kwargs() == {
        "outcomes": ("TE", "TR", "W"),
        "wave_covariates": ("erbto",),
        "include_hearing": True,
    }
    assert plan.coupling_names() == {
        ("W", "TE"): "g_W_TE",
        ("TR", "TE"): "g_TR_TE",
        ("W", "TR"): "g_W_TR",
    }
    assert plan.lagged_names() == {("W", "TE"): "h_W"}
    assert plan.diagnostic_vars()[:8] == [
        "g_W_TE",
        "g_TR_TE",
        "g_W_TR",
        "h_W",
        "b_hs",
        "b_hs_missing",
        "b_erbto",
        "b_erbto_missing",
    ]


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (L.LcsmModelSettings(outcomes=("W",)), "at least two"),
        (
            L.LcsmModelSettings(
                outcomes=("W", "L"),
                couplings=(("X", ("L",)),),
            ),
            "target 'X' is not in outcomes",
        ),
        (
            L.LcsmModelSettings(
                outcomes=("W", "L"),
                couplings=(("W", ("X",)),),
            ),
            "outside outcomes",
        ),
        (
            L.LcsmModelSettings(
                outcomes=("W", "L"),
                couplings=(("W", ("W",)),),
            ),
            "cannot couple to itself",
        ),
        (
            L.LcsmModelSettings(
                outcomes=("W", "L"),
                lagged_change_couplings=(("W", ("L",)),),
            ),
            "requires arm_window_intercepts=True",
        ),
        (
            L.LcsmModelSettings(
                outcomes=("W", "L"),
                covariate_block=("hs",),
            ),
            "must be declared together",
        ),
        (
            L.LcsmModelSettings(
                outcomes=("W", "L"),
                covariate_block=("hs",),
                covariate_targets=("X",),
            ),
            "outside outcomes",
        ),
        (
            L.LcsmModelSettings(
                outcomes=("W", "L"),
                dominance_pair=("W", "L"),
            ),
            "reciprocal level couplings",
        ),
        (
            L.LcsmModelSettings(
                outcomes=("W", "L"),
                use_process_noise=False,
                shared_process_noise=True,
            ),
            "requires use_process_noise=True",
        ),
    ],
)
def test_resolve_rejects_cross_field_contradictions(settings, message):
    with pytest.raises(ValueError, match=message):
        L.resolve_lcsm_run_plan(_spec(settings=settings))


def test_outcome_symbol_must_be_loaded():
    with pytest.raises(ValueError, match="outcome_symbol 'W'.*not in outcomes"):
        L.resolve_lcsm_run_plan(
            _spec(settings=L.LcsmModelSettings(outcomes=("L", "E")))
        )


def test_wrong_typed_settings_class_is_rejected():
    from language_reading_predictors.statistical_models.survival import (
        SurvivalModelSettings,
    )

    with pytest.raises(TypeError, match="requires LcsmModelSettings"):
        L.resolve_lcsm_run_plan(_spec(settings=SurvivalModelSettings()))


def test_invalid_setting_fails_before_context_reset_or_data_loading(monkeypatch):
    from language_reading_predictors.statistical_models.pipelines import lcsm as P

    touched = {"context": False, "data": False}

    def _context(*args, **kwargs):
        touched["context"] = True
        raise AssertionError("make_context must not run")

    def _data(*args, **kwargs):
        touched["data"] = True
        raise AssertionError("load_wave_panel must not run")

    monkeypatch.setattr(P, "make_context", _context)
    monkeypatch.setattr(P, "load_wave_panel", _data)

    with pytest.raises(ValueError, match="unknown LCSM setting"):
        P.fit_lcsm(_spec(coupling_prior_sgima=0.3))
    assert touched == {"context": False, "data": False}


def test_reporting_dispatch_and_recipe_use_the_attached_plan(tmp_path):
    spec = _spec(settings=L.LcsmModelSettings())
    plan = L.resolve_lcsm_run_plan(spec)
    ctx = SimpleNamespace(spec=spec, resolved_plan=plan, output_dir=str(tmp_path))

    assert R._resolved_run_plan(ctx) is plan
    path = R.write_model_recipe(ctx)
    assert path is not None
    text = (tmp_path / "model_recipe.md").read_text(encoding="utf-8")
    assert "validated LCSM run plan" in text
    assert "Cross-process couplings are adjusted or exploratory associations" in text
    assert "zero-divergence convergence gate" in text


def test_pipeline_has_no_direct_lcsm_setting_reads():
    from language_reading_predictors.statistical_models.pipelines import lcsm as P

    source = inspect.getsource(P.fit_lcsm)
    assert "spec.extra" not in source
    assert "ctx.spec.extra" not in source


def test_registered_models_are_typed_and_preserve_the_legacy_contract():
    specs = _registered_specs()
    assert len(specs) == 5

    for registered in specs:
        settings = registered.model_settings
        assert isinstance(settings, L.LcsmModelSettings)
        assert set(registered.extra) <= {"target_accept"}
        typed = L.resolve_lcsm_run_plan(registered)
        legacy_settings = asdict(settings)
        legacy_settings["couplings"] = (
            dict(settings.couplings) if settings.couplings is not None else None
        )
        legacy_settings["lagged_change_couplings"] = dict(
            settings.lagged_change_couplings
        )
        legacy = L.resolve_lcsm_run_plan(
            ModelSpec(
                model_id=registered.model_id,
                kind="lcsm",
                title=registered.title,
                outcome_symbol=registered.outcome_symbol,
                study_id="rli",
                extra={**legacy_settings, **registered.extra},
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
