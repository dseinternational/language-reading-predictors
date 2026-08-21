# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for typed joint-mechanism settings and run plans (#394 pillar 4)."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from language_reading_predictors.statistical_models import joint_mechanism as J
from language_reading_predictors.statistical_models import reporting as R
from language_reading_predictors.statistical_models.context import ModelSpec

_MODULES = (
    "language_reading_predictors.statistical_models.lrp_rli_jm_001",
    "language_reading_predictors.statistical_models.lrp_rli_jm_002",
)
_META_FIELDS = (
    "design_description",
    "estimand",
    "causal_status",
    "analysis_population",
    "missing_data_assumption",
)


def _spec(
    *,
    settings=None,
    mechanism_symbol: str | None = "L",
    study_id: str = "rli",
    **extra,
) -> ModelSpec:
    return ModelSpec(
        model_id="lrp-rli-jm-999",
        kind="joint_mechanism",
        title="test joint mechanism",
        mechanism_symbol=mechanism_symbol,
        study_id=study_id,
        family="joint_mechanism",
        estimand_type="association",
        causal_status="none",
        model_settings=settings,
        extra=extra,
    )


def _registered_specs() -> list[ModelSpec]:
    return [importlib.import_module(name).SPEC for name in _MODULES]


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"design": "other"}, ValueError, "levels.*transition"),
        ({"outcome_symbols": "WN"}, TypeError, "sequence of strings"),
        (
            {"outcome_symbols": ("W", "W")},
            ValueError,
            "duplicate symbols",
        ),
        ({"contrast": ("N", "")}, TypeError, "non-empty strings"),
        ({"confounder_symbols": ("G", "G")}, ValueError, "duplicate"),
        ({"include_group": 1}, TypeError, "must be a boolean"),
        ({"covariates": ("hs", "hs")}, ValueError, "duplicate"),
        ({"adjust_for": "hs"}, TypeError, "sequence of strings"),
        ({"predictor_slope_sigma": 0}, ValueError, "positive finite"),
    ],
)
def test_settings_reject_invalid_values(kwargs, error, message):
    with pytest.raises(error, match=message):
        J.JointMechanismModelSettings(**kwargs)


def test_settings_reject_unknown_legacy_key():
    with pytest.raises(
        ValueError,
        match="unknown joint-mechanism setting.*predictor_slope_sgima",
    ):
        J.JointMechanismModelSettings.from_legacy_extra(
            {"predictor_slope_sgima": 0.3},
            model_id="lrp-rli-jm-999",
        )


def test_typed_settings_allow_only_global_extra_keys():
    plan = J.resolve_joint_mechanism_run_plan(
        _spec(settings=J.JointMechanismModelSettings(), target_accept=0.99)
    )
    assert plan.settings_source == "typed"

    with pytest.raises(ValueError, match="cannot be split.*design"):
        J.resolve_joint_mechanism_run_plan(
            _spec(settings=J.JointMechanismModelSettings(), design="transition")
        )


def test_resolve_rejects_wrong_kind_and_study():
    wrong = ModelSpec(model_id="x", kind="itt", title="x")
    with pytest.raises(ValueError, match="expected kind 'joint_mechanism'"):
        J.resolve_joint_mechanism_run_plan(wrong)

    with pytest.raises(ValueError, match="requires study_id='rli'"):
        J.resolve_joint_mechanism_run_plan(_spec(study_id="rlm"))


def test_default_levels_plan_preserves_execution_contract():
    plan = J.resolve_joint_mechanism_run_plan(_spec())

    assert plan.settings_source == "legacy_extra"
    assert plan.design == "levels"
    assert plan.prepare_kwargs() == {
        "phase_mode": "levels",
        "outcomes": ("W", "N", "L"),
        "baseline_covariates": (),
    }
    assert plan.factory_kwargs() == {
        "design": "levels",
        "mechanism_symbol": "L",
        "outcome_symbols": ("W", "N"),
        "contrast": ("N", "W"),
        "adjust_for": (),
        "confounder_symbols": ("G", "A"),
        "include_group": True,
        "predictor_slope_sigma": 0.3,
    }
    available = {
        "alpha",
        "beta_mech",
        "delta_ls_decoding",
        "beta_group_nuisance",
        "gamma_A",
        "sigma_u_resid",
        "rho_outcome",
        "beta_mech_focal_given_held",
        "share_retained",
    }
    assert plan.diagnostic_vars(available) == [
        "alpha",
        "beta_mech",
        "delta_ls_decoding",
        "beta_group_nuisance",
        "gamma_A",
        "sigma_u_resid",
        "rho_outcome",
        "beta_mech_focal_given_held",
        "share_retained",
    ]
    assert plan.psense_vars(available) == [
        "beta_mech",
        "delta_ls_decoding",
        "rho_outcome",
        "beta_mech_focal_given_held",
    ]
    assert plan.likelihood == "binomial"
    assert plan.min_wave_rows == 10
    # Saturated per-child residual: PSIS-LOO is not computed for this design
    # (2026-08-21 joint-mechanism review, finding 2).
    assert plan.compute_loo is False


def test_transition_plan_owns_loader_factory_and_diagnostics():
    plan = J.resolve_joint_mechanism_run_plan(
        _spec(
            settings=J.JointMechanismModelSettings(
                design="transition",
                adjust_for=(
                    "hs",
                    "hs_missing",
                    "attend",
                    "deapp_c",
                    "deapp_c_missing",
                ),
            )
        )
    )

    assert plan.prepare_kwargs() == {
        "phase_mode": "all",
        "outcomes": ("W", "N", "L"),
        "covariates": ("attend",),
        "post_covariates": ("hs", "hs_missing", "deapp_c", "deapp_c_missing"),
        # Only the two fitted baselines are required — without this the loader's
        # default would also demand the mechanism's unused period-start score.
        "pre_required": ("W", "N"),
    }
    assert plan.factory_kwargs() == {
        "design": "transition",
        "mechanism_symbol": "L",
        "outcome_symbols": ("W", "N"),
        "contrast": ("N", "W"),
        "adjust_for": (
            "hs",
            "hs_missing",
            "attend",
            "deapp_c",
            "deapp_c_missing",
        ),
        "confounder_symbols": ("G", "A"),
        "include_group": True,
    }
    available = {
        "alpha",
        "beta_mech",
        "delta_ls_decoding",
        "beta_G",
        "gamma_A",
        "gamma_hs",
        "gamma_own",
        "alpha_phase",
        "kappa",
        "sigma_u_child",
        "rho_outcome",
    }
    assert plan.diagnostic_vars(available) == [
        "alpha",
        "beta_mech",
        "delta_ls_decoding",
        "beta_G",
        "gamma_A",
        "gamma_hs",
        "gamma_own",
        "alpha_phase",
        "kappa",
        "sigma_u_child",
        "rho_outcome",
    ]
    assert plan.psense_vars(available) == [
        "beta_mech",
        "delta_ls_decoding",
        "rho_outcome",
    ]
    assert plan.likelihood == "beta_binomial"
    assert plan.min_wave_rows is None
    # Child-level dependence over three transitions: leave-one-child-out LOO is
    # meaningful here (via the factory's ``loo_child_idx`` map) and stays on.
    assert plan.compute_loo is True
    assert plan.loo_unit == "child"


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (
            J.JointMechanismModelSettings(outcome_symbols=("W",)),
            "exactly two outcome_symbols",
        ),
        (
            J.JointMechanismModelSettings(contrast=("W", "E")),
            "two outcome_symbols exactly once",
        ),
        (
            J.JointMechanismModelSettings(confounder_symbols=("G", "GA")),
            "supports only.*G/A",
        ),
        (
            J.JointMechanismModelSettings(adjust_for=("hs",)),
            "transition-only",
        ),
        (
            J.JointMechanismModelSettings(
                design="transition",
                covariates=("blocks",),
            ),
            "levels-only",
        ),
        (
            J.JointMechanismModelSettings(
                design="transition",
                predictor_slope_sigma=0.3,
            ),
            "predictor_slope_sigma is levels-only",
        ),
    ],
)
def test_resolve_rejects_cross_field_contradictions(settings, message):
    with pytest.raises(ValueError, match=message):
        J.resolve_joint_mechanism_run_plan(_spec(settings=settings))


def test_mechanism_must_differ_from_outcomes():
    with pytest.raises(ValueError, match="must differ"):
        J.resolve_joint_mechanism_run_plan(
            _spec(
                mechanism_symbol="W",
                settings=J.JointMechanismModelSettings(),
            )
        )


def test_wrong_typed_settings_class_is_rejected():
    from language_reading_predictors.statistical_models.survival import (
        SurvivalModelSettings,
    )

    with pytest.raises(TypeError, match="requires JointMechanismModelSettings"):
        J.resolve_joint_mechanism_run_plan(_spec(settings=SurvivalModelSettings()))


def test_active_adjustment_cannot_add_an_undeclared_term():
    plan = J.resolve_joint_mechanism_run_plan(
        _spec(
            settings=J.JointMechanismModelSettings(
                covariates=("blocks", "hs"),
            )
        )
    )
    assert plan.with_active_adjustment(("blocks",)).active_adjustment == ("blocks",)
    with pytest.raises(ValueError, match="was not declared"):
        plan.with_active_adjustment(("blocks", "behav"))


def test_invalid_setting_fails_before_context_reset_or_data_loading(monkeypatch):
    from language_reading_predictors.statistical_models.pipelines import (
        joint_mechanism as P,
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

    with pytest.raises(ValueError, match="unknown joint-mechanism setting"):
        P.fit_joint_mechanism(_spec(predictor_slope_sgima=0.3))
    assert touched == {"context": False, "data": False}


def test_reporting_dispatch_and_recipe_use_the_attached_plan(tmp_path):
    spec = _spec(settings=J.JointMechanismModelSettings())
    plan = J.resolve_joint_mechanism_run_plan(spec)
    ctx = SimpleNamespace(spec=spec, resolved_plan=plan, output_dir=str(tmp_path))

    assert R._resolved_run_plan(ctx) is plan
    path = R.write_model_recipe(ctx)
    assert path is not None
    text = (tmp_path / "model_recipe.md").read_text(encoding="utf-8")
    assert "validated joint-mechanism run plan" in text
    assert "Adjusted association only" in text
    assert "zero-divergence convergence gate" in text
    # The default (levels) plan computes no PSIS-LOO and the recipe says why.
    assert "PSIS-LOO is not computed" in text
    assert "saturated per-child latent" in text


def test_pipeline_has_no_direct_joint_mechanism_setting_reads():
    from language_reading_predictors.statistical_models.pipelines import (
        joint_mechanism as P,
    )

    source = inspect.getsource(P)
    assert "spec.extra" not in source
    assert "ctx.spec.extra" not in source


def test_registered_models_are_typed_and_preserve_the_legacy_contract():
    specs = _registered_specs()
    assert len(specs) == 2
    assert {spec.model_settings.design for spec in specs} == {"levels", "transition"}

    for registered in specs:
        settings = registered.model_settings
        assert isinstance(settings, J.JointMechanismModelSettings)
        assert set(registered.extra) <= {"target_accept"}
        typed = J.resolve_joint_mechanism_run_plan(registered)
        legacy = J.resolve_joint_mechanism_run_plan(
            ModelSpec(
                model_id=registered.model_id,
                kind="joint_mechanism",
                title=registered.title,
                mechanism_symbol=registered.mechanism_symbol,
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
