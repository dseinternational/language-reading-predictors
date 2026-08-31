# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the correlated-factor typed settings and run plan (#394)."""

from __future__ import annotations

import inspect
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from language_reading_predictors.statistical_models import corr_factor as C
from language_reading_predictors.statistical_models import reporting as R
from language_reading_predictors.statistical_models.context import ModelSpec


def _spec(
    *,
    settings=None,
    study_id: str = "rli",
    outcome_symbol: str | None = "W",
    **extra,
) -> ModelSpec:
    return ModelSpec(
        model_id=f"lrp-{study_id}-mm-999",
        kind="corr_factor",
        title="test correlated factor",
        outcome_symbol=outcome_symbol,
        study_id=study_id,
        family="corr_factor",
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
        ({"domains": ()}, ValueError, "at least one"),
        (
            {"domains": (("vocabulary", ("R",)), ("code", ("R",)))},
            ValueError,
            "only one domain",
        ),
        ({"structural_covariates": "blocks"}, TypeError, "sequence of strings"),
        ({"use_age": 1}, TypeError, "boolean or None"),
        ({"loading_prior": "pooled"}, ValueError, "communality.*free"),
        ({"loading_mu": float("inf")}, ValueError, "must be finite"),
        ({"loading_sigma": 0}, ValueError, "positive and finite"),
        ({"predictor_slope_sigma": True}, TypeError, "must be a number"),
        ({"post_time": 0}, ValueError, "at least 1"),
        ({"single_indicator_reliability": 1.0}, ValueError, r"in \(0, 1\)"),
        ({"lkj_eta": -1}, ValueError, "positive and finite"),
    ],
)
def test_settings_reject_invalid_values(kwargs, error, message):
    with pytest.raises(error, match=message):
        C.CorrFactorModelSettings(**kwargs)


def test_settings_reject_unknown_legacy_key():
    with pytest.raises(ValueError, match="unknown correlated-factor setting.*domians"):
        C.CorrFactorModelSettings.from_legacy_extra(
            {"domians": {}},
            model_id="lrp-rli-mm-999",
        )


def test_typed_settings_allow_only_global_extra_keys():
    plan = C.resolve_corr_factor_run_plan(
        _spec(settings=C.CorrFactorModelSettings(), target_accept=0.999)
    )
    assert plan.settings_source == "typed"

    with pytest.raises(ValueError, match="cannot be split.*use_age"):
        C.resolve_corr_factor_run_plan(
            _spec(settings=C.CorrFactorModelSettings(), use_age=False)
        )


def test_resolve_rejects_wrong_kind_study_and_port_outcomes():
    wrong = ModelSpec(model_id="x", kind="itt", title="x", outcome_symbol="W")
    with pytest.raises(ValueError, match="expected kind 'corr_factor'"):
        C.resolve_corr_factor_run_plan(wrong)

    with pytest.raises(ValueError, match="study_id must be 'rli' or 'rlm'"):
        C.resolve_corr_factor_run_plan(_spec(study_id="other"))

    with pytest.raises(ValueError, match="RLI corr_factor requires an outcome"):
        C.resolve_corr_factor_run_plan(_spec(outcome_symbol=None))

    with pytest.raises(ValueError, match="measurement-only.*outcome_symbol=None"):
        C.resolve_corr_factor_run_plan(_spec(study_id="rlm"))


def test_default_legacy_rli_plan_preserves_execution_contract():
    plan = C.resolve_corr_factor_run_plan(_spec())

    assert plan.settings_source == "legacy_extra"
    assert plan.port == "rli"
    assert plan.domain_mapping() == {
        "vocabulary": ("R", "E"),
        "code": ("L", "B"),
        "grammar": ("F", "T"),
    }
    assert plan.rli_prepare_kwargs() == {
        "phase_mode": "span",
        "post_time": 4,
        "outcomes": ("W", "R", "E", "L", "B", "F", "T"),
        "covariates": ("blocks",),
    }
    assert plan.rli_factory_kwargs() == {
        "outcome_symbol": "W",
        "domains": plan.domain_mapping(),
        "structural_covariates": ("blocks",),
        "structural_factors": None,
        "use_group": False,
        "use_age": True,
        "loading_prior": "communality",
        "comm_alpha": 2.0,
        "comm_beta": 2.0,
        "loading_mu": 0.0,
        "loading_sigma": 1.0,
        "residual_sigma": 1.0,
        "predictor_slope_sigma": 0.3,
        "focal_slope_sigma": None,
        "lkj_eta": 2.0,
    }
    assert plan.observation_nodes == ("Z_obs", "y_post")
    assert plan.compute_loo is False


@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize(
    ("loading_prior", "knob", "message"),
    [
        ("communality", "loading_mu", "only apply to loading_prior='free'"),
        ("free", "comm_alpha", "only apply to loading_prior='communality'"),
    ],
)
def test_resolve_rejects_inactive_loading_knobs(
    typed, loading_prior, knob, message
):
    values = {"loading_prior": loading_prior, knob: 0.5}
    spec = (
        _spec(settings=C.CorrFactorModelSettings(**values))
        if typed
        else _spec(**values)
    )
    with pytest.raises(ValueError, match=message):
        C.resolve_corr_factor_run_plan(spec)


def test_rli_plan_rejects_invalid_domains_factors_and_rlm_settings():
    with pytest.raises(ValueError, match="require at least two indicators"):
        C.resolve_corr_factor_run_plan(
            _spec(
                settings=C.CorrFactorModelSettings(
                    domains=(("code", ("L",)),),
                )
            )
        )

    with pytest.raises(ValueError, match="not fitted domains.*decoding"):
        C.resolve_corr_factor_run_plan(
            _spec(
                settings=C.CorrFactorModelSettings(
                    structural_factors=("decoding",),
                )
            )
        )

    with pytest.raises(ValueError, match="RLM-only settings.*wave"):
        C.resolve_corr_factor_run_plan(
            _spec(settings=C.CorrFactorModelSettings(wave=3))
        )


def test_rlm_plan_preserves_measurement_only_contract():
    plan = C.resolve_corr_factor_run_plan(
        _spec(
            study_id="rlm",
            outcome_symbol=None,
            settings=C.CorrFactorModelSettings(
                domains=(
                    ("reading", ("basread", "basspel")),
                    ("memory", ("basdig",)),
                ),
                wave=3,
                single_indicator_reliability=0.8,
            ),
        )
    )

    assert plan.port == "rlm"
    assert plan.rlm_prepare_kwargs() == {
        "wave": 3,
        "measure_symbols": ("basread", "basspel", "basdig"),
    }
    assert plan.rlm_factory_kwargs() == {
        "domains": plan.domain_mapping(),
        "single_indicator_reliability": 0.8,
        "comm_alpha": 2.0,
        "comm_beta": 2.0,
        "lkj_eta": 2.0,
    }
    assert plan.diagnostic_vars() == [
        "lambda_free",
        "sigma_free",
        "factor_corr_pairs",
    ]
    assert plan.observation_nodes == ("Z_obs",)


def test_rlm_plan_rejects_rli_only_settings_and_legacy_study_conflict():
    with pytest.raises(ValueError, match="RLI-only settings.*use_age"):
        C.resolve_corr_factor_run_plan(
            _spec(
                study_id="rlm",
                outcome_symbol=None,
                settings=C.CorrFactorModelSettings(use_age=True),
            )
        )

    conflict = _spec(study_id="rlm", outcome_symbol=None)
    conflict.extra["study_id"] = "rli"
    with pytest.raises(ValueError, match="legacy study_id.*conflicts"):
        C.resolve_corr_factor_run_plan(conflict)


def test_active_covariates_update_factory_diagnostics_and_serialisation():
    plan = C.resolve_corr_factor_run_plan(
        _spec(
            settings=C.CorrFactorModelSettings(
                structural_covariates=("hs", "erbto_missing"),
                use_group=True,
            )
        )
    )
    active = plan.with_active_structural_covariates(("hs",))

    assert active.rli_factory_kwargs()["structural_covariates"] == ("hs",)
    assert "beta_hs" in active.diagnostic_vars()
    assert "beta_erbto_missing" not in active.diagnostic_vars()
    assert "beta_G" in active.diagnostic_vars()
    assert active.as_dict()["active_structural_covariates"] == ("hs",)


def test_wrong_typed_settings_class_is_rejected():
    from language_reading_predictors.statistical_models.survival import (
        SurvivalModelSettings,
    )

    with pytest.raises(TypeError, match="requires CorrFactorModelSettings"):
        C.resolve_corr_factor_run_plan(_spec(settings=SurvivalModelSettings()))


@pytest.mark.parametrize("port", ["rli", "rlm"])
def test_invalid_setting_fails_before_context_reset_or_data_loading(monkeypatch, port):
    from language_reading_predictors.statistical_models.pipelines import corr_factor as P

    touched = {"context": False, "data": False}

    def _context(*args, **kwargs):
        touched["context"] = True
        raise AssertionError("make_context must not run")

    def _data(*args, **kwargs):
        touched["data"] = True
        raise AssertionError("data loading must not run")

    monkeypatch.setattr(P, "make_context", _context)
    monkeypatch.setattr(P, "load_and_prepare", _data)
    spec = _spec(
        study_id=port,
        outcome_symbol=None if port == "rlm" else "W",
        domians={},
    )
    fit = P.fit_rlm_corr_factor if port == "rlm" else P.fit_correlated_factor

    with pytest.raises(ValueError, match="unknown correlated-factor setting"):
        fit(spec)
    assert touched == {"context": False, "data": False}


def test_reporting_dispatch_and_recipe_use_attached_plan(tmp_path):
    spec = _spec(settings=C.CorrFactorModelSettings())
    plan = C.resolve_corr_factor_run_plan(spec)
    ctx = SimpleNamespace(spec=spec, resolved_plan=plan, output_dir=str(tmp_path))

    assert R._resolved_run_plan(ctx) is plan
    path = R.write_model_recipe(ctx)
    assert path is not None
    text = (tmp_path / "model_recipe.md").read_text(encoding="utf-8")
    assert "validated correlated-factor run plan" in text
    assert "Descriptive measurement associations only" in text
    assert "Active structural covariates: blocks" in text


def test_pipelines_do_not_read_family_settings_from_extra():
    from language_reading_predictors.statistical_models.pipelines import corr_factor as P

    assert "spec.extra" not in inspect.getsource(P.fit_correlated_factor)
    assert "spec.extra" not in inspect.getsource(P.fit_rlm_corr_factor)


def test_registered_specs_are_typed_and_match_legacy_contracts():
    from language_reading_predictors.statistical_models.lrp_rli_mm_001 import (
        SPEC as RLI_001,
    )
    from language_reading_predictors.statistical_models.lrp_rli_mm_002 import (
        SPEC as RLI_002,
    )
    from language_reading_predictors.statistical_models.lrp_rli_mm_101 import (
        SPEC as RLI_101,
    )
    from language_reading_predictors.statistical_models.lrp_rli_mm_102 import (
        SPEC as RLI_102,
    )
    from language_reading_predictors.statistical_models.lrp_rlm_mm_001 import (
        SPEC as RLM_001,
    )

    for spec in (RLI_001, RLI_002, RLI_101, RLI_102, RLM_001):
        assert isinstance(spec.model_settings, C.CorrFactorModelSettings)
        # #637 stage 2: the sampler knob moved to the first-class
        # ``ModelSpec.target_accept``, so a registered spec carries no ``extra``
        # at all. The legacy adapter still accepts it, which is what the
        # comparison below exercises.
        assert spec.extra == {}
        assert spec.target_accept is not None
        typed = C.resolve_corr_factor_run_plan(spec)
        legacy_extra = asdict(spec.model_settings)
        legacy_extra["target_accept"] = spec.target_accept
        legacy = C.resolve_corr_factor_run_plan(
            ModelSpec(
                model_id=spec.model_id,
                kind=spec.kind,
                title=spec.title,
                outcome_symbol=spec.outcome_symbol,
                study_id=spec.study_id,
                extra=legacy_extra,
            )
        )
        assert typed.settings_source == "typed"
        assert legacy.settings_source == "legacy_extra"
        assert typed.domain_mapping() == legacy.domain_mapping()
        assert typed.diagnostic_vars() == legacy.diagnostic_vars()
        if typed.port == "rli":
            assert typed.rli_prepare_kwargs() == legacy.rli_prepare_kwargs()
            assert typed.rli_factory_kwargs() == legacy.rli_factory_kwargs()
        else:
            assert typed.rlm_prepare_kwargs() == legacy.rlm_prepare_kwargs()
            assert typed.rlm_factory_kwargs() == legacy.rlm_factory_kwargs()


# --- 2026-08-21 review fixes --------------------------------------------------


def test_declared_empty_structural_covariates_stay_empty():
    """Finding 8: a declared-empty covariate set is the natural spelling of an
    unadjusted structural leg and must not silently become blocks-adjusted."""
    defaulted = C.resolve_corr_factor_run_plan(_spec())
    assert "blocks" in defaulted.structural_covariates
    plan = C.resolve_corr_factor_run_plan(
        _spec(settings=C.CorrFactorModelSettings(structural_covariates=()))
    )
    assert plan.structural_covariates == ()


def test_factor_z_stays_out_of_the_curated_diagnostics():
    """Finding 3: the 153-element non-centred offset destroyed the trace/posterior
    figures and drowned psense; the gate covers it via the all-free-RV widening."""
    plan = C.resolve_corr_factor_run_plan(_spec())
    assert "factor_z" not in plan.diagnostic_vars()


def test_rlm_single_domain_diag_list_omits_factor_corr_pairs():
    """Finding 10: a single-domain declaration builds no factor_corr_pairs node,
    so the curated list must not name it."""
    plan = C.resolve_corr_factor_run_plan(
        _spec(
            study_id="rlm",
            outcome_symbol=None,
            settings=C.CorrFactorModelSettings(
                domains=(("memory", ("basdig",)),),
                wave=3,
                single_indicator_reliability=0.8,
            ),
        )
    )
    assert plan.diagnostic_vars() == ["lambda_free", "sigma_free"]
