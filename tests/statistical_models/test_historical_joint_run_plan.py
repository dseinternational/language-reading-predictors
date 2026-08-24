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
    "dispersion_prior_sigma": 0.25,
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


def _within_registered_spec() -> ModelSpec:
    return importlib.import_module("language_reading_predictors.statistical_models.lrp_rlm_jc_002").SPEC


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
        ({"dispersion_prior_sigma": float("inf")}, "positive finite"),
        ({"lkj_eta": -1.0}, "positive finite"),
        ({"sigma_within_prior_sigma": 0.0}, "positive finite"),
        ({"within_lkj_eta": float("nan")}, "positive finite"),
        ({"within_correlation": "yes"}, "must be a boolean"),
        (
            {"within_correlation": True, "extension_waves": (4,)},
            "balanced complete-case window",
        ),
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
        "dispersion_prior_sigma": default_of(build_rlm_joint_growth_model, "dispersion_prior_sigma"),
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
        "dispersion_prior_sigma": 0.25,
        "lkj_eta": 2.0,
    }
    for field in _META_FIELDS:
        assert isinstance(typed.as_dict()[field], str) and typed.as_dict()[field]


def test_registered_within_companion_resolves_balanced_dynamic_contract():
    registered = _within_registered_spec()
    assert isinstance(registered.model_settings, HJ.HistoricalJointModelSettings)
    assert registered.extra == {}

    plan = HJ.resolve_historical_joint_run_plan(registered)

    assert plan.within_correlation is True
    assert plan.prepare_kwargs() == {
        "waves": (1, 2, 3),
        "complete_case": True,
        "extension_waves": (),
    }
    assert plan.factory_kwargs() == {
        "measures": ("basread", "bpvs", "basdig"),
        "eta_prior_sigma": 1.5,
        "sigma_subject_prior_sigma": 1.0,
        "lkj_eta": 2.0,
        "within_correlation": True,
        "sigma_within_prior_sigma": 0.5,
        "within_lkj_eta": 2.0,
    }
    assert plan.diagnostic_vars() == [
        "eta_cell",
        "sigma_subject",
        "sigma_within",
        "within_corr_pairs",
        "measure_corr_pairs",
    ]
    assert plan.dispersion_prior_sigma is None
    assert plan.likelihood == "logistic_normal_binomial"
    assert "within-child correlation matrix" in plan.estimand
    assert "Extension waves are excluded" in plan.analysis_population
    assert plan.causal_status.startswith("Descriptive only")


# --- 2026-08-21 historical-families code review -----------------------------


def _review_spec(**settings) -> ModelSpec:
    return ModelSpec(
        model_id="lrp-rlm-jc-test",
        kind="historical_joint",
        title="test",
        outcome_symbol=None,
        study_id="rlm",
        model_settings=HJ.HistoricalJointModelSettings(**settings),
    )


def test_plan_nulls_every_prior_scale_the_fitted_model_lacks():
    """Finding 10: config.json must not name a prior the posterior lacks.

    The plan already nulled the concentration scale for the within-child branch but
    kept live within-child scales for the between-child model, which has neither
    of them.
    """
    between = HJ.resolve_historical_joint_run_plan(_review_spec(within_correlation=False))
    assert between.dispersion_prior_sigma == 0.25
    assert between.sigma_within_prior_sigma is None
    assert between.within_lkj_eta is None

    within = HJ.resolve_historical_joint_run_plan(
        _review_spec(within_correlation=True, extension_waves=())
    )
    assert within.dispersion_prior_sigma is None
    assert within.sigma_within_prior_sigma == 0.5
    assert within.within_lkj_eta == 2.0

    # The factory is fed only the arguments its branch actually takes.
    assert "dispersion_prior_sigma" not in within.factory_kwargs()
    assert "sigma_within_prior_sigma" not in between.factory_kwargs()


def test_a_kappa_scale_is_rejected_on_the_within_child_branch():
    """Finding 10: an explicitly-declared setting must never be silently discarded."""
    with pytest.raises(ValueError, match="no effect when within_correlation is\\s+true"):
        HJ.HistoricalJointModelSettings(
            within_correlation=True, dispersion_prior_sigma=0.5
        )
    # The default is not a declaration, so it stays acceptable.
    HJ.HistoricalJointModelSettings(within_correlation=True)


def test_declared_waves_are_checked_against_the_measure_catalogue():
    """Finding 10: the joint resolver checks every measure's available window."""
    with pytest.raises(ValueError, match="no data at waves"):
        HJ.resolve_historical_joint_run_plan(
            _review_spec(measures=("basread", "basmat"), waves=(1, 2, 3))
        )
    # basmat is wave-3+ only; a window inside it resolves.
    HJ.resolve_historical_joint_run_plan(
        _review_spec(measures=("basread", "basmat"), waves=(3, 4), extension_waves=(5,))
    )


def test_within_child_estimand_records_the_attenuation_and_scale_limits():
    """Finding 6: the structural limits belong in the estimand of record.

    The within-child residual carries the measurement noise (no Beta-Binomial
    concentration term in that branch) and the double centring shrinks the
    realised departures below ``sigma_within``. Both reach ``model_recipe.md``
    and ``config.json`` through the plan's estimand string.
    """
    plan = HJ.resolve_historical_joint_run_plan(_review_spec(within_correlation=True))
    estimand = plan.estimand.lower()
    assert "measurement noise" in estimand
    assert "attenuates" in estimand
    assert "sigma_within" in estimand
    # The between-child branch makes no such claim: it fits kappa separately.
    between = HJ.resolve_historical_joint_run_plan(_review_spec(within_correlation=False))
    assert "measurement noise" not in between.estimand.lower()


def test_registered_joint_models_declare_windows_their_measures_support():
    """Every registered declaration resolves under the new catalogue check."""
    for model_id in ("lrp-rlm-jc-001", "lrp-rlm-jc-002"):
        module = importlib.import_module(
            "language_reading_predictors.statistical_models."
            + model_id.replace("-", "_")
        )
        plan = HJ.resolve_historical_joint_run_plan(module.SPEC)
        assert plan.model_id == model_id


def test_the_registered_prior_companion_constant_matches_the_modules():
    """``HISTORICAL_JOINT_PRIOR_COMPANIONS`` is a second source of truth for a
    release control, so it must not drift from the modules that own the pairing
    (2026-08-23 joint audit, finding 5). The companion is built from its parent's
    frozen settings with ``dataclasses.replace``, and the *only* thing it may
    change is the within-scale prior — otherwise it varies more than the
    sensitivity it claims to be."""
    for parent_id, companion_id in HJ.HISTORICAL_JOINT_PRIOR_COMPANIONS.items():
        parent = importlib.import_module(
            "language_reading_predictors.statistical_models."
            + parent_id.replace("-", "_")
        ).SPEC
        companion = importlib.import_module(
            "language_reading_predictors.statistical_models."
            + companion_id.replace("-", "_")
        ).SPEC
        assert companion.model_id == companion_id
        assert companion.kind == parent.kind == "historical_joint"
        parent_plan = HJ.resolve_historical_joint_run_plan(parent).as_dict()
        companion_plan = HJ.resolve_historical_joint_run_plan(companion).as_dict()
        differing = {
            key
            for key in parent_plan
            if parent_plan[key] != companion_plan[key]
        }
        assert differing == {"model_id", "sigma_within_prior_sigma"}
        assert parent_plan["within_correlation"] is True
        assert (
            companion_plan["sigma_within_prior_sigma"]
            > parent_plan["sigma_within_prior_sigma"]
        )
