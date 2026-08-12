# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the RLI and RLM adjusted-association typed run plan (#394)."""

from __future__ import annotations

import inspect
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from language_reading_predictors.statistical_models import reporting as R
from language_reading_predictors.statistical_models.adjusted import (
    AdjustedModelSettings,
    AdjustedRunPlan,
    resolve_adjusted_run_plan,
)
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.lrp_rli_adj_065 import get_spec
from language_reading_predictors.statistical_models.lrp_rlm_adj_001 import (
    SPEC as RLM_SPEC,
)
from language_reading_predictors.statistical_models.lrp_rlm_adj_002 import (
    SPEC as RLM_DS_SPEC,
)
from language_reading_predictors.statistical_models.pipelines import (
    adjusted as pipeline,
)


def _spec(
    *,
    study_id: str = "rli",
    settings: object | None = None,
    extra: dict | None = None,
) -> ModelSpec:
    return ModelSpec(
        model_id=f"lrp-{study_id}-adj-test",
        kind="adjusted",
        title="typed adjusted test",
        outcome_symbol="W" if study_id == "rli" else "basread",
        study_id=study_id,
        model_settings=settings,
        extra=extra or {},
    )


def test_registered_adjusted_specs_are_typed_and_resolve_both_ports():
    rli = get_spec()
    assert isinstance(rli.model_settings, AdjustedModelSettings)
    assert isinstance(RLM_SPEC.model_settings, AdjustedModelSettings)
    assert isinstance(RLM_DS_SPEC.model_settings, AdjustedModelSettings)
    assert rli.extra == RLM_SPEC.extra == RLM_DS_SPEC.extra == {}
    rli_plan = resolve_adjusted_run_plan(rli)
    rlm_plan = resolve_adjusted_run_plan(RLM_SPEC)
    rlm_ds_plan = resolve_adjusted_run_plan(RLM_DS_SPEC)
    assert (rli_plan.port, rlm_plan.port) == ("rli", "rlm")
    assert rlm_ds_plan.port == "rlm"
    assert rli_plan.settings_source == rlm_plan.settings_source == "typed"
    assert rlm_ds_plan.settings_source == "typed"
    assert rlm_ds_plan.group_codes == (1,)
    assert rlm_ds_plan.predictor_measures == ("basdig", "bpvs", "bassim")
    assert rlm_ds_plan.use_age_predictor is False


@pytest.mark.parametrize(
    ("study_id", "settings", "extra"),
    [
        (
            "rli",
            AdjustedModelSettings(
                design="between_child",
                post_time=4,
                predictor_symbols=("L", "B"),
                language_composite_symbols=("R", "E", "F"),
                covariates=("blocks", "behav"),
                ses_covariates=("mumedupost16",),
            ),
            {
                "design": "between_child",
                "post_time": 4,
                "predictor_symbols": ("L", "B"),
                "language_composite_symbols": ("R", "E", "F"),
                "covariates": ("blocks", "behav"),
                "ses_covariates": ("mumedupost16",),
                "use_age_predictor": True,
                "predictor_slope_sigma": 0.3,
                "prior_sensitivity_sigmas": (0.5, 0.7),
            },
        ),
        (
            "rlm",
            AdjustedModelSettings(
                predictor_measures=("bpvs", "trog"),
                pre_wave=1,
                post_wave=3,
            ),
            {
                "study_id": "rlm",
                "predictor_measures": ("bpvs", "trog"),
                "use_age_predictor": True,
                "pre_wave": 1,
                "post_wave": 3,
                "predictor_slope_sigma": 0.3,
                "prior_sensitivity_sigmas": (0.5, 0.7),
            },
        ),
    ],
)
def test_typed_and_legacy_declarations_resolve_identically(
    study_id, settings, extra
):
    typed = resolve_adjusted_run_plan(_spec(study_id=study_id, settings=settings))
    legacy = resolve_adjusted_run_plan(_spec(study_id=study_id, extra=extra))
    assert asdict(typed) == {**asdict(legacy), "settings_source": "typed"}


def test_rli_plan_maps_loader_factory_and_diagnostic_contracts():
    plan = resolve_adjusted_run_plan(
        _spec(
            settings=AdjustedModelSettings(
                predictor_symbols=("L", "B"),
                language_composite_symbols=("R", "E"),
                covariates=("blocks", "hs_missing"),
                ses_covariates=("mumedupost16",),
                use_age_predictor=False,
            )
        )
    )
    assert plan.rli_prepare_kwargs() == {
        "phase_mode": "span",
        "post_time": 4,
        "outcomes": ("W", "L", "B", "R", "E"),
        "covariates": ("blocks", "hs_missing"),
    }
    assert plan.headline_predictors() == ("L", "B", "lang", "blocks", "hs_missing")
    factory = plan.rli_factory_kwargs()
    assert factory["predictors"] == plan.headline_predictors()
    assert factory["predictor_slope_sigma"] == 0.3
    assert plan.diagnostic_vars(("L",)) == [
        "alpha",
        "gamma_own",
        "kappa",
        "beta_L",
    ]


def test_rlm_plan_maps_loader_and_factory_contracts():
    plan = resolve_adjusted_run_plan(
        _spec(
            study_id="rlm",
            settings=AdjustedModelSettings(
                predictor_measures=("bpvs", "trog"),
                use_age_predictor=False,
                pre_wave=1,
                post_wave=3,
            ),
        )
    )
    assert plan.rlm_prepare_kwargs() == {
        "outcome": "basread",
        "predictor_measures": ("bpvs", "trog"),
        "include_age": False,
        "pre_wave": 1,
        "post_wave": 3,
        "group_codes": None,
    }
    assert plan.rlm_factory_kwargs(("bpvs",)) == {
        "predictors": ("bpvs",),
        "predictor_slope_sigma": 0.3,
    }


def test_rlm_group_subset_is_validated_and_propagated_before_io():
    plan = resolve_adjusted_run_plan(RLM_DS_SPEC)
    assert plan.rlm_prepare_kwargs()["group_codes"] == (1,)
    assert "Down syndrome" in plan.analysis_population
    recipe = plan.recipe_markdown(title="DS-only adjusted")
    assert "group code(s) 1" in recipe
    assert "No group nuisance term" in recipe
    assert "SES sensitivity" not in recipe

    with pytest.raises(ValueError, match="unknown RLM group_codes"):
        resolve_adjusted_run_plan(
            _spec(
                study_id="rlm",
                settings=AdjustedModelSettings(group_codes=(99,)),
            )
        )


def test_active_covariates_are_recorded_and_drive_ses_loader():
    plan = resolve_adjusted_run_plan(
        _spec(
            settings=AdjustedModelSettings(
                covariates=("blocks", "hs_missing"),
                ses_covariates=("mumedupost16",),
            )
        )
    )
    active = plan.with_active_covariates(("blocks",))
    assert active.declared_covariates == ("blocks", "hs_missing")
    assert active.active_covariates == ("blocks",)
    assert active.rli_prepare_kwargs(include_ses=True)["covariates"] == (
        "blocks",
        "mumedupost16",
    )
    with pytest.raises(ValueError, match="not declared"):
        plan.with_active_covariates(("behav",))


@pytest.mark.parametrize(
    ("study_id", "settings", "message"),
    [
        ("rli", AdjustedModelSettings(pre_wave=1), "RLM-only"),
        ("rli", AdjustedModelSettings(group_codes=(1,)), "RLM-only"),
        (
            "rlm",
            AdjustedModelSettings(predictor_symbols=("L",)),
            "RLI-only",
        ),
        (
            "rlm",
            AdjustedModelSettings(pre_wave=3, post_wave=1),
            "later",
        ),
    ],
)
def test_port_and_wave_constraints_fail_early(study_id, settings, message):
    with pytest.raises(ValueError, match=message):
        resolve_adjusted_run_plan(_spec(study_id=study_id, settings=settings))


def test_unknown_split_and_wrong_settings_type_are_rejected():
    with pytest.raises(ValueError, match="unknown adjusted setting"):
        resolve_adjusted_run_plan(_spec(extra={"predictrs": ("L",)}))
    with pytest.raises(ValueError, match="cannot be split"):
        resolve_adjusted_run_plan(
            _spec(
                settings=AdjustedModelSettings(),
                extra={"post_time": 4},
            )
        )
    with pytest.raises(TypeError, match="AdjustedModelSettings"):
        resolve_adjusted_run_plan(_spec(settings=object()))


def test_wrong_port_entrypoint_fails_before_context_or_data(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "make_context",
        lambda *args, **kwargs: pytest.fail("context created before validation"),
    )
    monkeypatch.setattr(
        pipeline,
        "load_and_prepare",
        lambda *args, **kwargs: pytest.fail("data loaded before validation"),
    )
    with pytest.raises(ValueError, match="RLM settings"):
        pipeline.fit_adjusted(RLM_SPEC)


def test_reporting_reuses_attached_plan_and_recipe():
    spec = get_spec()
    plan = resolve_adjusted_run_plan(spec)
    assert R._resolved_run_plan(SimpleNamespace(spec=spec, resolved_plan=plan)) is plan
    recipe = plan.recipe_markdown(title="Adjusted test")
    assert "Codex/GPT-5" in recipe
    assert "Adjusted between-child association" in recipe
    assert "child-level PSIS-LOO" in recipe
    assert isinstance(plan, AdjustedRunPlan)


def test_pipeline_has_no_direct_family_extra_reads():
    source = inspect.getsource(pipeline)
    assert "spec.extra" not in source
    assert "resolve_adjusted_run_plan(spec)" in source
