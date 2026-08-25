# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the single- and two-mediator typed run plans (#394)."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from language_reading_predictors.statistical_models import reporting as R
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.mediation_settings import (
    MediationModelSettings,
    MediationMultiModelSettings,
    MediationMultiRunPlan,
    MediationRunPlan,
    NamedConfounderCalibration,
    resolve_mediation_multi_run_plan,
    resolve_mediation_run_plan,
)
from language_reading_predictors.statistical_models.pipelines import (
    mediation as pipeline,
)

_MODEL_ROOT = Path(__file__).parents[2] / "src/language_reading_predictors/statistical_models"


def _spec(
    *,
    settings: object | None = None,
    extra: dict | None = None,
    kind: str = "mediation",
    adjustment: list[str] | None = None,
) -> ModelSpec:
    return ModelSpec(
        model_id="lrp-rli-med-test",
        kind=kind,
        title="typed mediation test",
        outcome_symbol="W",
        mechanism_symbol="L" if kind == "mediation" else None,
        adjustment=adjustment or ["G", "A", "E", "L_t1", "W_pre", "hs"],
        model_settings=settings,
        extra=extra or {},
    )


def _registered_specs() -> list[ModelSpec]:
    specs = []
    for path in sorted(_MODEL_ROOT.glob("lrp_rli_med_*.py")):
        module = importlib.import_module(
            f"language_reading_predictors.statistical_models.{path.stem}"
        )
        specs.append(module.SPEC)
    return specs


def test_every_registered_mediation_spec_is_typed_and_resolves():
    specs = _registered_specs()
    # 19 + the lrp-rli-med-387 phoneme-blending link companion (#619), which
    # copies med-087 and differs only in the outcome's score mean.
    assert len(specs) == 20
    assert {spec.kind for spec in specs} == {"mediation", "mediation_multi"}
    for spec in specs:
        assert spec.extra == {}
        if spec.kind == "mediation":
            assert isinstance(spec.model_settings, MediationModelSettings)
            plan = resolve_mediation_run_plan(spec)
        else:
            assert isinstance(spec.model_settings, MediationMultiModelSettings)
            plan = resolve_mediation_multi_run_plan(spec)
        assert plan.settings_source == "typed"
        assert plan.compute_loo is False


def test_single_typed_and_legacy_declarations_resolve_identically():
    settings = MediationModelSettings(
        outcomes=("N", "L", "W"),
        outcome_kind="bernoulli_offfloor",
        estimand="interventional",
        companion_of="lrp-rli-med-parent",
    )
    typed = _spec(settings=settings, adjustment=["G", "A", "L_t1", "W"])
    legacy = _spec(
        extra={
            "outcomes": ("N", "L", "W"),
            "outcome_kind": "bernoulli_offfloor",
            "estimand": "interventional",
            "companion_of": "lrp-rli-med-parent",
        },
        adjustment=["G", "A", "L_t1", "W"],
    )
    typed = replace(typed, outcome_symbol="N")
    legacy = replace(legacy, outcome_symbol="N")
    typed_plan = resolve_mediation_run_plan(typed)
    legacy_plan = resolve_mediation_run_plan(legacy)
    assert asdict(typed_plan) == {
        **asdict(legacy_plan),
        "settings_source": "typed",
    }


def test_multi_typed_and_legacy_declarations_resolve_identically():
    calibration = NamedConfounderCalibration(symbol="attend", label="IS")
    settings = MediationMultiModelSettings(
        mediators=("L", "N"),
        order=("L", "N"),
        chain=True,
        second_mediator_offfloor=True,
        # ``E`` is a declared bounded-measure confounder, so it must be loaded:
        # the resolver refuses a load set that omits one (#585 finding 3).
        outcomes=("W", "L", "N", "E"),
        named_confounder_calibration=calibration,
    )
    adjustment = ["G", "A", "W_pre", "L_t1", "E", "hs"]
    typed = _spec(kind="mediation_multi", settings=settings, adjustment=adjustment)
    legacy = _spec(
        kind="mediation_multi",
        adjustment=adjustment,
        extra={
            "mediators": ("L", "N"),
            "order": ("L", "N"),
            "chain": True,
            "second_mediator_offfloor": True,
            "outcomes": ("W", "L", "N", "E"),
            "named_confounder_calibration": {
                "symbol": "attend",
                "label": "IS",
            },
        },
    )
    typed_plan = resolve_mediation_multi_run_plan(typed)
    legacy_plan = resolve_mediation_multi_run_plan(legacy)
    assert asdict(typed_plan) == {
        **asdict(legacy_plan),
        "settings_source": "typed",
    }


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (MediationModelSettings(route_symbols=("L",)), "route_symbols"),
        (
            MediationModelSettings(
                mediator_kind="gaussian_composite", route_symbols=()
            ),
            "route_symbols",
        ),
        (
            MediationModelSettings(
                companion_of="parent", estimand="natural"
            ),
            "companion_of",
        ),
        (
            MediationModelSettings(
                period_stacked=True, estimand="interventional"
            ),
            "period-stacked",
        ),
    ],
)
def test_single_cross_field_constraints_fail_early(settings, message):
    with pytest.raises(ValueError, match=message):
        resolve_mediation_run_plan(_spec(settings=settings))


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (MediationMultiModelSettings(mediators=("L",)), "exactly two"),
        (
            MediationMultiModelSettings(
                mediators=("E", "L"), order=("E", "L")
            ),
            "first.*'L'",
        ),
        (
            MediationMultiModelSettings(
                mediators=("L", "B"), order=("L", "E")
            ),
            "permutation",
        ),
    ],
)
def test_multi_cross_field_constraints_fail_early(settings, message):
    with pytest.raises(ValueError, match=message):
        resolve_mediation_multi_run_plan(
            _spec(kind="mediation_multi", settings=settings)
        )


def test_unknown_and_split_declarations_are_rejected():
    with pytest.raises(ValueError, match="unknown mediation setting"):
        resolve_mediation_run_plan(_spec(extra={"outocmes": ("W", "L")}))
    with pytest.raises(ValueError, match="cannot be split"):
        resolve_mediation_run_plan(
            _spec(settings=MediationModelSettings(), extra={"outcomes": ("W", "L")})
        )
    with pytest.raises(ValueError, match="unknown multi-mediation setting"):
        resolve_mediation_multi_run_plan(
            _spec(kind="mediation_multi", extra={"medaitors": ("L", "E")})
        )


def test_plan_maps_loader_factory_and_observation_contracts():
    single = resolve_mediation_run_plan(
        ModelSpec(
            model_id="lrp-rli-med-test",
            kind="mediation",
            title="typed mediation test",
            outcome_symbol="N",
            mechanism_symbol="L",
            adjustment=["G", "A", "L_t1", "W", "hs"],
            model_settings=MediationModelSettings(
                outcomes=("N", "L", "W"),
                outcome_kind="bernoulli_offfloor",
            ),
        )
    )
    # ``pre_required`` is the resolved complete-case rule — the measures whose
    # baseline a leg actually uses — so a loaded-but-unmodelled measure cannot
    # drop a child (#585 finding 4).
    assert single.prepare_kwargs() == {
        "phase_mode": "itt",
        "covariates": ("hs",),
        "outcomes": ("N", "L", "W"),
        "drop_missing_pre": True,
        "pre_required": ("N", "L", "W"),
    }
    assert single.factory_kwargs()["confounder_symbols"] == ("W", "hs")
    assert single.observation_nodes == ("L_post", "y_offfloor")

    multi = resolve_mediation_multi_run_plan(
        _spec(
            kind="mediation_multi",
            adjustment=["G", "A", "W_pre", "L_t1", "E_t1", "R", "hs"],
            settings=MediationMultiModelSettings(
                named_confounder_calibration=NamedConfounderCalibration()
            ),
        )
    )
    assert multi.prepare_kwargs() == {
        "phase_mode": "itt",
        "covariates": ("hs", "attend"),
        "pre_required": ("W", "L", "E", "R"),
    }
    assert multi.factory_kwargs()["confounder_symbols"] == ("R", "hs")
    assert multi.observation_nodes == ("L_post", "E_post", "y_post")


def test_period_stacked_plan_owns_entrypoint_and_loader_contract():
    plan = resolve_mediation_run_plan(
        _spec(
            settings=MediationModelSettings(period_stacked=True),
            adjustment=["T", "A", "E", "L_pre", "W_pre", "hs", "deapp_c"],
        )
    )
    assert plan.entrypoint == "period_stacked"
    kwargs = plan.period_prepare_kwargs()
    assert kwargs["phase_mode"] == "all"
    assert kwargs["outcomes"] == ("W", "L", "E")
    assert set(kwargs) == {
        "phase_mode",
        "outcomes",
        "covariates",
        "post_covariates",
        "baseline_covariates",
        "pre_required",
    }
    assert kwargs["pre_required"] == ("W", "L", "E")


def test_effective_confounders_are_a_validated_subset():
    plan = resolve_mediation_run_plan(_spec(settings=MediationModelSettings()))
    active = plan.with_effective_confounders(("E",))
    assert active.declared_confounders == ("E", "hs")
    assert active.effective_confounders == ("E",)
    with pytest.raises(ValueError, match="not declared"):
        plan.with_effective_confounders(("R",))


def test_reporting_reuses_attached_plan_and_reconstructs_both_kinds():
    single = resolve_mediation_run_plan(_spec(settings=MediationModelSettings()))
    assert R._resolved_run_plan(SimpleNamespace(spec=_spec(), resolved_plan=single)) is single
    multi_spec = _spec(
        kind="mediation_multi", settings=MediationMultiModelSettings()
    )
    multi = R._resolved_run_plan(SimpleNamespace(spec=multi_spec, resolved_plan=None))
    assert isinstance(multi, MediationMultiRunPlan)


def test_pipeline_has_no_direct_family_extra_reads():
    source = inspect.getsource(pipeline)
    assert "spec.extra.get" not in source
    assert "resolve_mediation_run_plan(spec)" in source
    assert "resolve_mediation_multi_run_plan(spec)" in source


@pytest.mark.parametrize(
    ("entrypoint", "spec"),
    [
        (
            pipeline.fit_mediation,
            _spec(settings=MediationModelSettings(period_stacked=True)),
        ),
        (
            pipeline.fit_mediation_period_stacked,
            _spec(settings=MediationModelSettings()),
        ),
    ],
)
def test_wrong_single_entrypoint_fails_before_context_or_data(
    monkeypatch, entrypoint, spec
):
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
    with pytest.raises(ValueError, match="require"):
        entrypoint(spec)


def test_recipe_is_generated_from_the_resolved_contract():
    plan = resolve_mediation_run_plan(_spec(settings=MediationModelSettings()))
    recipe = plan.recipe_markdown(title="Mediation test")
    assert "Codex/GPT-5" in recipe
    assert "Active confounders: E, hs" in recipe
    assert "does not compute ordinary PSIS-LOO" in recipe
    assert isinstance(plan, MediationRunPlan)
