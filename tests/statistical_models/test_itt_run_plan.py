# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the typed, pre-data ITT specification boundary."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, replace
from types import SimpleNamespace

import pytest

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.itt import (
    IttModelSettings,
    build_itt_from_plan,
    declared_settings_dict,
    prepare_itt_data,
    resolve_itt_run_plan,
)


def _spec(
    settings: IttModelSettings | None = None,
    *,
    outcome: str = "W",
    extra: dict | None = None,
) -> ModelSpec:
    return ModelSpec(
        model_id="lrp-rli-itt-999",
        kind="itt",
        title="Teaching example",
        outcome_symbol=outcome,
        model_settings=settings,
        extra=extra or {},
    )


def test_typed_defaults_resolve_to_the_single_outcome_teaching_model():
    plan = resolve_itt_run_plan(_spec(IttModelSettings()))

    assert plan.settings_source == "typed"
    assert plan.outcomes == ("W",)
    assert plan.cross_symbols == ()
    assert plan.age_effect == "linear"
    assert plan.use_own_baseline is True
    assert plan.headline_likelihood == "beta_binomial"
    assert plan.prepare_kwargs() == {
        "phase_mode": "itt",
        "outcomes": ("W",),
        "covariates": (),
        "restrict_complete": (),
        "drop_missing_pre": True,
        "pre_required": None,
    }


def test_typed_settings_normalise_sequences_and_remain_immutable():
    settings = IttModelSettings(
        outcomes=["W"],  # type: ignore[arg-type]
        adjust_for=["blocks"],  # type: ignore[arg-type]
    )

    assert settings.outcomes == ("W",)
    assert settings.adjust_for == ("blocks",)
    with pytest.raises(FrozenInstanceError):
        settings.use_age_linear = False  # type: ignore[misc]


def test_floor_plan_makes_the_exploratory_estimand_explicit():
    plan = resolve_itt_run_plan(
        _spec(IttModelSettings.for_floor_outcome(), outcome="P")
    )

    assert plan.floor_rule is True
    assert plan.pre_required == ()
    assert plan.use_own_baseline is False
    assert plan.headline_likelihood == "bernoulli_offfloor"
    recipe = plan.recipe_markdown(title="Phoneme awareness")
    assert recipe.startswith("Note: Generated from the validated ITT run plan")
    assert "[!NOTE]" not in recipe
    assert "observed at the floor at t1" in recipe
    assert "Graded Beta-Binomial fits are secondary checks" in recipe


def test_covariate_moderator_is_loaded_once_and_gets_one_main_effect():
    plan = resolve_itt_run_plan(
        _spec(
            IttModelSettings(
                tau_moderator_symbol="blocks",
                tau_moderator_is_covariate=True,
            )
        )
    )

    assert plan.adjust_for == ()
    assert plan.covariates_to_load == ("blocks",)
    assert plan.prepare_kwargs()["covariates"] == ("blocks",)
    assert plan.factory_kwargs()["adjust_for"] == ()


def test_family_preparation_filters_constant_adjusters_from_the_build_contract():
    plan = resolve_itt_run_plan(
        _spec(IttModelSettings(adjust_for=("hs", "hs_missing")))
    )
    calls = []
    prepared = SimpleNamespace(covariates={"hs": [0.0, 1.0]})

    loaded, adjustment = prepare_itt_data(
        plan,
        loader=lambda **kwargs: calls.append(kwargs) or prepared,
    )

    assert loaded is prepared
    assert calls == [plan.prepare_kwargs()]
    assert adjustment == ("hs",)


def test_family_builder_consumes_the_validated_plan_without_reinterpreting_it():
    plan = resolve_itt_run_plan(_spec(IttModelSettings(adjust_for=("hs",))))
    prepared = SimpleNamespace()
    calls = []
    built = object()

    returned = build_itt_from_plan(
        plan,
        prepared,
        effective_adjustment=("hs",),
        builder=lambda data, **kwargs: calls.append((data, kwargs)) or built,
    )

    assert returned is built
    assert calls == [
        (
            prepared,
            {
                "likelihood": "beta_binomial",
                **plan.factory_kwargs(effective_adjustment=("hs",)),
            },
        )
    ]


def test_unknown_legacy_setting_is_rejected():
    spec = _spec(extra={"outcomes": ("W",), "use_age_lienar": True})

    with pytest.raises(ValueError, match="unknown ITT setting.*use_age_lienar"):
        resolve_itt_run_plan(spec)


def test_typed_and_legacy_settings_cannot_be_mixed():
    spec = _spec(IttModelSettings(), extra={"use_age_linear": True})

    with pytest.raises(ValueError, match="cannot be split"):
        resolve_itt_run_plan(spec)


def test_non_itt_typed_settings_are_serialized_from_the_typed_boundary():
    @dataclass(frozen=True, slots=True)
    class ExampleSettings:
        adjust_for: tuple[str, ...]
        use_nonlinear_age: bool

    spec = ModelSpec(
        model_id="lrp-rli-example-001",
        kind="example",
        title="Future typed family",
        model_settings=ExampleSettings(
            adjust_for=("age",),
            use_nonlinear_age=True,
        ),
        extra={"adjust_for": ("wrong_source",)},
    )

    assert declared_settings_dict(spec) == {
        "source": "typed",
        "adjust_for": ("age",),
        "use_nonlinear_age": True,
    }


def test_non_itt_typed_settings_reject_an_unserializable_boundary():
    spec = ModelSpec(
        model_id="lrp-rli-example-002",
        kind="example",
        title="Invalid future typed family",
        model_settings=object(),
    )

    with pytest.raises(TypeError, match="model_settings must be a dataclass instance"):
        declared_settings_dict(spec)


def test_typed_settings_cannot_overwrite_the_reserved_source_marker():
    @dataclass(frozen=True, slots=True)
    class AmbiguousSettings:
        source: str

    spec = ModelSpec(
        model_id="lrp-rli-example-003",
        kind="example",
        title="Ambiguous future typed family",
        model_settings=AmbiguousSettings(source="user supplied"),
    )

    with pytest.raises(ValueError, match="field 'source' is reserved"):
        declared_settings_dict(spec)


def test_legacy_outcomes_without_cross_symbols_keep_the_loaded_outcome_default():
    plan = resolve_itt_run_plan(_spec(extra={"outcomes": ("W",)}))

    assert plan.settings_source == "legacy_extra"
    assert plan.outcomes == ("W",)
    assert plan.cross_symbols == ()


@pytest.mark.parametrize(
    ("settings", "outcome", "message"),
    [
        (
            IttModelSettings(use_age_gp=True),
            "W",
            "use_age_gp and use_age_linear are mutually exclusive",
        ),
        (
            IttModelSettings(outcomes=("L",)),
            "W",
            "outcome_symbol 'W' must appear",
        ),
        (
            IttModelSettings(adjust_for=("blocks",), restrict_complete=("blocks",)),
            "W",
            "cannot both adjust for and only restrict",
        ),
        (
            IttModelSettings(
                pre_required=(),
                use_own_baseline=False,
                floor_rule=True,
                floor_rule_provenance="post_hoc",
                floor_estimand_role="exploratory",
            ),
            "W",
            "floor_rule is only registered",
        ),
        (
            replace(IttModelSettings.for_floor_outcome(), use_varying_tau=True),
            "P",
            "floor_rule cannot use a varying treatment effect",
        ),
        (
            replace(
                IttModelSettings.for_floor_outcome(),
                tau_moderator_symbol="blocks",
                tau_moderator_is_covariate=True,
            ),
            "P",
            "floor_rule cannot use treatment-effect moderation",
        ),
    ],
)
def test_contradictory_settings_fail_during_plan_resolution(
    settings: IttModelSettings,
    outcome: str,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        resolve_itt_run_plan(_spec(settings, outcome=outcome))
