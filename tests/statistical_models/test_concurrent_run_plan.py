# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the typed concurrent-associations settings and run plan (#394 pillar 4)."""

from __future__ import annotations

import glob
import importlib
import os

import pytest

from language_reading_predictors.statistical_models.concurrent import (
    ConcurrentModelSettings,
    ConcurrentRunPlan,
    resolve_concurrent_run_plan,
)
from language_reading_predictors.statistical_models.context import ModelSpec

_META_FIELDS = (
    "design",
    "estimand",
    "causal_status",
    "analysis_population",
    "missing_data_assumption",
)


def _concurrent_specs() -> list[ModelSpec]:
    """Every registered concurrent-associations model's SPEC."""
    root = os.path.dirname(
        importlib.import_module(
            "language_reading_predictors.statistical_models.concurrent"
        ).__file__
    )
    specs: list[ModelSpec] = []
    for path in sorted(glob.glob(os.path.join(root, "lrp_rli_ca_*.py"))):
        mod = importlib.import_module(
            "language_reading_predictors.statistical_models." + os.path.basename(path)[:-3]
        )
        spec = getattr(mod, "SPEC", None)
        if spec is not None and spec.kind == "concurrent":
            specs.append(spec)
    return specs


# --- settings validation ------------------------------------------------------


def test_settings_reject_non_bool_include_age():
    with pytest.raises(TypeError, match="include_age"):
        ConcurrentModelSettings(include_age=1)  # type: ignore[arg-type]


def test_settings_reject_string_predictor_symbols():
    with pytest.raises(TypeError, match="predictor_symbols"):
        ConcurrentModelSettings(predictor_symbols="L")  # type: ignore[arg-type]


def test_settings_reject_string_covariates():
    with pytest.raises(TypeError, match="covariates"):
        ConcurrentModelSettings(covariates="hs")  # type: ignore[arg-type]


def test_settings_reject_non_positive_sigma():
    with pytest.raises(ValueError, match="predictor_slope_sigma must be positive"):
        ConcurrentModelSettings(predictor_slope_sigma=0.0)


def test_settings_reject_bool_sigma():
    with pytest.raises(TypeError, match="predictor_slope_sigma"):
        ConcurrentModelSettings(predictor_slope_sigma=True)  # type: ignore[arg-type]


def test_from_legacy_extra_rejects_unknown_key():
    with pytest.raises(ValueError, match="unknown concurrent setting"):
        ConcurrentModelSettings.from_legacy_extra(
            {"predictor_symbols": ("L",), "predictor_symbol": ("B",)},  # typo
            model_id="lrp-rli-ca-999",
        )


def test_from_legacy_extra_round_trips_and_sigma_defaults_none():
    settings = ConcurrentModelSettings.from_legacy_extra(
        {
            "predictor_symbols": ("L", "B"),
            "covariates": ("hs", "blocks"),
            "include_age": False,
            "include_group": False,
        },
        model_id="lrp-rli-ca-999",
    )
    assert settings.predictor_symbols == ("L", "B")
    assert settings.covariates == ("hs", "blocks")
    assert settings.include_age is False
    assert settings.include_group is False
    # Absent -> None so the pipeline fills the factory default via _default_of.
    assert settings.predictor_slope_sigma is None


# --- resolve ------------------------------------------------------------------


def _spec(**extra) -> ModelSpec:
    return ModelSpec(
        model_id="lrp-rli-ca-000",
        kind="concurrent",
        title="test",
        outcome_symbol="W",
        extra=extra,
    )


def test_resolve_rejects_wrong_kind():
    spec = ModelSpec(model_id="x", kind="itt", title="t", outcome_symbol="W")
    with pytest.raises(ValueError, match="expected kind 'concurrent'"):
        resolve_concurrent_run_plan(spec)


def test_resolve_is_levels_frame_and_associational():
    plan = resolve_concurrent_run_plan(_spec(predictor_symbols=("L", "B")))
    assert plan.settings_source == "legacy_extra"
    assert plan.predictor_slope_sigma is None  # unset -> pipeline fills via _default_of
    prep = plan.prepare_kwargs()
    assert prep["phase_mode"] == "levels"
    # outcome first, then predictors, de-duplicated.
    assert prep["outcomes"] == ("W", "L", "B")
    assert prep["baseline_covariates"] == ()
    assert plan.causal_status.startswith("Associational")
    assert "Table-2 fallacy" in plan.estimand


def test_resolve_dedups_outcome_in_measure_outcomes():
    # A predictor equal to the outcome must not appear twice in the load list.
    plan = resolve_concurrent_run_plan(_spec(predictor_symbols=("W", "L")))
    assert plan.prepare_kwargs()["outcomes"] == ("W", "L")


def test_resolve_keeps_covariates_and_explicit_sigma():
    plan = resolve_concurrent_run_plan(
        _spec(covariates=("blocks", "hs"), predictor_slope_sigma=0.5)
    )
    assert plan.prepare_kwargs()["baseline_covariates"] == ("blocks", "hs")
    assert plan.predictor_slope_sigma == 0.5


def test_typed_settings_are_accepted_and_sourced():
    spec = ModelSpec(
        model_id="lrp-rli-ca-000",
        kind="concurrent",
        title="test",
        outcome_symbol="W",
        model_settings=ConcurrentModelSettings(predictor_symbols=("L",)),
    )
    plan = resolve_concurrent_run_plan(spec)
    assert plan.settings_source == "typed"
    assert plan.predictor_symbols == ("L",)


def test_split_settings_between_typed_and_extra_is_rejected():
    spec = ModelSpec(
        model_id="lrp-rli-ca-000",
        kind="concurrent",
        title="test",
        outcome_symbol="W",
        model_settings=ConcurrentModelSettings(),
        extra={"include_age": False},
    )
    with pytest.raises(ValueError, match="cannot be split"):
        resolve_concurrent_run_plan(spec)


# --- registered-specification coverage (acceptance criterion) -----------------


def test_every_registered_concurrent_model_resolves_with_metadata():
    """Every registered concurrent model resolves to a validated plan that records the
    design, estimand, causal status, analysis population and missing-data assumption
    (#394 pillar 4)."""
    specs = _concurrent_specs()
    assert len(specs) >= 11, f"expected the full concurrent suite, found {len(specs)}"
    for spec in specs:
        plan = resolve_concurrent_run_plan(spec)
        assert isinstance(plan, ConcurrentRunPlan)
        recorded = plan.as_dict()
        for field in _META_FIELDS:
            assert isinstance(recorded[field], str) and recorded[field], (
                f"{spec.model_id}: {field} not recorded"
            )
        # The outcome always loads as the first measure outcome.
        assert plan.prepare_kwargs()["outcomes"][0] == spec.outcome_symbol
