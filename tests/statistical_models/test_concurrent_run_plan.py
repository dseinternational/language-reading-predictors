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


def test_settings_reject_string_require_observed():
    with pytest.raises(TypeError, match="require_observed"):
        ConcurrentModelSettings(require_observed="hs")  # type: ignore[arg-type]


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
    # Absent -> None so the pipeline fills the factory default via default_of.
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
    assert plan.predictor_slope_sigma is None  # unset -> pipeline fills via default_of
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


def test_resolve_rejects_unknown_rli_measure_before_io():
    # Pre-I/O symbol validation (2026-08-21 concurrent review, finding 5): a
    # misspelled RLI measure must fail in resolve, before make_context can reset
    # an output directory — the RLM branch already validated its measures there.
    with pytest.raises(ValueError, match="unknown RLI measure"):
        resolve_concurrent_run_plan(_spec(predictor_symbols=("L", "ZZ")))


def test_resolve_keeps_covariates_and_explicit_sigma():
    plan = resolve_concurrent_run_plan(
        _spec(
            covariates=("blocks", "hs", "hs_missing"),
            predictor_slope_sigma=0.5,
        )
    )
    assert plan.prepare_kwargs()["baseline_covariates"] == (
        "blocks",
        "hs",
        "hs_missing",
    )
    assert plan.predictor_slope_sigma == 0.5


@pytest.mark.parametrize(
    ("parent", "indicator"),
    (
        ("hs", "hs_missing"),
        ("deapp_c", "deapp_c_missing"),
        ("erbto", "erbto_missing"),
    ),
)
def test_resolve_requires_indicator_or_complete_case(parent, indicator):
    with pytest.raises(ValueError, match=indicator):
        resolve_concurrent_run_plan(_spec(covariates=(parent,)))

    paired = resolve_concurrent_run_plan(
        _spec(covariates=(parent, indicator))
    )
    assert paired.covariates == (parent, indicator)

    complete_case = resolve_concurrent_run_plan(
        _spec(covariates=(parent,), require_observed=(parent,))
    )
    prep = complete_case.prepare_kwargs()
    assert prep["baseline_covariates"] == (parent, indicator)
    assert prep["require_observed"] == (parent,)


def test_resolve_rejects_orphan_indicator():
    with pytest.raises(ValueError, match="orphan missingness indicator"):
        resolve_concurrent_run_plan(_spec(covariates=("hs_missing",)))


def test_resolve_rejects_unknown_missingness_indicator():
    with pytest.raises(ValueError, match="unsupported missingness indicator"):
        resolve_concurrent_run_plan(_spec(covariates=("hearing_missing",)))


def test_resolve_rejects_unsupported_or_undeclared_complete_case():
    with pytest.raises(ValueError, match="supports only"):
        resolve_concurrent_run_plan(
            _spec(covariates=("blocks",), require_observed=("blocks",))
        )
    with pytest.raises(ValueError, match="must also be declared"):
        resolve_concurrent_run_plan(_spec(require_observed=("hs",)))


def test_resolve_rejects_indicator_and_complete_case_together():
    with pytest.raises(ValueError, match="cannot use both"):
        resolve_concurrent_run_plan(
            _spec(
                covariates=("hs", "hs_missing"),
                require_observed=("hs",),
            )
        )


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


def test_registered_concurrent_models_use_typed_paired_missingness_settings():
    specs = _concurrent_specs()
    # 11 primaries + the lrp-rli-ca-307 phoneme-blending link companion (#619),
    # which copies ca-007's full covariate set and differs only in the score mean.
    assert len(specs) == 12
    full = (
        "blocks",
        "hs",
        "hs_missing",
        "deapp_c",
        "deapp_c_missing",
        "erbto",
        "erbto_missing",
    )
    minimal = ("blocks", "hs", "hs_missing")
    for spec in specs:
        assert spec.extra == {}, spec.model_id
        assert isinstance(spec.model_settings, ConcurrentModelSettings), spec.model_id
        plan = resolve_concurrent_run_plan(spec)
        expected = minimal if spec.model_id.endswith(("010", "011")) else full
        assert plan.settings_source == "typed"
        assert plan.covariates == expected, spec.model_id
        assert plan.require_observed == ()


# --- the phoneme-blending response-link pair (#619) ---------------------------


def test_the_registered_blending_link_pair_is_paired_both_ways():
    """#619: ca-007 and ca-307 fit the same analysis under the two response links,
    each naming the other, and neither may release alone."""
    specs = {spec.model_id: spec for spec in _concurrent_specs()}
    primary = resolve_concurrent_run_plan(specs["lrp-rli-ca-007"])
    companion = resolve_concurrent_run_plan(specs["lrp-rli-ca-307"])
    assert primary.score_mean_link == "logit"
    assert companion.score_mean_link == "three_choice_guessing_floor"
    assert primary.required_link_companion_model_id == "lrp-rli-ca-307"
    assert companion.required_link_companion_model_id == "lrp-rli-ca-007"
    assert primary.link_sensitivity_required_for_release
    assert companion.link_sensitivity_required_for_release
    for field in (
        "outcome_symbol", "predictor_symbols", "covariates", "require_observed",
        "include_age", "include_group", "predictor_slope_sigma", "waves",
        "estimand", "causal_status", "analysis_population",
    ):
        assert getattr(primary, field) == getattr(companion, field), field


def test_only_a_blending_outcome_requires_the_link_pair():
    """The link governs the OUTCOME's score mean. ca-001..006 carry B as a
    standardised logit *predictor*, which models no B score mean, so they are out of
    scope — an association with blending is not a blending score."""
    specs = {spec.model_id: spec for spec in _concurrent_specs()}
    for spec in specs.values():
        plan = resolve_concurrent_run_plan(spec)
        assert plan.link_sensitivity_required_for_release == (
            plan.outcome_symbol == "B"
        ), spec.model_id
    # ca-001 has B among its predictors and is still out of scope.
    ca1 = resolve_concurrent_run_plan(specs["lrp-rli-ca-001"])
    assert "B" in ca1.predictor_symbols
    assert not ca1.link_sensitivity_required_for_release


def test_the_guessing_floor_link_is_rejected_for_a_non_blending_outcome():
    specs = {spec.model_id: spec for spec in _concurrent_specs()}
    from dataclasses import replace

    bad = replace(
        specs["lrp-rli-ca-001"],
        model_settings=replace(
            specs["lrp-rli-ca-001"].model_settings,
            score_mean_link="three_choice_guessing_floor",
        ),
    )
    with pytest.raises(ValueError, match="only valid for phoneme blending"):
        resolve_concurrent_run_plan(bad)


def test_settings_reject_an_unknown_score_mean_link():
    with pytest.raises(ValueError, match="score_mean_link must be one of"):
        ConcurrentModelSettings(score_mean_link="probit")


def test_the_link_reaches_the_recipe():
    specs = {spec.model_id: spec for spec in _concurrent_specs()}
    companion = resolve_concurrent_run_plan(specs["lrp-rli-ca-307"])
    recipe = companion.recipe_markdown(title="t")
    assert "Score-mean link: three_choice_guessing_floor" in recipe
    assert "lrp-rli-ca-007" in recipe
