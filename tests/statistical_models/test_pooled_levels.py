# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Contracts for the wave-pooled level-association family (``kind='pooled_levels'``)."""

from __future__ import annotations

import numpy as np
import pytest

from language_reading_predictors.statistical_models import pooled_levels as P
from language_reading_predictors.statistical_models.context import ModelSpec


def _spec(**extra) -> ModelSpec:
    base = {
        "adjust_for": ("hs", "hs_missing"),
        "ability_covariate": "blocks",
    }
    base.update(extra)
    return ModelSpec(
        model_id="lrp-rli-pl-000",
        kind="pooled_levels",
        title="test pooled levels",
        outcome_symbol="W",
        mechanism_symbol="L",
        extra=base,
    )


def test_ability_covariate_broadcasts_from_t1_not_per_row():
    """Block design is recorded once at t1; a per-row pull is NaN after t1."""
    plan = P.resolve_pooled_levels_run_plan(_spec())
    kwargs = plan.prepare_kwargs()

    assert kwargs["phase_mode"] == "levels"
    assert kwargs["baseline_covariates"] == ("blocks",)
    assert "blocks" not in kwargs["post_covariates"]


def test_exposure_and_outcome_must_differ():
    with pytest.raises(ValueError, match="trivially 1"):
        P.resolve_pooled_levels_run_plan(
            ModelSpec(
                model_id="lrp-rli-pl-000",
                kind="pooled_levels",
                title="degenerate",
                outcome_symbol="W",
                mechanism_symbol="W",
                extra={},
            )
        )


def test_pooling_without_a_child_random_intercept_is_refused():
    """The defect this family exists to avoid must not be reachable by setting."""
    with pytest.raises(ValueError, match="understates"):
        P.PooledLevelsModelSettings(use_subject_random_intercept=False)


def test_single_wave_cannot_ask_for_wave_intercepts():
    with pytest.raises(ValueError, match="at least two waves"):
        P.PooledLevelsModelSettings(waves=(2,), use_subject_random_intercept=True)


def test_unknown_setting_fails_fast():
    with pytest.raises(ValueError, match="unknown pooled_levels setting"):
        P.PooledLevelsModelSettings.from_extra({"nonsense": 1}, model_id="x")


def test_decomposition_is_the_default_and_names_the_between_term_focal():
    plan = P.resolve_pooled_levels_run_plan(_spec())

    assert plan.decompose_between_within is True
    assert plan.focal_term == "beta_between"
    assert {"beta_between", "beta_within"} <= set(plan.diagnostic_vars(("hs",)))


def test_blended_variant_reports_a_single_slope():
    plan = P.resolve_pooled_levels_run_plan(_spec(decompose_between_within=False))

    assert plan.focal_term == "beta_mech"
    names = set(plan.diagnostic_vars(("hs",)))
    assert "beta_mech" in names and "beta_between" not in names


def test_causal_status_refuses_a_causal_reading():
    plan = P.resolve_pooled_levels_run_plan(_spec())

    assert "Association only" in plan.causal_status
    assert "contemporaneous" in plan.causal_status


def test_between_and_within_regressors_are_orthogonal_by_construction():
    """The Mundlak split must give a child mean and a mean-zero within deviation."""
    rng = np.random.default_rng(0)
    child_idx = np.repeat(np.arange(20), 4)
    x = rng.normal(size=child_idx.size)
    bar = np.zeros_like(x)
    for c in np.unique(child_idx):
        m = child_idx == c
        bar[m] = x[m].mean()
    dev = x - bar

    for c in np.unique(child_idx):
        assert dev[child_idx == c].sum() == pytest.approx(0.0, abs=1e-12)
    assert np.corrcoef(bar, dev)[0, 1] == pytest.approx(0.0, abs=1e-8)
