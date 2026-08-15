# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Byrne/RLM growth-family port tests (#409 D4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm
import pytest

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.factories import build_growth_model
from language_reading_predictors.statistical_models.growth import (
    GrowthModelSettings,
    resolve_growth_run_plan,
)
from language_reading_predictors.statistical_models.lrp_rlm_gc_001 import SPEC
from language_reading_predictors.statistical_models.preprocessing import (
    load_rlm_growth_panel,
)


def _write_rlm_panel_csv(tmp_path, n_per_group: int = 10) -> str:
    rng = np.random.default_rng(409)
    rows: list[dict[str, object]] = []
    child_number = 0
    for group in (1, 2, 3):
        for _ in range(n_per_group):
            subject = f"R{child_number:03d}"
            age0 = 72 + child_number
            similarity = int(rng.integers(3, 19))
            for wave in (1, 2, 3):
                rows.append(
                    {
                        "subject_id": subject,
                        "time": wave,
                        "readgrp": group,
                        "age": age0 + 12 * (wave - 1),
                        "basread": int(
                            np.clip(
                                5 + 4 * wave + similarity + 5 * (group - 1)
                                + rng.normal(0, 2),
                                0,
                                90,
                            )
                        ),
                        "bassim": similarity if wave == 1 else np.nan,
                    }
                )
            child_number += 1

    frame = pd.DataFrame(rows)
    frame.loc[frame["subject_id"] == "R000", "bassim"] = np.nan
    frame.loc[
        (frame["subject_id"] == "R001") & (frame["time"].isin((2, 3))),
        "basread",
    ] = np.nan
    path = tmp_path / "rlm.csv"
    frame.to_csv(path, index=False)
    return str(path)


def _rlm_spec(**settings) -> ModelSpec:
    defaults = {
        "outcomes": ("basread",),
        "baseline_covariate": "bassim",
        "waves": (1, 2, 3),
        "baseline_scale": "logit_safe",
        "min_outcome_waves": 2,
        "adjust_for_group": True,
        "use_random_slope": False,
    }
    defaults.update(settings)
    return ModelSpec(
        model_id="lrp-rlm-gc-999",
        kind="growth",
        title="test",
        study_id="rlm",
        model_settings=GrowthModelSettings(**defaults),
    )


def test_registered_rlm_growth_plan_is_confirmed_and_group_adjusted():
    plan = resolve_growth_run_plan(SPEC)
    assert plan.study_id == "rlm"
    assert plan.outcomes == ("basread",)
    assert plan.baseline_covariate == "bassim"
    assert plan.waves == (1, 2, 3)
    assert plan.baseline_scale == "logit_safe"
    assert plan.min_outcome_waves == 2
    assert plan.adjust_for_group is True
    assert plan.use_random_slope is False
    assert plan.factory_kwargs()["adjust_for_group"] is True
    assert "reading-matched" in plan.causal_status


@pytest.mark.parametrize("outcome", ["basspel", "woco", "basnum", "basmat"])
def test_rlm_growth_rejects_unresolved_measure_inputs(outcome):
    with pytest.raises(ValueError, match="requires confirmed"):
        resolve_growth_run_plan(_rlm_spec(outcomes=(outcome,)))


def test_rlm_growth_rejects_unadjusted_pooled_cohort():
    with pytest.raises(ValueError, match="reading-group nuisance"):
        resolve_growth_run_plan(_rlm_spec(adjust_for_group=False))


def test_rlm_growth_loader_applies_baseline_and_trajectory_rules(tmp_path):
    panel = load_rlm_growth_panel(
        outcomes=("basread",),
        baseline_covariate="bassim",
        path=_write_rlm_panel_csv(tmp_path),
    )
    assert panel.study_id == "rlm"
    assert panel.n_children == 28
    assert panel.source_n_children == 30
    assert panel.excluded_children == 2
    assert panel.dropped_by_reason == {
        "missing_wave_1_baseline_covariate": 1,
        "fewer_than_minimum_outcome_waves": 1,
    }
    assert set(panel.group) == {1, 2, 3}
    assert panel.group_labels[3] == "Reading-matched"
    assert panel.outcome_labels["basread"] == "BAS word reading"
    assert panel.obs_mask["basread"].sum(axis=1).min() >= 2
    assert abs(float(panel.baseline["bassim"].mean())) < 1e-9
    assert abs(float(panel.baseline["bassim"].std(ddof=1)) - 1.0) < 1e-9
    assert panel.data_path.endswith("rlm.csv")
    assert len(panel.data_sha256) == 64


def test_group_adjusted_growth_factory_builds_and_samples_prior(tmp_path):
    panel = load_rlm_growth_panel(
        outcomes=("basread",),
        baseline_covariate="bassim",
        path=_write_rlm_panel_csv(tmp_path),
    )
    built = build_growth_model(
        panel,
        baseline_covariate="bassim",
        adjust_for_group=True,
        use_random_slope=False,
    )
    assert built.model.coords["reading_group"] == (1, 2, 3)
    assert built.model.named_vars_to_dims["alpha"] == ("reading_group", "outcome")
    assert built.model.named_vars_to_dims["beta"] == ("reading_group", "outcome")
    assert built.model.named_vars_to_dims["kappa"] == ("reading_group", "outcome")
    assert built.model.named_vars_to_dims["gamma"] == ("outcome",)
    assert built.model.named_vars_to_dims["delta"] == ("outcome",)
    assert "sigma_slope" not in built.model.named_vars
    assert "z_slope" not in built.model.named_vars
    with built.model:
        prior = pm.sample_prior_predictive(draws=4, random_seed=15)
    assert prior.prior_predictive["y_obs"].shape[-1] == int(
        panel.obs_mask["basread"].sum()
    )
