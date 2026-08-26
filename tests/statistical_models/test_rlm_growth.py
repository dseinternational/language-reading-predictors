# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Byrne/RLM growth-family port tests (#409 D4)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.factories import build_growth_model
from language_reading_predictors.statistical_models.growth import (
    GrowthModelSettings,
    exclude_growth_observation_cells,
    growth_influence_summary,
    growth_observation_index,
    growth_pareto_table,
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
    assert plan.observation_influence_sensitivity is True
    assert plan.factory_kwargs()["adjust_for_group"] is True
    assert "reading-matched" in plan.causal_status


@pytest.mark.parametrize("outcome", ["basspel", "woco", "basnum"])
def test_rlm_growth_rejects_unresolved_measure_inputs(outcome):
    with pytest.raises(ValueError, match="requires confirmed"):
        resolve_growth_run_plan(_rlm_spec(outcomes=(outcome,)))


def test_rlm_growth_rejects_basmat_with_only_one_paper_window_wave():
    with pytest.raises(ValueError, match=r"basmat.*fewer than 2 source waves"):
        resolve_growth_run_plan(_rlm_spec(outcomes=("basmat",)))


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


@pytest.mark.parametrize(
    ("value", "match"),
    [(7.5, "integer counts"), (-3.0, "below the valid lower bound")],
)
def test_rlm_growth_loader_rejects_invalid_outcome_counts(tmp_path, value, match):
    """#631 finding 5: only the 0..n_trials range was checked, so a fractional
    count passed and was silently truncated by the factory's int cast."""
    path = _write_rlm_panel_csv(tmp_path)
    frame = pd.read_csv(path)
    frame.loc[
        (frame["subject_id"] == "R002") & (frame["time"] == 2), "basread"
    ] = value
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=match):
        load_rlm_growth_panel(
            outcomes=("basread",), baseline_covariate="bassim", path=path
        )


def test_rlm_growth_loader_rejects_fractional_baseline(tmp_path):
    """#631 finding 5: the baseline is a bounded count feeding a Haldane logit,
    so exact integrality applies to it exactly as to the outcomes."""
    path = _write_rlm_panel_csv(tmp_path)
    frame = pd.read_csv(path)
    frame.loc[
        (frame["subject_id"] == "R002") & (frame["time"] == 1), "bassim"
    ] = 7.5
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="integer counts"):
        load_rlm_growth_panel(
            outcomes=("basread",), baseline_covariate="bassim", path=path
        )


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


def test_growth_observation_map_and_flagged_cell_exclusion(tmp_path):
    panel = load_rlm_growth_panel(
        outcomes=("basread",),
        baseline_covariate="bassim",
        path=_write_rlm_panel_csv(tmp_path),
    )
    mapping = growth_observation_index(panel)
    assert len(mapping) == int(panel.obs_mask["basread"].sum())
    assert mapping.iloc[0]["subject_id"] == panel.subject_ids[0]
    assert set(mapping["wave"]) == {1, 2, 3}

    loo = SimpleNamespace(
        pareto_k=np.linspace(0.1, 0.9, len(mapping)),
        good_k=0.7,
    )
    pareto = growth_pareto_table(panel, loo)
    flagged = pareto.loc[~pareto["loo_reliable"]]
    assert len(flagged) > 0
    assert pareto.iloc[0]["pareto_k"] == pytest.approx(0.9)

    selected = flagged.head(2)["observation_index"].to_numpy(dtype=int)
    sensitivity = exclude_growth_observation_cells(panel, selected)
    assert len(growth_observation_index(sensitivity)) == len(mapping) - 2
    assert sum(mask.sum() for mask in sensitivity.obs_mask.values()) == (
        len(mapping) - 2
    )
    assert sum(mask.sum() for mask in panel.obs_mask.values()) == len(mapping)

    first_child = mapping.loc[mapping["child_index"] == 0, "observation_index"]
    child_excluded = exclude_growth_observation_cells(
        panel, first_child.to_numpy(dtype=int)
    )
    assert child_excluded.n_children == panel.n_children - 1
    assert panel.subject_ids[0] not in set(child_excluded.subject_ids)
    assert child_excluded.dropped_by_reason["all_observed_cells_high_pareto"] == 1


def test_growth_influence_summary_compares_unpaired_marginal_posteriors():
    coords = {"chain": [0], "draw": np.arange(4), "outcome": ["basread"]}

    def _trace(gamma, delta):
        return SimpleNamespace(
            posterior=xr.Dataset(
                {
                    "gamma": (
                        ("chain", "draw", "outcome"),
                        np.asarray(gamma).reshape(1, 4, 1),
                    ),
                    "delta": (
                        ("chain", "draw", "outcome"),
                        np.asarray(delta).reshape(1, 4, 1),
                    ),
                },
                coords=coords,
            )
        )

    excluded = pd.DataFrame(
        {
            "observation_index": [2, 8],
            "subject_id": ["R001", "R003"],
            "wave": [3, 2],
            "outcome": ["basread", "basread"],
            "pareto_k": [1.1, 0.8],
        }
    )
    summary = growth_influence_summary(
        _trace([0.1, 0.2, 0.3, 0.4], [-0.4, -0.3, -0.2, -0.1]),
        _trace([-0.2, -0.1, 0.0, 0.1], [-0.3, -0.2, -0.1, 0.0]),
        excluded_cells=excluded,
        sensitivity_converged=True,
        n_fully_excluded_children=1,
    )
    assert list(summary["coefficient"]) == ["gamma", "delta"]
    gamma = summary.iloc[0]
    assert gamma["primary_median"] == pytest.approx(0.25)
    assert gamma["sensitivity_median"] == pytest.approx(-0.05)
    assert gamma["median_shift"] == pytest.approx(-0.30)
    assert bool(gamma["median_direction_stable"]) is False
    assert gamma["n_excluded_cells"] == 2
    assert gamma["n_excluded_children"] == 2
    assert gamma["n_fully_excluded_children"] == 1
    assert bool(gamma["sensitivity_converged"]) is True


def test_growth_report_distinguishes_withheld_from_missing_influence_results():
    partial = Path("docs/models/_partials/_results_growth.qmd").read_text("utf-8")
    assert "growth_influence_sensitivity.csv" in partial
    assert "trace-backed influence sensitivity completed" in partial
    assert "nor supplies exact LOO" in partial
