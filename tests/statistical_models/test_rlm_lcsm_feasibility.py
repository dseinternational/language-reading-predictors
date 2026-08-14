# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Guards for the Byrne/RLM LCSM pre-fit recovery study (#338/#409)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from language_reading_predictors.statistical_models.rlm_lcsm_feasibility import (
    OUTCOMES,
    REVERSE_EDGES,
    FeasibilityCriteria,
    aggregate_recovery,
    build_rlm_lcsm_recovery_model,
    edge_name,
    evaluate_candidate,
    load_rlm_feasibility_design,
    simulate_rlm_lcsm_counts,
    simulation_truth,
)


def test_real_design_uses_paper_waves_confirmed_measures_and_actual_masks():
    ds = load_rlm_feasibility_design("ds")
    pooled = load_rlm_feasibility_design("three_group")

    assert ds.n_children == 24
    assert pooled.n_children == 97
    assert ds.waves.tolist() == pooled.waves.tolist() == [1, 2, 3]
    assert ds.group_labels == ("Down syndrome",)
    assert pooled.group_labels == (
        "Down syndrome",
        "Average readers",
        "Reading-matched",
    )
    assert dict(zip(OUTCOMES, ds.n_trials.tolist(), strict=True)) == {
        "basread": 90,
        "bpvs": 32,
        "trog": 20,
        "basdig": 34,
    }
    assert ds.metadata()["complete_all_measure_children"] == 21
    assert pooled.metadata()["complete_all_measure_children"] == 68
    assert np.array_equal(ds.mask, np.isfinite(ds.counts))
    assert np.allclose(np.diag(ds.correlation_initial), 1.0)
    assert np.linalg.eigvalsh(ds.correlation_initial).min() > 0


def test_truth_has_only_the_pre_specified_reverse_edges_at_requested_strength():
    design = load_rlm_feasibility_design("three_group")
    truth = simulation_truth(design, reverse_strength=0.10)

    assert {edge: truth.coupling(*edge) for edge in REVERSE_EDGES} == {
        edge: 0.10 for edge in REVERSE_EDGES
    }
    assert truth.coupling("bpvs", "basdig") == 0.0
    assert truth.coupling("basread", "basread") == 0.0


def test_simulator_is_reproducible_and_respects_score_bounds():
    design = load_rlm_feasibility_design("ds")
    truth = simulation_truth(design, reverse_strength=0.10)
    left, latent_left = simulate_rlm_lcsm_counts(
        design, truth, np.random.default_rng(42)
    )
    right, latent_right = simulate_rlm_lcsm_counts(
        design, truth, np.random.default_rng(42)
    )

    assert np.array_equal(left, right)
    assert np.array_equal(latent_left, latent_right)
    assert left.shape == design.counts.shape
    assert np.all(left >= 0)
    assert np.all(left <= design.n_trials[None, None, :])


def test_recovery_model_exposes_exact_reverse_parameter_set():
    design = load_rlm_feasibility_design("ds")
    truth = simulation_truth(design, reverse_strength=0.10)
    counts, _ = simulate_rlm_lcsm_counts(design, truth, np.random.default_rng(7))
    model = build_rlm_lcsm_recovery_model(design, counts)

    reverse_names = {edge_name(*edge) for edge in REVERSE_EDGES}
    assert reverse_names <= set(model.named_vars)
    assert "y_obs" in model.observed_RVs[0].name
    assert model["y_data"].get_value().size == int(design.mask.sum())


def _summary_for(scope: str, *, support: float) -> pd.DataFrame:
    criteria = FeasibilityCriteria()
    rows = []
    for strength in (0.0, criteria.alternative_strength):
        for source, target in REVERSE_EDGES:
            rows.append(
                {
                    "scope": scope,
                    "reverse_strength": strength,
                    "parameter": edge_name(source, target),
                    "n_fitted": 40,
                    "true_value": strength,
                    "mean_median": strength,
                    "coverage_89": 0.90,
                    "support_rate": 0.05 if strength == 0 else support,
                    "zero_divergence_rate": 1.0,
                    "bias": 0.0,
                    "n_attempted": 40,
                    "fit_success_rate": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_gate_requires_every_reverse_edge_to_recover():
    passed = evaluate_candidate(_summary_for("three_group", support=0.85), "three_group")
    failed = evaluate_candidate(_summary_for("ds", support=0.50), "ds")

    assert passed["decision"] == "go"
    assert not passed["failures"]
    assert failed["decision"] == "no_go"
    assert len(failed["failures"]) == len(REVERSE_EDGES)


def test_aggregate_recovery_tracks_failed_attempts():
    rows = pd.DataFrame(
        [
            {
                "scope": "ds",
                "simulation": simulation,
                "reverse_strength": 0.10,
                "parameter": "g_basread_bpvs",
                "true_value": 0.10,
                "median": 0.09,
                "covered_89": True,
                "supported_positive": True,
                "divergences": 0,
            }
            for simulation in range(3)
        ]
    )
    attempted = pd.DataFrame(
        {
            "scope": ["ds"] * 4,
            "simulation": range(4),
            "reverse_strength": [0.10] * 4,
        }
    )

    summary = aggregate_recovery(rows, attempted=attempted).iloc[0]
    assert summary["n_fitted"] == 3
    assert summary["n_attempted"] == 4
    assert summary["fit_success_rate"] == 0.75
