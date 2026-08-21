# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the ordered publication decision (#394 design point 3).

The policy these stages apply is not new — it was assembled inline inside
``generate_key_findings``, and ``test_key_findings.py`` covers its outcomes for
every registered family. What is tested here is the boundary: that the decision
is one object, made in a declared order, reproducible from a stored directory,
and *consumed* by report finalisation rather than remade inside it.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from language_reading_predictors.statistical_models.artifacts import (
    ArtifactLog,
    ArtifactRecord,
    save_table,
)
from language_reading_predictors.statistical_models import release as release_module
from language_reading_predictors.statistical_models.release import (
    GROWTH_INFLUENCE_TRACE_FILENAME,
    MEDIATION_T3_TRACE_FILENAME,
    RELEASE_DECISION_FILENAME,
    ReleaseEvaluation,
    evaluate_publication,
    write_release_decision,
)
from language_reading_predictors.statistical_models.reporting import (
    KEY_FINDINGS_FILENAME,
    generate_key_findings,
)

REPO = Path(__file__).resolve().parents[2]
PARTIAL = REPO / "docs/models/_partials/_key_findings.qmd"
MEDIATION_PARTIAL = REPO / "docs/models/_partials/_results_mediation.qmd"


def _gate(passed: bool = True) -> dict:
    checks = {"rhat": passed, "ess": passed, "divergences": passed, "bfmi": passed}
    return {
        "passed": passed,
        "checks": checks,
        "divergences": 0 if passed else 3,
        "max_rhat": 1.001 if passed else 1.05,
        "min_ess": 1000.0 if passed else 90.0,
        "bfmi_per_chain": [0.8, 0.9] if passed else [0.2, 0.9],
    }


def _fit_dir(
    tmp_path: Path,
    *,
    gate_passed: bool = True,
    kind: str = "mechanism",
    config_name: str = "reporting",
) -> Path:
    """A minimal stored fit: a gate payload and a config, nothing else.

    ``mechanism`` is deliberately an *ungated* family, so the robustness stage is
    out of scope and each test exercises the stage it names.
    """
    d = tmp_path / f"lrp-test-001-{config_name}"
    d.mkdir(parents=True)
    (d / "diagnostics_summary.json").write_text(json.dumps(_gate(gate_passed)))
    (d / "config.json").write_text(
        json.dumps(
            {
                "model_id": "lrp-test-001",
                "kind": kind,
                "outcome_symbol": "W",
                "config_name": config_name,
            }
        )
    )
    return d


def _ctx(output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        output_dir=str(output_dir),
        tables={},
        artifacts=ArtifactLog(),
        spec=SimpleNamespace(model_id="lrp-test-001"),
    )


def _natural_mediation_fit_dir(tmp_path: Path) -> Path:
    """Minimal natural-mediation bundle with a clean, trace-backed t3 sub-fit."""
    d = _fit_dir(tmp_path, kind="mediation")
    config = json.loads((d / "config.json").read_text())
    config["extra"] = {"estimand": "natural"}
    (d / "config.json").write_text(json.dumps(config))
    (d / "mediation_summary_t3.csv").write_text(
        "quantity,converged,trace_file\n"
        f"NIE,True,{MEDIATION_T3_TRACE_FILENAME}\n"
    )
    (d / "subfit_provenance.csv").write_text(
        "label,role,converged,trace_file\n"
        f"lrp-test-001 t3 sensitivity,sensitivity,True,{MEDIATION_T3_TRACE_FILENAME}\n"
    )
    (d / MEDIATION_T3_TRACE_FILENAME).write_bytes(b"trace fixture")
    return d

def _growth_influence_fit_dir(tmp_path: Path) -> Path:
    """Minimal RLM growth bundle with one clean, trace-bound influence refit."""
    from language_reading_predictors.statistical_models.sensitivity import sha256_file

    d = _fit_dir(tmp_path, kind="growth", config_name="rep-lite")
    config = json.loads((d / "config.json").read_text())
    config.update(
        {
            "study_id": "rlm",
            "resolved_run_plan": {"observation_influence_sensitivity": True},
            "extra": {"observation_influence_converged": True},
            "publication_input_contract": {
                "schema_version": 1,
                "study_id": "rlm",
                "publication_ready": True,
                "dataset": {"source_provenance_confirmed": True},
                "measures": {},
                "blockers": [],
            },
        }
    )
    (d / "config.json").write_text(json.dumps(config))
    pd.DataFrame(
        {
            "observation_index": [0, 1, 2],
            "subject_id": ["R001", "R001", "R002"],
            "wave": [1, 2, 2],
            "outcome": ["basread", "basread", "basread"],
            "pareto_k": [0.82, 0.40, 0.40],
            "good_k_threshold": [0.70, 0.70, 0.70],
            "loo_reliable": [False, True, True],
        }
    ).to_csv(d / "pareto_k.csv", index=False)
    pd.DataFrame(
        {
            "coefficient": ["gamma", "delta"],
            "outcome": ["basread", "basread"],
            "primary_median": [-0.10, 0.20],
            "primary_lo89": [-0.20, 0.05],
            "primary_hi89": [-0.02, 0.35],
            "sensitivity_median": [-0.08, 0.25],
            "sensitivity_lo89": [-0.18, 0.10],
            "sensitivity_hi89": [0.01, 0.40],
            "median_direction_stable": [True, True],
            "intervals_overlap": [True, True],
            "n_excluded_cells": [1, 1],
            "n_excluded_children": [1, 1],
            "n_fully_excluded_children": [0, 0],
            "sensitivity_converged": [True, True],
        }
    ).to_csv(d / "growth_influence_sensitivity.csv", index=False)
    trace_path = d / GROWTH_INFLUENCE_TRACE_FILENAME
    trace_path.write_bytes(b"growth influence trace fixture")
    pd.DataFrame(
        {
            "label": ["lrp-test-001 high-Pareto observation-cell exclusion"],
            "role": ["sensitivity"],
            "converged": [True],
            "max_rhat": [1.002],
            "min_ess": [900.0],
            "min_bfmi": [0.7],
            "n_divergences": [0],
            "trace_file": [GROWTH_INFLUENCE_TRACE_FILENAME],
            "trace_sha256": [sha256_file(trace_path)],
        }
    ).to_csv(d / "subfit_provenance.csv", index=False)
    return d


def _write_clean_missingness_trace(
    path: Path, *, include_prior_groups: bool = True
) -> dict[str, float | int]:
    """Persist a deterministic trace that clears every raw sub-fit threshold."""

    rng = np.random.default_rng(811)
    shape = (4, 800)
    coords = {"chain": np.arange(shape[0]), "draw": np.arange(shape[1])}
    posterior = xr.Dataset(
        {
            name: (("chain", "draw"), rng.normal(size=shape))
            for name in (
                "alpha",
                "tau",
                "beta_screening_age",
                "beta_screening_word",
                "kappa",
            )
        },
        coords=coords,
    )
    sample_stats = xr.Dataset(
        {
            "diverging": (("chain", "draw"), np.zeros(shape, dtype=bool)),
            "energy": (("chain", "draw"), rng.normal(size=shape)),
        },
        coords=coords,
    )
    prior = xr.Dataset(
        {
            "p0_target": (
                ("chain", "draw", "target_id"),
                np.full((1, 2, 1), 0.4),
            ),
            "p1_target": (
                ("chain", "draw", "target_id"),
                np.full((1, 2, 1), 0.6),
            ),
        }
    )
    prior_predictive = xr.Dataset(
        {"y_post": (("chain", "draw", "obs_id"), np.full((1, 2, 1), 10))}
    )
    groups = {"posterior": posterior, "sample_stats": sample_stats}
    if include_prior_groups:
        groups.update({"prior": prior, "prior_predictive": prior_predictive})
    xr.DataTree.from_dict(groups).to_netcdf(path)
    if not include_prior_groups:
        return {}
    diagnostics, error = release_module._missingness_trace_diagnostics(path)
    assert error is None
    assert diagnostics is not None
    assert release_module._missingness_diagnostics_pass(diagnostics)
    return diagnostics


def _word_reading_missingness_fit_dir(tmp_path: Path) -> Path:
    """Minimal valid ITT-010 bundle, including every content binding."""

    from language_reading_predictors.statistical_models.itt_missingness import (
        DEFAULT_DELTA_ITEMS,
        LOST_TO_FOLLOW_UP_N,
        MISSINGNESS_PPC_FILENAME,
        MISSINGNESS_PRIOR_DRAWS,
        MISSINGNESS_PRIOR_FILENAME,
        MISSINGNESS_PROVENANCE_FILENAME,
        MISSINGNESS_SCENARIOS,
        MISSINGNESS_SUBFIT_LABEL,
        MISSINGNESS_SUMMARY_FILENAME,
        MISSINGNESS_TRACE_FILENAME,
        OBSERVED_CONTROL_N,
        OBSERVED_INTERVENTION_N,
        RLI_ARCHIVE_CSV_SHA256,
        RLI_ARCHIVE_DOI,
        RLI_LOCAL_WIDE_SHA256,
        RLI_RECONCILIATION_DIGEST,
        SCREENING_ALPHA_SIGMA,
        SCREENING_COVARIATES,
        WITHIN_ARCHIVE_W_MISSING_N,
        WORD_READING_N,
        sha256_file,
    )

    d = _fit_dir(tmp_path, kind="itt")
    config = json.loads((d / "config.json").read_text())
    config.update(
        {
            "model_id": "lrp-rli-itt-010",
            "outcome_symbol": "W",
            "resolved_run_plan": {
                "score_mean_link": "logit",
                "missingness_sensitivity_required_for_release": True,
                "missingness_plan": {
                    "source_doi": RLI_ARCHIVE_DOI,
                    "source_csv_sha256": RLI_ARCHIVE_CSV_SHA256,
                    "local_wide_sha256": RLI_LOCAL_WIDE_SHA256,
                    "reconciliation_digest": RLI_RECONCILIATION_DIGEST,
                    "screening_covariates": list(SCREENING_COVARIATES),
                    "randomised_n": 57,
                    "randomised_intervention_n": 29,
                    "randomised_control_n": 28,
                    "observed_intervention_n": OBSERVED_INTERVENTION_N,
                    "observed_control_n": OBSERVED_CONTROL_N,
                    "lost_to_follow_up_n": LOST_TO_FOLLOW_UP_N,
                    "within_archive_w_missing_n": WITHIN_ARCHIVE_W_MISSING_N,
                    "word_reading_n": WORD_READING_N,
                    "delta_items": list(DEFAULT_DELTA_ITEMS),
                    "scenarios": list(MISSINGNESS_SCENARIOS),
                    "common_estimand_class": "common_profile_standardisation",
                    "completion_estimand_class": "randomised_arm_factual_completion",
                    "intercept_prior_anchor": (
                        "mean_all_57_screening_word_reading_logit"
                    ),
                    "intercept_prior_sigma": SCREENING_ALPHA_SIGMA,
                    "prior_predictive_draws": MISSINGNESS_PRIOR_DRAWS,
                    "trace_filename": MISSINGNESS_TRACE_FILENAME,
                    "summary_filename": MISSINGNESS_SUMMARY_FILENAME,
                    "ppc_filename": MISSINGNESS_PPC_FILENAME,
                    "prior_check_filename": MISSINGNESS_PRIOR_FILENAME,
                    "provenance_filename": MISSINGNESS_PROVENANCE_FILENAME,
                },
            },
        }
    )
    (d / "config.json").write_text(json.dumps(config))
    pd.DataFrame(
        [{"prior": 0.01, "likelihood": 0.02, "diagnosis": "✓"}],
        index=["tau"],
    ).to_csv(d / "psense_summary.csv")

    trace_path = d / MISSINGNESS_TRACE_FILENAME
    diagnostics = _write_clean_missingness_trace(trace_path)
    trace_hash = sha256_file(trace_path)
    shared = {
        "target_population": "all 57 randomised screening profiles",
        "effect_items_median": 2.0,
        "effect_items_lo50": 1.0,
        "effect_items_hi50": 3.0,
        "effect_items_lo89": -1.0,
        "effect_items_hi89": 5.0,
        "intervention_mean_items_median": 12.0,
        "intervention_mean_items_lo50": 11.0,
        "intervention_mean_items_hi50": 13.0,
        "intervention_mean_items_lo89": 9.0,
        "intervention_mean_items_hi89": 15.0,
        "control_mean_items_median": 10.0,
        "control_mean_items_lo50": 9.0,
        "control_mean_items_hi50": 11.0,
        "control_mean_items_lo89": 7.0,
        "control_mean_items_hi89": 13.0,
        "prob_effect_positive": 0.8,
        "clipped_intervention_fraction": 0.0,
        "clipped_control_fraction": 0.0,
        "randomised_n": 57,
        "randomised_intervention_n": 29,
        "randomised_control_n": 28,
        "observed_intervention_n": OBSERVED_INTERVENTION_N,
        "observed_control_n": OBSERVED_CONTROL_N,
        "missing_intervention_n": 1,
        "missing_control_n": 3,
        "lost_to_follow_up_n": LOST_TO_FOLLOW_UP_N,
        "within_archive_w_missing_n": WITHIN_ARCHIVE_W_MISSING_N,
        "n_trials": WORD_READING_N,
        "source_sha256": RLI_ARCHIVE_CSV_SHA256,
        "converged": True,
        **diagnostics,
        "trace_file": MISSINGNESS_TRACE_FILENAME,
        "trace_sha256": trace_hash,
    }
    rows = [
        {
            **shared,
            "scenario": "screening_model_observed_profiles",
            "scenario_class": "bridge",
            "estimand_class": "common_profile_standardisation",
            "delta_intervention_items": None,
            "delta_control_items": None,
        },
        {
            **shared,
            "scenario": "mar_all_57",
            "scenario_class": "missing_at_random",
            "estimand_class": "common_profile_standardisation",
            "delta_intervention_items": None,
            "delta_control_items": None,
        },
        {
            **shared,
            "scenario": "jump_to_reference_intervention_nonstarter",
            "scenario_class": "reference_based",
            "estimand_class": "randomised_arm_factual_completion",
            "delta_intervention_items": None,
            "delta_control_items": None,
        },
    ]
    for delta_i in DEFAULT_DELTA_ITEMS:
        for delta_c in DEFAULT_DELTA_ITEMS:
            rows.append(
                {
                    **shared,
                    "scenario": f"delta_i_{delta_i:+g}_c_{delta_c:+g}",
                    "scenario_class": "arm_specific_delta_grid",
                    "estimand_class": "randomised_arm_factual_completion",
                    "delta_intervention_items": delta_i,
                    "delta_control_items": delta_c,
                }
            )
    pd.DataFrame(rows).to_csv(d / MISSINGNESS_SUMMARY_FILENAME, index=False)
    pd.DataFrame(
        [
            {"arm": "all", "n": 53},
            {"arm": "intervention", "n": 28},
            {"arm": "control", "n": 25},
        ]
    ).assign(
        observed_mean_items=10.0,
        posterior_predictive_mean_items=10.2,
        mean_absolute_prediction_error_items=2.0,
        coverage_50=0.5,
        coverage_89=0.9,
    ).to_csv(d / MISSINGNESS_PPC_FILENAME, index=False)
    prior_common = {
        "effect_items_median": 0.0,
        "effect_items_lo50": -2.0,
        "effect_items_hi50": 2.0,
        "effect_items_lo89": -5.0,
        "effect_items_hi89": 5.0,
        "intervention_mean_items_median": 12.0,
        "intervention_mean_items_lo89": 5.0,
        "intervention_mean_items_hi89": 20.0,
        "control_mean_items_median": 12.0,
        "control_mean_items_lo89": 5.0,
        "control_mean_items_hi89": 20.0,
        "prob_effect_positive": 0.5,
        "prior_predictive_mean_items_median": 12.0,
        "prior_predictive_mean_items_lo89": 2.0,
        "prior_predictive_mean_items_hi89": 25.0,
        "prior_predictive_floor_fraction_median": 0.1,
        "prior_predictive_ceiling_fraction_median": 0.1,
        "alpha_anchor_logit": -1.0,
        "alpha_anchor_items": 21.2,
        "alpha_sigma": SCREENING_ALPHA_SIGMA,
        "prior_draws": MISSINGNESS_PRIOR_DRAWS,
        "source_sha256": RLI_ARCHIVE_CSV_SHA256,
    }
    pd.DataFrame(
        [
            {
                "estimand": "common_profile_all_57",
                "target_population": (
                    "all 57 randomised screening profiles under both arms"
                ),
                **prior_common,
            },
            {
                "estimand": "randomised_arm_factual_mar",
                "target_population": (
                    "29 intervention-arm versus 28 control-arm screening profiles"
                ),
                **prior_common,
            },
        ]
    ).to_csv(d / MISSINGNESS_PRIOR_FILENAME, index=False)
    (d / "attrition_bounds.csv").write_text(
        "outcome,observed_intervention_n,observed_control_n,missing_intervention_n,missing_control_n,n_trials\n"
        "W,28,25,1,3,79\n"
    )
    pd.DataFrame(
        [
            {
                "label": MISSINGNESS_SUBFIT_LABEL,
                "role": "sensitivity",
                "converged": True,
                **diagnostics,
                "trace_file": MISSINGNESS_TRACE_FILENAME,
                "n_obs": 53,
                "n_children": 53,
                "data_digest": "fixture-digest",
            }
        ]
    ).to_csv(d / "subfit_provenance.csv", index=False)
    provenance = {
        "source": {
            "csv_sha256": RLI_ARCHIVE_CSV_SHA256,
            "local_wide_sha256": RLI_LOCAL_WIDE_SHA256,
            "reconciled_included_n": 54,
            "reconciliation_digest": RLI_RECONCILIATION_DIGEST,
        },
        "analysis": {
            "observed_outcome_n": 53,
            "target_profile_n": 57,
            "randomised_by_arm": {"intervention": 29, "control": 28},
            "observed_outcome_by_arm": {"intervention": 28, "control": 25},
            "lost_to_follow_up_n": LOST_TO_FOLLOW_UP_N,
            "within_archive_word_reading_missing_n": WITHIN_ARCHIVE_W_MISSING_N,
            "screening_covariates": list(SCREENING_COVARIATES),
            "delta_items_grid": list(DEFAULT_DELTA_ITEMS),
        },
        "trace": {
            "file": MISSINGNESS_TRACE_FILENAME,
            "sha256": trace_hash,
            "converged": True,
            **diagnostics,
        },
        "outputs": {
            "summary_file": MISSINGNESS_SUMMARY_FILENAME,
            "summary_sha256": sha256_file(d / MISSINGNESS_SUMMARY_FILENAME),
            "ppc_file": MISSINGNESS_PPC_FILENAME,
            "ppc_sha256": sha256_file(d / MISSINGNESS_PPC_FILENAME),
            "prior_check_file": MISSINGNESS_PRIOR_FILENAME,
            "prior_check_sha256": sha256_file(d / MISSINGNESS_PRIOR_FILENAME),
        },
    }
    (d / MISSINGNESS_PROVENANCE_FILENAME).write_text(json.dumps(provenance))
    return d


def _mutate_missingness_diagnostics(
    directory: Path,
    updates: dict[str, float | int],
    *,
    surfaces: tuple[str, ...],
    converged: bool = True,
) -> None:
    """Mutate selected redundant diagnostic surfaces while keeping hashes current."""

    from language_reading_predictors.statistical_models.itt_missingness import (
        MISSINGNESS_PROVENANCE_FILENAME,
        MISSINGNESS_SUMMARY_FILENAME,
        sha256_file,
    )

    summary_path = directory / MISSINGNESS_SUMMARY_FILENAME
    if "summary" in surfaces:
        summary = pd.read_csv(summary_path)
        for field, value in updates.items():
            summary[field] = value
        summary["converged"] = converged
        summary.to_csv(summary_path, index=False)

    provenance_path = directory / MISSINGNESS_PROVENANCE_FILENAME
    provenance = json.loads(provenance_path.read_text())
    if "provenance" in surfaces:
        provenance["trace"].update(updates)
        provenance["trace"]["converged"] = converged
    # A diagnostic mutation in the summary should exercise the redundant-evidence
    # check, not fail earlier merely because the summary's content hash is stale.
    provenance["outputs"]["summary_sha256"] = sha256_file(summary_path)
    provenance_path.write_text(json.dumps(provenance))

    if "subfit" in surfaces:
        subfits_path = directory / "subfit_provenance.csv"
        subfits = pd.read_csv(subfits_path)
        for field, value in updates.items():
            subfits[field] = value
        subfits["converged"] = converged
        subfits.to_csv(subfits_path, index=False)


def _refresh_missingness_trace_binding(directory: Path) -> None:
    """Update both content bindings after an intentional trace mutation."""

    from language_reading_predictors.statistical_models.itt_missingness import (
        MISSINGNESS_PROVENANCE_FILENAME,
        MISSINGNESS_SUMMARY_FILENAME,
        MISSINGNESS_TRACE_FILENAME,
        sha256_file,
    )

    trace_hash = sha256_file(directory / MISSINGNESS_TRACE_FILENAME)
    summary_path = directory / MISSINGNESS_SUMMARY_FILENAME
    summary = pd.read_csv(summary_path)
    summary["trace_sha256"] = trace_hash
    summary.to_csv(summary_path, index=False)

    provenance_path = directory / MISSINGNESS_PROVENANCE_FILENAME
    provenance = json.loads(provenance_path.read_text())
    provenance["trace"]["sha256"] = trace_hash
    provenance["outputs"]["summary_sha256"] = sha256_file(summary_path)
    provenance_path.write_text(json.dumps(provenance))


# --- the stages, and their order ------------------------------------------


def test_a_clean_ungated_fit_publishes(tmp_path):
    decision = evaluate_publication(_fit_dir(tmp_path))
    assert decision.publishable and decision.status == "ok"
    assert decision.scientific_publication_eligible
    assert decision.robustness is None  # ungated family, no robustness verdict
    assert decision.config["model_id"] == "lrp-test-001"


@pytest.mark.parametrize("config_name", ["dev", "test"])
def test_diagnostic_presets_keep_local_reports_but_fail_scientific_release(
    tmp_path, config_name
):
    decision = evaluate_publication(_fit_dir(tmp_path, config_name=config_name))

    assert decision.publishable
    assert decision.development_only
    assert not decision.scientific_publication_eligible
    assert decision.sampling_preset == config_name
    assert "diagnostic-only" in decision.publication_qualification
    assert decision.summary() == "ok (development-only)"


@pytest.mark.parametrize("config_name", ["rep-lite", "reporting"])
def test_publication_sampling_presets_are_scientifically_eligible(
    tmp_path, config_name
):
    decision = evaluate_publication(_fit_dir(tmp_path, config_name=config_name))

    assert decision.publishable
    assert decision.scientific_publication_eligible
    assert not decision.development_only
    assert decision.sampling_preset == config_name
    assert decision.publication_qualification == ""


def test_legacy_fit_without_saved_preset_uses_its_directory_suffix(tmp_path):
    d = _fit_dir(tmp_path, config_name="reporting")
    config = json.loads((d / "config.json").read_text())
    config.pop("config_name")
    (d / "config.json").write_text(json.dumps(config))

    decision = evaluate_publication(d)

    assert decision.sampling_preset == "reporting"
    assert decision.scientific_publication_eligible


def test_unknown_or_directory_inconsistent_preset_fails_closed(tmp_path):
    d = _fit_dir(tmp_path, config_name="dev")
    config = json.loads((d / "config.json").read_text())
    config["config_name"] = "reporting"
    (d / "config.json").write_text(json.dumps(config))

    decision = evaluate_publication(d)

    assert decision.publishable  # local diagnostic rendering remains available
    assert decision.development_only
    assert not decision.scientific_publication_eligible
    assert "disagrees" in decision.publication_qualification


def test_clean_natural_mediation_requires_and_accepts_trace_backed_t3(tmp_path):
    decision = evaluate_publication(_natural_mediation_fit_dir(tmp_path))
    assert decision.publishable


@pytest.mark.parametrize("failure", ["summary", "provenance"])
def test_natural_mediation_t3_gate_failure_withholds_the_whole_release(
    tmp_path, failure
):
    d = _natural_mediation_fit_dir(tmp_path)
    if failure == "summary":
        (d / "mediation_summary_t3.csv").write_text(
            "quantity,converged,trace_file\n"
            f"NIE,False,{MEDIATION_T3_TRACE_FILENAME}\n"
        )
    elif failure == "provenance":
        (d / "subfit_provenance.csv").write_text(
            "label,role,converged,trace_file\n"
            f"lrp-test-001 t3 sensitivity,sensitivity,,{MEDIATION_T3_TRACE_FILENAME}\n"
        )
    decision = evaluate_publication(d)
    assert (decision.status, decision.stage) == ("gate_failed", "computation")
    assert not decision.publishable
    assert any("mediation t3 sensitivity" in check for check in decision.failing_checks)


@pytest.mark.parametrize("failure", ["summary", "provenance", "trace"])
def test_natural_mediation_t3_artifact_failure_does_not_misstate_sampling(
    tmp_path, failure
):
    d = _natural_mediation_fit_dir(tmp_path)
    artifact = {
        "summary": d / "mediation_summary_t3.csv",
        "provenance": d / "subfit_provenance.csv",
        "trace": d / MEDIATION_T3_TRACE_FILENAME,
    }[failure]
    artifact.unlink()

    decision = evaluate_publication(d)
    assert (decision.status, decision.stage) == ("artifacts_incomplete", "artifacts")
    assert not decision.publishable
    assert any(artifact.name in item for item in decision.missing_artifacts)


def test_mediation_t3_gate_does_not_apply_to_interventional_companion(tmp_path):
    d = _fit_dir(tmp_path, kind="mediation")
    config = json.loads((d / "config.json").read_text())
    config["extra"] = {"estimand": "interventional"}
    (d / "config.json").write_text(json.dumps(config))
    assert evaluate_publication(d).publishable

def test_clean_growth_influence_bundle_allows_release(tmp_path):
    assert evaluate_publication(_growth_influence_fit_dir(tmp_path)).publishable


def test_growth_influence_direction_change_withholds_at_robustness_stage(tmp_path):
    d = _growth_influence_fit_dir(tmp_path)
    summary = pd.read_csv(d / "growth_influence_sensitivity.csv")
    gamma = summary["coefficient"] == "gamma"
    summary.loc[gamma, "sensitivity_median"] = 0.08
    summary.loc[gamma, "sensitivity_lo89"] = -0.05
    summary.loc[gamma, "sensitivity_hi89"] = 0.18
    summary.loc[gamma, "median_direction_stable"] = False
    summary.to_csv(d / "growth_influence_sensitivity.csv", index=False)

    decision = evaluate_publication(d)

    assert (decision.status, decision.stage) == (
        "robustness_unresolved",
        "robustness",
    )
    assert "median direction" in decision.reason
    payload = generate_key_findings(d, decision=decision)
    assert payload["status"] == "robustness_unresolved"
    assert payload["sentences"] == []
    assert "release" not in payload


def test_growth_influence_separated_intervals_withhold_at_robustness_stage(tmp_path):
    d = _growth_influence_fit_dir(tmp_path)
    summary = pd.read_csv(d / "growth_influence_sensitivity.csv")
    gamma = summary["coefficient"] == "gamma"
    summary.loc[gamma, "sensitivity_median"] = -0.005
    summary.loc[gamma, "sensitivity_lo89"] = -0.01
    summary.loc[gamma, "sensitivity_hi89"] = -0.001
    summary.loc[gamma, "intervals_overlap"] = False
    summary.to_csv(d / "growth_influence_sensitivity.csv", index=False)

    decision = evaluate_publication(d)

    assert (decision.status, decision.stage) == (
        "robustness_unresolved",
        "robustness",
    )
    assert "overlapping 89% intervals" in decision.reason


def test_growth_influence_nonconvergence_withholds_at_computation_stage(tmp_path):
    d = _growth_influence_fit_dir(tmp_path)
    summary = pd.read_csv(d / "growth_influence_sensitivity.csv")
    summary["sensitivity_converged"] = False
    summary.to_csv(d / "growth_influence_sensitivity.csv", index=False)

    decision = evaluate_publication(d)

    assert (decision.status, decision.stage) == ("gate_failed", "computation")
    assert any("growth observation-cell" in item for item in decision.failing_checks)


def test_growth_influence_missing_trace_withholds_at_artifact_stage(tmp_path):
    d = _growth_influence_fit_dir(tmp_path)
    (d / GROWTH_INFLUENCE_TRACE_FILENAME).unlink()

    decision = evaluate_publication(d)

    assert (decision.status, decision.stage) == (
        "artifacts_incomplete",
        "artifacts",
    )
    assert GROWTH_INFLUENCE_TRACE_FILENAME in decision.missing_artifacts


def test_growth_influence_refit_not_required_when_all_pareto_values_reliable(tmp_path):
    d = _growth_influence_fit_dir(tmp_path)
    pareto = pd.read_csv(d / "pareto_k.csv")
    pareto["pareto_k"] = 0.4
    pareto["loo_reliable"] = True
    pareto.to_csv(d / "pareto_k.csv", index=False)
    (d / "growth_influence_sensitivity.csv").unlink()
    (d / "subfit_provenance.csv").unlink()
    (d / GROWTH_INFLUENCE_TRACE_FILENAME).unlink()
    config = json.loads((d / "config.json").read_text())
    config["observation_influence_converged"] = None
    (d / "config.json").write_text(json.dumps(config))

    assert evaluate_publication(d).publishable


def test_word_reading_primary_cannot_bypass_missingness_gate_with_a_stale_plan(
    tmp_path,
):
    d = _fit_dir(tmp_path, kind="itt")
    config = json.loads((d / "config.json").read_text())
    config.update(
        {
            "model_id": "lrp-rli-itt-010",
            "outcome_symbol": "W",
            "resolved_run_plan": {},
        }
    )
    (d / "config.json").write_text(json.dumps(config))

    decision = evaluate_publication(d)

    assert (decision.status, decision.stage) == ("artifacts_incomplete", "artifacts")
    assert any("missingness sensitivity is undeclared" in item for item in decision.missing_artifacts)


def test_declared_word_reading_missingness_bundle_is_required(tmp_path):
    from language_reading_predictors.statistical_models.itt_missingness import (
        DEFAULT_DELTA_ITEMS,
        LOST_TO_FOLLOW_UP_N,
        MISSINGNESS_SCENARIOS,
        MISSINGNESS_SUMMARY_FILENAME,
        MISSINGNESS_PPC_FILENAME,
        MISSINGNESS_PRIOR_DRAWS,
        MISSINGNESS_PRIOR_FILENAME,
        MISSINGNESS_PROVENANCE_FILENAME,
        MISSINGNESS_TRACE_FILENAME,
        RLI_ARCHIVE_CSV_SHA256,
        RLI_ARCHIVE_DOI,
        RLI_LOCAL_WIDE_SHA256,
        RLI_RECONCILIATION_DIGEST,
        SCREENING_ALPHA_SIGMA,
        SCREENING_COVARIATES,
        WITHIN_ARCHIVE_W_MISSING_N,
        WORD_READING_N,
    )

    d = _fit_dir(tmp_path, kind="itt")
    config = json.loads((d / "config.json").read_text())
    config.update(
        {
            "model_id": "lrp-rli-itt-010",
            "outcome_symbol": "W",
            "resolved_run_plan": {
                "missingness_sensitivity_required_for_release": True,
                "missingness_plan": {
                    "source_doi": RLI_ARCHIVE_DOI,
                    "source_csv_sha256": RLI_ARCHIVE_CSV_SHA256,
                    "local_wide_sha256": RLI_LOCAL_WIDE_SHA256,
                    "reconciliation_digest": RLI_RECONCILIATION_DIGEST,
                    "screening_covariates": list(SCREENING_COVARIATES),
                    "randomised_n": 57,
                    "randomised_intervention_n": 29,
                    "randomised_control_n": 28,
                    "observed_intervention_n": 28,
                    "observed_control_n": 25,
                    "lost_to_follow_up_n": LOST_TO_FOLLOW_UP_N,
                    "within_archive_w_missing_n": WITHIN_ARCHIVE_W_MISSING_N,
                    "word_reading_n": WORD_READING_N,
                    "delta_items": list(DEFAULT_DELTA_ITEMS),
                    "scenarios": list(MISSINGNESS_SCENARIOS),
                    "common_estimand_class": "common_profile_standardisation",
                    "completion_estimand_class": "randomised_arm_factual_completion",
                    "intercept_prior_anchor": (
                        "mean_all_57_screening_word_reading_logit"
                    ),
                    "intercept_prior_sigma": SCREENING_ALPHA_SIGMA,
                    "prior_predictive_draws": MISSINGNESS_PRIOR_DRAWS,
                    "trace_filename": MISSINGNESS_TRACE_FILENAME,
                    "summary_filename": MISSINGNESS_SUMMARY_FILENAME,
                    "ppc_filename": MISSINGNESS_PPC_FILENAME,
                    "prior_check_filename": MISSINGNESS_PRIOR_FILENAME,
                    "provenance_filename": MISSINGNESS_PROVENANCE_FILENAME,
                },
            },
        }
    )
    (d / "config.json").write_text(json.dumps(config))

    decision = evaluate_publication(d)

    assert (decision.status, decision.stage) == ("artifacts_incomplete", "artifacts")
    assert "itt_missingness_sensitivity.csv" in decision.missing_artifacts
    assert MISSINGNESS_TRACE_FILENAME in decision.missing_artifacts


def test_complete_word_reading_missingness_bundle_is_publishable(tmp_path):
    decision = evaluate_publication(_word_reading_missingness_fit_dir(tmp_path))

    assert decision.publishable
    assert decision.status == "ok"


def test_mutated_word_reading_missingness_table_fails_its_content_binding(tmp_path):
    from language_reading_predictors.statistical_models.itt_missingness import (
        MISSINGNESS_SUMMARY_FILENAME,
    )

    d = _word_reading_missingness_fit_dir(tmp_path)
    summary = pd.read_csv(d / MISSINGNESS_SUMMARY_FILENAME)
    summary.loc[
        summary["scenario"].eq("screening_model_observed_profiles"),
        "effect_items_median",
    ] = 99.0
    summary.to_csv(d / MISSINGNESS_SUMMARY_FILENAME, index=False)

    decision = evaluate_publication(d)

    assert (decision.status, decision.stage) == ("artifacts_incomplete", "artifacts")
    assert any("invalid output binding" in item for item in decision.missing_artifacts)


@pytest.mark.parametrize(
    ("field", "failed_value"),
    [
        ("max_rhat", 1.0100001),
        ("min_ess", 399.999),
        ("min_bfmi", 0.299999),
        ("n_divergences", 1),
    ],
)
def test_raw_missingness_thresholds_override_a_stored_true_verdict(
    tmp_path, monkeypatch, field, failed_value
):
    d = _word_reading_missingness_fit_dir(tmp_path)
    raw = {
        "max_rhat": 1.001,
        "min_ess": 1000.0,
        "min_bfmi": 0.8,
        "n_divergences": 0,
    }
    raw[field] = failed_value
    _mutate_missingness_diagnostics(
        d,
        raw,
        surfaces=("summary", "provenance", "subfit"),
        converged=True,
    )
    monkeypatch.setattr(
        release_module,
        "_missingness_trace_diagnostics",
        lambda _path: (raw, None),
    )

    decision = evaluate_publication(d)

    assert (decision.status, decision.stage) == ("gate_failed", "computation")
    assert any("raw sampling-quality thresholds" in item for item in decision.failing_checks)


@pytest.mark.parametrize(
    ("surface", "field", "replacement", "filename"),
    [
        ("summary", "max_rhat", 1.0, "itt_missingness_sensitivity.csv"),
        ("provenance", "min_ess", 999.0, "itt_missingness_provenance.json"),
        ("subfit", "min_bfmi", 0.8, "subfit_provenance.csv"),
    ],
)
def test_mutated_missingness_diagnostic_surface_cannot_disagree_with_trace(
    tmp_path, surface, field, replacement, filename
):
    d = _word_reading_missingness_fit_dir(tmp_path)
    _mutate_missingness_diagnostics(
        d,
        {field: replacement},
        surfaces=(surface,),
    )

    decision = evaluate_publication(d)

    assert (decision.status, decision.stage) == ("artifacts_incomplete", "artifacts")
    assert any(
        filename in item and "do not match the trace" in item
        for item in decision.missing_artifacts
    )


def test_missingness_trace_must_retain_prior_and_prior_predictive_groups(tmp_path):
    from language_reading_predictors.statistical_models.itt_missingness import (
        MISSINGNESS_TRACE_FILENAME,
    )

    d = _word_reading_missingness_fit_dir(tmp_path)
    _write_clean_missingness_trace(
        d / MISSINGNESS_TRACE_FILENAME,
        include_prior_groups=False,
    )
    _refresh_missingness_trace_binding(d)

    decision = evaluate_publication(d)

    assert (decision.status, decision.stage) == ("artifacts_incomplete", "artifacts")
    assert any(
        MISSINGNESS_TRACE_FILENAME in item
        and "missing required trace group" in item
        for item in decision.missing_artifacts
    )


def test_mutated_missingness_prior_check_cannot_pass_its_registered_contract(tmp_path):
    from language_reading_predictors.statistical_models.itt_missingness import (
        MISSINGNESS_PRIOR_FILENAME,
        MISSINGNESS_PROVENANCE_FILENAME,
        sha256_file,
    )

    d = _word_reading_missingness_fit_dir(tmp_path)
    prior_path = d / MISSINGNESS_PRIOR_FILENAME
    prior = pd.read_csv(prior_path)
    prior["prior_draws"] = 999
    prior.to_csv(prior_path, index=False)
    provenance_path = d / MISSINGNESS_PROVENANCE_FILENAME
    provenance = json.loads(provenance_path.read_text())
    provenance["outputs"]["prior_check_sha256"] = sha256_file(prior_path)
    provenance_path.write_text(json.dumps(provenance))

    decision = evaluate_publication(d)

    assert (decision.status, decision.stage) == ("artifacts_incomplete", "artifacts")
    assert any(
        MISSINGNESS_PRIOR_FILENAME in item and "registered draw count" in item
        for item in decision.missing_artifacts
    )


def test_missing_diagnostics_stops_at_the_inputs_stage(tmp_path):
    d = _fit_dir(tmp_path)
    (d / "diagnostics_summary.json").unlink()
    decision = evaluate_publication(d)
    assert (decision.status, decision.stage) == ("not_available", "inputs")
    assert "diagnostics_summary.json is missing" in decision.reason
    # The model is still identified: which fit could not be decided is part of
    # the record, not something the reader has to infer from the directory name.
    assert decision.config["model_id"] == "lrp-test-001"


def test_an_unreadable_config_stops_at_the_inputs_stage(tmp_path):
    d = _fit_dir(tmp_path)
    (d / "config.json").write_text("{not json")
    decision = evaluate_publication(d)
    assert (decision.status, decision.stage) == ("not_available", "inputs")
    assert "could not be parsed" in decision.reason


def test_non_rli_fit_without_input_contract_fails_closed(tmp_path):
    d = _fit_dir(tmp_path)
    config = json.loads((d / "config.json").read_text())
    config["study_id"] = "rlm"
    (d / "config.json").write_text(json.dumps(config))

    decision = evaluate_publication(d)

    assert (decision.status, decision.stage) == ("inputs_unresolved", "inputs")
    assert "no valid publication input contract" in decision.reason
    assert decision.input_failures


def test_unresolved_fit_time_input_contract_withholds_findings(tmp_path):
    d = _fit_dir(tmp_path)
    config = json.loads((d / "config.json").read_text())
    config.update(
        {
            "study_id": "rlm",
            "publication_input_contract": {
                "schema_version": 1,
                "study_id": "rlm",
                "publication_ready": False,
                "dataset": {},
                "measures": {},
                "blockers": ["dataset source provenance is unresolved"],
            },
        }
    )
    (d / "config.json").write_text(json.dumps(config))

    decision = evaluate_publication(d)
    payload = generate_key_findings(d, decision=decision)

    assert (decision.status, decision.stage) == ("inputs_unresolved", "inputs")
    assert decision.as_dict()["input_failures"] == [
        "dataset source provenance is unresolved"
    ]
    assert payload["status"] == "inputs_unresolved"
    assert payload["input_failures"] == ["dataset source provenance is unresolved"]
    assert payload["sentences"] == []


def test_resolved_fit_time_input_contract_allows_later_stages(tmp_path):
    d = _fit_dir(tmp_path)
    config = json.loads((d / "config.json").read_text())
    config.update(
        {
            "study_id": "rlm",
            "publication_input_contract": {
                "schema_version": 1,
                "study_id": "rlm",
                "publication_ready": True,
                "dataset": {"source_provenance_confirmed": True},
                "measures": {},
                "blockers": [],
            },
        }
    )
    (d / "config.json").write_text(json.dumps(config))

    decision = evaluate_publication(d)

    assert decision.status == "ok"


def test_sampling_gate_outranks_unresolved_scientific_inputs(tmp_path):
    d = _fit_dir(tmp_path, gate_passed=False)
    config = json.loads((d / "config.json").read_text())
    config["study_id"] = "rlm"
    (d / "config.json").write_text(json.dumps(config))

    decision = evaluate_publication(d)

    assert (decision.status, decision.stage) == ("gate_failed", "computation")


def test_a_failed_gate_stops_at_the_computation_stage(tmp_path):
    decision = evaluate_publication(_fit_dir(tmp_path, gate_passed=False))
    assert (decision.status, decision.stage) == ("gate_failed", "computation")
    assert decision.failing_checks
    assert not decision.publishable


def test_the_gate_outranks_every_later_stage(tmp_path):
    """An unconverged fit is not asked whether its artefacts are complete.

    Ordering is the substance of this boundary: a reader must never be told a
    fit's outputs are incomplete when the real objection is that it did not
    converge.
    """
    d = _fit_dir(tmp_path, gate_passed=False)
    log = ArtifactLog()
    log.record(
        ArtifactRecord(
            name="tau_summary",
            filename="tau_summary.csv",
            kind="table",
            required=True,
            status="written",
        )
    )  # recorded, never written to disk
    decision = evaluate_publication(d, artifacts=log)
    assert decision.stage == "computation"
    assert decision.missing_artifacts == ()


# --- the artefact stage ---------------------------------------------------


def test_a_required_artefact_absent_from_disk_withholds(tmp_path):
    d = _fit_dir(tmp_path)
    log = ArtifactLog()
    log.record(
        ArtifactRecord(
            name="tau_summary",
            filename="tau_summary.csv",
            kind="table",
            required=True,
            status="written",
        )
    )
    decision = evaluate_publication(d, artifacts=log)
    assert (decision.status, decision.stage) == ("artifacts_incomplete", "artifacts")
    assert decision.missing_artifacts == ("tau_summary.csv",)
    assert "tau_summary.csv" in decision.reason


def test_an_optional_artefact_absent_from_disk_does_not_withhold(tmp_path):
    """The required/optional split is what makes the artefact stage safe.

    A plotting backend hiccup already records its own failure and must not
    suppress a fit's findings.
    """
    d = _fit_dir(tmp_path)
    log = ArtifactLog()
    log.record(
        ArtifactRecord(
            name="tau forest",
            filename="tau_forest.png",
            kind="figure",
            required=False,
            status="skipped",
            error_type="RuntimeError",
            error="backend unavailable",
        )
    )
    assert evaluate_publication(d, artifacts=log).publishable


def test_a_required_artefact_that_is_on_disk_does_not_withhold(tmp_path):
    d = _fit_dir(tmp_path)
    ctx = _ctx(d)
    import pandas as pd

    save_table(ctx, "tau_summary", pd.DataFrame([{"term": "tau", "median": 0.2}]))
    assert evaluate_publication(d, artifacts=ctx.artifacts).publishable


def test_the_decision_is_reproducible_from_a_stored_manifest(tmp_path):
    """No run context: the artefact inventory is read back from the manifest.

    This is the contract every release evaluator in this module keeps — a stored
    fit can be re-decided without refitting.
    """
    d = _fit_dir(tmp_path)
    manifest = {
        "model_id": "lrp-test-001",
        "artifacts": [
            {
                "filename": "tau_summary.csv",
                "name": "tau_summary",
                "kind": "table",
                "required": True,
                "status": "written",
            }
        ],
    }
    (d / "artifact_manifest.json").write_text(json.dumps(manifest))
    decision = evaluate_publication(d)  # no artifacts= argument
    assert decision.status == "artifacts_incomplete"
    assert decision.missing_artifacts == ("tau_summary.csv",)


# --- the record -----------------------------------------------------------


def test_the_decision_is_written_and_recorded(tmp_path):
    d = _fit_dir(tmp_path)
    ctx = _ctx(d)
    decision = evaluate_publication(d, artifacts=ctx.artifacts)
    record = write_release_decision(ctx, decision)

    written = json.loads((d / RELEASE_DECISION_FILENAME).read_text())
    assert written == record
    assert written["status"] == "ok" and written["publishable"] is True
    assert written["model_id"] == "lrp-test-001"
    assert ctx.artifacts.records[RELEASE_DECISION_FILENAME].kind == "json"


def test_the_summary_line_names_the_stage_that_objected(tmp_path):
    clean = evaluate_publication(_fit_dir(tmp_path))
    assert clean.summary() == "ok"
    failed = evaluate_publication(_fit_dir(tmp_path / "b", gate_passed=False))
    assert "computation" in failed.summary()


# --- consumption by the findings generator --------------------------------


def test_key_findings_consumes_the_decision_it_is_given(tmp_path):
    """Finalisation *passes* the decision; the generator does not remake it.

    The directory here would pass on its own, so a payload that says otherwise
    can only have come from the supplied decision.
    """
    d = _fit_dir(tmp_path)
    supplied = ReleaseEvaluation(
        status="gate_failed",
        stage="computation",
        reason="the automatic sampling-quality gate failed",
        failing_checks=("divergences",),
        config={"model_id": "lrp-test-001", "kind": "mechanism"},
    )
    payload = generate_key_findings(d, decision=supplied)
    assert payload["status"] == "gate_failed"
    assert payload["failing_checks"] == ["divergences"]
    assert json.loads((d / KEY_FINDINGS_FILENAME).read_text()) == payload


def test_key_findings_evaluates_for_itself_when_given_nothing(tmp_path):
    """The regeneration scripts pass an output directory and nothing else."""
    d = _fit_dir(tmp_path, gate_passed=False)
    payload = generate_key_findings(d)
    assert payload["status"] == "gate_failed"
    assert payload["model_id"] == "lrp-test-001"


def test_an_incomplete_fit_reaches_the_findings_payload(tmp_path):
    d = _fit_dir(tmp_path)
    log = ArtifactLog()
    log.record(
        ArtifactRecord(
            name="tau_summary",
            filename="tau_summary.csv",
            kind="table",
            required=True,
            status="written",
        )
    )
    decision = evaluate_publication(d, artifacts=log)
    payload = generate_key_findings(d, decision=decision)
    assert payload["status"] == "artifacts_incomplete"
    assert payload["missing_artifacts"] == ["tau_summary.csv"]
    assert payload["sentences"] == []


# --- the reader end of the boundary ---------------------------------------


@pytest.mark.parametrize(
    "status",
    [
        "gate_failed",
        "inputs_unresolved",
        "robustness_unresolved",
        "artifacts_incomplete",
    ],
)
def test_the_partial_renders_every_withholding_status(status):
    """A withhold the reader sees as a soft note is not a withhold.

    Policy split across the pipeline, the readers and the Quarto partials is the
    defect this boundary exists to close, so a status the evaluator can produce
    must have a branch here rather than falling through to the "not available"
    fallback.
    """
    source = PARTIAL.read_text(encoding="utf-8")
    assert f'_kf_status == "{status}"' in source
    branch = source.split(f'_kf_status == "{status}"', 1)[1].split("elif ")[0]
    assert "_scientific_results_released = False" in branch
    assert "callout-important" in branch


def test_report_fail_closes_t3_table_on_verdict_and_trace():
    source = MEDIATION_PARTIAL.read_text(encoding="utf-8")
    assert '"converged" in mediation_summary_t3.columns' in source
    assert "_has(_t3_trace)" in source
    assert "_mediation_display(mediation_summary_t3) if _t3_ready else None" in source
    assert "Temporal-ordering sensitivity suppressed" in source


# ---------------------------------------------------------------------------
# Floored-outcome per-class treatment (2026-08-20 ITT review, finding 2:
# notes/202608201205-itt-code-review-findings.md). The grid gates *whether* a
# floor-rule fit may speak; the class-specific note/qualification mirrors the
# graded branch so the floored path cannot say less about prior dependence
# than the policy promises.
# ---------------------------------------------------------------------------


def _floor_fit_dir(
    tmp_path: Path, *, prior: float, likelihood: float, diagnosis: str
) -> Path:
    d = tmp_path / "lrp-rli-itt-009-reporting"
    d.mkdir(parents=True)
    (d / "config.json").write_text(
        json.dumps(
            {
                "model_id": "lrp-rli-itt-009",
                "kind": "itt",
                "outcome_symbol": "P",
                "resolved_run_plan": {"floor_rule": True},
            }
        )
    )
    pd.DataFrame(
        [{"prior": prior, "likelihood": likelihood, "diagnosis": diagnosis}],
        index=["tau"],
    ).to_csv(d / "psense_summary.csv")
    return d


def _ready_grid(monkeypatch) -> None:
    monkeypatch.setattr(
        release_module, "load_primary_floor_reference", lambda *a, **k: object()
    )
    monkeypatch.setattr(
        release_module, "evaluate_floor_sensitivity", lambda *a, **k: {"ready": True}
    )


def test_floored_conflict_release_carries_the_attenuation_note(
    tmp_path, monkeypatch
):
    """A released floored ``prior_data_conflict`` must carry the lower-bound note
    the module policy promises, exactly as the graded branch does."""
    d = _floor_fit_dir(
        tmp_path,
        prior=0.12,
        likelihood=0.30,
        diagnosis="potential prior-data conflict",
    )
    _ready_grid(monkeypatch)
    decision = release_module.evaluate_release(d)
    assert decision.status == "release"
    assert decision.tau_class == "prior_data_conflict"
    assert decision.floor_grid_required is True
    assert decision.floor_grid_ready is True
    assert "lower bound" in decision.note
    assert "attenuates" in decision.reason


def test_floored_prior_dominant_qualifies_on_grid_evidence(tmp_path, monkeypatch):
    """A prior-dominant floored fit with a validated grid is qualified, not
    released bare — the grid is the estimand-matched analogue of the graded
    branch's trace-bound sweep, which also yields qualify, never release."""
    d = _floor_fit_dir(
        tmp_path,
        prior=0.40,
        likelihood=0.01,
        diagnosis="potential strong prior / weak likelihood",
    )
    _ready_grid(monkeypatch)
    decision = release_module.evaluate_release(d)
    assert decision.status == "qualify"
    assert decision.tau_class == "prior_dominant"
    assert "prior-informed and exploratory" in decision.note
    assert "floor_tau_prior_sensitivity.csv" in decision.evidence


def test_floored_clean_diagnosis_still_releases_bare(tmp_path):
    """A clean tau diagnosis requires no grid and no note (unchanged path)."""
    d = _floor_fit_dir(tmp_path, prior=0.01, likelihood=0.30, diagnosis="✓")
    decision = release_module.evaluate_release(d)
    assert decision.status == "release"
    assert decision.tau_class == "clear"
    assert decision.floor_grid_required is False
    assert decision.note == ""


# ---------------------------------------------------------------------------
# Mandatory phoneme-blending link pair (2026-08-20 ITT review, finding 1:
# notes/202608201205-itt-code-review-findings.md). The pairing was enforced
# only by the key-findings builder and the copied report partial;
# release_decision.json said publishable for an unpaired B fit.
# ---------------------------------------------------------------------------


def _blending_fit_dir(
    tmp_path: Path,
    *,
    model_id: str = "lrp-rli-itt-008",
    psense_diagnosis: str = "✓",
) -> Path:
    d = tmp_path / f"{model_id}-reporting"
    d.mkdir(parents=True)
    (d / "diagnostics_summary.json").write_text(json.dumps(_gate(True)))
    (d / "config.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "kind": "itt",
                "outcome_symbol": "B",
                "resolved_run_plan": {
                    "link_sensitivity_required_for_release": True,
                    "score_mean_link": "logit",
                },
            }
        )
    )
    pd.DataFrame(
        [{"prior": 0.01, "likelihood": 0.30, "diagnosis": psense_diagnosis}],
        index=["tau"],
    ).to_csv(d / "psense_summary.csv")
    return d


def test_unpaired_blending_fit_is_withheld_by_the_release_decision(
    tmp_path, monkeypatch
):
    from language_reading_predictors.statistical_models import (
        blending_sensitivity as bs,
    )

    d = _blending_fit_dir(tmp_path)
    monkeypatch.setattr(
        bs,
        "evaluate_local_blending_link_sensitivity",
        lambda *a, **k: {
            "required": True,
            "ready": False,
            "reason": "stale for testing",
        },
    )
    evaluation = evaluate_publication(d)
    assert evaluation.status == "robustness_unresolved"
    assert evaluation.stage == "robustness"
    assert "phoneme-blending link pair" in evaluation.reason
    assert "stale for testing" in evaluation.reason
    assert evaluation.publishable is False


def test_paired_blending_fit_passes_the_release_decision(tmp_path, monkeypatch):
    from language_reading_predictors.statistical_models import (
        blending_sensitivity as bs,
    )

    d = _blending_fit_dir(tmp_path)
    monkeypatch.setattr(
        bs,
        "evaluate_local_blending_link_sensitivity",
        lambda *a, **k: {"required": True, "ready": True, "reason": ""},
    )
    evaluation = evaluate_publication(d)
    assert evaluation.status == "ok"
    assert evaluation.publishable is True


def test_declared_link_sensitivity_outside_the_registered_pair_fails_closed(
    tmp_path,
):
    """A future B-outcome ITT fit outside 008/108 declares the pairing in its
    plan but has no registered bundle; it must withhold, not release unpaired."""
    d = _blending_fit_dir(tmp_path, model_id="lrp-rli-itt-998")
    evaluation = evaluate_publication(d)
    assert evaluation.status == "robustness_unresolved"
    assert evaluation.stage == "robustness"
    assert "no registered blending-link bundle" in evaluation.reason


# ---------------------------------------------------------------------------
# Joint dependence pairing (2026-08-21 joint review, finding 3:
# notes/202608211100-joint-family-code-review.md). The contrast parents'
# dependence_note has always demanded the LKJ companion pass the house gate
# before the contrast counts as dependence-checked; the release decision now
# verifies the registered companion beside the fit and attaches the
# dependence-unchecked qualifier when it cannot.
# ---------------------------------------------------------------------------


def _joint_contrast_fit_dir(
    tmp_path: Path,
    *,
    model_id: str = "lrp-rli-itt-015",
    companion: str | None = "lrp-rli-itt-215",
    data_sha256: str = "abc123",
) -> Path:
    d = tmp_path / f"{model_id}-reporting"
    d.mkdir(parents=True)
    (d / "diagnostics_summary.json").write_text(json.dumps(_gate(True)))
    contrast: dict = {"left": "TE", "right": "UE"}
    if companion is not None:
        contrast["dependence_companion"] = companion
    (d / "config.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "kind": "joint",
                "config_name": "reporting",
                "data_sha256": data_sha256,
                "resolved_run_plan": {
                    "use_residual_correlation": False,
                    "contrast": contrast,
                },
            }
        )
    )
    pd.DataFrame(
        [
            {"prior": 0.01, "likelihood": 0.30, "diagnosis": "✓"},
            {"prior": 0.02, "likelihood": 0.25, "diagnosis": "✓"},
        ],
        index=["tau[TE]", "tau[UE]"],
    ).to_csv(d / "psense_summary.csv")
    return d


def _joint_companion_dir(
    tmp_path: Path,
    *,
    model_id: str = "lrp-rli-itt-215",
    publishable: bool = True,
    data_sha256: str = "abc123",
) -> Path:
    d = tmp_path / f"{model_id}-reporting"
    d.mkdir(parents=True)
    (d / "config.json").write_text(
        json.dumps(
            {"model_id": model_id, "kind": "joint", "data_sha256": data_sha256}
        )
    )
    (d / RELEASE_DECISION_FILENAME).write_text(
        json.dumps(
            {
                "status": "ok" if publishable else "robustness_unresolved",
                "publishable": publishable,
            }
        )
    )
    return d


def test_joint_contrast_with_ready_companion_releases_without_the_qualifier(
    tmp_path,
):
    d = _joint_contrast_fit_dir(tmp_path)
    _joint_companion_dir(tmp_path)
    evaluation = evaluate_publication(d)
    assert evaluation.status == "ok"
    assert evaluation.publishable is True
    assert "dependence-unchecked" not in (evaluation.robustness.note or "")


def test_joint_contrast_without_its_companion_releases_with_the_qualifier(
    tmp_path,
):
    """A qualify-note, not a withhold: the marginal effects are valid without
    the companion — only the paired contrast's interval is unchecked."""
    d = _joint_contrast_fit_dir(tmp_path)
    evaluation = evaluate_publication(d)
    assert evaluation.status == "ok"
    assert evaluation.publishable is True
    note = evaluation.robustness.note
    assert "lrp-rli-itt-215" in note
    assert "dependence-unchecked" in note


def test_joint_companion_on_different_data_attaches_the_qualifier(tmp_path):
    d = _joint_contrast_fit_dir(tmp_path)
    _joint_companion_dir(tmp_path, data_sha256="something-else")
    evaluation = evaluate_publication(d)
    assert evaluation.status == "ok"
    assert "dependence-unchecked" in evaluation.robustness.note
    assert "same input data" in evaluation.robustness.note


def test_joint_companion_that_withholds_attaches_the_qualifier(tmp_path):
    d = _joint_contrast_fit_dir(tmp_path)
    _joint_companion_dir(tmp_path, publishable=False)
    evaluation = evaluate_publication(d)
    assert evaluation.status == "ok"
    assert "dependence-unchecked" in evaluation.robustness.note
    assert "withholds publication" in evaluation.robustness.note


def test_a_stored_joint_plan_without_the_companion_field_is_unaffected(tmp_path):
    """Every pre-field stored fit re-decides identically: no companion id in the
    plan means no check, mirroring how the level-factors focal_term fallback
    keeps stored decisions reproducible."""
    d = _joint_contrast_fit_dir(tmp_path, companion=None)
    evaluation = evaluate_publication(d)
    assert evaluation.status == "ok"
    assert (evaluation.robustness.note or "") == ""


def test_gate_skips_a_pooled_level_factors_plan():
    """2026-08-20 level-factors review, finding 4: a post-#552 pooled level fit
    records ``focal_term`` as explicitly null — its pooled ``beta_grp`` mixes
    post-crossover waves and is never a randomised contrast — so there is no
    causal headline to gate, and the ``b_grp_time[1]`` fallback (a term the
    pooled posterior structurally lacks) must never be consulted."""
    pooled = {
        "kind": "level_factors",
        "resolved_run_plan": {"group_by_time": False, "focal_term": None},
    }
    assert release_module.gate_applies(pooled) is False


def test_gate_keeps_the_pre_552_level_fallback_and_reads_the_plan_focal_term():
    """A stored pre-#552 fit's plan has no ``focal_term`` key at all; it was
    fitted free, so the gate still applies and falls back to ``b_grp_time[1]``
    — the presence of the key, not its value, is what distinguishes the pooled
    exemption. A t1-referenced plan names ``d_grp_time[t2]``."""
    stored = {"kind": "level_factors", "resolved_run_plan": {"group_by_time": True}}
    assert release_module.gate_applies(stored) is True
    assert release_module.causal_term_for(stored) == "b_grp_time[1]"
    t1_referenced = {
        "kind": "level_factors",
        "resolved_run_plan": {"group_by_time": True, "focal_term": "d_grp_time[t2]"},
    }
    assert release_module.gate_applies(t1_referenced) is True
    assert release_module.causal_term_for(t1_referenced) == "d_grp_time[t2]"
