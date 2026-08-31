# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Release-contract tests for the phoneme-blending response-link pair."""

from __future__ import annotations

import json
import pytest
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from language_reading_predictors.statistical_models import blending_sensitivity as bs
from language_reading_predictors.statistical_models.itt_missingness import (
    MISSINGNESS_RENDERED_SCIENTIFIC_ARTIFACTS,
)
from language_reading_predictors.statistical_models.sensitivity import sha256_file


def _fake_artifact_hash_manifest(character: str) -> str:
    return json.dumps(
        {
            name: character * 64
            for name in bs.BLENDING_RENDERED_SCIENTIFIC_ARTIFACTS
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _write_scientific_artifacts(directory: Path, *, label: str) -> str:
    for name in bs.BLENDING_RENDERED_SCIENTIFIC_ARTIFACTS:
        (directory / name).write_bytes(f"{label}:{name}".encode())
    return json.dumps(
        {
            name: sha256_file(directory / name)
            for name in bs.BLENDING_RENDERED_SCIENTIFIC_ARTIFACTS
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _manifest_rows() -> pd.DataFrame:
    shared = {
        "schema_version": bs.BLENDING_SENSITIVITY_SCHEMA_VERSION,
        "config": "reporting",
        "outcome": "B",
        "sensitivity_of": bs.BLENDING_PRIMARY_MODEL_ID,
        "data_sha256": "a" * 64,
        "environment_lock_sha256": "b" * 64,
        "source_commit": "c" * 40,
        "n": 54,
        "n_intervention": 28,
        "n_control": 26,
        "subject_order_sha256": "d" * 64,
        "treatment_order_sha256": "e" * 64,
        "sampling_draws": 6000,
        "sampling_tune": 6000,
        "sampling_chains": 6,
        "sampling_target_accept": 0.95,
        "sampling_random_seed": 47,
        "ci_prob": 0.89,
        "converged": True,
        "effect_items_median": 0.7,
        "effect_items_lo": -0.1,
        "effect_items_hi": 1.4,
        "prob_effect_positive": 0.9,
        "prob_meaningful_benefit": 0.4,
        "prob_practically_negligible": 0.5,
        "prior_effect_items_median": 0.0,
        "prior_effect_items_lo": -1.3,
        "prior_effect_items_hi": 1.3,
        "max_rhat": 1.001,
        "min_ess": 800.0,
        "min_bfmi": 0.8,
        "n_divergences": 0,
        "loo_elpd": -10.0,
        "loo_p": 2.0,
        "pareto_k_max": 0.4,
        "good_k_threshold": 0.7,
        "loo_reliable": True,
        "guessing_floor_minus_logit_elpd": 1.0,
        "guessing_floor_minus_logit_elpd_se": 0.5,
    }
    return pd.DataFrame(
        [
            {
                **shared,
                "model_id": bs.BLENDING_PRIMARY_MODEL_ID,
                "score_mean_link": "logit",
                "config_sha256": "1" * 64,
                "trace_sha256": "2" * 64,
                "trace_file": f"{bs.BLENDING_PRIMARY_MODEL_ID}-{'2' * 16}.nc",
                "row_map_sha256": "5" * 64,
                "row_map_file": (
                    f"{bs.BLENDING_PRIMARY_MODEL_ID}-rows-{'5' * 16}.csv"
                ),
                "scientific_artifacts_sha256": _fake_artifact_hash_manifest("7"),
            },
            {
                **shared,
                "model_id": bs.BLENDING_COMPANION_MODEL_ID,
                "score_mean_link": "three_choice_guessing_floor",
                "config_sha256": "3" * 64,
                "trace_sha256": "4" * 64,
                "trace_file": f"{bs.BLENDING_COMPANION_MODEL_ID}-{'4' * 16}.nc",
                "row_map_sha256": "6" * 64,
                "row_map_file": (
                    f"{bs.BLENDING_COMPANION_MODEL_ID}-rows-{'6' * 16}.csv"
                ),
                "scientific_artifacts_sha256": _fake_artifact_hash_manifest("8"),
                "loo_elpd": -9.0,
            },
        ]
    )


def test_bound_scientific_artifacts_match_the_itt_results_partial():
    partial = (
        Path(__file__).resolve().parents[2]
        / "docs/models/_partials/_results_itt.qmd"
    ).read_text(encoding="utf-8")
    consumed = set(
        re.findall(r'_(?:csv|img|has)\("([^\"]+\.(?:csv|png))"', partial)
    )

    missingness_artifacts = set(MISSINGNESS_RENDERED_SCIENTIFIC_ARTIFACTS)
    assert missingness_artifacts <= consumed
    assert (consumed - missingness_artifacts) | {"prior_pushforward.csv"} == set(
        bs.BLENDING_RENDERED_SCIENTIFIC_ARTIFACTS
    )
    assert "analysis_set.csv" in consumed
    assert "attrition_bounds.csv" in consumed


def test_evaluator_requires_exact_paired_links(tmp_path):
    rows = _manifest_rows()
    validated: list[str] = []

    def validate(path: Path, row) -> None:
        validated.append(f"{row['model_id']}:{path.name}")

    status = bs.evaluate_blending_link_sensitivity(
        rows,
        trace_root=tmp_path,
        trace_validator=validate,
        row_map_validator=lambda _path, _row: None,
    )
    assert status["ready"] is True
    assert validated == [
        f"{bs.BLENDING_PRIMARY_MODEL_ID}:{bs.BLENDING_PRIMARY_MODEL_ID}-{'2' * 16}.nc",
        f"{bs.BLENDING_COMPANION_MODEL_ID}:{bs.BLENDING_COMPANION_MODEL_ID}-{'4' * 16}.nc",
    ]
    assert status["archive_ready"] is True
    assert status["release_ready"] is False
    assert status["scientific_artifacts_bound"] is True
    assert status["scientific_artifacts_current"] is False

    rows.loc[1, "sampling_draws"] = 4000
    status = bs.evaluate_blending_link_sensitivity(
        rows,
        trace_root=tmp_path,
        trace_validator=validate,
        row_map_validator=lambda _path, _row: None,
    )
    assert status["ready"] is False
    assert status["paired"] is False

    rows = _manifest_rows()
    rows.loc[0, "scientific_artifacts_sha256"] = "{}"
    status = bs.evaluate_blending_link_sensitivity(
        rows,
        trace_root=tmp_path,
        trace_validator=validate,
        row_map_validator=lambda _path, _row: None,
    )
    assert status["ready"] is False
    assert "ITT report contract" in status["reason"]


def test_evaluator_rejects_shared_trace_bytes_and_incoherent_loo_difference(tmp_path):
    rows = _manifest_rows()
    rows.loc[1, "trace_sha256"] = rows.loc[0, "trace_sha256"]
    rows.loc[1, "trace_file"] = (
        f"{bs.BLENDING_COMPANION_MODEL_ID}-{'2' * 16}.nc"
    )
    status = bs.evaluate_blending_link_sensitivity(
        rows,
        trace_root=tmp_path,
        trace_validator=lambda _path, _row: None,
        row_map_validator=lambda _path, _row: None,
    )
    assert status["ready"] is False
    assert "identical trace bytes" in status["reason"]

    rows = _manifest_rows()
    rows.loc[1, "guessing_floor_minus_logit_elpd"] = 0.25
    status = bs.evaluate_blending_link_sensitivity(
        rows,
        trace_root=tmp_path,
        trace_validator=lambda _path, _row: None,
        row_map_validator=lambda _path, _row: None,
    )
    assert status["ready"] is False
    assert "LOO difference" in status["reason"]


def test_evaluator_binds_summary_to_current_primary_bytes(tmp_path):
    rows = _manifest_rows()
    directories = {}
    for index, model_id in enumerate(
        (bs.BLENDING_PRIMARY_MODEL_ID, bs.BLENDING_COMPANION_MODEL_ID), start=1
    ):
        directory = tmp_path / model_id
        directory.mkdir()
        (directory / "config.json").write_text(f"config-{index}")
        (directory / "trace.nc").write_text(f"trace-{index}")
        (directory / "pareto_k.csv").write_text(f"row-map-{index}")
        artifact_manifest = _write_scientific_artifacts(
            directory, label=f"fit-{index}"
        )
        directories[model_id] = directory
        rows.loc[
            rows["model_id"] == model_id, "scientific_artifacts_sha256"
        ] = artifact_manifest
        rows.loc[rows["model_id"] == model_id, "config_sha256"] = sha256_file(
            directory / "config.json"
        )
        rows.loc[rows["model_id"] == model_id, "trace_sha256"] = sha256_file(
            directory / "trace.nc"
        )
        rows.loc[rows["model_id"] == model_id, "row_map_sha256"] = sha256_file(
            directory / "pareto_k.csv"
        )
        trace_sha = rows.loc[
            rows["model_id"] == model_id, "trace_sha256"
        ].iat[0]
        row_map_sha = rows.loc[
            rows["model_id"] == model_id, "row_map_sha256"
        ].iat[0]
        rows.loc[rows["model_id"] == model_id, "trace_file"] = (
            f"{model_id}-{trace_sha[:16]}.nc"
        )
        rows.loc[rows["model_id"] == model_id, "row_map_file"] = (
            f"{model_id}-rows-{row_map_sha[:16]}.csv"
        )

    status = bs.evaluate_blending_link_sensitivity(
        rows,
        trace_root=tmp_path,
        primary_model_dirs=directories,
        trace_validator=lambda _path, _row: None,
        row_map_validator=lambda _path, _row: None,
    )
    assert status["ready"] is True
    assert status["scientific_artifacts_current"] is True

    companion_artifact = (
        directories[bs.BLENDING_COMPANION_MODEL_ID] / "predicted_scores.png"
    )
    original_artifact = companion_artifact.read_bytes()
    companion_artifact.write_text("changed companion scientific figure")
    status = bs.evaluate_blending_link_sensitivity(
        rows,
        trace_root=tmp_path,
        primary_model_dirs=directories,
        trace_validator=lambda _path, _row: None,
        row_map_validator=lambda _path, _row: None,
    )
    assert status["ready"] is False
    assert "scientific report artefact has changed" in status["reason"]
    assert "predicted_scores.png" in status["reason"]
    companion_artifact.write_bytes(original_artifact)

    (directories[bs.BLENDING_PRIMARY_MODEL_ID] / "config.json").write_text("changed")
    status = bs.evaluate_blending_link_sensitivity(
        rows,
        trace_root=tmp_path,
        primary_model_dirs=directories,
        trace_validator=lambda _path, _row: None,
        row_map_validator=lambda _path, _row: None,
    )
    assert status["ready"] is False
    assert "config has changed" in status["reason"]


def test_local_evaluator_hashes_both_current_fit_directories(tmp_path):
    rows = _manifest_rows()
    models = tmp_path / "models"
    directories = {}
    for index, model_id in enumerate(
        (bs.BLENDING_PRIMARY_MODEL_ID, bs.BLENDING_COMPANION_MODEL_ID), start=1
    ):
        directory = models / f"{model_id}-reporting"
        directory.mkdir(parents=True)
        config = {
            "model_id": model_id,
            "outcome_symbol": "B",
            "resolved_run_plan": {
                "score_mean_link": dict(bs.BLENDING_LINK_MODELS)[model_id]
            },
            "model_settings": {
                "score_mean_link": dict(bs.BLENDING_LINK_MODELS)[model_id]
            },
        }
        (directory / "config.json").write_text(json.dumps(config))
        (directory / "trace.nc").write_text(f"trace-{index}")
        (directory / "pareto_k.csv").write_text(f"row-map-{index}")
        artifact_manifest = _write_scientific_artifacts(
            directory, label=f"local-fit-{index}"
        )
        directories[model_id] = directory
        mask = rows["model_id"] == model_id
        rows.loc[mask, "scientific_artifacts_sha256"] = artifact_manifest
        for column, filename in (
            ("config_sha256", "config.json"),
            ("trace_sha256", "trace.nc"),
            ("row_map_sha256", "pareto_k.csv"),
        ):
            rows.loc[mask, column] = sha256_file(directory / filename)
        trace_sha = str(rows.loc[mask, "trace_sha256"].iat[0])
        row_map_sha = str(rows.loc[mask, "row_map_sha256"].iat[0])
        rows.loc[mask, "trace_file"] = f"{model_id}-{trace_sha[:16]}.nc"
        rows.loc[mask, "row_map_file"] = (
            f"{model_id}-rows-{row_map_sha[:16]}.csv"
        )
    # The local check byte-binds the installed copy to the central archive
    # manifest (finding 1, notes/202608201205-itt-code-review-findings.md); the
    # fixture layout mirrors production: <root>/models/<fit> beside
    # <root>/blending_link_sensitivity/.
    archive = tmp_path / "blending_link_sensitivity"
    archive.mkdir()
    rows.to_csv(archive / bs.BLENDING_SENSITIVITY_FILENAME, index=False)
    for directory in directories.values():
        shutil.copyfile(
            archive / bs.BLENDING_SENSITIVITY_FILENAME,
            directory / bs.BLENDING_SENSITIVITY_FILENAME,
        )

    status = bs.evaluate_local_blending_link_sensitivity(
        directories[bs.BLENDING_PRIMARY_MODEL_ID]
    )
    assert status["ready"] is True
    assert len(status["summary_sha256"]) == 64

    report_table = directories[bs.BLENDING_COMPANION_MODEL_ID] / "rope_summary.csv"
    report_table.write_text("replaced companion result table")
    status = bs.evaluate_local_blending_link_sensitivity(
        directories[bs.BLENDING_PRIMARY_MODEL_ID]
    )
    assert status["ready"] is False
    assert "scientific report artefact has changed" in status["reason"]
    assert "rope_summary.csv" in status["reason"]

    # Restore the manifest-bound bytes before exercising the independent trace check.
    report_table.write_bytes(
        b"local-fit-2:rope_summary.csv"
    )
    (directories[bs.BLENDING_COMPANION_MODEL_ID] / "trace.nc").write_text(
        "replaced companion trace"
    )
    status = bs.evaluate_local_blending_link_sensitivity(
        directories[bs.BLENDING_PRIMARY_MODEL_ID]
    )
    assert status["ready"] is False
    assert "trace has changed" in status["reason"]


def test_local_evaluator_requires_the_central_archive_manifest(tmp_path):
    """Finding 1 (notes/202608201205-itt-code-review-findings.md): with no
    central archive manifest the installed pair CSV is only self-referential
    evidence, so the check fails closed rather than trusting it."""
    rows = _manifest_rows()
    models = tmp_path / "models"
    directory = models / f"{bs.BLENDING_PRIMARY_MODEL_ID}-reporting"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(
        json.dumps({"model_id": bs.BLENDING_PRIMARY_MODEL_ID})
    )
    rows.to_csv(directory / bs.BLENDING_SENSITIVITY_FILENAME, index=False)

    status = bs.evaluate_local_blending_link_sensitivity(directory)
    assert status["required"] is True
    assert status["ready"] is False
    assert "central B link archive manifest is missing" in status["reason"]


def test_local_evaluator_rejects_an_installed_summary_the_archive_never_validated(
    tmp_path,
):
    """An edited installed CSV whose hash columns stay coherent must not be
    quotable: only the byte-identical, build-validated archive manifest counts."""
    rows = _manifest_rows()
    models = tmp_path / "models"
    directory = models / f"{bs.BLENDING_PRIMARY_MODEL_ID}-reporting"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(
        json.dumps({"model_id": bs.BLENDING_PRIMARY_MODEL_ID})
    )
    archive = tmp_path / "blending_link_sensitivity"
    archive.mkdir()
    rows.to_csv(archive / bs.BLENDING_SENSITIVITY_FILENAME, index=False)
    edited = rows.copy()
    edited.loc[:, "effect_items_median"] = 9.9
    edited.to_csv(directory / bs.BLENDING_SENSITIVITY_FILENAME, index=False)

    status = bs.evaluate_local_blending_link_sensitivity(directory)
    assert status["required"] is True
    assert status["ready"] is False
    assert "does not match the validated central archive manifest" in status["reason"]


def _fit_record(
    model_dir: Path,
    *,
    model_id: str,
    link: str,
    trace_text: str,
) -> bs._FitRecord:
    model_dir.mkdir(parents=True)
    (model_dir / "trace.nc").write_text(trace_text)
    (model_dir / "pareto_k.csv").write_text(f"rows for {model_id}")
    _write_scientific_artifacts(model_dir, label=model_id)
    settings = {"source": "typed", "score_mean_link": link}
    resolved_run_plan = {
        "model_id": model_id,
        "score_mean_link": link,
        "required_link_companion_model_id": (
            bs.BLENDING_COMPANION_MODEL_ID
            if model_id == bs.BLENDING_PRIMARY_MODEL_ID
            else bs.BLENDING_PRIMARY_MODEL_ID
        ),
        "link_sensitivity_required_for_release": True,
        "outcome_symbol": "B",
    }
    return bs._FitRecord(
        model_dir=model_dir,
        config={
            "model_settings": settings,
            "resolved_run_plan": resolved_run_plan,
            "ci_prob": 0.89,
        },
        model_id=model_id,
        score_mean_link=link,
        config_name="reporting",
        config_sha256=("1" if model_id == bs.BLENDING_PRIMARY_MODEL_ID else "3") * 64,
        trace_sha256=sha256_file(model_dir / "trace.nc"),
        row_map_sha256=sha256_file(model_dir / "pareto_k.csv"),
        data_sha256="a" * 64,
        environment_lock_sha256="b" * 64,
        source_commit="c" * 40,
        n_obs=4,
        subject_ids=("a", "b", "c", "d"),
        treatment=(1, 1, 0, 0),
        sampling={
            "draws": 100,
            "tune": 100,
            "chains": 2,
            "target_accept": 0.95,
            "random_seed": 47,
        },
        summary={
            "tau_prob_median": 0.1,
            "tau_prob_lo": -0.05,
            "tau_prob_hi": 0.2,
            "prob_ame_pos": 0.8,
        },
        rope={"prob_benefit_ge_delta": 0.4, "prob_in_rope": 0.5},
        prior={
            "prior_items_median": 0.0,
            "prior_items_lo": -1.3,
            "prior_items_hi": 1.3,
        },
        convergence={
            "max_rhat": 1.0,
            "min_ess": 800.0,
            "min_bfmi": 0.8,
            "n_divergences": 0,
        },
        loo_elpd=-10.0,
        loo_p=2.0,
        pareto_k_max=0.4,
        good_k_threshold=0.7,
        loo_reliable=True,
        loo_i=np.array([-2.0, -2.5, -3.0, -2.5]),
        pareto_k=np.array([0.2, 0.3, 0.4, 0.1]),
    )


def test_builder_writes_content_addressed_archive_and_report_copies(
    tmp_path, monkeypatch
):
    models = tmp_path / "models"
    archive = tmp_path / "archive"
    primary = _fit_record(
        models / f"{bs.BLENDING_PRIMARY_MODEL_ID}-reporting",
        model_id=bs.BLENDING_PRIMARY_MODEL_ID,
        link="logit",
        trace_text="primary trace",
    )
    companion = _fit_record(
        models / f"{bs.BLENDING_COMPANION_MODEL_ID}-reporting",
        model_id=bs.BLENDING_COMPANION_MODEL_ID,
        link="three_choice_guessing_floor",
        trace_text="companion trace",
    )
    records = iter((primary, companion))
    monkeypatch.setattr(bs, "_load_fit_record", lambda *args, **kwargs: next(records))
    monkeypatch.setattr(
        bs,
        "evaluate_blending_link_sensitivity",
        lambda *args, **kwargs: {
            "ready": True,
            "archive_ready": True,
            "release_ready": True,
        },
    )

    result = bs.build_blending_link_sensitivity(models, archive)
    assert len(result) == 2
    assert set(result["score_mean_link"]) == {
        "logit",
        "three_choice_guessing_floor",
    }
    for row in result.itertuples():
        assert set(json.loads(row.scientific_artifacts_sha256)) == set(
            bs.BLENDING_RENDERED_SCIENTIFIC_ARTIFACTS
        )
        path = archive / row.trace_file
        assert path.is_file()
        assert sha256_file(path) == row.trace_sha256
        row_map = archive / row.row_map_file
        assert row_map.is_file()
        assert sha256_file(row_map) == row.row_map_sha256
    assert (archive / bs.BLENDING_SENSITIVITY_FILENAME).is_file()
    assert (
        primary.model_dir / bs.BLENDING_SENSITIVITY_FILENAME
    ).is_file()
    assert (
        companion.model_dir / bs.BLENDING_SENSITIVITY_FILENAME
    ).is_file()


# --- the level family's registered pair (#584 decision 2) ---------------------


def _level_fit_dir(
    root: Path,
    model_id: str,
    *,
    link: str,
    config_name: str = "reporting",
    gate_passed: bool = True,
    digest: str = "d1g3st",
    data_sha256: str = "a" * 64,
    n_obs: int = 215,
    items_median: float = 0.64,
) -> Path:
    """A minimal stored level-factor B fit: config, gate verdict and ROPE card."""
    directory = root / f"{model_id}-{config_name}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "kind": "level_factors",
                "outcome_symbol": "B",
                "config_name": config_name,
                "data_sha256": data_sha256,
                "n_obs": n_obs,
                "fitted_data_identity": {"digest": digest},
                "resolved_run_plan": {
                    "score_mean_link": link,
                    "link_sensitivity_required_for_release": True,
                    "required_link_companion_model_id": (
                        "lrp-rli-lf-106"
                        if model_id == "lrp-rli-lf-006"
                        else "lrp-rli-lf-006"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )
    (directory / "diagnostics_summary.json").write_text(
        json.dumps(
            {
                "divergences": 0 if gate_passed else 47,
                "max_rhat": 1.0 if gate_passed else 1.9,
                "min_ess": 4000.0 if gate_passed else 12.0,
                "bfmi_per_chain": [0.9, 0.9] if gate_passed else [0.05, 0.9],
                "checks": {
                    "rhat": gate_passed,
                    "ess": gate_passed,
                    "divergences": gate_passed,
                    "bfmi": gate_passed,
                },
                "passed": gate_passed,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [{"items_median": items_median, "items_lo": -0.2, "items_hi": 1.4, "pd": 0.9}]
    ).to_csv(directory / "rope_summary.csv", index=False)
    return directory


def test_level_pair_is_ready_when_both_links_are_fitted_on_the_same_rows(tmp_path):
    models = tmp_path / "models"
    primary = _level_fit_dir(models, "lrp-rli-lf-006", link="logit")
    _level_fit_dir(
        models, "lrp-rli-lf-106", link="three_choice_guessing_floor", items_median=0.43
    )
    status = bs.evaluate_level_blending_link_pair(primary, plan_checker=_plan_is_current)
    assert status["required"] and status["ready"], status
    cards = status["cards"]
    assert cards["lrp-rli-lf-006"]["score_mean_link"] == "logit"
    assert cards["lrp-rli-lf-106"]["items_median"] == 0.43
    # Either side of the pair sees the same verdict.
    companion_status = bs.evaluate_level_blending_link_pair(
        models / "lrp-rli-lf-106-reporting", plan_checker=_plan_is_current)
    assert companion_status["ready"]


def test_level_pair_is_not_ready_without_its_twin(tmp_path):
    models = tmp_path / "models"
    primary = _level_fit_dir(models, "lrp-rli-lf-006", link="logit")
    status = bs.evaluate_level_blending_link_pair(primary, plan_checker=_plan_is_current)
    assert status["required"] and not status["ready"]
    assert "not present beside this one" in status["reason"]


def test_level_pair_requires_the_registered_pairing_even_on_a_stale_plan(tmp_path):
    """The requirement is derived from the registered ids as well as the stored
    plan, so a fit whose plan predates the pairing cannot bypass the gate."""
    models = tmp_path / "models"
    primary = _level_fit_dir(models, "lrp-rli-lf-006", link="logit")
    config = json.loads((primary / "config.json").read_text(encoding="utf-8"))
    config["resolved_run_plan"].pop("link_sensitivity_required_for_release")
    status = bs.evaluate_level_blending_link_pair(primary, config=config, plan_checker=_plan_is_current)
    assert status["required"] and not status["ready"]


def test_level_pair_rejects_an_unconverged_side(tmp_path):
    models = tmp_path / "models"
    primary = _level_fit_dir(models, "lrp-rli-lf-006", link="logit")
    _level_fit_dir(
        models,
        "lrp-rli-lf-106",
        link="three_choice_guessing_floor",
        gate_passed=False,
    )
    status = bs.evaluate_level_blending_link_pair(primary, plan_checker=_plan_is_current)
    assert not status["ready"]
    assert "convergence gate" in status["reason"]


def test_level_pair_rejects_different_fitted_rows(tmp_path):
    models = tmp_path / "models"
    primary = _level_fit_dir(models, "lrp-rli-lf-006", link="logit")
    _level_fit_dir(
        models,
        "lrp-rli-lf-106",
        link="three_choice_guessing_floor",
        digest="0th3r",
    )
    status = bs.evaluate_level_blending_link_pair(primary, plan_checker=_plan_is_current)
    assert not status["ready"]
    assert "fitted rows" in status["reason"]


def test_level_pair_rejects_two_fits_under_the_same_link(tmp_path):
    models = tmp_path / "models"
    primary = _level_fit_dir(models, "lrp-rli-lf-006", link="logit")
    _level_fit_dir(models, "lrp-rli-lf-106", link="logit")
    status = bs.evaluate_level_blending_link_pair(primary, plan_checker=_plan_is_current)
    assert not status["ready"]
    assert "opposite score-mean links" in status["reason"]


def test_level_pair_ignores_a_non_blending_level_fit(tmp_path):
    models = tmp_path / "models"
    directory = models / "lrp-rli-lf-001-reporting"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(
        json.dumps(
            {
                "model_id": "lrp-rli-lf-001",
                "kind": "level_factors",
                "outcome_symbol": "W",
                "config_name": "reporting",
                "resolved_run_plan": {"score_mean_link": "logit"},
            }
        ),
        encoding="utf-8",
    )
    status = bs.evaluate_level_blending_link_pair(directory, plan_checker=_plan_is_current)
    assert not status["required"] and status["ready"]


def test_release_gate_withholds_an_unpaired_level_blending_fit(tmp_path):
    from language_reading_predictors.statistical_models import release


    models = tmp_path / "models"
    primary = _level_fit_dir(models, "lrp-rli-lf-006", link="logit")
    config = json.loads((primary / "config.json").read_text(encoding="utf-8"))
    failures = release._blending_pair_release_failures(primary, config)
    assert failures and "lrp-rli-lf-006 + lrp-rli-lf-106" in failures[0]
    _level_fit_dir(models, "lrp-rli-lf-106", link="three_choice_guessing_floor")
    assert release._blending_pair_release_failures(primary, config) == ()



# --- the gain family's pair (#596) --------------------------------------------


def _gain_fit_dir(
    root: Path,
    model_id: str,
    *,
    link: str,
    kind: str = "gain_factors",
    outcome_symbol: str = "B",
    config_name: str = "reporting",
    gate_passed: bool = True,
    digest: str = "ad7c861af4c22af5",
    data_sha256: str = "b" * 64,
    n_obs: int = 161,
    items_median: float = 0.835,
    required: bool = True,
) -> Path:
    """A minimal stored gain-factor B fit: config, gate verdict and ROPE card."""
    directory = root / f"{model_id}-{config_name}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "kind": kind,
                "outcome_symbol": outcome_symbol,
                "config_name": config_name,
                "data_sha256": data_sha256,
                "n_obs": n_obs,
                "fitted_data_identity": {"digest": digest},
                "resolved_run_plan": {
                    "score_mean_link": link,
                    "link_sensitivity_required_for_release": required,
                    "required_link_companion_model_id": (
                        "lrp-rli-gf-306"
                        if model_id == "lrp-rli-gf-006"
                        else "lrp-rli-gf-006"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )
    (directory / "diagnostics_summary.json").write_text(
        json.dumps(
            {
                "divergences": 0 if gate_passed else 47,
                "max_rhat": 1.0 if gate_passed else 1.9,
                "min_ess": 4000.0 if gate_passed else 12.0,
                "bfmi_per_chain": [0.9, 0.9] if gate_passed else [0.05, 0.9],
                "checks": {
                    "rhat": gate_passed,
                    "ess": gate_passed,
                    "divergences": gate_passed,
                    "bfmi": gate_passed,
                },
                "passed": gate_passed,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [{"items_median": items_median, "items_lo": 0.086, "items_hi": 1.58, "pd": 0.96}]
    ).to_csv(directory / "rope_summary.csv", index=False)
    return directory


def test_gain_pair_is_ready_when_both_links_are_fitted_on_the_same_rows(tmp_path):
    models = tmp_path / "models"
    primary = _gain_fit_dir(models, "lrp-rli-gf-006", link="logit")
    _gain_fit_dir(
        models,
        "lrp-rli-gf-306",
        link="three_choice_guessing_floor",
        items_median=0.44,
    )
    status = bs.evaluate_gain_blending_link_pair(primary)
    assert status["required"] and status["ready"], status
    cards = status["cards"]
    assert cards["lrp-rli-gf-006"]["score_mean_link"] == "logit"
    assert cards["lrp-rli-gf-306"]["items_median"] == 0.44
    # Either side of the pair sees the same verdict.
    assert bs.evaluate_gain_blending_link_pair(
        models / "lrp-rli-gf-306-reporting"
    )["ready"]


def test_gain_pair_is_not_ready_without_its_twin(tmp_path):
    models = tmp_path / "models"
    primary = _gain_fit_dir(models, "lrp-rli-gf-006", link="logit")
    status = bs.evaluate_gain_blending_link_pair(primary)
    assert status["required"] and not status["ready"]
    assert "not present beside this one" in status["reason"]


def test_gain_pair_requires_the_registered_pairing_even_on_a_stale_plan(tmp_path):
    """gf-006's stored reporting fit predates the pairing, so its plan carries no
    ``link_sensitivity_required_for_release``. The requirement is derived from the
    registered ids too, so that fit fails closed rather than releasing unpaired."""
    models = tmp_path / "models"
    primary = _gain_fit_dir(models, "lrp-rli-gf-006", link="logit")
    config = json.loads((primary / "config.json").read_text(encoding="utf-8"))
    config["resolved_run_plan"].pop("link_sensitivity_required_for_release")
    status = bs.evaluate_gain_blending_link_pair(primary, config=config)
    assert status["required"] and not status["ready"]


def test_gain_pair_rejects_an_unconverged_side(tmp_path):
    models = tmp_path / "models"
    primary = _gain_fit_dir(models, "lrp-rli-gf-006", link="logit")
    _gain_fit_dir(
        models,
        "lrp-rli-gf-306",
        link="three_choice_guessing_floor",
        gate_passed=False,
    )
    status = bs.evaluate_gain_blending_link_pair(primary)
    assert not status["ready"]
    assert "convergence gate" in status["reason"]


def test_gain_pair_rejects_different_fitted_rows(tmp_path):
    models = tmp_path / "models"
    primary = _gain_fit_dir(models, "lrp-rli-gf-006", link="logit")
    _gain_fit_dir(
        models,
        "lrp-rli-gf-306",
        link="three_choice_guessing_floor",
        digest="0th3r",
    )
    status = bs.evaluate_gain_blending_link_pair(primary)
    assert not status["ready"]
    assert "fitted rows" in status["reason"]


def test_gain_pair_rejects_two_fits_under_the_same_link(tmp_path):
    models = tmp_path / "models"
    primary = _gain_fit_dir(models, "lrp-rli-gf-006", link="logit")
    _gain_fit_dir(models, "lrp-rli-gf-306", link="logit")
    status = bs.evaluate_gain_blending_link_pair(primary)
    assert not status["ready"]
    assert "opposite score-mean links" in status["reason"]


def test_gain_pair_ignores_a_non_blending_gain_fit(tmp_path):
    models = tmp_path / "models"
    directory = _gain_fit_dir(
        models, "lrp-rli-gf-001", link="logit", outcome_symbol="W", required=False
    )
    status = bs.evaluate_gain_blending_link_pair(directory)
    assert not status["required"] and status["ready"]


def test_gain_pair_exempts_the_b_variants(tmp_path):
    """#596 scope decision: the treated-only and moderation variants publish no
    paired headline, so their plans do not declare the pairing and the gate lets
    them through rather than demanding floor twins that do not exist."""
    models = tmp_path / "models"
    for model_id in ("lrp-rli-gf-106", "lrp-rli-gf-206"):
        directory = _gain_fit_dir(models, model_id, link="logit", required=False)
        status = bs.evaluate_gain_blending_link_pair(directory)
        assert not status["required"] and status["ready"], model_id


def test_release_gate_withholds_an_unpaired_gain_blending_fit(tmp_path):
    from language_reading_predictors.statistical_models import release


    models = tmp_path / "models"
    primary = _gain_fit_dir(models, "lrp-rli-gf-006", link="logit")
    config = json.loads((primary / "config.json").read_text(encoding="utf-8"))
    failures = release._blending_pair_release_failures(primary, config)
    assert failures and "lrp-rli-gf-006 + lrp-rli-gf-306" in failures[0]
    _gain_fit_dir(models, "lrp-rli-gf-306", link="three_choice_guessing_floor")
    assert release._blending_pair_release_failures(primary, config) == ()


def test_release_gate_lets_the_exempt_gain_b_variants_through(tmp_path):
    from language_reading_predictors.statistical_models import release


    models = tmp_path / "models"
    for model_id in ("lrp-rli-gf-106", "lrp-rli-gf-206"):
        directory = _gain_fit_dir(models, model_id, link="logit", required=False)
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        assert release._blending_pair_release_failures(directory, config) == ()


# --- the aligned family's pair (#619) -----------------------------------------


def _aligned_fit_dir(
    root: Path,
    model_id: str,
    *,
    link: str,
    outcome_symbol: str = "B",
    config_name: str = "reporting",
    gate_passed: bool = True,
    digest: str = "a11gn3d",
    data_sha256: str = "c" * 64,
    n_obs: int = 54,
    items_median: float = 0.61,
    required: bool = True,
) -> Path:
    """A minimal stored aligned B fit: config, gate verdict and cohort marginal."""
    directory = root / f"{model_id}-{config_name}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "kind": "aligned",
                "outcome_symbol": outcome_symbol,
                "config_name": config_name,
                "data_sha256": data_sha256,
                "n_obs": n_obs,
                "fitted_data_identity": {"digest": digest},
                "resolved_run_plan": {
                    "score_mean_link": link,
                    "link_sensitivity_required_for_release": required,
                    "required_link_companion_model_id": (
                        "lrp-rli-al-306"
                        if model_id == "lrp-rli-al-006"
                        else "lrp-rli-al-006"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )
    (directory / "diagnostics_summary.json").write_text(
        json.dumps(
            {
                "divergences": 0 if gate_passed else 47,
                "max_rhat": 1.0 if gate_passed else 1.9,
                "min_ess": 4000.0 if gate_passed else 12.0,
                "bfmi_per_chain": [0.9, 0.9] if gate_passed else [0.05, 0.9],
                "checks": {
                    "rhat": gate_passed,
                    "ess": gate_passed,
                    "divergences": gate_passed,
                    "bfmi": gate_passed,
                },
                "passed": gate_passed,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "trt_items_median": items_median,
                "trt_items_lo": -0.4,
                "trt_items_hi": 1.7,
                "prob_trt_pos": 0.82,
            }
        ]
    ).to_csv(directory / "cohort_marginal.csv", index=False)
    return directory


def test_aligned_pair_is_ready_when_both_links_are_fitted_on_the_same_rows(tmp_path):
    models = tmp_path / "models"
    primary = _aligned_fit_dir(models, "lrp-rli-al-006", link="logit")
    _aligned_fit_dir(
        models,
        "lrp-rli-al-306",
        link="three_choice_guessing_floor",
        items_median=0.37,
    )
    status = bs.evaluate_aligned_blending_link_pair(primary, plan_checker=_plan_is_current)
    assert status["required"] and status["ready"], status
    assert status["cards"]["lrp-rli-al-306"]["items_median"] == 0.37


def test_aligned_pair_is_not_ready_without_its_twin(tmp_path):
    models = tmp_path / "models"
    primary = _aligned_fit_dir(models, "lrp-rli-al-006", link="logit")
    status = bs.evaluate_aligned_blending_link_pair(primary, plan_checker=_plan_is_current)
    assert status["required"] and not status["ready"]
    assert "not present beside this one" in status["reason"]


def test_aligned_pair_rejects_different_fitted_rows(tmp_path):
    models = tmp_path / "models"
    primary = _aligned_fit_dir(models, "lrp-rli-al-006", link="logit")
    _aligned_fit_dir(
        models, "lrp-rli-al-306", link="three_choice_guessing_floor", digest="0th3r"
    )
    status = bs.evaluate_aligned_blending_link_pair(primary, plan_checker=_plan_is_current)
    assert not status["ready"]
    assert "fitted rows" in status["reason"]


def test_aligned_pair_ignores_a_non_blending_aligned_fit(tmp_path):
    models = tmp_path / "models"
    directory = _aligned_fit_dir(
        models, "lrp-rli-al-001", link="logit", outcome_symbol="W", required=False
    )
    status = bs.evaluate_aligned_blending_link_pair(directory, plan_checker=_plan_is_current)
    assert not status["required"] and status["ready"]


def test_release_gate_withholds_an_unpaired_aligned_blending_fit(tmp_path):
    from language_reading_predictors.statistical_models import release


    models = tmp_path / "models"
    primary = _aligned_fit_dir(models, "lrp-rli-al-006", link="logit")
    config = json.loads((primary / "config.json").read_text(encoding="utf-8"))
    failures = release._blending_pair_release_failures(primary, config)
    assert failures and "lrp-rli-al-006 + lrp-rli-al-306" in failures[0]
    _aligned_fit_dir(models, "lrp-rli-al-306", link="three_choice_guessing_floor")
    assert release._blending_pair_release_failures(primary, config) == ()


# --- the concurrent family's pair (#619), whose card is a table ---------------


def _concurrent_fit_dir(
    root: Path,
    model_id: str,
    *,
    link: str,
    outcome_symbol: str = "B",
    config_name: str = "reporting",
    gate_passed: bool = True,
    digest: str = "c0ncurr3nt",
    data_sha256: str = "d" * 64,
    n_obs: int = 54,
    n_marginal_rows: int = 12,
    required: bool = True,
) -> Path:
    """A minimal stored concurrent B fit, whose card is a marginals *table*."""
    directory = root / f"{model_id}-{config_name}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "kind": "concurrent",
                "outcome_symbol": outcome_symbol,
                "config_name": config_name,
                "data_sha256": data_sha256,
                "n_obs": n_obs,
                "fitted_data_identity": {"digest": digest},
                "resolved_run_plan": {
                    "score_mean_link": link,
                    "link_sensitivity_required_for_release": required,
                    "required_link_companion_model_id": (
                        "lrp-rli-ca-307"
                        if model_id == "lrp-rli-ca-007"
                        else "lrp-rli-ca-007"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )
    (directory / "diagnostics_summary.json").write_text(
        json.dumps(
            {
                "divergences": 0 if gate_passed else 47,
                "max_rhat": 1.0 if gate_passed else 1.9,
                "min_ess": 4000.0 if gate_passed else 12.0,
                "bfmi_per_chain": [0.9, 0.9] if gate_passed else [0.05, 0.9],
                "checks": {
                    "rhat": gate_passed,
                    "ess": gate_passed,
                    "divergences": gate_passed,
                    "bfmi": gate_passed,
                },
                "passed": gate_passed,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"timepoint": 1, "term": f"X{i}", "scale": "+1 SD", "items_median": 0.1 * i}
            for i in range(n_marginal_rows)
        ]
    ).to_csv(directory / "concurrent_marginals.csv", index=False)
    return directory


def test_concurrent_pair_is_ready_without_a_scalar_card(tmp_path):
    """This family publishes a table and names no headline row, so the pairing rests
    on the identity evidence plus the table's shape."""
    models = tmp_path / "models"
    primary = _concurrent_fit_dir(models, "lrp-rli-ca-007", link="logit")
    _concurrent_fit_dir(models, "lrp-rli-ca-307", link="three_choice_guessing_floor")
    status = bs.evaluate_concurrent_blending_link_pair(primary, plan_checker=_plan_is_current)
    assert status["required"] and status["ready"], status
    card = status["cards"]["lrp-rli-ca-007"]
    assert card["card_rows"] == 12
    assert "items_median" not in card


def test_concurrent_pair_rejects_a_different_marginals_shape(tmp_path):
    """Two fits that produced different marginal sets are not the same analysis."""
    models = tmp_path / "models"
    primary = _concurrent_fit_dir(models, "lrp-rli-ca-007", link="logit")
    _concurrent_fit_dir(
        models,
        "lrp-rli-ca-307",
        link="three_choice_guessing_floor",
        n_marginal_rows=9,
    )
    status = bs.evaluate_concurrent_blending_link_pair(primary, plan_checker=_plan_is_current)
    assert not status["ready"]
    assert "concurrent marginals shape" in status["reason"]


def test_concurrent_pair_rejects_an_empty_marginals_table(tmp_path):
    models = tmp_path / "models"
    primary = _concurrent_fit_dir(models, "lrp-rli-ca-007", link="logit")
    companion = _concurrent_fit_dir(
        models, "lrp-rli-ca-307", link="three_choice_guessing_floor"
    )
    pd.DataFrame(columns=["timepoint", "term"]).to_csv(
        companion / "concurrent_marginals.csv", index=False
    )
    status = bs.evaluate_concurrent_blending_link_pair(primary, plan_checker=_plan_is_current)
    assert not status["ready"]
    assert "empty" in status["reason"]


def test_concurrent_pair_ignores_a_fit_where_b_is_only_a_predictor(tmp_path):
    """The link governs blending as the OUTCOME. The six siblings that carry B as a
    standardised logit predictor model no B score mean, so they are out of scope."""
    models = tmp_path / "models"
    directory = _concurrent_fit_dir(
        models, "lrp-rli-ca-001", link="logit", outcome_symbol="W", required=False
    )
    status = bs.evaluate_concurrent_blending_link_pair(directory, plan_checker=_plan_is_current)
    assert not status["required"] and status["ready"]


def test_release_gate_withholds_an_unpaired_concurrent_blending_fit(tmp_path):
    from language_reading_predictors.statistical_models import release


    models = tmp_path / "models"
    primary = _concurrent_fit_dir(models, "lrp-rli-ca-007", link="logit")
    config = json.loads((primary / "config.json").read_text(encoding="utf-8"))
    failures = release._blending_pair_release_failures(primary, config)
    assert failures and "lrp-rli-ca-007 + lrp-rli-ca-307" in failures[0]
    _concurrent_fit_dir(models, "lrp-rli-ca-307", link="three_choice_guessing_floor")
    assert release._blending_pair_release_failures(primary, config) == ()


def test_every_natural_scale_reporting_helper_accepts_the_score_mean_link():
    """The quiet failure mode this whole policy guards against (#619).

    A helper that turns ``eta`` into probability/items output but takes no
    ``score_mean_link`` will silently summarise a floor-link posterior on the
    ordinary link — a wrong number wearing the right label, in a published CSV. That
    is exactly what ``treatment_marginal_effect``, ``association_marginals``,
    ``concurrent_marginals`` and ``marginal_prior_pushforward`` each did until they
    were fixed, and ``concurrent_marginals`` was missed once because a stale
    line-number lookup made it *look* already done.

    Pinning the inventory means a new natural-scale helper has to opt in explicitly,
    rather than defaulting to a hidden ordinary-link assumption.
    """
    import inspect

    from language_reading_predictors.statistical_models import (
        figure_artifacts as _figures,
    )
    from language_reading_predictors.statistical_models import (
        predicted_scores as _predicted,
    )
    from language_reading_predictors.statistical_models import reporting as _reporting

    required = {
        _reporting: (
            "tau_summary_itt",
            "rope_summary",
            "rope_sensitivity",
            "prior_pushforward",
            "marginal_prior_pushforward",
            "did_summary",
            "treatment_marginal_effect",
            "association_marginals",
            "concurrent_marginals",
            "level_t2_marginal_effect",
        ),
        _figures: ("save_rope_plot",),
        _predicted: ("write_predicted_scores_artifacts",),
    }
    missing = [
        f"{module.__name__.rsplit('.', 1)[-1]}.{name}"
        for module, names in required.items()
        for name in names
        if "score_mean_link"
        not in inspect.signature(getattr(module, name)).parameters
    ]
    assert not missing, (
        "these natural-scale helpers take no score_mean_link, so they would "
        f"publish ordinary-link numbers from a floor-link posterior: {missing}"
    )


# --- the dose family's pair (#619) --------------------------------------------


def _dose_fit_dir(
    root: Path,
    model_id: str,
    *,
    link: str,
    outcome_symbol: str = "B",
    config_name: str = "reporting",
    gate_passed: bool = True,
    digest: str = "d05e",
    data_sha256: str = "e" * 64,
    n_obs: int = 160,
    items_median: float = 0.017,
    required: bool = True,
) -> Path:
    """A minimal stored dose-response B fit, carded by its dose marginal summary."""
    directory = root / f"{model_id}-{config_name}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "kind": "dose_response",
                "outcome_symbol": outcome_symbol,
                "config_name": config_name,
                "data_sha256": data_sha256,
                "n_obs": n_obs,
                "fitted_data_identity": {"digest": digest},
                "resolved_run_plan": {
                    "score_mean_link": link,
                    "link_sensitivity_required_for_release": required,
                    "required_link_companion_model_id": (
                        "lrp-rli-dose-384"
                        if model_id == "lrp-rli-dose-084"
                        else "lrp-rli-dose-084"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )
    (directory / "diagnostics_summary.json").write_text(
        json.dumps(
            {
                "divergences": 0 if gate_passed else 47,
                "max_rhat": 1.0 if gate_passed else 1.9,
                "min_ess": 4000.0 if gate_passed else 12.0,
                "bfmi_per_chain": [0.9, 0.9] if gate_passed else [0.05, 0.9],
                "checks": {
                    "rhat": gate_passed,
                    "ess": gate_passed,
                    "divergences": gate_passed,
                    "bfmi": gate_passed,
                },
                "passed": gate_passed,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "items_median": items_median,
                "items_lo": -0.74,
                "items_hi": 0.72,
                "prob_pos": 0.515,
            }
        ]
    ).to_csv(directory / "dose_marginal_summary.csv", index=False)
    return directory


def test_dose_pair_is_ready_when_both_links_are_fitted_on_the_same_rows(tmp_path):
    models = tmp_path / "models"
    primary = _dose_fit_dir(models, "lrp-rli-dose-084", link="logit")
    _dose_fit_dir(
        models,
        "lrp-rli-dose-384",
        link="three_choice_guessing_floor",
        items_median=0.009,
    )
    status = bs.evaluate_dose_blending_link_pair(primary, plan_checker=_plan_is_current)
    assert status["required"] and status["ready"], status
    assert status["cards"]["lrp-rli-dose-384"]["items_median"] == 0.009


def test_dose_pair_is_not_ready_without_its_twin(tmp_path):
    models = tmp_path / "models"
    primary = _dose_fit_dir(models, "lrp-rli-dose-084", link="logit")
    status = bs.evaluate_dose_blending_link_pair(primary, plan_checker=_plan_is_current)
    assert status["required"] and not status["ready"]
    assert "not present beside this one" in status["reason"]


def test_dose_pair_rejects_different_fitted_rows(tmp_path):
    models = tmp_path / "models"
    primary = _dose_fit_dir(models, "lrp-rli-dose-084", link="logit")
    _dose_fit_dir(
        models, "lrp-rli-dose-384", link="three_choice_guessing_floor", digest="0th3r"
    )
    status = bs.evaluate_dose_blending_link_pair(primary, plan_checker=_plan_is_current)
    assert not status["ready"]
    assert "fitted rows" in status["reason"]


def test_dose_pair_ignores_a_non_blending_dose_fit(tmp_path):
    models = tmp_path / "models"
    directory = _dose_fit_dir(
        models, "lrp-rli-dose-077", link="logit", outcome_symbol="W", required=False
    )
    status = bs.evaluate_dose_blending_link_pair(directory, plan_checker=_plan_is_current)
    assert not status["required"] and status["ready"]


def test_release_gate_withholds_an_unpaired_dose_blending_fit(tmp_path):
    from language_reading_predictors.statistical_models import release


    models = tmp_path / "models"
    primary = _dose_fit_dir(models, "lrp-rli-dose-084", link="logit")
    config = json.loads((primary / "config.json").read_text(encoding="utf-8"))
    failures = release._blending_pair_release_failures(primary, config)
    assert failures and "lrp-rli-dose-084 + lrp-rli-dose-384" in failures[0]
    _dose_fit_dir(models, "lrp-rli-dose-384", link="three_choice_guessing_floor")
    assert release._blending_pair_release_failures(primary, config) == ()


# --- the mediation family's pair, and the symbol-keyed default (#619) ---------


def _mediation_fit_dir(
    root: Path,
    model_id: str,
    *,
    link: str,
    outcome_symbol: str = "B",
    config_name: str = "reporting",
    gate_passed: bool = True,
    digest: str = "med87",
    data_sha256: str = "f" * 64,
    n_obs: int = 53,
    total_items: float = 0.674,
    n_quantities: int = 4,
    required: bool = True,
) -> Path:
    """A minimal stored mediation B fit, carded by its decomposition table."""
    directory = root / f"{model_id}-{config_name}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "kind": "mediation",
                "outcome_symbol": outcome_symbol,
                "config_name": config_name,
                "data_sha256": data_sha256,
                "n_obs": n_obs,
                "fitted_data_identity": {"digest": digest},
                "resolved_run_plan": {
                    "score_mean_link": link,
                    "link_sensitivity_required_for_release": required,
                    "required_link_companion_model_id": (
                        "lrp-rli-med-387"
                        if model_id == "lrp-rli-med-087"
                        else "lrp-rli-med-087"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )
    (directory / "diagnostics_summary.json").write_text(
        json.dumps(
            {
                "divergences": 0 if gate_passed else 47,
                "max_rhat": 1.0 if gate_passed else 1.9,
                "min_ess": 4000.0 if gate_passed else 12.0,
                "bfmi_per_chain": [0.9, 0.9] if gate_passed else [0.05, 0.9],
                "checks": {
                    "rhat": gate_passed,
                    "ess": gate_passed,
                    "divergences": gate_passed,
                    "bfmi": gate_passed,
                },
                "passed": gate_passed,
            }
        ),
        encoding="utf-8",
    )
    quantities = ["total", "NDE", "NIE", "proportion_mediated"][:n_quantities]
    pd.DataFrame(
        [
            {
                "quantity": q,
                "words_median": total_items if q == "total" else 0.1,
                "words_lo": -0.148,
                "words_hi": 1.498,
                "prob_pos": 0.908,
            }
            for q in quantities
        ]
    ).to_csv(directory / "mediation_summary.csv", index=False)
    return directory


def test_mediation_pair_selects_the_total_row_from_the_decomposition(tmp_path):
    """The card is a multi-row decomposition; the headline is the ``total`` row, so
    the audit record shows the number the report leads with rather than a
    shape-only check."""
    models = tmp_path / "models"
    primary = _mediation_fit_dir(models, "lrp-rli-med-087", link="logit")
    _mediation_fit_dir(
        models,
        "lrp-rli-med-387",
        link="three_choice_guessing_floor",
        total_items=0.41,
    )
    status = bs.evaluate_mediation_blending_link_pair(primary, plan_checker=_plan_is_current)
    assert status["required"] and status["ready"], status
    assert status["cards"]["lrp-rli-med-087"]["items_median"] == 0.674
    assert status["cards"]["lrp-rli-med-387"]["items_median"] == 0.41
    assert status["cards"]["lrp-rli-med-087"]["card_rows"] == 4


def test_mediation_pair_rejects_a_decomposition_of_a_different_shape(tmp_path):
    """A twin reporting a different set of quantities is not the same decomposition."""
    models = tmp_path / "models"
    primary = _mediation_fit_dir(models, "lrp-rli-med-087", link="logit")
    _mediation_fit_dir(
        models,
        "lrp-rli-med-387",
        link="three_choice_guessing_floor",
        n_quantities=3,
    )
    status = bs.evaluate_mediation_blending_link_pair(primary, plan_checker=_plan_is_current)
    assert not status["ready"]
    assert "mediation summary shape" in status["reason"]


def test_mediation_pair_rejects_a_card_with_no_total_row(tmp_path):
    models = tmp_path / "models"
    primary = _mediation_fit_dir(models, "lrp-rli-med-087", link="logit")
    companion = _mediation_fit_dir(
        models, "lrp-rli-med-387", link="three_choice_guessing_floor"
    )
    table = pd.read_csv(companion / "mediation_summary.csv")
    table["quantity"] = table["quantity"].replace({"total": "IDE"})
    table.to_csv(companion / "mediation_summary.csv", index=False)
    status = bs.evaluate_mediation_blending_link_pair(primary, plan_checker=_plan_is_current)
    assert not status["ready"]
    assert "exactly one quantity='total' row" in status["reason"]


def test_mediation_pair_is_not_ready_without_its_twin(tmp_path):
    models = tmp_path / "models"
    primary = _mediation_fit_dir(models, "lrp-rli-med-087", link="logit")
    status = bs.evaluate_mediation_blending_link_pair(primary, plan_checker=_plan_is_current)
    assert status["required"] and not status["ready"]
    assert "not present beside this one" in status["reason"]


def test_mediation_pair_exempts_the_interventional_relabelling(tmp_path):
    """med-187 declares companion_of and reproduces med-087's numbers under an
    interventional relabelling, so the pairing governs the parent, not the alias."""
    models = tmp_path / "models"
    directory = _mediation_fit_dir(
        models, "lrp-rli-med-187", link="logit", required=False
    )
    status = bs.evaluate_mediation_blending_link_pair(directory, plan_checker=_plan_is_current)
    assert not status["required"] and status["ready"]


def test_release_gate_withholds_an_unpaired_mediation_blending_fit(tmp_path):
    from language_reading_predictors.statistical_models import release


    models = tmp_path / "models"
    primary = _mediation_fit_dir(models, "lrp-rli-med-087", link="logit")
    config = json.loads((primary / "config.json").read_text(encoding="utf-8"))
    failures = release._blending_pair_release_failures(primary, config)
    assert failures and "lrp-rli-med-087 + lrp-rli-med-387" in failures[0]
    _mediation_fit_dir(models, "lrp-rli-med-387", link="three_choice_guessing_floor")
    assert release._blending_pair_release_failures(primary, config) == ()


def test_a_blending_fit_in_an_ungated_family_fails_closed(tmp_path):
    """#608 decision 1, implemented in #619: the gate's default is keyed on the
    OUTCOME SYMBOL, not on ``kind``.

    Before this, the dispatch returned early for every unlisted kind — which is how
    four families published unpaired ``B`` results for months without anything
    failing. A ``B`` fit in a family with no registered pairing is a fit whose
    response-link sensitivity nothing can verify, so it must not publish.
    """
    from language_reading_predictors.statistical_models import release

    failures = release._blending_pair_release_failures(
        tmp_path,
        {"kind": "pooled_levels", "outcome_symbol": "B", "model_id": "lrp-rli-pl-999"},
    )
    assert failures
    assert "no registered response-link pair gate" in failures[0]
    # A non-B outcome in the same ungated family is untouched.
    assert (
        release._blending_pair_release_failures(
            tmp_path,
            {
                "kind": "pooled_levels",
                "outcome_symbol": "W",
                "model_id": "lrp-rli-pl-001",
            },
        )
        == ()
    )


def test_every_family_registering_a_blending_model_has_a_gate():
    """The fail-closed default must not be load-bearing for anything registered.

    If a registered ``B`` model's family had no gate, the default above would
    withhold it — correct, but a surprise discovered at release time rather than
    here. This asserts the two stay in step.
    """
    import glob
    import importlib
    import os

    from language_reading_predictors.statistical_models import release

    root = os.path.dirname(
        importlib.import_module(
            "language_reading_predictors.statistical_models.mediation"
        ).__file__
    )
    gated = set(release._BLENDING_PAIR_GATES) | {"itt"}
    ungated = set()
    for path in sorted(glob.glob(os.path.join(root, "lrp_*.py"))):
        module = importlib.import_module(
            "language_reading_predictors.statistical_models."
            + os.path.basename(path)[:-3]
        )
        spec = getattr(module, "SPEC", None)
        if spec is not None and getattr(spec, "outcome_symbol", None) == "B":
            if spec.kind not in gated:
                ungated.add((spec.kind, spec.model_id))
    assert not ungated, (
        "these registered B models are in families with no pair gate, so they would "
        f"fail closed at release: {sorted(ungated)}"
    )


# --- the run-plan + provenance binding (#608 decision 2, as amended) ----------


@pytest.fixture(autouse=True)
def monkeypatch_plan_currency_fixture(monkeypatch, request):
    """Stub the run-plan currency check for every fixture-based test in this module.

    The synthetic fits here carry stub run plans, so the real checker -- which
    re-resolves each model's module -- would report every stub field as stale and
    fail tests that are about something else entirely.

    A test that is *about* the checker opts out with ``@pytest.mark.real_plan_currency``.
    Without that escape hatch those tests would be stubbed too and would pass
    vacuously, which is worse than not having them.
    """
    if request.node.get_closest_marker("real_plan_currency") is not None:
        return
    from language_reading_predictors.statistical_models import blending_sensitivity

    monkeypatch.setattr(
        blending_sensitivity, "_stale_plan_fields", lambda *_a, **_k: []
    )


def _plan_is_current(model_id: str, kind: str, stored):
    """A currency checker that reports no staleness.

    The synthetic fixtures in this module carry stub run plans, so the real checker
    -- which re-resolves the module and would report every stub field as stale --
    cannot be used against them. Injecting this is the same idiom
    ``evaluate_blending_link_sensitivity`` already uses for its trace and row-map
    validators; the staleness behaviour itself is tested against the real checker
    below and against the two genuinely stale stored fits in the repository.
    """
    return []


def _plan_pair(tmp_path, *, primary_plan=None, companion_plan=None):
    """A ready gain pair whose two halves' stored run plans can be perturbed."""
    models = tmp_path / "models"
    primary = _gain_fit_dir(models, "lrp-rli-gf-006", link="logit")
    companion = _gain_fit_dir(
        models, "lrp-rli-gf-306", link="three_choice_guessing_floor"
    )
    for directory, extra in ((primary, primary_plan), (companion, companion_plan)):
        if not extra:
            continue
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        config["resolved_run_plan"].update(extra)
        (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return primary


def test_pair_rejects_halves_whose_run_plans_differ_beyond_the_link(monkeypatch):
    """The defect this check exists for (#619 review).

    A stored ``lrp-rli-med-087`` fitted before #600 lacked the g-formula's per-leg
    baseline terms; its companion, built from current code, had them. Every check
    that existed then passed — data, fitted rows, sampling config, card shape — and
    the published comparison was confounded between the link and the model. What
    separates the two is the resolved run plan, and nothing was comparing it.
    """
    from language_reading_predictors.statistical_models import blending_sensitivity

    a = {"skill_symbols": ["L", "E", "TE"], "adjust_for": ["hs"]}
    b = {"skill_symbols": ["L", "E", "TE"], "adjust_for": ["hs", "erbto"]}
    assert blending_sensitivity._comparable_plan(a) != blending_sensitivity._comparable_plan(b)
    mismatched = sorted(
        name
        for name in set(a) | set(b)
        if blending_sensitivity._normalise_plan_value(a.get(name))
        != blending_sensitivity._normalise_plan_value(b.get(name))
    )
    assert mismatched == ["adjust_for"]


def test_the_plan_comparison_survives_the_json_round_trip():
    """Resolvers produce tuples; ``config.json`` returns lists.

    Comparing them raw reports every tuple-valued field as drifted, which is a fact
    about serialisation and would drown the real signal. This is the normalisation
    that keeps the check meaningful — an earlier draft of it flagged fits made
    minutes before, which is how the bug was noticed.
    """
    from language_reading_predictors.statistical_models import blending_sensitivity

    resolved = {"adjust_for": ("hs", "hs_missing"), "nested": ({"a": (1, 2)},)}
    stored = json.loads(json.dumps({"adjust_for": ["hs", "hs_missing"],
                                    "nested": [{"a": [1, 2]}]}))
    assert blending_sensitivity._comparable_plan(resolved) == (
        blending_sensitivity._comparable_plan(stored)
    )


def test_the_link_and_prose_fields_are_excluded_from_the_comparison():
    """A pair MUST differ in the link and in the pairing bookkeeping derived from
    it, and its generated prose embeds the link clause. Comparing those would fail
    every pair for being a pair."""
    from language_reading_predictors.statistical_models import blending_sensitivity

    logit = {
        "score_mean_link": "logit",
        "required_link_companion_model_id": "lrp-rli-gf-306",
        "design": "... ordinary inverse-logit ...",
        "settings_source": "legacy_extra",
        "adjust_for": ("hs",),
    }
    floor = {
        "score_mean_link": "three_choice_guessing_floor",
        "required_link_companion_model_id": "lrp-rli-gf-006",
        "design": "... mapped onto [1/3, 1] ...",
        "settings_source": "typed",
        "adjust_for": ("hs",),
    }
    assert blending_sensitivity._comparable_plan(logit) == (
        blending_sensitivity._comparable_plan(floor)
    )


def test_every_gated_family_has_a_run_plan_resolver():
    """The currency check fails closed when a family has no resolver, so a gated
    family without one would withhold its pair. The two maps must stay in step."""
    from language_reading_predictors.statistical_models import blending_sensitivity
    from language_reading_predictors.statistical_models import release

    missing = sorted(set(release._BLENDING_PAIR_GATES) - set(blending_sensitivity._PLAN_RESOLVERS))
    assert not missing, (
        f"these gated families have no run-plan resolver, so their pairs would fail "
        f"the staleness check: {missing}"
    )


def test_provenance_is_recorded_and_surfaced_but_not_required_to_match():
    """A companion is registered in a later commit than its primary by construction,
    so requiring one commit would fail-close five of six live pairs for a fact about
    git history. The note says where each half came from instead."""
    from language_reading_predictors.statistical_models import blending_sensitivity

    note = blending_sensitivity._pair_provenance_note(
        {
            "model_id": "lrp-rli-gf-006",
            "source_commit": "a" * 40,
            "source_dirty": False,
            "environment_lock_sha256": "e" * 64,
        },
        {
            "model_id": "lrp-rli-gf-306",
            "source_commit": "b" * 40,
            "source_dirty": True,
            "environment_lock_sha256": "f" * 64,
        },
    )
    assert "different source commits" in note
    assert "uncommitted changes" in note and "lrp-rli-gf-306" in note
    assert "different environment locks" in note
    # Identical provenance says nothing at all.
    same = {
        "model_id": "x",
        "source_commit": "a" * 40,
        "source_dirty": False,
        "environment_lock_sha256": "e" * 64,
    }
    assert blending_sensitivity._pair_provenance_note(same, {**same, "model_id": "y"}) == ""


def test_a_ready_pair_does_not_leak_private_card_fields(tmp_path):
    """The comparable plan is an implementation detail of the check, not part of the
    published card."""
    models = tmp_path / "models"
    primary = _gain_fit_dir(models, "lrp-rli-gf-006", link="logit")
    _gain_fit_dir(models, "lrp-rli-gf-306", link="three_choice_guessing_floor")
    status = bs.evaluate_gain_blending_link_pair(primary, plan_checker=_plan_is_current)
    for card in status["cards"].values():
        assert not any(name.startswith("_") for name in card), sorted(card)
        assert "source_commit" in card and "environment_lock_sha256" in card


@pytest.mark.real_plan_currency
def test_the_real_currency_checker_passes_a_current_fit():
    """Exercised against the real resolver, not the stub the fixture installs.

    Originally pinned to ``lrp-rli-gf-306``, which went deliberately plan-stale
    when #575 changed the gain family's settings ahead of the refit batch — a
    pinned exemplar turns this test into a canary for whichever family last
    changed, which is the currency *checker*'s job, not this test's. Instead,
    walk the pair families' stored reporting fits and require that the checker
    passes at least one genuinely current real fit (post-batch: all of them).
    """
    import json as _json

    from language_reading_predictors.statistical_models import blending_sensitivity
    from language_reading_predictors.statistical_models.registry import (
        discover_models,
    )

    models_root = Path("output/statistical_models/models")
    if not models_root.is_dir():
        pytest.skip("no stored fits in this checkout")
    checked = 0
    for model_id, lazy in sorted(discover_models().items()):
        directory = models_root / f"{model_id}-reporting"
        if not (directory / "config.json").is_file():
            continue
        try:
            spec = lazy.load().SPEC
        except Exception:  # noqa: BLE001 - not this test's concern
            continue
        if spec.kind not in blending_sensitivity._PLAN_RESOLVERS:
            continue
        config = _json.loads((directory / "config.json").read_text(encoding="utf-8"))
        plan = config.get("resolved_run_plan")
        if not plan:
            continue
        stored = blending_sensitivity._comparable_plan(plan)
        checked += 1
        if not blending_sensitivity._stale_plan_fields(model_id, spec.kind, stored):
            return  # the checker passed a real, current fit
    if checked == 0:
        pytest.skip("no pair-family reporting fit with a stored plan is present")
    pytest.fail(
        f"none of the {checked} stored pair-family reporting fits is plan-current; "
        "either every family changed without its refit batch, or the checker "
        "broke — both need attention"
    )


@pytest.mark.real_plan_currency
def test_the_real_currency_checker_catches_a_stale_stored_plan():
    """The check earns its place on real evidence, not a synthetic perturbation.

    ``lrp-rli-lf-006-reporting.pre-594-estimand-20260825`` is the fit this repository
    actually published before #619's review: its card was produced by the estimand
    #594 superseded, and it also predates #598's dispersion and child-SD priors.
    Nothing in the pipeline flagged it. This asserts the checker names those fields.
    """
    import json as _json

    from language_reading_predictors.statistical_models import blending_sensitivity

    directory = Path(
        "output/statistical_models/models/"
        "lrp-rli-lf-006-reporting.pre-594-estimand-20260825"
    )
    if not (directory / "config.json").is_file():
        pytest.skip("the pre-#594 lf-006 backup is not present in this checkout")
    config = _json.loads((directory / "config.json").read_text(encoding="utf-8"))
    stored = blending_sensitivity._comparable_plan(config["resolved_run_plan"])
    stale = blending_sensitivity._stale_plan_fields(
        "lrp-rli-lf-006", "level_factors", stored
    )
    assert "standardisation_balance_term" in stale, stale
    assert "sigma_child_prior_sigma" in stale, stale


@pytest.mark.real_plan_currency
def test_the_currency_checker_fails_closed_for_a_family_with_no_resolver():
    """A check that cannot run must refuse, not pass silently.

    The kind here is deliberately fictional. Since #637 stage 4 the resolver
    lookup is derived from the family descriptors, so *every* registered family
    has one — including ``pooled_levels``, which this test used to name because
    it was missing from a hand-maintained seven-entry subset. That was a
    bookkeeping gap rather than a scientific one; the fail-closed behaviour it
    exercised is still required for a kind the package does not describe.
    """
    from language_reading_predictors.statistical_models import blending_sensitivity

    with pytest.raises(ValueError, match="no registered run-plan resolver"):
        blending_sensitivity._stale_plan_fields("lrp-rli-xx-001", "not_a_family", {})


def test_the_currency_checker_now_covers_every_registered_family():
    """The subset is gone: a gated family cannot be absent from the lookup."""
    from language_reading_predictors.statistical_models import blending_sensitivity
    from language_reading_predictors.statistical_models import definitions

    assert set(blending_sensitivity._PLAN_RESOLVERS) == set(definitions.KINDS)
