# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Release-contract tests for the phoneme-blending response-link pair."""

from __future__ import annotations

import json
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
