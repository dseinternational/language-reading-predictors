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
    status = bs.evaluate_level_blending_link_pair(primary)
    assert status["required"] and status["ready"], status
    cards = status["cards"]
    assert cards["lrp-rli-lf-006"]["score_mean_link"] == "logit"
    assert cards["lrp-rli-lf-106"]["items_median"] == 0.43
    # Either side of the pair sees the same verdict.
    companion_status = bs.evaluate_level_blending_link_pair(
        models / "lrp-rli-lf-106-reporting"
    )
    assert companion_status["ready"]


def test_level_pair_is_not_ready_without_its_twin(tmp_path):
    models = tmp_path / "models"
    primary = _level_fit_dir(models, "lrp-rli-lf-006", link="logit")
    status = bs.evaluate_level_blending_link_pair(primary)
    assert status["required"] and not status["ready"]
    assert "not present beside this one" in status["reason"]


def test_level_pair_requires_the_registered_pairing_even_on_a_stale_plan(tmp_path):
    """The requirement is derived from the registered ids as well as the stored
    plan, so a fit whose plan predates the pairing cannot bypass the gate."""
    models = tmp_path / "models"
    primary = _level_fit_dir(models, "lrp-rli-lf-006", link="logit")
    config = json.loads((primary / "config.json").read_text(encoding="utf-8"))
    config["resolved_run_plan"].pop("link_sensitivity_required_for_release")
    status = bs.evaluate_level_blending_link_pair(primary, config=config)
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
    status = bs.evaluate_level_blending_link_pair(primary)
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
    status = bs.evaluate_level_blending_link_pair(primary)
    assert not status["ready"]
    assert "fitted rows" in status["reason"]


def test_level_pair_rejects_two_fits_under_the_same_link(tmp_path):
    models = tmp_path / "models"
    primary = _level_fit_dir(models, "lrp-rli-lf-006", link="logit")
    _level_fit_dir(models, "lrp-rli-lf-106", link="logit")
    status = bs.evaluate_level_blending_link_pair(primary)
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
    status = bs.evaluate_level_blending_link_pair(directory)
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
