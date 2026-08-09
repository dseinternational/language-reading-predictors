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

import pytest

from language_reading_predictors.statistical_models.artifacts import (
    ArtifactLog,
    ArtifactRecord,
    save_table,
)
from language_reading_predictors.statistical_models.release import (
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


def _fit_dir(tmp_path: Path, *, gate_passed: bool = True, kind: str = "mechanism") -> Path:
    """A minimal stored fit: a gate payload and a config, nothing else.

    ``mechanism`` is deliberately an *ungated* family, so the robustness stage is
    out of scope and each test exercises the stage it names.
    """
    d = tmp_path / "lrp-test-001-dev"
    d.mkdir(parents=True)
    (d / "diagnostics_summary.json").write_text(json.dumps(_gate(gate_passed)))
    (d / "config.json").write_text(
        json.dumps({"model_id": "lrp-test-001", "kind": kind, "outcome_symbol": "W"})
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


# --- the stages, and their order ------------------------------------------


def test_a_clean_ungated_fit_publishes(tmp_path):
    decision = evaluate_publication(_fit_dir(tmp_path))
    assert decision.publishable and decision.status == "ok"
    assert decision.robustness is None  # ungated family, no robustness verdict
    assert decision.config["model_id"] == "lrp-test-001"


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
    "status", ["gate_failed", "robustness_unresolved", "artifacts_incomplete"]
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
