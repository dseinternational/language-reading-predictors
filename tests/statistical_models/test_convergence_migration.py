# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Read both the 0.13 diagnostic contract and conservative legacy summaries."""

import pytest

from language_reading_predictors.statistical_models.convergence import convergence_gate_failures


def _summary():
    return dict(
        passed=True,
        scan_completed=True,
        max_rhat=1.0,
        min_ess=900.0,
        divergences=0,
        bfmi_per_chain=[0.8, 0.9],
        unassessable_parameters=[],
        checks=dict(rhat=True, ess=True, divergences=True, bfmi=True, diagnostics_assessable=True),
    )


def test_completed_unassessable_scan_is_not_reported_as_failed_scan():
    summary = _summary()
    summary.update(passed=False, max_rhat=None, min_ess=None, unassessable_parameters=["stuck"])
    summary["checks"].update(rhat=False, ess=False, diagnostics_assessable=False)
    failures = convergence_gate_failures(summary)
    assert "parameter diagnostics could not be assessed" in failures
    assert "convergence summary incomplete" not in failures
    assert not any("scan failed" in failure for failure in failures)


@pytest.mark.parametrize("scan", [False, None, "true"])
def test_failed_empty_or_invalid_scan_cannot_pass(scan):
    summary = _summary()
    summary["scan_completed"] = scan
    assert convergence_gate_failures(summary) == ["diagnostic scan failed or returned no parameters"]


def test_unassessable_names_override_inconsistent_pass():
    summary = _summary()
    summary["unassessable_parameters"] = ["stuck"]
    assert convergence_gate_failures(summary) == ["parameter diagnostics could not be assessed"]


def test_legacy_summary_requires_numeric_extrema():
    summary = _summary()
    del summary["scan_completed"]
    assert convergence_gate_failures(summary) == []
    summary["max_rhat"] = None
    assert convergence_gate_failures(summary) == ["convergence summary incomplete"]


def test_completed_assessable_scan_passes_and_null_bfmi_fails():
    summary = _summary()
    assert convergence_gate_failures(summary) == []
    summary["bfmi_per_chain"] = [None, 0.8]
    assert convergence_gate_failures(summary) == ["sampling energy (BFMI)"]


@pytest.mark.parametrize("mode", ["failed", "empty", "unassessable"])
def test_real_writer_scan_status_and_json_nulls(tmp_path, monkeypatch, mode):
    import json
    from types import SimpleNamespace
    import numpy as np
    import pandas as pd
    from dse_research_utils.statistics import diagnostics as shared

    def scan(*args, **kwargs):
        if mode == "failed":
            raise ValueError("test scan failure")
        return pd.DataFrame({"r_hat": [np.nan], "ess_bulk": [np.nan], "ess_tail": [np.nan]}, index=["stuck"]).iloc[
            : 0 if mode == "empty" else 1
        ]

    monkeypatch.setattr(shared.az, "summary", scan)
    monkeypatch.setattr(shared, "_bfmi_per_chain", lambda _: np.array([0.8, 0.9]))
    summary = shared.write_diagnostics_summary(SimpleNamespace(), str(tmp_path))
    stored = json.loads(
        (tmp_path / "diagnostics_summary.json").read_text(),
        parse_constant=lambda text: pytest.fail(f"invalid JSON constant {text}"),
    )
    assert summary == stored
    assert stored["scan_completed"] is (mode == "unassessable")
    assert stored["max_rhat"] is None and stored["min_ess"] is None
    assert stored["passed"] is False
    assert convergence_gate_failures(stored)
