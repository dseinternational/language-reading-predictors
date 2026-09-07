# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The research report's shared data access, exercised as the report runs it (#662).

``docs/report/_report_data.qmd`` is a Quarto include: its Python block is the
report's only route to fitted output, and nothing in the Python suite imports it.
These tests execute that block and drive the three file states the shared
``report.readers`` facts distinguish — present, missing and *present but
unparsable* — because the third used to be indistinguishable from the second and
therefore rendered as a reassuring "pending fit" placeholder.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
QMD = REPO / "docs/report/_report_data.qmd"


def _report_namespace(out_root: Path) -> dict:
    """Execute the report's setup block against a temporary output root."""
    source = QMD.read_text(encoding="utf-8")
    blocks = re.findall(r"```\{python\}\n(.*?)```", source, flags=re.DOTALL)
    assert len(blocks) == 1, "the shared include should hold exactly one code block"
    namespace: dict = {"__name__": "_report_data"}
    exec(compile(blocks[0], str(QMD), "exec"), namespace)
    from dse_research_utils.report.data import ReportData

    namespace["OUT_ROOT"] = out_root
    namespace["_report"] = ReportData(
        lambda model_id, config: out_root / f"{model_id}-{config}",
        default_config="reporting",
    )
    return namespace


def _fit_dir(out_root: Path, model_id: str) -> Path:
    directory = out_root / f"{model_id}-reporting"
    directory.mkdir(parents=True)
    return directory


@pytest.fixture
def report(tmp_path):
    return _report_namespace(tmp_path)


def _text(rendered) -> str:
    return getattr(rendered, "data", str(rendered))


def test_a_missing_fit_still_renders_the_pending_placeholder(report, tmp_path):
    assert report["load_summary"]("absent", "rope_summary") is None
    assert report["load_diagnostics"]("absent") is None
    assert report["unreadable_artifacts"] == {}
    rendered = _text(report["show_gated"]("absent", None, "the word-reading card"))
    assert "not been fitted" in rendered or "pending" in rendered.lower()


def test_a_clean_gate_publishes_a_readable_table(report, tmp_path):
    directory = _fit_dir(tmp_path, "clean")
    (directory / "rope_summary.csv").write_text(
        "quantity,median\ntau,0.42\n", encoding="utf-8"
    )
    (directory / "diagnostics_summary.json").write_text(
        json.dumps(
            {
                "passed": True,
                "scan_completed": True,
                "checks": {
                    "rhat": True,
                    "ess": True,
                    "divergences": True,
                    "bfmi": True,
                    "diagnostics_assessable": True,
                },
                "divergences": 0,
                "max_rhat": 1.0,
                "min_ess": 4000.0,
                "bfmi_per_chain": [0.9, 0.9],
            }
        ),
        encoding="utf-8",
    )
    table = report["load_summary"]("clean", "rope_summary")
    assert table is not None and list(table["quantity"]) == ["tau"]
    assert report["show_gated"]("clean", table, "the card") is table
    assert report["unreadable_artifacts"] == {}


def test_an_unparsable_gate_is_recorded_and_withholds_a_readable_table(
    report, tmp_path
):
    """The corrupt file is named; the readable CSV beside it is not published."""
    directory = _fit_dir(tmp_path, "corrupt-gate")
    (directory / "rope_summary.csv").write_text(
        "quantity,median\ntau,0.42\n", encoding="utf-8"
    )
    (directory / "diagnostics_summary.json").write_text(
        '{"passed": true, "checks"', encoding="utf-8"
    )

    table = report["load_summary"]("corrupt-gate", "rope_summary")
    assert report["load_diagnostics"]("corrupt-gate") is None
    assert report["unreadable_artifacts"] == {
        ("corrupt-gate", "diagnostics_summary.json"): "parse_error"
    }
    rendered = _text(report["show_gated"]("corrupt-gate", table, "the card"))
    assert "Unreadable artefact" in rendered
    assert "diagnostics_summary.json" in rendered
    assert "0.42" not in rendered
    assert "diagnostics_summary.json" in report["unreadable_artifacts_markdown"]()


def test_a_nonfinite_json_token_is_a_failure_not_a_value(report, tmp_path):
    """``NaN`` is not JSON; the strict reader refuses it rather than inventing one."""
    directory = _fit_dir(tmp_path, "nonfinite")
    (directory / "diagnostics_summary.json").write_text(
        '{"passed": true, "max_rhat": NaN}', encoding="utf-8"
    )
    assert report["load_diagnostics"]("nonfinite") is None
    assert report["unreadable_artifacts"] == {
        ("nonfinite", "diagnostics_summary.json"): "parse_error"
    }


def test_an_unparsable_result_table_is_recorded_and_withheld(report, tmp_path):
    directory = _fit_dir(tmp_path, "corrupt-table")
    (directory / "rope_summary.csv").write_text(
        'quantity,median\n"tau,0.42\n9,9,9,9\n', encoding="utf-8"
    )

    assert report["load_summary"]("corrupt-table", "rope_summary") is None
    assert report["unreadable_artifacts"] == {
        ("corrupt-table", "rope_summary.csv"): "parse_error"
    }
    rendered = _text(report["show_gated"]("corrupt-table", None, "the card"))
    assert "Unreadable artefact" in rendered
    assert "rope_summary.csv" in rendered


def test_a_header_only_table_is_present_and_empty(report, tmp_path):
    """A fit that wrote a header and no rows is readable, not a read failure."""
    directory = _fit_dir(tmp_path, "empty-table")
    (directory / "rope_summary.csv").write_text("quantity,median\n", encoding="utf-8")

    table = report["load_summary"]("empty-table", "rope_summary")
    assert table is not None and table.empty
    assert report["unreadable_artifacts"] == {}
