# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the single artefact interface and manifest (#394 steps 2-3)."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pandas as pd
import pytest

from language_reading_predictors.statistical_models.artifacts import (
    ArtifactLog,
    guard_optional,
    save_table,
    write_manifest,
)


def _ctx(tmp_path):
    return SimpleNamespace(
        output_dir=str(tmp_path),
        tables={},
        artifacts=ArtifactLog(),
        spec=SimpleNamespace(model_id="lrp-rli-test-001"),
    )


def _frame():
    return pd.DataFrame({"term": ["tau", "beta"], "median": [0.25, -0.1]})


def test_save_table_writes_the_same_bytes_as_the_legacy_idiom(tmp_path):
    ctx = _ctx(tmp_path / "new")
    os.makedirs(ctx.output_dir)
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    df = _frame()

    save_table(ctx, "tau_summary", df)
    df.to_csv(os.path.join(legacy_dir, "tau_summary.csv"), index=False)

    new = (tmp_path / "new" / "tau_summary.csv").read_bytes()
    old = (legacy_dir / "tau_summary.csv").read_bytes()
    assert new == old


def test_save_table_index_true_matches_the_legacy_matrix_write(tmp_path):
    ctx = _ctx(tmp_path / "new")
    os.makedirs(ctx.output_dir)
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    matrix = pd.DataFrame([[1.0, 0.4], [0.4, 1.0]], index=["W", "L"], columns=["W", "L"])

    save_table(ctx, "factor_correlation", matrix, index=True)
    matrix.to_csv(os.path.join(legacy_dir, "factor_correlation.csv"))

    new = (tmp_path / "new" / "factor_correlation.csv").read_bytes()
    old = (legacy_dir / "factor_correlation.csv").read_bytes()
    assert new == old


def test_save_table_registers_and_records(tmp_path):
    ctx = _ctx(tmp_path)
    df = save_table(ctx, "tau_summary", _frame())

    assert ctx.tables["tau_summary"] is df
    record = ctx.artifacts.records["tau_summary.csv"]
    assert record.name == "tau_summary"
    assert record.kind == "table"
    assert record.status == "written"
    assert record.required is True
    assert record.n_rows == 2
    assert record.columns == ("term", "median")


def test_save_table_register_false_still_writes_and_records(tmp_path):
    ctx = _ctx(tmp_path)
    save_table(ctx, "analysis_rows", _frame(), register=False)

    assert "analysis_rows" not in ctx.tables
    assert os.path.exists(tmp_path / "analysis_rows.csv")
    assert ctx.artifacts.records["analysis_rows.csv"].status == "written"


def test_save_table_required_columns_fail_loud_before_writing(tmp_path):
    ctx = _ctx(tmp_path)
    with pytest.raises(ValueError, match="missing required column"):
        save_table(
            ctx, "tau_summary", _frame(), required_columns=["term", "ci_low"]
        )
    assert not os.path.exists(tmp_path / "tau_summary.csv")
    assert "tau_summary" not in ctx.tables


def test_save_table_tolerates_a_minimal_duck_typed_context(tmp_path):
    minimal = SimpleNamespace(output_dir=str(tmp_path))
    save_table(minimal, "prior_pushforward", _frame())
    assert os.path.exists(tmp_path / "prior_pushforward.csv")


def test_guard_optional_swallows_warns_and_records(tmp_path, capsys):
    ctx = _ctx(tmp_path)
    with guard_optional(ctx, "ppc_summary.csv", filename="ppc_summary.csv", kind="table"):
        raise RuntimeError("no posterior predictive group")

    out = capsys.readouterr().out
    assert "ppc_summary.csv skipped: no posterior predictive group" in out
    record = ctx.artifacts.records["ppc_summary.csv"]
    assert record.status == "skipped"
    assert record.required is False
    assert record.error_type == "RuntimeError"
    assert record.error == "no posterior predictive group"


def test_guard_optional_does_not_swallow_keyboard_interrupt(tmp_path):
    ctx = _ctx(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        with guard_optional(ctx, "ppc_summary.csv"):
            raise KeyboardInterrupt


def test_a_success_after_a_recorded_skip_wins(tmp_path):
    ctx = _ctx(tmp_path)
    with guard_optional(ctx, "rope_summary.csv", filename="rope_summary.csv", kind="table"):
        raise RuntimeError("first branch unavailable")
    save_table(ctx, "rope_summary", _frame())

    assert ctx.artifacts.records["rope_summary.csv"].status == "written"


def test_write_manifest_reconciles_recorded_and_untracked(tmp_path):
    ctx = _ctx(tmp_path)
    save_table(ctx, "tau_summary", _frame())
    with guard_optional(ctx, "ppc_summary.csv", filename="ppc_summary.csv", kind="table"):
        raise RuntimeError("skipped on purpose")
    (tmp_path / "rank_plot.png").write_bytes(b"png")
    nested = tmp_path / "_partials"
    nested.mkdir()
    (nested / "_header.qmd").write_text("{{}}", encoding="utf-8")

    manifest = write_manifest(ctx)

    assert manifest["model_id"] == "lrp-rli-test-001"
    by_name = {e["filename"]: e for e in manifest["artifacts"]}
    assert by_name["tau_summary.csv"]["status"] == "written"
    assert by_name["ppc_summary.csv"]["status"] == "skipped"
    assert by_name["ppc_summary.csv"]["error_type"] == "RuntimeError"
    assert by_name["rank_plot.png"]["status"] == "untracked"
    assert by_name["rank_plot.png"]["kind"] == "figure"
    assert by_name[os.path.join("_partials", "_header.qmd")]["kind"] == "report"
    assert manifest["n_written"] == 1
    assert manifest["n_skipped"] == 1
    assert manifest["n_untracked"] == 2
    filenames = [e["filename"] for e in manifest["artifacts"]]
    assert filenames == sorted(filenames)
    # The manifest lists everything except itself, and is valid JSON on disk.
    assert "artifact_manifest.json" not in by_name
    on_disk = json.loads((tmp_path / "artifact_manifest.json").read_text("utf-8"))
    assert on_disk["n_written"] == 1


def test_write_manifest_marks_a_vanished_recorded_write_missing(tmp_path):
    ctx = _ctx(tmp_path)
    save_table(ctx, "tau_summary", _frame())
    os.remove(tmp_path / "tau_summary.csv")

    manifest = write_manifest(ctx)

    by_name = {e["filename"]: e for e in manifest["artifacts"]}
    assert by_name["tau_summary.csv"]["status"] == "missing"
    assert manifest["n_missing"] == 1


def test_pipeline_writes_tables_only_through_the_artifact_interface():
    """Characterisation guard (#394): no direct ``to_csv`` in the monolith.

    Every published table in ``pipeline.py`` goes through :func:`save_table`,
    so a regression back to the inline write-and-register idiom fails here
    rather than silently re-fragmenting the artefact record.
    """
    import language_reading_predictors.statistical_models.pipeline as pipeline_module

    source = open(pipeline_module.__file__, encoding="utf-8").read()
    assert ".to_csv(" not in source
