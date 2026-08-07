# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Single artefact interface for statistical-model fits (#394 steps 2-3).

Historically every family pipeline wrote its tables with an inline
``df.to_csv(os.path.join(ctx.output_dir, ...))`` followed by a manual
``ctx.tables[...]`` registration, and guarded optional figures with ad-hoc
``except Exception`` blocks that printed a warning and dropped the failure on
the floor. That idiom appeared at over a hundred call sites in the monolithic
pipeline, so nothing recorded which artefacts a fit produced, which optional
ones were skipped, or why.

This module centralises the mechanics without changing behaviour:

- :func:`save_table` is the one operation that writes a table CSV into the
  fit's output directory, registers it on the run context, optionally
  validates required columns, and records the artefact.
- :func:`guard_optional` reproduces the warn-and-continue guard around
  optional artefacts (figures, coverage tables) while persisting a structured
  record of what was skipped and why.
- :func:`write_manifest` assembles ``artifact_manifest.json`` at report
  finalisation: every recorded artefact plus a reconciliation scan of the
  output directory, so files written by helpers that have not yet migrated to
  this interface (shared ``dse_research_utils`` writers, plot helpers, the
  report template copy) still appear, marked ``untracked``.

The interface duck-types the context (``output_dir`` / ``tables`` /
``artifacts`` attributes) so lightweight sweep and test harnesses that build a
minimal context object keep working; registration and recording are simply
skipped when the corresponding attribute is absent.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import pandas as pd
from rich import print as rprint


# Manifest kind inferred from the file extension during the reconciliation
# scan. Anything unrecognised is reported as "other" rather than dropped.
_KIND_BY_EXTENSION = {
    ".csv": "table",
    ".json": "json",
    ".md": "text",
    ".nc": "netcdf",
    ".pdf": "figure",
    ".png": "figure",
    ".qmd": "report",
    ".svg": "figure",
    ".txt": "text",
    ".yml": "report",
}

MANIFEST_FILENAME = "artifact_manifest.json"


@dataclass(slots=True)
class ArtifactRecord:
    """One artefact of a fit: a written table, or a skipped optional output."""

    name: str
    """Logical name (the ``ctx.tables`` registration key, or the skip label)."""

    filename: str
    """Path relative to the fit's output directory."""

    kind: str
    """``table`` | ``figure`` | ``json`` | ``netcdf`` | ``report`` | ``text`` | ``other``."""

    required: bool
    """Whether the fit treats this artefact as required (a failure raises) or
    optional (a failure warns, is recorded, and the fit continues)."""

    status: str
    """``written`` | ``skipped`` (``untracked`` is added by the manifest scan)."""

    n_rows: int | None = None
    columns: tuple[str, ...] | None = None
    error_type: str | None = None
    error: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "name": self.name,
            "kind": self.kind,
            "required": self.required,
            "status": self.status,
            "n_rows": self.n_rows,
            "columns": list(self.columns) if self.columns is not None else None,
            "error_type": self.error_type,
            "error": self.error,
        }


@dataclass(slots=True)
class ArtifactLog:
    """Per-fit record of artefacts, keyed by filename (last write wins).

    Mutually-exclusive branches of a family pipeline may target the same
    filename (``rope_summary.csv`` on the graded versus off-floor routes), and
    a retried optional artefact may succeed after a recorded skip; keying by
    filename keeps the log consistent with what is actually on disk.
    """

    records: dict[str, ArtifactRecord] = field(default_factory=dict)

    def record(self, record: ArtifactRecord) -> None:
        self.records[record.filename] = record

    @property
    def written(self) -> list[ArtifactRecord]:
        return [r for r in self.records.values() if r.status == "written"]

    @property
    def skipped(self) -> list[ArtifactRecord]:
        return [r for r in self.records.values() if r.status == "skipped"]


def _log_of(ctx: Any) -> ArtifactLog | None:
    log = getattr(ctx, "artifacts", None)
    return log if isinstance(log, ArtifactLog) else None


def save_table(
    ctx: Any,
    name: str,
    df: pd.DataFrame,
    *,
    filename: str | None = None,
    required_columns: Sequence[str] | None = None,
    index: bool = False,
    register: bool = True,
    required: bool = True,
) -> pd.DataFrame:
    """Write ``df`` into the fit's output directory, register and record it.

    One operation replaces the historical three-line idiom (``to_csv`` +
    ``ctx.tables[...] =`` + nothing recorded). ``filename`` defaults to
    ``{name}.csv``; ``index=True`` preserves the matrix CSVs that publish
    their row labels. ``register=False`` preserves the few artefacts that were
    deliberately never registered on the context (per-measure loop outputs,
    row manifests). ``required_columns`` fails loudly before anything is
    written, so a schema drift cannot publish a partial table.

    Returns ``df`` unchanged so call sites can keep chaining.
    """
    resolved = filename if filename is not None else f"{name}.csv"
    if required_columns:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"artefact {resolved!r} is missing required column(s) "
                f"{missing}; present: {list(df.columns)}"
            )
    df.to_csv(os.path.join(ctx.output_dir, resolved), index=index)
    if register:
        tables = getattr(ctx, "tables", None)
        if tables is not None:
            tables[name] = df
    log = _log_of(ctx)
    if log is not None:
        log.record(
            ArtifactRecord(
                name=name,
                filename=resolved,
                kind="table",
                required=required,
                status="written",
                n_rows=int(len(df)),
                columns=tuple(str(c) for c in df.columns),
            )
        )
    return df


def record_artifact(
    ctx: Any,
    name: str,
    *,
    filename: str | None = None,
    kind: str = "table",
    required: bool = True,
    df: pd.DataFrame | None = None,
) -> None:
    """Record an artefact written by a writer this interface does not own.

    Some writers deliberately keep their own write mechanics — the atomic
    temp-file-and-rename writers shared with the post-hoc regeneration scripts
    (``psense_summary.csv``), and helpers that take an output directory rather
    than a fit context (``predicted_scores.csv``, ``mechanism_curve_items.csv``).
    This records the artefact on the fit's log without writing anything, so the
    manifest reports it as ``written`` (with shape when ``df`` is supplied)
    instead of ``untracked``.
    """
    log = _log_of(ctx)
    if log is None:
        return
    log.record(
        ArtifactRecord(
            name=name,
            filename=filename if filename is not None else f"{name}.csv",
            kind=kind,
            required=required,
            status="written",
            n_rows=int(len(df)) if df is not None else None,
            columns=tuple(str(c) for c in df.columns) if df is not None else None,
        )
    )


@contextmanager
def guard_optional(
    ctx: Any,
    label: str,
    *,
    filename: str | None = None,
    kind: str = "figure",
    verb: str = "skipped",
) -> Iterator[None]:
    """Warn-and-continue guard for optional artefacts, recording any skip.

    Behaviour-preserving replacement for the pipeline's ad-hoc ``except
    Exception`` blocks: an expensive fit must never be lost to a plotting or
    summary hiccup, so the failure prints the same ``[yellow]{label} {verb}``
    warning and the fit continues — but the failure type and message are now
    persisted to the artefact manifest instead of scrolling away. ``verb``
    reproduces each historical guard's own wording ("skipped", "failed",
    "not written") so converted sites stay message-identical. Only
    ``Exception`` is caught (``KeyboardInterrupt``/``SystemExit`` propagate),
    matching the guards this replaces.
    """
    try:
        yield
    except Exception as exc:  # noqa: BLE001 - an optional artefact must not fail a fit
        rprint(f"[yellow]{label} {verb}: {exc}[/yellow]")
        log = _log_of(ctx)
        if log is not None:
            log.record(
                ArtifactRecord(
                    name=label,
                    filename=filename if filename is not None else label,
                    kind=kind,
                    required=False,
                    status="skipped",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            )


def _scan_output_dir(output_dir: str) -> list[str]:
    """Relative paths of every file under ``output_dir`` (sorted, stable)."""
    found: list[str] = []
    for root, _dirs, files in os.walk(output_dir):
        for fname in files:
            if fname == ".DS_Store":
                continue
            rel = os.path.relpath(os.path.join(root, fname), output_dir)
            found.append(rel)
    return sorted(found)


def write_manifest(ctx: Any) -> dict[str, Any]:
    """Write ``artifact_manifest.json`` reconciling the log with the directory.

    Recorded artefacts carry their full provenance (status, shape, any skip
    reason). Files present on disk but never routed through this interface —
    figures from plot helpers, shared-writer diagnostics, the copied report
    template — are listed as ``untracked`` with a kind inferred from their
    extension, so the manifest is a complete inventory of the fit directory
    from the first adoption and the recorded/untracked split measures how much
    of the pipeline has migrated. A recorded skip whose file nevertheless
    exists (a later retry succeeded outside the guard) is reported as it was
    recorded; disk presence is authoritative only for ``untracked`` entries.
    """
    log = _log_of(ctx)
    records = dict(log.records) if log is not None else {}
    on_disk = _scan_output_dir(ctx.output_dir)
    # Stems that have a figure file: an untracked CSV sharing a stem with a
    # .png/.svg is that figure's data sidecar (``save_styled_figure(data=...)``),
    # not a not-yet-migrated table, and is classified accordingly.
    figure_stems = {
        os.path.splitext(rel)[0]
        for rel in on_disk
        if os.path.splitext(rel)[1].lower() in {".png", ".svg"}
    }
    entries: list[dict[str, Any]] = []
    for rel in on_disk:
        if rel == MANIFEST_FILENAME:
            continue
        if rel in records:
            entries.append(records.pop(rel).to_json_dict())
        else:
            stem, ext = os.path.splitext(rel)
            ext = ext.lower()
            kind = _KIND_BY_EXTENSION.get(ext, "other")
            if ext == ".csv" and stem in figure_stems:
                kind = "figure_data"
            entries.append(
                {
                    "filename": rel,
                    "name": None,
                    "kind": kind,
                    "required": None,
                    "status": "untracked",
                    "n_rows": None,
                    "columns": None,
                    "error_type": None,
                    "error": None,
                }
            )
    # Remaining records have no file on disk: recorded skips, plus any recorded
    # write whose file has since vanished (surfaced rather than silently lost).
    for rec in records.values():
        entry = rec.to_json_dict()
        if rec.status == "written":
            entry["status"] = "missing"
        entries.append(entry)
    entries.sort(key=lambda e: e["filename"])
    counts = {
        "n_written": sum(1 for e in entries if e["status"] == "written"),
        "n_skipped": sum(1 for e in entries if e["status"] == "skipped"),
        "n_untracked": sum(1 for e in entries if e["status"] == "untracked"),
        "n_missing": sum(1 for e in entries if e["status"] == "missing"),
    }
    manifest = {
        "model_id": getattr(getattr(ctx, "spec", None), "model_id", None),
        "artifacts": entries,
        **counts,
    }
    path = os.path.join(ctx.output_dir, MANIFEST_FILENAME)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    return manifest
