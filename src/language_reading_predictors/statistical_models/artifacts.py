# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Write, register and record artefacts from statistical-model fits.

``save_table`` validates and writes tables. ``guard_optional`` records optional
failures while the fit continues. ``write_manifest`` reconciles these records
with files on disk, including outputs from other writers as ``untracked``.

Contexts must supply ``output_dir``. Registration and recording are skipped
when the optional ``tables`` or ``artifacts`` attributes are absent.
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
    ".scss": "report",
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

    ``filename`` defaults to ``{name}.csv``. Use ``index=True`` to retain row
    labels and ``register=False`` to omit the ``ctx.tables`` entry. Missing
    ``required_columns`` raise before the table is written.

    Returns ``df`` unchanged so call sites can keep chaining.
    """
    resolved = filename if filename is not None else f"{name}.csv"
    if required_columns:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"artefact {resolved!r} is missing required column(s) {missing}; present: {list(df.columns)}"
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

    Catch ``Exception``, print ``{label} {verb}`` with the error, and record its
    type and message in the manifest. ``KeyboardInterrupt`` and ``SystemExit``
    propagate. Use only for outputs whose failure may leave the fit running.
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

    Recorded artefacts retain their status, shape and skip reason. Files from
    other writers appear as ``untracked``, with kind inferred from the extension.
    A recorded write with no file becomes ``missing``. A recorded skip retains
    that status even if a later writer created the file without updating the log.
    """
    log = _log_of(ctx)
    records = dict(log.records) if log is not None else {}
    on_disk = _scan_output_dir(ctx.output_dir)
    # Stems that have a figure file: an untracked CSV sharing a stem with a
    # .png/.svg is that figure's data sidecar (``save_styled_figure(data=...)``),
    # not a not-yet-migrated table, and is classified accordingly.
    figure_stems = {os.path.splitext(rel)[0] for rel in on_disk if os.path.splitext(rel)[1].lower() in {".png", ".svg"}}
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
