# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Backfill ``dependence_identification.csv`` over existing joint fit dirs.

The table (2026-08-22 ITT audit, finding 3) measures how far a fit's LKJ residual
dependence block was informed by the data, by comparing each block parameter's
posterior SD against the prior SD read from the fit's *own* persisted ``prior``
group. It is written at fit time by ``pipelines.joint``; this script computes it
from a stored ``trace.nc`` so the registered companions need no refit.

It then rewrites ``release_decision.json`` and ``key_findings.json`` through the
same generators the pipeline uses, because the prior-dominated qualifier
(``release._dependence_identification_note``) reads the new table.

Targets follow ``regenerate_key_findings.py``:

    regenerate_dependence_identification.py all
    regenerate_dependence_identification.py lrp-rli-itt-215
    regenerate_dependence_identification.py lrp-rli-itt-215-reporting

A fit without the block is skipped, so ``all`` is safe. Honours the output-root
override (``DSE_LRP_OUTPUT_DIR`` or ``--output-dir``).
"""

from __future__ import annotations

import argparse
import json
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console

from language_reading_predictors import paths as _paths
from language_reading_predictors.statistical_models.release import (
    evaluate_publication,
    write_release_decision,
)
from language_reading_predictors.statistical_models.reporting import (
    dependence_identification_summary,
    generate_key_findings,
)

_console = Console()
_FILENAME = "dependence_identification.csv"
_MANIFEST = "artifact_manifest.json"


def _subdirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))


def resolve_targets(target: str) -> list[Path]:
    root = _paths.stat_models_dir()
    if target == "all":
        return _subdirs(root)
    return [d for d in _subdirs(root) if d.name == target or d.name.startswith(f"{target}-")]


def _record_in_manifest(directory: Path) -> None:
    """Add the backfilled table to the manifest without rewriting the rest.

    ``artifacts.write_manifest`` reconciles a live ``ArtifactLog`` against a
    directory scan; called here it would find no log and reclassify every
    recorded artefact as ``untracked``, destroying the provenance the manifest
    exists to hold. So the entry is appended in place and marked ``untracked``,
    which is exactly what a file written outside the log is.
    """
    path = directory / _MANIFEST
    if not path.is_file():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return
    entries = payload.get("artifacts")
    if not isinstance(entries, list):
        return
    if any(str(e.get("filename")) == _FILENAME for e in entries):
        return
    entries.append(
        {
            "filename": _FILENAME,
            "name": None,
            "kind": "table",
            "required": None,
            "status": "untracked",
            "n_rows": None,
            "columns": None,
            "error_type": None,
            "error": None,
        }
    )
    payload["artifacts"] = sorted(entries, key=lambda e: str(e.get("filename")))
    payload["n_untracked"] = int(payload.get("n_untracked") or 0) + 1
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    import arviz as az

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", help="'all', a model id, or a fit dir name")
    parser.add_argument("--ci-prob", type=float, default=None,
                        help="Credible-interval mass (default: the fit's own config.json)")
    parser.add_argument("--output-dir", default=None, help="Output root override")
    args = parser.parse_args()
    if args.output_dir:
        _paths.set_output_root(args.output_dir)
    _console.print(f"Output root: {_paths.describe_output_root()}")

    targets = resolve_targets(args.target)
    if not targets:
        raise SystemExit(f"No fit output directories matched {args.target!r}.")

    for directory in targets:
        trace_path = directory / "trace.nc"
        if not trace_path.is_file():
            continue
        ci_prob = args.ci_prob
        if ci_prob is None:
            with suppress(Exception):
                ci_prob = float(
                    json.loads((directory / "config.json").read_text(encoding="utf-8"))["ci_prob"]
                )
        if ci_prob is None:
            ci_prob = 0.89
        trace = None
        try:
            trace = az.from_netcdf(trace_path)
            frame = dependence_identification_summary(trace, ci_prob=ci_prob)
        except Exception as exc:  # noqa: BLE001 - report and continue
            _console.print(f"  [yellow]{directory.name}: {type(exc).__name__}: {exc}[/yellow]")
            continue
        finally:
            with suppress(Exception):
                if trace is not None:
                    trace.close()
        if frame is None or frame.empty:
            continue

        frame.to_csv(directory / _FILENAME, index=False)
        _record_in_manifest(directory)
        decision = evaluate_publication(directory)
        write_release_decision(SimpleNamespace(output_dir=str(directory)), decision)
        payload = generate_key_findings(directory, decision=decision)
        verdicts = ", ".join(
            f"{row.parameter}={row.verdict}"
            for row in frame.loc[frame["role"] == "residual correlation"].itertuples()
        )
        _console.print(
            f"  {directory.name}: {len(frame)} rows [{verdicts}] "
            f"-> release {decision.status}, findings {payload['status']}"
        )


if __name__ == "__main__":
    main()
