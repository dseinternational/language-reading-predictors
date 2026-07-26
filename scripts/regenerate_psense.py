#!/usr/bin/env python
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Backfill ``psense_summary.csv`` / ``psense.png`` over existing fit output dirs.

Power-scaling prior sensitivity (#381) is emitted at fit time by every family
pipeline, but that wiring landed in #408 / #416 — so every fit stored *before*
those merged carries no psense artefacts, and its report shows no flags because
none were **measured**, not because the estimand was measured clean. That
distinction matters most for exactly the deliverables #381 names as the most
prior-dependent: horseshoe rankings, HSGP mechanism curves, the EiV slope.

Power-scaling is importance reweighting over the draws already in hand, **not** a
refit, so the gap closes without resampling anything: a stored trace carrying the
``log_prior`` and ``log_likelihood`` groups can be measured after the fact and the
numbers belong to the same posterior the published report was written from.

The reported-parameter set comes from each fit's own ``diagnostics.csv`` rather
than from re-deriving it off the spec. That file *is* the fit's record of what it
reported, so the backfill covers exactly the parameters the report shows, and it
stays correct for a fit whose spec has since been edited.

Targets mirror ``regenerate_key_findings.py``:

    regenerate_psense.py all                        # every fit output dir
    regenerate_psense.py lrp-rli-hs-001             # one model, all its -<config> dirs
    regenerate_psense.py lrp-rli-hs-001-reporting   # one specific fit dir

Fits whose trace predates the ``log_prior`` wiring cannot be repaired here and are
reported as skips with that reason — they need a refit, which is a separate call.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd
from rich.console import Console

from language_reading_predictors import paths as _paths
from language_reading_predictors.statistical_models.diagnostics import (
    psense_artifacts,
)

_console = Console()


def _subdirs(root: Path) -> list[Path]:
    """Published fit directories, excluding in-flight output transactions.

    ``StatisticalFitContext.reset_output_dir`` stages each run in a *hidden* sibling
    (``.<id>-<config>.staging-XXXX``) and promotes it only on success, so a dotted
    directory is either a run in progress or an abandoned one. Backfilling into it
    writes artefacts that are about to be discarded — or, worse, races a live fit.
    """
    if not root.is_dir():
        return []
    return sorted(
        d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")
    )


def resolve_targets(target: str) -> list[Path]:
    """Fit output dirs for the requested target (statistical models only)."""
    root = _paths.stat_models_dir()
    if target == "all":
        return _subdirs(root)
    return [
        d for d in _subdirs(root) if d.name == target or d.name.startswith(f"{target}-")
    ]


def reported_var_names(fit_dir: Path) -> list[str]:
    """Base parameter names the fit reported, in the order it reported them.

    ``diagnostics.csv`` carries one row per *element* (``beta[L]``, ``beta[R]``,
    ...); ``psense_summary`` takes the base name and expands it itself. Order is
    preserved because it drives the psense figure's row layout.
    """
    path = fit_dir / "diagnostics.csv"
    if not path.exists():
        return []
    index = pd.read_csv(path, index_col=0).index
    seen: dict[str, None] = {}
    for label in index:
        seen.setdefault(str(label).split("[")[0].strip(), None)
    return list(seen)


def _trace_groups(idata) -> set[str]:
    return {g.rstrip("/").split("/")[-1] for g in idata.groups}


def backfill(fit_dir: Path, *, force: bool, dry_run: bool) -> tuple[str, str]:
    """Return ``(status, detail)`` for one fit directory."""
    summary = fit_dir / "psense_summary.csv"
    if summary.exists() and not force:
        return "present", "psense_summary.csv already written"
    if not (fit_dir / "trace.nc").exists():
        return "skipped", "no trace.nc"

    var_names = reported_var_names(fit_dir)
    if not var_names:
        return "skipped", "no diagnostics.csv, so no reported-parameter set to measure"

    import arviz as az

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        idata = az.from_netcdf(fit_dir / "trace.nc")
    groups = _trace_groups(idata)
    missing = {"log_prior", "log_likelihood"} - groups
    if missing:
        return (
            "needs refit",
            f"trace predates the {'/'.join(sorted(missing))} wiring; "
            "power-scaling cannot be reconstructed without resampling",
        )

    # Only measure what the posterior actually carries: a spec edit since the fit can
    # leave a name in diagnostics.csv that this trace never sampled, and psense would
    # fail the whole model over one absent term.
    present = set(az.extract(idata, group="posterior").data_vars)
    measurable = [n for n in var_names if n in present]
    dropped = [n for n in var_names if n not in present]
    if not measurable:
        return "skipped", f"none of {var_names} are in the posterior group"

    if dry_run:
        detail = f"would measure {len(measurable)} parameters"
        if dropped:
            detail += f" (not in posterior: {', '.join(dropped)})"
        return "would write", detail

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = psense_artifacts(idata, str(fit_dir), measurable)
    if frame is None:
        return "failed", "psense_summary could not be computed (see warning above)"

    detail = f"{len(frame)} rows"
    if "diagnosis" in frame.columns:
        flagged = int((frame["diagnosis"].astype(str).str.strip() != "✓").sum())
        detail += f", {flagged} flagged"
    if dropped:
        detail += f" (not in posterior: {', '.join(dropped)})"
    return "written", detail


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target", help="'all', a model id, or a fit dir name (<id>-<config>)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even where psense_summary.csv already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be written without touching any file",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output root override (takes precedence over DSE_LRP_OUTPUT_DIR)",
    )
    args = parser.parse_args()
    if args.output_dir:
        _paths.set_output_root(args.output_dir)
    _console.print(f"Output root: {_paths.describe_output_root()}")

    targets = resolve_targets(args.target)
    if not targets:
        raise SystemExit(f"No fit output directories matched {args.target!r}.")

    tally: dict[str, int] = {}
    for fit_dir in targets:
        status, detail = backfill(fit_dir, force=args.force, dry_run=args.dry_run)
        tally[status] = tally.get(status, 0) + 1
        colour = {
            "written": "green",
            "would write": "cyan",
            "present": "dim",
            "skipped": "yellow",
            "needs refit": "yellow",
            "failed": "red",
        }.get(status, "white")
        _console.print(f"[{colour}]{status:11}[/{colour}] {fit_dir.name}: {detail}")

    _console.print()
    _console.print(", ".join(f"{k}: {v}" for k, v in sorted(tally.items())))
    # A fit that needs a refit is a real coverage gap, not a no-op: surface it in the
    # exit status so a sweep cannot look clean while leaving estimands unmeasured.
    if tally.get("failed") or tally.get("needs refit"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
