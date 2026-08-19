# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regenerate ``standardised_couplings.csv`` over stored ``lcsm`` fit directories.

The SD-standardised level -> change couplings (and the contrasts between sources
of the same target) are a pure function of the stored posterior — the couplings
and the latent levels ``x_latent`` — so they can be backfilled over fits made
before the table existed (2026-08-19) without sampling anything. The writer is
the pipeline's own (``pipelines.lcsm.write_standardised_couplings``), so a fit
made after this date and a regenerated older fit carry byte-identical tables for
the same posterior.

Targets:

    regenerate_lcsm_standardised_couplings.py all              # every lcsm fit dir
    regenerate_lcsm_standardised_couplings.py lrp-rli-lcsm-067 # one model (all its -<config> dirs)
    regenerate_lcsm_standardised_couplings.py lrp-rli-lcsm-067-reporting

Honours the output-root override (``DSE_LRP_OUTPUT_DIR`` or ``--output-dir``).
The fit-time ``artifact_manifest.json`` is not rewritten, so a manifest scan of a
regenerated directory will list the new table as untracked until the next refit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import xarray as xr
from rich.console import Console

from language_reading_predictors import paths as _paths
from language_reading_predictors.statistical_models.lcsm import resolve_lcsm_run_plan
from language_reading_predictors.statistical_models.pipelines.lcsm import (
    write_standardised_couplings,
)
from language_reading_predictors.statistical_models.registry import discover_models

_console = Console()


def _subdirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")
    )


def resolve_targets(target: str) -> list[Path]:
    """Stored ``lcsm`` fit dirs for the requested target."""
    root = _paths.stat_models_dir()
    candidates = (
        _subdirs(root)
        if target == "all"
        else [
            d
            for d in _subdirs(root)
            if d.name == target or d.name.startswith(f"{target}-")
        ]
    )
    out: list[Path] = []
    for d in candidates:
        cfg_path = d / "config.json"
        if not cfg_path.exists():
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        if cfg.get("kind") == "lcsm":
            out.append(d)
    return out


def regenerate(fit_dir: Path) -> int:
    """Write the table for one stored fit; return the number of rows written."""
    with open(fit_dir / "config.json") as f:
        cfg = json.load(f)
    model_id = str(cfg["model_id"])
    spec = discover_models()[model_id].SPEC
    plan = resolve_lcsm_run_plan(spec)
    post = xr.open_dataset(fit_dir / "trace.nc", group="posterior")
    try:
        ctx = SimpleNamespace(
            output_dir=str(fit_dir),
            reporting=SimpleNamespace(ci_prob=float(cfg.get("ci_prob", 0.89))),
        )
        df = write_standardised_couplings(ctx, post, plan.coupling_names())
    finally:
        post.close()
    return 0 if df is None else int(len(df))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target", help="'all', a model id, or a fit dir name (<id>-<config>)"
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
        raise SystemExit(f"No lcsm fit output directories matched {args.target!r}.")
    for d in targets:
        n = regenerate(d)
        _console.print(f"  {d.name}: {n} rows written")


if __name__ == "__main__":
    main()
