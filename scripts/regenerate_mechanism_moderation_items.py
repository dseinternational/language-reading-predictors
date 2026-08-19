# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regenerate ``moderation_items.csv`` over stored moderated ``mechanism`` fit dirs.

The words-scale moderation contrast (the interquartile-cell exposure increments
at the low and high moderator cell, their difference in outcome items, the
logit-additive benchmark and the logit-scale interaction over the same cells) is
a pure function of the stored posterior, its ``constant_data`` and the fitted
rows, so it can be backfilled over fits made before the table existed
(2026-08-19) without sampling anything. The writer is the pipeline's own
(``pipelines.mechanism.write_moderation_items``), so a fit made after this date
and a regenerated older fit carry byte-identical tables for the same posterior.

The fitted rows are recovered by rebuilding the model (no sampling) through the
family's own plan — the same construction a leave-one-out refit uses — and the
writer refuses to proceed unless the re-loaded exposure and moderator values
reproduce the stored standardised vectors exactly, so a data or row change since
the fit cannot silently produce a table.

Targets:

    regenerate_mechanism_moderation_items.py all                # every moderated mechanism fit dir
    regenerate_mechanism_moderation_items.py lrp-rli-mech-061   # one model (all its -<config> dirs)
    regenerate_mechanism_moderation_items.py lrp-rli-mech-061-reporting

Honours the output-root override (``DSE_LRP_OUTPUT_DIR`` or ``--output-dir``).
The fit-time ``artifact_manifest.json`` is not rewritten, so a manifest scan of a
regenerated directory will list the new table as untracked until the next refit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import arviz as az
from rich.console import Console

from language_reading_predictors import paths as _paths
from language_reading_predictors.statistical_models.mechanism import (
    build_mechanism_for_plan,
    resolve_mechanism_plan,
    resolve_mechanism_run_plan,
)
from language_reading_predictors.statistical_models.pipelines.mechanism import (
    write_moderation_items,
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
    """Stored moderated ``mechanism`` fit dirs for the requested target."""
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
        if not cfg_path.exists() or not (d / "trace.nc").exists():
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        if cfg.get("kind") != "mechanism":
            continue
        plan = cfg.get("resolved_run_plan") or {}
        if plan.get("moderator_symbol") and plan.get("include_interaction", True):
            out.append(d)
    return out


def regenerate(fit_dir: Path) -> int:
    """Write the table for one stored fit; return the number of rows written."""
    with open(fit_dir / "config.json") as f:
        cfg = json.load(f)
    model_id = str(cfg["model_id"])
    spec = discover_models()[model_id].SPEC
    run_plan = resolve_mechanism_run_plan(spec)
    plan = resolve_mechanism_plan(spec, run_plan=run_plan)
    # The factory subsets the analysis frame to the fitted rows; rebuilding (no
    # sampling) is the one construction that yields exactly those rows.
    built = build_mechanism_for_plan(plan)
    trace = az.from_netcdf(fit_dir / "trace.nc")
    n_fitted = int(trace.posterior.sizes["obs_id"])
    if int(built.prepared.n_obs) != n_fitted:
        raise SystemExit(
            f"{fit_dir.name}: rebuilt frame has {built.prepared.n_obs} rows but the "
            f"stored posterior has {n_fitted}; refusing to write."
        )
    ctx = SimpleNamespace(
        spec=spec,
        resolved_plan=run_plan,
        prepared=built.prepared,
        trace=trace,
        output_dir=str(fit_dir),
        reporting=SimpleNamespace(ci_prob=float(cfg.get("ci_prob", 0.89))),
    )
    df = write_moderation_items(ctx)
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
        raise SystemExit(
            f"No moderated mechanism fit output directories matched {args.target!r}."
        )
    for d in targets:
        n = regenerate(d)
        _console.print(f"  {d.name}: {n} rows written")


if __name__ == "__main__":
    main()
