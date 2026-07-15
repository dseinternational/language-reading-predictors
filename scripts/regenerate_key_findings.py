# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regenerate ``key_findings.json`` over existing statistical-model output dirs.

The key-findings box (#320) is generated at fit time by the pipeline; this
script re-runs the same generator over already-fitted output directories, so
sentence-template fixes and backfill of pre-#320 fits need no refit.

Targets:

    regenerate_key_findings.py all                # every output/statistical_models/models/<id>-<cfg> dir
    regenerate_key_findings.py lrp-rli-itt-010    # a single model (all its -<config> dirs)
    regenerate_key_findings.py lrp-rli-itt-010-dev  # one specific fit dir

Honours the output-root override (``DSE_LRP_OUTPUT_DIR`` or ``--output-dir``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from language_reading_predictors import paths as _paths
from language_reading_predictors.statistical_models.reporting import (
    generate_key_findings,
)

_console = Console()


def _subdirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(d for d in root.iterdir() if d.is_dir())


def resolve_targets(target: str) -> list[Path]:
    """Fit output dirs for the requested target (statistical models only)."""
    root = _paths.stat_models_dir()
    if target == "all":
        return _subdirs(root)
    # Statistical dirs are named "<id>-<config>"; accept either form.
    return [
        d
        for d in _subdirs(root)
        if d.name == target or d.name.startswith(f"{target}-")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", help="'all', a model id, or a fit dir name (<id>-<config>)")
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
    for d in targets:
        payload = generate_key_findings(d)
        detail = (
            f"{len(payload['sentences'])} sentences"
            if payload["status"] == "ok"
            else payload.get("reason") or ", ".join(payload.get("failing_checks", []))
        )
        _console.print(f"  {d.name}: {payload['status']} ({detail})")


if __name__ == "__main__":
    main()
