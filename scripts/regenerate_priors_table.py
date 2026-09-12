#!/usr/bin/env python
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Refresh explanations and missing density panels for a stored fit.

Only checked family rebuilds are supported. The recorded run plan, variable names
and prior distributions must match before the script writes anything. Existing
panels still used by the corrected table are retained. The trace is never sampled.
Use --dry-run to inspect the changes.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd
from rich.console import Console

from language_reading_predictors import paths as _paths
from language_reading_predictors.statistical_models import priors as _priors
from language_reading_predictors.statistical_models.registry import discover_models

_console = Console()

TABLE = "priors_table.csv"


def _subdirs(root: Path) -> list[Path]:
    """Published fit directories, excluding in-flight output transactions."""
    if not root.is_dir():
        return []
    return sorted(
        d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")
    )


def resolve_targets(target: str) -> list[Path]:
    root = _paths.stat_models_dir()
    return [
        d for d in _subdirs(root) if d.name == target or d.name.startswith(f"{target}-")
    ]


def _normalise(value: object) -> str:
    """Compare a CSV round trip to a fresh frame: an absent panel reads as NaN."""
    text = "" if value is None else str(value)
    return "" if text in {"nan", "None"} else text


def _spec_for(model_id: str):
    lazy = discover_models().get(model_id)
    if lazy is None:
        raise LookupError(f"{model_id} is not a discoverable model module")
    spec = getattr(lazy.load(), "SPEC", None)
    if spec is None:
        raise LookupError(f"{model_id}'s module declares no SPEC")
    return spec


def _build_joint_mechanism_levels(spec, config: dict):
    """Rebuild the artefact-hosting wave of a joint-mechanism levels fit.

    The levels design fits one model per wave and one wave hosts the fit-level
    files. Which wave that was is recorded in the fit's own ``config.json``, so the
    rebuild targets the same rows the stored table describes rather than guessing.
    """
    from language_reading_predictors.statistical_models import joint_mechanism as _jm
    from language_reading_predictors.statistical_models.factories.joint_mechanism import build_joint_mechanism_model
    from language_reading_predictors.statistical_models.preprocessing import (
        _subset_prepared,
        load_and_prepare,
    )

    plan = _jm.resolve_joint_mechanism_run_plan(spec)
    if plan.design != "levels":
        return None, f"{spec.model_id} is the {plan.design!r} design, not levels"
    timepoint = (config.get("extra") or {}).get("artifact_hosting_timepoint")
    if timepoint is None:
        return None, "config.json records no artifact_hosting_timepoint"

    recorded = config.get("resolved_run_plan")
    if not recorded:
        return None, "no recorded run plan to validate the rebuild"
    prepared_all = load_and_prepare(**plan.prepare_kwargs())
    active = tuple(c for c in plan.declared_adjustment if c in prepared_all.covariates)
    if active != plan.active_adjustment:
        plan = plan.with_active_adjustment(active)
    if json.loads(json.dumps(plan.as_dict())) != recorded:
        return None, "the current plan differs from the stored fit; refit before replacing its priors"
    sub = _subset_prepared(prepared_all, prepared_all.phase == int(timepoint) - 1)
    return build_joint_mechanism_model(sub, **plan.factory_kwargs()), None


#: ``ModelSpec.kind`` -> a function rebuilding that family's model without
#: sampling. Deliberately per family and deliberately incomplete: a family is
#: added when its rebuild has been checked against a stored fit, not before.
BUILDERS = {"joint_mechanism": _build_joint_mechanism_levels}


def regenerate(fit_dir: Path, *, dry_run: bool) -> tuple[str, str]:
    """Return ``(status, detail)`` for one fit directory."""
    stored_path = fit_dir / TABLE
    if not stored_path.exists():
        return "skipped", f"no {TABLE}"
    config_path = fit_dir / "config.json"
    if not config_path.exists():
        return "skipped", "no config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    kind = str(config.get("kind") or "")
    builder = BUILDERS.get(kind)
    if builder is None:
        return "skipped", f"no checked rebuild for kind {kind!r}"

    spec = _spec_for(str(config["model_id"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built, reason = builder(spec, config)
    if built is None:
        return "skipped", reason or "rebuild unavailable"

    stored = pd.read_csv(stored_path)
    rebuilt_names = [rv.name for rv in built.model.free_RVs]
    # The rebuild must describe the *stored* model. A changed variable set means
    # the module has moved since the fit, and the table would then document a model
    # nobody ran.
    if list(stored["parameter"]) != rebuilt_names:
        return (
            "needs refit",
            "the rebuilt model's variables differ from the stored table "
            f"({sorted(set(rebuilt_names) ^ set(stored['parameter']))}); "
            "the module has changed since this fit",
        )

    table = _priors.priors_table(built.model)
    distributions_match = all(
        str(old).replace(" ", "") == str(new).replace(" ", "")
        for old, new in zip(stored["distribution"], table["distribution"], strict=True)
    )
    if not distributions_match:
        return "needs refit", "the rebuilt prior distributions differ from the stored table"
    changed = [
        f"{row.parameter}: {column} {getattr(old, column)!r} -> {getattr(row, column)!r}"
        for old, row in zip(stored.itertuples(), table.itertuples(), strict=True)
        for column in ("distribution", "role", "rationale", "panel")
        if _normalise(getattr(old, column)) != _normalise(getattr(row, column))
    ]
    panels = _priors.model_prior_panels(built.model)
    missing = [
        f"prior_{key}.{ext}" for key in panels for ext in ("png", "svg")
        if not (fit_dir / f"prior_{key}.{ext}").exists()
    ]
    if missing:
        changed.append(f"missing density panels: {', '.join(missing)}")
    if not changed:
        return "unchanged", "the stored table already matches the rebuilt model"
    if dry_run:
        return "would rewrite", "; ".join(changed)

    existing = {path.name for path in fit_dir.glob("prior_*.*")}
    for key, density in panels.items():
        if any(not (fit_dir / f"prior_{key}.{ext}").exists() for ext in ("png", "svg")):
            _priors.plot_and_save(density, str(fit_dir), f"prior_{key}", title=_priors.model_prior_panel_title(built.model, key))
    added = {path.name for path in fit_dir.glob("prior_*.*")} - existing
    table.to_csv(stored_path, index=False)
    orphaned = _drop_orphaned_panels(fit_dir, table)
    _prune_manifest(fit_dir, orphaned, added=added)
    detail = "; ".join(changed)
    if orphaned:
        detail += f" [removed {', '.join(sorted(orphaned))}]"
    return "rewritten", detail


def _drop_orphaned_panels(fit_dir: Path, table: pd.DataFrame) -> set[str]:
    """Remove unused density panels, retaining overlays and predictive checks."""
    wanted = {str(panel) for panel in table["panel"] if str(panel) not in {"", "nan"}}
    removed: set[str] = set()
    for panel in _priors.prior_density_panel_files(fit_dir):
        key = panel.stem.removeprefix("prior_")
        if key in wanted:
            continue
        panel.unlink()
        removed.add(panel.name)
    return removed


def _prune_manifest(fit_dir: Path, removed: set[str], *, added: set[str] | None = None) -> None:
    """Update manifest rows for changed panels only.

    A full rescan would also absorb whatever has appeared in the directory since
    the fit — a rendered ``index.html`` and its Quarto asset tree — turning a
    fit-time inventory into a directory listing.
    """
    added = added or set()
    if not removed and not added:
        return
    path = fit_dir / "artifact_manifest.json"
    if not path.exists():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    kept = [
        entry for entry in manifest.get("artifacts", [])
        if entry.get("filename") not in removed
    ]
    dropped = len(manifest.get("artifacts", [])) - len(kept)
    known = {entry.get("filename") for entry in kept}
    new_entries = [{"filename": name, "status": "untracked"} for name in sorted(added - known)]
    manifest["artifacts"] = kept + new_entries
    manifest["n_untracked"] = int(manifest.get("n_untracked", 0)) - dropped + len(new_entries)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", help="model id, or a specific <id>-<config> fit dir")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would change without writing anything",
    )
    args = parser.parse_args(argv)

    targets = resolve_targets(args.target)
    if not targets:
        _console.print(f"[red]No fit directories matched {args.target!r}[/red]")
        return 1
    failures = 0
    for fit_dir in targets:
        status, detail = regenerate(fit_dir, dry_run=args.dry_run)
        colour = {
            "rewritten": "green",
            "would rewrite": "cyan",
            "unchanged": "dim",
            "skipped": "yellow",
            "needs refit": "red",
        }.get(status, "white")
        failures += status == "needs refit"
        _console.print(f"[{colour}]{fit_dir.name}: {status}[/{colour}] — {detail}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
