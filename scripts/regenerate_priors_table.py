#!/usr/bin/env python
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Rewrite ``priors_table.csv`` and the prior-PDF panels over a stored fit.

The published prior table used to derive each parameter's **role**, **rationale**
and **panel** from its name: an exact-name map, then a prefix, a suffix, and in
several branches the rendered distribution string. #637 stage 2 replaced that for
every variable built through a named constructor, which records a
``PriorDescriptor`` when it is created — but a descriptor exists only in a model
that has been *built*, so a fit stored before that change keeps whatever the name
map said.

Where the two disagreed, the name map was wrong. ``lrp-rli-jm-001`` is the case
that prompted this script: its levels design deliberately carries ``beta_mech`` on
``predictor_slope_prior`` — ``Normal(0, 0.3)``, matched to ``ca-010`` / ``ca-011``
— while the name map keyed on the *name* ``beta_mech`` and published "Linear-
mechanism slope beta_mech ~ Normal(0, 1)" beside a ``distribution`` column that
correctly read ``Normal(0, 0.3)``, with a ``panel`` pointing at the wider density.
A reader deciding whether a flat fitted slope is evidence or prior shrinkage was
shown the wrong prior.

**No resampling.** The prior table is a property of the model's *structure*, not
of its posterior: this rebuilds the model from the fit's own recorded plan and
re-runs the same writer the fit used, then checks the rebuilt free-variable set
against the stored table before writing anything. Nothing else in the directory is
touched, and the stored trace is never opened.

    regenerate_priors_table.py lrp-rli-jm-001-reporting          # one fit dir
    regenerate_priors_table.py lrp-rli-jm-001 --dry-run          # show the diff

A family whose build this script cannot reproduce is reported as a skip with that
reason rather than half-written: the alternative is a table describing a model
other than the one that was fitted, which is the defect being repaired.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from rich.console import Console

from language_reading_predictors import paths as _paths
from language_reading_predictors.statistical_models import priors as _priors
from language_reading_predictors.statistical_models.prior_artifacts import (
    _prior_table_overrides,
)
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
    from language_reading_predictors.statistical_models.factories import (
        build_joint_mechanism_model,
    )
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

    prepared_all = load_and_prepare(**plan.prepare_kwargs())
    active = tuple(c for c in plan.declared_adjustment if c in prepared_all.covariates)
    if active != plan.active_adjustment:
        plan = plan.with_active_adjustment(active)
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

    context = SimpleNamespace(
        spec=spec,
        model=built.model,
        prepared=built.prepared,
        resolved_plan=getattr(built, "plan", None),
        output_dir=str(fit_dir),
        tables={},
        reporting=SimpleNamespace(ci_prob=float(config.get("ci_prob") or 0.89)),
    )
    # Exactly what ``emit_priors`` will write, family overrides included. Previewing
    # the bare table would show a diff the write does not make — and would have
    # reported two regressions here that the overrides in fact prevent.
    ctor_overrides, role_overrides, rationale_overrides = _prior_table_overrides(context)
    table = _priors.priors_table(
        built.model,
        ctor_overrides=ctor_overrides,
        role_overrides=role_overrides,
        rationale_overrides=rationale_overrides,
    )
    changed = [
        f"{row.parameter}: {column} {getattr(old, column)!r} -> {getattr(row, column)!r}"
        for old, row in zip(stored.itertuples(), table.itertuples(), strict=True)
        for column in ("distribution", "role", "rationale", "panel")
        if _normalise(getattr(old, column)) != _normalise(getattr(row, column))
    ]
    if not changed:
        return "unchanged", "the stored table already matches the rebuilt model"
    if dry_run:
        return "would rewrite", "; ".join(changed)

    table.to_csv(stored_path, index=False)
    orphaned = _drop_orphaned_panels(fit_dir, table)
    _prune_manifest(fit_dir, orphaned)
    detail = "; ".join(changed)
    if orphaned:
        detail += f" [removed {', '.join(sorted(orphaned))}]"
    return "rewritten", detail


def _drop_orphaned_panels(fit_dir: Path, table: pd.DataFrame) -> set[str]:
    """Delete prior-PDF panels the corrected table no longer points at.

    Only the orphans. ``emit_priors`` would redraw *every* panel, and a panel
    redrawn now is laid out by today's matplotlib rather than the fit's — different
    canvas dimensions for an identical density, which would make four unrelated
    figures differ from the rest of the corpus for no reason. Leaving the orphan is
    not an option either: ``prior_beta_mech.png`` plots ``Normal(0, 1)``, a density
    this model does not use, which is the defect being repaired.
    """
    wanted = {str(panel) for panel in table["panel"] if str(panel) not in {"", "nan"}}
    removed: set[str] = set()
    for key in _priors.ALL_PRIORS:
        if key in wanted:
            continue
        for ext in ("png", "svg"):
            panel = fit_dir / f"prior_{key}.{ext}"
            if panel.exists():
                panel.unlink()
                removed.add(panel.name)
    return removed


def _prune_manifest(fit_dir: Path, removed: set[str]) -> None:
    """Drop manifest rows for the deleted panels, changing nothing else.

    A full rescan would also absorb whatever has appeared in the directory since
    the fit — a rendered ``index.html`` and its Quarto asset tree — turning a
    fit-time inventory into a directory listing.
    """
    if not removed:
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
    manifest["artifacts"] = kept
    manifest["n_untracked"] = int(manifest.get("n_untracked", 0)) - dropped
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
