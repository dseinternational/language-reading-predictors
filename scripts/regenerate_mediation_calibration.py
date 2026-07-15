# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regenerate intervention-session calibration over existing mediation fits.

The #324 calibration is generated at fit time after ``mediation_sensitivity.csv``.
This script backfills or refreshes it from the existing sensitivity and
dose-response artefacts plus the source data; it builds the mediation data
container to recover the exact sample and standardisation but does not sample or
change any fitted model.

Targets::

    regenerate_mediation_calibration.py all
    regenerate_mediation_calibration.py lrp-rli-med-086
    regenerate_mediation_calibration.py lrp-rli-med-086-reporting

The supported set is MED-059, MED-086 and MED-087.  MED-092 has a different
period-stacked exposure/estimand and remains outside this calibration; the
two-mediator MED-064 calibration is handled separately by its fit pipeline (#335).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from rich.console import Console

from language_reading_predictors import paths as _paths
from language_reading_predictors.statistical_models import factories as _factories
from language_reading_predictors.statistical_models.mediation_calibration import (
    IS_CALIBRATION_SOURCES,
    generate_is_calibration,
)
from language_reading_predictors.statistical_models.pipeline import (
    prepare_mediation_data,
)
from language_reading_predictors.statistical_models.registry import discover_models

_console = Console()


def _subdirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(d for d in root.iterdir() if d.is_dir())


def resolve_targets(target: str) -> list[Path]:
    """Supported fitted-model directories matching an id or id-plus-config."""
    candidates = _subdirs(_paths.stat_models_dir())
    supported = tuple(IS_CALIBRATION_SOURCES)
    if target == "all":
        return [d for d in candidates if any(d.name.startswith(f"{mid}-") for mid in supported)]
    if target in IS_CALIBRATION_SOURCES:
        return [d for d in candidates if d.name.startswith(f"{target}-")]
    return [
        d
        for d in candidates
        if d.name == target
        and any(d.name.startswith(f"{mid}-") for mid in supported)
    ]


def _config_name(output_dir: Path, model_id: str) -> str:
    prefix = f"{model_id}-"
    if not output_dir.name.startswith(prefix):
        raise ValueError(f"{output_dir.name!r} is not a fit directory for {model_id}")
    return output_dir.name[len(prefix) :]


def regenerate_one(output_dir: Path, models: dict) -> pd.DataFrame:
    """Regenerate one calibration CSV without posterior sampling."""
    config_path = output_dir / "config.json"
    sensitivity_path = output_dir / "mediation_sensitivity.csv"
    summary_path = output_dir / "mediation_sensitivity_summary.csv"
    if not config_path.exists() or not sensitivity_path.exists() or not summary_path.exists():
        raise ValueError("config or mediation sensitivity artefacts are missing")
    with config_path.open() as f:
        config_json = json.load(f)
    model_id = config_json["model_id"]
    if model_id not in IS_CALIBRATION_SOURCES:
        raise ValueError(f"{model_id} is not an IS-calibration target")
    module = models[model_id]
    spec = module.SPEC
    prepared, confounders = prepare_mediation_data(spec)
    built, med_data = _factories.build_mediation_model(
        prepared,
        mediator_symbol=spec.mechanism_symbol or "L",
        outcome_symbol=spec.outcome_symbol or "W",
        confounder_symbols=confounders,
        mediator_kind=spec.extra.get("mediator_kind", "beta_binomial"),
        route_symbols=tuple(spec.extra.get("route_symbols", ())),
        outcome_kind=spec.extra.get("outcome_kind", "beta_binomial"),
    )
    sweep = pd.read_csv(sensitivity_path)
    sensitivity_summary = pd.read_csv(summary_path).iloc[0].to_dict()
    result = generate_is_calibration(
        spec,
        config=_config_name(output_dir, model_id),
        output_dir=output_dir,
        prepared=built.prepared,
        med=med_data,
        sweep=sweep,
        sensitivity_summary=sensitivity_summary,
    )
    if result is None:
        raise ValueError(f"{model_id} unexpectedly produced no IS calibration")
    result.to_csv(output_dir / "mediation_is_calibration.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", help="'all', a supported model id, or a fit directory name")
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
        raise SystemExit(f"No supported mediation fit directories matched {args.target!r}.")
    models = discover_models()
    failures = 0
    for output_dir in targets:
        try:
            result = regenerate_one(output_dir, models)
            row = result.iloc[0]
            detail = row.get("verdict", row.get("reason", "unknown"))
            _console.print(f"  {output_dir.name}: {row['status']} ({detail})")
        except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
            failures += 1
            _console.print(f"  {output_dir.name}: failed ({exc})", style="bold red")
    if failures:
        raise SystemExit(f"Calibration regeneration failed for {failures} fit(s).")


if __name__ == "__main__":
    main()
