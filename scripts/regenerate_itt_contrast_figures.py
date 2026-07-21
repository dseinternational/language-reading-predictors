# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regenerate the ITT randomised-contrast figures straight from saved traces.

Two figure families are redrawn from ``trace.nc`` with no resampling, so colour
or layout changes can be backfilled onto existing single-outcome ITT fits:

* the predicted-scores panel + icon array (``predicted_scores``); and
* the intervention-vs-no-intervention overlap curves (``arm_overlap_mean`` and,
  for graded outcomes, ``arm_overlap_predictive``).

Both reuse the fit-time generators, so the emitted CSVs are byte-for-byte the
same as a fresh fit (identical draws under the recorded seed) — only the figures
change. The script also refreshes each dir's copied ``_partials`` so a
subsequent ``quarto render`` picks up any report-template changes.

Targets:

    regenerate_itt_contrast_figures.py all                # every single-outcome ITT fit dir
    regenerate_itt_contrast_figures.py lrp-rli-itt-010    # one model (all its -<config> dirs)
    regenerate_itt_contrast_figures.py lrp-rli-itt-010-reporting  # one specific fit dir

Only ``kind='itt'`` single-outcome fits are eligible; joint (multi-outcome) ITT
fits and any model declaring a treatment-effect moderator are skipped with a
note. Honours the output-root override (``DSE_LRP_OUTPUT_DIR`` or
``--output-dir``).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import arviz as az
import numpy as np
from rich.console import Console

from language_reading_predictors import paths as _paths
from language_reading_predictors.statistical_models import reporting as _report
from language_reading_predictors.statistical_models.arm_overlap import (
    write_arm_overlap_artifacts,
)
from language_reading_predictors.statistical_models.effect_plots import (
    write_rope_figures,
)
from language_reading_predictors.statistical_models.measures import (
    MEASURES,
    ROPE_DELTA,
    ROPE_DELTA_PROB,
)
from language_reading_predictors.statistical_models.predicted_scores import (
    write_predicted_scores_artifacts,
)

_console = Console()
_REPO_ROOT = Path(__file__).resolve().parents[1]
_PARTIALS_SRC = _REPO_ROOT / "docs" / "models" / "_partials"

# Reference-population and contrast-status strings, kept in step with the two
# ITT call sites in pipeline.py so backfilled CSVs read identically to fresh fits.
_GRADED = {
    "population": "new child; covariate profiles drawn from the fitted ITT analysis rows",
    "contrast_status": "randomised contrast (ITT)",
    "event_label": "off the floor at follow-up",
}
_FLOOR = {
    "population": (
        "new child; covariate profiles drawn from the baseline-floored "
        "at-risk analysis rows"
    ),
    "contrast_status": "randomised contrast (floor-rule subgroup ITT)",
    "event_label": "off the floor at t2",
}


def _subdirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(d for d in root.iterdir() if d.is_dir())


def resolve_targets(target: str) -> list[Path]:
    root = _paths.stat_models_dir()
    if target == "all":
        return _subdirs(root)
    return [
        d
        for d in _subdirs(root)
        if d.name == target or d.name.startswith(f"{target}-")
    ]


def _regenerate_one(fit_dir: Path) -> str:
    """Redraw both contrast figure families for one fit dir; return a status string."""
    config_path = fit_dir / "config.json"
    trace_path = fit_dir / "trace.nc"
    if not config_path.exists() or not trace_path.exists():
        return "skipped (no config.json/trace.nc)"
    config = json.loads(config_path.read_text())
    if config.get("kind") != "itt":
        return f"skipped (kind={config.get('kind')!r}, not itt)"
    plan = config.get("resolved_run_plan") or {}
    symbol = plan.get("outcome_symbol")
    if not symbol:
        return "skipped (no outcome_symbol; joint fit)"
    if plan.get("tau_moderator_symbol"):
        return "skipped (treatment-effect moderator not reconstructable from trace)"
    if symbol not in MEASURES:
        return f"skipped (unknown outcome symbol {symbol!r})"

    trace = az.from_netcdf(str(trace_path))
    if "G" not in trace.constant_data:
        return "skipped (no treatment indicator G in constant_data)"
    G = np.asarray(trace.constant_data["G"].values, dtype=float)

    floored = bool(plan.get("floor_rule"))
    likelihood = (
        "bernoulli" if plan.get("headline_likelihood") == "bernoulli_offfloor"
        else "beta_binomial"
    )
    n_trials = 1 if floored else int(MEASURES[symbol].n_trials)
    ci_prob = float(config.get("ci_prob", 0.89))
    random_seed = (config.get("sampling") or {}).get("random_seed")
    strings = _FLOOR if floored else _GRADED
    delta = ROPE_DELTA_PROB.get(symbol) if floored else ROPE_DELTA.get(symbol)

    common = dict(
        outcome_symbol=symbol,
        item_label=MEASURES[symbol].label,
        G=G,
        n_trials=n_trials,
        term="tau",
        varying_term="" if floored else "tau_i",
        likelihood=likelihood,
        ci_prob=ci_prob,
        random_seed=random_seed,
    )

    write_predicted_scores_artifacts(
        str(fit_dir), trace, delta=delta, split=True, **common, **strings
    )
    tables = write_arm_overlap_artifacts(str(fit_dir), trace, **common, **strings)

    # ROPE effect + benefit-curve as individual files. Recompute the items-scale
    # effect draws exactly as _save_rope_plot does at fit time.
    _, ame_prob = _report._itt_ame_draws(
        trace, G=G, term="tau", varying_term="" if floored else "tau_i",
    )
    write_rope_figures(
        str(fit_dir), ame_prob * float(n_trials),
        symbol=symbol, delta=delta, n_trials=n_trials, split=True,
    )

    # Refresh the copied partials so a re-render surfaces any template changes.
    if _PARTIALS_SRC.is_dir():
        shutil.copytree(_PARTIALS_SRC, fit_dir / "_partials", dirs_exist_ok=True)

    figs = ["predicted_scores", "predicted_effect", "rope_summary",
            "rope_benefit_curve", *sorted(tables)]
    return f"ok ({', '.join(figs)})"


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
    n_ok = 0
    for d in targets:
        try:
            status = _regenerate_one(d)
        except Exception as exc:  # pragma: no cover - defensive per-dir isolation
            status = f"[red]failed: {exc}[/red]"
        if status.startswith("ok"):
            n_ok += 1
        _console.print(f"  {d.name}: {status}")
    _console.print(f"Regenerated contrast figures for {n_ok} ITT fit(s).")


if __name__ == "__main__":
    main()
