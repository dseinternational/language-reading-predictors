#!/usr/bin/env python
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Backfill the #586 / #602 mechanism artefacts over stored fits, without resampling.

The #586 batch changes four things a stored ``reporting`` fit already has the
information to answer, none of which needs a new posterior:

* ``priors_table.csv`` and the prior density panels — the mechanism lengthscale was
  panelled from the shared ``InverseGamma(3, 1)`` constructor and captioned with a
  hard-coded ``InverseGamma(5, 5)`` rationale, whatever the model fitted (finding 3);
* ``readiness_threshold.csv`` — the located steepest interval now carries its scale,
  its boundary and stability diagnostics and a ``knee_well_defined`` verdict, so a
  report can stop calling every net rise a knee (finding 1);
* ``exposure_support.csv`` — fitted exposure support by period and arm, so structural
  non-overlap is visible rather than inferred (finding 2);
* ``config.json``'s ``effective_adjustment`` — the moderation terms that carry
  coefficients but were never named in the fitted record (finding 9).

#602 adds the family's single declared natural-scale estimand, which is likewise pure
post-processing of a stored posterior:

* ``mechanism_summary.csv`` — now two labelled rows, the headline interquartile
  contrast standardised over the fitted rows and the observed-range contrast as an
  explicit secondary, each carrying a machine-readable ``estimand``;
* ``mechanism_curve_items.csv`` and its figure — the same standardised quantity across
  the observed exposure range, so the worked-example points lie on the curve;
* ``mechanism_curve.csv`` — the logit-scale view, standardised over the same rows and
  deduplicated to one row per distinct exposure value;
* ``readiness_threshold.csv`` — gains ``items_*`` columns locating the steepest
  interval of the *expected-items* curve under the same reference population;
* ``dispersion_summary.csv`` — the fitted concentration against its prior and the
  implied variance-inflation factor (#605);
* ``config.json``'s ``extra.mechanism_items`` — the reference points the report
  partial renders its caption from.

The observed-range row reproduces each fit's previously-published headline exactly
(verified over the stored mech-058/097/101 traces), so the regeneration adds the
headline rather than silently restating the old number as a new estimand.

``key_findings.json`` is regenerated afterwards (it reads the CSVs above), by
delegating to ``regenerate_key_findings.py`` rather than duplicating its gate
interlock.

**What this script deliberately cannot fix.** Four models change their *fitted rows
or specification* in this batch — mech-063, mech-163 (the ``pre_required`` contract),
mech-158 (matched to mech-058) and mech-191 (restricted to on-intervention periods).
Their stored posteriors were sampled on a different analysis frame, so no amount of
post-processing makes them current: they are reported as ``needs refit`` and skipped,
and the run exits non-zero so a sweep cannot look clean while leaving them stale.

The model is rebuilt from its spec **without sampling** so the prior writers have a
PyMC graph to read, and the rebuilt frame's row count is checked against the stored
``config.json`` before anything is written — a mismatch means the analysis frame has
moved under the trace, which is exactly the ``needs refit`` case.

Targets mirror ``regenerate_psense.py``:

    regenerate_mechanism_artefacts.py all
    regenerate_mechanism_artefacts.py lrp-rli-mech-058
    regenerate_mechanism_artefacts.py lrp-rli-mech-058-reporting
"""

from __future__ import annotations

import argparse
import importlib
import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from rich.console import Console

from language_reading_predictors import paths as _paths
from language_reading_predictors.statistical_models import mechanism as _mechanism
from language_reading_predictors.statistical_models import reporting as _report
from language_reading_predictors.statistical_models.adjustment import (
    effective_adjustment,
)
from language_reading_predictors.statistical_models.measures import MEASURES
from language_reading_predictors.statistical_models.pipelines.mechanism import (
    _COVARIATE_EXPOSURE_LABELS,
    _exposure_term_records,
)

_console = Console()

#: Models whose fitted rows or specification changed in #586. Post-processing a
#: posterior sampled on the old frame would publish current-looking artefacts over a
#: stale fit, which is worse than leaving the gap visible.
NEEDS_REFIT: dict[str, str] = {
    "lrp-rli-mech-063": "pre_required contract (#586 finding 4): 151 -> 155 rows",
    "lrp-rli-mech-163": "pre_required contract (#586 finding 4): 151 -> 155 rows",
    "lrp-rli-mech-158": "matched to mech-058 (#586 finding 5): outcomes, HSGP basis, "
    "lengthscale prior",
    "lrp-rli-mech-191": "on-intervention restriction (#586 finding 2): 156 -> 128 rows",
}


def _subdirs(root: Path) -> list[Path]:
    """Published fit directories, excluding transactions and manual backups.

    A published directory is exactly ``<model-id>-<config>``. A leading dot is an
    in-flight or abandoned output transaction (``.reset_output_dir`` stages there),
    and a dot anywhere else marks a manual backup copy — the house convention for a
    remediation batch is ``<dir>.pre-review-fix-<date>`` beside the live directory.
    Backfilling into either writes artefacts nobody will read, or silently updates a
    snapshot that exists precisely to preserve the pre-change state.
    """
    if not root.is_dir():
        return []
    return sorted(d for d in root.iterdir() if d.is_dir() and "." not in d.name)


def resolve_targets(target: str) -> list[Path]:
    """Mechanism fit output dirs for the requested target."""
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
    return [d for d in candidates if "-mech-" in d.name]


def _spec_for(model_id: str):
    """The registered ``SPEC`` for a mechanism model id, or ``None``."""
    module = "language_reading_predictors.statistical_models." + model_id.replace(
        "lrp-rli-mech-", "lrp_rli_mech_"
    ).replace("-", "_")
    try:
        return importlib.import_module(module).SPEC
    except (ImportError, AttributeError):
        return None


def _regenerate(fit_dir: Path, *, dry_run: bool) -> tuple[str, str]:
    """Return ``(status, detail)`` for one fit directory."""
    config_path = fit_dir / "config.json"
    if not config_path.exists():
        return "skipped", "no config.json"
    with open(config_path) as handle:
        stored = json.load(handle)
    model_id = stored.get("model_id", "")
    if model_id in NEEDS_REFIT:
        return "needs refit", NEEDS_REFIT[model_id]
    if not (fit_dir / "trace.nc").exists():
        return "skipped", "no trace.nc"
    spec = _spec_for(model_id)
    if spec is None:
        return "skipped", f"no registered module for {model_id}"

    import arviz as az

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = _mechanism.resolve_mechanism_plan(spec)
        built = _mechanism.build_mechanism_for_plan(plan)
        trace = az.from_netcdf(fit_dir / "trace.nc")
    # The *fitted* frame, i.e. after the factory keep-mask — the same object the
    # pipeline sees, because ``stages.attach_built`` replaces ``ctx.prepared`` with
    # ``built.prepared``. Using ``plan.prepared`` here would feed the writers the
    # pre-mask frame (157 rows against the trace's 156 for mech-058) and either raise
    # or, worse, silently misalign the exposure vector against the posterior.
    prepared = built.prepared
    run_plan = plan.run_plan

    # The rebuilt frame must be the one the stored posterior was sampled on. Compare
    # against the *fitted* counts — the factory keep-mask drops rows the loader kept,
    # so ``prepared.n_obs`` is systematically the larger number and comparing it
    # would flag every model.
    fitted_obs = len(built.model.coords["obs_id"])
    fitted_children = len(built.model.coords["child"])
    stored_obs, stored_children = stored.get("n_obs"), stored.get("n_children")
    if stored_obs is not None and int(stored_obs) != fitted_obs:
        return (
            "needs refit",
            f"analysis frame moved: stored {stored_obs} fitted rows, rebuild gives "
            f"{fitted_obs}",
        )
    if stored_children is not None and int(stored_children) != fitted_children:
        return (
            "needs refit",
            f"analysis frame moved: stored {stored_children} children, rebuild gives "
            f"{fitted_children}",
        )

    written: list[str] = []
    if dry_run:
        return "would write", (
            "priors_table.csv + panels, mechanism_summary.csv (both declared "
            "contrasts), mechanism_curve_items.csv + figure, mechanism_curve.csv, "
            "readiness_threshold.csv (+ items scale), dispersion_summary.csv, "
            "exposure_support.csv, config.json effective_adjustment + mechanism_items"
        )

    # -- priors (finding 3) -------------------------------------------------
    # ``emit_priors`` needs only ``spec``/``model``/``output_dir``/``resolved_plan``;
    # a namespace stands in for the fit context so nothing re-runs a sampler and no
    # output directory is reset.
    from language_reading_predictors.statistical_models.prior_artifacts import (
        emit_priors,
    )

    emit_priors(
        SimpleNamespace(
            spec=spec,
            model=built.model,
            output_dir=str(fit_dir),
            resolved_plan=run_plan,
            artifact_manifest=None,
        )
    )
    written.append("priors_table.csv")

    ci_prob = float(stored.get("ci_prob", 0.89))

    # -- the declared natural-scale estimand (#602) --------------------------
    items_worked = _regenerate_declared_estimand(
        fit_dir, trace, run_plan, prepared, ci_prob=ci_prob
    )
    if items_worked:
        written.append("mechanism_summary.csv, mechanism_curve_items.csv")
    written.append("mechanism_curve.csv")
    _regenerate_logit_curve(fit_dir, trace, run_plan, prepared)
    dispersion = _dispersion_summary(trace, run_plan, prepared, ci_prob=ci_prob)
    if dispersion is not None:
        dispersion.to_csv(fit_dir / "dispersion_summary.csv", index=False)
        written.append("dispersion_summary.csv")

    # -- steepest interval (finding 1, plus the #602 items scale) ------------
    if not run_plan.linear_mechanism and "f_mech" in trace.posterior:
        if run_plan.mechanism_is_covariate:
            scaler = prepared.covariate_scalers.get(run_plan.mechanism_symbol)
            z = np.asarray(
                prepared.covariates[run_plan.mechanism_symbol], dtype=float
            )
            values = scaler.inverse(z) if scaler is not None else z
            summary = _report.readiness_threshold(
                trace, exposure_values=values, ci_prob=ci_prob
            )
            x_obs = values
        else:
            n_trials = MEASURES[run_plan.mechanism_symbol].n_trials
            summary = _report.readiness_threshold(
                trace, n_trials=n_trials, ci_prob=ci_prob
            )
            ell = np.asarray(
                trace.constant_data["mech_post_logit"].values
            ).reshape(-1)
            x_obs = np.clip(
                (n_trials + 1.0) / (1.0 + np.exp(-ell)) - 0.5, 0.0, float(n_trials)
            )
        summary.update(
            _items_scale_knee(trace, run_plan, prepared, x_obs=x_obs, ci_prob=ci_prob)
        )
        pd.DataFrame([summary]).to_csv(
            fit_dir / "readiness_threshold.csv", index=False
        )
        written.append(
            "readiness_threshold.csv"
            + ("" if summary["knee_well_defined"] else " (not a qualified knee)")
        )

    # -- exposure support (finding 2) ---------------------------------------
    support = _exposure_support(run_plan, prepared)
    if support is not None:
        support.to_csv(fit_dir / "exposure_support.csv", index=False)
        written.append("exposure_support.csv")

    # -- fitted adjustment record (finding 9) -------------------------------
    record = effective_adjustment(
        spec,
        prepared,
        measure_confounders=tuple(
            s for s in plan.confounders if s in ("G", "A") or s in MEASURES
        ),
        adjust_for=plan.adjust_for,
        requested_adjust_for=run_plan.adjust_for
        + ((run_plan.ability_covariate,) if run_plan.ability_covariate else ()),
        baseline_symbol=run_plan.adjust_baseline_symbol,
        moderator_symbol=run_plan.moderator_symbol,
        moderator_is_covariate=run_plan.moderator_is_covariate,
        moderator_interaction=(
            run_plan.moderator_symbol is not None and run_plan.include_interaction
        ),
        exposure_terms=_exposure_term_records(run_plan, prepared),
    )
    stored.setdefault("extra", {})["effective_adjustment"] = record
    if items_worked:
        stored["extra"]["mechanism_items"] = items_worked
    stored["resolved_run_plan"] = run_plan.as_dict()
    with open(config_path, "w") as handle:
        json.dump(stored, handle, indent=2)
    written.append("config.json")

    if _sync_report_template(fit_dir, model_id):
        written.append("index.qmd + _partials")

    return "written", ", ".join(written)


def _sync_report_template(fit_dir: Path, model_id: str) -> bool:
    """Re-copy the report template and shared partials beside a stored fit.

    Each fit directory carries its own ``_partials`` snapshot from fit time, so a
    backfill that rewrites the CSVs but leaves the partials stale renders the new
    numbers through the old prose — or, worse, publishes a section a current partial
    would have withheld. Refreshing them here keeps the regeneration honest, and is
    the same copy ``publication.copy_report_template`` performs at fit time.
    """
    import shutil

    template = _paths.DOCS_DIR / "models" / model_id / "index.qmd"
    partials = _paths.DOCS_DIR / "models" / "_partials"
    copied = False
    if template.exists():
        shutil.copy(template, fit_dir / "index.qmd")
        copied = True
    if partials.is_dir():
        shutil.copytree(partials, fit_dir / "_partials", dirs_exist_ok=True)
        copied = True
    return copied


def _items_axis(run_plan, prepared) -> tuple[np.ndarray, str, int | None]:
    """Mirror ``pipelines.mechanism._mechanism_items_axis`` over a stored fit."""
    symbol = run_plan.mechanism_symbol
    if run_plan.mechanism_is_covariate:
        scaler = prepared.covariate_scalers.get(symbol)
        z = np.asarray(prepared.covariates[symbol], dtype=float)
        values = scaler.inverse(z) if scaler is not None else z
        return values, _COVARIATE_EXPOSURE_LABELS.get(symbol, symbol), None
    if run_plan.mechanism_at_pre:
        return (
            np.asarray(prepared.pre_counts[symbol], dtype=float),
            f"{MEASURES[symbol].label} (period start)",
            MEASURES[symbol].n_trials,
        )
    return (
        np.asarray(prepared.post_counts[symbol], dtype=float),
        MEASURES[symbol].label,
        MEASURES[symbol].n_trials,
    )


def _regenerate_declared_estimand(
    fit_dir: Path, trace, run_plan, prepared, *, ci_prob: float
) -> dict:
    """Rewrite ``mechanism_summary.csv`` and the items curve on the #602 estimand."""
    from language_reading_predictors.statistical_models.mechanism_items import (
        mechanism_summary_table,
        write_mechanism_items_artifacts,
    )

    outcome = run_plan.outcome_symbol
    x_exposure, exposure_label, exposure_n_trials = _items_axis(run_plan, prepared)
    worked = write_mechanism_items_artifacts(
        str(fit_dir),
        trace,
        x_exposure=x_exposure,
        outcome_symbol=outcome,
        outcome_label=MEASURES[outcome].label,
        n_trials_outcome=MEASURES[outcome].n_trials,
        exposure_label=exposure_label,
        exposure_is_covariate=run_plan.mechanism_is_covariate,
        exposure_n_trials=exposure_n_trials,
        ci_prob=ci_prob,
        ref_quantiles=run_plan.items_ref_quantiles,
        outcome_off_floor=False,
    )
    symbol = run_plan.mechanism_symbol
    mechanism_summary_table(
        worked,
        exposure_unit=(
            f"{symbol} raw-score units"
            if run_plan.mechanism_is_covariate
            else f"{symbol} items"
        ),
    ).to_csv(fit_dir / "mechanism_summary.csv", index=False)
    return worked


def _regenerate_logit_curve(fit_dir: Path, trace, run_plan, prepared) -> None:
    """Rewrite ``mechanism_curve.csv`` as the row-standardised logit contribution."""
    from language_reading_predictors.statistical_models.mechanism_items import (
        resolve_mechanism_terms,
    )
    from language_reading_predictors.statistical_models.preprocessing import logit_safe

    symbol = run_plan.mechanism_symbol
    if run_plan.mechanism_is_covariate:
        scaler = prepared.covariate_scalers.get(symbol)
        z = np.asarray(prepared.covariates[symbol], dtype=float)
        x_vals = scaler.inverse(z) if scaler is not None else z
        x_col = "mech_x"
    elif run_plan.mechanism_at_pre:
        x_vals = np.asarray(prepared.pre_logit[symbol], dtype=float)
        x_col = "mech_logit"
    else:
        x_vals = np.asarray(
            logit_safe(prepared.post_counts[symbol], MEASURES[symbol].n_trials),
            dtype=float,
        )
        x_col = "mech_logit"
    terms = resolve_mechanism_terms(
        trace, x_exposure=x_vals, exposure_n_trials=None, group="posterior"
    )
    xs = np.unique(x_vals)
    f_ord = np.stack(
        [
            np.broadcast_to(
                terms.contribution_at(float(v)),
                (x_vals.size, terms.fitted.shape[1]),
            ).mean(axis=0)
            for v in xs
        ]
    )
    pd.DataFrame(
        {
            x_col: xs,
            "f_mean": f_ord.mean(axis=1),
            "f_lo": np.quantile(f_ord, 0.055, axis=1),
            "f_hi": np.quantile(f_ord, 0.945, axis=1),
            "f_lo50": np.quantile(f_ord, 0.25, axis=1),
            "f_hi50": np.quantile(f_ord, 0.75, axis=1),
        }
    ).to_csv(fit_dir / "mechanism_curve.csv", index=False)


def _items_scale_knee(trace, run_plan, prepared, *, x_obs, ci_prob: float) -> dict:
    """Mirror ``pipelines.mechanism._items_scale_knee`` over a stored fit."""
    from language_reading_predictors.statistical_models.mechanism_items import (
        standardised_items_by_row,
    )

    outcome = run_plan.outcome_symbol
    x_exposure, _label, exposure_n_trials = _items_axis(run_plan, prepared)
    try:
        items_rows = standardised_items_by_row(
            trace,
            x_exposure=x_exposure,
            n_trials_outcome=MEASURES[outcome].n_trials,
            exposure_n_trials=(
                None if run_plan.mechanism_is_covariate else exposure_n_trials
            ),
        )
        items = _report.readiness_threshold(
            trace,
            exposure_values=np.asarray(x_obs, dtype=float),
            ci_prob=ci_prob,
            curve=items_rows,
            scale="expected_items",
        )
    except (KeyError, ValueError):
        return {}
    keep = (
        "knee_count_median",
        "knee_count_ci_low",
        "knee_count_ci_high",
        "half_rise_count_median",
        "slope_below_knee_median",
        "slope_above_knee_median",
        "increasing_frac",
        "steepest_interval_index",
        "steepest_interval_share",
        "boundary_pinned",
        "prob_slope_above_gt_below",
        "knee_well_defined",
        "scale",
    )
    return {f"items_{k}": items[k] for k in keep if k in items}


def _dispersion_summary(trace, run_plan, prepared, *, ci_prob: float):
    """Mirror ``pipelines.mechanism._write_dispersion_summary`` over a stored fit."""
    if "kappa" not in trace.posterior:
        return None
    outcome = run_plan.outcome_symbol
    n_trials = int(MEASURES[outcome].n_trials)
    kappa = np.asarray(trace.posterior["kappa"].values, dtype=float).ravel()
    inflation = (kappa + n_trials) / (kappa + 1.0)
    lo_q, hi_q = (1 - ci_prob) / 2, 1 - (1 - ci_prob) / 2
    family = run_plan.kappa_prior_family
    default_sigma = 0.25 if family == "halfnormal_inverse_sqrt" else 50.0
    sigma = default_sigma if run_plan.kappa_sigma is None else float(run_plan.kappa_sigma)
    return pd.DataFrame(
        [
            {
                "outcome_symbol": outcome,
                "n_trials": n_trials,
                "kappa_prior_family": family,
                "kappa_prior_sigma": sigma,
                "kappa_prior_label": (
                    f"1/sqrt(kappa) ~ HalfNormal({sigma:g})"
                    if family == "halfnormal_inverse_sqrt"
                    else f"kappa ~ HalfNormal({sigma:g})"
                ),
                "reaches_near_binomial": family == "halfnormal_inverse_sqrt",
                "kappa_median": float(np.median(kappa)),
                "kappa_lo": float(np.quantile(kappa, lo_q)),
                "kappa_hi": float(np.quantile(kappa, hi_q)),
                "variance_inflation_median": float(np.median(inflation)),
                "variance_inflation_lo": float(np.quantile(inflation, lo_q)),
                "variance_inflation_hi": float(np.quantile(inflation, hi_q)),
                "kappa_for_10pct_of_binomial": 10.0 * (n_trials - 1) - 1.0,
                "prob_within_10pct_of_binomial": float(np.mean(inflation <= 1.1)),
                "ci_prob": ci_prob,
                "n_obs": int(prepared.n_obs),
            }
        ]
    )


def _exposure_support(run_plan, prepared) -> pd.DataFrame | None:
    """Mirror ``pipelines.mechanism._write_exposure_support`` over a stored fit."""
    symbol = run_plan.mechanism_symbol
    if run_plan.mechanism_is_covariate:
        scaler = prepared.covariate_scalers.get(symbol)
        z = np.asarray(prepared.covariates[symbol], dtype=float)
        values = scaler.inverse(z) if scaler is not None else z
        unit = f"{symbol} raw score"
    elif symbol in prepared.post_counts:
        values = np.asarray(prepared.post_counts[symbol], dtype=float)
        unit = f"{symbol} items"
    else:
        return None
    arm_label = {0: "wait-list", 1: "immediate"}
    rows = []
    for phase in sorted({int(p) for p in prepared.phase}):
        for arm in sorted({int(g) for g in prepared.G}):
            cell = values[(prepared.phase == phase) & (prepared.G == arm)]
            if not cell.size:
                continue
            rows.append(
                {
                    "phase": phase,
                    "period": f"t{phase + 1}->t{phase + 2}",
                    "arm": arm_label.get(arm, str(arm)),
                    "exposure_unit": unit,
                    "n_rows": int(cell.size),
                    "n_at_zero": int((cell <= 0).sum()),
                    "min": float(np.min(cell)),
                    "q25": float(np.quantile(cell, 0.25)),
                    "median": float(np.median(cell)),
                    "q75": float(np.quantile(cell, 0.75)),
                    "max": float(np.max(cell)),
                }
            )
    return pd.DataFrame(rows) if rows else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target", help="'all', a model id, or a fit dir name (<id>-<config>)"
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
        raise SystemExit(f"No mechanism fit output directories matched {args.target!r}.")

    tally: dict[str, int] = {}
    for fit_dir in targets:
        status, detail = _regenerate(fit_dir, dry_run=args.dry_run)
        tally[status] = tally.get(status, 0) + 1
        colour = {
            "written": "green",
            "would write": "cyan",
            "skipped": "yellow",
            "needs refit": "yellow",
            "failed": "red",
        }.get(status, "white")
        _console.print(f"[{colour}]{status:11}[/{colour}] {fit_dir.name}: {detail}")

    _console.print()
    _console.print(", ".join(f"{k}: {v}" for k, v in sorted(tally.items())))
    _console.print(
        "\nNow re-run scripts/regenerate_key_findings.py over the same targets: the "
        "key-findings box reads the CSVs written above, and its gate interlock lives "
        "there."
    )
    if tally.get("failed") or tally.get("needs refit"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
