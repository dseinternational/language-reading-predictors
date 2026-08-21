# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Audit high-Pareto observations in a completed historical-growth fit.

The historical-growth likelihood scores one child-wave row at a time while
conditioning on that child's fitted random intercept. This command maps those
rows, excludes every point above the saved fit's ArviZ threshold in one matched
direct refit, gates every free variable and compares all reported growth
quantities. It tests coefficient stability; it does not repair the approximate
LOO score or estimate new-child predictive accuracy.

Usage::

    python scripts/historical_growth_influence.py lrp-rlm-hg-009 --config reporting
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import UTC, datetime
from multiprocessing import freeze_support
from pathlib import Path
from typing import Any

import arviz as az
import pandas as pd
import pymc as pm

from language_reading_predictors import paths
from language_reading_predictors.statistical_models import diagnostics as _diag
from language_reading_predictors.statistical_models import reporting as _report
from language_reading_predictors.statistical_models.datasets import resolve_dataset
from language_reading_predictors.statistical_models.factories import (
    build_historical_growth_model,
)
from language_reading_predictors.statistical_models.historical_growth import (
    exclude_historical_growth_observations,
    historical_growth_influence_summary,
    historical_growth_pareto_table,
    resolve_historical_growth_run_plan,
)
from language_reading_predictors.statistical_models.preprocessing import (
    load_longitudinal_panel,
)
from language_reading_predictors.statistical_models.registry import discover_models
from language_reading_predictors.statistical_models.sensitivity import sha256_file

# The fit writes its own required, manifest-recorded ``pareto_k.csv``; this audit
# must not overwrite it. Overwriting left the fit's manifest describing a file
# with different columns, and made the bundle's ``primary_pareto_k_sha256`` a
# hash of the script's own output rather than of a primary-fit artefact
# (2026-08-21 historical-families review, finding 11). Write the enriched
# row-mapped table under its own name instead.
PARETO_FILENAME = "pareto_k_rows.csv"
PRIMARY_PARETO_FILENAME = "pareto_k.csv"
SUMMARY_FILENAME = "historical_growth_influence_sensitivity.csv"
PROVENANCE_FILENAME = "historical_growth_influence_provenance.json"
TRACE_FILENAME = "trace_historical_growth_influence_sensitivity.nc"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required completed-fit artefact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _atomic_write_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(value: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _registered_spec(model_id: str):
    canonical = model_id.strip().lower()
    models = discover_models()
    if canonical not in models:
        raise ValueError(f"unknown registered statistical model: {model_id!r}")
    spec = models[canonical].SPEC
    if spec.kind != "historical_growth":
        raise ValueError(
            "historical-growth influence supports kind='historical_growth', "
            f"not {spec.kind!r}"
        )
    return spec


def _validate_primary(
    spec: Any,
    config_name: str,
    model_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = _read_json(model_dir / "config.json")
    diagnostics = _read_json(model_dir / "diagnostics_summary.json")
    if metadata.get("model_id") != spec.model_id:
        raise ValueError(
            f"completed fit is for {metadata.get('model_id')!r}, not {spec.model_id!r}"
        )
    if metadata.get("kind") != spec.kind:
        raise ValueError("completed fit kind does not match the registered model")
    if metadata.get("config_name") != config_name:
        raise ValueError(
            f"completed fit config is {metadata.get('config_name')!r}, "
            f"not {config_name!r}"
        )
    if not _report.convergence_gate_clean_passed(diagnostics):
        raise ValueError("completed primary fit did not pass its convergence gate")
    if metadata.get("publication_input_contract", {}).get("publication_ready") is not True:
        raise ValueError("completed primary fit did not pass its publication input contract")

    plan = resolve_historical_growth_run_plan(spec)
    stored_plan = metadata.get("resolved_run_plan")
    current_plan = json.loads(json.dumps(plan.as_dict()))
    if stored_plan != current_plan:
        raise ValueError("completed fit's resolved run plan has drifted from the registry")
    return metadata, diagnostics


def run(
    model_id: str,
    *,
    config: str,
    cores: int | None,
) -> pd.DataFrame | None:
    """Run one trace-bound historical-growth influence audit."""
    spec = _registered_spec(model_id)
    plan = resolve_historical_growth_run_plan(spec)
    model_dir = paths.stat_models_dir() / f"{spec.model_id}-{config}"
    metadata, _primary_diagnostics = _validate_primary(
        spec, config, model_dir
    )

    config_path = model_dir / "config.json"
    primary_trace_path = model_dir / "trace.nc"
    primary_config_sha256 = sha256_file(config_path)
    primary_trace_sha256 = sha256_file(primary_trace_path)
    recorded_trace_sha256 = metadata.get("trace_sha256")
    if recorded_trace_sha256 != primary_trace_sha256:
        raise ValueError(
            "primary trace hash does not match the completed fit's config.json"
        )

    dataset, measures = resolve_dataset(plan.study_id)
    panel = load_longitudinal_panel(
        dataset,
        [measures[plan.measure]],
        **plan.prepare_kwargs(),
    )
    primary_trace = az.from_netcdf(primary_trace_path)
    loo = az.loo(primary_trace, pointwise=True)
    pareto = historical_growth_pareto_table(panel, loo, measure=plan.measure)
    _atomic_write_csv(pareto, model_dir / PARETO_FILENAME)
    # Hash the fit's OWN untouched table, so the bundle is bound to a primary-fit
    # artefact rather than to something this run just wrote.
    pareto_path = model_dir / PRIMARY_PARETO_FILENAME
    if not pareto_path.is_file():
        raise FileNotFoundError(
            f"required completed-fit artefact is missing: {pareto_path}"
        )
    flagged = pareto.loc[~pareto["loo_reliable"]].copy()

    base_provenance: dict[str, Any] = {
        "schema_version": 1,
        "generated_by": "Codex/GPT-5",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "model_id": spec.model_id,
        "config": config,
        "sensitivity_target": "all observation rows above ArviZ good_k",
        "loo_unit": plan.loo_unit,
        "interpretation": (
            "Separate direct row-exclusion refit for coefficient stability; "
            "not exact LOO and not new-child prediction."
        ),
        "primary_config_sha256": primary_config_sha256,
        "primary_trace_sha256": primary_trace_sha256,
        "primary_pareto_k_sha256": sha256_file(pareto_path),
        "pareto_threshold": float(pareto["good_k_threshold"].iloc[0]),
        "max_pareto_k": float(pareto["pareto_k"].max()),
        "n_flagged_rows": int(len(flagged)),
        "flagged_observation_indices": [
            int(value) for value in flagged["observation_index"]
        ],
    }
    if flagged.empty:
        base_provenance["status"] = "no_flagged_observations"
        _atomic_write_json(base_provenance, model_dir / PROVENANCE_FILENAME)
        print("No observation exceeds the ArviZ Pareto-k threshold.")
        return None

    sensitivity_panel = exclude_historical_growth_observations(
        panel, flagged["observation_index"].to_numpy(dtype=int)
    )
    built = build_historical_growth_model(
        sensitivity_panel,
        **plan.factory_kwargs(),
    )
    sampling = metadata.get("sampling")
    required_sampling = {"draws", "tune", "chains", "target_accept", "random_seed"}
    if not isinstance(sampling, dict) or not required_sampling.issubset(sampling):
        raise ValueError("completed fit lacks its full sampling contract")
    chains = int(sampling["chains"])
    sampling_cores = min(chains, cores if cores is not None else (os.cpu_count() or 1))
    if sampling_cores < 1:
        raise ValueError("cores must be at least one")

    with built.model:
        sensitivity_trace = pm.sample(
            draws=int(sampling["draws"]),
            tune=int(sampling["tune"]),
            chains=chains,
            cores=sampling_cores,
            target_accept=float(sampling["target_accept"]),
            nuts_sampler="nutpie",
            return_inferencedata=True,
            random_seed=int(sampling["random_seed"]),
            progressbar=False,
        )

    free_variables = [rv.name for rv in built.model.free_RVs]
    convergence = _diag.subfit_convergence(
        sensitivity_trace,
        label=f"{spec.model_id} high-Pareto observation exclusion",
        var_names=free_variables,
    )
    sensitivity_trace.posterior.attrs[
        "historical_growth_influence_sampling_json"
    ] = json.dumps({**sampling, "cores": sampling_cores, "nuts_sampler": "nutpie"})
    sensitivity_trace.posterior.attrs[
        "historical_growth_influence_identity_json"
    ] = json.dumps(base_provenance, sort_keys=True)

    trace_path = model_dir / TRACE_FILENAME
    temporary_trace = trace_path.with_name(
        f".{trace_path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        sensitivity_trace.to_netcdf(temporary_trace)
        os.replace(temporary_trace, trace_path)
    finally:
        temporary_trace.unlink(missing_ok=True)
    sensitivity_trace_sha256 = sha256_file(trace_path)

    summary = historical_growth_influence_summary(
        primary_trace,
        sensitivity_trace,
        primary_panel=panel,
        sensitivity_panel=sensitivity_panel,
        measure=plan.measure,
        excluded_rows=flagged,
        sensitivity_converged=convergence["converged"],
    )
    summary.insert(0, "model_id", spec.model_id)
    summary.insert(1, "config", config)
    summary["primary_config_sha256"] = primary_config_sha256
    summary["primary_trace_sha256"] = primary_trace_sha256
    summary["primary_pareto_k_sha256"] = base_provenance[
        "primary_pareto_k_sha256"
    ]
    summary["sensitivity_trace_file"] = TRACE_FILENAME
    summary["sensitivity_trace_sha256"] = sensitivity_trace_sha256
    summary["sampling_draws"] = int(sampling["draws"])
    summary["sampling_tune"] = int(sampling["tune"])
    summary["sampling_chains"] = chains
    summary["sampling_cores"] = sampling_cores
    summary["sampling_target_accept"] = float(sampling["target_accept"])
    summary["sampling_random_seed"] = int(sampling["random_seed"])
    summary["sampling_nuts_sampler"] = "nutpie"
    summary["free_variables"] = json.dumps(free_variables)
    for key, value in convergence.items():
        summary[key] = value
    summary_path = model_dir / SUMMARY_FILENAME
    _atomic_write_csv(summary, summary_path)

    provenance = {
        **base_provenance,
        "status": (
            "completed" if convergence["converged"] is True else "not_converged"
        ),
        "sensitivity_trace_file": TRACE_FILENAME,
        "sensitivity_trace_sha256": sensitivity_trace_sha256,
        "sensitivity_summary_file": SUMMARY_FILENAME,
        "sensitivity_summary_sha256": sha256_file(summary_path),
        "sampling": {**sampling, "cores": sampling_cores, "nuts_sampler": "nutpie"},
        "free_variables": free_variables,
        "convergence": convergence,
    }
    _atomic_write_json(provenance, model_dir / PROVENANCE_FILENAME)

    print(
        f"Flagged {len(flagged)} of {len(pareto)} rows "
        f"(max k={pareto['pareto_k'].max():.3f}); "
        f"sensitivity converged={convergence['converged']}."
    )
    print(
        summary[
            [
                "label",
                "readgrp_label",
                "primary_q50",
                "sensitivity_q50",
                "median_shift",
                "median_direction_stable",
                "intervals_overlap",
            ]
        ].to_string(index=False)
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="registered historical-growth model id")
    parser.add_argument(
        "--config",
        default="reporting",
        help="completed fit configuration to reproduce (default: reporting)",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=None,
        help="sampling cores (default: min of saved chain count and available CPUs)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "override the output root (highest precedence, above "
            "DSE_LRP_OUTPUT_DIR); relative layout is unchanged"
        ),
    )
    args = parser.parse_args()
    if args.cores is not None and args.cores < 1:
        parser.error("--cores must be at least one")

    paths.set_output_root(args.output_dir)
    print(f"Output root: {paths.describe_output_root()}")
    run(args.model, config=args.config, cores=args.cores)


if __name__ == "__main__":
    freeze_support()
    main()
