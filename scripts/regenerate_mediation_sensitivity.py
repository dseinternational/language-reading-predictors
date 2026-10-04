# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Recompute single-mediator sensitivity from a verified saved posterior.

Run with a fitted-model directory. Data, model settings, fitted rows and the
likelihood graph must match before any output is written. This calculation
changes neither the posterior nor its original fit provenance.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import arviz as az
import numpy as np
import pandas as pd

from language_reading_predictors.statistical_models.artifacts import (
    ArtifactLog,
    ArtifactRecord,
    save_table,
    write_manifest,
)
from language_reading_predictors.statistical_models.factories.mediation import build_mediation_model
from language_reading_predictors.statistical_models.key_findings import generate_key_findings
from language_reading_predictors.statistical_models.mediation import sensitivity_sweep
from language_reading_predictors.statistical_models.mediation_settings import resolve_mediation_run_plan
from language_reading_predictors.statistical_models.pipelines.mediation import prepare_mediation_data
from language_reading_predictors.statistical_models.registry import discover_models
from language_reading_predictors.statistical_models.release import evaluate_publication, write_release_decision
from language_reading_predictors.statistical_models.run_metadata import (
    _fitted_data_identity,
    _json_safe,
    _model_design_identity,
    _sha256_path,
    fitted_subject_identity,
)


def _require_matching_fit(config, plan, built):
    contract = config.get("reuse_contract") or {}
    context = SimpleNamespace(prepared=built.prepared, model=built.model)
    current = {
        "data_sha256": built.prepared.data_sha256,
        "n_obs": built.prepared.n_obs,
        "n_children": built.prepared.n_children,
        "resolved_run_plan": _json_safe(plan.as_dict()),
        "fitted_subject_identity": fitted_subject_identity(built.prepared),
        "fitted_data_identity": _fitted_data_identity(context),
        "model_design_identity": _model_design_identity(context),
    }
    mismatches = [name for name, value in current.items() if contract.get(name) != value]
    if mismatches:
        raise ValueError("saved mediation fit does not match its reconstruction: " + ", ".join(mismatches))


def _existing_log(directory):
    manifest = json.loads((directory / "artifact_manifest.json").read_text())
    log = ArtifactLog()
    for entry in manifest["artifacts"]:
        if entry.get("status") not in {"written", "skipped"}:
            continue
        log.record(
            ArtifactRecord(
                name=entry["name"],
                filename=entry["filename"],
                kind=entry["kind"],
                required=entry["required"],
                status=entry["status"],
                n_rows=entry.get("n_rows"),
                columns=tuple(entry["columns"]) if entry.get("columns") else None,
                error_type=entry.get("error_type"),
                error=entry.get("error"),
            )
        )
    return log


def regenerate(directory: Path, *, dry_run=False):
    config = json.loads((directory / "config.json").read_text())
    lazy = discover_models()[config["model_id"]]
    spec = lazy.load().SPEC
    plan = resolve_mediation_run_plan(spec)
    if plan.entrypoint != "single":
        raise ValueError("this regenerator supports single-mediator ITT fits only")
    prepared, confounders = prepare_mediation_data(spec)
    plan = plan.with_effective_confounders(confounders)
    built, med = build_mediation_model(prepared, **plan.factory_kwargs())
    _require_matching_fit(config, plan, built)
    primary = pd.read_csv(directory / "mediation_summary.csv")
    trace_path = directory / "trace.nc"
    if dry_run:
        if not trace_path.is_file():
            raise ValueError("trace.nc is missing")
        return None
    trace = az.from_netcdf(trace_path)
    sweep, summary = sensitivity_sweep(
        trace,
        med,
        ci_prob=float(config["ci_prob"]),
        interventional=plan.estimand == "interventional",
        score_mean_link=built.payload.score_mean_link,
    )
    effect = "IIE" if plan.estimand == "interventional" else "NIE"
    main = primary.set_index("quantity").loc[effect]
    zero = sweep.set_index("delta").loc[0]
    if not np.allclose(
        main[["prob_median", "prob_lo", "prob_hi", "prob_pos"]].to_numpy(dtype=float),
        zero[["nie_median", "nie_lo", "nie_hi", "nie_prob_pos"]].to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("recomputed zero-bias sensitivity does not match the saved primary effect")
    ctx = SimpleNamespace(output_dir=str(directory), tables={}, artifacts=_existing_log(directory), spec=spec)
    save_table(ctx, "mediation_sensitivity", sweep)
    save_table(ctx, "mediation_sensitivity_summary", pd.DataFrame([summary]), register=False)
    record = {
        "note": "Regenerated by an LLM-based AI tool (Codex/GPT-6).",
        "regenerated_at": datetime.now(timezone.utc).isoformat(),
        "score_mean_link": built.payload.score_mean_link,
        "trace_sha256": _sha256_path(trace_path),
        "regenerator_sha256": _sha256_path(__file__),
        "verified_contract_fields": [
            "data_sha256",
            "n_obs",
            "n_children",
            "resolved_run_plan",
            "fitted_subject_identity",
            "fitted_data_identity",
            "model_design_identity",
        ],
        "zero_bias_matches_primary": True,
    }
    (directory / "mediation_sensitivity_regeneration.json").write_text(json.dumps(record, indent=2) + "\n")
    decision = evaluate_publication(directory)
    write_release_decision(ctx, decision)
    generate_key_findings(directory, decision=decision)
    write_manifest(ctx)
    return sweep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit_directory", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = regenerate(args.fit_directory, dry_run=args.dry_run)
    print("Verified saved fit." if result is None else f"Regenerated {len(result)} sensitivity rows.")


if __name__ == "__main__":
    main()
