# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Stress-test unresolved Byrne denominators and a denominator-free count model.

This is an empirical likelihood sensitivity, not a registered model fit.  It
retains the historical-growth mean/random-intercept structure and compares the
observed-maximum Beta-Binomial denominator with 2x and 4x stress denominators,
plus a Negative-Binomial count likelihood that uses no score ceiling.

Examples::

    python scripts/rlm_denominator_sensitivity.py --mode smoke --measure basspel
    python scripts/rlm_denominator_sensitivity.py --mode study
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from multiprocessing import freeze_support
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from language_reading_predictors.statistical_models import diagnostics as _diag
from language_reading_predictors.statistical_models import historical as _historical
from language_reading_predictors.statistical_models.datasets import resolve_dataset
from language_reading_predictors.statistical_models.preprocessing import (
    load_longitudinal_panel,
)
from language_reading_predictors.statistical_models.rlm_denominator_sensitivity import (
    aggregate_sensitivity,
    build_sensitivity_model,
    sensitivity_variants,
)
from language_reading_predictors.statistical_models.rlm_sensitivity_contract import (
    DENOMINATOR_FACTORS,
    MAX_MEDIAN_RANGE_FRACTION,
    RLM_SENSITIVITY_WINDOWS,
    UNRESOLVED_MEASURES,
)
from language_reading_predictors.statistical_models.sensitivity import sha256_file

def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "study"), default="smoke")
    parser.add_argument(
        "--measure",
        choices=("all", *UNRESOLVED_MEASURES),
        default="all",
    )
    parser.add_argument("--draws", type=int, default=None)
    parser.add_argument("--tune", type=int, default=None)
    parser.add_argument("--chains", type=int, default=None)
    parser.add_argument("--cores", type=int, default=None)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument(
        "--reuse-traces",
        action="store_true",
        help="recompute tables from the exact saved variant traces without sampling",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "output/statistical_models/sensitivity/rlm-denominator-likelihood"
        ),
    )
    return parser.parse_args()


def _posterior_predictive_diagnostics(
    trace,
    *,
    observed: np.ndarray,
    observed_maximum: int,
) -> dict[str, float]:
    draws = trace.posterior_predictive["score"].values.reshape(-1, observed.size)
    lo50, hi50 = np.quantile(draws, (0.25, 0.75), axis=0)
    lo90, hi90 = np.quantile(draws, (0.05, 0.95), axis=0)
    return {
        "ppc_50_coverage": float(np.mean((observed >= lo50) & (observed <= hi50))),
        "ppc_90_coverage": float(np.mean((observed >= lo90) & (observed <= hi90))),
        "ppc_draw_share_above_observed_maximum": float(
            np.mean(draws > observed_maximum)
        ),
        "ppc_draw_share_above_twice_observed_maximum": float(
            np.mean(draws > 2 * observed_maximum)
        ),
        "ppc_draw_q99": float(np.quantile(draws, 0.99)),
        "ppc_draw_q999": float(np.quantile(draws, 0.999)),
        "ppc_draw_maximum": float(np.max(draws)),
    }


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the requested sensitivity suite and persist its audit artefacts."""

    if args.draws is not None and args.draws < 1:
        raise ValueError("--draws must be positive")
    if args.tune is not None and args.tune < 1:
        raise ValueError("--tune must be positive")
    if args.chains is not None and args.chains < 1:
        raise ValueError("--chains must be positive")
    if args.cores is not None and args.cores < 1:
        raise ValueError("--cores must be positive")
    if not 0 < args.target_accept < 1:
        raise ValueError("--target-accept must lie strictly between zero and one")

    smoke = args.mode == "smoke"
    draws = args.draws or (300 if smoke else 1_500)
    tune = args.tune or (300 if smoke else 1_500)
    chains = args.chains or (2 if smoke else 4)
    cores = min(chains, args.cores or (os.cpu_count() or 1))
    selected = (
        UNRESOLVED_MEASURES
        if args.measure == "all"
        else (args.measure,)
    )
    dataset, measures = resolve_dataset("rlm")
    source = pd.read_csv(dataset.path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    decisions: dict[str, object] = {}
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "generated_by": "Codex/GPT-5",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "mode": args.mode,
        "measures": list(selected),
        "denominator_factors": list(DENOMINATOR_FACTORS),
        "sampling": {
            "draws": draws,
            "tune": tune,
            "chains": chains,
            "cores": cores,
            "target_accept": args.target_accept,
            "nuts_sampler": "nutpie",
            "reuse_traces": args.reuse_traces,
        },
        "decision_rule": {
            "all_variants_converged": True,
            "all_median_directions_stable": True,
            "all_joint_89_intervals_overlap": True,
            "maximum_median_range_fraction_observed_max": (
                MAX_MEDIAN_RANGE_FRACTION
            ),
        },
        "interpretation": (
            "Exploratory empirical likelihood sensitivity. Stress denominators "
            "are not instrument claims; no result confirms a ceiling or clears "
            "the existing publication gate."
        ),
        "results": {},
    }

    for measure_index, measure in enumerate(selected):
        core_waves, extension_waves = RLM_SENSITIVITY_WINDOWS[measure]
        panel = load_longitudinal_panel(
            dataset,
            [measures[measure]],
            waves=core_waves,
            complete_case=True,
            extension_waves=extension_waves,
        )
        observed = panel.long[measure].to_numpy(dtype=int)
        observed_maximum = int(
            pd.to_numeric(source[measure], errors="coerce").max()
        )
        growth_parts: list[pd.DataFrame] = []
        diagnostic_rows: list[dict[str, object]] = []
        variant_manifest: dict[str, object] = {}

        print(
            f"{measure}: n={panel.n_subjects} children, {len(observed)} rows, "
            f"observed maximum={observed_maximum}",
            flush=True,
        )
        for variant_index, variant in enumerate(
            sensitivity_variants(observed_maximum)
        ):
            seed = 20_260_816 + 100 * measure_index + variant_index
            started = time.monotonic()
            variant_dir = args.output_dir / measure / variant.name
            variant_dir.mkdir(parents=True, exist_ok=True)
            row: dict[str, object] = {
                "measure": measure,
                **variant.as_dict(),
                "sampling_seed": seed,
            }
            print(f"  fitting {variant.name}", flush=True)
            try:
                built = build_sensitivity_model(
                    panel,
                    measure=measure,
                    variant=variant,
                )
                trace_path = variant_dir / "trace.nc"
                if args.reuse_traces:
                    if not trace_path.is_file():
                        raise FileNotFoundError(
                            f"--reuse-traces requires {trace_path}"
                        )
                    trace = az.from_netcdf(trace_path)
                else:
                    with built.model:
                        prior = pm.sample_prior_predictive(
                            draws=min(1_000, draws * chains),
                            random_seed=seed + 10_000,
                        )
                        trace = pm.sample(
                            draws=draws,
                            tune=tune,
                            chains=chains,
                            cores=cores,
                            target_accept=args.target_accept,
                            nuts_sampler="nutpie",
                            random_seed=seed,
                            return_inferencedata=True,
                            progressbar=False,
                        )
                        trace = pm.sample_posterior_predictive(
                            trace,
                            var_names=["score"],
                            random_seed=seed + 20_000,
                            progressbar=False,
                            extend_inferencedata=True,
                        )
                        for group in ("prior", "prior_predictive"):
                            if group in prior.children:
                                trace[group] = prior[group]

                free_variables = [rv.name for rv in built.model.free_RVs]
                convergence = _diag.subfit_convergence(
                    trace,
                    label=f"{measure} {variant.name}",
                    var_names=free_variables,
                )
                row.update(convergence)
                row.update(
                    _posterior_predictive_diagnostics(
                        trace,
                        observed=observed,
                        observed_maximum=observed_maximum,
                    )
                )
                row["status"] = "ok"
                growth = _historical.growth_summary(
                    trace,
                    built.prepared,
                    measure,
                )
                growth.insert(0, "measure", measure)
                growth.insert(1, "variant", variant.name)
                growth.insert(2, "likelihood", variant.likelihood)
                growth.insert(3, "denominator", variant.denominator)
                growth_parts.append(growth)

                if not args.reuse_traces:
                    trace.to_netcdf(trace_path)
                row["trace_file"] = str(trace_path.relative_to(args.output_dir))
                row["trace_sha256"] = sha256_file(trace_path)
                variant_manifest[variant.name] = {
                    **variant.as_dict(),
                    "trace_file": row["trace_file"],
                    "trace_sha256": row["trace_sha256"],
                    "free_variables": free_variables,
                }
            except Exception as error:  # pragma: no cover - recorded for audit
                row.update(
                    {
                        "status": "failed",
                        "converged": False,
                        "error": f"{type(error).__name__}: {error}"[:500],
                    }
                )
                variant_manifest[variant.name] = {
                    **variant.as_dict(),
                    "error": row["error"],
                }
            row["seconds"] = round(time.monotonic() - started, 1)
            diagnostic_rows.append(row)
            print(
                f"    {row['status']}; converged={row.get('converged')}; "
                f"{row['seconds']}s",
                flush=True,
            )

        diagnostics = pd.DataFrame(diagnostic_rows)
        growth = (
            pd.concat(growth_parts, ignore_index=True)
            if growth_parts
            else pd.DataFrame()
        )
        if growth.empty:
            comparison = pd.DataFrame()
            decision: dict[str, object] = {
                "status": "no_go",
                "reason": "no sensitivity fit produced a growth table",
            }
        else:
            comparison, decision = aggregate_sensitivity(
                growth,
                diagnostics,
                observed_maximum=observed_maximum,
            )
        measure_dir = args.output_dir / measure
        diagnostics_path = measure_dir / "diagnostics.csv"
        growth_path = measure_dir / "growth.csv"
        comparison_path = measure_dir / "comparison.csv"
        _write_frame(diagnostics, diagnostics_path)
        _write_frame(growth, growth_path)
        _write_frame(comparison, comparison_path)
        decisions[measure] = decision
        manifest["results"][measure] = {
            "observed_maximum": observed_maximum,
            "n_subjects": panel.n_subjects,
            "n_rows": len(observed),
            "core_waves": list(core_waves),
            "extension_waves": list(extension_waves),
            "variants": variant_manifest,
            "tables": {
                "diagnostics": {
                    "file": str(diagnostics_path.relative_to(args.output_dir)),
                    "sha256": sha256_file(diagnostics_path),
                },
                "growth": {
                    "file": str(growth_path.relative_to(args.output_dir)),
                    "sha256": sha256_file(growth_path),
                },
                "comparison": {
                    "file": str(comparison_path.relative_to(args.output_dir)),
                    "sha256": sha256_file(comparison_path),
                },
            },
            "decision": decision,
        }
        print(json.dumps(decision, indent=2), flush=True)

    manifest["decisions"] = decisions
    (args.output_dir / "decision.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output_dir}", flush=True)
    return manifest


def main() -> None:
    run(_arguments())


if __name__ == "__main__":
    freeze_support()
    main()
