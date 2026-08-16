# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run the participant Bayesian-bootstrap robustness check for Byrne growth.

The analysis uses paired raw-score changes and therefore needs neither a score
denominator nor a count likelihood. It is intentionally a non-registered
Phase-A sensitivity and remains subordinate to the instrument documentation
gate.

Examples::

    python scripts/rlm_growth_bootstrap.py --mode smoke --measure basspel
    python scripts/rlm_growth_bootstrap.py --mode study
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from language_reading_predictors.statistical_models.datasets import resolve_dataset
from language_reading_predictors.statistical_models.preprocessing import (
    load_longitudinal_panel,
)
from language_reading_predictors.statistical_models.rlm_sensitivity_contract import (
    MAX_MEDIAN_RANGE_FRACTION,
    RLM_SENSITIVITY_WINDOWS,
    UNRESOLVED_MEASURES,
)
from language_reading_predictors.statistical_models.rlm_growth_bootstrap import (
    MONTE_CARLO_TOLERANCE_FRACTION,
    compare_bootstrap_with_likelihoods,
    monte_carlo_stability,
    participant_bayesian_bootstrap_growth,
)
from language_reading_predictors.statistical_models.sensitivity import sha256_file

DEFAULT_SEED = 26_081_600


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "study"), default="smoke")
    parser.add_argument(
        "--measure",
        choices=("all", *UNRESOLVED_MEASURES),
        default="all",
    )
    parser.add_argument("--draws", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path(
            "output/statistical_models/sensitivity/rlm-denominator-likelihood"
        ),
        help="trace-backed output from scripts/rlm_denominator_sensitivity.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "output/statistical_models/sensitivity/rlm-growth-bootstrap"
        ),
    )
    return parser.parse_args()


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _load_reference(
    root: Path,
    *,
    measure: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    decision_path = root / "decision.json"
    if not decision_path.is_file():
        raise FileNotFoundError(
            f"Missing {decision_path}; run the full denominator sensitivity first"
        )
    manifest = json.loads(decision_path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("mode") != "study":
        raise ValueError("likelihood reference must be a schema-1 full study")
    try:
        result = manifest["results"][measure]
        variants = result["variants"]
        growth_record = result["tables"]["growth"]
        reference_passed = result["decision"]["status"] == "pass"
    except KeyError as error:
        raise ValueError(
            f"likelihood reference lacks complete {measure!r} metadata"
        ) from error
    if not variants:
        raise ValueError(f"likelihood reference has no {measure!r} variants")

    trace_records: dict[str, dict[str, str]] = {}
    for variant, record in variants.items():
        trace_path = root / record["trace_file"]
        expected_hash = record["trace_sha256"]
        if not trace_path.is_file():
            raise FileNotFoundError(f"likelihood reference trace missing: {trace_path}")
        actual_hash = sha256_file(trace_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"likelihood reference trace hash mismatch for {variant!r}"
            )
        trace_records[variant] = {
            "trace_file": str(trace_path),
            "trace_sha256": actual_hash,
        }

    growth_path = root / growth_record["file"]
    if not growth_path.is_file():
        raise FileNotFoundError(f"likelihood reference table missing: {growth_path}")
    growth_hash = sha256_file(growth_path)
    if growth_hash != growth_record["sha256"]:
        raise ValueError(f"likelihood growth hash mismatch for {measure!r}")
    growth = pd.read_csv(growth_path)
    expected_variants = tuple(variants)
    if set(growth["variant"].astype(str)) != set(expected_variants):
        raise ValueError(
            f"likelihood growth variants do not match manifest for {measure!r}"
        )
    return growth, {
        "decision_file": str(decision_path),
        "decision_sha256": sha256_file(decision_path),
        "growth_file": str(growth_path),
        "growth_sha256": growth_hash,
        "reference_passed": reference_passed,
        "expected_variants": list(expected_variants),
        "traces": trace_records,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run and persist the requested denominator-free sensitivity."""

    if args.draws is not None and args.draws < 1:
        raise ValueError("--draws must be positive")
    if args.seed < 1:
        raise ValueError("--seed must be positive")
    draws = args.draws or (10_000 if args.mode == "smoke" else 200_000)
    selected = (
        UNRESOLVED_MEASURES
        if args.measure == "all"
        else (args.measure,)
    )
    dataset, measures = resolve_dataset("rlm")
    source = pd.read_csv(dataset.path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "generated_by": "Codex/GPT-5",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "mode": args.mode,
        "measures": list(selected),
        "bootstrap": {
            "method": "participant Bayesian bootstrap",
            "weight_prior": "independent Dirichlet(1, ..., 1) within group",
            "draws_per_simulation": draws,
            "primary_seed": args.seed,
            "replicate_seed_offset": 1_000,
            "paired_endpoint_rule": (
                "renormalise shared participant weights over children observed "
                "at both endpoints"
            ),
        },
        "source": {
            "path": str(Path(dataset.path).resolve()),
            "sha256": sha256_file(Path(dataset.path)),
        },
        "decision_rule": {
            "likelihood_reference_passed": True,
            "independent_simulation_maximum_quantile_difference_fraction": (
                MONTE_CARLO_TOLERANCE_FRACTION
            ),
            "all_five_method_median_directions_stable": True,
            "all_five_method_89_intervals_overlap": True,
            "maximum_five_method_median_range_fraction_observed_max": (
                MAX_MEDIAN_RANGE_FRACTION
            ),
        },
        "interpretation": (
            "Empirical Phase-A raw-growth robustness only. No result identifies "
            "a score ceiling, validates a score definition, repairs another "
            "model family, or clears the publication gate."
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
        observed_maximum = int(
            pd.to_numeric(source[measure], errors="coerce").max()
        )
        likelihood_growth, reference = _load_reference(
            args.reference_dir,
            measure=measure,
        )
        seed = args.seed + 100 * measure_index
        replicate_seed = seed + 1_000
        print(
            f"{measure}: {panel.n_subjects} children; {draws:,} draws x 2",
            flush=True,
        )
        primary = participant_bayesian_bootstrap_growth(
            panel,
            measure=measure,
            draws=draws,
            seed=seed,
        )
        replicate = participant_bayesian_bootstrap_growth(
            panel,
            measure=measure,
            draws=draws,
            seed=replicate_seed,
        )
        monte_carlo, monte_carlo_decision = monte_carlo_stability(
            primary,
            replicate,
            observed_maximum=observed_maximum,
        )
        comparison, decision = compare_bootstrap_with_likelihoods(
            primary,
            likelihood_growth,
            observed_maximum=observed_maximum,
            expected_likelihood_variants=tuple(reference["expected_variants"]),
            likelihood_reference_passed=bool(reference["reference_passed"]),
            monte_carlo_passed=monte_carlo_decision["status"] == "pass",
        )

        measure_dir = args.output_dir / measure
        _write_frame(primary, measure_dir / "bootstrap_growth.csv")
        _write_frame(monte_carlo, measure_dir / "monte_carlo_stability.csv")
        _write_frame(comparison, measure_dir / "likelihood_comparison.csv")
        manifest["results"][measure] = {
            "observed_maximum_operational_only": observed_maximum,
            "n_subjects": panel.n_subjects,
            "n_rows": panel.n_obs,
            "core_waves": list(core_waves),
            "extension_waves": list(extension_waves),
            "primary_seed": seed,
            "replicate_seed": replicate_seed,
            "reference": reference,
            "monte_carlo_decision": monte_carlo_decision,
            "decision": decision,
        }
        print(
            f"  Monte Carlo {monte_carlo_decision['status']}; "
            f"five-method comparison {decision['status']}",
            flush=True,
        )

    statuses = [
        result["decision"]["status"] for result in manifest["results"].values()
    ]
    manifest["overall_status"] = (
        "pass" if statuses and all(status == "pass" for status in statuses) else "no_go"
    )
    output_path = args.output_dir / "decision.json"
    output_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {output_path}; overall={manifest['overall_status']}", flush=True)
    return manifest


def main() -> None:
    run(_arguments())


if __name__ == "__main__":
    main()
