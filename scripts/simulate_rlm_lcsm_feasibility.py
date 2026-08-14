# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run the pre-fit Byrne/RLM latent change-score recovery study.

Examples
--------
Smoke-test one dataset per candidate::

    python scripts/simulate_rlm_lcsm_feasibility.py --mode smoke

Run the pre-specified study and persist its audit tables::

    python scripts/simulate_rlm_lcsm_feasibility.py --mode study --n-sims 40
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm

from language_reading_predictors.statistical_models.rlm_lcsm_feasibility import (
    FeasibilityCriteria,
    aggregate_recovery,
    build_rlm_lcsm_recovery_model,
    evaluate_candidate,
    load_rlm_feasibility_design,
    recovery_rows,
    simulate_rlm_lcsm_counts,
    simulation_truth,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "study"), default="smoke")
    parser.add_argument("--n-sims", type=int, default=40)
    parser.add_argument("--draws", type=int, default=None)
    parser.add_argument("--tune", type=int, default=None)
    parser.add_argument("--chains", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/statistical_models/feasibility/rlm-lcsm"),
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if args.n_sims < 1:
        raise ValueError("--n-sims must be positive")
    smoke = args.mode == "smoke"
    n_sims = 1 if smoke else args.n_sims
    draws = args.draws or (300 if smoke else 500)
    tune = args.tune or (300 if smoke else 600)
    chains = args.chains or (2 if smoke else 4)
    criteria = FeasibilityCriteria()
    scopes = ("ds", "three_group")
    strengths = (criteria.alternative_strength,) if smoke else (
        0.0,
        criteria.alternative_strength,
    )

    replicate_rows: list[dict[str, object]] = []
    attempts: list[dict[str, object]] = []
    designs: dict[str, object] = {}
    for scope_index, scope in enumerate(scopes):
        design = load_rlm_feasibility_design(scope)
        designs[scope] = design.metadata()
        print(
            f"{scope}: N={design.n_children}, groups={design.group_labels}, "
            f"waves={design.waves.tolist()}, observed={int(design.mask.sum())}",
            flush=True,
        )
        for strength_index, strength in enumerate(strengths):
            truth = simulation_truth(design, reverse_strength=strength)
            for simulation in range(n_sims):
                seed = 10_000 * scope_index + 1_000 * strength_index + simulation + 1
                attempts.append(
                    {
                        "scope": scope,
                        "reverse_strength": strength,
                        "simulation": simulation,
                        "seed": seed,
                    }
                )
                started = time.monotonic()
                try:
                    counts, _ = simulate_rlm_lcsm_counts(
                        design,
                        truth,
                        np.random.default_rng(seed),
                    )
                    model = build_rlm_lcsm_recovery_model(design, counts)
                    with model:
                        idata = pm.sample(
                            draws=draws,
                            tune=tune,
                            chains=chains,
                            cores=chains,
                            nuts_sampler="nutpie",
                            target_accept=0.95,
                            random_seed=50_000 + seed,
                            progressbar=False,
                            compute_convergence_checks=False,
                        )
                    replicate_rows.extend(
                        recovery_rows(
                            idata,
                            scope=scope,
                            simulation=simulation,
                            truth=truth,
                            support_threshold=criteria.posterior_support_threshold,
                        )
                    )
                    status = "ok"
                except Exception as error:  # pragma: no cover - surfaced in artefact
                    status = f"failed: {type(error).__name__}: {error}"[:240]
                attempts[-1]["status"] = status
                attempts[-1]["seconds"] = round(time.monotonic() - started, 1)
                print(
                    f"  strength={strength:.2f} sim={simulation + 1}/{n_sims} "
                    f"{status} ({attempts[-1]['seconds']}s)",
                    flush=True,
                )

    rows = pd.DataFrame(replicate_rows)
    attempted = pd.DataFrame(attempts)
    if rows.empty:
        raise RuntimeError("all feasibility fits failed")
    summary = aggregate_recovery(rows, attempted=attempted)
    decisions = {
        scope: evaluate_candidate(summary, scope, criteria=criteria)
        for scope in scopes
        if not smoke
    }
    print("\nRecovery summary", flush=True)
    print(summary.to_string(index=False), flush=True)
    if decisions:
        print("\nDecisions", flush=True)
        print(json.dumps(decisions, indent=2), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output_dir / "replicates.csv", index=False)
    attempted.to_csv(args.output_dir / "attempts.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    metadata = {
        "generated_by": "Codex/GPT-5",
        "mode": args.mode,
        "n_sims": n_sims,
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "designs": designs,
        "criteria": asdict(criteria),
        "decisions": decisions,
    }
    (args.output_dir / "decision.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"\nWrote {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
