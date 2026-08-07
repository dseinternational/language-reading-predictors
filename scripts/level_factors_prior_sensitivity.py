# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Treatment-prior sensitivity sweep for the level-factor family (#389 criterion 6).

The robustness release gate (#482) covers ``level_factors`` on the randomised t2
contrast ``b_grp_time[1]``, and for a prior-dominant fit it names its release
evidence exactly: *"a tau_prior_sensitivity.csv treatment-prior sweep, computed
from this fit's own trace, showing the sign of the effect is stable across the
grid"* (``release._standard_sweep_evidence``). No non-ITT runner produced that
artefact until now. This script is the level-family runner:

- For each requested outcome (default: the five the #389 review names — W, L, P,
  B, N, all proximal-tier) it resolves the registered primary model's **typed run
  plan**, rebuilds the model in-process with the focal contrast's prior moved
  across the proximal grid (0.25 / 0.5 / 0.75; the registered scale included),
  holding everything else — data, adjustment set, anchored intercepts (#389
  finding 2), likelihood — at the registered specification.
- Each cell is gated on the full convergence criteria over all free variables
  (R-hat <= 1.01, ESS >= 400, BFMI >= 0.3, zero divergences); an unconverged cell
  is not evidence and blocks that outcome's report-local copy.
- Rows carry the standard sweep's full column set, the primary fit's
  ``config.json`` / ``trace.nc`` sha256 bindings, and a content-addressed cell
  trace, so the gate's evidence check can verify the sweep belongs to exactly the
  primary it sits beside. A row's ``gamma_own_sigma`` is NaN — a levels model has
  no own-baseline term — and the off-floor outcomes (P, N) report the risk
  difference through ``n_trials = 1``.
- With ``--attach`` (reporting config only) the per-outcome rows are written to
  ``<fit-dir>/tau_prior_sensitivity.csv`` beside each primary whose cells all
  converged.

Usage:
    python scripts/level_factors_prior_sensitivity.py --config reporting --attach
    python scripts/level_factors_prior_sensitivity.py --config reporting \
        --outcomes W P --attach
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm

import dse_research_utils.statistics.models.sampling as _sampling
from language_reading_predictors import paths as _paths
from language_reading_predictors.statistical_models.factories import (
    build_level_factors_model,
)
from language_reading_predictors.statistical_models.measures import MEASURES
from language_reading_predictors.statistical_models.sensitivity import (
    LEVEL_SENSITIVITY_MODEL_IDS,
    LEVEL_SENSITIVITY_OUTCOMES,
    STANDARD_SENSITIVITY_FILENAME,
    STANDARD_SENSITIVITY_PROVENANCE_ATTR,
    STANDARD_SENSITIVITY_PROXIMAL_TAU_SIGMAS,
    STANDARD_SENSITIVITY_SAMPLING_ATTR,
    PrimaryStandardReference,
    load_primary_level_reference,
    sha256_file,
)

TAU_SIGMAS = STANDARD_SENSITIVITY_PROXIMAL_TAU_SIGMAS
SENSITIVITY_AXIS = "tau"
KAPPA_SIGMA = 50.0


def _resolve_plan(outcome: str):
    """The registered primary's typed run plan (single source of truth)."""
    import importlib

    from language_reading_predictors.statistical_models.level_factors import (
        resolve_level_factors_run_plan,
    )

    model_id = LEVEL_SENSITIVITY_MODEL_IDS[outcome]
    module = importlib.import_module(
        "language_reading_predictors.statistical_models."
        + model_id.replace("-", "_")
    )
    return resolve_level_factors_run_plan(module.SPEC)


def _persist_trace(trace, *, sensitivity_dir: Path, semantic_file: Path) -> tuple[Path, str]:
    """Atomically install an immutable cell trace named by its content digest."""
    trace_dir = sensitivity_dir / semantic_file.parent
    trace_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{semantic_file.stem}-", suffix=".nc", dir=trace_dir
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        trace.to_netcdf(temporary)
        digest = sha256_file(temporary)
        destination = trace_dir / f"{semantic_file.stem}-{digest[:12]}.nc"
        if destination.exists():
            if sha256_file(destination) != digest:
                raise RuntimeError(
                    f"level sensitivity trace digest-prefix collision: {destination}"
                )
            temporary.unlink()
        else:
            os.replace(temporary, destination)
        return destination.relative_to(sensitivity_dir), digest
    finally:
        if temporary.exists():
            temporary.unlink()


def _fit_cell(
    outcome: str,
    tau_sigma: float,
    *,
    config: str,
    sampling,
    sensitivity_dir: Path,
    primary_reference: PrimaryStandardReference,
) -> dict:
    """Fit one level-factor sweep cell and return its standard-schema row."""
    from language_reading_predictors.statistical_models import diagnostics as _diag
    from language_reading_predictors.statistical_models.preprocessing import (
        load_and_prepare,
    )
    from language_reading_predictors.statistical_models.reporting import (
        REPORTING_CI_PROB,
        level_t2_marginal_effect,
    )

    plan = _resolve_plan(outcome)
    prepared = load_and_prepare(**plan.prepare_kwargs())
    plan.validate_prepared(prepared)
    effective = tuple(c for c in plan.adjust_for if c in prepared.covariates)
    built = build_level_factors_model(
        prepared,
        **plan.factory_kwargs(effective_adjustment=effective),
        tau_prior_sigma=tau_sigma,
    )
    with built.model:
        trace = pm.sample(
            draws=sampling.draws,
            tune=sampling.tune,
            chains=sampling.chains,
            cores=sampling.cores,
            target_accept=sampling.target_accept,
            nuts_sampler="nutpie",
            return_inferencedata=True,
            random_seed=sampling.random_seed,
            progressbar=False,
        )

    ability = (
        built.prepared.covariates[plan.ability_covariate]
        if plan.ability_covariate is not None
        else None
    )
    contrast_draws, ame_prob = level_t2_marginal_effect(
        trace,
        phase=built.prepared.phase,
        G=built.prepared.G,
        ability=ability,
    )
    lo_q = (1.0 - REPORTING_CI_PROB) / 2.0
    n_trials = 1 if plan.off_floor else int(MEASURES[outcome].n_trials)
    kappa_draws = (
        trace.posterior["kappa"].stack(sample=("chain", "draw")).values
        if "kappa" in trace.posterior
        else np.array([np.nan])
    )
    free_names = [rv.name for rv in built.model.free_RVs]
    convergence = _diag.subfit_convergence(
        trace, label=f"{outcome} {SENSITIVITY_AXIS}", var_names=free_names
    )

    # Arm counts over all fitted rows, matching load_primary_level_reference's
    # definition so the binding check is like-for-like; the t2 per-arm split (the
    # rows the randomised contrast is actually estimated from) rides along as
    # informative extra columns.
    G = np.asarray(built.prepared.G)
    phase = np.asarray(built.prepared.phase)
    n = int(built.prepared.n_obs)
    n_intervention = int(np.sum(G == 1))
    n_control = int(np.sum(G == 0))
    data_sha256 = str(getattr(built.prepared, "data_sha256", ""))
    if (
        primary_reference.config_name != config
        or primary_reference.outcome != outcome
        or data_sha256 != primary_reference.data_sha256
        or n != primary_reference.n
        or n_intervention != primary_reference.n_intervention
        or n_control != primary_reference.n_control
    ):
        raise RuntimeError(
            f"{outcome} level sensitivity does not match its current primary "
            "level-factor data, sample, arm counts, or config"
        )

    row = {
        **primary_reference.manifest_values(),
        "config": config,
        "outcome": outcome,
        "n_trials": n_trials,
        "sensitivity_axis": SENSITIVITY_AXIS,
        "tau_sigma": tau_sigma,
        # A levels model has no own-baseline term; the column exists so the row
        # carries the standard sweep's full schema.
        "gamma_own_sigma": np.nan,
        "kappa_sigma": np.nan if plan.off_floor else KAPPA_SIGMA,
        "use_precision_terms": True,  # the linear-age precision term is always built
        "data_sha256": data_sha256,
        "n": n,
        "n_intervention": n_intervention,
        "n_control": n_control,
        "n_t2_intervention": int(np.sum(G[phase == 1] == 1)),
        "n_t2_control": int(np.sum(G[phase == 1] == 0)),
        "pd": float(np.mean(ame_prob > 0)),
        "tau_logit_mean": float(np.mean(contrast_draws)),
        "tau_logit_lo": float(np.quantile(contrast_draws, lo_q)),
        "tau_logit_hi": float(np.quantile(contrast_draws, 1.0 - lo_q)),
        "ci_width_logit": float(
            np.quantile(contrast_draws, 1.0 - lo_q) - np.quantile(contrast_draws, lo_q)
        ),
        "tau_sd_logit": float(np.std(contrast_draws)),
        "kappa_median": float(np.nanmedian(kappa_draws)),
        "items_mean": float(np.mean(ame_prob)) * n_trials,
        "items_lo": float(np.quantile(ame_prob, lo_q)) * n_trials,
        "items_hi": float(np.quantile(ame_prob, 1.0 - lo_q)) * n_trials,
        "converged": convergence["converged"],
        "max_rhat": convergence["max_rhat"],
        "min_ess": convergence["min_ess"],
        "min_bfmi": convergence["min_bfmi"],
        "n_divergences": convergence["n_divergences"],
        "free_variables": "|".join(free_names),
        "n_free_variables": len(free_names),
        "convergence_scope": "all_free_variables",
        "sampling_draws": sampling.draws,
        "sampling_tune": sampling.tune,
        "sampling_chains": sampling.chains,
        "sampling_cores": sampling.cores,
        "sampling_target_accept": sampling.target_accept,
        "sampling_random_seed": sampling.random_seed,
        "sampling_nuts_sampler": "nutpie",
    }
    # Level-family provenance stamped on the cell trace. Deliberately NOT the ITT
    # ``standard_trace_provenance`` — that validator asserts the ITT free-variable
    # layout and would mislabel this family — but the same identity content.
    provenance = {
        "schema_version": 1,
        "model_kind": "level_factors",
        "config": config,
        "outcome": outcome,
        "focal_term": "b_grp_time[1]",
        "sensitivity_axis": SENSITIVITY_AXIS,
        "tau_sigma": tau_sigma,
        "likelihood": plan.likelihood,
        "n_trials": n_trials,
        "data_sha256": data_sha256,
        "n": n,
        "primary_model_id": str(row["primary_model_id"]),
        "primary_config_sha256": str(row["primary_config_sha256"]),
        "primary_trace_sha256": str(row["primary_trace_sha256"]),
        "free_variables": free_names,
        "sampling": {
            "draws": sampling.draws,
            "tune": sampling.tune,
            "chains": sampling.chains,
            "cores": sampling.cores,
            "target_accept": sampling.target_accept,
            "random_seed": sampling.random_seed,
            "nuts_sampler": "nutpie",
        },
    }
    trace.posterior.attrs[STANDARD_SENSITIVITY_SAMPLING_ATTR] = json.dumps(
        provenance["sampling"], sort_keys=True, separators=(",", ":")
    )
    trace.posterior.attrs[STANDARD_SENSITIVITY_PROVENANCE_ATTR] = json.dumps(
        provenance, sort_keys=True, separators=(",", ":")
    )
    sigma_token = f"{tau_sigma:g}".replace(".", "p")
    semantic = (
        Path("traces")
        / f"level-{config}"
        / f"trace_{outcome}_tau-{sigma_token}.nc"
    )
    trace_file, trace_sha256 = _persist_trace(
        trace, sensitivity_dir=sensitivity_dir, semantic_file=semantic
    )
    row.update(trace_file=trace_file.as_posix(), trace_sha256=trace_sha256)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="dev", help="sampling preset (dev/test/reporting)")
    ap.add_argument(
        "--outcomes", nargs="+", default=list(LEVEL_SENSITIVITY_OUTCOMES),
        help="level outcomes to sweep (default: the #389 review's W L P B N)",
    )
    ap.add_argument(
        "--attach", action="store_true",
        help=(
            "write each outcome's rows to <fit-dir>/tau_prior_sensitivity.csv "
            "beside its primary (only when every cell converged)"
        ),
    )
    ap.add_argument(
        "--output-dir", type=str, default=None,
        help="override the output root (above DSE_LRP_OUTPUT_DIR); layout unchanged",
    )
    args = ap.parse_args()

    unknown = sorted(set(args.outcomes) - set(LEVEL_SENSITIVITY_MODEL_IDS))
    if unknown:
        ap.error(
            f"unsupported level outcomes: {unknown}; choose from "
            f"{sorted(LEVEL_SENSITIVITY_MODEL_IDS)}"
        )

    _paths.set_output_root(args.output_dir)
    print(f"Output root: {_paths.describe_output_root()}")
    models_root = Path(_paths.stat_dir()) / "models"
    sensitivity_dir = Path(_paths.stat_dir()) / "level_tau_prior_sensitivity"
    sensitivity_dir.mkdir(parents=True, exist_ok=True)

    sampling = _sampling.get_sampling_configuration(args.config, random_seed=20260701)

    rows: list[dict] = []
    attach_ready: dict[str, bool] = {}
    for outcome in args.outcomes:
        model_id = LEVEL_SENSITIVITY_MODEL_IDS[outcome]
        primary_dir = models_root / f"{model_id}-{args.config}"
        reference = load_primary_level_reference(
            primary_dir, outcome, config_name=args.config
        )
        outcome_rows = []
        for tau_sigma in TAU_SIGMAS:
            print(f"--- {outcome} ({model_id}): tau_sigma={tau_sigma} ---")
            row = _fit_cell(
                outcome,
                float(tau_sigma),
                config=args.config,
                sampling=sampling,
                sensitivity_dir=sensitivity_dir,
                primary_reference=reference,
            )
            print(
                f"    tau_logit_mean={row['tau_logit_mean']:+.3f} "
                f"[{row['tau_logit_lo']:+.3f}, {row['tau_logit_hi']:+.3f}] "
                f"converged={row['converged']}"
            )
            outcome_rows.append(row)
        rows.extend(outcome_rows)
        all_ok = all(bool(r["converged"]) for r in outcome_rows)
        attach_ready[outcome] = all_ok
        if args.attach:
            if not all_ok:
                print(
                    f"    NOT attaching {outcome}: one or more cells failed the "
                    "convergence gate"
                )
                continue
            frame = pd.DataFrame(outcome_rows)
            destination = primary_dir / STANDARD_SENSITIVITY_FILENAME
            frame.to_csv(destination, index=False)
            print(f"    attached {destination}")

    combined = pd.DataFrame(rows)
    combined_path = sensitivity_dir / f"level_{STANDARD_SENSITIVITY_FILENAME}"
    combined.to_csv(combined_path, index=False)
    print(f"\nWrote {combined_path} ({len(combined)} rows)")
    for outcome, ok in attach_ready.items():
        state = "all cells converged" if ok else "HAS UNCONVERGED CELLS"
        print(f"  {outcome}: {state}")


if __name__ == "__main__":
    main()
