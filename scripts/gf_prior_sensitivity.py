# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Treatment-prior sensitivity sweep for the gain-factor family (#391).

The robustness release gate (#482) covers ``gain_factors`` on ``beta_trt``
(``release.causal_term_for``), and for a prior-dominant fit it names its release
evidence exactly: *"a tau_prior_sensitivity.csv treatment-prior sweep, computed
from this fit's own trace, showing the sign of the effect is stable across the
grid"* (``release._standard_sweep_evidence``). The two off-floor primaries
(gf-005 P, gf-011 N) shipped ``robustness_unresolved`` because no gain-factor
runner produced that artefact; after the #391 findings 2+3 respec their refits
need it under the new specification. This script is the gain-factor family
runner, sharing the level/did runners' attach discipline via :mod:`sensitivity`:

- For each requested model (default: the two historically prior-dominant
  off-floor fits; pass ``--models`` for any other fit a refit's power-scaling
  flags) it resolves the registered primary's **typed run plan** and rebuilds
  the model in-process with the ``beta_trt`` prior moved across its outcome
  tier's grid (proximal 0.25 / 0.5 / 0.75; distal 0.2 / 0.25 / 0.3 / 0.5),
  holding everything else — data, adjustment set, likelihood, the off-floor
  pre indicator — at the registered specification. Like the did sweep the set
  is keyed by **model id**: the taught-vocabulary outcomes each have two
  registered primaries.
- Cells reproduce the primary's own sampling contract: the preset must match
  on draws/tune/chains, and the primary's recorded ``target_accept`` is
  adopted for every cell (with ``--cell-target-accept`` able only to RAISE it).
- Each cell is gated on the full convergence criteria over all free variables
  (R-hat <= 1.01, ESS >= 400, BFMI >= 0.3, zero divergences); an unconverged
  cell is not evidence and blocks that model's report-local copy.
- Rows carry the standard sweep's full column set, the primary fit's
  ``config.json`` / ``trace.nc`` sha256 bindings, and a content-addressed cell
  trace. Items columns hold the family's own headline translation — the
  period-1 average marginal effect of the on-intervention toggle (a risk
  difference for the off-floor fits, ``n_trials`` items otherwise). The
  headline specs are interaction-free (#391 finding 3), so no moderator
  netting arises.
- With ``--attach`` the per-model rows are written to
  ``<fit-dir>/tau_prior_sensitivity.csv`` beside each primary whose cells all
  converged (trace-backed, digest-verified, atomic, rolled back on failure).

Usage:
    python scripts/gf_prior_sensitivity.py --config reporting --attach
    python scripts/gf_prior_sensitivity.py --config reporting \
        --models lrp-rli-gf-005 --attach
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
from scipy.special import expit

import dse_research_utils.statistics.models.sampling as _sampling
from language_reading_predictors import paths as _paths
from language_reading_predictors.statistical_models.factories import (
    build_gain_factors_model,
)
from language_reading_predictors.statistical_models.measures import (
    DISTAL_OUTCOMES,
    MEASURES,
)
from language_reading_predictors.statistical_models.sensitivity import (
    GF_SENSITIVITY_MODEL_IDS,
    STANDARD_SENSITIVITY_DISTAL_TAU_SIGMAS,
    STANDARD_SENSITIVITY_FILENAME,
    STANDARD_SENSITIVITY_PROVENANCE_ATTR,
    STANDARD_SENSITIVITY_PROXIMAL_TAU_SIGMAS,
    STANDARD_SENSITIVITY_SAMPLING_ATTR,
    PrimaryStandardReference,
    assert_primary_sampling_contract,
    attach_outcome_bundle,
    load_primary_gf_reference,
    persist_sensitivity_trace,
)

KAPPA_SIGMA = 50.0
# Unswept nuisance-prior scales, recorded per row: the graded fits keep the
# shared gamma_own default; the off-floor fits carry the binary
# off-floor-at-pre indicator's Normal(0, 1) (#391 finding 2 decision).
GAMMA_OWN_SIGMA_GRADED = 0.25
GAMMA_OWN_SIGMA_OFFFLOOR = 1.0

#: The kappa-axis cells (#575 finding 10a), indexed by the integer passed as the
#: cell's ``sigma``: the registered concentration prior refitted for a
#: like-for-like anchor, then the near-Binomial-capable dispersion-scale
#: parameterisation at the two documented widths (the dispersion sweep's idiom).
KAPPA_AXIS_CELLS: dict[int, tuple[str, float | None]] = {
    0: ("halfnormal_concentration", None),
    1: ("halfnormal_inverse_sqrt", 0.25),
    2: ("halfnormal_inverse_sqrt", 0.5),
}
GAMMA_OWN_AXIS_SIGMAS = (0.25, 0.5)

# Default sweep set: the two fits whose pre-respec release decisions were
# prior-dominant withholds. Which fits NEED evidence is each refit's own
# power-scaling diagnosis — pass --models for any other flagged primary.
GF_SENSITIVITY_DEFAULT_MODEL_IDS = ("lrp-rli-gf-005", "lrp-rli-gf-011")


def assert_gf_sampling_contract(
    sampling, reference: PrimaryStandardReference, *, config: str
) -> None:
    """The gain-factor variant of the shared contract: match the preset on
    draws/tune/chains and *adopt* the primary's recorded ``target_accept``
    (mirroring the did runner — a registered spec may override the preset)."""
    assert_primary_sampling_contract(
        sampling,
        reference,
        config=config,
        keys=("draws", "tune", "chains"),
        label=f"{reference.model_id} gain-factor",
    )


def _resolve_plan(model_id: str):
    """The registered primary's typed run plan (single source of truth)."""
    import importlib

    from language_reading_predictors.statistical_models.gain_factors import (
        resolve_gain_factors_run_plan,
    )

    module = importlib.import_module(
        "language_reading_predictors.statistical_models."
        + model_id.replace("-", "_")
    )
    return resolve_gain_factors_run_plan(module.SPEC)


def _grid_for(plan) -> tuple[float, ...]:
    """The beta_trt prior grid for the outcome's tau tier."""
    if plan.outcome_symbol in DISTAL_OUTCOMES:
        return STANDARD_SENSITIVITY_DISTAL_TAU_SIGMAS
    return STANDARD_SENSITIVITY_PROXIMAL_TAU_SIGMAS


def _items_translation(
    trace, built, plan, *, n_trials: int, ci_prob: float
) -> tuple[float, float, float]:
    """(items_mean, items_lo, items_hi): the period-1 AME of the trt toggle.

    The family's own headline translation (``treatment_marginal_effect`` with
    ``row_mask`` = the period-1 rows): per draw, remove the fitted on-intervention
    contribution from each period-1 row's linear predictor, then contrast the
    toggle. The headline specs carry no treatment interactions (#391 finding 3),
    so the netting reduces to ``beta_trt`` alone.
    """
    lo_q = (1.0 - ci_prob) / 2.0
    posterior = trace.posterior
    eta = (
        posterior["eta"]
        .stack(sample=("chain", "draw"))
        .transpose("obs_id", "sample")
        .values
    )
    beta = posterior["beta_trt"].stack(sample=("chain", "draw")).values.ravel()
    trt = np.asarray(
        ((built.prepared.G == 1) | (built.prepared.phase >= 1)), dtype=float
    )
    p1 = np.asarray(built.prepared.phase) == 0
    eta_off = eta[p1] - np.outer(trt[p1], beta)
    delta = expit(eta_off + beta[None, :]) - expit(eta_off)
    items = delta.mean(axis=0) * float(n_trials)
    return (
        float(np.mean(items)),
        float(np.quantile(items, lo_q)),
        float(np.quantile(items, 1.0 - lo_q)),
    )


def _fit_cell(
    model_id: str,
    sigma: float,
    *,
    config: str,
    sampling,
    sensitivity_dir: Path,
    primary_reference: PrimaryStandardReference,
    cell_target_accept: float | None = None,
    axis: str = "tau",
) -> dict:
    """Fit one gain-factor sweep cell and return its standard-schema row.

    ``axis`` selects what the cell varies (#575 finding 10): ``"tau"`` (the
    standard beta_trt prior grid, the release-evidence artefact), ``"kappa"``
    (``sigma`` indexes :data:`KAPPA_AXIS_CELLS`) or ``"gamma_own"`` (``sigma``
    is the Normal(1, s) width). Non-tau axes write to their own sweep
    directories and never enter the standard-schema release artefact.
    """
    from language_reading_predictors.statistical_models import diagnostics as _diag
    from language_reading_predictors.statistical_models.preprocessing import (
        load_and_prepare,
    )
    from language_reading_predictors.statistical_models.reporting import (
        REPORTING_CI_PROB,
    )

    plan = _resolve_plan(model_id)
    outcome = plan.outcome_symbol
    prepared = load_and_prepare(**plan.prepare_kwargs())
    adjust_for = tuple(c for c in plan.adjust_for if c in prepared.covariates)
    kwargs = plan.factory_kwargs(effective_adjustment=adjust_for)
    # Axis dispatch (#575 finding 10): the tau axis moves the beta_trt prior
    # scale exactly as before; the kappa axis moves the dispersion-prior family
    # (the near-Binomial-capable 1/sqrt parameterisation against the registered
    # HalfNormal(50)); the gamma_own axis runs the prior's own documented
    # 0.25-vs-0.5 width sensitivity on graded fits.
    if axis == "tau":
        kwargs["trt_prior_sigma"] = sigma
    elif axis == "kappa":
        if plan.off_floor:
            raise RuntimeError(
                f"{model_id}: the kappa axis applies to graded Beta-Binomial "
                "fits only (an off-floor Bernoulli fit has no dispersion)"
            )
        family, kappa_sigma_value = KAPPA_AXIS_CELLS[int(sigma)]
        kwargs["kappa_prior_family"] = family
        kwargs["kappa_sigma"] = kappa_sigma_value
    elif axis == "gamma_own":
        if plan.off_floor:
            raise RuntimeError(
                f"{model_id}: the gamma_own axis applies to graded fits only "
                "(the off-floor path carries the indicator prior instead)"
            )
        kwargs["gamma_own_prior_sigma"] = sigma
    else:  # pragma: no cover - argparse restricts the choices
        raise RuntimeError(f"unknown sensitivity axis {axis!r}")
    built = build_gain_factors_model(prepared, **kwargs)
    # Cells adopt the primary's own recorded target_accept (see
    # assert_gf_sampling_contract); --cell-target-accept may only RAISE it —
    # stricter integration does not change the posterior, and a single
    # escalated bundle is honest where per-cell seed selection would not be.
    target_accept = float(primary_reference.sampling["target_accept"])
    if cell_target_accept is not None:
        target_accept = max(target_accept, float(cell_target_accept))
    with built.model:
        trace = pm.sample(
            draws=sampling.draws,
            tune=sampling.tune,
            chains=sampling.chains,
            cores=sampling.cores,
            target_accept=target_accept,
            nuts_sampler="nutpie",
            return_inferencedata=True,
            random_seed=sampling.random_seed,
            progressbar=False,
        )

    focal_draws = (
        trace.posterior["beta_trt"].stack(sample=("chain", "draw")).values.ravel()
    )
    lo_q = (1.0 - REPORTING_CI_PROB) / 2.0
    n_trials = 1 if plan.off_floor else int(MEASURES[outcome].n_trials)
    items_mean, items_lo, items_hi = _items_translation(
        trace, built, plan, n_trials=n_trials, ci_prob=REPORTING_CI_PROB
    )
    kappa_draws = (
        trace.posterior["kappa"].stack(sample=("chain", "draw")).values
        if "kappa" in trace.posterior
        else np.array([np.nan])
    )
    free_names = [rv.name for rv in built.model.free_RVs]
    convergence = _diag.subfit_convergence(
        trace, label=f"{model_id} tau", var_names=free_names
    )

    trt = np.asarray(
        ((built.prepared.G == 1) | (built.prepared.phase >= 1)), dtype=float
    )
    phase = np.asarray(built.prepared.phase)
    p1 = phase == 0
    n = int(built.prepared.n_obs)
    n_intervention = int(np.sum(trt[p1] == 1))
    n_control = int(np.sum(trt[p1] == 0))
    data_sha256 = str(getattr(built.prepared, "data_sha256", ""))
    if (
        primary_reference.config_name != config
        or primary_reference.model_id != model_id
        or primary_reference.outcome != outcome
        or data_sha256 != primary_reference.data_sha256
        or n != primary_reference.n
        or n_intervention != primary_reference.n_intervention
        or n_control != primary_reference.n_control
    ):
        raise RuntimeError(
            f"{model_id} gain-factor sensitivity does not match its current "
            "primary data, sample, period-1 arm counts, or config"
        )

    row = {
        **primary_reference.manifest_values(),
        "config": config,
        "outcome": outcome,
        "n_trials": n_trials,
        "sensitivity_axis": axis,
        "tau_sigma": sigma if axis == "tau" else np.nan,
        "gamma_own_sigma": (
            sigma
            if axis == "gamma_own"
            else (
                GAMMA_OWN_SIGMA_OFFFLOOR if plan.off_floor else GAMMA_OWN_SIGMA_GRADED
            )
        ),
        "kappa_prior_family": kwargs.get(
            "kappa_prior_family", "halfnormal_concentration"
        ),
        "kappa_sigma": (
            np.nan
            if plan.off_floor
            else (
                kwargs.get("kappa_sigma")
                if axis == "kappa" and kwargs.get("kappa_sigma") is not None
                else KAPPA_SIGMA
            )
        ),
        "use_precision_terms": True,
        "data_sha256": data_sha256,
        "n": n,
        "n_intervention": n_intervention,
        "n_control": n_control,
        "n_p1_rows": int(np.sum(p1)),
        "pd": float(np.mean(focal_draws > 0)),
        "tau_logit_mean": float(np.mean(focal_draws)),
        "tau_logit_lo": float(np.quantile(focal_draws, lo_q)),
        "tau_logit_hi": float(np.quantile(focal_draws, 1.0 - lo_q)),
        "ci_width_logit": float(
            np.quantile(focal_draws, 1.0 - lo_q) - np.quantile(focal_draws, lo_q)
        ),
        "tau_sd_logit": float(np.std(focal_draws)),
        "kappa_median": float(np.nanmedian(kappa_draws)),
        "items_mean": items_mean,
        "items_lo": items_lo,
        "items_hi": items_hi,
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
        "sampling_target_accept": target_accept,
        "sampling_random_seed": sampling.random_seed,
        "sampling_nuts_sampler": "nutpie",
    }
    # Gain-factor provenance stamped on the cell trace: same identity content as
    # the did runner, keyed by the swept model (TR/TE each have two primaries).
    provenance = {
        "schema_version": 1,
        "model_kind": "gain_factors",
        "config": config,
        "outcome": outcome,
        "model_id": model_id,
        "focal_term": "beta_trt",
        "sensitivity_axis": axis,
        "tau_sigma": sigma if axis == "tau" else None,
        "axis_value": sigma,
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
            "target_accept": target_accept,
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
    sigma_token = f"{sigma:g}".replace(".", "p")
    token = model_id.removeprefix("lrp-rli-")
    semantic = (
        Path("traces")
        / f"gf-{config}"
        / f"trace_{token}_{axis}-{sigma_token}.nc"
    )
    trace_file, trace_sha256 = persist_sensitivity_trace(
        trace,
        sensitivity_dir=sensitivity_dir,
        semantic_file=semantic,
        label="gain-factor sensitivity",
    )
    row.update(trace_file=trace_file.as_posix(), trace_sha256=trace_sha256)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="dev", help="sampling preset (dev/test/reporting)")
    ap.add_argument(
        "--models", nargs="+", default=list(GF_SENSITIVITY_DEFAULT_MODEL_IDS),
        help=(
            "gain-factor primary ids to sweep (default: the two historically "
            "prior-dominant off-floor fits gf-005/gf-011; any non-companion "
            "primary is accepted)"
        ),
    )
    ap.add_argument(
        "--attach", action="store_true",
        help=(
            "install each model's trace-backed bundle beside its primary "
            "(manifest + digest-verified cell traces; only when every cell "
            "converged and the gate's own evidence check passes)"
        ),
    )
    ap.add_argument(
        "--reattach", action="store_true",
        help=(
            "skip fitting: re-install bundles from the sweep directory's "
            "existing combined CSV (re-verifying every binding and trace hash)"
        ),
    )
    ap.add_argument(
        "--output-dir", type=str, default=None,
        help="override the output root (above DSE_LRP_OUTPUT_DIR); layout unchanged",
    )
    ap.add_argument(
        "--seed", type=int, default=20260701,
        help=(
            "cell sampling seed (recorded per row); a reseed is the legitimate "
            "first response to a stochastic divergence-only cell failure — the "
            "draws/tune/chains/target-accept contract is unchanged"
        ),
    )
    ap.add_argument(
        "--axis", choices=("tau", "kappa", "gamma_own"), default="tau",
        help=(
            "what the sweep varies (#575 finding 10): 'tau' is the standard "
            "beta_trt prior grid and release-evidence artefact; 'kappa' fits "
            "the dispersion-prior-family cells (registered HalfNormal(50) "
            "anchor, then 1/sqrt(kappa) ~ HalfNormal(0.25)/(0.5)); 'gamma_own' "
            "runs the documented 0.25-vs-0.5 own-baseline width sensitivity. "
            "Non-tau axes write to their own sweep directories and cannot be "
            "attached as release evidence"
        ),
    )
    ap.add_argument(
        "--cell-target-accept", type=float, default=None,
        help=(
            "escalate the cells' target_accept ABOVE the primary's recorded "
            "value (each cell runs at max(primary, this); recorded per row in "
            "sampling_target_accept beside the primary's own value)"
        ),
    )
    args = ap.parse_args()

    unknown = sorted(set(args.models) - set(GF_SENSITIVITY_MODEL_IDS))
    if unknown:
        ap.error(
            f"unsupported gain-factor sensitivity models: {unknown}; choose from "
            f"{sorted(GF_SENSITIVITY_MODEL_IDS)}"
        )

    _paths.set_output_root(args.output_dir)
    print(f"Output root: {_paths.describe_output_root()}")
    models_root = Path(_paths.stat_dir()) / "models"
    _axis_dirs = {
        "tau": "gf_tau_prior_sensitivity",
        "kappa": "gf_kappa_prior_sensitivity",
        "gamma_own": "gf_gamma_own_prior_sensitivity",
    }
    sensitivity_dir = Path(_paths.stat_dir()) / _axis_dirs[args.axis]
    sensitivity_dir.mkdir(parents=True, exist_ok=True)
    if args.axis != "tau" and (args.attach or args.reattach):
        ap.error(
            "--attach/--reattach apply to the tau axis only: the kappa and "
            "gamma_own sweeps are reported sensitivity artefacts, not per-fit "
            "release evidence"
        )

    sampling = _sampling.get_sampling_configuration(args.config, random_seed=args.seed)
    combined_path = sensitivity_dir / f"gf_{STANDARD_SENSITIVITY_FILENAME}"

    if args.reattach:
        # Re-install from the existing combined manifest without refitting:
        # every binding (current primary hashes, trace digests, convergence,
        # sign, model identity) is re-verified by attach_outcome_bundle, so a
        # stale or tampered sweep cannot be re-exposed. Rows are selected by
        # primary_model_id — the TR/TE outcomes each have two primaries.
        combined = pd.read_csv(combined_path)
        for model_id in args.models:
            primary_dir = models_root / f"{model_id}-{args.config}"
            reference = load_primary_gf_reference(
                primary_dir, model_id, config_name=args.config
            )
            assert_gf_sampling_contract(sampling, reference, config=args.config)
            model_rows = combined.loc[
                combined["primary_model_id"].astype(str) == model_id
            ]
            destination = attach_outcome_bundle(
                model_rows,
                outcome=reference.outcome,
                primary_dir=primary_dir,
                sensitivity_dir=sensitivity_dir,
                reference=reference,
            )
            print(f"re-attached {destination}")
        return

    rows: list[dict] = []
    attach_ready: dict[str, bool] = {}
    for model_id in args.models:
        primary_dir = models_root / f"{model_id}-{args.config}"
        reference = load_primary_gf_reference(
            primary_dir, model_id, config_name=args.config
        )
        # Preset must match the primary on draws/tune/chains; the primary's own
        # recorded target_accept is adopted per cell.
        assert_gf_sampling_contract(sampling, reference, config=args.config)
        plan = _resolve_plan(model_id)
        model_rows = []
        if args.axis == "tau":
            grid = tuple(float(v) for v in _grid_for(plan))
        elif args.axis == "kappa":
            grid = tuple(float(k) for k in sorted(KAPPA_AXIS_CELLS))
        else:
            grid = tuple(float(v) for v in GAMMA_OWN_AXIS_SIGMAS)
        for sigma in grid:
            print(
                f"--- {model_id} ({plan.outcome_symbol}, beta_trt): "
                f"axis={args.axis} value={sigma} ---"
            )
            row = _fit_cell(
                model_id,
                float(sigma),
                config=args.config,
                sampling=sampling,
                sensitivity_dir=sensitivity_dir,
                primary_reference=reference,
                cell_target_accept=args.cell_target_accept,
                axis=args.axis,
            )
            print(
                f"    tau_logit_mean={row['tau_logit_mean']:+.3f} "
                f"[{row['tau_logit_lo']:+.3f}, {row['tau_logit_hi']:+.3f}] "
                f"converged={row['converged']}"
            )
            model_rows.append(row)
        rows.extend(model_rows)
        all_ok = all(bool(r["converged"]) for r in model_rows)
        attach_ready[model_id] = all_ok
        if args.attach:
            if not all_ok:
                print(
                    f"    NOT attaching {model_id}: one or more cells failed "
                    "the convergence gate"
                )
                continue
            destination = attach_outcome_bundle(
                pd.DataFrame(model_rows),
                outcome=reference.outcome,
                primary_dir=primary_dir,
                sensitivity_dir=sensitivity_dir,
                reference=reference,
            )
            print(f"    attached {destination}")

    combined = pd.DataFrame(rows)
    # A subset rerun (e.g. reseeding one model's stochastic divergence-only
    # failure) replaces that model's rows and preserves the others, so the
    # combined manifest stays reattachable for every swept fit.
    if combined_path.exists():
        previous = pd.read_csv(combined_path)
        swept = set(combined["primary_model_id"].astype(str))
        previous = previous.loc[
            ~previous["primary_model_id"].astype(str).isin(swept)
        ]
        combined = pd.concat([previous, combined], ignore_index=True)
    combined.to_csv(combined_path, index=False)
    print(f"\nWrote {combined_path} ({len(combined)} rows)")
    for model_id, ok in attach_ready.items():
        state = "all cells converged" if ok else "HAS UNCONVERGED CELLS"
        print(f"  {model_id}: {state}")


if __name__ == "__main__":
    main()
