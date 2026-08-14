# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Treatment-prior sensitivity sweep for the did family (#390).

The robustness release gate (#482) covers ``did`` on the plan's focal term —
``tau_t2`` for the arm-by-wave models, the dose slope for the dose companions
(``release.causal_term_for`` mirrors ``DiDRunPlan.effect_term``) — and for a
prior-dominant fit it names its release evidence exactly: *"a
tau_prior_sensitivity.csv treatment-prior sweep, computed from this fit's own
trace, showing the sign of the effect is stable across the grid"*
(``release._standard_sweep_evidence``). Four did fits ship
``robustness_unresolved`` because no did runner produced that artefact. This
script is the did-family runner, sharing the level runner's attach discipline
via :mod:`sensitivity` (#488 review):

- For each requested model (default: the four withheld fits — did-001 W and
  did-013 W on ``tau_t2``, did-003 B on ``tau_t2``, did-007 L on ``mu_dose`` —
  plus the did-101 independent-prior intercept companion, whose ``tau_t2``
  flags prior-dominant exactly as its anchored parent's does) it resolves the
  registered primary's **typed run plan**, rebuilds the model
  in-process with the focal term's prior moved across its grid (the proximal
  tau grid 0.25 / 0.5 / 0.75 for the tau_t2 fits; 0.5 / 1.0 / 1.5 around the
  ``Normal(0, 1)`` default for the dose slope), holding everything else —
  data, waves, anchored intercept, likelihood — at the registered
  specification. Unlike the level sweep the set is keyed by **model id**: two
  withheld fits share outcome W.
- Cells reproduce the primary's own sampling contract: the preset must match
  on draws/tune/chains, and the primary's recorded ``target_accept`` is
  adopted for every cell (did-007's registered spec overrides the preset with
  0.97, and evidence for that fit must be sampled under that fit's contract).
- Each cell is gated on the full convergence criteria over all free variables
  (R-hat <= 1.01, ESS >= 400, BFMI >= 0.3, zero divergences); an unconverged
  cell is not evidence and blocks that model's report-local copy.
- Rows carry the standard sweep's full column set, the primary fit's
  ``config.json`` / ``trace.nc`` sha256 bindings, and a content-addressed cell
  trace. ``tau_sigma`` / ``tau_logit_*`` are the schema's generic focal-term
  columns: for did-007 they hold the ``mu_dose`` prior scale and posterior
  (``sensitivity_axis`` records which term was swept). Items columns use the
  family's own translations — the wave-standardised t2 arm gap
  (``did_summary``) for arm-by-wave fits, the +1 SD session pushforward
  (``dose_marginal_summary``'s definition) for dose fits.
- With ``--attach`` the per-model rows are written to
  ``<fit-dir>/tau_prior_sensitivity.csv`` beside each primary whose cells all
  converged (trace-backed, digest-verified, atomic, rolled back on failure).

Usage:
    python scripts/did_prior_sensitivity.py --config reporting --attach
    python scripts/did_prior_sensitivity.py --config reporting \
        --models lrp-rli-did-007 --attach
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
from language_reading_predictors.statistical_models.factories import build_did_model
from language_reading_predictors.statistical_models.fitted_payloads import (
    DidDosePayload,
)
from language_reading_predictors.statistical_models.measures import (
    DISTAL_OUTCOMES,
    MEASURES,
)
from language_reading_predictors.statistical_models.sensitivity import (
    DID_SENSITIVITY_MODEL_IDS,
    DID_SENSITIVITY_MU_DOSE_SIGMAS,
    STANDARD_SENSITIVITY_DISTAL_TAU_SIGMAS,
    STANDARD_SENSITIVITY_FILENAME,
    STANDARD_SENSITIVITY_PROVENANCE_ATTR,
    STANDARD_SENSITIVITY_PROXIMAL_TAU_SIGMAS,
    STANDARD_SENSITIVITY_SAMPLING_ATTR,
    PrimaryStandardReference,
    assert_primary_sampling_contract,
    attach_outcome_bundle,
    load_primary_did_reference,
    persist_sensitivity_trace,
)

KAPPA_SIGMA = 50.0
# The dose models' own-baseline coefficient gamma_t1 keeps the shared
# gamma_own_prior default; it is never swept here.
GAMMA_OWN_SIGMA_DEFAULT = 0.25


def assert_did_sampling_contract(
    sampling, reference: PrimaryStandardReference, *, config: str
) -> None:
    """The did variant of the shared contract: match the preset on
    draws/tune/chains and *adopt* the primary's recorded ``target_accept``
    (a registered spec may override the preset — did-007 does, at 0.97)."""
    assert_primary_sampling_contract(
        sampling,
        reference,
        config=config,
        keys=("draws", "tune", "chains"),
        label=f"{reference.model_id} did",
    )


def _resolve_plan(model_id: str):
    """The registered primary's typed run plan (single source of truth)."""
    import importlib

    from language_reading_predictors.statistical_models.did import (
        resolve_did_run_plan,
    )

    module = importlib.import_module(
        "language_reading_predictors.statistical_models."
        + model_id.replace("-", "_")
    )
    return resolve_did_run_plan(module.SPEC)


def _grid_for(plan) -> tuple[float, ...]:
    """The focal term's prior grid: the outcome-tier tau grid, or the dose
    slope's own grid around its Normal(0, 1) default."""
    if plan.effect_term in ("mu_dose", "beta_dose"):
        return DID_SENSITIVITY_MU_DOSE_SIGMAS
    if plan.outcome_symbol in DISTAL_OUTCOMES:
        return STANDARD_SENSITIVITY_DISTAL_TAU_SIGMAS
    return STANDARD_SENSITIVITY_PROXIMAL_TAU_SIGMAS


def _items_translation(
    trace, built, plan, *, n_trials: int, ci_prob: float
) -> tuple[float, float, float]:
    """(items_mean, items_lo, items_hi) on the family's own estimand scale.

    Arm-by-wave fits use ``did_summary``'s wave-standardised t2 arm gap — the
    exact quantity the primary's report and key findings carry. Dose fits use
    ``_write_dose_slope_summary``'s natural-scale marginal: raise the
    standardised session dose by 1 on every fitted row with that row's
    period-specific slope.
    """
    from language_reading_predictors.statistical_models.reporting import did_summary

    lo_q = (1.0 - ci_prob) / 2.0
    posterior = trace.posterior
    if not plan.dose:
        summary = did_summary(
            trace,
            ci_prob=ci_prob,
            n_trials=n_trials,
            off_floor=plan.off_floor,
            wave=np.asarray(built.prepared.phase, dtype=np.int64),
        )
        return (
            float(summary["tau_t2_items_mean"]),
            float(summary["tau_t2_items_lo"]),
            float(summary["tau_t2_items_hi"]),
        )
    eta = (
        posterior["eta"]
        .stack(sample=("chain", "draw"))
        .transpose("obs_id", "sample")
        .values
    )
    if plan.period_varying:
        stacked = posterior["beta_dose_phase"].stack(sample=("chain", "draw"))
        phase_dim = next(d for d in stacked.dims if d != "sample")
        slopes = stacked.transpose(phase_dim, "sample").values
        delta_eta = slopes[np.asarray(built.prepared.phase, dtype=np.int64)]
    else:
        slope = posterior["beta_dose"].stack(sample=("chain", "draw")).values
        delta_eta = np.broadcast_to(slope[None, :], eta.shape)
    items = (expit(eta + delta_eta) - expit(eta)).mean(axis=0) * float(n_trials)
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
) -> dict:
    """Fit one did sweep cell and return its standard-schema row."""
    from language_reading_predictors.statistical_models import diagnostics as _diag
    from language_reading_predictors.statistical_models.preprocessing import (
        load_and_prepare,
    )
    from language_reading_predictors.statistical_models.reporting import (
        REPORTING_CI_PROB,
    )

    plan = _resolve_plan(model_id)
    focal = plan.effect_term
    axis = "tau" if focal == "tau_t2" else focal
    outcome = plan.outcome_symbol
    prepared = load_and_prepare(**plan.prepare_kwargs())
    kwargs = plan.factory_kwargs()
    if focal == "tau_t2":
        kwargs["tau_t2_prior_sigma"] = sigma
    else:
        kwargs["dose_slope_prior_sigma"] = sigma
    built = build_did_model(prepared, **kwargs)
    # Cells adopt the primary's own recorded target_accept (see
    # assert_did_sampling_contract): evidence must be sampled under the
    # registered fit's contract, including any per-model override. An explicit
    # --cell-target-accept may only *raise* it (stricter integration does not
    # change the posterior; a looser sampler than the primary's would relax the
    # contract the runner exists to enforce).
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

    focal_draws = trace.posterior[focal].stack(sample=("chain", "draw")).values.ravel()
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
        trace, label=f"{model_id} {axis}", var_names=free_names
    )

    G = np.asarray(built.prepared.G)
    phase = np.asarray(built.prepared.phase)
    n = int(built.prepared.n_obs)
    n_intervention = int(np.sum(G == 1))
    n_control = int(np.sum(G == 0))
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
            f"{model_id} did sensitivity does not match its current primary "
            "did data, sample, arm counts, or config"
        )

    row = {
        **primary_reference.manifest_values(),
        "config": config,
        "outcome": outcome,
        "n_trials": n_trials,
        "sensitivity_axis": axis,
        "tau_sigma": sigma,
        # The arm-by-wave models have no own-baseline term; the dose models
        # keep gamma_t1 at its (unswept) shared default.
        "gamma_own_sigma": GAMMA_OWN_SIGMA_DEFAULT if plan.dose else np.nan,
        "kappa_sigma": np.nan if plan.off_floor else KAPPA_SIGMA,
        "use_precision_terms": bool(plan.use_age),
        "data_sha256": data_sha256,
        "n": n,
        "n_intervention": n_intervention,
        "n_control": n_control,
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
    if plan.dose:
        payload = built.require_payload(DidDosePayload, family="did sensitivity")
        row["n_treated_rows"] = int(np.sum(np.asarray(payload.treated) == 1))
    else:
        row["n_t2_intervention"] = int(np.sum(G[phase == 1] == 1))
        row["n_t2_control"] = int(np.sum(G[phase == 1] == 0))
    # Did-family provenance stamped on the cell trace: same identity content as
    # the ITT/level runners, keyed by the swept model rather than the outcome.
    provenance = {
        "schema_version": 1,
        "model_kind": "did",
        "config": config,
        "outcome": outcome,
        "model_id": model_id,
        "focal_term": focal,
        "sensitivity_axis": axis,
        "tau_sigma": sigma,
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
        / f"did-{config}"
        / f"trace_{token}_{axis}-{sigma_token}.nc"
    )
    trace_file, trace_sha256 = persist_sensitivity_trace(
        trace,
        sensitivity_dir=sensitivity_dir,
        semantic_file=semantic,
        label="did sensitivity",
    )
    row.update(trace_file=trace_file.as_posix(), trace_sha256=trace_sha256)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="dev", help="sampling preset (dev/test/reporting)")
    ap.add_argument(
        "--models", nargs="+", default=list(DID_SENSITIVITY_MODEL_IDS),
        help=(
            "did model ids to sweep (default: the four #390 withheld fits plus "
            "the did-101 intercept companion)"
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
        "--cell-target-accept", type=float, default=None,
        help=(
            "escalate the cells' target_accept ABOVE the primary's recorded "
            "value (each cell runs at max(primary, this); recorded per row in "
            "sampling_target_accept beside the primary's own value). For a fit "
            "whose cells show seed-hopping divergence-only failures at the "
            "primary's contract: stricter integration never changes the "
            "posterior, and a single escalated bundle is honest where "
            "per-cell seed selection would not be"
        ),
    )
    args = ap.parse_args()

    unknown = sorted(set(args.models) - set(DID_SENSITIVITY_MODEL_IDS))
    if unknown:
        ap.error(
            f"unsupported did sensitivity models: {unknown}; choose from "
            f"{sorted(DID_SENSITIVITY_MODEL_IDS)}"
        )

    _paths.set_output_root(args.output_dir)
    print(f"Output root: {_paths.describe_output_root()}")
    models_root = Path(_paths.stat_dir()) / "models"
    sensitivity_dir = Path(_paths.stat_dir()) / "did_tau_prior_sensitivity"
    sensitivity_dir.mkdir(parents=True, exist_ok=True)

    sampling = _sampling.get_sampling_configuration(args.config, random_seed=args.seed)
    combined_path = sensitivity_dir / f"did_{STANDARD_SENSITIVITY_FILENAME}"

    if args.reattach:
        # Re-install from the existing combined manifest without refitting:
        # every binding (current primary hashes, trace digests, convergence,
        # sign, model identity) is re-verified by attach_outcome_bundle, so a
        # stale or tampered sweep cannot be re-exposed. Rows are selected by
        # primary_model_id — two swept fits share outcome W.
        combined = pd.read_csv(combined_path)
        for model_id in args.models:
            primary_dir = models_root / f"{model_id}-{args.config}"
            reference = load_primary_did_reference(
                primary_dir, model_id, config_name=args.config
            )
            assert_did_sampling_contract(sampling, reference, config=args.config)
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
        reference = load_primary_did_reference(
            primary_dir, model_id, config_name=args.config
        )
        # Preset must match the primary on draws/tune/chains; the primary's own
        # target_accept (did-007: a registered 0.97 override) is adopted below.
        assert_did_sampling_contract(sampling, reference, config=args.config)
        plan = _resolve_plan(model_id)
        model_rows = []
        for sigma in _grid_for(plan):
            print(
                f"--- {model_id} ({plan.outcome_symbol}, {plan.effect_term}): "
                f"sigma={sigma} ---"
            )
            row = _fit_cell(
                model_id,
                float(sigma),
                config=args.config,
                sampling=sampling,
                sensitivity_dir=sensitivity_dir,
                primary_reference=reference,
                cell_target_accept=args.cell_target_accept,
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
