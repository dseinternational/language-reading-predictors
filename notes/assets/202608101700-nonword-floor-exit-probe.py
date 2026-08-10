# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Promotion probe for the lagged word-reading/nonword floor-exit association.

This script implements the pre-fit decision in
``notes/202608101700-nonword-floor-exit-method-decision.md``. It fits a Bernoulli
full model containing standardised ``log1p(W_pre)`` and a genuine nested null that
removes only that term. The grid crosses the full true-floor population and the
``W_pre <= 25`` instrument-tail sensitivity with slope priors ``Normal(0, 0.3)``
and ``Normal(0, 1)``.

Nothing fitted here is causal or publication-ready. The script writes diagnostic,
comparison, risk-difference and row-identity evidence to ``output/notes``; it does
not register a statistical model or bypass the production release gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from dse_research_utils.statistics.diagnostics import (
    BFMI_THRESHOLD,
    ESS_THRESHOLD,
    RHAT_MAX,
)
from scipy.special import expit

from language_reading_predictors.data_utils import load_data
from language_reading_predictors.statistical_models.sampling_quality import (
    sampling_quality,
)

CI = 0.89
SEED = 20260810
WR = "ewrswr"
NW = "nonword"
RISK_REFERENCES = (0.0, 5.0, 25.0)
SLOPE_PRIOR_SDS = (0.3, 1.0)
COVARIATE_NAMES = (
    "age_z",
    "arm",
    "on_intervention",
    "hearing_z",
    "hearing_missing",
    "speech_z",
    "speech_missing",
)


@dataclass(frozen=True, slots=True)
class ReferenceDesign:
    """Frozen centring/filling constants shared by every sensitivity fit."""

    age_mean: float
    age_sd: float
    hearing_mean: float
    hearing_sd: float
    speech_mean: float
    speech_sd: float
    log_wr_mean: float
    log_wr_sd: float


@dataclass(frozen=True, slots=True)
class PreparedProbe:
    """One analysis population and its frozen design arrays."""

    label: str
    frame: pd.DataFrame
    X: np.ndarray
    y: np.ndarray
    wr_z: np.ndarray
    subject_index: np.ndarray
    subject_labels: tuple[str, ...]
    row_sha256: str
    observed_sha256: str


def _positive_sd(values: np.ndarray, *, name: str) -> float:
    sd = float(np.std(np.asarray(values, dtype=float)))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError(f"{name} has no positive finite standard deviation")
    return sd


def transition_frame() -> pd.DataFrame:
    """Observed t1-to-t2, t2-to-t3 and t3-to-t4 transitions at the NW floor."""
    raw = load_data()
    frame = raw.loc[raw["time"] <= 3].copy()
    frame["wr_pre"] = frame[WR]
    frame["nw_pre"] = frame[NW]
    frame["nw_post"] = frame[f"{NW}_next"]
    frame["arm"] = (frame["group"] == 1).astype(int)
    frame["on_int"] = frame["on_intervention"].astype(int)
    required = [
        "subject_id",
        "time",
        "wr_pre",
        "nw_pre",
        "nw_post",
        "age",
        "group",
        "on_intervention",
    ]
    frame = frame.dropna(subset=required)
    frame = frame.loc[frame["nw_pre"] == 0].copy()
    frame["y_exit"] = (frame["nw_post"] > 0).astype(int)
    return frame.sort_values(["subject_id", "time"], kind="stable").reset_index(
        drop=True
    )


def reference_design(frame: pd.DataFrame) -> ReferenceDesign:
    """Derive constants once from the full true-floor population."""
    hearing = frame["hearing_c"].to_numpy(float)
    speech = frame["deapp_c"].to_numpy(float)
    hearing_mean = float(np.nanmean(hearing))
    speech_mean = float(np.nanmean(speech))
    hearing_filled = np.where(np.isnan(hearing), hearing_mean, hearing)
    speech_filled = np.where(np.isnan(speech), speech_mean, speech)
    age = frame["age"].to_numpy(float)
    log_wr = np.log1p(frame["wr_pre"].to_numpy(float))
    return ReferenceDesign(
        age_mean=float(np.mean(age)),
        age_sd=_positive_sd(age, name="age"),
        hearing_mean=hearing_mean,
        hearing_sd=_positive_sd(hearing_filled, name="hearing"),
        speech_mean=speech_mean,
        speech_sd=_positive_sd(speech_filled, name="speech"),
        log_wr_mean=float(np.mean(log_wr)),
        log_wr_sd=_positive_sd(log_wr, name="log1p word reading"),
    )


def _digest_rows(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256(b"dse-lrp-nw-floor-exit-rows-v1\0")
    for row in frame[["subject_id", "time"]].itertuples(index=False):
        value = f"{row.subject_id}\0{int(row.time)}".encode()
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


def _digest_observed(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256(b"dse-lrp-nw-floor-exit-observed-v1\0")
    columns = (
        "subject_id",
        "time",
        "wr_pre",
        "nw_pre",
        "nw_post",
        "age",
        "arm",
        "on_int",
        "hearing_c",
        "deapp_c",
    )
    payload = frame.loc[:, columns].to_csv(index=False, lineterminator="\n")
    digest.update(payload.encode())
    return digest.hexdigest()


def prepare_probe(
    frame: pd.DataFrame,
    design: ReferenceDesign,
    *,
    label: str,
    max_words: float | None,
) -> PreparedProbe:
    """Prepare one sensitivity population using the frozen reference constants."""
    sub = frame if max_words is None else frame.loc[frame["wr_pre"] <= max_words]
    sub = sub.copy().reset_index(drop=True)
    hearing = sub["hearing_c"].to_numpy(float)
    speech = sub["deapp_c"].to_numpy(float)
    hearing_missing = np.isnan(hearing)
    speech_missing = np.isnan(speech)
    hearing_filled = np.where(hearing_missing, design.hearing_mean, hearing)
    speech_filled = np.where(speech_missing, design.speech_mean, speech)
    X = np.column_stack(
        [
            (sub["age"].to_numpy(float) - design.age_mean) / design.age_sd,
            sub["arm"].to_numpy(float),
            sub["on_int"].to_numpy(float),
            (hearing_filled - design.hearing_mean) / design.hearing_sd,
            hearing_missing.astype(float),
            (speech_filled - design.speech_mean) / design.speech_sd,
            speech_missing.astype(float),
        ]
    )
    wr_z = (
        np.log1p(sub["wr_pre"].to_numpy(float)) - design.log_wr_mean
    ) / design.log_wr_sd
    subject = pd.Categorical(sub["subject_id"])
    return PreparedProbe(
        label=label,
        frame=sub,
        X=X,
        y=sub["y_exit"].to_numpy(int),
        wr_z=wr_z,
        subject_index=subject.codes.astype(int),
        subject_labels=tuple(str(value) for value in subject.categories),
        row_sha256=_digest_rows(sub),
        observed_sha256=_digest_observed(sub),
    )


def build_model(
    prepared: PreparedProbe,
    *,
    include_word_reading: bool,
    slope_prior_sd: float,
) -> pm.Model:
    """Build the full or null model; the latter removes only ``b_wr``."""
    if slope_prior_sd not in SLOPE_PRIOR_SDS:
        raise ValueError(f"unsupported slope prior SD {slope_prior_sd}")
    coords = {
        "obs_id": np.arange(len(prepared.y)),
        "covariate": COVARIATE_NAMES,
        "child": prepared.subject_labels,
    }
    with pm.Model(coords=coords) as model:
        alpha = pm.Normal("alpha", 0.0, 1.5)
        beta = pm.Normal("beta", 0.0, 0.3, dims="covariate")
        sigma_child = pm.HalfNormal("sigma_child", 0.5)
        u_raw = pm.Normal("u_raw", 0.0, 1.0, dims="child")
        eta = (
            alpha
            + pm.math.dot(prepared.X, beta)
            + (sigma_child * u_raw)[prepared.subject_index]
        )
        if include_word_reading:
            b_wr = pm.Normal("b_wr", 0.0, slope_prior_sd)
            eta = eta + b_wr * prepared.wr_z
        pm.Deterministic("p_exit", pm.math.sigmoid(eta), dims="obs_id")
        pm.Bernoulli("y_exit", logit_p=eta, observed=prepared.y, dims="obs_id")
    return model


def _fit(
    prepared: PreparedProbe,
    *,
    include_word_reading: bool,
    slope_prior_sd: float,
    draws: int,
    tune: int,
    chains: int,
    seed: int,
):
    model = build_model(
        prepared,
        include_word_reading=include_word_reading,
        slope_prior_sd=slope_prior_sd,
    )
    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=min(chains, 4),
            target_accept=0.95,
            nuts_sampler="nutpie",
            random_seed=seed,
            progressbar=False,
            compute_convergence_checks=False,
            idata_kwargs={"log_likelihood": True},
        )
    return model, trace


def fit_diagnostics(trace, model: pm.Model) -> dict[str, Any]:
    """Unrounded project convergence gate over every free variable."""
    var_names = [rv.name for rv in model.free_RVs]
    summary = az.summary(
        trace,
        var_names=var_names,
        round_to="none",
        kind="diagnostics",
    )
    signals = sampling_quality(trace, var_names=var_names)
    min_bulk = float(summary["ess_bulk"].min())
    min_tail = float(summary["ess_tail"].min())
    gate_pass = bool(
        np.isfinite(signals.max_rhat)
        and signals.max_rhat <= RHAT_MAX
        and np.isfinite(min_bulk)
        and min_bulk >= ESS_THRESHOLD
        and np.isfinite(min_tail)
        and min_tail >= ESS_THRESHOLD
        and signals.min_bfmi is not None
        and signals.min_bfmi >= BFMI_THRESHOLD
        and signals.n_divergences == 0
    )
    return {
        "max_rhat": signals.max_rhat,
        "min_ess": signals.min_ess,
        "min_ess_bulk": min_bulk,
        "min_ess_tail": min_tail,
        "min_bfmi": signals.min_bfmi,
        "n_divergences": signals.n_divergences,
        "gate_pass": gate_pass,
        "gate_scope": "all_free_variables",
        "free_variables": ",".join(var_names),
    }


def _interval(draws: np.ndarray) -> dict[str, float]:
    values = np.asarray(draws, dtype=float).ravel()
    lo_q = (1 - CI) / 2
    hi_q = 1 - lo_q
    return {
        "median": float(np.median(values)),
        "lo50": float(np.quantile(values, 0.25)),
        "hi50": float(np.quantile(values, 0.75)),
        "lo89": float(np.quantile(values, lo_q)),
        "hi89": float(np.quantile(values, hi_q)),
        "prob_positive": float(np.mean(values > 0)),
    }


def risk_difference_summary(
    trace,
    prepared: PreparedProbe,
    design: ReferenceDesign,
) -> list[dict[str, Any]]:
    """Transition-standardised floor-exit risk differences from the full fit."""
    posterior = trace.posterior
    alpha = posterior["alpha"].stack(sample=("chain", "draw")).values
    beta = posterior["beta"].stack(sample=("chain", "draw")).values
    sigma = posterior["sigma_child"].stack(sample=("chain", "draw")).values
    u_raw = posterior["u_raw"].stack(sample=("chain", "draw")).values
    b_wr = posterior["b_wr"].stack(sample=("chain", "draw")).values
    base = (
        alpha[None, :]
        + prepared.X @ beta
        + sigma[None, :] * u_raw[prepared.subject_index, :]
    )
    rows = []
    for reference in RISK_REFERENCES[1:]:
        z0 = (np.log1p(RISK_REFERENCES[0]) - design.log_wr_mean) / design.log_wr_sd
        z1 = (np.log1p(reference) - design.log_wr_mean) / design.log_wr_sd
        risk0 = expit(base + b_wr[None, :] * z0).mean(axis=0)
        risk1 = expit(base + b_wr[None, :] * z1).mean(axis=0)
        diff = risk1 - risk0
        row: dict[str, Any] = {
            "reference_low_words": RISK_REFERENCES[0],
            "reference_high_words": reference,
            **{f"risk_low_{key}": value for key, value in _interval(risk0).items()},
            **{f"risk_high_{key}": value for key, value in _interval(risk1).items()},
            **{f"risk_difference_{key}": value for key, value in _interval(diff).items()},
        }
        rows.append(row)
    return rows


def slope_summary(trace) -> dict[str, float]:
    """Posterior summary for one SD increase in ``log1p(W_pre)``."""
    draws = trace.posterior["b_wr"].values
    return {f"slope_{key}": value for key, value in _interval(draws).items()}


def loo_comparison(
    full_trace,
    null_trace,
    *,
    full_gate_pass: bool,
    null_gate_pass: bool,
) -> dict[str, Any]:
    """Nested row-level PSIS-LOO comparison with an explicit validity verdict."""
    full = az.loo(full_trace, var_name="y_exit", pointwise=True)
    null = az.loo(null_trace, var_name="y_exit", pointwise=True)
    full_i = np.asarray(full.elpd_i.values, dtype=float).ravel()
    null_i = np.asarray(null.elpd_i.values, dtype=float).ravel()
    if full_i.shape != null_i.shape:
        raise ValueError("full and null pointwise LOO arrays do not align")
    delta_i = full_i - null_i
    elpd_difference = float(np.sum(delta_i))
    difference_se = float(np.sqrt(delta_i.size * np.var(delta_i, ddof=1)))
    full_good_k = float(full.good_k)
    null_good_k = float(null.good_k)
    full_k = np.asarray(full.pareto_k.values, dtype=float).ravel()
    null_k = np.asarray(null.pareto_k.values, dtype=float).ravel()
    full_bad = int(np.sum(full_k > full_good_k))
    null_bad = int(np.sum(null_k > null_good_k))
    comparison_valid = bool(
        full_gate_pass and null_gate_pass and full_bad == 0 and null_bad == 0
    )
    if not full_gate_pass or not null_gate_pass:
        invalid_reason = "one or both fits failed the sampling-quality gate"
    elif full_bad or null_bad:
        invalid_reason = (
            "PSIS-LOO has unreliable points; exact row-level refits are required "
            "before this comparison can support promotion"
        )
    else:
        invalid_reason = ""
    return {
        "loo_unit": "child_period_transition_conditional_on_child_intercept",
        "full_elpd": float(full.elpd),
        "null_elpd": float(null.elpd),
        "elpd_difference_full_minus_null": elpd_difference,
        "difference_se": difference_se,
        "full_max_pareto_k": float(np.max(full_k)),
        "null_max_pareto_k": float(np.max(null_k)),
        "full_good_k": full_good_k,
        "null_good_k": null_good_k,
        "full_unreliable_points": full_bad,
        "null_unreliable_points": null_bad,
        "comparison_valid": comparison_valid,
        "invalid_reason": invalid_reason,
        "evidence_class": (
            "invalid"
            if not comparison_valid
            else "full_discriminating"
            if elpd_difference >= 4
            else "null_discriminating"
            if elpd_difference <= -4
            else "inconclusive"
        ),
    }


def promotion_decision(
    diagnostics: pd.DataFrame,
    comparisons: pd.DataFrame,
    risk_differences: pd.DataFrame,
) -> dict[str, Any]:
    """Apply the decision's pre-specified promotion rule without discretion."""
    primary_key = ("all_words", 0.3)
    rd5 = risk_differences.loc[
        risk_differences["reference_high_words"] == 5.0
    ].copy()
    primary_rows = rd5.loc[
        (rd5["population"] == primary_key[0])
        & (rd5["slope_prior_sd"] == primary_key[1])
    ]
    if len(primary_rows) != 1:
        raise ValueError("promotion table must contain one primary 0-to-5 risk contrast")
    primary_rd = float(primary_rows.iloc[0]["risk_difference_median"])
    all_fits_pass = bool(diagnostics["gate_pass"].all())
    all_comparisons_valid = bool(comparisons["comparison_valid"].all())
    all_comparisons_discriminating = bool(
        (comparisons["elpd_difference_full_minus_null"] >= 4).all()
    )
    primary_material = primary_rd >= 0.10
    direction_stable = bool((rd5["risk_difference_prob_positive"] >= 0.95).all())
    magnitude_stable = bool(
        (rd5["risk_difference_median"] - primary_rd).abs().le(0.10).all()
    )
    checks = {
        "all_eight_fits_pass_computational_gate": all_fits_pass,
        "all_four_loo_comparisons_valid": all_comparisons_valid,
        "all_four_full_models_discriminating_by_at_least_4_elpd": (
            all_comparisons_discriminating
        ),
        "primary_0_to_5_risk_difference_at_least_0_10": primary_material,
        "all_0_to_5_risk_differences_prob_positive_at_least_0_95": direction_stable,
        "sensitivity_0_to_5_medians_within_0_10_of_primary": magnitude_stable,
    }
    promoted = all(checks.values())
    return {
        "schema_version": 1,
        "status": "promote" if promoted else "do_not_promote",
        "promoted": promoted,
        "checks": checks,
        "primary_risk_difference_0_to_5": primary_rd,
        "rule": (
            "Promote only if all fits pass, all comparisons are valid and favour the "
            "full model by at least 4 elpd, the primary 0-to-5-word median risk "
            "difference is at least 0.10, every sensitivity has posterior probability "
            "of a positive difference at least 0.95, and every sensitivity median is "
            "within 0.10 of the primary median."
        ),
    }


def _identity(prepared: PreparedProbe, design: ReferenceDesign) -> dict[str, Any]:
    frame = prepared.frame
    return {
        "population": prepared.label,
        "n_transitions": len(frame),
        "n_children": int(frame["subject_id"].nunique()),
        "n_floor_exits": int(frame["y_exit"].sum()),
        "n_hearing_missing": int(frame["hearing_c"].isna().sum()),
        "n_speech_missing": int(frame["deapp_c"].isna().sum()),
        "row_sha256": prepared.row_sha256,
        "observed_sha256": prepared.observed_sha256,
        "reference_design": asdict(design),
    }


def run_grid(
    *,
    output_dir: Path,
    draws: int,
    tune: int,
    chains: int,
) -> dict[str, Any]:
    """Fit the eight-model promotion grid and persist its audit evidence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    base = transition_frame()
    design = reference_design(base)
    populations = (
        prepare_probe(base, design, label="all_words", max_words=None),
        prepare_probe(base, design, label="words_le_25", max_words=25),
    )
    (output_dir / "analysis_identity.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "row_policy": (
                    "N_pre == 0 with observed W_pre, N_post, age, arm and current "
                    "treatment; hearing/speech mean-filled with missing indicators"
                ),
                "loo_unit": "child-period transition conditional on child intercept",
                "populations": [_identity(population, design) for population in populations],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    diagnostic_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    for population_index, prepared in enumerate(populations):
        for prior_index, prior_sd in enumerate(SLOPE_PRIOR_SDS):
            traces = {}
            gates = {}
            for model_index, (model_name, include_wr) in enumerate(
                (("null", False), ("full", True))
            ):
                seed = SEED + 100 * population_index + 10 * prior_index + model_index
                model, trace = _fit(
                    prepared,
                    include_word_reading=include_wr,
                    slope_prior_sd=prior_sd,
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    seed=seed,
                )
                diagnostics = fit_diagnostics(trace, model)
                traces[model_name] = trace
                gates[model_name] = bool(diagnostics["gate_pass"])
                diagnostic_rows.append(
                    {
                        "population": prepared.label,
                        "slope_prior_sd": prior_sd,
                        "model": model_name,
                        "n_transitions": len(prepared.y),
                        "n_children": len(prepared.subject_labels),
                        "n_floor_exits": int(prepared.y.sum()),
                        "row_sha256": prepared.row_sha256,
                        "observed_sha256": prepared.observed_sha256,
                        **diagnostics,
                    }
                )
                trace.to_netcdf(
                    output_dir
                    / f"trace_{prepared.label}_prior-{prior_sd:g}_{model_name}.nc"
                )
            comparison_rows.append(
                {
                    "population": prepared.label,
                    "slope_prior_sd": prior_sd,
                    "row_sha256": prepared.row_sha256,
                    **loo_comparison(
                        traces["full"],
                        traces["null"],
                        full_gate_pass=gates["full"],
                        null_gate_pass=gates["null"],
                    ),
                }
            )
            for row in risk_difference_summary(traces["full"], prepared, design):
                risk_rows.append(
                    {
                        "population": prepared.label,
                        "slope_prior_sd": prior_sd,
                        "row_sha256": prepared.row_sha256,
                        **slope_summary(traces["full"]),
                        **row,
                    }
                )

    diagnostics_table = pd.DataFrame(diagnostic_rows)
    comparisons_table = pd.DataFrame(comparison_rows)
    risk_table = pd.DataFrame(risk_rows)
    diagnostics_table.to_csv(output_dir / "fit_diagnostics.csv", index=False)
    comparisons_table.to_csv(output_dir / "loo_comparisons.csv", index=False)
    risk_table.to_csv(output_dir / "risk_differences.csv", index=False)
    decision = promotion_decision(diagnostics_table, comparisons_table, risk_table)
    (output_dir / "promotion_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/notes/202608101700-nonword-floor-exit"),
    )
    parser.add_argument("--draws", type=int, default=3000)
    parser.add_argument("--tune", type=int, default=3000)
    parser.add_argument("--chains", type=int, default=4)
    args = parser.parse_args()
    if min(args.draws, args.tune, args.chains) < 1:
        parser.error("draws, tune and chains must be positive")
    decision = run_grid(
        output_dir=args.output_dir,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    print(f"Evidence written to {args.output_dir}")


if __name__ == "__main__":
    main()
