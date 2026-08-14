# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pre-fit recovery study for the Byrne/RLM reciprocal LCSM question.

This module is deliberately not a registered model family.  It supplies the
real observation design, a generative four-process latent change-score model,
the matching PyMC recovery model, and an explicit go/no-go rule.  The command
line harness lives in ``scripts/simulate_rlm_lcsm_feasibility.py``.

The two candidates differ only in population and mean structure:

``ds``
    Down-syndrome children only, with transition-specific intercepts.
``three_group``
    All three cohorts, with group-by-transition intercepts and couplings shared
    across groups and transitions.

Both use the paper-compatible first three annual waves and the repository's
actual cell-level missingness.  The three reverse couplings are adjusted
predictive associations, never causal effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy.special import expit, logit

from language_reading_predictors.statistical_models.datasets import (
    RLM_DATASET,
    RLM_GROUP_LABELS,
    RLM_MEASURES,
)
from language_reading_predictors.statistical_models.likelihood import (
    beta_binomial_from_logit,
)

CandidateScope = Literal["ds", "three_group"]

OUTCOMES = ("basread", "bpvs", "trog", "basdig")
REVERSE_EDGES = (
    ("basread", "bpvs"),
    ("basread", "trog"),
    ("basread", "basdig"),
)
FORWARD_EDGES = (
    ("basdig", "bpvs"),
    ("bpvs", "trog"),
    ("basdig", "basread"),
    ("bpvs", "basread"),
)
MODEL_EDGES = FORWARD_EDGES + REVERSE_EDGES
PRIMARY_WAVES = (1, 2, 3)


def edge_name(source: str, target: str) -> str:
    """Return the stable PyMC parameter name for one directed coupling."""

    return f"g_{source}_{target}"


@dataclass(frozen=True, slots=True)
class RlmFeasibilityDesign:
    """Fixed real-data design used by every simulated replicate."""

    scope: CandidateScope
    subject_ids: np.ndarray
    group_codes: np.ndarray
    group_index: np.ndarray
    group_labels: tuple[str, ...]
    waves: np.ndarray
    counts: np.ndarray
    mask: np.ndarray
    logits: np.ndarray
    age_std: np.ndarray
    n_trials: np.ndarray
    anchors: np.ndarray
    sigma_initial: np.ndarray
    correlation_initial: np.ndarray
    sigma_process: np.ndarray
    data_path: Path

    @property
    def n_children(self) -> int:
        return int(self.subject_ids.size)

    @property
    def n_groups(self) -> int:
        return len(self.group_labels)

    @property
    def n_waves(self) -> int:
        return int(self.waves.size)

    def metadata(self) -> dict[str, Any]:
        """Return a JSON-ready audit record for the simulation output."""

        return {
            "scope": self.scope,
            "n_children": self.n_children,
            "n_groups": self.n_groups,
            "group_labels": list(self.group_labels),
            "waves": self.waves.tolist(),
            "outcomes": list(OUTCOMES),
            "n_trials": dict(zip(OUTCOMES, self.n_trials.tolist(), strict=True)),
            "observed_cells": {
                symbol: int(self.mask[:, :, index].sum())
                for index, symbol in enumerate(OUTCOMES)
            },
            "complete_all_measure_children": int(self.mask.all(axis=(1, 2)).sum()),
            "initial_correlation": self.correlation_initial.tolist(),
            "data_path": str(self.data_path),
        }


@dataclass(frozen=True, slots=True)
class RlmSimulationTruth:
    """Parameters used to generate one simulation scenario."""

    reverse_strength: float
    mu_initial: np.ndarray
    change_intercept: np.ndarray
    self_feedback: np.ndarray
    age_slope: np.ndarray
    coupling_matrix: np.ndarray
    sigma_initial: np.ndarray
    correlation_initial: np.ndarray
    sigma_process: np.ndarray
    kappa: np.ndarray

    def coupling(self, source: str, target: str) -> float:
        index = {symbol: position for position, symbol in enumerate(OUTCOMES)}
        return float(self.coupling_matrix[index[target], index[source]])


@dataclass(frozen=True, slots=True)
class FeasibilityCriteria:
    """Pre-specified Monte Carlo gate for one candidate design."""

    alternative_strength: float = 0.10
    posterior_support_threshold: float = 0.90
    min_fit_success_rate: float = 0.95
    min_zero_divergence_rate: float = 0.95
    max_abs_bias: float = 0.05
    min_coverage_89: float = 0.75
    max_coverage_89: float = 1.00
    min_support_rate: float = 0.80
    max_null_support_rate: float = 0.15


def _pivot(
    frame: pd.DataFrame,
    *,
    subject_ids: np.ndarray,
    waves: np.ndarray,
    column: str,
) -> np.ndarray:
    wide = frame.pivot(index="subject_id", columns="time", values=column)
    return wide.reindex(index=subject_ids, columns=waves).to_numpy(dtype=float)


def _corrected_logit(counts: np.ndarray, n_trials: int) -> np.ndarray:
    out = np.full_like(counts, np.nan, dtype=float)
    present = np.isfinite(counts)
    out[present] = logit((counts[present] + 0.5) / (n_trials + 1.0))
    return out


def load_rlm_feasibility_design(
    scope: CandidateScope,
    *,
    path: str | Path | None = None,
) -> RlmFeasibilityDesign:
    """Load the first-three-wave RLM design without inspecting outcome findings."""

    if scope not in {"ds", "three_group"}:
        raise ValueError("scope must be 'ds' or 'three_group'")
    data_path = Path(path) if path is not None else RLM_DATASET.path
    frame = pd.read_csv(data_path)
    frame = frame.loc[frame["time"].isin(PRIMARY_WAVES)].copy()
    if scope == "ds":
        frame = frame.loc[frame["readgrp"] == 1].copy()

    subject_ids = np.sort(frame["subject_id"].unique())
    waves = np.asarray(PRIMARY_WAVES, dtype=int)
    group_wide = _pivot(
        frame,
        subject_ids=subject_ids,
        waves=waves,
        column="readgrp",
    )
    group_codes = group_wide[:, 0].astype(int)
    if not np.all(group_wide == group_codes[:, None]):
        raise ValueError("readgrp must be observed and constant within child")
    unique_codes = np.sort(np.unique(group_codes))
    if scope == "ds" and unique_codes.tolist() != [1]:
        raise ValueError("the Down-syndrome candidate must contain readgrp=1 only")
    expected = [1, 2, 3] if scope == "three_group" else [1]
    if unique_codes.tolist() != expected:
        raise ValueError(f"unexpected group codes: {unique_codes.tolist()}")
    code_to_index = {code: index for index, code in enumerate(unique_codes)}
    group_index = np.asarray([code_to_index[code] for code in group_codes], dtype=int)
    group_labels = tuple(RLM_GROUP_LABELS[int(code)] for code in unique_codes)

    n_trials = np.asarray([RLM_MEASURES[symbol].n_trials for symbol in OUTCOMES])
    if not all(RLM_MEASURES[symbol].n_trials_confirmed for symbol in OUTCOMES):
        raise ValueError("all feasibility outcomes require confirmed denominators")
    count_arrays = [
        _pivot(frame, subject_ids=subject_ids, waves=waves, column=symbol)
        for symbol in OUTCOMES
    ]
    counts = np.stack(count_arrays, axis=2)
    mask = np.isfinite(counts)
    logits = np.stack(
        [
            _corrected_logit(counts[:, :, index], int(n_trials[index]))
            for index in range(len(OUTCOMES))
        ],
        axis=2,
    )

    age = _pivot(frame, subject_ids=subject_ids, waves=waves, column="age")
    age = (
        pd.DataFrame(age, columns=waves)
        .interpolate(axis=1, limit_direction="both")
        .to_numpy(dtype=float)
    )
    if not np.isfinite(age).all():
        raise ValueError("age must be recoverable for every child-wave cell")
    age_std = (age - age.mean()) / age.std(ddof=0)

    anchors = np.empty((unique_codes.size, waves.size, len(OUTCOMES)))
    for group in range(unique_codes.size):
        for wave in range(waves.size):
            for outcome in range(len(OUTCOMES)):
                cells = logits[group_index == group, wave, outcome]
                cells = cells[np.isfinite(cells)]
                if not cells.size:
                    raise ValueError(
                        "every group-wave-outcome cell needs an observed anchor"
                    )
                anchors[group, wave, outcome] = cells.mean()

    initial_residuals: list[list[float]] = [[] for _ in OUTCOMES]
    process_residuals: list[list[float]] = [[] for _ in OUTCOMES]
    for child in range(subject_ids.size):
        group = group_index[child]
        for outcome in range(len(OUTCOMES)):
            value = logits[child, 0, outcome]
            if np.isfinite(value):
                initial_residuals[outcome].append(value - anchors[group, 0, outcome])
            for wave in range(waves.size - 1):
                left = logits[child, wave, outcome]
                right = logits[child, wave + 1, outcome]
                if np.isfinite(left) and np.isfinite(right):
                    observed_change = right - left
                    anchor_change = (
                        anchors[group, wave + 1, outcome]
                        - anchors[group, wave, outcome]
                    )
                    process_residuals[outcome].append(observed_change - anchor_change)

    sigma_initial = np.asarray(
        [np.clip(np.std(values, ddof=1), 0.30, 1.20) for values in initial_residuals]
    )
    wave_one = logits[:, 0, :].copy()
    wave_one = wave_one - anchors[group_index, 0, :]
    complete_wave_one = wave_one[np.isfinite(wave_one).all(axis=1)]
    empirical_correlation = np.corrcoef(complete_wave_one, rowvar=False)
    # Shrink the small-sample correlation toward independence before using it as
    # a simulation truth.  This retains the stable common-cause signal without
    # pretending the empirical 4x4 matrix is known exactly.
    correlation_initial = 0.50 * empirical_correlation + 0.50 * np.eye(len(OUTCOMES))
    eigenvalues, eigenvectors = np.linalg.eigh(correlation_initial)
    correlation_initial = (eigenvectors * np.clip(eigenvalues, 1e-6, None)) @ eigenvectors.T
    scale = np.sqrt(np.diag(correlation_initial))
    correlation_initial = correlation_initial / np.outer(scale, scale)
    sigma_process = np.asarray(
        [np.clip(np.std(values, ddof=1), 0.25, 0.80) for values in process_residuals]
    )

    return RlmFeasibilityDesign(
        scope=scope,
        subject_ids=subject_ids,
        group_codes=group_codes,
        group_index=group_index,
        group_labels=group_labels,
        waves=waves,
        counts=counts,
        mask=mask,
        logits=logits,
        age_std=age_std,
        n_trials=n_trials,
        anchors=anchors,
        sigma_initial=sigma_initial,
        correlation_initial=correlation_initial,
        sigma_process=sigma_process,
        data_path=data_path,
    )


def simulation_truth(
    design: RlmFeasibilityDesign,
    *,
    reverse_strength: float,
) -> RlmSimulationTruth:
    """Construct a realistic truth while preserving empirical mean trajectories."""

    if not np.isfinite(reverse_strength) or reverse_strength < 0:
        raise ValueError("reverse_strength must be a non-negative finite number")
    index = {symbol: position for position, symbol in enumerate(OUTCOMES)}
    coupling = np.zeros((len(OUTCOMES), len(OUTCOMES)), dtype=float)
    for source, target in FORWARD_EDGES:
        coupling[index[target], index[source]] = 0.08
    for source, target in REVERSE_EDGES:
        coupling[index[target], index[source]] = reverse_strength

    self_feedback = np.full(len(OUTCOMES), -0.30)
    age_slope = np.full(len(OUTCOMES), 0.05)
    mean_age = np.empty((design.n_groups, design.n_waves))
    for group in range(design.n_groups):
        mean_age[group] = design.age_std[design.group_index == group].mean(axis=0)

    change_intercept = np.empty(
        (design.n_groups, design.n_waves - 1, len(OUTCOMES))
    )
    for group in range(design.n_groups):
        for transition in range(design.n_waves - 1):
            previous = design.anchors[group, transition]
            change_intercept[group, transition] = (
                design.anchors[group, transition + 1]
                - previous
                - self_feedback * previous
                - coupling @ previous
                - age_slope * mean_age[group, transition]
            )

    return RlmSimulationTruth(
        reverse_strength=float(reverse_strength),
        mu_initial=design.anchors[:, 0, :].copy(),
        change_intercept=change_intercept,
        self_feedback=self_feedback,
        age_slope=age_slope,
        coupling_matrix=coupling,
        sigma_initial=design.sigma_initial.copy(),
        correlation_initial=design.correlation_initial.copy(),
        sigma_process=design.sigma_process.copy(),
        kappa=np.full(len(OUTCOMES), 30.0),
    )


def simulate_rlm_lcsm_counts(
    design: RlmFeasibilityDesign,
    truth: RlmSimulationTruth,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate bounded scores, returning counts and their latent logit states."""

    n_outcomes = len(OUTCOMES)
    latent = np.empty((design.n_children, design.n_waves, n_outcomes))
    covariance_initial = (
        truth.sigma_initial[:, None]
        * truth.correlation_initial
        * truth.sigma_initial[None, :]
    )
    cholesky_initial = np.linalg.cholesky(covariance_initial)
    latent[:, 0, :] = (
        truth.mu_initial[design.group_index]
        + rng.normal(size=(design.n_children, n_outcomes)) @ cholesky_initial.T
    )
    for transition in range(design.n_waves - 1):
        previous = latent[:, transition, :]
        mean_change = (
            truth.change_intercept[design.group_index, transition]
            + previous * truth.self_feedback
            + previous @ truth.coupling_matrix.T
            + design.age_std[:, transition, None] * truth.age_slope
        )
        innovation = (
            rng.normal(size=(design.n_children, n_outcomes)) * truth.sigma_process
        )
        latent[:, transition + 1, :] = previous + mean_change + innovation

    probability = np.clip(expit(latent), 1e-6, 1.0 - 1e-6)
    counts = np.empty_like(latent, dtype=np.int64)
    for outcome in range(n_outcomes):
        alpha = probability[:, :, outcome] * truth.kappa[outcome]
        beta = (1.0 - probability[:, :, outcome]) * truth.kappa[outcome]
        beta_probability = rng.beta(alpha, beta)
        counts[:, :, outcome] = rng.binomial(
            int(design.n_trials[outcome]), beta_probability
        )
    return counts, latent


def build_rlm_lcsm_recovery_model(
    design: RlmFeasibilityDesign,
    counts: np.ndarray,
) -> pm.Model:
    """Build the exact four-process model fitted to each simulated dataset."""

    if counts.shape != design.counts.shape:
        raise ValueError(
            f"counts shape {counts.shape} does not match design {design.counts.shape}"
        )
    n_outcomes = len(OUTCOMES)
    idx_i, idx_t, idx_k = np.nonzero(design.mask)
    observed = counts[idx_i, idx_t, idx_k].astype(np.int64)
    coords = {
        "child": design.subject_ids,
        "wave": design.waves,
        "trans": design.waves[1:],
        "outcome": list(OUTCOMES),
        "group": list(design.group_labels),
        "y_obs_cell": np.arange(observed.size),
    }
    outcome_index = {symbol: position for position, symbol in enumerate(OUTCOMES)}

    with pm.Model(coords=coords) as model:
        age = pm.Data("age_std", design.age_std, dims=("child", "wave"))
        group = pm.Data("group_index", design.group_index, dims="child")
        y_data = pm.Data("y_data", observed, dims="y_obs_cell")

        mu_initial = pm.Normal(
            "mu_initial",
            mu=design.anchors[:, 0, :],
            sigma=0.75,
            dims=("group", "outcome"),
        )
        initial_cholesky, _, _ = pm.LKJCholeskyCov(
            "initial_cholesky",
            n=n_outcomes,
            eta=2.0,
            sd_dist=pm.HalfNormal.dist(1.0, shape=n_outcomes),
            compute_corr=True,
        )
        z_initial = pm.Normal("z_initial", 0.0, 1.0, dims=("child", "outcome"))
        change_intercept = pm.Normal(
            "change_intercept",
            0.0,
            1.0,
            dims=("group", "trans", "outcome"),
        )
        self_feedback = pm.Normal(
            "self_feedback", -0.30, 0.20, dims="outcome"
        )
        age_slope = pm.Normal("age_slope", 0.0, 0.30, dims="outcome")
        coupling = {
            (source, target): pm.Normal(edge_name(source, target), 0.0, 0.30)
            for source, target in MODEL_EDGES
        }
        sigma_process = pm.HalfNormal("sigma_process", 0.50, dims="outcome")
        z_process = pm.Normal(
            "z_process",
            0.0,
            1.0,
            dims=("child", "trans", "outcome"),
        )
        kappa = pm.HalfNormal("kappa", 50.0, dims="outcome")

        states: list[pt.TensorVariable] = [
            mu_initial[group] + z_initial @ initial_cholesky.T
        ]
        for transition in range(design.n_waves - 1):
            previous = states[-1]
            mean_change = (
                change_intercept[group, transition, :]
                + previous * self_feedback
                + age[:, transition, None] * age_slope
            )
            for source, target in MODEL_EDGES:
                mean_change = pt.set_subtensor(
                    mean_change[:, outcome_index[target]],
                    mean_change[:, outcome_index[target]]
                    + coupling[(source, target)] * previous[:, outcome_index[source]],
                )
            states.append(
                previous
                + mean_change
                + z_process[:, transition, :] * sigma_process
            )
        latent = pm.Deterministic(
            "latent_state",
            pt.stack(states, axis=1),
            dims=("child", "wave", "outcome"),
        )
        linear = np.ravel_multi_index(
            (idx_i, idx_t, idx_k),
            (design.n_children, design.n_waves, n_outcomes),
        )
        beta_binomial_from_logit(
            "y_obs",
            latent.reshape((-1,))[linear],
            n_trials=design.n_trials[idx_k],
            kappa=kappa[idx_k],
            observed=y_data,
            dims="y_obs_cell",
        )
    return model


def recovery_rows(
    idata: az.InferenceData,
    *,
    scope: CandidateScope,
    simulation: int,
    truth: RlmSimulationTruth,
    ci_prob: float = 0.89,
    support_threshold: float = 0.90,
) -> list[dict[str, Any]]:
    """Extract one auditable recovery row per pre-specified reverse edge."""

    tail = (1.0 - ci_prob) / 2.0
    divergences = int(np.asarray(idata.sample_stats["diverging"]).sum())
    rows: list[dict[str, Any]] = []
    for source, target in REVERSE_EDGES:
        name = edge_name(source, target)
        samples = np.asarray(idata.posterior[name]).reshape(-1)
        lower, median, upper = np.quantile(samples, [tail, 0.5, 1.0 - tail])
        true_value = truth.coupling(source, target)
        p_positive = float(np.mean(samples > 0.0))
        rows.append(
            {
                "scope": scope,
                "simulation": simulation,
                "reverse_strength": truth.reverse_strength,
                "source": source,
                "target": target,
                "parameter": name,
                "true_value": true_value,
                "median": float(median),
                "lower_89": float(lower),
                "upper_89": float(upper),
                "p_positive": p_positive,
                "covered_89": bool(lower <= true_value <= upper),
                "interval_positive": bool(lower > 0.0),
                "supported_positive": bool(p_positive >= support_threshold),
                "divergences": divergences,
            }
        )
    return rows


def aggregate_recovery(
    rows: pd.DataFrame,
    *,
    attempted: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate replicate rows into bias, coverage, support, and sampling rates."""

    required = {
        "scope",
        "simulation",
        "reverse_strength",
        "parameter",
        "true_value",
        "median",
        "covered_89",
        "supported_positive",
        "divergences",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"recovery rows missing columns: {sorted(missing)}")
    grouped = rows.groupby(
        ["scope", "reverse_strength", "parameter"], sort=True, observed=True
    )
    summary = grouped.agg(
        n_fitted=("simulation", "nunique"),
        true_value=("true_value", "first"),
        mean_median=("median", "mean"),
        coverage_89=("covered_89", "mean"),
        support_rate=("supported_positive", "mean"),
        zero_divergence_rate=("divergences", lambda values: float(np.mean(values == 0))),
    ).reset_index()
    summary["bias"] = summary["mean_median"] - summary["true_value"]
    if attempted is not None:
        attempts = (
            attempted.groupby(["scope", "reverse_strength"], observed=True)
            .agg(n_attempted=("simulation", "nunique"))
            .reset_index()
        )
        summary = summary.merge(attempts, on=["scope", "reverse_strength"], how="left")
        summary["fit_success_rate"] = summary["n_fitted"] / summary["n_attempted"]
    else:
        summary["n_attempted"] = summary["n_fitted"]
        summary["fit_success_rate"] = 1.0
    return summary


def evaluate_candidate(
    summary: pd.DataFrame,
    scope: CandidateScope,
    *,
    criteria: FeasibilityCriteria = FeasibilityCriteria(),
) -> dict[str, Any]:
    """Apply the pre-specified all-edges go/no-go rule to one candidate."""

    candidate = summary.loc[summary["scope"] == scope].copy()
    expected = {edge_name(source, target) for source, target in REVERSE_EDGES}
    failures: list[str] = []
    checks: list[dict[str, Any]] = []
    for parameter in sorted(expected):
        null = candidate.loc[
            (candidate["parameter"] == parameter)
            & np.isclose(candidate["reverse_strength"], 0.0)
        ]
        alternative = candidate.loc[
            (candidate["parameter"] == parameter)
            & np.isclose(
                candidate["reverse_strength"], criteria.alternative_strength
            )
        ]
        if len(null) != 1 or len(alternative) != 1:
            failures.append(f"{parameter}: missing null or alternative scenario")
            continue
        nrow = null.iloc[0]
        arow = alternative.iloc[0]
        values = {
            "parameter": parameter,
            "fit_success_rate": float(
                min(nrow["fit_success_rate"], arow["fit_success_rate"])
            ),
            "zero_divergence_rate": float(
                min(nrow["zero_divergence_rate"], arow["zero_divergence_rate"])
            ),
            "abs_bias": abs(float(arow["bias"])),
            "coverage_89": float(arow["coverage_89"]),
            "support_rate": float(arow["support_rate"]),
            "null_support_rate": float(nrow["support_rate"]),
        }
        checks.append(values)
        conditions = {
            "fit success": values["fit_success_rate"] >= criteria.min_fit_success_rate,
            "zero divergences": values["zero_divergence_rate"]
            >= criteria.min_zero_divergence_rate,
            "bias": values["abs_bias"] <= criteria.max_abs_bias,
            "coverage": criteria.min_coverage_89
            <= values["coverage_89"]
            <= criteria.max_coverage_89,
            "positive support": values["support_rate"] >= criteria.min_support_rate,
            "null calibration": values["null_support_rate"]
            <= criteria.max_null_support_rate,
        }
        failures.extend(
            f"{parameter}: {name} failed" for name, passed in conditions.items() if not passed
        )
    missing_parameters = expected - set(candidate["parameter"])
    failures.extend(f"{parameter}: absent from summary" for parameter in missing_parameters)
    return {
        "scope": scope,
        "decision": "go" if not failures else "no_go",
        "criteria": asdict(criteria),
        "checks": checks,
        "failures": failures,
    }
