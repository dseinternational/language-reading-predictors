# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Quantitative posterior-predictive calibration for bounded ITT scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from language_reading_predictors.statistical_models.preprocessing import PreparedData


def _baseline_bands(values: np.ndarray) -> np.ndarray:
    """Return stable low/middle/high labels, retaining missing baselines."""

    values = np.asarray(values, dtype=float)
    labels = np.full(values.size, "baseline_missing", dtype=object)
    finite = np.isfinite(values)
    if not finite.any():
        return labels
    observed = values[finite]
    if np.unique(observed).size == 1:
        labels[finite] = "baseline_all"
        return labels

    codes = pd.qcut(observed, q=3, labels=False, duplicates="drop")
    codes = np.asarray(codes, dtype=int)
    n_bands = int(codes.max()) + 1
    names = (
        ["baseline_low", "baseline_high"]
        if n_bands == 2
        else ["baseline_low", "baseline_middle", "baseline_high"]
    )
    labels[finite] = np.asarray(names, dtype=object)[codes]
    return labels


def _metric_summary(
    replicated: np.ndarray,
    observed: float,
    *,
    prefix: str,
    ci_prob: float,
) -> dict[str, float | bool]:
    alpha = (1.0 - ci_prob) / 2.0
    lo = float(np.quantile(replicated, alpha))
    hi = float(np.quantile(replicated, 1.0 - alpha))
    upper_tail = float(np.mean(replicated >= observed))
    lower_tail = float(np.mean(replicated <= observed))
    return {
        f"observed_{prefix}": float(observed),
        f"ppc_{prefix}_median": float(np.median(replicated)),
        f"ppc_{prefix}_lo": lo,
        f"ppc_{prefix}_hi": hi,
        f"ppc_{prefix}_two_sided_tail": min(1.0, 2.0 * min(upper_tail, lower_tail)),
        f"ppc_{prefix}_outside_interval": bool(observed < lo or observed > hi),
    }


def score_ppc_by_arm_and_baseline(
    prepared: PreparedData,
    outcome_symbol: str,
    replicated_counts: np.ndarray,
    *,
    row_indices: np.ndarray | None = None,
    observed_counts: np.ndarray | None = None,
    n_trials: int | None = None,
    ci_prob: float = 0.95,
) -> pd.DataFrame:
    """Summarise replicated score location, floor and ceiling by trial arm/band.

    ``replicated_counts`` has arbitrary sample dimensions followed by one row
    dimension. ``row_indices`` maps that last dimension to ``prepared`` (needed
    for flattened joint likelihoods); it defaults to every prepared row.
    ``observed_counts`` can replace the prepared score, as for the Bernoulli
    post-hoc off-floor exploratory headline. All metrics use proportion correct, so outcomes with
    different denominators remain interpretable.
    """

    if not 0 < ci_prob < 1:
        raise ValueError("ci_prob must be in (0, 1)")
    replicated = np.asarray(replicated_counts, dtype=float)
    if replicated.ndim < 2:
        raise ValueError("replicated_counts needs sample and row dimensions")
    replicated = replicated.reshape(-1, replicated.shape[-1])
    if row_indices is None:
        row_indices = np.arange(prepared.n_obs)
    row_indices = np.asarray(row_indices, dtype=int)
    if row_indices.ndim != 1 or row_indices.size != replicated.shape[1]:
        raise ValueError("row_indices must align with replicated predictive cells")
    if np.any((row_indices < 0) | (row_indices >= prepared.n_obs)):
        raise ValueError("row_indices contains an out-of-range prepared row")

    if observed_counts is None:
        observed = np.asarray(prepared.post_counts[outcome_symbol], dtype=float)[
            row_indices
        ]
    else:
        observed = np.asarray(observed_counts, dtype=float)
    if observed.shape != (row_indices.size,):
        raise ValueError("observed_counts must have one value per predictive cell")

    denominator = int(
        prepared.n_trials[outcome_symbol] if n_trials is None else n_trials
    )
    if denominator <= 0:
        raise ValueError("n_trials must be positive")
    finite = np.isfinite(observed)
    if np.any((observed[finite] < 0) | (observed[finite] > denominator)):
        raise ValueError("observed scores lie outside the requested denominator")

    G = np.asarray(prepared.G, dtype=int)[row_indices]
    baseline = np.asarray(prepared.pre_logit[outcome_symbol], dtype=float)[row_indices]
    bands = _baseline_bands(baseline)
    rows = []
    for g, arm in ((1, "intervention"), (0, "control")):
        for band in dict.fromkeys(bands[G == g]):
            mask = (G == g) & (bands == band) & finite
            if not mask.any():
                continue
            obs = observed[mask] / denominator
            rep = replicated[:, mask] / denominator
            rows.append(
                {
                    "outcome": outcome_symbol,
                    "arm": arm,
                    "baseline_band": str(band),
                    "n": int(mask.sum()),
                    **_metric_summary(
                        rep.mean(axis=1),
                        float(obs.mean()),
                        prefix="mean_proportion",
                        ci_prob=ci_prob,
                    ),
                    **_metric_summary(
                        (rep == 0).mean(axis=1),
                        float(np.mean(obs == 0)),
                        prefix="floor_rate",
                        ci_prob=ci_prob,
                    ),
                    **_metric_summary(
                        (rep == 1).mean(axis=1),
                        float(np.mean(obs == 1)),
                        prefix="ceiling_rate",
                        ci_prob=ci_prob,
                    ),
                }
            )
    return pd.DataFrame(rows)
