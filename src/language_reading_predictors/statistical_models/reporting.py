# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Post-fit reporting helpers shared across LRP52-LRP58."""

from __future__ import annotations

import json
import os

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import expit

from language_reading_predictors.statistical_models.context import (
    StatisticalFitContext,
)


def tau_summary_itt(
    trace: xr.DataTree,
    *,
    hdi_prob: float,
    pre_logit_mean: float,
    gamma_own_mean: float,
    alpha_mean: float,
) -> dict[str, float]:
    """Summarise the treatment effect ``tau`` on both scales for an ITT model.

    The probability-scale marginal effect is approximated at the sample-mean
    logit-scale baseline, with the fixed-effects intercept and own-baseline
    contribution held at their posterior means. It is a point-estimate
    sensitivity summary - not a causal contrast.
    """
    draws = trace.posterior["tau"].stack(sample=("chain", "draw")).values
    tau_mean = float(np.mean(draws))
    lower, upper = np.quantile(draws, [(1 - hdi_prob) / 2, 1 - (1 - hdi_prob) / 2])

    baseline_eta = alpha_mean + gamma_own_mean * pre_logit_mean
    marginal = expit(baseline_eta + draws) - expit(baseline_eta)
    marg_mean = float(np.mean(marginal))
    marg_lo, marg_hi = np.quantile(
        marginal, [(1 - hdi_prob) / 2, 1 - (1 - hdi_prob) / 2]
    )
    prob_pos = float(np.mean(draws > 0))

    return {
        "tau_logit_mean": tau_mean,
        "tau_logit_lo": float(lower),
        "tau_logit_hi": float(upper),
        "tau_prob_mean": marg_mean,
        "tau_prob_lo": float(marg_lo),
        "tau_prob_hi": float(marg_hi),
        "prob_tau_pos": prob_pos,
    }


def tau_summary_joint(
    trace: xr.DataTree,
    outcomes: list[str],
    hdi_prob: float,
) -> pd.DataFrame:
    """Return a DataFrame summarising tau_k for each outcome (logit scale)."""
    draws = trace.posterior["tau"].stack(sample=("chain", "draw")).values  # (K, n_sample)
    out = []
    lo_q = (1 - hdi_prob) / 2
    hi_q = 1 - lo_q
    for k, s in enumerate(outcomes):
        d = draws[k]
        out.append(
            {
                "outcome": s,
                "tau_mean": float(np.mean(d)),
                "tau_lo": float(np.quantile(d, lo_q)),
                "tau_hi": float(np.quantile(d, hi_q)),
                "prob_pos": float(np.mean(d > 0)),
            }
        )
    return pd.DataFrame(out)


def tau_contrast_matrix(
    trace: xr.DataTree,
    outcomes: list[str],
) -> pd.DataFrame:
    """Compute P(tau_k > tau_j) for every outcome pair."""
    draws = trace.posterior["tau"].stack(sample=("chain", "draw")).values  # (K, n_sample)
    K = draws.shape[0]
    M = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i == j:
                M[i, j] = np.nan
            else:
                M[i, j] = float(np.mean(draws[i] > draws[j]))
    return pd.DataFrame(M, index=outcomes, columns=outcomes)


def write_run_metadata(context: StatisticalFitContext, extra: dict | None = None) -> None:
    """Persist a ``config.json`` and basic metrics for the report."""
    out = context.output_dir
    os.makedirs(out, exist_ok=True)
    spec = context.spec
    cfg = {
        "model_id": spec.model_id,
        "kind": spec.kind,
        "title": spec.title,
        "outcome_symbol": spec.outcome_symbol,
        "mechanism_symbol": spec.mechanism_symbol,
        "adjustment": spec.adjustment,
        "n_obs": context.prepared.n_obs if context.prepared else None,
        "n_children": context.prepared.n_children if context.prepared else None,
        "n_phases": context.prepared.n_phases if context.prepared else None,
        "dropped_rows": context.prepared.dropped_rows if context.prepared else None,
        "hdi": context.reporting.hdi,
        "sampling": {
            "draws": context.sampling.draws,
            "tune": context.sampling.tune,
            "chains": context.sampling.chains,
            "target_accept": context.sampling.target_accept,
            "random_seed": context.sampling.random_seed,
        },
        "extra": extra or {},
    }
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)


def write_loo_summary(context: StatisticalFitContext) -> None:
    if context.loo is None:
        return
    out = context.output_dir
    path = os.path.join(out, "loo.txt")
    with open(path, "w") as f:
        f.write(str(context.loo))


def loo_delta(loo_a: az.ELPDData, loo_b: az.ELPDData) -> dict[str, float]:
    """Delta-ELPD between two models using ArviZ compare."""
    df = az.compare({"a": loo_a, "b": loo_b})
    return {
        "d_elpd": float(df.loc["a", "elpd_loo"] - df.loc["b", "elpd_loo"]),
        "d_se": float(df.loc["a", "dse"] if "dse" in df.columns else np.nan),
    }
