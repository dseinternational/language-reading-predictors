# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Horseshoe calculations and summaries."""

from __future__ import annotations
import arviz as az
import numpy as np
import pandas as pd
import xarray as xr


def horseshoe_ranking(trace: xr.DataTree, *, delta: float = 0.1) -> pd.DataFrame:
    """Per-predictor ranking from a horseshoe fit (LRPHS, #116 Phase E).

    One row per predictor: ``p_abs_gt_delta`` = posterior ``P(|beta_k| > delta)``
    (the ranking key), the posterior median/mean/sd and 89% HDI (``beta_hdi_lo`` /
    ``beta_hdi_hi``, an actual highest-density interval via :func:`arviz.hdi`, not
    equal-tailed percentiles) of the standardised coefficient, its ``sign``, and
    ``lambda_mean`` (mean local shrinkage — small ⇒ shrunk toward zero). ``delta``
    is on the logit / per-SD scale (the minimally-interesting coefficient). Ranked
    by ``p_abs_gt_delta`` descending — the horseshoe analogue of the GB
    permutation-importance order.
    """
    posterior = trace.posterior
    beta = posterior["beta"]  # (chain, draw, predictor)
    predictors = [str(p) for p in beta.coords["predictor"].values]
    lam = posterior["hs_lambda"] if "hs_lambda" in posterior else None
    rows = []
    for i, name in enumerate(predictors):
        b = beta.isel(predictor=i).stack(sample=("chain", "draw")).values  # (S,)
        mean = float(np.mean(b))
        median = float(np.median(b))
        hdi = np.asarray(az.hdi(b, prob=0.89))  # 89% highest-density interval
        row = {
            "predictor": name,
            "p_abs_gt_delta": float(np.mean(np.abs(b) > delta)),
            "beta_median": median,
            "beta_mean": mean,
            "beta_sd": float(np.std(b)),
            "beta_hdi_lo": float(hdi[0]),
            "beta_hdi_hi": float(hdi[1]),
            # Direction from the median — the house lead statistic, and the same
            # statistic the key-findings box reads, so the CSV and the box cannot
            # disagree on a spike-and-slab posterior whose mean and median
            # straddle zero (2026-08-21 review, finding 10).
            "sign": "+" if median > 0 else ("-" if median < 0 else "0"),
        }
        if lam is not None:
            row["lambda_mean"] = float(lam.isel(predictor=i).stack(sample=("chain", "draw")).values.mean())
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("p_abs_gt_delta", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df
