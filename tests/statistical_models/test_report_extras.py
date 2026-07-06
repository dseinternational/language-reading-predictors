# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the #125 reporting additions: prior pushforward, ROPE markdown,
and the evidence-label rollout into the plain summary cards."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from language_reading_predictors.statistical_models.reporting import (
    evidence_label,
    factor_summary,
    prior_pushforward,
    rope_markdown,
    tau_summary_itt,
)


def _ds(eta, tau, *, extra=None):
    n_chain, n_draw, n_obs = eta.shape
    data = {
        "eta": (("chain", "draw", "obs_id"), eta),
        "tau": (("chain", "draw"), tau),
    }
    for k, v in (extra or {}).items():
        data[k] = (("chain", "draw"), v)
    return xr.Dataset(
        data,
        coords={
            "chain": np.arange(n_chain),
            "draw": np.arange(n_draw),
            "obs_id": np.arange(n_obs),
        },
    )


def test_prior_pushforward_reads_prior_group():
    rng = np.random.default_rng(1)
    eta = rng.normal(0.0, 1.0, (1, 500, 10))
    tau = rng.normal(0.0, 0.5, (1, 500))
    G = (rng.random(10) > 0.5).astype(float)
    trace = SimpleNamespace(prior=_ds(eta, tau))
    out = prior_pushforward(trace, G=G, n_trials=79, ci_prob=0.95)
    assert out["n_trials"] == 79
    # Prior centred at 0 -> items median near 0; 95% range wide but finite.
    assert abs(out["prior_items_median"]) < 5
    assert out["prior_items_lo"] < out["prior_items_hi"]
    assert out["prior_items_lo50"] >= out["prior_items_lo"]
    assert out["prior_items_hi50"] <= out["prior_items_hi"]


def test_tau_summary_itt_has_direction_label():
    rng = np.random.default_rng(2)
    eta = rng.normal(0.0, 1.0, (2, 300, 8))
    tau = rng.normal(0.5, 0.2, (2, 300))
    G = (rng.random(8) > 0.5).astype(float)
    out = tau_summary_itt(SimpleNamespace(posterior=_ds(eta, tau)), ci_prob=0.95, G=G)
    assert "direction_label" in out
    assert out["direction_label"] == evidence_label(out["prob_tau_pos"])


def test_factor_summary_has_direction_label():
    rng = np.random.default_rng(3)
    beta = rng.normal(0.3, 0.2, (2, 300))
    ds = xr.Dataset(
        {"beta_trt": (("chain", "draw"), beta)},
        coords={"chain": np.arange(2), "draw": np.arange(300)},
    )
    df = factor_summary(
        SimpleNamespace(posterior=ds), ["beta_trt"], ci_prob=0.95, causal_terms=("beta_trt",)
    )
    assert "direction_label" in df.columns
    row = df.iloc[0]
    assert row["role"] == "causal"
    assert row["median"] == np.median(beta)
    assert row["direction_label"] == evidence_label(row["prob_positive"])


def test_rope_markdown_items_and_risk_difference():
    base = {
        "tau_logit_median": 0.3, "tau_logit_lo50": 0.1, "tau_logit_hi50": 0.5,
        "tau_logit_lo": -0.1, "tau_logit_hi": 0.7,
        "items_median": 2.0, "items_lo50": 1.0, "items_hi50": 3.0,
        "items_lo": -0.5, "items_hi": 4.5, "delta_items": 1.0,
        "pd": 0.97, "prob_benefit_ge_delta": 0.85, "prob_in_rope": 0.05,
        "prob_harm_ge_delta": 0.01, "direction_label": "moderate",
        "benefit_label": "suggestive",
    }
    md = rope_markdown(pd.DataFrame([base]), "letter sounds")
    assert "letter sounds" in md
    assert "+2.0 items" in md
    assert "δ" in md

    # Floored risk-difference scale: median ×100 to percentage points, flagged provisional.
    rd = dict(base, delta_scale="risk_difference", provisional_delta=True, items_median=0.12,
              items_lo50=0.05, items_hi50=0.20, items_lo=-0.02, items_hi=0.30, delta_items=0.10)
    md2 = rope_markdown(pd.DataFrame([rd]), "P(off-floor)")
    assert "percentage points" in md2
    assert "provisional" in md2
    assert "+12.0" in md2
