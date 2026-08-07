# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Statistical validation of the gain-factor off-floor parameterisation (#391).

The #391 finding 2 decision (2026-07-22) replaced the off-floor path's own-baseline
handling with an always-on **binary off-floor-at-pre indicator** main effect
(``gamma_own_offfloor``). This module recovers that parameterisation from simulated
data through ``build_gain_factors_model`` itself with the real sampler — the
acceptance criterion asks for exactly this: the floor-rule baseline
parameterisation documented AND covered by a recovery test.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit as _expit


def _prepared_offfloor_panel(
    *,
    n_children_per_arm: int,
    truth: dict[str, float | np.ndarray],
    n_trials: int,
    seed: int,
):
    """A PreparedData period-stacked panel simulated from the off-floor DGP.

    Children carry a Normal(0, sigma_child) random intercept. Each child-period
    row has a pre count (zero-inflated, so the off-floor-at-pre indicator has
    both classes in both arms) and a Bernoulli off-floor-at-post outcome whose
    logit follows the factory's linear predictor: intercept + period offsets +
    the indicator main effect + the on-intervention term + the intercept. Ages
    are outcome-independent, so the fitted precision term has a true
    coefficient of zero.
    """
    from language_reading_predictors.statistical_models.preprocessing import (
        PreparedData,
        standardise,
    )

    rng = np.random.default_rng(seed)
    n_children = 2 * n_children_per_arm
    child_g = np.repeat([1.0, 0.0], n_children_per_arm)
    u_child = rng.normal(0.0, truth["sigma_child"], size=n_children)
    ages = rng.uniform(60.0, 120.0, size=n_children)

    child_idx = np.repeat(np.arange(n_children), 3)
    phase = np.tile(np.arange(3), n_children)
    G = child_g[child_idx]
    trt = ((G == 1) | (phase >= 1)).astype(float)

    # Pre counts: ~55% of rows at the floor, the rest with small positive counts,
    # independent of arm (period-1 baselines predate randomised exposure).
    at_floor_pre = rng.random(child_idx.size) < 0.55
    pre_counts = np.where(
        at_floor_pre, 0.0, rng.integers(1, n_trials // 2, size=child_idx.size)
    ).astype(float)
    indicator = (pre_counts > 0).astype(float)
    # Empirical logit of the pre proportion — only its finiteness matters here
    # (the off-floor path uses the indicator, not the graded logit).
    pre_logit = np.log(
        (pre_counts + 0.5) / (n_trials - pre_counts + 0.5)
    )

    wave_offset = np.asarray(truth["wave_offset"], dtype=float)
    eta = (
        truth["alpha"]
        + wave_offset[phase]
        + truth["gamma_own_offfloor"] * indicator
        + truth["beta_trt"] * trt
        + u_child[child_idx]
    )
    y = rng.binomial(1, _expit(eta)).astype(float)

    a_months = ages[child_idx] + 5.0 * phase
    a_std, age_scaler = standardise(a_months)
    prepared = PreparedData(
        subject_ids=np.asarray([f"c{i:03d}" for i in child_idx]),
        child_idx=child_idx.astype(np.int64),
        phase=phase.astype(np.int64),
        G=G,
        A_months=a_months,
        A_std=a_std,
        age_scaler=age_scaler,
        pre_logit={"P": pre_logit},
        pre_counts={"P": pre_counts},
        post_counts={"P": y},
        n_trials={"P": n_trials},
        n_obs=int(y.size),
        n_children=n_children,
        n_phases=3,
        dropped_rows=0,
        phase_mode="all",
    )
    return prepared, float(u_child.std(ddof=1))


def test_offfloor_indicator_parameterisation_recovers_truth() -> None:
    """#391 finding 2: the production off-floor model — Bernoulli likelihood,
    always-on binary off-floor-at-pre indicator main effect, on-intervention
    term, non-centred child random intercept — recovers its own generating
    parameters. The indicator coefficient is the load-bearing check: under the
    pre-decision specification the control arm was forced flat in the baseline
    and a trt x own interaction absorbed this signal instead."""
    import pymc as pm

    from language_reading_predictors.statistical_models.factories import (
        build_gain_factors_model,
    )

    truth = {
        "alpha": -1.6,
        "wave_offset": np.asarray([0.0, 0.30, 0.50]),
        "gamma_own_offfloor": 2.0,
        "beta_trt": 0.6,
        "sigma_child": 0.5,
    }
    prepared, realised_sigma_child = _prepared_offfloor_panel(
        n_children_per_arm=40, truth=truth, n_trials=26, seed=20260807
    )
    built = build_gain_factors_model(
        prepared, outcome_symbol="P", likelihood="bernoulli_offfloor"
    )
    names = {v.name for v in built.model.free_RVs}
    assert "gamma_own_offfloor" in names and "gamma_own" not in names
    with built.model:
        trace = pm.sample(
            draws=500,
            tune=500,
            chains=2,
            cores=2,
            target_accept=0.9,
            nuts_sampler="nutpie",
            return_inferencedata=True,
            random_seed=20260807,
            progressbar=False,
        )

    posterior = trace.posterior

    def _draws(name: str) -> np.ndarray:
        return posterior[name].values.ravel()

    def _covers(draws: np.ndarray, value: float) -> bool:
        lo, hi = np.quantile(draws, (0.03, 0.97))
        return bool(lo <= value <= hi)

    # The decision's target: the baseline-status main effect is recovered, not
    # forced to zero and not leaked into an interaction (none is fitted).
    g_own = _draws("gamma_own_offfloor")
    assert abs(float(g_own.mean()) - truth["gamma_own_offfloor"]) < 0.5
    assert _covers(g_own, truth["gamma_own_offfloor"])
    assert float(np.mean(g_own > 0)) > 0.99

    # The randomised term (identified by the period-1 contrast) is covered; its
    # N(0, 0.5) tier prior shrinks the point estimate, so coverage — not a tight
    # mean tolerance — is the honest assertion at this sample size.
    assert _covers(_draws("beta_trt"), truth["beta_trt"])

    # Combined period-1 intercept (alpha and the free period offsets split the
    # level between them, so only their sum at phase 0 is sharply identified).
    combined = _draws("alpha") + posterior["alpha_phase"].values[..., 0].ravel()
    assert _covers(combined, truth["alpha"])

    # A single realisation only exposes the realised intercept spread.
    assert _covers(_draws("sigma_child"), realised_sigma_child)

    # Simulated ages are outcome-independent: the precision term is null.
    assert abs(float(_draws("gamma_A").mean())) < 0.25
