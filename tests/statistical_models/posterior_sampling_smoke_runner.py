# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Isolated fixed-seed PyMC/nutpie smoke run invoked by the test suite."""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pymc as pm
from scipy.special import expit


@dataclass(frozen=True, slots=True)
class SyntheticIttData:
    group: np.ndarray
    age_std: np.ndarray
    pre_logit: np.ndarray
    post_counts: np.ndarray
    n_trials: int

    @property
    def n_obs(self) -> int:
        return int(self.group.size)


def synthetic_itt_data(seed: int = 361) -> SyntheticIttData:
    rng = np.random.default_rng(seed)
    n_obs = 18
    n_trials = 24
    group = np.tile([0, 1], n_obs // 2).astype(float)
    age_months = np.linspace(66.0, 102.0, n_obs)
    age_std = (age_months - age_months.mean()) / age_months.std(ddof=1)
    pre_counts = rng.integers(3, 19, size=n_obs)
    pre_logit = np.log((pre_counts + 0.5) / (n_trials - pre_counts + 0.5))
    eta = -0.15 + 0.45 * group + 0.85 * pre_logit + 0.10 * age_std
    mean = expit(eta)
    concentration = 18.0
    child_probability = rng.beta(
        mean * concentration,
        (1.0 - mean) * concentration,
    )
    post_counts = rng.binomial(n_trials, child_probability).astype(float)
    return SyntheticIttData(
        group=group,
        age_std=age_std,
        pre_logit=pre_logit,
        post_counts=post_counts,
        n_trials=n_trials,
    )


def main() -> None:
    prepared = synthetic_itt_data()
    with pm.Model(coords={"obs_id": np.arange(prepared.n_obs)}):
        group = pm.Data("G", prepared.group, dims="obs_id")
        age_std = pm.Data("A_std", prepared.age_std, dims="obs_id")
        pre_logit = pm.Data("own_pre_logit", prepared.pre_logit, dims="obs_id")
        alpha = pm.Normal("alpha", mu=0.0, sigma=1.5)
        tau = pm.Normal("tau", mu=0.0, sigma=0.5)
        gamma_own = pm.Normal("gamma_own", mu=1.0, sigma=0.25)
        gamma_age = pm.Normal("gamma_A", mu=0.0, sigma=0.3)
        kappa = pm.HalfNormal("kappa", sigma=50.0)
        eta = pm.Deterministic(
            "eta",
            alpha + tau * group + gamma_own * pre_logit + gamma_age * age_std,
            dims="obs_id",
        )
        mean = pm.math.sigmoid(eta)
        pm.BetaBinomial(
            "y_post",
            n=prepared.n_trials,
            alpha=mean * kappa,
            beta=(1.0 - mean) * kappa,
            observed=prepared.post_counts.astype(int),
            dims="obs_id",
        )
        trace = pm.sample(
            draws=60,
            tune=60,
            chains=2,
            cores=1,
            target_accept=0.90,
            nuts_sampler="nutpie",
            random_seed=20260719,
            progressbar=False,
            compute_convergence_checks=False,
            return_inferencedata=True,
        )
        trace = pm.sample_posterior_predictive(
            trace,
            var_names=["y_post"],
            random_seed=20260719,
            progressbar=False,
            extend_inferencedata=True,
        )

    y_rep = trace.posterior_predictive["y_post"].values
    print(
        json.dumps(
            {
                "eta_shape": list(trace.posterior["eta"].shape),
                "posterior_predictive_shape": list(y_rep.shape),
                "tau_finite": bool(
                    np.isfinite(trace.posterior["tau"].values).all()
                ),
                "tau_shape": list(trace.posterior["tau"].shape),
                "y_in_bounds": bool(
                    (y_rep >= 0).all() and (y_rep <= prepared.n_trials).all()
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
