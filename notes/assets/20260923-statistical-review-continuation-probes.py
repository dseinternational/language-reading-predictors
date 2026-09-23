# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Check release interpretations and missingness algebra without study refits.

Drafted by Codex/GPT-6. Assertions reproduce current defects or named invariants.
All generated CSV fixtures live in an automatically removed local directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import arviz_stats as azs
import numpy as np
import pandas as pd
from scipy.stats import beta, norm
import xarray as xr

from language_reading_predictors.statistical_models import itt_missingness as missing
from language_reading_predictors.statistical_models.release.dependence import (
    _dependence_identification_note,
    _joint_width_channels,
)
from language_reading_predictors.statistical_models.release.robustness import evaluate_release
from language_reading_predictors.statistical_models.summaries.dependence import (
    dependence_identification_summary,
)


ROOT = Path(__file__).resolve().parents[2]


def conflict_is_not_attenuation(directory: Path) -> dict:
    """An exact normal model with zero-centred priors can reverse a coefficient.

    Observed vector y has likelihood N(theta, covariance); theta has prior N(0, I).
    Both power-scaling scores are computed by the installed ArviZ implementation.
    This is an algebraic counterexample, not a fit of any study family.
    """
    y = np.array([1.0, 4.0])
    covariance = np.array([[1.0, 0.9], [0.9, 1.0]])
    precision = np.linalg.inv(covariance)
    posterior_covariance = np.linalg.inv(precision + np.eye(2))
    posterior_mean = posterior_covariance @ precision @ y
    rng = np.random.default_rng(230923)
    draws = rng.multivariate_normal(posterior_mean, posterior_covariance, size=(4, 20000))
    residual = draws - y
    trace = xr.DataTree.from_dict(
        {
            "posterior": xr.Dataset({"tau": (("chain", "draw"), draws[:, :, 0])}),
            "log_prior": xr.Dataset({"theta": (("chain", "draw"), -0.5 * (draws**2).sum(axis=-1))}),
            "log_likelihood": xr.Dataset(
                {"y": (("chain", "draw"), -0.5 * np.einsum("...i,ij,...j->...", residual, precision, residual))}
            ),
        }
    )
    summary = azs.psense_summary(trace, var_names=["tau"], round_to=8)
    summary.to_csv(directory / "psense_summary.csv")
    decision = evaluate_release(directory, {"kind": "itt", "outcome_symbol": "W"})
    stronger_mean = np.linalg.solve(np.eye(2) + 1.01 * covariance, y)
    weaker_mean = np.linalg.solve(np.eye(2) + 0.99 * covariance, y)
    assert y[0] > 0 > posterior_mean[0]
    assert stronger_mean[0] < posterior_mean[0] < weaker_mean[0] < 0
    assert decision.tau_class == "prior_data_conflict"
    assert decision.status == "release" and "lower bound" in decision.note
    # Even in one dimension, shrinkage toward zero is no lower bound on truth.
    # If theta=0, an observation y=2 is possible under N(theta, 1).
    # With prior N(0, 1), the posterior is exactly N(1, 1/2).
    scalar_mean, scalar_variance = 1.0, 0.5
    return {
        "likelihood_centre_tau": float(y[0]),
        "zero_centred_joint_prior_posterior_mean_tau": float(posterior_mean[0]),
        "posterior_tau_mean_at_prior_powers": {
            "0.99": float(weaker_mean[0]),
            "1.00": float(posterior_mean[0]),
            "1.01": float(stronger_mean[0]),
        },
        "computed_power_sensitivity": summary.loc["tau"].to_dict(),
        "release_decision": decision.as_dict(),
        "scalar_counterexample": {
            "data_generating_theta": 0.0,
            "observed_y": 2.0,
            "posterior_mean": scalar_mean,
            "posterior_probability_below_its_mean": 0.5,
            "posterior_probability_positive": float(norm.cdf(scalar_mean / np.sqrt(scalar_variance))),
        },
    }


def interval_width_is_not_covariance(directory: Path) -> dict:
    """Independent uniform effects have nonzero width-implied correlation.

    A and B are independent U(0, 0.1). Their central 89% widths are 0.089.
    The difference has a triangular distribution with exact 89% width below.
    The comparator uses independent normal variables with the same marginal
    interval widths. Both pairs have exactly zero covariance by construction.
    """
    parent, companion = directory / "parent", directory / "companion"
    parent.mkdir()
    companion.mkdir()
    width = 0.1 * 0.89
    frame = pd.DataFrame({"outcome": ["A", "B"], "ame_prob_lo": [0.0055] * 2, "ame_prob_hi": [0.0945] * 2})
    for path in (parent, companion):
        frame.to_csv(path / "tau_summary.csv", index=False)
    parent_width = np.sqrt(2) * width
    companion_width = 2 * 0.1 * (1 - np.sqrt(0.11))
    result = _joint_width_channels(
        parent_dir=parent,
        companion_dir=companion,
        outcomes=("A", "B"),
        parent_width=float(parent_width),
        companion_width=float(companion_width),
    )
    assert result["channel_status"] == "measured"
    assert result["companion_implied_ame_correlation"] < -0.12
    assert result["dominant_width_channel"] == "cross_outcome_covariance"
    assert result["covariance_channel_share"] > 0.999999
    return {"true_correlation_both_pairs": 0.0, "exact_uniform_difference_width": companion_width, **result}


def unchanged_sd_is_not_unchanged_distribution(directory: Path) -> dict:
    """Two bounded correlation distributions with equal variance, different means.

    Prior 2*Beta(4,4)-1 is the two-outcome LKJ(4) marginal, variance 1/9.
    Comparator posterior 2*Beta(4.592,1.968)-1 has mean 0.4 and variance 1/9.
    Deterministic midpoint quantiles remove random Monte Carlo variation.
    """
    u = (np.arange(40000) + 0.5) / 40000
    prior = 2 * beta.ppf(u, 4, 4) - 1
    posterior = 2 * beta.ppf(u, 4.592, 1.968) - 1

    def dataset(values):
        return xr.Dataset(
            {"u_corr_pair": (("chain", "draw", "outcome_pair"), values.reshape(4, 10000, 1))},
            coords={"outcome_pair": ["A|B"]},
        )

    tree = xr.DataTree.from_dict({"prior": dataset(prior), "posterior": dataset(posterior)})
    frame = dependence_identification_summary(tree, ci_prob=0.89)
    assert frame is not None
    row = frame.iloc[0]
    assert np.isclose(row.posterior_prior_sd_ratio, 1, atol=0.0001)
    assert row.verdict == "prior-dominated"
    frame.to_csv(directory / "dependence_identification.csv", index=False)
    note = _dependence_identification_note(directory)
    assert "did not move off its prior" in note
    return {
        "exact_prior_mean": 0.0,
        "exact_posterior_mean": 0.4,
        "exact_variance_both_distributions": 1 / 9,
        "posterior_probability_positive": float(1 - beta.cdf(0.5, 4.592, 1.968)),
        "summary": row.to_dict(),
        "generated_note": note,
    }


def missingness_target_invariants() -> dict:
    """Independent checks of the actual archive masks and completion denominators."""
    data = missing.load_randomised_w_archive(missing.RLI_ARCHIVE_LOCAL_CSV)
    p0 = np.linspace(0.2, 0.5, 57)
    p1 = p0 + 0.1
    observed = data.target_outcome_observed
    posterior = xr.Dataset(
        {
            name: (("chain", "draw", dimension), np.broadcast_to(values, (2, 5, len(values))))
            for name, dimension, values in (
                ("p0_target", "target_id", p0),
                ("p1_target", "target_id", p1),
                ("p0_observed_profiles", "obs_id", p0[observed]),
                ("p1_observed_profiles", "obs_id", p1[observed]),
            )
        }
    )
    result = missing.summarise_missingness_sensitivity(SimpleNamespace(posterior=posterior), data)
    rows = result.set_index("scenario")
    factual = rows.loc["delta_i_+0_c_+0", "effect_items_median"]
    grid = result.loc[result.scenario_class == "arm_specific_delta_grid"]
    expected = factual + grid.delta_intervention_items / 29 - 3 * grid.delta_control_items / 28
    error = float(np.max(np.abs(grid.effect_items_median - expected)))
    assert error < 1e-12
    assert np.isclose(rows.loc["mar_all_57", "effect_items_median"], 7.9)
    assert not np.isclose(factual, 7.9)
    missing_i = (data.target_G == 1) & ~observed
    missing_c = (data.target_G == 0) & ~observed
    assert (int(missing_i.sum()), int(missing_c.sum())) == (1, 3)
    return {
        "observed_rows": data.n_obs,
        "target_profiles": len(p0),
        "missing_by_arm": [int(missing_i.sum()), int(missing_c.sum())],
        "common_profile_effect_items": 7.9,
        "factual_completion_effect_items": float(factual),
        "delta_grid_max_absolute_algebra_error": error,
        "checked_grid_cells": len(grid),
    }


def main() -> None:
    with TemporaryDirectory(prefix=".review-continuation-", dir=ROOT) as temporary:
        directory = Path(temporary)
        result = {
            "authorship": "Drafted by Codex/GPT-6",
            "scope": "Synthetic counterexamples and archive-mask algebra; no production refits or revalidation.",
            "prior_conflict_interpretation": conflict_is_not_attenuation(directory),
            "interval_width_covariance": interval_width_is_not_covariance(directory),
            "dependence_sd_ratio": unchanged_sd_is_not_unchanged_distribution(directory),
            "missingness_positive_checks": missingness_target_invariants(),
        }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
