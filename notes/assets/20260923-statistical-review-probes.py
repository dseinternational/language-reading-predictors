# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Reproduce source-review counterexamples without fitting study models.

Drafted by Codex/GPT-6. Run from the repository root with uv run python.
These probes report current behaviour, not expected behaviour after a fix.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import norm

from language_reading_predictors.data_utils import load_and_filter
from language_reading_predictors.models.objective import robust_huber_delta
from language_reading_predictors.models.permutation import (
    pooled_permutation_deltas,
    subject_block_permutation_indices,
)
from language_reading_predictors.statistical_models.new_child_predictive import (
    NewChildPlan,
    NewChildValidation,
    _half_split_error,
)
from language_reading_predictors.statistical_models.summaries.itt import tau_summary_itt
from language_reading_predictors.statistical_models.summaries.rope import rope_summary


def huber_target() -> dict:
    y = np.array([0.0] * 9 + [10.0])
    delta = robust_huber_delta(y).delta
    root = brentq(lambda centre: np.clip(y - centre, -delta, delta).mean(), 0, 10)
    assert np.isclose(root, delta / 9)
    assert not np.isclose(root, y.mean())
    return {"mean": y.mean(), "huber_optimum": root, "delta": delta}


class WavePredictor:
    def predict(self, frame):
        return frame["wave"].to_numpy(dtype=float)


def permutation_alignment() -> dict:
    # A and B have the same trajectory x(t)=t, but B missed wave 2.
    groups = np.array(["A", "A", "A", "B", "B"])
    wave = np.array([1, 2, 3, 1, 3])
    frame = pd.DataFrame({"wave": wave})
    deltas = pooled_permutation_deltas(
        [WavePredictor()],
        frame,
        wave,
        [np.arange(5)],
        groups,
        {"wave": [0]},
        n_repeats=100,
        seed=47,
    )["wave"]
    assert deltas.max() > 0
    original = np.array([1, 2, 1, 2])
    equal_groups = np.array(["A", "A", "B", "B"])
    reorder = np.array([0, 1, 3, 2])
    before, after = [], []
    for seed in range(100):
        donor = subject_block_permutation_indices(equal_groups, np.random.default_rng(seed))
        before.append(float(np.mean((original - original[donor]) ** 2)))
        after.append(float(np.mean((original[reorder] - original[reorder][donor]) ** 2)))
    assert max(before) == 0 and max(after) > 0
    return {
        "identical_wave_trajectory_with_missing_wave_mean_rmse_delta": deltas.mean(),
        "identical_wave_trajectory_with_missing_wave_max_rmse_delta": deltas.max(),
        "same_complete_data_original_order_mean_mse_delta": np.mean(before),
        "same_complete_data_reordered_mean_mse_delta": np.mean(after),
    }


def study_wave_alignment() -> list[dict]:
    results = []
    for target in ["ewrswr", "ewrswr_gain", "nonword", "nonword_gain"]:
        df, _x, _y, groups = load_and_filter(target, ["time"], None)
        wave = df["time"].to_numpy(dtype=int)
        changed = []
        for repeat in range(50):
            donor = subject_block_permutation_indices(groups, np.random.default_rng([47, repeat]))
            changed.append(float(np.mean(wave != wave[donor])))
        results.append({"target": target, "rows": len(df), "mean_fraction_different_wave": np.mean(changed)})
    return results


def moderated_direction() -> dict:
    rng = np.random.default_rng(42)
    chains, draws = 4, 500
    tau = rng.normal(0.2, 0.001, (chains, draws))
    interaction = rng.normal(2, 0.001, (chains, draws))
    moderator = np.array([-1.0, 1.0])
    group = np.array([0.0, 1.0])
    baseline = np.array([0.0, 10.0])
    delta = tau[..., None] + interaction[..., None] * moderator
    eta = baseline + delta * group
    trace = xr.DataTree()
    trace["posterior"] = xr.Dataset(
        {
            "tau": (("chain", "draw"), tau),
            "gamma_tau_int": (("chain", "draw"), interaction),
            "eta": (("chain", "draw", "obs_id"), eta),
        }
    )
    kwargs = {"G": group, "moderators": [("gamma_tau_int", moderator)], "ci_prob": 0.89}
    headline = tau_summary_itt(trace, **kwargs)
    card = rope_summary(trace, n_trials=10, delta=1, **kwargs)
    assert headline["prob_ame_pos"] == 0 and card["pd"] == 1
    return {
        "headline_prob_benefit": headline["prob_ame_pos"],
        "rope_prob_benefit": card["pd"],
        "rope_items_median": card["items_median"],
    }


def predictive_gate() -> dict:
    common = dict(
        plan=NewChildPlan(child_dims=("child",), latent_vars=("z",)),
        n_children=1,
        posterior_draws_used=1000,
        elpd=-10.0,
        elpd_se=1.0,
        p_loo=1.0,
        pointwise_elpd=np.array([-10.0]),
        good_k=0.7,
        latents_redrawn=("z",),
        observed_nodes=("y",),
        latent_mc_error=0.0,
    )
    missing_k = NewChildValidation(**common, pareto_k=np.array([np.nan]))
    assert missing_k.reliable
    # Same mean log term, different tail. PSIS weights depend on that tail.
    left = np.array([-1.0, -9.0]).reshape(1, 2, 1)
    right = np.array([-5.0, -5.0]).reshape(1, 2, 1)
    error = _half_split_error([left, right], [1, 1], ["z"])
    loo_left = -(logsumexp(-left.ravel()) - np.log(2))
    loo_right = -(logsumexp(-right.ravel()) - np.log(2))
    assert error == 0 and abs(loo_left - loo_right) > 3
    return {
        "nan_pareto_k_marked_reliable": missing_k.reliable,
        "half_split_error": error,
        "half_split_raw_importance_loo_left": loo_left,
        "half_split_raw_importance_loo_right": loo_right,
        "note": "Two-draw algebra isolates the pre-PSIS integration check; it is not a study LOO estimate.",
    }


def unequal_intervals() -> dict:
    # Exact linear growth in elapsed months, with a baseline-dependent rate.
    baseline = np.tile(np.array([-1.0, 0.0, 1.0]), 2)
    interval = np.repeat(np.array([5.0, 8.0]), 3)
    change = (1 + 0.2 * baseline) * interval
    period = np.repeat(np.eye(2), 3, axis=0)
    design = np.column_stack([period, baseline])
    coefficient = np.linalg.lstsq(design, change, rcond=None)[0]
    residual = change - design @ coefficient
    assert np.isclose(np.max(np.abs(residual)), 0.3)
    return {
        "per_month_baseline_slope": 0.2,
        "true_interval_slopes": [1.0, 1.6],
        "fitted_common_slope_with_period_intercepts": coefficient[-1],
        "max_absolute_residual": np.max(np.abs(residual)),
        "note": "Period intercepts cannot absorb elapsed-time differences in the baseline slope.",
    }


def timing_scenarios_are_not_bounds() -> dict:
    # Both arms have the same trajectory and there is no treatment effect.
    # All growth occurs in the short extra interval before the later assessment.
    early_months, later_months = 5.0, 5.44
    early_gain, later_gain = 0.0, 10.0
    gap = later_months - early_months
    rate_products = [gap * early_gain / early_months, gap * later_gain / later_months]
    actual_timing_difference = later_gain - early_gain
    assert max(rate_products) < actual_timing_difference
    return {
        "whole_period_rate_products": rate_products,
        "actual_difference_due_only_to_timing": actual_timing_difference,
        "note": "A monotone trajectory alone does not make whole-period average rates bound the endpoint effect.",
    }


def pit_interpretation() -> dict:
    # For Y ~ N(0, 1) and forecast N(mu, sigma), F_PIT(u) = Phi(mu + sigma*Phi^-1(u)).
    grid = np.array([0.25, 0.5, 0.75])
    shifted = norm.cdf(1.0 + norm.ppf(grid)) - grid
    too_wide = norm.cdf(2.0 * norm.ppf(grid)) - grid
    too_narrow = norm.cdf(0.5 * norm.ppf(grid)) - grid
    assert shifted[1] > 0.3
    assert too_wide[1] == too_narrow[1] == 0
    return {
        "probabilities": grid.tolist(),
        "correct_spread_wrong_location_ecdf_minus_uniform": shifted.tolist(),
        "too_wide_forecast_ecdf_minus_uniform": too_wide.tolist(),
        "too_narrow_forecast_ecdf_minus_uniform": too_narrow.tolist(),
    }


def registry_check() -> dict:
    from collections import Counter

    from language_reading_predictors.statistical_models.family_registry import resolve_run_plan
    from language_reading_predictors.statistical_models.registry import discover_models

    counts = Counter()
    moderated_itt = []
    for key, entry in discover_models().items():
        module = entry.load()
        spec = module.SPEC if hasattr(module, "SPEC") else module.get_spec()
        plan = resolve_run_plan(spec)
        assert spec.model_id == key
        counts[spec.kind] += 1
        if spec.kind == "itt" and (
            getattr(plan, "use_varying_tau", False) or getattr(plan, "tau_moderator_symbol", None) is not None
        ):
            moderated_itt.append(key)
    return {
        "models_resolved": sum(counts.values()),
        "by_family": dict(sorted(counts.items())),
        "registered_moderated_itt": moderated_itt,
    }


def main() -> None:
    result = {
        "huber_target": huber_target(),
        "permutation_alignment": permutation_alignment(),
        "study_wave_alignment": study_wave_alignment(),
        "moderated_direction": moderated_direction(),
        "predictive_gate": predictive_gate(),
        "unequal_intervals": unequal_intervals(),
        "timing_scenarios_are_not_bounds": timing_scenarios_are_not_bounds(),
        "pit_interpretation": pit_interpretation(),
        "registry": registry_check(),
    }
    path = Path(__file__).with_suffix(".json")
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
