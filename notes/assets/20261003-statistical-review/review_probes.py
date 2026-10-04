# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Deterministic review probes. No posterior fitting; JSON contains aggregate results."""

from pathlib import Path
import json
import tempfile
import sys

sys.path.insert(0, str(Path.cwd()))

import numpy as np

from language_reading_predictors.statistical_models.preprocessing import load_and_prepare, logit_safe, standardise
from language_reading_predictors.statistical_models.pooled_levels import (
    resolve_pooled_levels_run_plan,
    build_pooled_levels_model,
)
from language_reading_predictors.statistical_models.lrp_rli_pl_001 import SPEC
from language_reading_predictors.statistical_models.factories.mediation import build_mediation_model
from language_reading_predictors.statistical_models.mediation import decompose, sensitivity_sweep
from language_reading_predictors.statistical_models.run_metadata import fitted_subject_identity
from language_reading_predictors.statistical_models.dose_response import resolve_dose_response_run_plan
from language_reading_predictors.statistical_models.factories.dose_response import build_dose_response_model
from language_reading_predictors.statistical_models.lrp_rli_dose_077 import SPEC as DOSE_SPEC
from tests.statistical_models.test_factories import _write_synthetic
from tests.statistical_models.test_mediation import _fake_trace

result = {}
plan = resolve_pooled_levels_run_plan(SPEC)
prepared = load_and_prepare(**plan.prepare_kwargs())
built = build_pooled_levels_model(prepared, **plan.factory_kwargs())
y = np.asarray(built.model.rvs_to_values[built.model["y_post"]].eval())
x = built.prepared.post_counts["L"]
mask = np.isfinite(x) & np.isfinite(built.prepared.post_counts["W"])
z_haldane, scale = standardise(logit_safe(x[mask], 32))
z_clip = built.model["mech_post_logit_std"].get_value()
result["pooled_levels_current_data"] = {
    "prepared_rows": prepared.n_obs,
    "likelihood_rows": int(y.size),
    "recorded_fitted_identity_rows": fitted_subject_identity(built.prepared)["n_rows"],
    "payload_fitted_rows": built.payload.n_fitted_rows,
    "rows_dropped_in_factory": built.payload.n_dropped_incomplete,
    "exposure_zero_rows": int((x[mask] == 0).sum()),
    "exposure_ceiling_rows": int((x[mask] == 32).sum()),
    "max_absolute_z_difference": float(np.max(np.abs(z_haldane - z_clip))),
    "transform_correlation": float(np.corrcoef(z_haldane, z_clip)[0, 1]),
    "zero_haldane_logit": float(logit_safe(np.array([0.0]), 32)[0]),
    "zero_clipped_logit": float(np.log(0.001 / 0.999)),
}
with tempfile.TemporaryDirectory(prefix="lrp-review-med-") as temp:
    prep = load_and_prepare(
        path=_write_synthetic(Path(temp), n_children=15), phase_mode="itt", outcomes=("B", "L", "W")
    )
    mbuilt, med = build_mediation_model(
        prep,
        mediator_symbol="L",
        outcome_symbol="B",
        confounder_symbols=("W",),
        score_mean_link="three_choice_guessing_floor",
    )
    names = [str(rv.name) for rv in mbuilt.model.free_RVs]
    trace = _fake_trace(
        names,
        positive=[n for n in names if n.startswith("kappa")],
        values={"b_G": 0.4, "b_M": 0.5, "b_GM": 0.1, "a_G": 0.7},
    )
    primary = decompose(trace, med, score_mean_link="three_choice_guessing_floor").set_index("quantity")
    ordinary, _ = sensitivity_sweep(trace, med, n_deltas=3)
    correct, _ = sensitivity_sweep(trace, med, n_deltas=3, score_mean_link="three_choice_guessing_floor")
    result["mediation_floor_sensitivity"] = {
        "primary_nie_median": float(primary.loc["NIE", "prob_median"]),
        "legacy_default_sweep_zero_nie_median": float(ordinary.iloc[0].nie_median),
        "correct_sweep_zero_nie_median": float(correct.iloc[0].nie_median),
        "ordinary_to_correct_median_ratio": float(ordinary.iloc[0].nie_median / correct.iloc[0].nie_median),
        "all_sweep_interval_columns_ratio_1_5": bool(
            np.allclose(ordinary[["nie_median", "nie_lo", "nie_hi"]], 1.5 * correct[["nie_median", "nie_lo", "nie_hi"]])
        ),
        "direction_probabilities_unchanged": bool(np.allclose(ordinary.nie_prob_pos, correct.nie_prob_pos)),
    }
dp = resolve_dose_response_run_plan(DOSE_SPEC)
dbuilt = build_dose_response_model(load_and_prepare(**dp.prepare_kwargs()), **dp.factory_kwargs())
dprep, dose = dbuilt.prepared, dbuilt.payload
p1 = (dprep.phase == 0) & dose.treated
result["dose_period1_design"] = {
    "period1_immediate_rows": int(p1.sum()),
    "pooled_treated_mean_sessions": float(dose.dose_scaler.mean),
    "period1_immediate_mean_sessions": float(dose.raw_attend[p1].mean()),
    "period1_immediate_mean_between_regressor": float(dose.dose_between[p1].mean()),
    "period1_immediate_mean_within_regressor": float(dose.dose_within[p1].mean()),
    "later_periods_contribute_to_child_dose_mean": True,
}
rng = np.random.default_rng(20261003)
trait = rng.normal(0, 2, size=(100_000, 1))
true_x = trait + rng.normal(0, 0.1, size=(100_000, 4))
x_observed = true_x + rng.normal(0, 2, size=true_x.shape)
y_true = true_x.copy()  # True direct contemporaneous effect is exactly one.
xb, yb = x_observed.mean(axis=1), y_true.mean(axis=1)
xw = x_observed - xb[:, None]
yw = y_true - yb[:, None]
result["within_between_counterexample"] = {
    "true_direct_effect": 1.0,
    "estimated_between_slope": float(np.cov(xb, yb)[0, 1] / np.var(xb, ddof=1)),
    "estimated_within_slope": float(np.sum(xw * yw) / np.sum(xw * xw)),
    "n_synthetic_children": 100_000,
    "n_waves": 4,
    "predictor_measurement_error_sd": 2.0,
}
assert (
    result["pooled_levels_current_data"]["recorded_fitted_identity_rows"]
    == result["pooled_levels_current_data"]["likelihood_rows"]
)
assert result["pooled_levels_current_data"]["max_absolute_z_difference"] < 1e-12
assert result["mediation_floor_sensitivity"]["all_sweep_interval_columns_ratio_1_5"]
assert result["mediation_floor_sensitivity"]["direction_probabilities_unchanged"]
assert result["within_between_counterexample"]["estimated_between_slope"] > 0.5
assert result["within_between_counterexample"]["estimated_within_slope"] < 0.01
print(json.dumps(result, indent=2))
