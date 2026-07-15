# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the post-hoc floor rule and its ``pre_required`` data-layer path.

The floor rule classifies a heavily-floored outcome (>= 40% of post-scores at
zero at t2) for an exploratory binary transition analysis. Nonword ``N`` must be
loadable without its missing baselines silently disappearing before eligibility
is reported by arm.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.floor import (
    FLOOR_THRESHOLD,
    baseline_floor_eligibility_by_arm,
    baseline_floor_status_bounds,
    binary_transition_missingness_bounds,
    is_floored,
    proportion_at_zero,
)
from language_reading_predictors.statistical_models.factories import build_itt_model
from language_reading_predictors.statistical_models.measures import (
    ITT_OUTCOMES,
    MEASURES,
)
from language_reading_predictors.statistical_models.preprocessing import (
    load_and_prepare,
    restrict_to_baseline_floored,
)


def _write_floored(tmp_path, n_children: int = 30, seed: int = 5):
    """Synthetic data where phonetic spelling (P) and nonword (N) are floored
    (~80% zero at every wave), with three missing nonword *baselines* (t1)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_children):
        sid = f"S{i:03d}"
        g = int(rng.integers(1, 3))
        age_base = int(rng.integers(60, 110))
        for t in (1, 2, 3, 4):
            row = {
                V.SUBJECT_ID: sid,
                V.TIME: t,
                V.GROUP: g,
                V.AGE: age_base + 6 * (t - 1),
            }
            for s in ITT_OUTCOMES:
                m = MEASURES[s]
                row[m.column] = int(rng.integers(0, m.n_trials + 1))
            # Floor P and N: mostly zero.
            row[V.SPPHON] = 0 if rng.random() < 0.8 else int(rng.integers(1, 10))
            row[V.NONWORD] = 0 if rng.random() < 0.8 else int(rng.integers(1, 7))
            rows.append(row)
    df = pd.DataFrame(rows)
    # Inject three missing nonword baselines (t1) to exercise ``pre_required``.
    t1_idx = df.index[df[V.TIME] == 1][:3]
    df.loc[t1_idx, V.NONWORD] = np.nan
    path = tmp_path / "rli_floored.csv"
    df.to_csv(path, index=False)
    return path


def test_is_floored_flags_floored_outcomes(tmp_path):
    p = _write_floored(tmp_path)
    prep = load_and_prepare(path=p, phase_mode="itt", outcomes=("P", "L"))
    assert is_floored(prep, "P")
    assert proportion_at_zero(prep, "P") >= FLOOR_THRESHOLD
    # A non-floored standardised outcome does not trip the rule.
    assert not is_floored(prep, "L")


def test_is_floored_nonword_post_only(tmp_path):
    p = _write_floored(tmp_path)
    prep = load_and_prepare(
        path=p, phase_mode="itt", outcomes=("N",), pre_required=()
    )
    assert is_floored(prep, "N")
    assert proportion_at_zero(prep, "N") >= FLOOR_THRESHOLD


def test_pre_required_exempts_missing_nonword_baseline(tmp_path):
    """With ``pre_required=()`` the missing nonword baselines do not drop rows,
    while the default (baseline required) drops them."""
    p = _write_floored(tmp_path)
    with pytest.warns(UserWarning):
        prep_drop = load_and_prepare(path=p, phase_mode="itt", outcomes=("N",))
    prep_keep = load_and_prepare(
        path=p, phase_mode="itt", outcomes=("N",), pre_required=()
    )
    assert prep_drop.n_obs == 27  # three missing-baseline children dropped
    assert prep_keep.n_obs == 30  # all kept (post present, group/age present)


def test_pre_required_must_be_subset_of_outcomes(tmp_path):
    p = _write_floored(tmp_path)
    with pytest.raises(ValueError):
        load_and_prepare(
            path=p, phase_mode="itt", outcomes=("P",), pre_required=("R",)
        )


def _write_known_eligibility(tmp_path):
    """Twelve children with known arm-specific N floor eligibility."""
    path = _write_floored(tmp_path, n_children=12, seed=17)
    df = pd.read_csv(path)
    pre = [0, 0, 0, 1, np.nan, np.nan, 0, 0, 0, 0, 1, np.nan]
    post = [1, 0, 2, 0, 1, 0, 1, 0, 0, 2, 0, 1]
    for i in range(12):
        sid = f"S{i:03d}"
        group = 1 if i < 6 else 2
        df.loc[df[V.SUBJECT_ID] == sid, V.GROUP] = group
        df.loc[
            (df[V.SUBJECT_ID] == sid) & (df[V.TIME] == 1), V.NONWORD
        ] = pre[i]
        df.loc[
            (df[V.SUBJECT_ID] == sid) & (df[V.TIME] == 2), V.NONWORD
        ] = post[i]
    df.to_csv(path, index=False)
    return path


def _write_joint_transition_missingness(tmp_path):
    """Twelve children covering every eligibility/outcome missingness category."""
    path = _write_floored(tmp_path, n_children=12, seed=23)
    df = pd.read_csv(path)
    pre = [0, 0, 0, 0, np.nan, np.nan, 0, 0, 0, 0, np.nan, np.nan]
    post = [1, 0, 2, np.nan, 1, np.nan, 1, 0, 0, np.nan, 0, np.nan]
    for i in range(12):
        sid = f"S{i:03d}"
        group = 1 if i < 6 else 2
        df.loc[df[V.SUBJECT_ID] == sid, V.GROUP] = group
        df.loc[
            (df[V.SUBJECT_ID] == sid) & (df[V.TIME] == 1), V.NONWORD
        ] = pre[i]
        df.loc[
            (df[V.SUBJECT_ID] == sid) & (df[V.TIME] == 2), V.NONWORD
        ] = post[i]
    df.to_csv(path, index=False)
    return path


def test_floor_eligibility_exposes_missing_baselines_and_matches_model(tmp_path):
    """Loader -> eligibility -> restriction -> PyMC build keeps exactly 7 rows.

    This exercises the fitted analysis-set path without running NUTS. Missing
    baseline values remain visible in the audit table but never become eligible.
    """
    path = _write_known_eligibility(tmp_path)
    prepared = load_and_prepare(
        path=path, phase_mode="itt", outcomes=("N",), pre_required=()
    )
    eligibility = baseline_floor_eligibility_by_arm(prepared, "N").set_index("arm")
    assert prepared.n_obs == 12
    assert eligibility.loc["intervention", "n_pre_missing"] == 2
    assert eligibility.loc["intervention", "n_exploratory_eligible"] == 3
    assert eligibility.loc["control", "n_pre_missing"] == 1
    assert eligibility.loc["control", "n_exploratory_eligible"] == 4

    at_risk = restrict_to_baseline_floored(prepared, "N")
    assert at_risk.n_obs == int(eligibility["n_exploratory_eligible"].sum()) == 7
    assert np.bincount(at_risk.G, minlength=2).tolist() == [4, 3]
    assert np.isfinite(at_risk.pre_logit["N"]).all()

    built = build_itt_model(
        at_risk,
        outcome_symbol="N",
        likelihood="bernoulli_offfloor",
        cross_symbols=(),
        use_age_linear=True,
        use_own_baseline=False,
    )
    assert built.prepared.n_obs == 7
    observed = built.model.rvs_to_values[built.model["y_offfloor"]].eval()
    assert np.array_equal(
        observed, (built.prepared.post_counts["N"] > 0).astype(np.int64)
    )


def test_floor_status_bounds_enumerate_unknown_baseline_eligibility(tmp_path):
    prepared = load_and_prepare(
        path=_write_known_eligibility(tmp_path),
        phase_mode="itt",
        outcomes=("N",),
        pre_required=(),
    )

    row = baseline_floor_status_bounds(prepared, "N").iloc[0]

    assert row.observed_known_eligibility_difference == pytest.approx((2 / 3) - (2 / 4))
    assert row.eligibility_status_lower == pytest.approx((2 / 4) - (3 / 5))
    assert row.eligibility_status_upper == pytest.approx((3 / 4) - (2 / 4))
    assert row.intervention_unknown_eligibility_n == 2
    assert row.control_unknown_eligibility_n == 1


def test_binary_transition_bounds_jointly_complete_archive_and_absent_children(
    tmp_path,
):
    path = _write_joint_transition_missingness(tmp_path)
    with pytest.warns(UserWarning):
        prepared = load_and_prepare(
            path=path,
            phase_mode="itt",
            outcomes=("N",),
            pre_required=(),
        )

    bounds = binary_transition_missingness_bounds(
        prepared,
        "N",
        randomised_by_g={1: 7, 0: 8},
    ).set_index("scope")

    archive = bounds.loc["archived_dataset"]
    assert archive.intervention_fixed_eligible_observed_n == 3
    assert archive.intervention_fixed_events_n == 2
    assert archive.intervention_known_eligible_missing_post_n == 1
    assert archive.intervention_unknown_eligibility_observed_event_n == 1
    assert archive.intervention_unknown_eligibility_missing_post_n == 1
    assert archive.control_fixed_eligible_observed_n == 3
    assert archive.control_fixed_events_n == 1
    assert archive.control_known_eligible_missing_post_n == 1
    assert archive.control_unknown_eligibility_observed_zero_n == 1
    assert archive.control_unknown_eligibility_missing_post_n == 1
    assert archive.intervention_min_risk == pytest.approx(2 / 5)
    assert archive.intervention_max_risk == pytest.approx(5 / 6)
    assert archive.control_min_risk == pytest.approx(1 / 6)
    assert archive.control_max_risk == pytest.approx(3 / 5)
    assert archive.risk_difference_lower == pytest.approx((2 / 5) - (3 / 5))
    assert archive.risk_difference_upper == pytest.approx((5 / 6) - (1 / 6))

    full = bounds.loc["full_randomised_population"]
    assert full.intervention_absent_randomised_n == 1
    assert full.control_absent_randomised_n == 2
    assert full.intervention_min_risk == pytest.approx(2 / 6)
    assert full.intervention_max_risk == pytest.approx(6 / 7)
    assert full.control_min_risk == pytest.approx(1 / 8)
    assert full.control_max_risk == pytest.approx(5 / 7)
    assert full.risk_difference_lower == pytest.approx((2 / 6) - (5 / 7))
    assert full.risk_difference_upper == pytest.approx((6 / 7) - (1 / 8))


@pytest.mark.parametrize(
    ("column", "value", "match"),
    [
        (V.GROUP, 1.5, "Group codes must be exactly"),
        (V.NONWORD, 0.5, "must contain integer counts"),
        (V.NONWORD, -1, "must lie in"),
    ],
)
def test_binary_transition_bounds_reject_invalid_raw_values(
    tmp_path, column, value, match
):
    path = _write_known_eligibility(tmp_path)
    prepared = load_and_prepare(
        path=path,
        phase_mode="itt",
        outcomes=("N",),
        pre_required=(),
    )
    data = pd.read_csv(path)
    row = data.index[data[V.TIME] == 1][0]
    data[column] = data[column].astype(float)
    data.loc[row, column] = value
    data.to_csv(path, index=False)

    with pytest.raises(ValueError, match=match):
        binary_transition_missingness_bounds(
            prepared,
            "N",
            randomised_by_g={1: 6, 0: 6},
        )


@pytest.mark.parametrize(
    ("symbol", "eligible_by_arm", "missing_by_arm"),
    [
        ("P", {"intervention": 24, "control": 17}, {"intervention": 0, "control": 0}),
        ("N", {"intervention": 21, "control": 15}, {"intervention": 1, "control": 2}),
    ],
)
def test_current_floor_model_analysis_counts(
    symbol, eligible_by_arm, missing_by_arm
):
    """Lock the actual outcome-available subgroup counts used by P/N fits."""
    prepared = load_and_prepare(
        phase_mode="itt", outcomes=(symbol,), pre_required=()
    )
    eligibility = baseline_floor_eligibility_by_arm(prepared, symbol).set_index("arm")
    assert prepared.n_obs == 53
    for arm in ("intervention", "control"):
        assert eligibility.loc[arm, "n_exploratory_eligible"] == eligible_by_arm[arm]
        assert eligibility.loc[arm, "n_pre_missing"] == missing_by_arm[arm]

    at_risk = restrict_to_baseline_floored(prepared, symbol)
    built = build_itt_model(
        at_risk,
        outcome_symbol=symbol,
        likelihood="bernoulli_offfloor",
        cross_symbols=(),
        use_age_linear=True,
        use_own_baseline=False,
    )
    assert built.prepared.n_obs == sum(eligible_by_arm.values())
    assert {
        "control": int((built.prepared.G == 0).sum()),
        "intervention": int((built.prepared.G == 1).sum()),
    } == eligible_by_arm


@pytest.mark.parametrize(
    ("symbol", "archive_bounds", "full_bounds"),
    [
        ("P", (0.125, 0.1805555556), (0.03, 0.22)),
        ("N", (0.2670454545, 0.3650793651), (0.1570048309, 0.4)),
    ],
)
def test_current_floor_transition_missingness_bounds(
    symbol, archive_bounds, full_bounds
):
    with pytest.warns(UserWarning):
        prepared = load_and_prepare(
            phase_mode="itt", outcomes=(symbol,), pre_required=()
        )
    bounds = binary_transition_missingness_bounds(prepared, symbol).set_index("scope")

    assert bounds.loc[
        "archived_dataset", "risk_difference_lower"
    ] == pytest.approx(archive_bounds[0])
    assert bounds.loc[
        "archived_dataset", "risk_difference_upper"
    ] == pytest.approx(archive_bounds[1])
    assert bounds.loc[
        "full_randomised_population", "risk_difference_lower"
    ] == pytest.approx(full_bounds[0])
    assert bounds.loc[
        "full_randomised_population", "risk_difference_upper"
    ] == pytest.approx(full_bounds[1])


def test_floor_specs_override_generic_itt_estimand_metadata():
    from language_reading_predictors.statistical_models.lrp_rli_itt_009 import (
        SPEC as P_SPEC,
    )
    from language_reading_predictors.statistical_models.lrp_rli_itt_011 import (
        SPEC as N_SPEC,
    )

    for spec in (P_SPEC, N_SPEC):
        assert spec.design == (
            "waitlist_randomised_t1_to_t2_observed_baseline_floor_subgroup"
        )
        assert spec.estimand_type == (
            "post_hoc_exploratory_available_case_subgroup_risk_difference"
        )
        assert "observed_baseline_floor_subgroup" in spec.causal_status
