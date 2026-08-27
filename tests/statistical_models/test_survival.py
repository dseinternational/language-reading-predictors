# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for the floor-sitter survival family (#230 §5, #293 review).

Covers the two pieces most likely to regress and cheap to test without sampling:
- ``prepare_survival``'s person-period expansion — at-risk entry, event placement,
  flicker → first-crossing, censoring / drop of a mid-study gap, and the
  intervention-aligned ``treated`` coding — on a tiny hand-built long frame.
- ``reporting._readiness_knee`` — the knee finder — on synthetic ``f_mech`` draws.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from language_reading_predictors.statistical_models import reporting
from language_reading_predictors.statistical_models.survival import prepare_survival


def _fixture() -> pd.DataFrame:
    """Six children x 4 waves; nonword (N) is the floored outcome under test.

    C1 off at t2 (immediate); C2 off at t4 (waitlist, 3 rows); C3 never off (immediate,
    censored, 3 rows); C4 t2 missing (dropped — no interval placeable); C5 off-floor at
    t1 (excluded); C6 flicker 0,2,0,0 (waitlist, event at first crossing t2).
    """
    rows = []
    spec = {
        # sid: (group, [nonword t1..t4], yarclet@t1, ewrswr@t1, age@t1)
        "C1": (1, [0, 3, np.nan, np.nan], 10, 5, 60),
        "C2": (2, [0, 0, 0, 2], 15, 8, 62),
        "C3": (1, [0, 0, 0, 0], 20, 10, 58),
        "C4": (1, [0, np.nan, 4, 5], 12, 6, 61),
        "C5": (2, [2, 3, np.nan, np.nan], 18, 9, 63),
        "C6": (2, [0, 2, 0, 0], 25, 12, 64),
    }
    for sid, (grp, nonword, yl, ew, age) in spec.items():
        for t in (1, 2, 3, 4):
            rows.append({
                "subject_id": sid, "time": t, "group": grp,
                "nonword": nonword[t - 1],
                "yarclet": yl if t == 1 else np.nan,
                "ewrswr": ew if t == 1 else np.nan,
                "age": age if t == 1 else np.nan,
            })
    return pd.DataFrame(rows)


def test_person_period_expansion_counts_and_placement():
    p = prepare_survival("N", df=_fixture())
    # C1,C2,C3,C4,C6 are at floor at t1 (C5 off-floor -> excluded).
    assert p.n_at_risk_children == 5
    # C4 drops (t2 missing -> no interval placeable); the other four contribute rows.
    assert p.n_children == 4
    assert p.dropped_rows == 1
    # Rows: C1(1) + C2(3) + C3(3) + C6(1) = 8; events: C1, C2, C6 = 3.
    assert p.n_obs == 8
    assert p.n_events == 3
    # Events land at the first off-floor interval: C1 & C6 at interval 0, C2 at interval 2.
    event_intervals = sorted(p.interval_idx[p.event == 1].tolist())
    assert event_intervals == [0, 0, 2]
    # Baseline age was added as a covariate (docstring/impl reconciliation).
    assert set(p.covariates) == {"L0", "W0", "A0"}


def test_intervention_aligned_treated_coding():
    p = prepare_survival("N", df=_fixture())
    G, k, treated = p.G, p.interval_idx, p.treated
    # Immediate arm (G==1) is treated in every interval.
    assert np.all(treated[G == 1] == 1)
    # Waitlist arm (G==0): untreated in interval 0, treated from interval 1 (crossover).
    assert np.all(treated[(G == 0) & (k == 0)] == 0)
    assert np.all(treated[(G == 0) & (k >= 1)] == 1)


def test_treatment_window_controls_where_tau_enters_eta():
    """Under the default "randomised" window tau reaches only the interval-1
    treated rows; the legacy "pooled" comparator keeps the all-interval shift
    (2026-08-21 survival review, finding 1)."""
    from language_reading_predictors.statistical_models.survival import (
        build_survival_model,
    )

    p = prepare_survival("N", df=_fixture())

    def _tau_reach(window: str) -> np.ndarray:
        import pytensor

        built = build_survival_model(p, treatment_window=window)
        m = built.model
        point = m.initial_point()
        rvs = list(m.free_RVs)
        eta_fn = pytensor.function(
            [m[rv.name] for rv in rvs], m["eta"], on_unused_input="ignore"
        )
        zero = [np.zeros_like(point[rv.name]) for rv in rvs]
        one = [
            np.ones_like(point[rv.name])
            if rv.name == "tau"
            else np.zeros_like(point[rv.name])
            for rv in rvs
        ]
        return np.asarray(eta_fn(*one)) - np.asarray(eta_fn(*zero))

    randomised = _tau_reach("randomised")
    pooled = _tau_reach("pooled")
    expected_randomised = (p.treated * (p.interval_idx == 0)).astype(float)
    assert np.allclose(randomised, expected_randomised)
    assert np.allclose(pooled, p.treated.astype(float))

    with pytest.raises(ValueError, match="randomised.*pooled"):
        build_survival_model(p, treatment_window="all")


def test_survival_model_carries_the_child_loo_map():
    """#631 finding 14: the factory persists the row-to-child map so PSIS-LOO
    aggregates each child's person-period rows before leaving them out —
    row-level LOO would leak the held-out event through the later rows."""
    from language_reading_predictors.statistical_models.survival import (
        build_survival_model,
    )

    p = prepare_survival("N", df=_fixture())
    built = build_survival_model(p)
    assert "loo_child_idx" in built.model.named_vars
    values = built.model["loo_child_idx"].get_value()
    assert values.shape == (p.n_obs,)
    assert np.array_equal(np.asarray(values), p.child_idx)


def test_child_aggregation_recognises_the_survival_likelihood_node():
    """The aggregator must not be tied to the ``y_post`` node name: survival's
    likelihood is ``y_event`` (#631 finding 14)."""
    import xarray as xr

    from language_reading_predictors.statistical_models.diagnostics import (
        _joint_log_likelihood_by_child,
    )

    rng = np.random.default_rng(3)
    rows_to_child = np.array([0, 0, 1, 1, 1, 2], dtype=np.int64)
    ll = rng.normal(size=(2, 5, rows_to_child.size))
    trace = xr.DataTree.from_dict(
        {
            "log_likelihood": xr.Dataset(
                {"y_event": (("chain", "draw", "obs_id"), ll)}
            ),
            "constant_data": xr.Dataset(
                {"loo_child_idx": (("obs_id",), rows_to_child)}
            ),
        }
    )
    aggregated = _joint_log_likelihood_by_child(trace)
    assert aggregated is not None
    assert aggregated.sizes["loo_child"] == 3
    expected_child0 = ll[..., :2].sum(axis=-1)
    assert np.allclose(aggregated.values[..., 0], expected_child0)


def test_prepare_survival_rejects_duplicate_rows_and_bad_group_codes():
    """Source-integrity fail-loud, matching the shared loaders (finding 5)."""
    dup = pd.concat([_fixture(), _fixture().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate \\(subject, time\\) rows"):
        prepare_survival("N", df=dup)

    bad = _fixture()
    bad.loc[bad.index[0], "group"] = 3
    with pytest.raises(ValueError, match="Group codes must be exactly 1"):
        prepare_survival("N", df=bad)

    unstable = _fixture()
    c1 = unstable["subject_id"] == "C1"
    unstable.loc[c1 & (unstable["time"] == 4), "group"] = 2
    with pytest.raises(ValueError, match="Group code changes within child"):
        prepare_survival("N", df=unstable)


def test_per_child_standardisation_is_row_count_independent():
    # C3 contributes 3 rows and C1 one row; a per-child scaler must not up-weight C3.
    p = prepare_survival("N", df=_fixture())
    # Baseline L0 mean/SD are over the 4 unique children (10,15,20,25), not the 8 rows.
    sc = p.covariate_scalers["L0"]
    assert sc.mean == np.mean([10, 15, 20, 25])


def test_readiness_knee_finds_a_late_rising_curve():
    # f_mech flat below ~20 letter sounds, rising above -> knee should be high.
    n_trials, n_obs = 32, 48
    ell = np.log((np.linspace(0, 32, n_obs) + 0.5) / (n_trials - np.linspace(0, 32, n_obs) + 0.5))
    lvals = (n_trials + 1.0) / (1.0 + np.exp(-ell)) - 0.5
    fmean = np.where(lvals < 20, 0.0, (lvals - 20) * 0.3)
    f = np.repeat(fmean[:, None], 60, axis=1)  # (n_obs, draws), noise-free
    out = reporting._readiness_knee(f, ell, n_trials=n_trials, n_bins=6)
    assert 0.0 <= out["knee_count_median"] <= float(n_trials)
    assert out["knee_count_median"] > 15.0  # rises only in the upper range
    assert out["slope_above_knee_median"] >= out["slope_below_knee_median"]
    assert out["increasing_frac"] == 1.0  # noise-free rising curve
    assert out["n_obs"] == n_obs
