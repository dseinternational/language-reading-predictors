# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the pre-specified floor rule and the ``pre_required`` data-layer
exemption it relies on (issue #119).

The floor rule classifies a heavily-floored outcome (>= 40% of post-scores at
zero at t2) so it can take the binary off-floor primary estimand; the post-only
outcome ``N`` (nonword) must be loadable without its missing baseline silently
dropping rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.floor import (
    FLOOR_THRESHOLD,
    is_floored,
    proportion_at_zero,
)
from language_reading_predictors.statistical_models.measures import (
    ITT_OUTCOMES,
    MEASURES,
)
from language_reading_predictors.statistical_models.preprocessing import (
    load_and_prepare,
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
