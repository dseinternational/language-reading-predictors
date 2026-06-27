# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for diagnostics helpers (issue #125 Area 3 / step 0b)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from language_reading_predictors.statistical_models import diagnostics as diag


def test_interval_cols_matches_eti_and_hdi():
    cols = ["mean", "sd", "eti95_lb", "eti95_ub", "ess_bulk", "ess_tail", "r_hat"]
    assert diag._interval_cols(cols) == ["eti95_lb", "eti95_ub"]
    # Legacy HDI naming is still recognised.
    assert diag._interval_cols(["hdi_3%", "hdi_97%", "mean"]) == ["hdi_3%", "hdi_97%"]
    assert diag._interval_cols(["mean", "sd"]) == []


def test_bfmi_per_chain_matches_reference():
    rng = np.random.default_rng(0)
    energy = rng.normal(size=(2, 500))
    ss = xr.Dataset({"energy": (("chain", "draw"), energy)})
    trace = SimpleNamespace(sample_stats=ss)
    bf = diag._bfmi_per_chain(trace)
    assert bf is not None and len(bf) == 2
    for c in range(2):
        e = energy[c]
        ref = float(np.sum(np.diff(e) ** 2) / np.sum((e - e.mean()) ** 2))
        assert bf[c] == pytest.approx(ref)


def test_bfmi_per_chain_handles_missing_energy():
    trace = SimpleNamespace(sample_stats=xr.Dataset({}))
    assert diag._bfmi_per_chain(trace) is None


def test_thin_for_plots_thins_large_traces_only():
    # Build a DataTree-like object exposing .posterior.sizes and .isel.
    big = xr.Dataset(
        {"tau": (("chain", "draw"), np.zeros((6, 6000)))},
        coords={"chain": range(6), "draw": range(6000)},
    )
    dt = xr.DataTree.from_dict({"posterior": big})
    thinned = diag.thin_for_plots(dt, max_draws=1000)
    total = thinned.posterior.sizes["chain"] * thinned.posterior.sizes["draw"]
    # 36000 draws thinned to ~max_draws (small per-chain rounding overshoot is fine).
    assert total <= 1100
    assert total < 36000

    # A small trace is returned unchanged.
    small = xr.DataTree.from_dict(
        {
            "posterior": xr.Dataset(
                {"tau": (("chain", "draw"), np.zeros((2, 250)))},
                coords={"chain": range(2), "draw": range(250)},
            )
        }
    )
    assert diag.thin_for_plots(small, max_draws=1000) is small
