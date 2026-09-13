# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Derived diagnostics require the original chain and draw layout."""

import arviz as az
import numpy as np
import pytest
import xarray as xr

from language_reading_predictors.statistical_models.posteriors import derived_mc_diagnostics
from language_reading_predictors.statistical_models.mediation import _proportion_row


def _draws():
    rng = np.random.default_rng(731)
    return rng.normal(size=(4, 1000)) + np.array([-2, 2, -2, 2])[:, None]


def test_complete_draws_keep_between_chain_information():
    draws = _draws()
    actual = derived_mc_diagnostics(draws.ravel(), n_chains=4, n_draws=1000)
    expected = xr.DataArray(draws, dims=("chain", "draw"))
    assert actual["ess_bulk"] == pytest.approx(float(az.ess(expected, method="bulk")))
    assert actual["ess_tail"] == pytest.approx(float(az.ess(expected, method="tail")))
    assert actual["mcse_median"] == pytest.approx(float(az.mcse(expected, method="median")))
    assert actual["ess_bulk"] < 10


@pytest.mark.parametrize("missing", [np.nan, np.inf, -np.inf, "removed"])
def test_missing_draws_do_not_turn_multiple_chains_into_one(missing):
    draws = _draws().ravel()
    if missing == "removed":
        draws = draws[1:]
    else:
        draws[0] = missing
    actual = derived_mc_diagnostics(draws, n_chains=4, n_draws=1000, prefix="effect_")
    assert set(actual) == {"effect_ess_bulk", "effect_ess_tail", "effect_mcse_median"}
    assert all(np.isnan(value) for value in actual.values())


def test_undefined_mediated_ratios_keep_summaries_but_have_no_chain_diagnostics():
    total = np.ones(4000)
    total[0] = 0
    actual = _proportion_row(np.full(4000, 0.4), total, 0.055, 0.945, n_chains=4, n_draws=1000)
    assert actual["prob_median"] == pytest.approx(0.4)
    assert np.isnan(actual["ess_bulk"])
    assert np.isnan(actual["ess_tail"])
    assert np.isnan(actual["mcse_median"])


@pytest.mark.parametrize("chains, draws", [(0, 100), (2, 0), (-1, 100)])
def test_invalid_chain_dimensions_are_rejected(chains, draws):
    with pytest.raises(ValueError, match="must both be positive"):
        derived_mc_diagnostics(np.ones(100), n_chains=chains, n_draws=draws)
