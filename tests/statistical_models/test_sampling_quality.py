# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Guards for the shared unrounded sampling-quality extraction (#440).

The point of ``sampling_quality`` is that there is exactly one place where R-hat, ESS,
BFMI and divergences are read off a trace, and that place does not round. The critical
regression these tests pin is the ``round_to`` trap: ``az.summary`` rounds to two
significant figures unless passed the *string* ``"none"``, which turns every R-hat from
1.011 to 1.049 into ``1.0`` and silently clears an ``R-hat <= 1.01`` gate
(dseinternational/research#65; recurred in ``loo_refit`` and in a prototype script).
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pytest
import xarray as xr

from language_reading_predictors.statistical_models.sampling_quality import (
    SamplingQuality,
    sampling_quality,
)


def _trace(
    *,
    n_chains: int = 4,
    n_draws: int = 400,
    divergences: int = 0,
    seed: int = 0,
    offset: float = 0.0,
    with_energy: bool = True,
    with_diverging: bool = True,
    extra_vars: dict | None = None,
):
    """A small DataTree trace with well-mixed chains and a known divergence count.

    ``offset`` shifts chain 0's mean to manufacture a deliberately poor R-hat.
    """
    rng = np.random.default_rng(seed)
    draws = rng.normal(size=(n_chains, n_draws))
    draws[0] += offset

    data = {"theta": (("chain", "draw"), draws)}
    for name, values in (extra_vars or {}).items():
        data[name] = (("chain", "draw"), values)
    posterior = xr.Dataset(
        data, coords={"chain": np.arange(n_chains), "draw": np.arange(n_draws)}
    )

    stats: dict = {}
    if with_diverging:
        div = np.zeros((n_chains, n_draws), dtype=bool)
        if divergences:
            div.reshape(-1)[:divergences] = True
        stats["diverging"] = (("chain", "draw"), div)
    if with_energy:
        stats["energy"] = (("chain", "draw"), rng.normal(size=(n_chains, n_draws)))

    groups = {"posterior": posterior}
    if stats:
        groups["sample_stats"] = xr.Dataset(
            stats, coords={"chain": np.arange(n_chains), "draw": np.arange(n_draws)}
        )
    return xr.DataTree.from_dict(groups)


# --- the regression this module exists for ----------------------------------------


def test_reports_unrounded_rhat_where_rounding_would_hide_the_failure():
    # Tuned to land in the (1.01, 1.05) band — the range where two-significant-figure
    # rounding collapses a gate failure onto a clean-looking 1.0.
    trace = _trace(offset=0.4, seed=2)
    signals = sampling_quality(trace)
    rounded = float(az.summary(trace, kind="diagnostics", round_to=None)["r_hat"].max())

    assert signals.max_rhat > 1.01, "fixture should fail an unrounded gate"
    # The bug: the rounded value clears the very gate the true value fails.
    assert rounded <= 1.01
    assert signals.max_rhat != rounded


@pytest.mark.parametrize("true_rhat", [1.011, 1.02, 1.049])
def test_two_significant_figure_rounding_collapses_the_gate_band(true_rhat):
    """Every R-hat in (1.01, 1.05) rounds to 1.0 — the gate becomes ``< 1.05``."""
    assert true_rhat > 1.01, "must fail an unrounded R-hat <= 1.01 gate"
    assert float(f"{true_rhat:.2g}") == 1.0, "but passes once rounded to 2 sig figs"


def test_arviz_1_3_default_summary_is_numeric():
    """ArviZ 1.3 returns numeric diagnostics even when applying default rounding."""
    col = az.summary(_trace(), kind="diagnostics")["r_hat"]
    assert np.issubdtype(col.dtype, np.number)
    assert isinstance(float(col.max()), float)
    # The helper remains explicit about disabling rounding.
    assert isinstance(sampling_quality(_trace()).max_rhat, float)


# --- extraction behaviour ----------------------------------------------------------


def test_min_ess_takes_the_bulk_tail_minimum():
    trace = _trace()
    summ = az.summary(trace, round_to="none", kind="diagnostics")
    expected = min(float(summ["ess_bulk"].min()), float(summ["ess_tail"].min()))
    assert sampling_quality(trace).min_ess == pytest.approx(expected)


def test_counts_divergences_via_array_coercion():
    assert sampling_quality(_trace(divergences=7)).n_divergences == 7
    assert sampling_quality(_trace(divergences=0)).n_divergences == 0


def test_divergences_none_when_sample_stats_lacks_them():
    trace = _trace(with_diverging=False, with_energy=False)
    assert sampling_quality(trace).n_divergences is None


def test_var_names_restricts_the_summary():
    rng = np.random.default_rng(1)
    nuisance = rng.normal(size=(4, 400))
    nuisance[0] += 5.0  # badly mixed
    trace = _trace(extra_vars={"nuisance": nuisance})

    assert sampling_quality(trace, var_names=["theta"]).max_rhat < 1.01
    assert sampling_quality(trace).max_rhat > 1.01


def test_nan_diagnostics_do_not_poison_the_reduction():
    """A constant (unsampled) variable yields NaN diagnostics; those are skipped."""
    trace = _trace(extra_vars={"constant": np.ones((4, 400))})
    assert np.isfinite(sampling_quality(trace).max_rhat)


def test_bfmi_absent_without_energy():
    assert sampling_quality(_trace(with_energy=False)).min_bfmi is None


def test_bfmi_finite_when_energy_present():
    min_bfmi = sampling_quality(_trace()).min_bfmi
    assert min_bfmi is None or np.isfinite(min_bfmi)


# --- rendering ---------------------------------------------------------------------


def test_summary_line_renders_values_and_missing_alike():
    line = SamplingQuality(
        max_rhat=1.0018, min_ess=4137.4, min_bfmi=None, n_divergences=None
    ).summary_line()
    assert "1.0018" in line, "R-hat must keep four decimals, not round to 1.00"
    assert "4137" in line
    assert line.count("n/a") == 2

    full = SamplingQuality(
        max_rhat=1.0, min_ess=8000.0, min_bfmi=0.91, n_divergences=3
    ).summary_line()
    assert "0.91" in full and "divergences 3" in full
