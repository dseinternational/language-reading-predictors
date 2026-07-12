# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for diagnostics helpers (issue #125 Area 3 / step 0b)."""

from __future__ import annotations

from types import SimpleNamespace

import arviz as az
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


def _synthetic_trace(shift, *, n=800, chains=4, seed=1, n_div=0):
    """A DataTree with a tunable between-chain mean shift and divergence count.

    ``shift`` sets each chain's mean to ``shift * chain_index``, so a larger shift
    pushes R-hat up while ESS stays comfortable — enough to land R-hat in the
    (1.01, 1.05) band that the rounding bug (issue #274 item 1) would hide.
    """
    rng = np.random.default_rng(seed)
    draws = np.stack([rng.normal(loc=shift * c, scale=1.0, size=n) for c in range(chains)])
    post = xr.Dataset(
        {"tau": (("chain", "draw"), draws)},
        coords={"chain": range(chains), "draw": range(n)},
    )
    div = np.zeros((chains, n), dtype=bool)
    if n_div:
        div.reshape(-1)[:n_div] = True
    ss = xr.Dataset(
        {"diverging": (("chain", "draw"), div)},
        coords={"chain": range(chains), "draw": range(n)},
    )
    return xr.DataTree.from_dict({"posterior": post, "sample_stats": ss})


def test_subfit_convergence_gates_on_unrounded_rhat():
    # Regression for issue #274 item 1: a true max R-hat in (1.01, 1.05) must FAIL
    # the gate. The bug was az.summary(round_to=None) rounding to 2 sig figs, so a
    # 1.0156 R-hat rounded to 1.0 and slipped through the <= 1.01 gate.
    dt = _synthetic_trace(0.12)
    res = diag.subfit_convergence(dt, label="borderline", var_names=["tau"])

    # Core regression check (independent of ArviZ's rounding behaviour): the gate
    # reports the UNROUNDED max R-hat — it matches an explicit round_to="none"
    # reference — and therefore fails.
    ref = float(az.summary(dt, var_names=["tau"], round_to="none", kind="diagnostics")["r_hat"].max())
    assert res["max_rhat"] == pytest.approx(ref)
    assert diag.RHAT_MAX < res["max_rhat"] < 1.05  # genuinely in the hidden band
    assert res["min_ess"] >= diag.ESS_THRESHOLD  # so only R-hat can fail the gate
    assert res["converged"] is False

    # Illustration (not load-bearing): with ArviZ's current default 2-sig-fig
    # rounding the same R-hat rounds to 1.0 and would have slipped the <= 1.01 gate.
    # Guarded so a future ArviZ change to the round_to=None default cannot break the
    # regression test above.
    rounded = float(az.summary(dt, var_names=["tau"], round_to=None, kind="diagnostics")["r_hat"].max())
    if rounded != pytest.approx(ref):  # ArviZ still rounds round_to=None
        assert rounded <= diag.RHAT_MAX  # would have slipped through


def test_subfit_convergence_passes_clean_and_flags_divergences():
    clean = diag.subfit_convergence(_synthetic_trace(0.0), label="clean", var_names=["tau"])
    assert clean["converged"] is True
    assert clean["n_divergences"] == 0

    div = diag.subfit_convergence(_synthetic_trace(0.0, n_div=3), label="div", var_names=["tau"])
    assert div["n_divergences"] == 3
    assert div["converged"] is False  # zero-divergence gate is strict


def test_gate_var_names_unions_free_rvs_with_curated_and_filters_present():
    # Issue #274 item 2: the gate must scan the model's free RVs (incl. the
    # per-child intercept vector) unioned with the curated headline terms, and
    # drop any name a given fit does not instantiate.
    rv = lambda name: SimpleNamespace(name=name)  # noqa: E731
    model = SimpleNamespace(free_RVs=[rv("mu"), rv("u_child_raw"), rv("sigma_child")])
    post = xr.Dataset(
        {
            k: (("chain", "draw"), np.zeros((2, 5)))
            for k in ("mu", "u_child_raw", "sigma_child", "tau")
        }
    )
    ctx = SimpleNamespace(model=model, trace=SimpleNamespace(posterior=post))

    names = diag._gate_var_names(ctx, ["tau", "beta_absent"])

    assert "u_child_raw" in names  # free RV now gated
    assert "tau" in names  # curated headline deterministic kept
    assert "beta_absent" not in names  # not present in posterior -> dropped
    assert len(names) == len(set(names))  # de-duplicated


def test_gate_var_names_falls_back_without_model():
    ctx = SimpleNamespace(model=None, trace=None)
    assert diag._gate_var_names(ctx, ["tau"]) == ["tau"]


def test_thin_posterior_only_keeps_prior_full():
    # Issue #270 item 1: thinning must not decimate the small 1-chain prior group.
    dt = xr.DataTree.from_dict(
        {
            "posterior": xr.Dataset(
                {"tau": (("chain", "draw"), np.zeros((6, 6000)))},
                coords={"chain": range(6), "draw": range(6000)},
            ),
            "prior": xr.Dataset(
                {"tau": (("chain", "draw"), np.zeros((1, 1000)))},
                coords={"chain": range(1), "draw": range(1000)},
            ),
        }
    )
    thinned = diag.thin_posterior_only(dt, max_draws=1000)
    post_total = thinned.posterior.sizes["chain"] * thinned.posterior.sizes["draw"]
    assert post_total <= 1100  # posterior thinned
    assert post_total < 36000
    assert thinned.prior.sizes["draw"] == 1000  # prior untouched (was the bug)

    # A small posterior is returned unchanged.
    small = xr.DataTree.from_dict(
        {"posterior": xr.Dataset({"tau": (("chain", "draw"), np.zeros((2, 250)))})}
    )
    assert diag.thin_posterior_only(small, max_draws=1000) is small
