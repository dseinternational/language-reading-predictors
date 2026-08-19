# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The words-scale moderation contrast for moderated mechanism fits (2026-08-19).

``gamma_int`` is a product term on the logit scale; on a bounded outcome its sign
is not a statement about items. ``moderation_items.csv`` evaluates the fitted
surface in items at the interquartile cells and carries the logit-additive
benchmark, so the items-scale reading is computed rather than inferred.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.special import expit

from language_reading_predictors.statistical_models.pipelines.mechanism import (
    moderation_items_rows,
    write_moderation_items,
)
from language_reading_predictors.statistical_models.preprocessing import logit_safe


def _standardise(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std(ddof=1)


def _synthetic(
    *,
    gamma_int: float,
    gamma_mod: float = 0.4,
    beta: float = 0.8,
    alpha: float = -2.0,
    n_obs: int = 60,
    seed: int = 0,
    curve: str = "linear",
):
    """A linear-mechanism (or HSGP-shaped) posterior with two draws per chain and
    constant parameters, so every items-scale quantity can be hand-computed."""
    rng = np.random.default_rng(seed)
    exposure = rng.integers(0, 33, size=n_obs).astype(float)  # L counts of 32
    moderator = rng.integers(0, 11, size=n_obs).astype(float)  # B counts of 10
    z_x = _standardise(logit_safe(exposure, 32))
    z_m = _standardise(logit_safe(moderator, 10))
    chain, draw = 2, 2
    if curve == "linear":
        f = beta * z_x
    else:
        f = beta * np.tanh(z_x)  # any exposure-only curve
    eta = alpha + f + gamma_mod * z_m + gamma_int * z_x * z_m
    eta_full = np.broadcast_to(eta, (chain, draw, n_obs)).copy()
    data = {
        "eta": (("chain", "draw", "obs_id"), eta_full),
        "gamma_mod": (("chain", "draw"), np.full((chain, draw), gamma_mod)),
        "gamma_int": (("chain", "draw"), np.full((chain, draw), gamma_int)),
    }
    if curve == "linear":
        data["beta_mech"] = (("chain", "draw"), np.full((chain, draw), beta))
    else:
        data["f_mech"] = (
            ("chain", "draw", "f_mech_dim_0"),
            np.broadcast_to(f, (chain, draw, n_obs)).copy(),
        )
    post = xr.Dataset(
        data,
        coords={
            "chain": np.arange(chain),
            "draw": np.arange(draw),
            "obs_id": np.arange(n_obs),
        },
    )
    constant = xr.Dataset(
        {
            "z_mech_logit": (("obs_id",), z_x),
            "z_moderator": (("obs_id",), z_m),
        },
        coords={"obs_id": np.arange(n_obs)},
    )
    return post, constant, exposure, moderator, dict(alpha=alpha, beta=beta, f=f, z_x=z_x, z_m=z_m)


def _rows(post, constant, exposure, moderator):
    return moderation_items_rows(
        post,
        constant,
        exposure_counts=exposure,
        exposure_n_trials=32,
        moderator_values=moderator,
        moderator_n_trials=10,
        outcome_n_trials=79,
        ci_prob=0.89,
        exposure_symbol="L",
        moderator_symbol="B",
        outcome_symbol="W",
        moderator_unit="B items",
    )


def test_rows_match_a_hand_computation_on_the_linear_mechanism():
    gamma_int, gamma_mod, beta, alpha = -0.3, 0.4, 0.8, -2.0
    post, constant, exposure, moderator, t = _synthetic(
        gamma_int=gamma_int, gamma_mod=gamma_mod, beta=beta, alpha=alpha
    )
    rows = _rows(post, constant, exposure, moderator)
    by = {r["quantity"]: r for r in rows}
    inter = by["interaction"]
    x_lo, x_hi = inter["exposure_low"], inter["exposure_high"]
    m_lo, m_hi = inter["moderator_low"], inter["moderator_high"]
    # Cells are the interquartile values snapped to observed counts.
    assert x_lo in exposure and x_hi in exposure and m_lo in moderator and m_hi in moderator
    assert x_lo < x_hi and m_lo < m_hi
    # Hand computation: the factory's standardisation recovered exactly, the
    # curve at a cell is beta * z(x), and every other term is held at its fitted
    # value for each row (here eta_base == alpha for every row) then averaged.
    lx, lm = logit_safe(exposure, 32), logit_safe(moderator, 10)

    def zx(x):
        return (logit_safe(np.array([x]), 32)[0] - lx.mean()) / lx.std(ddof=1)

    def zm(m):
        return (logit_safe(np.array([m]), 10)[0] - lm.mean()) / lm.std(ddof=1)

    def expected(x, m, g):
        return 79 * expit(alpha + beta * zx(x) + gamma_mod * zm(m) + g * zx(x) * zm(m))

    inc_lo = expected(x_hi, m_lo, gamma_int) - expected(x_lo, m_lo, gamma_int)
    inc_hi = expected(x_hi, m_hi, gamma_int) - expected(x_lo, m_hi, gamma_int)
    assert by["increment_at_moderator_low"]["median"] == pytest.approx(inc_lo, abs=1e-9)
    assert by["increment_at_moderator_high"]["median"] == pytest.approx(inc_hi, abs=1e-9)
    assert inter["median"] == pytest.approx(inc_hi - inc_lo, abs=1e-9)
    bench = (expected(x_hi, m_hi, 0.0) - expected(x_lo, m_hi, 0.0)) - (
        expected(x_hi, m_lo, 0.0) - expected(x_lo, m_lo, 0.0)
    )
    assert by["interaction_if_logit_additive"]["median"] == pytest.approx(bench, abs=1e-9)
    assert by["interaction_logit"]["median"] == pytest.approx(
        gamma_int * (zx(x_hi) - zx(x_lo)) * (zm(m_hi) - zm(m_lo)), abs=1e-9
    )
    assert sum(r["quantity"] == "cell_mean" for r in rows) == 4
    assert {r["scale"] for r in rows} == {"items", "logit"}
    assert inter["exposure_unit"] == "L items" and inter["outcome_unit"] == "W items"
    assert inter["n_obs"] == 60 and inter["ci_prob"] == 0.89


def test_logit_additivity_is_items_synergy_below_the_midpoint_and_the_benchmark_equals_it():
    """With gamma_int = 0 the items-scale interaction IS the benchmark, and with two
    positive effects and p < 0.5 it is positive — the bounded-scale fact the table
    exists to make explicit (a negative gamma_int can be items-additivity)."""
    post, constant, exposure, moderator, _ = _synthetic(gamma_int=0.0, alpha=-2.5)
    by = {r["quantity"]: r for r in _rows(post, constant, exposure, moderator)}
    assert by["interaction"]["median"] == pytest.approx(
        by["interaction_if_logit_additive"]["median"], abs=1e-12
    )
    assert by["interaction"]["median"] > 0
    assert by["interaction_logit"]["median"] == 0.0


def test_hsgp_curve_is_read_off_the_fitted_rows():
    post, constant, exposure, moderator, t = _synthetic(gamma_int=-0.2, curve="hsgp")
    by = {r["quantity"]: r for r in _rows(post, constant, exposure, moderator)}
    inter = by["interaction"]
    x_lo, x_hi, m_lo, m_hi = (
        inter["exposure_low"],
        inter["exposure_high"],
        inter["moderator_low"],
        inter["moderator_high"],
    )
    def f_at(x):
        return t["f"][int(np.flatnonzero(exposure == x)[0])]

    def zx(x):
        return t["z_x"][int(np.flatnonzero(exposure == x)[0])]

    def zm(m):
        return t["z_m"][int(np.flatnonzero(moderator == m)[0])]

    def expected(x, m):
        return 79 * expit(t["alpha"] + f_at(x) + 0.4 * zm(m) - 0.2 * zx(x) * zm(m))

    want = (expected(x_hi, m_hi) - expected(x_lo, m_hi)) - (
        expected(x_hi, m_lo) - expected(x_lo, m_lo)
    )
    assert inter["median"] == pytest.approx(want, abs=1e-9)


def test_row_identity_guard_refuses_values_that_do_not_reproduce_the_fit():
    post, constant, exposure, moderator, _ = _synthetic(gamma_int=-0.1)
    wrong = moderator.copy()
    wrong[0] = 10.0 if wrong[0] != 10.0 else 9.0
    with pytest.raises(ValueError, match="moderator"):
        _rows(post, constant, exposure, wrong)
    with pytest.raises(ValueError, match="exposure"):
        _rows(post, constant, exposure[:-1], moderator[:-1])


def test_degenerate_cells_return_no_rows():
    post, constant, exposure, moderator, _ = _synthetic(gamma_int=-0.1)
    # A moderator constant on the fitted rows cannot be standardised by the
    # factory either; here it trips the guard before any cell is formed.
    with pytest.raises(ValueError, match="constant"):
        _rows(post, constant, exposure, np.full_like(moderator, 5.0))


def test_writer_uses_the_fitted_rows_and_skips_fits_without_an_interaction(tmp_path):
    post, constant, exposure, moderator, _ = _synthetic(gamma_int=-0.3)
    trace = SimpleNamespace(posterior=post, constant_data=constant)
    prepared = SimpleNamespace(
        post_counts={"L": exposure, "B": moderator},
        n_trials={"W": 79, "L": 32, "B": 10},
        A_months=np.zeros_like(exposure),
        covariates={},
        covariate_scalers={},
    )
    plan = SimpleNamespace(
        moderator_symbol="B",
        include_interaction=True,
        mechanism_is_covariate=False,
        mechanism_at_pre=False,
        moderator_is_covariate=False,
        mechanism_symbol="L",
        outcome_symbol="W",
    )
    ctx = SimpleNamespace(
        spec=SimpleNamespace(model_id="lrp-test-mech"),
        prepared=prepared,
        trace=trace,
        output_dir=str(tmp_path),
        reporting=SimpleNamespace(ci_prob=0.89),
    )
    import language_reading_predictors.statistical_models.pipelines.mechanism as mod

    orig = mod._mechanism_run_plan
    mod._mechanism_run_plan = lambda c: plan
    try:
        df = write_moderation_items(ctx)
        assert df is not None and len(df) == 9
        on_disk = pd.read_csv(tmp_path / "moderation_items.csv")
        assert list(on_disk["quantity"]) == list(df["quantity"])
        assert set(on_disk["quantity"]) >= {
            "increment_at_moderator_low",
            "increment_at_moderator_high",
            "interaction",
            "interaction_if_logit_additive",
            "interaction_logit",
        }
        plan.include_interaction = False
        assert write_moderation_items(ctx) is None
        plan.include_interaction = True
        plan.mechanism_is_covariate = True
        assert write_moderation_items(ctx) is None
    finally:
        mod._mechanism_run_plan = orig
