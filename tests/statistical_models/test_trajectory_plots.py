# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for :mod:`statistical_models.trajectory_plots` (#317).

Covers the population g-computation arithmetic behind the group trajectory, the
seeded worst-k child selection for the small multiples, the floor-rule branch, and
end-to-end artefact emission for both the obs_id families and the masked panel
families.
"""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import xarray as xr
from scipy.special import expit

from language_reading_predictors.statistical_models import trajectory_plots as tp


# ---------------------------------------------------------------------------
# Synthetic traces / panels
# ---------------------------------------------------------------------------


def _obsid_trace(
    *,
    n_chain=2,
    n_draw=50,
    n_children=8,
    n_waves=4,
    seed=3,
    off_floor=False,
    sigma=0.7,
):
    """obs_id trace: one row per child per wave, ``eta``/``u_child``/``sigma_child``
    in ``posterior`` and the replicated + observed outcome node in the PPC groups."""
    rng = np.random.default_rng(seed)
    n_obs = n_children * n_waves
    child_idx = np.repeat(np.arange(n_children), n_waves)
    wave = np.tile(np.arange(n_waves), n_children)
    arm = (child_idx % 2).astype(int)  # alternate arms, constant within child

    u_vals = rng.normal(0.0, sigma, size=(n_chain, n_draw, n_children))
    base = rng.normal(-0.5 + 0.3 * wave, 0.2, size=(n_chain, n_draw, n_obs))
    eta = base + u_vals[:, :, child_idx]
    sigma_child = np.full((n_chain, n_draw), sigma)

    obs_node = "y_offfloor" if off_floor else "y_post"
    n_trials = 1 if off_floor else 30
    p = expit(eta)
    ppc = rng.binomial(n_trials, p).astype(float)
    observed = ppc[0, 0].copy()

    posterior = xr.Dataset(
        {
            "eta": (("chain", "draw", "obs_id"), eta),
            "u_child": (("chain", "draw", "child"), u_vals),
            "sigma_child": (("chain", "draw"), sigma_child),
        },
        coords={
            "chain": np.arange(n_chain),
            "draw": np.arange(n_draw),
            "obs_id": np.arange(n_obs),
            "child": np.arange(n_children),
        },
    )
    pp = xr.Dataset(
        {obs_node: (("chain", "draw", "obs_id"), ppc)},
        coords={"chain": np.arange(n_chain), "draw": np.arange(n_draw), "obs_id": np.arange(n_obs)},
    )
    od = xr.Dataset({obs_node: (("obs_id",), observed)}, coords={"obs_id": np.arange(n_obs)})
    trace = SimpleNamespace(posterior=posterior, posterior_predictive=pp, observed_data=od)
    return trace, dict(arm=arm, wave=wave, child_idx=child_idx, n_obs=n_obs)


def _panel_and_trace(*, n_chain=2, n_draw=40, n_children=6, seed=5):
    """A minimal ``WavePanel``-like object plus a latent-grid trace (growth/lcsm)."""
    rng = np.random.default_rng(seed)
    outcomes = ("W", "L")
    waves = [1, 2, 3, 4]
    T, K = len(waves), len(outcomes)
    n_trials = {"W": 30, "L": 32}
    counts, obs_mask = {}, {}
    for s in outcomes:
        c = rng.integers(0, n_trials[s], size=(n_children, T)).astype(float)
        m = rng.random((n_children, T)) > 0.1  # ~10% missing
        c[~m] = np.nan
        counts[s] = c
        obs_mask[s] = m
    panel = SimpleNamespace(
        outcomes=outcomes,
        waves=waves,
        n_children=n_children,
        n_trials=n_trials,
        counts=counts,
        obs_mask=obs_mask,
    )
    latent = rng.normal(-0.5, 0.6, size=(n_chain, n_draw, n_children, T, K))
    kappa = rng.uniform(20, 60, size=(n_chain, n_draw, K))
    posterior = xr.Dataset(
        {
            "x_latent": (("chain", "draw", "child", "wave", "outcome"), latent),
            "kappa": (("chain", "draw", "outcome"), kappa),
        },
        coords={
            "chain": np.arange(n_chain),
            "draw": np.arange(n_draw),
            "child": np.arange(n_children),
            "wave": waves,
            "outcome": list(outcomes),
        },
    )
    trace = SimpleNamespace(posterior=posterior)
    return panel, trace


# ---------------------------------------------------------------------------
# select_children
# ---------------------------------------------------------------------------


def test_select_children_reproducible_and_includes_worst():
    k = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.05, 0.4])
    a, worst_a = tp.select_children(k, n_children=8, n_total=5, n_worst=3, seed=42)
    b, worst_b = tp.select_children(k, n_children=8, n_total=5, n_worst=3, seed=42)
    assert a == b and worst_a == worst_b  # seed -> identical
    assert worst_a == {1, 3, 5}  # the three highest-k children
    assert worst_a.issubset(set(a))
    assert len(a) == 5 and len(set(a)) == 5  # capped, de-duplicated


def test_select_children_no_pareto_k_is_pure_random():
    a, worst = tp.select_children(None, n_children=10, n_total=4, seed=7)
    assert worst == set()
    assert len(a) == 4
    b, _ = tp.select_children(None, n_children=10, n_total=4, seed=7)
    assert a == b


def test_select_children_caps_to_available():
    a, _ = tp.select_children(None, n_children=3, n_total=12, seed=1)
    assert sorted(a) == [0, 1, 2]


# ---------------------------------------------------------------------------
# marginal cell probabilities (g-computation) and observed means
# ---------------------------------------------------------------------------


def test_gh_marginal_symmetric_zero_is_half():
    eta0 = np.zeros((4, 6))
    p = tp._gh_marginal_prob(eta0, sigma=np.full(6, 1.3), n_gh=20)
    np.testing.assert_allclose(p, 0.5, atol=1e-10)


def test_gh_marginal_no_sigma_is_expit():
    eta0 = np.array([[-1.0, 0.5], [2.0, -0.3]])
    np.testing.assert_allclose(tp._gh_marginal_prob(eta0, sigma=None, n_gh=16), expit(eta0))


def test_marginal_cell_probabilities_gcomputation_no_re():
    # With no random effect the cell value is the plain mean of expit(eta) over the
    # cell's rows — the g-computation average.
    eta = np.array(
        [
            [0.0, 1.0],  # arm0 wave0
            [0.5, -0.5],  # arm0 wave0
            [1.0, 2.0],  # arm1 wave0
        ]
    )
    arm = np.array([0, 0, 1])
    wave = np.array([0, 0, 0])
    cells = tp.marginal_cell_probabilities(eta, arm=arm, wave=wave)
    np.testing.assert_allclose(cells[(0, 0)], expit(eta[:2]).mean(axis=0))
    np.testing.assert_allclose(cells[(1, 0)], expit(eta[2:3]).mean(axis=0))


def test_marginal_cell_probabilities_removes_child_intercept():
    # sigma_child = 0: integrating adds nothing, so the marginal prob is expit(eta - u).
    n_obs, S = 4, 3
    eta = np.full((n_obs, S), 2.0)
    u_rows = np.tile(np.array([2.0, 2.0, -1.0, -1.0])[:, None], (1, S))
    cells = tp.marginal_cell_probabilities(
        eta, arm=np.array([0, 0, 1, 1]), wave=np.zeros(n_obs, int),
        u_child_rows=u_rows, sigma_child=np.zeros(S),
    )
    np.testing.assert_allclose(cells[(0, 0)], expit(0.0))  # eta - u = 0
    np.testing.assert_allclose(cells[(1, 0)], expit(3.0))


def test_observed_cell_means():
    obs = np.array([2.0, 4.0, 10.0, 20.0])
    means = tp.observed_cell_means(obs, arm=np.array([0, 0, 1, 1]), wave=np.zeros(4, int))
    assert means[(0, 0)] == pytest.approx(3.0)
    assert means[(1, 0)] == pytest.approx(15.0)


def test_child_predictive_bands_shapes_and_order():
    ppc = np.arange(3 * 100, dtype=float).reshape(3, 100)
    bands = tp.child_predictive_bands(ppc, ci_prob=0.9)
    for key in ("median", "lo", "hi", "lo50", "hi50"):
        assert bands[key].shape == (3,)
    assert np.all(bands["lo"] <= bands["median"]) and np.all(bands["median"] <= bands["hi"])


# ---------------------------------------------------------------------------
# obs_id writers (end-to-end)
# ---------------------------------------------------------------------------


def test_group_trajectory_writer_graded(tmp_path):
    trace, d = _obsid_trace()
    summary = tp.write_group_arm_trajectory(
        str(tmp_path), trace, arm=d["arm"], wave=d["wave"], child_idx=d["child_idx"],
        n_trials=30, outcome_symbol="W", item_label="Word reading", off_floor=False,
    )
    assert (tmp_path / "group_trajectory.png").exists()
    assert (tmp_path / "group_trajectory.svg").exists()
    assert (tmp_path / "group_trajectory.csv").exists()
    assert {"arm", "wave", "observed_mean", "predicted_median", "predicted_lo", "predicted_hi"} <= set(summary.columns)
    # Two arms x four waves.
    assert len(summary) == 8
    # Items scale: medians within [0, n_trials].
    assert summary["predicted_median"].between(0, 30).all()


def test_group_trajectory_writer_off_floor_is_probability(tmp_path):
    trace, d = _obsid_trace(off_floor=True)
    summary = tp.write_group_arm_trajectory(
        str(tmp_path), trace, arm=d["arm"], wave=d["wave"], child_idx=d["child_idx"],
        n_trials=1, outcome_symbol="P", item_label="Phonetic spelling", off_floor=True,
        obs_node="y_offfloor",
    )
    assert (summary["scale"] == "off_floor_probability").all()
    assert summary["predicted_median"].between(0, 1).all()
    assert summary["observed_mean"].between(0, 1).all()


def test_child_fit_obsid_includes_worst_and_writes(tmp_path):
    trace, d = _obsid_trace(n_children=10)
    # Force child 7 to be the worst-fitting.
    pk = np.full(d["n_obs"], 0.1)
    pk[d["child_idx"] == 7] = 0.95
    summary = tp.write_child_fit_obsid(
        str(tmp_path), trace, wave=d["wave"], child_idx=d["child_idx"], n_trials=30,
        outcome_symbol="W", item_label="Word reading", pareto_k=pk, seed=11,
    )
    assert (tmp_path / "child_fit_panels.png").exists()
    assert (tmp_path / "child_fit_panels.csv").exists()
    assert 7 in set(summary.loc[summary.worst_fitting, "child_index"])
    # Reproducible child sample across identical seeds.
    s2 = tp.write_child_fit_obsid(
        str(tmp_path / "again"), trace, wave=d["wave"], child_idx=d["child_idx"],
        n_trials=30, outcome_symbol="W", item_label="Word reading", pareto_k=pk, seed=11,
    )
    assert list(summary["child_index"].unique()) == list(s2["child_index"].unique())


def test_child_fit_obsid_off_floor_probability_scale(tmp_path):
    trace, d = _obsid_trace(off_floor=True)
    summary = tp.write_child_fit_obsid(
        str(tmp_path), trace, wave=d["wave"], child_idx=d["child_idx"], n_trials=1,
        outcome_symbol="P", item_label="Phonetic spelling", off_floor=True,
        obs_node="y_offfloor",
    )
    assert (summary["scale"] == "off_floor_probability").all()
    assert summary["predicted_median"].between(0, 1).all()


# ---------------------------------------------------------------------------
# panel writers (growth / lcsm)
# ---------------------------------------------------------------------------


def test_outcome_trajectory_writer(tmp_path):
    panel, trace = _panel_and_trace()
    summary = tp.write_outcome_trajectory(
        str(tmp_path), trace, panel, latent_name="x_latent", name="group_trajectory"
    )
    assert (tmp_path / "group_trajectory.png").exists()
    assert (tmp_path / "group_trajectory.csv").exists()
    assert set(summary["outcome"].unique()) == {"W", "L"}
    assert len(summary) == 2 * len(panel.waves)


def test_child_fit_panel_writer(tmp_path):
    panel, trace = _panel_and_trace(n_children=8)
    # Pareto-k over the flattened observed cells (nonzero-mask order).
    idx_i = tp._panel_child_index(panel)
    pk = np.full(idx_i.shape[0], 0.1)
    pk[idx_i == 5] = 0.99  # child 5 worst
    summary = tp.write_child_fit_panel(
        str(tmp_path), trace, panel, latent_name="x_latent", focal_symbol="W",
        kappa_name="kappa", pareto_k=pk, seed=9,
    )
    assert (tmp_path / "child_fit_panels.png").exists()
    assert (tmp_path / "child_fit_panels.csv").exists()
    assert (summary["outcome"] == "W").all()
    assert 5 in set(summary.loc[summary.worst_fitting, "child_index"])
