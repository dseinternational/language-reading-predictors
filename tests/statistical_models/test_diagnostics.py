# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for diagnostics helpers (issue #125 Area 3 / step 0b)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from language_reading_predictors.statistical_models import diagnostics as diag


def test_run_psense_removes_stale_summary_when_recomputation_fails(
    tmp_path: Path,
    monkeypatch,
):
    import arviz_stats as azs

    summary_path = tmp_path / "psense_summary.csv"
    summary_path.write_text("stale\n", encoding="utf-8")
    context = SimpleNamespace(
        output_dir=str(tmp_path),
        trace=object(),
        tables={"psense_summary": "stale"},
    )

    def _fail(*_args, **_kwargs):
        raise ValueError("diagnostic failed")

    monkeypatch.setattr(azs, "psense_summary", _fail)
    monkeypatch.setattr(diag, "_save_pc", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(diag, "thin_for_plots", lambda trace: trace)

    diag.run_psense(context, var_names=["tau"])

    assert not summary_path.exists()
    assert "psense_summary" not in context.tables


def test_run_psense_atomically_replaces_summary(
    tmp_path: Path,
    monkeypatch,
):
    import arviz_stats as azs

    context = SimpleNamespace(
        output_dir=str(tmp_path),
        trace=object(),
        tables={},
    )
    expected = pd.DataFrame({"diagnosis": ["✓"]}, index=["tau"])
    monkeypatch.setattr(azs, "psense_summary", lambda *_args, **_kwargs: expected)
    monkeypatch.setattr(diag, "_save_pc", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(diag, "thin_for_plots", lambda trace: trace)
    real_replace = diag.os.replace
    replacements: list[tuple[Path, Path]] = []

    def _record_replace(source, destination):
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(diag.os, "replace", _record_replace)

    diag.run_psense(context, var_names=["tau"])

    summary_path = tmp_path / "psense_summary.csv"
    assert len(replacements) == 1
    assert replacements[0][1] == summary_path
    assert replacements[0][0].parent == tmp_path
    assert replacements[0][0].name.startswith(".psense_summary-")
    assert summary_path.is_file()
    assert not replacements[0][0].exists()
    assert context.tables["psense_summary"].equals(expected)


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


def test_joint_log_likelihood_is_aggregated_by_child():
    # Four flattened cells: child 0 has two outcomes, child 1 one, child 2 one.
    values = np.array([[[1.0, 2.0, 4.0, 8.0], [10.0, 20.0, 40.0, 80.0]]])
    trace = xr.DataTree.from_dict(
        {
            "log_likelihood": xr.Dataset(
                {"y_post": (("chain", "draw", "cell"), values)}
            ),
            "constant_data": xr.Dataset(
                {
                    "G": ("obs_id", np.array([1.0, 0.0, 1.0])),
                    "y_post_cell_row": ("cell", np.array([0, 0, 1, 2])),
                }
            ),
        }
    )
    got = diag._joint_log_likelihood_by_child(trace)
    assert got is not None
    np.testing.assert_allclose(got.values, [[[3.0, 4.0, 8.0], [30.0, 40.0, 80.0]]])
    assert got.attrs["loo_unit"] == "child"


def test_joint_predictive_selection_never_pools_outcome_denominators():
    # Cells are interleaved A, B, A, B. Selecting B must return exactly the two B
    # columns, not all four counts in one incompatible-denominator histogram.
    values = np.array([[[1.0, 101.0, 2.0, 102.0]]])
    prior = xr.Dataset(
        {"tau": (("chain", "draw", "outcome"), np.zeros((1, 1, 2)))},
        coords={"outcome": ["A", "B"]},
    )
    samples = SimpleNamespace(
        prior=prior,
        prior_predictive=xr.Dataset(
            {"y_post": (("chain", "draw", "cell"), values)}
        ),
        constant_data=xr.Dataset(
            {"y_post_cell_outcome": ("cell", np.array([0, 1, 0, 1]))}
        ),
    )
    context = SimpleNamespace(
        prior_samples=samples,
        trace=None,
        spec=SimpleNamespace(extra={"outcomes": ("A", "B")}),
        model=None,
    )
    selected, symbol = diag._predictive_values_for_outcome(
        context,
        samples,
        group="prior_predictive",
        node="y_post",
        outcome_symbol="B",
    )
    assert symbol == "B"
    np.testing.assert_array_equal(selected, np.array([[[101.0, 102.0]]]))


def test_joint_predictive_selection_uses_one_coordinate_fallback():
    values = np.array([[[1.0, 101.0, 2.0, 102.0]]])
    prior = xr.Dataset(
        {"tau": (("chain", "draw", "outcome"), np.zeros((1, 1, 2)))},
        coords={"outcome": ["A", "B"]},
    )
    samples = SimpleNamespace(
        prior=prior,
        prior_predictive=xr.Dataset(
            {"y_post": (("chain", "draw", "cell"), values)}
        ),
        constant_data=xr.Dataset(
            {"y_post_cell_outcome": ("cell", np.array([0, 1, 0, 1]))}
        ),
    )
    context = SimpleNamespace(
        prior_samples=samples,
        trace=None,
        spec=SimpleNamespace(extra={}),
        model=None,
    )

    selected, symbol = diag._predictive_values_for_outcome(
        context,
        samples,
        group="prior_predictive",
        node="y_post",
        outcome_symbol="B",
    )

    assert symbol == "B"
    np.testing.assert_array_equal(selected, np.array([[[101.0, 102.0]]]))


def test_joint_predictive_selection_fails_closed_on_bad_map():
    samples = SimpleNamespace(
        prior=xr.Dataset(
            {"tau": (("chain", "draw", "outcome"), np.zeros((1, 1, 2)))},
            coords={"outcome": ["A", "B"]},
        ),
        prior_predictive=xr.Dataset(
            {"y_post": (("chain", "draw", "cell"), np.zeros((1, 1, 4)))}
        ),
        constant_data=xr.Dataset(
            {"y_post_cell_outcome": ("bad_cell", np.array([0, 1, 0]))}
        ),
    )
    context = SimpleNamespace(
        prior_samples=samples,
        trace=None,
        spec=SimpleNamespace(extra={"outcomes": ("A", "B")}),
        model=None,
    )
    with pytest.raises(ValueError, match="does not align"):
        diag._predictive_values_for_outcome(
            context,
            samples,
            group="prior_predictive",
            node="y_post",
            outcome_symbol="A",
        )


def test_predictive_histogram_uses_identical_count_bins(monkeypatch):
    calls = []

    def record_hist(_values, *, bins, **_kwargs):
        calls.append(np.asarray(bins))

    monkeypatch.setattr(diag.plt, "hist", record_hist)

    diag._overlay_count_histograms(
        np.array([0.0, 1.0, 5.0]),
        np.array([0.0, 2.0, 3.0]),
        predictive_label="posterior predictive",
    )

    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0], calls[1])
    np.testing.assert_array_equal(calls[0], np.arange(-0.5, 6.0, 1.0))


def test_joint_loo_pit_tree_selects_matching_outcome_cells():
    posterior = xr.Dataset(
        {
            "tau": (
                ("chain", "draw", "outcome"),
                np.zeros((1, 3, 2)),
            )
        },
        coords={"chain": [0], "draw": range(3), "outcome": ["A", "B"]},
    )
    observed = xr.Dataset({"y_post": ("cell", np.array([1, 101, 2, 102]))})
    replicated = xr.Dataset(
        {
            "y_post": (
                ("chain", "draw", "cell"),
                np.arange(12).reshape(1, 3, 4),
            )
        }
    )
    log_likelihood = xr.Dataset(
        {
            "y_post": (
                ("chain", "draw", "cell"),
                -np.arange(12, dtype=float).reshape(1, 3, 4),
            )
        }
    )
    trace = xr.DataTree.from_dict(
        {
            "posterior": posterior,
            "observed_data": observed,
            "posterior_predictive": replicated,
            "log_likelihood": log_likelihood,
            "constant_data": xr.Dataset(
                {"y_post_cell_outcome": ("cell", np.array([0, 1, 0, 1]))}
            ),
        }
    )
    context = SimpleNamespace(
        trace=trace,
        prior_samples=None,
        spec=SimpleNamespace(extra={}),
        model=None,
    )

    selected = diag._joint_outcome_predictive_tree(context, "B")

    np.testing.assert_array_equal(
        selected.observed_data["y_post"].values, np.array([101, 102])
    )
    np.testing.assert_array_equal(
        selected.posterior_predictive["y_post"].values,
        replicated["y_post"].values[..., [1, 3]],
    )
    np.testing.assert_array_equal(
        selected.log_likelihood["y_post"].values,
        log_likelihood["y_post"].values[..., [1, 3]],
    )


def _synthetic_trace(
    shift, *, n=800, chains=4, seed=1, n_div=0, kappa_shift=None
):
    """A DataTree with a tunable between-chain mean shift and divergence count.

    ``shift`` sets each chain's mean to ``shift * chain_index``, so a larger shift
    pushes R-hat up while ESS stays comfortable — enough to land R-hat in the
    (1.01, 1.05) band that the rounding bug (issue #274 item 1) would hide.
    """
    rng = np.random.default_rng(seed)
    draws = np.stack([rng.normal(loc=shift * c, scale=1.0, size=n) for c in range(chains)])
    variables = {"tau": (("chain", "draw"), draws)}
    if kappa_shift is not None:
        kappa = np.stack(
            [
                rng.normal(loc=kappa_shift * c, scale=1.0, size=n)
                for c in range(chains)
            ]
        )
        variables["kappa"] = (("chain", "draw"), kappa)
    post = xr.Dataset(
        variables,
        coords={"chain": range(chains), "draw": range(n)},
    )
    div = np.zeros((chains, n), dtype=bool)
    if n_div:
        div.reshape(-1)[:n_div] = True
    energy = rng.normal(size=(chains, n))
    ss = xr.Dataset(
        {
            "diverging": (("chain", "draw"), div),
            "energy": (("chain", "draw"), energy),
        },
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
    assert clean["min_bfmi"] >= diag.BFMI_THRESHOLD
    assert clean["n_divergences"] == 0

    div = diag.subfit_convergence(_synthetic_trace(0.0, n_div=3), label="div", var_names=["tau"])
    assert div["n_divergences"] == 3
    assert div["converged"] is False  # zero-divergence gate is strict


def test_subfit_convergence_catches_bad_nuisance_parameter():
    """A well-mixed tau must not hide a non-mixing kappa in a secondary fit."""
    trace = _synthetic_trace(0.0, kappa_shift=0.5)

    tau_only = diag.subfit_convergence(trace, label="tau-only", var_names=["tau"])
    complete = diag.subfit_convergence(
        trace, label="all-free-rvs", var_names=["tau", "kappa"]
    )

    assert tau_only["converged"] is True
    assert complete["max_rhat"] > diag.RHAT_MAX
    assert complete["converged"] is False


def test_subfit_convergence_flags_low_bfmi(monkeypatch):
    monkeypatch.setattr(diag, "_bfmi_per_chain", lambda _trace: np.asarray([0.2, 0.8]))
    result = diag.subfit_convergence(
        _synthetic_trace(0.0), label="low-bfmi", var_names=["tau"]
    )
    assert result["min_bfmi"] == pytest.approx(0.2)
    assert result["converged"] is False


def test_subfit_convergence_marks_diagnostic_errors_unchecked(monkeypatch):
    def fail_summary(*_args, **_kwargs):
        raise RuntimeError("synthetic diagnostics failure")

    monkeypatch.setattr(diag.az, "summary", fail_summary)
    result = diag.subfit_convergence(
        _synthetic_trace(0.0), label="uncheckable", var_names=["tau"]
    )
    assert result == {
        "converged": None,
        "max_rhat": None,
        "min_ess": None,
        "min_bfmi": None,
        "n_divergences": None,
    }


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


def test_prior_posterior_overlay_raises_subplot_limit_for_curated_vectors(
    monkeypatch, tmp_path
):
    # The full joint ITT overlay has five ten-outcome arrays (50 panels), above
    # ArviZ's default 40-panel safety limit.  An explicit curated selection must
    # render without permanently changing the process-wide rcParams setting.
    variables = {
        name: (("chain", "draw", "outcome"), np.zeros((1, 2, 10)))
        for name in ("alpha", "tau", "gamma_own", "kappa", "gamma_A")
    }
    trace = xr.DataTree.from_dict(
        {
            "posterior": xr.Dataset(variables),
            "prior": xr.Dataset(variables),
        }
    )
    context = SimpleNamespace(output_dir=str(tmp_path), trace=trace)
    observed: dict[str, object] = {}

    import arviz_plots as azp

    def fake_plot_prior_posterior(_trace, *, var_names, **kwargs):
        observed["limit"] = az.rcParams["plot.max_subplots"]
        observed["var_names"] = var_names
        observed["plot_kwargs"] = kwargs
        return object()

    monkeypatch.setattr(azp, "plot_prior_posterior", fake_plot_prior_posterior)
    monkeypatch.setattr(
        diag,
        "_save_pc",
        lambda _out, make, _name, title=None: observed.update(
            result=make(), title=title
        ),
    )
    original_limit = az.rcParams["plot.max_subplots"]
    selected = ["alpha", "tau", "gamma_own", "kappa", "gamma_A"]

    diag.save_prior_posterior_plot(context, var_names=selected)

    assert observed["limit"] == 50
    assert observed["var_names"] == selected
    assert observed["plot_kwargs"] == {
        "col_wrap": 5,
        "figure_kwargs": {
            "figsize": (22.0, 34.0),
            "gridspec_kw": {"hspace": 0.85, "wspace": 0.25},
        },
    }
    assert az.rcParams["plot.max_subplots"] == original_limit
