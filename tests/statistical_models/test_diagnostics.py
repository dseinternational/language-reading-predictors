# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for diagnostics helpers (issue #125 Area 3 / step 0b)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

from dse_research_utils.statistics import sampling_quality as sampling_quality_mod
from language_reading_predictors.statistical_models import diagnostics as diag
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.reporting import (
    REUSE_CONTRACT_KEY,
    _reuse_compatibility_contract,
    require_reuse_compatibility,
    write_run_metadata,
)


def _primary_reuse_context(tmp_path):
    source = tmp_path / "published"
    current = tmp_path / "current"
    source.mkdir()
    current.mkdir()
    observed = np.asarray([2, 4, 3])
    with pm.Model() as model:
        p = pm.Beta("p", 2.0, 2.0)
        pm.Binomial("y", n=5, p=p, observed=observed)
    context = SimpleNamespace(
        spec=ModelSpec(
            model_id="lrp-rli-hg-999",
            kind="historical_growth",
            title="primary reuse contract",
        ),
        prepared=SimpleNamespace(
            subject_ids=np.asarray(["S1", "S2", "S3"], dtype=object),
            n_obs=3,
            n_children=3,
            n_phases=1,
            n_waves=None,
            dropped_rows=0,
            dropped_by_reason={},
            data_sha256="a" * 64,
        ),
        model=model,
        reporting=SimpleNamespace(config_name="reporting", ci_prob=0.89),
        sampling=SimpleNamespace(
            draws=6000,
            tune=6000,
            chains=6,
            cores=5,
            target_accept=0.95,
            random_seed=47,
        ),
        resolved_plan=None,
        output_dir=str(current),
        final_output_dir=str(source),
    )
    trace_path = source / "trace.nc"
    trace_path.write_bytes(b"persisted posterior bytes")
    # Publish through the real writer (#637 stage 1). The fixture used to
    # serialise ``_reuse_compatibility_contract`` straight into ``config.json``,
    # which is why nothing noticed that ``write_run_metadata`` never persisted
    # ``model_design_identity``: the reader was only ever shown a config the
    # writer could not have produced.
    write_run_metadata(replace_namespace(context, output_dir=str(source)))
    return context, source


def replace_namespace(context: SimpleNamespace, **overrides) -> SimpleNamespace:
    """A copy of a namespace fit context with selected attributes replaced."""

    return SimpleNamespace(**{**vars(context), **overrides})


def test_reuse_trace_never_falls_back_to_fresh_sampling(tmp_path, monkeypatch):
    context = SimpleNamespace(final_output_dir=str(tmp_path))
    monkeypatch.setenv("DSE_LRP_REUSE_TRACE", "1")
    monkeypatch.setattr(
        diag.pm,
        "sample",
        lambda **_kwargs: pytest.fail("reuse-trace must never run fresh NUTS"),
    )

    with pytest.raises(FileNotFoundError, match="refusing to run fresh NUTS"):
        diag.sample_posterior(context)


def test_metadata_written_by_a_normal_fit_passes_reuse_validation(tmp_path):
    """The writer-to-reader round trip, with nothing changed in between.

    ``_REUSE_CONFIG_FIELDS`` named ``model_design_identity`` and
    ``_reuse_compatibility_contract`` computed it, but ``write_run_metadata``
    persisted only ``fitted_data_identity`` — so an unchanged round trip failed
    with a field the fit had never had the chance to record (#637 stage 1).
    """
    context, source = _primary_reuse_context(tmp_path)
    stored = json.loads((source / "config.json").read_text(encoding="utf-8"))
    assert stored[REUSE_CONTRACT_KEY]["model_design_identity"]["structure_sha256"]

    require_reuse_compatibility(context, source)


def test_the_documented_field_list_matches_the_serialised_contract(tmp_path):
    """The doc list and the contract cannot drift apart unnoticed."""
    from language_reading_predictors.statistical_models.reporting import (
        _REUSE_CONFIG_FIELDS,
    )

    context, _source = _primary_reuse_context(tmp_path)
    contract = _reuse_compatibility_contract(context)
    assert set(_REUSE_CONFIG_FIELDS) == set(contract) - {"schema_version"}


def test_reuse_is_refused_when_the_contract_schema_version_moves(tmp_path):
    context, source = _primary_reuse_context(tmp_path)
    config_path = source / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config[REUSE_CONTRACT_KEY]["schema_version"] = 0
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        require_reuse_compatibility(context, source)


@pytest.mark.parametrize(
    "field",
    [
        "config_name",
        "data_sha256",
        "n_obs",
        "fitted_data_identity",
        "model_design_identity",
        "environment_lock_sha256",
        "resolved_run_plan",
        "sampling",
    ],
)
def test_reuse_still_fails_closed_on_every_bound_contract_field(tmp_path, field):
    """Data, executable design, environment and plan identity all still bind."""
    context, source = _primary_reuse_context(tmp_path)
    config_path = source / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config[REUSE_CONTRACT_KEY][field] = "drifted"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        require_reuse_compatibility(context, source)


@pytest.mark.parametrize("field", ["config_name", "trace_sha256"])
def test_primary_reuse_rejects_contract_or_trace_hash_drift_before_loading(
    tmp_path, monkeypatch, field
):
    context, source = _primary_reuse_context(tmp_path)
    config_path = source / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if field == "trace_sha256":
        config[field] = "0" * 64
    else:
        config[REUSE_CONTRACT_KEY][field] = "dev"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    monkeypatch.setenv("DSE_LRP_REUSE_TRACE", "1")
    monkeypatch.setattr(
        diag.az,
        "from_netcdf",
        lambda *_args, **_kwargs: pytest.fail(
            "an incompatible primary trace must not be loaded"
        ),
    )
    monkeypatch.setattr(
        diag.pm,
        "sample",
        lambda **_kwargs: pytest.fail("reuse-trace must never run fresh NUTS"),
    )

    with pytest.raises(ValueError, match=field):
        diag.sample_posterior(context)


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


def test_psense_excludes_the_child_level_loo_aggregate_from_the_likelihood(tmp_path):
    """2026-08-22 adjusted-family review, finding 1.

    ``compute_log_likelihood_and_loo`` writes the child-summed ``y_post_child``
    into the ``log_likelihood`` group beside the row-level ``y_post`` so PSIS-LOO
    can leave a whole child out. arviz-stats' power scaling sums every variable in
    that group unless told otherwise, so the likelihood was counted twice: every
    published likelihood sensitivity of ``lrp-rlm-adj-006`` and of the eight joint
    / joint-mechanism fits was doubled. The summary written by ``psense_artifacts``
    must equal the one computed on ``y_post`` alone.
    """
    import arviz_stats as azs

    rng = np.random.default_rng(3)
    chains, draws, rows, children = 2, 200, 12, 4
    theta = rng.normal(size=(chains, draws))
    # Row log-likelihoods that depend on theta, and their within-child sums.
    ll_rows = -0.5 * (theta[..., None] - rng.normal(size=rows)) ** 2
    child_idx = np.repeat(np.arange(children), rows // children)
    ll_child = np.stack(
        [ll_rows[..., child_idx == c].sum(axis=-1) for c in range(children)],
        axis=-1,
    )
    trace = xr.DataTree.from_dict(
        {
            "posterior": xr.Dataset(
                {"theta": (("chain", "draw"), theta)},
                coords={"chain": np.arange(chains), "draw": np.arange(draws)},
            ),
            "log_prior": xr.Dataset(
                {"theta": (("chain", "draw"), -0.5 * theta**2)},
                coords={"chain": np.arange(chains), "draw": np.arange(draws)},
            ),
            "log_likelihood": xr.Dataset(
                {
                    "y_post": (("chain", "draw", "obs_id"), ll_rows),
                    diag.LOO_CHILD_AGGREGATE_NODE: (
                        ("chain", "draw", "child"),
                        ll_child,
                    ),
                },
                coords={
                    "chain": np.arange(chains),
                    "draw": np.arange(draws),
                    "obs_id": np.arange(rows),
                    "child": np.arange(children),
                },
            ),
            "sample_stats": xr.Dataset(
                {"diverging": (("chain", "draw"), np.zeros((chains, draws), bool))},
                coords={"chain": np.arange(chains), "draw": np.arange(draws)},
            ),
        }
    )
    assert diag.psense_likelihood_var_names(trace) == ["y_post"]
    single = xr.DataTree.from_dict(
        {
            "posterior": trace.posterior.to_dataset(),
            "log_prior": trace.log_prior.to_dataset(),
            "log_likelihood": trace.log_likelihood.to_dataset().drop_vars(
                diag.LOO_CHILD_AGGREGATE_NODE
            ),
        }
    )
    assert diag.psense_likelihood_var_names(single) is None
    expected = azs.psense_summary(single, var_names=["theta"])
    written = diag.psense_artifacts(trace, str(tmp_path), ["theta"])
    assert written is not None
    expected_df = expected.to_dataframe() if hasattr(expected, "to_dataframe") else pd.DataFrame(expected)
    np.testing.assert_allclose(
        written["likelihood"].to_numpy(dtype=float),
        expected_df["likelihood"].to_numpy(dtype=float),
    )
    # And it is genuinely different from the doubled (both-variables) value.
    doubled = azs.psense_summary(trace, var_names=["theta"])
    doubled_df = doubled.to_dataframe() if hasattr(doubled, "to_dataframe") else pd.DataFrame(doubled)
    assert float(doubled_df["likelihood"].iloc[0]) > float(
        expected_df["likelihood"].iloc[0]
    )


def _psense_trace(posterior_names: tuple[str, ...]) -> xr.DataTree:
    """A minimal trace with the groups ``plot_psense_dist`` needs."""
    rng = np.random.default_rng(0)
    chains, draws, obs = 2, 200, 30
    posterior = xr.Dataset(
        {n: (("chain", "draw"), rng.normal(size=(chains, draws))) for n in posterior_names}
    )
    return xr.DataTree.from_dict(
        {
            "posterior": posterior,
            "log_prior": posterior.copy(deep=True),
            "log_likelihood": xr.Dataset(
                {"y": (("chain", "draw", "obs"), rng.normal(size=(chains, draws, obs)))},
                coords={"obs": np.arange(obs)},
            ),
        }
    )


def test_psense_plot_view_leaves_clash_free_traces_alone():
    trace = _psense_trace(("tau", "kappa"))
    view, var_names = diag._psense_plot_view(trace, ["tau"])
    assert view is trace
    assert var_names == ["tau"]


def test_psense_plot_view_renames_reserved_posterior_names():
    trace = _psense_trace(("alpha", "tau", "kappa"))

    # Unrequested clash: the variable still has to be renamed, because
    # plot_psense_dist resamples the whole posterior regardless of var_names.
    view, var_names = diag._psense_plot_view(trace, ["tau"])
    assert var_names == ["tau"]
    assert "alpha" not in view.posterior.to_dataset().variables
    assert "alpha (parameter)" in view.posterior.to_dataset().variables

    # Requested clash: the mapping is applied to var_names too, so the parameter
    # keeps its panel in the figure.
    _, var_names = diag._psense_plot_view(trace, ["alpha", "tau"])
    assert var_names == ["alpha (parameter)", "tau"]


def test_psense_plot_view_survives_an_unexpected_trace():
    sentinel = object()
    view, var_names = diag._psense_plot_view(sentinel, ["tau"])
    assert view is sentinel
    assert var_names == ["tau"]


@pytest.mark.parametrize("var_names", [["tau"], ["alpha", "tau"]])
def test_psense_plot_view_lets_plot_psense_dist_run(var_names):
    """Regression for issue #340 — the un-viewed trace raises, the view does not."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import arviz_plots as azp

    trace = _psense_trace(("alpha", "tau", "kappa"))
    try:
        with pytest.raises(ValueError, match="alpha already exists"):
            azp.plot_psense_dist(trace, var_names=var_names, backend="matplotlib")

        view, mapped = diag._psense_plot_view(trace, var_names)
        azp.plot_psense_dist(view, var_names=mapped, backend="matplotlib")
    finally:
        plt.close("all")


def test_psense_layout_only_overrides_multi_row_grids():
    trace = _psense_trace(("tau", "kappa"))

    assert diag._psense_layout(trace, ["tau"]) == ({}, {})
    assert diag._psense_layout(object(), ["tau"]) == ({}, {})

    plot_kwargs, rc = diag._psense_layout(trace, ["tau", "kappa"])
    figure_kwargs = plot_kwargs["figure_kwargs"]
    assert figure_kwargs["figsize"][1] >= 2 * 2.0
    # gridspec_kw must NOT be set: the house style's constrained layout
    # overrides it and collapses the panels.
    assert set(figure_kwargs) == {"figsize"}
    # Four panels is under ArviZ's default guard, so it stays untouched.
    assert rc == {}


def test_psense_layout_raises_the_subplot_guard_for_wide_selections():
    rng = np.random.default_rng(0)
    posterior = xr.Dataset(
        {"b": (("chain", "draw", "time"), rng.normal(size=(2, 50, 25)))},
        coords={"time": np.arange(25)},
    )
    trace = xr.DataTree.from_dict({"posterior": posterior})

    _, rc = diag._psense_layout(trace, ["b"])
    # 25 coordinate levels x prior/likelihood = 50 panels, past the default 40.
    assert rc == {"plot.max_subplots": 50}


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
    assert got.dims == ("chain", "draw", "loo_child")
    assert got.attrs["loo_unit"] == "child"


def test_marked_repeated_rows_are_aggregated_by_child():
    values = np.array([[[1.0, 2.0, 4.0], [10.0, 20.0, 40.0]]])
    trace = xr.DataTree.from_dict(
        {
            "log_likelihood": xr.Dataset(
                {"y_post": (("chain", "draw", "obs_id"), values)}
            ),
            "constant_data": xr.Dataset(
                {"loo_child_idx": ("obs_id", np.array([0, 1, 0]))}
            ),
        }
    )
    got = diag._joint_log_likelihood_by_child(trace)
    assert got is not None
    np.testing.assert_allclose(got.values, [[[5.0, 2.0], [50.0, 20.0]]])
    assert got.sizes["loo_child"] == 2
    assert got.attrs["aggregation"] == "sum over repeated child rows"


def _repeated_transition_joint_trace() -> xr.DataTree:
    """Two children x two outcomes x three transitions: the ``jm-002`` shape.

    Twelve flattened cells. ``loo_child_idx`` marks the six that belong to each
    child, and ``y_post_cell_outcome`` marks the six that belong to each outcome, so
    the two diagnostics that read this trace slice it along different axes.
    """
    values = np.arange(1.0, 13.0).reshape(1, 1, 12)
    child = np.array([0] * 6 + [1] * 6)
    outcome = np.tile(np.repeat([0, 1], 3), 2)
    return xr.DataTree.from_dict(
        {
            "posterior": xr.Dataset(
                {"tau": (("chain", "draw", "outcome"), np.zeros((1, 1, 2)))},
                coords={"chain": [0], "draw": [0], "outcome": ["W", "N"]},
            ),
            "observed_data": xr.Dataset({"y_post": ("cell", np.arange(12))}),
            "posterior_predictive": xr.Dataset(
                {"y_post": (("chain", "draw", "cell"), values)}
            ),
            "log_likelihood": xr.Dataset(
                {"y_post": (("chain", "draw", "cell"), -values)}
            ),
            "constant_data": xr.Dataset(
                {
                    "loo_child_idx": ("cell", child),
                    "y_post_cell_outcome": ("cell", outcome),
                }
            ),
        }
    )


def test_child_aggregation_covers_both_outcomes_and_every_transition():
    """The declared holdout unit with repeated transitions *and* several outcomes.

    #588 acceptance criterion: LOO tests must include multiple outcomes and multiple
    transitions per child and preserve the declared unit. Leaving out one cell would
    predict a child's word-reading transition while their other transitions and their
    nonword cells stayed in the posterior; the stored LOO leaves out the child.
    """
    got = diag._joint_log_likelihood_by_child(_repeated_transition_joint_trace())
    assert got is not None
    assert got.sizes["loo_child"] == 2
    assert got.attrs["loo_unit"] == "child"
    # Child 0 holds cells 1-6, child 1 cells 7-12 -- both outcomes, all three
    # transitions, summed within child.
    np.testing.assert_allclose(got.values, [[[-21.0, -57.0]]])


def test_the_outcome_loo_pit_tree_keeps_the_cell_unit_the_stored_loo_aggregates():
    """The two diagnostics answer different questions, and the code says which.

    2026-08-23 joint audit, finding 6. The outcome-specific LOO-PIT tree carries the
    focal outcome's raw cells, so ArviZ recomputes leave-one-*cell*-out weights: on
    this trace six units against the stored LOO's two children. Neither validates the
    other, and the figure title names the unit it actually leaves out.
    """
    trace = _repeated_transition_joint_trace()
    context = SimpleNamespace(
        trace=trace, prior_samples=None, spec=SimpleNamespace(extra={}), model=None
    )
    selected = diag._joint_outcome_predictive_tree(context, "N")
    focal = selected.log_likelihood["y_post"]
    cell_dim = next(d for d in focal.dims if d not in {"chain", "draw"})
    # Three transitions for each of the two children, kept as separate units.
    assert focal.sizes[cell_dim] == 6
    np.testing.assert_array_equal(
        selected.observed_data["y_post"].values, np.array([3, 4, 5, 9, 10, 11])
    )
    aggregated = diag._joint_log_likelihood_by_child(trace)
    assert aggregated.sizes["loo_child"] < focal.sizes[cell_dim]
    assert diag.JOINT_LOO_PIT_UNIT_LABEL == "conditional leave-one-cell-out"


def test_marked_repeated_rows_ignore_per_row_G_for_the_unit_count():
    """With ``loo_child_idx`` present the unit count comes from the map itself: a
    per-observation ``G`` in a stacked repeated-row design has one entry per ROW,
    and reading its size would append silent zero-likelihood phantom children to
    the LOO (2026-08-21 joint-mechanism review, finding 3)."""
    values = np.array([[[1.0, 2.0, 4.0], [10.0, 20.0, 40.0]]])
    trace = xr.DataTree.from_dict(
        {
            "log_likelihood": xr.Dataset(
                {"y_post": (("chain", "draw", "cell"), values)}
            ),
            "constant_data": xr.Dataset(
                {
                    "G": ("obs_id", np.array([1.0, 0.0, 1.0])),
                    "loo_child_idx": ("cell", np.array([0, 1, 0])),
                }
            ),
        }
    )
    got = diag._joint_log_likelihood_by_child(trace)
    assert got is not None
    assert got.sizes["loo_child"] == 2
    np.testing.assert_allclose(got.values, [[[5.0, 2.0], [50.0, 20.0]]])


def test_child_level_influence_maps_repeated_rows_to_one_subject():
    context = SimpleNamespace(
        loo=SimpleNamespace(
            pareto_k=np.array([0.2, 0.8]),
            good_k=0.7,
        ),
        prepared=SimpleNamespace(
            subject_ids=np.array(["A", "B", "A", "B"]),
            child_idx=np.array([0, 1, 0, 1]),
        ),
    )
    frame, threshold, n_flagged = diag.influence_diagnostics(context)
    assert threshold == 0.7
    assert n_flagged == 1
    assert frame["subject_id"].tolist() == ["B", "A"]
    assert frame["observation_index"].tolist() == [1, 0]


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


def _joint_trace_without_tau(posterior_var: str = "beta_mech") -> xr.DataTree:
    """A joint-shaped trace whose reported coefficient is not called ``tau``."""
    rng = np.random.default_rng(0)
    n_draw, n_cell = 40, 4
    posterior = xr.Dataset(
        {
            posterior_var: (
                ("chain", "draw", "outcome"),
                rng.normal(size=(2, n_draw, 2)),
            )
        },
        coords={
            "chain": [0, 1],
            "draw": range(n_draw),
            "outcome": ["W", "N"],
        },
    )
    return xr.DataTree.from_dict(
        {
            "posterior": posterior,
            "observed_data": xr.Dataset(
                {"y_post": ("cell", np.array([1, 101, 2, 102]))}
            ),
            "posterior_predictive": xr.Dataset(
                {
                    "y_post": (
                        ("chain", "draw", "cell"),
                        rng.integers(0, 100, size=(2, n_draw, n_cell)),
                    )
                }
            ),
            "log_likelihood": xr.Dataset(
                {
                    "y_post": (
                        ("chain", "draw", "cell"),
                        -rng.random((2, n_draw, n_cell)),
                    )
                }
            ),
            "constant_data": xr.Dataset(
                {"y_post_cell_outcome": ("cell", np.array([0, 1, 0, 1]))}
            ),
        }
    )


def test_joint_loo_pit_tree_falls_back_when_tau_is_absent():
    """A joint family whose reported coefficient is not ``tau`` must still get a
    tree. The hard ``posterior['tau']`` requirement raised a ``KeyError`` that
    :func:`save_joint_loo_pit_plot` swallowed, so the ``joint_mechanism`` family's
    two promised per-outcome LOO-PIT plots were always omitted even though the fit
    completed (#427 review)."""
    context = SimpleNamespace(
        trace=_joint_trace_without_tau(),
        prior_samples=None,
        spec=SimpleNamespace(extra={}),
        model=None,
    )

    selected = diag._joint_outcome_predictive_tree(context, "N")

    # The fallback carries *a* posterior group for the relative-ESS calculation.
    assert "beta_mech" in selected.posterior
    np.testing.assert_array_equal(
        selected.observed_data["y_post"].values, np.array([101, 102])
    )
    # An explicitly named variable is honoured, and a missing one still raises
    # rather than silently picking something else.
    assert "beta_mech" in diag._joint_outcome_predictive_tree(
        context, "N", posterior_var="beta_mech"
    ).posterior
    with pytest.raises(KeyError, match="tau"):
        diag._joint_outcome_predictive_tree(context, "N", posterior_var="tau")


def test_save_joint_loo_pit_plot_writes_file_without_tau(tmp_path):
    """Artefact-level check: the per-outcome LOO-PIT PNG is actually written for a
    joint family with no ``tau``. Asserting on the *file* is the point — the
    previous failure mode was a plot that never appeared while the fit reported
    success."""
    context = SimpleNamespace(
        trace=_joint_trace_without_tau(),
        prior_samples=None,
        spec=SimpleNamespace(extra={}),
        model=None,
        output_dir=str(tmp_path),
    )

    diag.save_joint_loo_pit_plot(context, "N", filename_stem="loo_pit_n")

    assert (tmp_path / "loo_pit_n.png").exists()


def test_the_joint_loo_pit_figure_names_the_unit_it_actually_leaves_out(
    tmp_path, monkeypatch
):
    """This plot subsets one outcome's flattened cells and keeps no child map, so it
    leaves out one **cell** — the omitted cell's child keeps its other transitions,
    its other outcome and its fitted random effect. Presenting it as the calibration
    companion to a leave-one-child-out PSIS-LOO overstates what it checks, so the
    unit is on the figure (2026-08-23 joint-mechanism follow-up review, finding 4)."""
    titles: list[str] = []
    monkeypatch.setattr(
        diag,
        "_save_pc",
        lambda out, build, filename, title=None: titles.append(title),
    )
    context = SimpleNamespace(
        trace=_joint_trace_without_tau(),
        prior_samples=None,
        spec=SimpleNamespace(extra={}),
        model=None,
        output_dir=str(tmp_path),
    )

    diag.save_joint_loo_pit_plot(context, "N", filename_stem="loo_pit_n")
    assert diag.JOINT_LOO_PIT_UNIT_LABEL == "conditional leave-one-cell-out"
    assert titles == [
        "Conditional leave-one-cell-out PIT calibration (N) — "
        "the child's other cells remain observed"
    ]

    # A family whose likelihood really is one cell per child may say so.
    titles.clear()
    diag.save_joint_loo_pit_plot(
        context, "N", filename_stem="loo_pit_n", unit_label="leave-one-child-out"
    )
    # …and gets its own label alone: the "other cells remain observed" clause is a
    # property of the cell-level unit, so it must not travel with a different one.
    assert titles == ["Leave-one-child-out PIT calibration (N)"]


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
    # BFMI is read by the shared ``sampling_quality`` extractor, so that is the
    # seam to patch; ``diag._bfmi_per_chain`` is no longer on this call path.
    # The extractor moved to ``dse_research_utils`` in v0.12.0, so the patch has
    # to target the module that resolves the name.
    monkeypatch.setattr(
        sampling_quality_mod, "_bfmi_per_chain", lambda _trace: np.asarray([0.2, 0.8])
    )
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
        # No scan ran, so nothing is *known* to be unassessable — the whole
        # verdict is ``None``/unchecked, which is the distinction this test pins.
        "unassessable_parameters": "",
        "structurally_constant_parameters": "",
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


def test_compute_log_likelihood_and_prior_strict_controls_reraise(monkeypatch):
    """The psense-only path (strict=False) must not abort a fit when
    ``compute_log_likelihood`` cannot evaluate the likelihood, while the LOO
    path (strict=True) must still re-raise. Guards the #416 robustness contract.

    The patch target is the name **bound in** ``diagnostics`` — not ``diag.pm`` —
    because the module imports it from ``pymc.stats`` directly. Patching the pymc
    root namespace would still succeed (the attribute exists there) but would patch
    nothing the module actually calls, silently voiding this test.
    """
    import pymc as pm

    with pm.Model() as model:
        mu = pm.Normal("mu", 0.0, 1.0)
        pm.Normal("y", mu, 1.0, observed=np.array([0.0, 1.0, -1.0]))
        trace = pm.sample(
            tune=5,
            draws=5,
            chains=1,
            cores=1,
            progressbar=False,
            random_seed=1,
            compute_convergence_checks=False,
        )

    def _boom(*_args, **_kwargs):
        raise RuntimeError("cannot evaluate log-likelihood")

    monkeypatch.setattr(diag, "compute_log_likelihood", _boom, raising=True)

    # LOO contract: strict=True re-raises the failure.
    strict_ctx = SimpleNamespace(model=model, trace=trace.copy())
    with pytest.raises(RuntimeError, match="cannot evaluate log-likelihood"):
        diag.compute_log_likelihood_and_prior(strict_ctx, strict=True)

    # psense-only contract: strict=False swallows it (no raise) and still adds the
    # log_prior group, which power-scaling sensitivity needs.
    lenient_ctx = SimpleNamespace(model=model, trace=trace.copy())
    diag.compute_log_likelihood_and_prior(lenient_ctx, strict=False)
    assert "log_prior" in lenient_ctx.trace
    assert "log_likelihood" not in lenient_ctx.trace


def _prior_draws_as_posterior(model, *, draws: int = 6, seed: int = 3):
    """A cheap stand-in posterior: prior draws relabelled as the posterior group.

    ``compute_log_prior`` / ``compute_log_likelihood`` only ever read the posterior
    group's variable names, shapes and values, so prior draws exercise exactly the
    naming seam under test without paying for NUTS.
    """
    import pymc as pm

    with model:
        prior = pm.sample_prior_predictive(draws=draws, random_seed=seed)
    tree = xr.DataTree()
    tree["posterior"] = prior.prior
    return tree


def _lkj_corr_model():
    """A minimal model carrying the transform whose name contains an underscore."""
    import pymc as pm

    with pm.Model() as model:
        pm.LKJCorr("corr", n=3, eta=2.0)
        sigma = pm.HalfNormal("sigma", 1.0)
        pm.Normal("y", 0.0, sigma, observed=np.array([0.4, -1.1, 0.2, 0.9, -0.3]))
    return model


def test_log_density_model_returns_ordinary_models_unchanged():
    """Only underscore-named transforms need repairing (#453).

    Identity, not equality: an untouched model is the strongest possible statement
    that the ordinary log-density path is bit-for-bit what it was before the fix.
    """
    import pymc as pm

    with pm.Model() as model:
        mu = pm.Normal("mu", 0.0, 1.0)
        sigma = pm.HalfNormal("sigma", 1.0)  # log transform
        pm.Beta("p", 2.0, 2.0)  # logodds transform
        pm.Normal("y", mu, sigma, observed=np.array([0.0, 1.0, -1.0]))

    assert diag.log_density_model(model) is model


def test_pymc_still_mangles_underscore_named_transforms():
    """Pin the upstream bug itself, so a PyMC upgrade cannot silently change it.

    ``get_untransformed_name`` drops a fixed three underscore-separated components,
    which only round-trips when ``transform.name`` has no underscore of its own.
    ``LKJCorr``'s default ``cholesky_corr`` transform does. If this assertion ever
    fails, upstream has fixed it and :func:`log_density_model` can become a no-op.
    """
    from pymc.util import get_transformed_name, get_untransformed_name

    class _Transform:
        def __init__(self, name: str) -> None:
            self.name = name

    for clean in ("log", "logodds", "ordered", "cholesky-cov"):
        assert (
            get_untransformed_name(get_transformed_name("v", _Transform(clean)))
            == "v"
        )

    for broken in ("cholesky_corr", "log_exp_m1"):
        assert (
            get_untransformed_name(get_transformed_name("v", _Transform(broken)))
            != "v"
        )


def test_log_density_model_repairs_the_lkjcorr_naming_seam():
    """An ``LKJCorr`` model can have both log-density groups computed (#453)."""
    import pymc as pm
    from pymc.stats import compute_log_prior

    model = _lkj_corr_model()
    trace = _prior_draws_as_posterior(model)

    # Unrepaired, PyMC refuses both groups on the name mismatch, not on the density.
    with pytest.raises(ValueError, match="exact match required"):
        compute_log_prior(trace.copy(), model=model, progressbar=False)

    repaired = diag.log_density_model(model)
    assert repaired is not model
    assert {value.name for value in repaired.value_vars} == {
        rv.name for rv in repaired.free_RVs
    }

    out = compute_log_prior(trace.copy(), model=repaired, progressbar=False)
    out = pm.compute_log_likelihood(out, model=repaired, progressbar=False)
    assert set(out.log_prior.data_vars) == {"corr", "sigma"}
    assert "y" in out.log_likelihood.data_vars
    assert np.isfinite(out.log_prior["corr"].values).all()


def test_repaired_lkjcorr_log_prior_matches_a_direct_logp_evaluation():
    """The repair renames; it must not rescale (#453 acceptance criterion).

    The substantive risk in reconciling names at this seam is passing values on the
    wrong scale — which would yield plausible numbers rather than an error. Checked
    against ``pm.logp`` on the bare distribution, evaluated at the same draws.
    """
    import pymc as pm
    import pytensor.tensor as pt
    from pymc.stats import compute_log_prior

    model = _lkj_corr_model()
    trace = _prior_draws_as_posterior(model)
    repaired = diag.log_density_model(model)
    out = compute_log_prior(trace.copy(), model=repaired, progressbar=False)

    draws = np.asarray(trace["posterior"]["corr"].values)
    flat = draws.reshape(-1, *draws.shape[2:])
    value = pt.tensor("value", shape=flat.shape[1:], dtype=draws.dtype)
    logp_fn = pm.compile([value], pm.logp(pm.LKJCorr.dist(n=3, eta=2.0), value))

    expected = np.array([float(np.asarray(logp_fn(draw))) for draw in flat])
    got = np.asarray(out.log_prior["corr"].values).reshape(-1)
    np.testing.assert_allclose(got, expected, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# 2026-08-22 ITT audit regressions (issue #577)
# ---------------------------------------------------------------------------


def _degenerate_trace(*, include_constant: bool = True):
    """A clean trace, optionally carrying one unassessable parameter.

    ``stuck`` is constant across every draw and chain, so ArviZ reports a
    non-finite R-hat for it. The gate's NaN-skipping reductions could not see
    that: mixed with a healthy parameter it produced finite extrema, empty
    failing lists and ``passed=true`` (2026-08-22 ITT audit, finding 1).
    """
    rng = np.random.default_rng(0)
    nc, nd = 4, 1000
    variables = {"good": (("chain", "draw"), rng.normal(size=(nc, nd)))}
    if include_constant:
        variables["stuck"] = (("chain", "draw"), np.full((nc, nd), 2.5))
    coords = {"chain": np.arange(nc), "draw": np.arange(nd)}
    posterior = xr.Dataset(variables, coords=coords)
    sample_stats = xr.Dataset(
        {
            "diverging": (("chain", "draw"), np.zeros((nc, nd), dtype=bool)),
            "energy": (("chain", "draw"), rng.normal(size=(nc, nd))),
        },
        coords=coords,
    )
    return az.from_dict({"posterior": posterior, "sample_stats": sample_stats})


def test_sampling_quality_names_variables_whose_diagnostics_are_non_finite():
    from language_reading_predictors.statistical_models.sampling_quality import (
        sampling_quality,
    )

    signals = sampling_quality(
        _degenerate_trace(), var_names=["good", "stuck"]
    )
    # The NaN-skipping extrema still report the healthy parameter, which is the
    # right extraction behaviour - but the skipped row is now named.
    assert np.isfinite(signals.max_rhat)
    assert signals.unassessable == ("stuck",)

    clean = sampling_quality(
        _degenerate_trace(include_constant=False), var_names=["good"]
    )
    assert clean.unassessable == ()


def test_subfit_gate_fails_when_a_parameter_cannot_be_assessed():
    from language_reading_predictors.statistical_models.diagnostics import (
        subfit_convergence,
    )

    verdict = subfit_convergence(
        _degenerate_trace(), label="audit-577", var_names=["good", "stuck"]
    )
    # Zero divergences, R-hat 1.0003 and ESS ~3900 on the parameter it *could*
    # measure: without the unassessable check this passed.
    assert verdict["n_divergences"] == 0
    assert verdict["max_rhat"] <= 1.01
    assert verdict["unassessable_parameters"] == "stuck"
    assert verdict["converged"] is False

    clean = subfit_convergence(
        _degenerate_trace(include_constant=False),
        label="audit-577-clean",
        var_names=["good"],
    )
    assert clean["unassessable_parameters"] == ""
    assert clean["converged"] is True



def test_a_clean_fit_records_that_the_assessable_check_ran(tmp_path):
    """The key must be written whether or not the scan finds anything.

    Writing only on failure left every clean fit's ``diagnostics_summary.json``
    without ``diagnostics_assessable``, which reads identically to a fit from
    before the check existed — and left the file disagreeing with the ``tables``
    entry built from the same payload. Caught by inspecting a real refit.
    """
    context = SimpleNamespace(
        trace=_degenerate_trace(include_constant=False),
        output_dir=str(tmp_path),
        tables={},
        model=None,
    )
    payload = diag.write_diagnostics_summary(context, var_names=["good"])

    stored = json.loads((tmp_path / "diagnostics_summary.json").read_text())
    assert stored["checks"]["diagnostics_assessable"] is True
    assert stored["unassessable_parameters"] == []
    assert stored["passed"] is True
    # The file and the in-memory table must agree.
    assert stored["checks"] == payload["checks"]
    assert context.tables["diagnostics_summary"]["checks"] == stored["checks"]


def test_an_unassessable_parameter_is_recorded_and_fails_the_stored_gate(tmp_path):
    from language_reading_predictors.statistical_models.reporting import (
        convergence_gate_failures,
    )

    context = SimpleNamespace(
        trace=_degenerate_trace(),
        output_dir=str(tmp_path),
        tables={},
        model=None,
    )
    diag.write_diagnostics_summary(context, var_names=["good", "stuck"])

    stored = json.loads((tmp_path / "diagnostics_summary.json").read_text())
    assert stored["checks"]["diagnostics_assessable"] is False
    assert stored["unassessable_parameters"] == ["stuck"]
    assert stored["passed"] is False
    # And the release gate reads it back as a failure without needing to know the
    # check's name: it fails closed on any unrecognised non-True check.
    assert convergence_gate_failures(stored)


# ---------------------------------------------------------------------------
# 2026-08-22 ITT audit regressions (issue #577, finding 6)
# ---------------------------------------------------------------------------


def _two_itt_models(**overrides):
    """The registered W model, plus a variant built from the same rows."""
    from language_reading_predictors.statistical_models.factories import build_itt_model
    from language_reading_predictors.statistical_models.itt import (
        prepare_itt_data,
        resolve_itt_run_plan,
    )
    from language_reading_predictors.statistical_models.lrp_rli_itt_010 import SPEC
    from language_reading_predictors.statistical_models.pipelines.itt import (
        load_and_prepare,
    )

    plan = resolve_itt_run_plan(SPEC)
    prepared, _ = prepare_itt_data(plan, loader=load_and_prepare)
    kwargs = plan.factory_kwargs()
    return (
        prepared,
        build_itt_model(prepared, **kwargs),
        build_itt_model(prepared, **{**kwargs, **overrides}),
    )


def test_the_structure_signature_moves_when_a_prior_default_moves():
    """A code change the declared plan cannot see must refuse trace reuse.

    The contract checked the plan, the data hash, the row keys and the observed
    arrays — none of which move when a prior default, a term or a denominator is
    edited in the factory.
    """
    from language_reading_predictors.statistical_models import reporting as R

    _prepared, registered, retuned = _two_itt_models(tau_sigma=0.9)
    a = R._model_design_identity(SimpleNamespace(model=registered.model))
    b = R._model_design_identity(SimpleNamespace(model=retuned.model))
    assert a["structure_sha256"] and a["design_sha256"]
    assert a["structure_sha256"] != b["structure_sha256"]
    # The rows and predictors are untouched, so the design digest must not move.
    assert a["design_sha256"] == b["design_sha256"]


def test_the_design_digest_moves_when_a_predictor_is_rebuilt():
    """``describe_fitted_data`` digests outcomes and row keys, not predictors.

    A silently rebuilt covariate — same children, same scores, different
    standardisation — passed every check the contract had.
    """
    from dataclasses import replace as _replace

    from language_reading_predictors.statistical_models import reporting as R
    from language_reading_predictors.statistical_models.factories import build_itt_model
    from language_reading_predictors.statistical_models.itt import (
        prepare_itt_data,
        resolve_itt_run_plan,
    )
    from language_reading_predictors.statistical_models.lrp_rli_itt_010 import SPEC
    from language_reading_predictors.statistical_models.pipelines.itt import (
        load_and_prepare,
    )
    from language_reading_predictors.statistical_models.subfits import (
        describe_fitted_data,
    )

    plan = resolve_itt_run_plan(SPEC)
    prepared, _ = prepare_itt_data(plan, loader=load_and_prepare)
    rebuilt = _replace(prepared, A_std=np.asarray(prepared.A_std, dtype=float) * 1.01)
    built = build_itt_model(prepared, **plan.factory_kwargs())
    built_rebuilt = build_itt_model(rebuilt, **plan.factory_kwargs())

    a = R._model_design_identity(SimpleNamespace(model=built.model))
    b = R._model_design_identity(SimpleNamespace(model=built_rebuilt.model))
    assert a["design_sha256"] != b["design_sha256"]
    # Structure is unchanged — only the numbers inside the Data nodes moved. The
    # graph identity records each shared node's dtype and shape, not its contents,
    # so the two hashes stay independent and a refusal can name which one moved.
    assert a["structure_sha256"] == b["structure_sha256"]
    # And this is precisely what the pre-existing identity could not see.
    assert describe_fitted_data(
        SimpleNamespace(model=built.model, prepared=prepared)
    ).digest == describe_fitted_data(
        SimpleNamespace(model=built_rebuilt.model, prepared=rebuilt)
    ).digest


def test_the_reuse_contract_carries_the_new_bindings():
    from language_reading_predictors.statistical_models.reporting import (
        _REUSE_CONFIG_FIELDS,
    )

    assert "model_design_identity" in _REUSE_CONFIG_FIELDS
    assert "environment_lock_sha256" in _REUSE_CONFIG_FIELDS


def test_reuse_is_refused_against_a_fit_predating_the_bindings(tmp_path, monkeypatch):
    """Fail closed: those posteriors were never checked this way."""
    context, source = _primary_reuse_context(tmp_path)
    config_path = source / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config[REUSE_CONTRACT_KEY].pop("model_design_identity", None)
    config[REUSE_CONTRACT_KEY].pop("environment_lock_sha256", None)
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="model_design_identity"):
        require_reuse_compatibility(context, source)


def test_reuse_is_refused_against_a_fit_predating_the_serialised_contract(tmp_path):
    """A config carrying only the historical top-level subset is refused."""
    context, source = _primary_reuse_context(tmp_path)
    config_path = source / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.pop(REUSE_CONTRACT_KEY)
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match=REUSE_CONTRACT_KEY):
        require_reuse_compatibility(context, source)


# ---------------------------------------------------------------------------
# Structural-constant reclassification (2026-08-26 batch)
# ---------------------------------------------------------------------------


def _lkj_style_posterior():
    """A posterior with an LKJ-style Cholesky: fixed corner/zero, mixing slope."""
    import numpy as np
    import xarray as xr

    rng = np.random.default_rng(3)
    chol = np.zeros((4, 100, 2, 2))
    chol[:, :, 0, 0] = 1.0  # structural unit corner
    chol[:, :, 0, 1] = 0.0  # structural upper zero
    chol[:, :, 1, 0] = rng.normal(0.3, 0.05, (4, 100))  # mixes
    chol[:, :, 1, 1] = rng.normal(0.9, 0.02, (4, 100))  # mixes
    stuck = np.zeros((4, 100))
    stuck[:] = np.arange(4)[:, None] * 0.1  # constant WITHIN chains, differs ACROSS
    posterior = xr.Dataset(
        {
            "measure_corr_chol": (("chain", "draw", "d0", "d1"), chol),
            "stuck": (("chain", "draw"), stuck),
        }
    )

    _record_test_cholesky(posterior)
    return posterior


def _record_test_cholesky(posterior):
    import pymc as pm
    from language_reading_predictors.statistical_models.structural_constants import record_structural_constraints

    with pm.Model() as model:
        pm.LKJCorr("measure_corr_chol", n=2, eta=2)
    record_structural_constraints(posterior, model)


def test_split_structurally_constant_separates_lkj_entries_from_stuck_ones():
    from language_reading_predictors.statistical_models.diagnostics import (
        split_structurally_constant,
    )

    posterior = _lkj_style_posterior()
    structural, genuine = split_structurally_constant(
        posterior,
        ["measure_corr_chol[0, 0]", "measure_corr_chol[0, 1]", "stuck", "absent"],
    )
    assert structural == ["measure_corr_chol[0, 0]", "measure_corr_chol[0, 1]"]
    # A parameter constant within chains but different across them is a stuck
    # sampler, never a structural constant; an unresolvable name fails closed.
    assert genuine == ["stuck", "absent"]


def test_reclassify_structural_constants_flips_the_verdict_and_rewrites(tmp_path):
    import json

    from language_reading_predictors.statistical_models.diagnostics import (
        reclassify_structural_constants,
    )

    posterior = _lkj_style_posterior()
    summary = {
        "passed": False,
        "checks": {"rhat": True, "ess": True, "diagnostics_assessable": False},
        "unassessable_parameters": [
            "measure_corr_chol[0, 0]",
            "measure_corr_chol[0, 1]",
        ],
    }
    path = tmp_path / "diagnostics_summary.json"
    path.write_text(json.dumps(summary))
    out = reclassify_structural_constants(summary, posterior, path=str(path))
    assert out["passed"] is True
    assert out["checks"]["diagnostics_assessable"] is True
    assert out["unassessable_parameters"] == []
    assert out["structurally_constant_parameters"] == [
        "measure_corr_chol[0, 0]",
        "measure_corr_chol[0, 1]",
    ]
    assert json.loads(path.read_text())["passed"] is True


def test_reclassify_keeps_genuine_unassessables_failing(tmp_path):
    from language_reading_predictors.statistical_models.diagnostics import (
        reclassify_structural_constants,
    )

    posterior = _lkj_style_posterior()
    summary = {
        "passed": False,
        "checks": {"rhat": True, "diagnostics_assessable": False},
        "unassessable_parameters": ["measure_corr_chol[0, 0]", "stuck"],
    }
    out = reclassify_structural_constants(summary, posterior, path=None)
    assert out["passed"] is False
    assert out["checks"]["diagnostics_assessable"] is False
    assert out["unassessable_parameters"] == ["stuck"]
    assert out["structurally_constant_parameters"] == ["measure_corr_chol[0, 0]"]


def test_subfit_convergence_reclassifies_structural_constants():
    import numpy as np
    import xarray as xr

    from language_reading_predictors.statistical_models.diagnostics import (
        subfit_convergence,
    )

    rng = np.random.default_rng(9)
    chol = np.zeros((4, 200, 2, 2))
    chol[:, :, 0, 0] = 1.0
    chol[:, :, 1, 0] = rng.normal(0.3, 0.05, (4, 200))
    chol[:, :, 1, 1] = rng.normal(0.9, 0.02, (4, 200))
    post = xr.Dataset(
        {
            "measure_corr_chol": (("chain", "draw", "d0", "d1"), chol),
            "beta": (("chain", "draw"), rng.normal(0, 1, (4, 200))),
        },
        coords={"chain": range(4), "draw": range(200)},
    )
    ss = xr.Dataset(
        {
            "diverging": (("chain", "draw"), np.zeros((4, 200), dtype=bool)),
            "energy": (("chain", "draw"), rng.normal(size=(4, 200))),
        },
        coords={"chain": range(4), "draw": range(200)},
    )
    trace = xr.DataTree.from_dict({"posterior": post, "sample_stats": ss})
    _record_test_cholesky(trace.posterior)
    result = subfit_convergence(trace, label="lkj subfit")
    assert result["converged"] is True
    assert result["unassessable_parameters"] == ""
    assert "measure_corr_chol[0, 0]" in result["structurally_constant_parameters"]


# --- the derived-estimand gate must fail on names the summary lacks (#631 f.10)


def _derived_gate_ctx(tmp_path):
    (tmp_path / "diagnostics_summary.json").write_text(
        json.dumps({"checks": {"rhat": True}, "passed": True})
    )
    return SimpleNamespace(output_dir=str(tmp_path), tables={})


def _derived_row(name: str) -> dict:
    return {
        "quantity": name,
        "ess_bulk": 4000.0,
        "ess_tail": 4000.0,
        "mcse_median": 0.001,
        "prob_lo": -1.0,
        "prob_hi": 1.0,
    }


@pytest.mark.parametrize(
    "quantities",
    [("total", "NDE", "NIE"), ("total", "IDE", "IIE")],
    ids=["natural", "interventional"],
)
def test_derived_gate_passes_a_complete_branch_summary(tmp_path, quantities):
    ctx = _derived_gate_ctx(tmp_path)
    summary = pd.DataFrame([_derived_row(name) for name in quantities])
    payload = diag.gate_derived_estimands(ctx, summary, quantities=quantities)
    assert payload["checks"]["derived_estimands"] is True
    assert payload["passed"] is True


def test_derived_gate_fails_an_empty_summary(tmp_path):
    ctx = _derived_gate_ctx(tmp_path)
    payload = diag.gate_derived_estimands(
        ctx, pd.DataFrame(columns=["quantity"]), quantities=("total", "NDE", "NIE")
    )
    assert payload["checks"]["derived_estimands"] is False
    assert payload["passed"] is False
    assert payload["derived_estimands_failing"] == [
        "total (missing)",
        "NDE (missing)",
        "NIE (missing)",
    ]


def test_derived_gate_fails_a_partial_summary_naming_the_missing(tmp_path):
    ctx = _derived_gate_ctx(tmp_path)
    summary = pd.DataFrame([_derived_row("total")])
    payload = diag.gate_derived_estimands(
        ctx, summary, quantities=("total", "NDE", "NIE")
    )
    assert payload["passed"] is False
    assert payload["derived_estimands_failing"] == ["NDE (missing)", "NIE (missing)"]


def test_derived_gate_fails_a_duplicated_quantity(tmp_path):
    ctx = _derived_gate_ctx(tmp_path)
    summary = pd.DataFrame(
        [_derived_row("total"), _derived_row("total"), _derived_row("NDE"), _derived_row("NIE")]
    )
    payload = diag.gate_derived_estimands(
        ctx, summary, quantities=("total", "NDE", "NIE")
    )
    assert payload["passed"] is False
    assert payload["derived_estimands_failing"] == ["total (duplicated)"]


def test_derived_gate_still_fails_a_present_row_below_the_ess_floor(tmp_path):
    ctx = _derived_gate_ctx(tmp_path)
    rows = [_derived_row("total"), _derived_row("NDE"), _derived_row("NIE")]
    rows[1]["ess_bulk"] = 12.0
    payload = diag.gate_derived_estimands(
        ctx, pd.DataFrame(rows), quantities=("total", "NDE", "NIE")
    )
    assert payload["passed"] is False
    assert payload["derived_estimands_failing"] == ["NDE"]
