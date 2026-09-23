# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pins for the GB pipeline's clustering linkage and SHAP-interaction artefacts (#631).

Finding 18: Ward linkage is correctly defined only for Euclidean distances, so
the 1 − dcor dissimilarity must be clustered with AVERAGE linkage — both in
``EstimatorPipeline.feature_selection_diagnostics`` and in
``scripts/rank_predictors.py``'s cut-height sensitivity, which replicates it.

Finding 20c: the SHAP interaction CSV and heatmap must share one convention —
the summed symmetric |SHAP interaction| — so a pair's heatmap cell equals its
table value.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from dse_research_utils.ml.feature_groups import linkage_from_dissimilarity
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from language_reading_predictors.data_variables import Variables

from language_reading_predictors.models.base_pipeline import (
    EstimatorPipeline,
    summed_symmetric_interactions,
)

_RANK_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "rank_predictors.py"


@pytest.fixture(scope="module")
def rank_predictors():
    spec = importlib.util.spec_from_file_location("rank_predictors_linkage", _RANK_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── linkage method (finding 18) ────────────────────────────────────────────────


def test_feature_selection_diagnostics_uses_average_linkage():
    """Pin the linkage method: average, never Ward, on the 1 − dcor dissimilarity.

    Since #662 the pipeline builds its tree through
    ``cluster_ranking.average_linkage_tree``, so this checks the *tree* rather
    than the call spelling: it must equal SciPy average linkage on the same
    dissimilarity, and differ from Ward. The shared constructor also refuses
    Ward outright, which is asserted here so a later caller cannot reintroduce
    it by passing a method through.
    """
    from language_reading_predictors.models.cluster_ranking import average_linkage_tree

    src = inspect.getsource(EstimatorPipeline.feature_selection_diagnostics)
    assert "average_linkage_tree(" in src
    assert "hierarchy.ward(" not in src

    rng = np.random.default_rng(18)
    n = 9
    dissim = rng.random((n, n))
    dissim = (dissim + dissim.T) / 2.0
    np.fill_diagonal(dissim, 0.0)
    condensed = squareform(dissim, checks=False)

    tree = average_linkage_tree(dissim)
    np.testing.assert_array_equal(tree, hierarchy.average(condensed))
    assert not np.allclose(tree, hierarchy.ward(condensed))

    with pytest.raises(ValueError, match="Euclidean-only"):
        linkage_from_dissimilarity(dissim, method="ward")


def test_rank_predictors_average_linkage_matches_scipy_average(rank_predictors):
    """``average_linkage`` must reproduce scipy average linkage on the same
    mean-filled distance-correlation dissimilarity (and differ from Ward)."""
    from language_reading_predictors.stats_utils import distance_corr_matrix

    rng = np.random.default_rng(7)
    n = 60
    a = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "a0": a,
            "a1": a + 0.1 * rng.normal(size=n),
            "b0": rng.normal(size=n),
            "b1": rng.normal(size=n),
            "c0": rng.normal(size=n),
        }
    )

    Z = rank_predictors.average_linkage(X)

    Xf = X.replace({pd.NA: np.nan}).astype("float64")
    Xf = Xf.fillna(Xf.mean())
    dissim = 1.0 - distance_corr_matrix(Xf)
    np.fill_diagonal(dissim, 0.0)
    np.clip(dissim, 0.0, 1.0, out=dissim)
    condensed = squareform(dissim, checks=False)

    assert np.allclose(Z, hierarchy.average(condensed))
    assert not np.allclose(Z, hierarchy.ward(condensed))


# ── SHAP interaction convention (finding 20c) ──────────────────────────────────


def test_summed_symmetric_interactions_table_matches_heatmap():
    """The plotted matrix must equal the saved pair values (#631 finding 20c)."""
    rng = np.random.default_rng(11)
    feats = ["w", "x", "y", "z"]
    mean_abs = rng.uniform(0.0, 1.0, size=(4, 4))  # deliberately asymmetric

    inter_df, heat = summed_symmetric_interactions(mean_abs, feats)

    # Heatmap matrix: summed symmetric with a zero diagonal.
    assert np.allclose(heat, heat.T)
    assert np.allclose(np.diag(heat), 0.0)

    # Every CSV pair row equals the corresponding heatmap cell, and both carry
    # the summed [i, j] + [j, i] convention.
    pos = {f: i for i, f in enumerate(feats)}
    assert len(inter_df) == 6  # 4 choose 2 off-diagonal pairs
    for _, row in inter_df.iterrows():
        i, j = pos[row["feature_a"]], pos[row["feature_b"]]
        assert row["mean_abs_interaction"] == pytest.approx(heat[i, j])
        assert row["mean_abs_interaction"] == pytest.approx(
            mean_abs[i, j] + mean_abs[j, i]
        )

    # Ranked descending.
    vals = inter_df["mean_abs_interaction"].to_numpy()
    assert np.all(np.diff(vals) <= 0)


def _small_pipeline(tmp_path):
    from language_reading_predictors.models.common import ModelConfig, RunConfig

    pipe = EstimatorPipeline(
        ModelConfig(
            model_id="review", description="Review test", target_var="y",
            predictor_vars=["signal", "wave"], model_params={},
        ),
        RunConfig.from_name("dev"),
    )
    pipe.context.output_dir = tmp_path
    return pipe


@pytest.mark.parametrize("fail_at", ["permutation_importance_analysis", "report", None])
def test_metrics_mark_only_a_completed_fit(tmp_path, monkeypatch, fail_at):
    pipe = _small_pipeline(tmp_path)
    # An earlier successful run must not leave its completion marker behind.
    marker = tmp_path / "metrics.json"
    marker.write_text("old metrics")

    def stage(name):
        def run():
            if name == fail_at:
                raise RuntimeError("stage failed")
        return run

    for name in (
        "prepare_data", "configure_model", "cross_validate", "fit_model", "evaluate",
        "permutation_importance_analysis", "construct_importance", "cluster_ranking_analysis", "report",
    ):
        monkeypatch.setattr(pipe, name, stage(name))
    def save_metrics(*, fit_complete=False):
        assert fit_complete is True
        marker.write_text("new metrics")
    monkeypatch.setattr(pipe, "save_metrics", save_metrics)
    if fail_at is None:
        pipe.fit()
        assert marker.read_text() == "new metrics"
    else:
        with pytest.raises(RuntimeError, match="stage failed"):
            pipe.fit()
        assert not marker.exists()


def test_bootstrap_importance_preserves_child_trajectories(tmp_path):
    from sklearn.linear_model import LinearRegression

    pipe = _small_pipeline(tmp_path)
    groups = np.repeat(np.arange(24), 3)
    # Every child has the same wave profile. A whole-child permutation cannot
    # change it; a row shuffle invents trajectories and falsely assigns importance.
    wave = np.tile([0.0, 1.0, 2.0], 24)
    signal = np.repeat(np.linspace(-1.0, 1.0, 24), 3)
    pipe.context.X = pd.DataFrame({"signal": signal, "wave": wave})
    pipe.context.y = pd.Series(3.0 * signal + 10.0 * wave)
    pipe.context.groups = pd.Series(groups)
    pipe.context.df = pd.DataFrame({Variables.TIME: wave})
    pipe.context.pipeline = LinearRegression()

    pipe.stability_selection(n_bootstraps=4, n_repeats=5, top_k=1)

    result = pipe.context.dataframes["stability_selection"].set_index("feature")
    assert result.loc["wave", "importance_mean"] == pytest.approx(0.0, abs=1e-12)
    assert result.loc["signal", "importance_mean"] > 1.0
    assert result.loc["signal", "appearance_rate_top_k"] == 1.0
    support = pd.read_csv(tmp_path / "stability_permutation_support.csv")
    assert support["used"].all()
    assert (support["n_movable_subjects"] == support["n_oob_subjects"]).all()


def test_bootstrap_requires_two_out_of_bag_children(tmp_path):
    from sklearn.linear_model import LinearRegression

    pipe = _small_pipeline(tmp_path)
    pipe.context.X = pd.DataFrame({"signal": [0.0, 0.0, 1.0, 1.0], "wave": [0.0, 1.0, 0.0, 1.0]})
    pipe.context.y = pd.Series([1.0, 2.0, 3.0, 4.0])
    pipe.context.groups = pd.Series(["a", "a", "b", "b"])
    pipe.context.pipeline = LinearRegression()

    with pytest.raises(RuntimeError, match="out-of-bag subjects sharing an assessment schedule"):
        pipe.stability_selection(n_bootstraps=3, n_repeats=2)
    assert not (tmp_path / "stability_selection.csv").exists()


def test_bootstrap_without_alternative_donors_records_exclusions(tmp_path):
    from sklearn.linear_model import LinearRegression

    pipe = _small_pipeline(tmp_path)
    pipe.context.X = pd.DataFrame({"signal": np.arange(12)})
    pipe.context.y = pd.Series(np.arange(12))
    pipe.context.groups = pd.Series(np.arange(12))
    pipe.context.df = pd.DataFrame({Variables.TIME: np.arange(12)})
    pipe.context.pipeline = LinearRegression()
    with pytest.raises(RuntimeError, match="sharing an assessment schedule"):
        pipe.stability_selection(n_bootstraps=4, n_repeats=2)
    support = pd.read_csv(tmp_path / "stability_permutation_support.csv")
    assert len(support) == 4
    assert not support["used"].any()
    assert support["n_movable_subjects"].sum() == 0
    assert not (tmp_path / "stability_selection.csv").exists()


def test_locked_completion_record_preserves_the_previous_fit(tmp_path, monkeypatch):
    pipe = _small_pipeline(tmp_path)
    marker = tmp_path / "metrics.json"
    marker.write_text("completed")
    report = tmp_path / "index.qmd"
    report.write_text("previous report")
    unlink = Path.unlink

    def refuse_marker(path, *args, **kwargs):
        if path == marker:
            raise PermissionError("locked completion record")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", refuse_marker)
    with pytest.raises(PermissionError, match="locked completion record"):
        pipe.fit()
    assert marker.read_text() == "completed"
    assert report.read_text() == "previous report"


def test_config_records_loaded_data_identity_without_rehashing_it(tmp_path):
    import hashlib
    import json

    pipe = _small_pipeline(tmp_path)
    source = tmp_path / "source.csv"
    source.write_bytes(b"score\n1\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    pipe.context.df = pd.DataFrame({"score": [1]})
    pipe.context.df.attrs.update(data_path=str(source), data_sha256=digest)
    # If the source changes after loading, metadata must still identify what was fitted.
    source.write_bytes(b"score\n2\n")
    pipe.save_config()

    config = json.loads((tmp_path / "config.json").read_text())
    assert config["data_path"] == str(source)
    assert config["data_sha256"] == digest
    assert config["provenance"]["source"]["commit"]
    lock = tmp_path / config["environment_lock_file"]
    assert hashlib.sha256(lock.read_bytes()).hexdigest() == config["environment_lock_sha256"]


@pytest.fixture(params=["fit", "tune"])
def clear_outputs(request):
    if request.param == "fit":
        from language_reading_predictors.models.base_pipeline import _clear_directory
        return _clear_directory
    script = _RANK_SCRIPT.with_name("tune_model.py")
    spec = importlib.util.spec_from_file_location("tune_model_cleanup", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._clear_directory


def test_output_cleanup_does_not_suppress_deletion_failure(tmp_path, monkeypatch, clear_outputs):
    (tmp_path / "old.csv").write_text("old result")

    def locked(*args, **kwargs):
        raise PermissionError("locked output")

    monkeypatch.setattr(Path, "unlink", locked)
    with pytest.raises(PermissionError, match="locked output"):
        clear_outputs(tmp_path)


def test_output_cleanup_unlinks_directory_symlinks(tmp_path, clear_outputs):
    output = tmp_path / "output"
    output.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    sentinel = target / "keep.txt"
    sentinel.write_text("keep")
    try:
        (output / "linked").symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks unavailable on this platform")
    clear_outputs(output)
    assert list(output.iterdir()) == []
    assert sentinel.read_text() == "keep"
