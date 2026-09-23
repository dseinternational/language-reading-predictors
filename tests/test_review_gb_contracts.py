# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression cases raised in the review of the statistical corrections."""

import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from language_reading_predictors.models._reporting import gb_permutation_support_markdown, gb_ranking_markdown
from language_reading_predictors.models.cluster_ranking import aggregate_cluster_importance, assemble_ranking, cluster_ranking_table
from language_reading_predictors.models.permutation import PERMUTATION_DESIGN_VERSION, SubjectPermutationDesign, pooled_permutation_deltas


def _ranking():
    return pd.DataFrame({"member": ["signal", "time"], "perm_imp_mean": [1., np.nan],
                         "mean_abs_shap": [1., 20.], "sign": ["+", "+"]})


@pytest.mark.parametrize("metadata", [None, {}, {"permutation_design": "subject_blocks_same_wave_schedule_v1"},
                                      {"permutation_design": PERMUTATION_DESIGN_VERSION}])
def test_support_file_alone_cannot_certify_headline_rankings(tmp_path, metadata):
    pd.DataFrame({"n_subjects": [3], "n_movable_subjects": [3]}).to_csv(tmp_path / "permutation_schedule_support.csv", index=False)
    if metadata is not None:
        (tmp_path / "config.json").write_text(json.dumps(metadata))
    text = gb_ranking_markdown(_ranking(), artifact_dir=tmp_path)
    current = metadata == {"permutation_design": PERMUTATION_DESIGN_VERSION}
    assert ("headline ranking is withheld" in text) == (not current)
    assert ("The model leans most on" in text) == current
    if current:
        assert "Not assessable under this permutation design: `time`" in text
        assert "The model leans most on `signal`" in text
        assert "`signal`, `time`" not in text


def test_conflicting_ranking_and_fit_versions_are_not_accepted(tmp_path):
    (tmp_path / "ranking_meta.json").write_text(json.dumps({"permutation_design": PERMUTATION_DESIGN_VERSION}))
    (tmp_path / "config.json").write_text(json.dumps({"permutation_design": "old"}))
    assert "headline ranking is withheld" in gb_permutation_support_markdown(tmp_path)


def test_schedule_blocks_are_built_once_and_unchanged_columns_are_unranked(monkeypatch):
    from language_reading_predictors.models import permutation

    groups = np.repeat(["a", "b", "c"], 2)
    waves = np.tile([1, 2], 3)
    X = pd.DataFrame({"signal": [1., 1., 2., 2., 3., 3.], "time": waves})
    y = X["signal"].to_numpy() + waves
    estimator = LinearRegression().fit(X, y)
    original = permutation._schedule_blocks
    calls = []

    def record(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(permutation, "_schedule_blocks", record)
    design = SubjectPermutationDesign.from_labels(groups, waves)
    assert design.support()["n_movable_subjects"].sum() == 3
    result = pooled_permutation_deltas([estimator], X, y, [np.arange(6)], groups,
                                      {"signal": [0], "time": [1], "mixed": [0, 1]},
                                      n_repeats=10, seed=47, waves=waves, design=design)
    assert len(calls) == 1
    assert np.isnan(result["time"]).all()
    assert result["signal"].mean() > 0
    np.testing.assert_equal(result["signal"], result["mixed"])


def test_missing_importance_stays_unranked_through_saved_cluster_tables(tmp_path):
    perm = pd.DataFrame({"feature": ["signal", "time", "fixed"], "importance_mean": [1., np.nan, np.nan],
                         "importance_std": [.1, np.nan, np.nan]})
    clusters = pd.DataFrame({"feature": ["signal", "time", "fixed"], "cluster_id": [1, 1, 2]})
    clusters.to_csv(tmp_path / "cluster_table.csv", index=False)
    pd.DataFrame({"feature": ["signal", "time", "fixed"], "shap_mean_abs": [1., 10., .2],
                  "feature_shap_spearman": [.8, .9, .1]}).to_csv(tmp_path / "shap_direction_diagnostics.csv", index=False)
    cluster_imp = aggregate_cluster_importance(perm, clusters)
    pipe = SimpleNamespace(context=SimpleNamespace(output_dir=tmp_path, perm_importance_df=perm))
    ranking = assemble_ranking(pipe, "outcome", [], cluster_imp)
    ranking.to_csv(tmp_path / "predictor_ranking.csv", index=False)
    saved = pd.read_csv(tmp_path / "predictor_ranking.csv").set_index("member")
    assert pd.isna(saved.loc["time", "within_cluster_rank"])
    assert pd.isna(saved.loc["fixed", "cluster_rank"])
    assert saved.loc["time", "permutation_status"] == "not assessable under this design"
    summary = cluster_ranking_table(cluster_imp, ranking, []).set_index("cluster_id")
    assert summary.loc[1, "representative"] == "signal"
    assert pd.isna(summary.loc[2, "representative"])


TEMPLATES = sorted((Path(__file__).resolve().parents[1] / "docs/models").glob("lrp-rli-gb*/index.qmd"))


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda p: p.parent.name)
@pytest.mark.parametrize("current", [False, True])
def test_all_boosting_headlines_and_support_sections_execute(tmp_path, monkeypatch, capsys, template, current):
    _ranking().to_csv(tmp_path / "predictor_ranking.csv", index=False)
    config = {"permutation_design": PERMUTATION_DESIGN_VERSION} if current else {}
    (tmp_path / "config.json").write_text(json.dumps(config))
    pd.DataFrame({"n_subjects": [3], "n_movable_subjects": [3]}).to_csv(tmp_path / "permutation_schedule_support.csv", index=False)
    monkeypatch.chdir(tmp_path)
    blocks = re.findall(r"```\{python\}\n(.*?)\n```", template.read_text(encoding="utf-8"), re.S)
    selected = [block for block in blocks if "gb_ranking_markdown" in block or "gb_permutation_support_markdown" in block]
    assert len(selected) == 2
    for block in selected:
        exec(compile(block, str(template), "exec"), {"config": config})
        text = capsys.readouterr().out
        assert ("headline ranking is withheld" in text) == (not current)
        if current:
            assert "not assessable" in text
