# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the GB findings-prose helper (issue #208)."""

from __future__ import annotations

import pandas as pd

from language_reading_predictors.models._reporting import gb_ranking_markdown


def _ranking() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cluster_id": [1, 2, 3, 3],
            "member": ["attend", "age", "yarclet", "blending"],
            "perm_imp_mean": [0.07, 0.04, 0.0, 0.0],
            "mean_abs_shap": [0.49, 0.60, 0.10, 0.05],
            "sign": ["+", "-", "+", "0"],
            "same_skill_of_outcome": [False, False, True, False],
        }
    )


def test_lead_predictors_and_direction():
    md = gb_ranking_markdown(_ranking(), target_label="word-reading gain")
    assert "word-reading gain" in md
    assert "`attend`" in md and "`age`" in md
    # sign "+" -> larger; sign "-" -> smaller
    assert "higher values predict a larger" in md
    assert "higher values predict a smaller" in md


def test_perm_vs_shap_disagreement_flagged():
    md = gb_ranking_markdown(_ranking())
    # perm lead is attend (0.07); SHAP lead is age (0.60)
    assert "disagree on the lead predictor" in md
    assert "`attend`" in md and "`age`" in md


def test_leakage_flag_reported():
    md = gb_ranking_markdown(_ranking())
    assert "Leakage caution" in md
    assert "`yarclet`" in md


def test_collinear_cluster_grouped():
    md = gb_ranking_markdown(_ranking())
    assert "Collinear predictors" in md
    assert "yarclet" in md and "blending" in md


def test_empty_or_missing_columns_degrades_gracefully():
    assert "produced at the reporting/test tiers" in gb_ranking_markdown(
        pd.DataFrame()
    )
    assert "produced at the reporting/test tiers" in gb_ranking_markdown(
        pd.DataFrame({"not_member": [1]}), target_label="x"
    )
