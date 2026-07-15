# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Focused output-contract guards for the concurrent-associations family (#312)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from language_reading_predictors.statistical_models.lrp_rli_ca_001 import (
    SPEC as CA001,
)
from language_reading_predictors.statistical_models.lrp_rli_ca_002 import (
    SPEC as CA002,
)
from language_reading_predictors.statistical_models.lrp_rli_ca_003 import (
    SPEC as CA003,
)
from language_reading_predictors.statistical_models.lrp_rli_ca_004 import (
    SPEC as CA004,
)
from language_reading_predictors.statistical_models.lrp_rli_ca_005 import (
    SPEC as CA005,
)
from language_reading_predictors.statistical_models.lrp_rli_ca_006 import (
    SPEC as CA006,
)
from language_reading_predictors.statistical_models.pipeline import (
    _ca_label,
    _ca_margin_fields,
    _ca_sd_margin,
    _write_concurrent_outputs,
)


def test_concurrent_core_symbols_have_reader_labels():
    labels = {_ca_label(symbol) for symbol in ("W", "L", "B", "TR", "TE", "R", "E")}
    assert "W" not in labels
    assert "Word reading" in labels
    assert len(labels) == 7


def test_concurrent_specs_share_family_design_and_core_conditionals():
    specs = (CA001, CA002, CA003, CA004, CA005, CA006)
    core = {"W", "L", "B", "TR", "TE", "R", "E"}
    for spec in specs:
        assert spec.family == "concurrent"
        assert spec.design == "per-wave cross-sectional conditional associations"
        assert spec.estimand_type == "association"
        assert spec.causal_status == "none"
        assert set(spec.extra["predictor_symbols"]) == core - {spec.outcome_symbol}


def test_concurrent_wide_margin_fields_cover_probability_and_items():
    row = pd.Series(
        {
            "prob_median": 0.1,
            "prob_lo": 0.01,
            "prob_hi": 0.2,
            "prob_lo90": 0.02,
            "prob_hi90": 0.18,
            "items_median": 7.9,
            "items_lo": 0.79,
            "items_hi": 15.8,
            "items_lo90": 1.58,
            "items_hi90": 14.22,
        }
    )
    fields = _ca_margin_fields("biv", row)
    assert fields["biv_ame_prob_median"] == pytest.approx(0.1)
    assert fields["biv_ame_items_median"] == pytest.approx(7.9)
    assert len(fields) == 10


def test_concurrent_sd_margin_requires_one_row():
    df = pd.DataFrame(
        {
            "term": ["L", "L"],
            "scale": ["+1 SD", "+3 items"],
            "prob_median": [0.1, 0.05],
        }
    )
    assert _ca_sd_margin(df, "L")["prob_median"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match=r"Expected one \+1 SD marginal"):
        _ca_sd_margin(df, "B")


def test_concurrent_output_writer_enforces_full_published_fit_contract(tmp_path):
    predictors = ("L", "B", "TR", "TE", "R", "E")
    margin_summary = pd.Series(
        {
            "prob_median": 0.1,
            "prob_lo": 0.01,
            "prob_hi": 0.2,
            "prob_lo90": 0.02,
            "prob_hi90": 0.18,
            "items_median": 7.9,
            "items_lo": 0.79,
            "items_hi": 15.8,
            "items_lo90": 1.58,
            "items_hi90": 14.22,
        }
    )
    association_rows = []
    marginal_rows = []
    diagnostic_rows = []
    for timepoint in range(1, 5):
        diagnostic_rows.append(
            {
                "timepoint": timepoint,
                "fit_kind": "adjusted",
                "predictor": "all",
                "n": 53,
                "n_predictors": 6,
                "converged": True,
                "max_rhat": 1.001,
                "min_ess": 1000.0,
                "min_bfmi": 0.9,
                "n_divergences": 0,
            }
        )
        for predictor in predictors:
            row = {
                "timepoint": timepoint,
                "predictor": predictor,
                "label": _ca_label(predictor),
                "n": 53,
                "predictor_n": 52,
                "predictor_imputed_n": 1,
                "ame_contrast": "+1 SD",
                "adj_converged": True,
                "biv_converged": True,
            }
            for prefix in ("adj", "biv"):
                row.update(
                    {
                        f"{prefix}_mean": 0.1,
                        f"{prefix}_lo": 0.01,
                        f"{prefix}_hi": 0.2,
                        f"{prefix}_lo90": 0.02,
                        f"{prefix}_hi90": 0.18,
                        f"{prefix}_prob_pos": 0.95,
                        **_ca_margin_fields(prefix, margin_summary),
                    }
                )
            association_rows.append(row)
            for adjustment in ("adjusted", "bivariate"):
                for scale in ("+1 SD", "+3 items"):
                    marginal_rows.append(
                        {
                            "timepoint": timepoint,
                            "adjustment": adjustment,
                            "term": predictor,
                            "role": "association",
                            "scale": scale,
                            "prob_median": 0.1,
                            "prob_lo": 0.01,
                            "prob_hi": 0.2,
                            "prob_lo90": 0.02,
                            "prob_hi90": 0.18,
                            "items_median": 7.9,
                            "items_lo": 0.79,
                            "items_hi": 15.8,
                            "items_lo90": 1.58,
                            "items_hi90": 14.22,
                            "prob_pos": 0.95,
                            "label": _ca_label(predictor),
                            "converged": True,
                        }
                    )
            diagnostic_rows.append(
                {
                    "timepoint": timepoint,
                    "fit_kind": "bivariate",
                    "predictor": predictor,
                    "n": 53,
                    "n_predictors": 1,
                    "converged": True,
                    "max_rhat": 1.001,
                    "min_ess": 1000.0,
                    "min_bfmi": 0.9,
                    "n_divergences": 0,
                }
            )

    ctx = SimpleNamespace(output_dir=str(tmp_path), tables={})
    association_df, marginal_df, diagnostic_df = _write_concurrent_outputs(
        ctx,
        association_rows=association_rows,
        marginal_frames=[pd.DataFrame(marginal_rows)],
        diagnostic_rows=diagnostic_rows,
    )

    assert association_df.shape == (24, 41)
    assert set(marginal_df["adjustment"]) == {"adjusted", "bivariate"}
    assert set(marginal_df["scale"]) == {"+1 SD", "+3 items"}
    assert marginal_df.shape == (96, 18)
    assert diagnostic_df.shape == (28, 10)
    assert set(diagnostic_df.columns) >= {
        "max_rhat",
        "min_ess",
        "min_bfmi",
        "n_divergences",
    }
    for stem in (
        "concurrent_associations",
        "concurrent_marginals",
        "concurrent_fit_diagnostics",
    ):
        written = pd.read_csv(tmp_path / f"{stem}.csv")
        assert len(written) == len(ctx.tables[stem])

    with pytest.raises(ValueError, match="prob_lo90"):
        _write_concurrent_outputs(
            ctx,
            association_rows=association_rows,
            marginal_frames=[pd.DataFrame(marginal_rows).drop(columns="prob_lo90")],
            diagnostic_rows=diagnostic_rows,
        )


def test_shared_concurrent_results_partial_is_focal_outcome_neutral():
    repo = Path(__file__).resolve().parents[2]
    text = (repo / "docs/models/_partials/_results_concurrent.qmd").read_text()
    assert "word reading" not in text.lower()
    assert "words read" not in text.lower()
    assert "`adjustment`" in text
    assert "concurrent_fit_diagnostics.csv" in text


def test_concurrent_spec_docs_avoid_unqualified_attenuation_claims():
    repo = Path(__file__).resolve().parents[2]
    model_dir = repo / "src/language_reading_predictors/statistical_models"
    for number in range(1, 7):
        text = (model_dir / f"lrp_rli_ca_{number:03d}.py").read_text().lower()
        assert "associations attenuate toward zero" not in text
        assert "gap is itself informative" not in text
