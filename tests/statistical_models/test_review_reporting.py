# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Saved-table and report contracts for the September methodology corrections."""

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import pytest

from language_reading_predictors.statistical_models.new_child_predictive import NewChildPlan, NewChildValidation


PARTIALS = Path(__file__).resolve().parents[2] / "docs/models/_partials"


def _blocks(name):
    return re.findall(r"```\{python\}\n(.*?)\n```", (PARTIALS / name).read_text(encoding="utf-8"), re.S)


def _validation_row():
    return NewChildValidation(
        plan=NewChildPlan(child_dims=("child",)), n_children=2, posterior_draws_used=1000,
        elpd=-10, elpd_se=2, p_loo=1, pointwise_elpd=np.array([-4., -6.]),
        pareto_k=np.array([.1, .2]), good_k=.7, latents_redrawn=(), observed_nodes=("y",),
    ).summary_row()


@pytest.mark.parametrize("change", [{}, {"reliable": "False"}, {"validation_schema_version": 1},
                                   {"max_pareto_k": np.nan}, {"elpd_se": np.nan}])
def test_new_child_csv_reader_requires_current_finite_evidence(tmp_path, capsys, change):
    row = _validation_row() | change
    path = tmp_path / "new_child_loo.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    saved = pd.read_csv(path)
    namespace = {"config": {}, "_csv": lambda name: saved if name == path.name else None}
    for block in _blocks("_new_child_validation.qmd"):
        exec(compile(block, "_new_child_validation.qmd", "exec"), namespace)
    output = capsys.readouterr().out
    if change:
        assert "The ELPD is withheld" in output
        assert "**-10.0**" not in output
    else:
        assert "**-10.0**" in output


@pytest.mark.parametrize("method", ["legacy_interval_widths", "paired_ame_draws_v1"])
def test_joint_report_does_not_reuse_the_retired_width_decomposition(tmp_path, capsys, method):
    record = {
        "status": "compared", "channel_status": "measured", "channel_method": method,
        "median_shift": 0., "direction_probability_shift": 0., "interval_width_ratio": 1.,
        "contrast_variance_change": 0., "marginal_variance_channel": .01, "covariance_variance_channel": -.01,
    }
    (tmp_path / "release_decision.json").write_text(json.dumps({"dependence_contrast": record}))
    block = next(block for block in _blocks("_results_joint.qmd") if "_dep_path =" in block)
    exec(compile(block, "_results_joint.qmd", "exec"), {"_here": tmp_path, "json": json, "config": {}})
    output = capsys.readouterr().out
    if method == "paired_ame_draws_v1":
        assert "variance change" in output
    else:
        assert "older interval-width tables cannot supply it" in output
    assert "implied posterior correlation" not in output


def test_legacy_dependence_labels_are_translated_to_spread_only(capsys):
    frame = pd.DataFrame([{"parameter": "rho", "role": "residual correlation", "verdict": "prior-dominated",
                           "posterior_sd": .33, "prior_sd": .33, "posterior_prior_sd_ratio": 1.}])
    block = next(block for block in _blocks("_results_joint.qmd") if "_dep_id =" in block)
    exec(compile(block, "_results_joint.qmd", "exec"), {
        "_joint_correlated": True, "_csv": lambda name, **kwargs: frame if name == "dependence_identification.csv" else None,
    })
    output = capsys.readouterr().out
    assert "little or no contraction" in output
    assert "similar standard deviations, not equal distributions" in output
    assert "prior-dominated" not in output
