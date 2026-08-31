# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Only a real absence may produce an ``unavailable`` prior check (#637 stage 1).

``prior_pushforward.csv`` exists because a *missing* artefact reads as a clean one
(#381): "no flags" and "not measured" look identical on a rendered page. The
families that write it caught every ``Exception`` while computing it, so a
``KeyError``, a wrong dimension or a schema defect produced the same
``status="unavailable"`` row as an honestly absent prior group — and the release
stage checks the file's presence, not its row status.

These tests pin the repaired contract: :class:`PriorEvidenceUnavailable` is the
only condition that may be recorded, everything else fails the fit, and an
``unavailable`` row attaches a named publication qualification rather than
withholding a release.
"""

from __future__ import annotations

import ast
import json
import pathlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from language_reading_predictors.statistical_models import prior_artifacts as PA
from language_reading_predictors.statistical_models.prior_artifacts import (
    PriorEvidenceUnavailable,
    require_prior_evidence,
)

PIPELINES = pathlib.Path(PA.__file__).parent / "pipelines"


def _tree(prior_vars: dict | None, *, n_obs: int = 4, n_draws: int = 8):
    """A trace-shaped DataTree, optionally without a ``prior`` group at all."""
    groups = {
        "posterior": xr.Dataset(
            {"alpha": (("chain", "draw"), np.zeros((1, n_draws)))},
            coords={"chain": [0], "draw": list(range(n_draws))},
        )
    }
    if prior_vars is not None:
        data = {}
        for name, value in prior_vars.items():
            array = np.asarray(value, dtype=float)
            if array.ndim == 0:
                data[name] = (("chain", "draw"), np.full((1, n_draws), float(array)))
            else:
                data[name] = (
                    ("chain", "draw", "obs_id"),
                    np.broadcast_to(array, (1, n_draws, array.shape[-1])).copy(),
                )
        groups["prior"] = xr.Dataset(
            data,
            coords={
                "chain": [0],
                "draw": list(range(n_draws)),
                "obs_id": list(range(n_obs)),
            },
        )
    return xr.DataTree.from_dict(groups)


def _ctx(tmp_path, tree):
    return SimpleNamespace(
        output_dir=str(tmp_path),
        prior_samples=tree,
        trace=tree,
        tables={},
        reporting=SimpleNamespace(ci_prob=0.89),
        spec=SimpleNamespace(model_id="lrp-test-001"),
    )


# ---------------------------------------------------------------------------
# The narrow exception itself
# ---------------------------------------------------------------------------


def test_an_absent_prior_group_is_an_expected_absence():
    with pytest.raises(PriorEvidenceUnavailable, match="not sampled or persisted"):
        require_prior_evidence(_tree(None), terms=("tau",), what="the check")


def test_an_absent_term_names_itself():
    with pytest.raises(PriorEvidenceUnavailable, match="eta"):
        require_prior_evidence(
            _tree({"tau": 0.2}), terms=("tau", "eta"), what="the check"
        )


def test_a_present_group_with_every_term_is_returned():
    group = require_prior_evidence(
        _tree({"tau": 0.2, "eta": np.zeros(4)}), terms=("tau", "eta")
    )
    assert "tau" in group


def test_the_exception_is_a_lookup_error_not_a_catch_all():
    """It must be catchable narrowly and never shadow an arbitrary failure."""
    assert issubclass(PriorEvidenceUnavailable, LookupError)
    assert not issubclass(ValueError, PriorEvidenceUnavailable)
    assert not issubclass(TypeError, PriorEvidenceUnavailable)


# ---------------------------------------------------------------------------
# The shared row builders
# ---------------------------------------------------------------------------


def test_marginal_rows_record_an_absent_term_and_still_write_the_table(tmp_path):
    ctx = _ctx(tmp_path, _tree({"eta": np.zeros(4)}))
    rows = PA.marginal_pushforward_rows(
        ctx, [("beta_missing", "an absent coefficient")], n_trials=10
    )
    assert [row["status"] for row in rows] == ["unavailable"]
    assert "beta_missing" in str(rows[0]["reason"])


def test_marginal_rows_let_a_wrong_dimension_fail_the_run(tmp_path):
    """A defect must not be filed as missing evidence.

    ``eta`` here carries a dimension the transform cannot stack, which is a
    schema error — before #637 it became a valid table row saying the prior check
    was unavailable.
    """
    tree = xr.DataTree.from_dict(
        {
            "prior": xr.Dataset(
                {
                    "beta": (("chain", "draw"), np.zeros((1, 4))),
                    "eta": (("chain", "draw", "wrong_dim"), np.zeros((1, 4, 3))),
                },
                coords={"chain": [0], "draw": [0, 1, 2, 3], "wrong_dim": [0, 1, 2]},
            )
        }
    )
    ctx = _ctx(tmp_path, tree)
    with pytest.raises(ValueError, match="obs_id"):
        PA.marginal_pushforward_rows(ctx, [("beta", "a coefficient")], n_trials=10)


def test_marginal_rows_propagate_a_defect_rather_than_recording_it(
    tmp_path, monkeypatch
):
    ctx = _ctx(tmp_path, _tree({"beta": 0.3, "eta": np.zeros(4)}))

    def explode(*_args, **_kwargs):
        raise KeyError("obs_id")

    monkeypatch.setattr(PA._report, "marginal_prior_pushforward", explode)
    with pytest.raises(KeyError, match="obs_id"):
        PA.marginal_pushforward_rows(ctx, [("beta", "a coefficient")], n_trials=10)


def test_at_mean_rows_record_an_absent_prior_group(tmp_path):
    ctx = _ctx(tmp_path, _tree(None))
    rows = PA.at_mean_pushforward_rows(
        ctx,
        [("beta_x", "one SD on x")],
        n_trials=10,
        own_pre_logit_mean=0.0,
    )
    assert [row["status"] for row in rows] == ["unavailable"]
    assert "prior group" in str(rows[0]["reason"])


def test_at_mean_rows_propagate_a_defect_rather_than_recording_it(
    tmp_path, monkeypatch
):
    ctx = _ctx(tmp_path, _tree({"alpha": 0.1, "gamma_own": 0.2, "beta_x": 0.3}))

    def explode(*_args, **_kwargs):
        raise ValueError("pushforward schema drift")

    monkeypatch.setattr(PA._report, "pushforward_values", explode)
    with pytest.raises(ValueError, match="schema drift"):
        PA.at_mean_pushforward_rows(
            ctx,
            [("beta_x", "one SD on x")],
            n_trials=10,
            own_pre_logit_mean=0.0,
        )


# ---------------------------------------------------------------------------
# A family call site, end to end
# ---------------------------------------------------------------------------


def test_the_mechanism_curve_check_records_only_an_absent_prior_group(tmp_path):
    from language_reading_predictors.statistical_models.pipelines.mechanism import (
        _write_mechanism_prior_pushforward,
    )

    ctx = _ctx(tmp_path, _tree(None))
    _write_mechanism_prior_pushforward(
        ctx,
        x_exposure=np.arange(4, dtype=float),
        outcome="W",
        exposure_label="letter sounds",
        exposure_n_trials=26,
        ref_quantiles=(0.25, 0.75),
    )
    table = pd.read_csv(tmp_path / "prior_pushforward.csv")
    assert list(table["status"]) == ["unavailable"]
    assert "prior group" in table.loc[0, "reason"]


def test_the_mechanism_curve_check_fails_on_an_unregistered_outcome(tmp_path):
    """A schema defect, not an absence — it used to be filed as one."""
    from language_reading_predictors.statistical_models.pipelines.mechanism import (
        _write_mechanism_prior_pushforward,
    )

    ctx = _ctx(tmp_path, _tree({"eta": np.zeros(4)}))
    with pytest.raises(KeyError):
        _write_mechanism_prior_pushforward(
            ctx,
            x_exposure=np.arange(4, dtype=float),
            outcome="NOT_A_MEASURE",
            exposure_label="letter sounds",
            exposure_n_trials=26,
            ref_quantiles=(0.25, 0.75),
        )


def test_the_mechanism_curve_check_fails_on_a_mismatched_exposure_vector(tmp_path):
    from language_reading_predictors.statistical_models.pipelines.mechanism import (
        _write_mechanism_prior_pushforward,
    )

    ctx = _ctx(tmp_path, _tree({"eta": np.zeros(4)}, n_obs=4))
    with pytest.raises(ValueError, match="rows but the fit has"):
        _write_mechanism_prior_pushforward(
            ctx,
            x_exposure=np.arange(9, dtype=float),
            outcome="W",
            exposure_label="letter sounds",
            exposure_n_trials=26,
            ref_quantiles=(0.25, 0.75),
        )


# ---------------------------------------------------------------------------
# No family may reinstate the blanket handler
# ---------------------------------------------------------------------------


def _blanket_handlers(path: pathlib.Path) -> list[str]:
    """Functions that build an ``unavailable`` row under a catch-all ``except``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        names = {
            child.attr if isinstance(child, ast.Attribute) else child.id
            for child in ast.walk(node)
            if isinstance(child, (ast.Attribute, ast.Name))
        }
        if "unavailable_pushforward" not in names:
            continue
        caught = node.type
        caught_names = {
            child.id
            for child in ast.walk(caught)
            if isinstance(child, ast.Name)
        } if caught is not None else {"<bare except>"}
        if not caught_names <= {"PriorEvidenceUnavailable"}:
            offenders.append(f"{path.name}:{node.lineno} catches {sorted(caught_names)}")
    return offenders


def test_no_family_records_unavailable_prior_evidence_from_a_catch_all():
    """The defect was structural: four families, one blanket ``except Exception``.

    Kept as a dependency-style guard rather than a source-shape check — it reads
    what the handler *catches*, which is precisely the contract.
    """
    offenders: list[str] = []
    for path in [*sorted(PIPELINES.glob("*.py")), pathlib.Path(PA.__file__)]:
        offenders.extend(_blanket_handlers(path))
    assert offenders == [], offenders


# ---------------------------------------------------------------------------
# Release policy
# ---------------------------------------------------------------------------


def _unavailable_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "estimand": "beta_cohort",
                "estimand_label": "the per-protocol cohort contrast",
                "role": "association",
                "scale": "items",
                "status": "unavailable",
                "reason": "this fit has no persisted prior group",
                "n_trials": 0,
            }
        ]
    )


def test_unavailable_prior_evidence_qualifies_but_does_not_withhold(tmp_path):
    """The stated policy (#637 stage 1), exercised on a stored directory."""
    from .test_release_decision import _fit_dir
    from language_reading_predictors.statistical_models.release import (
        evaluate_publication,
    )

    directory = _fit_dir(tmp_path)
    _unavailable_table().to_csv(directory / "prior_pushforward.csv", index=False)

    evaluation = evaluate_publication(directory)
    assert evaluation.publishable is True
    assert "beta_cohort" in evaluation.publication_qualification
    assert "prior" in evaluation.publication_qualification
    assert json.dumps(evaluation.as_dict())


def test_an_available_prior_check_attaches_no_qualification(tmp_path):
    from .test_release_decision import _fit_dir
    from language_reading_predictors.statistical_models.release import (
        evaluate_publication,
    )

    directory = _fit_dir(tmp_path)
    table = _unavailable_table()
    table.loc[0, "status"] = "available"
    table.to_csv(directory / "prior_pushforward.csv", index=False)

    evaluation = evaluate_publication(directory)
    assert evaluation.publishable is True
    assert "prior check" not in evaluation.publication_qualification


def test_a_legacy_bare_numeric_table_attaches_no_qualification(tmp_path):
    """115 stored fits predate the labelled schema; every row in them is available."""
    from .test_release_decision import _fit_dir
    from language_reading_predictors.statistical_models.release import (
        evaluate_publication,
    )

    directory = _fit_dir(tmp_path)
    pd.DataFrame(
        [{"prior_logit_median": -0.01, "prior_items_median": -0.03, "n_trials": 12}]
    ).to_csv(directory / "prior_pushforward.csv", index=False)

    evaluation = evaluate_publication(directory)
    assert evaluation.publishable is True
    assert "prior check" not in evaluation.publication_qualification
