# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Numerical checks for held-out prediction, independent of fold sampling."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pymc as pm
import pytest
import xarray as xr
from scipy.stats import norm

from language_reading_predictors.statistical_models.new_child_evidence import new_child_kfold_verdict
from language_reading_predictors.statistical_models.new_child_kfold import KFoldPlan, KFoldValidation, _score_held_out
from language_reading_predictors.statistical_models.new_child_predictive import NewChildPlan


def _validation():
    return KFoldValidation(
        plan=NewChildPlan(child_dims=("child",)),
        kfold=KFoldPlan(n_folds=2),
        n_children=4,
        n_scored=4,
        elpd=-10.0,
        elpd_se=1.0,
        pointwise_elpd=np.full(4, -2.5),
        fold_of_child=np.array([0, 0, 1, 1]),
        fold_converged={0: True, 1: True},
        latents_redrawn=("u",),
        observed_nodes=("y_post",),
        pointwise_batch_elpd=np.full((2, 4), -2.5),
        integration_diagnostics={0: {"stable": True, "n_latent_draws": 64}, 1: {"stable": True, "n_latent_draws": 64}},
    )


@pytest.mark.parametrize(
    "change",
    [
        {"pointwise_batch_elpd": None},
        {"pointwise_batch_elpd": np.zeros((1, 4))},
        {"pointwise_batch_elpd": np.zeros((2, 3))},
        {"pointwise_batch_elpd": np.empty((0, 4))},
        {"pointwise_batch_elpd": np.full((2, 4), np.nan)},
        {"pointwise_batch_elpd": np.array([[-2.5] * 4, [-1.5] * 4])},
        {"integration_diagnostics": {}},
        {"pointwise_elpd": np.array([-2.5, -2.5, -2.5, np.inf])},
        {"elpd_se": np.nan},
        {"fold_converged": {0: True}},
    ],
)
def test_complete_coverage_and_convergence_cannot_replace_numerical_evidence(change):
    result = replace(_validation(), **change)
    assert not result.complete
    assert np.isnan(result.summary_row()["elpd_kfold"])
    assert not new_child_kfold_verdict(result.summary_row()).reliable


def test_live_and_csv_verdicts_agree_and_do_not_trust_a_legacy_complete_flag():
    row = _validation().summary_row()
    assert new_child_kfold_verdict(row).reliable
    assert new_child_kfold_verdict({k: str(v) for k, v in row.items()}).reliable
    for field, value in (
        ("validation_schema_version", None),
        ("max_pointwise_batch_difference", 0.2),
        ("total_batch_difference", 2.0),
        ("integration_stable", False),
        ("n_latent_draws_max_used", np.inf),
    ):
        assert not new_child_kfold_verdict({**row, field: value}).reliable


@pytest.mark.parametrize(
    "kwargs", [{"pointwise_tolerance": 0}, {"total_tolerance": np.nan}, {"max_latent_draws": 2}, {"n_latent_draws": 1}]
)
def test_invalid_integration_budgets_and_tolerances_are_refused(kwargs):
    with pytest.raises(ValueError):
        KFoldPlan(**kwargs)


def test_integration_choices_do_not_change_the_identity_of_saved_fold_fits():
    original = KFoldPlan()
    changed = replace(original, max_latent_draws=1024, n_latent_draws=128, pointwise_tolerance=0.05)
    assert original.partition_identity() == changed.partition_identity()
    assert original.as_dict() != changed.as_dict()


def _controlled_score(monkeypatch, *, pattern, max_draws=32):
    calls = []
    seeds = []
    model = pm.Model()
    posterior = xr.Dataset({"alpha": (("chain", "draw"), np.zeros((1, 2)))})
    dataset = xr.Dataset(
        {
            "u": (("chain", "draw", "child"), np.zeros((1, 2, 2))),
            "y_post": (("chain", "draw", "row"), np.zeros((1, 2, 2))),
        }
    )

    def predictive(*args, **kwargs):
        seeds.append(kwargs["random_seed"])
        return SimpleNamespace(posterior_predictive=dataset)

    def likelihood(*args, **kwargs):
        index = len(calls)
        calls.append(index)
        values = np.full((1, 2, 2), pattern(index))
        return SimpleNamespace(log_likelihood=xr.Dataset({"y_post": (("chain", "draw", "row"), values)}))

    monkeypatch.setattr(pm, "sample_posterior_predictive", predictive)
    monkeypatch.setattr(pm, "compute_log_likelihood", likelihood)
    scored = _score_held_out(
        model,
        NewChildPlan(child_dims=("child",)),
        KFoldPlan(n_folds=2, n_latent_draws=2, max_latent_draws=max_draws),
        transplanted=posterior,
        latents=("u",),
        nodes=("y_post",),
        maps={"y_post": np.array([0, 1])},
        n_children=2,
        density_model=model,
        held_out=np.array([0, 1]),
        fold=0,
    )
    assert len(seeds) == len(set(seeds))
    assert len(scored.predictive["y_post"]) == 1
    return scored


def test_disagreement_increases_the_budget_without_new_fold_sampling(monkeypatch):
    result = _controlled_score(monkeypatch, pattern=lambda i: -2.0 if i == 1 else 0.0)
    assert result.diagnostics["stable"]
    assert result.diagnostics["n_latent_draws"] == 32
    # Average predictive densities, not their logarithms.
    expected = np.log((31 + np.exp(-2)) / 32)
    np.testing.assert_allclose(result.scored, expected)


def test_persistent_batch_disagreement_exhausts_the_budget_and_fails(monkeypatch):
    result = _controlled_score(monkeypatch, pattern=lambda i: -2.0 * (i % 2), max_draws=8)
    assert not result.diagnostics["stable"]
    assert result.diagnostics["n_latent_draws"] == 8
    assert result.diagnostics["max_pointwise_batch_difference"] == pytest.approx(2)


def test_nonfinite_density_cannot_pass_batch_checks(monkeypatch):
    with np.errstate(invalid="ignore"):
        result = _controlled_score(monkeypatch, pattern=lambda i: -np.inf)
    assert not result.diagnostics["stable"]
    assert not result.diagnostics["finite"]


def test_real_latent_integration_agrees_with_an_analytic_gaussian_predictive():
    observed = np.array([0.5, -0.5])
    with pm.Model(coords={"child": [0, 1], "row": [0, 1]}) as model:
        alpha = pm.Normal("alpha", 0, 1)
        u = pm.Normal("u", 0, 1, dims="child")
        pm.Normal("y_post", alpha + u, 1, observed=observed, dims="row")
    posterior = xr.Dataset({"alpha": (("chain", "draw"), np.zeros((1, 1500)))})
    result = _score_held_out(
        model,
        NewChildPlan(child_dims=("child",)),
        KFoldPlan(n_folds=2, n_latent_draws=4, max_latent_draws=16),
        transplanted=posterior,
        latents=("u",),
        nodes=("y_post",),
        maps={"y_post": np.array([0, 1])},
        n_children=2,
        density_model=model,
        held_out=np.array([0, 1]),
        fold=0,
    )
    assert result.diagnostics["stable"]
    np.testing.assert_allclose(result.scored, norm.logpdf(observed, scale=np.sqrt(2)), atol=0.03)
