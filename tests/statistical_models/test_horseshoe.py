# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the regularized-horseshoe ranking cross-check (#116 Phase E).

The builder tests are smoke tests on synthetic data (matching the rest of
``test_factories``): each variant *builds*, exposes the expected horseshoe RVs,
and can draw a small prior predictive sample. Full posterior correctness is
validated by the end-to-end dev fits. The ranking test is deterministic: a
synthetic posterior with one planted strong coefficient must come out on top with
``p_abs_gt_delta`` near 1, and the schema must match what the report partial reads.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pymc as pm
import xarray as xr

from language_reading_predictors.statistical_models.factories import (
    build_horseshoe_model,
)
from language_reading_predictors.statistical_models.preprocessing import (
    load_and_prepare,
)
from language_reading_predictors.statistical_models.reporting import (
    horseshoe_ranking,
)

from .test_factories import _write_synthetic

_HS_CORE = {"hs_tau", "hs_c2", "hs_lambda", "hs_z", "alpha", "kappa"}
_PREDICTORS = ["L", "R", "E", "T", "age"]


def _posterior(beta, predictors, lam=None):
    """Wrap (chain, draw, predictor) arrays as an object exposing ``.posterior``.

    Mirrors ``test_reporting._trace`` so the ranking test is independent of the
    installed ArviZ version's ``from_dict`` behaviour.
    """
    n_chain, n_draw, _ = beta.shape
    data = {"beta": (("chain", "draw", "predictor"), beta)}
    if lam is not None:
        data["hs_lambda"] = (("chain", "draw", "predictor"), lam)
    ds = xr.Dataset(
        data,
        coords={
            "chain": np.arange(n_chain),
            "draw": np.arange(n_draw),
            "predictor": list(predictors),
        },
    )
    return SimpleNamespace(posterior=ds)


def test_horseshoe_gain_builds(tmp_path):
    """Gain framing (span): own-baseline term + horseshoe over standardised T1
    baselines; ``beta`` has one entry per predictor; prior predictive is drawable."""
    p = _write_synthetic(tmp_path)
    prep = load_and_prepare(path=p, phase_mode="span")
    built = build_horseshoe_model(prep, outcome_symbol="W", predictors=_PREDICTORS)

    names = {v.name for v in built.model.free_RVs}
    assert _HS_CORE.issubset(names)
    assert "gamma_own" in names  # gain conditions on its own baseline
    beta = built.model["beta"]
    assert beta.eval().shape == (len(_PREDICTORS),)
    assert list(built.model.coords["predictor"]) and len(
        built.model.coords["predictor"]
    ) == len(_PREDICTORS)

    with built.model:
        pp = pm.sample_prior_predictive(draws=5, random_seed=1)
    assert pp.prior_predictive["y_post"].shape[-1] == prep.n_obs


def test_horseshoe_level_builds_age_as_predictor(tmp_path):
    """Level framing (levels): subject random intercept + horseshoe over concurrent
    standardised levels; no own-baseline term. Because ``age`` is in the predictor
    list it is ranked as a horseshoe coefficient and the separate unshrunk
    ``gamma_A`` slope is suppressed, so age is not double-counted (#160 review)."""
    p = _write_synthetic(tmp_path)
    prep = load_and_prepare(path=p, phase_mode="levels")
    built = build_horseshoe_model(
        prep, outcome_symbol="W", predictors=_PREDICTORS, gain=False
    )

    names = {v.name for v in built.model.free_RVs}
    assert _HS_CORE.issubset(names)
    assert "gamma_own" not in names  # level model is concurrent, not baseline-conditioned
    assert "gamma_A" not in names  # age is horseshoe-ranked, so no separate fixed slope
    assert "u_child_raw" in names  # non-centred subject random intercept
    assert built.model["beta"].eval().shape == (len(_PREDICTORS),)  # incl. age

    with built.model:
        pp = pm.sample_prior_predictive(draws=5, random_seed=2)
    assert pp.prior_predictive["y_post"].shape[-1] == prep.n_obs


def test_horseshoe_level_age_adjusted_when_not_a_predictor(tmp_path):
    """When age is *not* a horseshoe predictor, the fixed ``gamma_A`` age slope is
    added (age adjusted for but not ranked) — the complement of the guard above."""
    p = _write_synthetic(tmp_path)
    prep = load_and_prepare(path=p, phase_mode="levels")
    built = build_horseshoe_model(
        prep, outcome_symbol="W", predictors=["L", "R", "E", "T"], gain=False
    )
    names = {v.name for v in built.model.free_RVs}
    assert "gamma_A" in names  # age adjusted for, not ranked
    assert built.model["beta"].eval().shape == (4,)  # no age column in the design


def test_horseshoe_gain_requires_span_or_itt(tmp_path):
    """A levels-mode frame must be rejected by the gain path (guards the caller)."""
    import pytest

    p = _write_synthetic(tmp_path)
    prep = load_and_prepare(path=p, phase_mode="levels")
    with pytest.raises(ValueError, match="phase_mode"):
        build_horseshoe_model(prep, outcome_symbol="W", predictors=_PREDICTORS, gain=True)


def test_horseshoe_ranking_schema_and_order():
    """A planted strong coefficient ranks first; the schema matches the report."""
    rng = np.random.default_rng(0)
    chains, draws = 2, 400
    beta = np.stack(
        [
            rng.normal(1.0, 0.10, (chains, draws)),   # strong +  -> should rank 1
            rng.normal(0.0, 0.02, (chains, draws)),   # noise
            rng.normal(-0.6, 0.10, (chains, draws)),  # moderate -
            rng.normal(0.0, 0.02, (chains, draws)),   # noise
        ],
        axis=-1,
    )
    lam = np.abs(rng.normal(0.0, 1.0, (chains, draws, 4))) + 1e-3
    idata = _posterior(beta, ["signal", "n1", "moderate", "n2"], lam=lam)

    df = horseshoe_ranking(idata, delta=0.1)

    expected_cols = {
        "rank", "predictor", "p_abs_gt_delta", "beta_mean", "beta_sd",
        "beta_hdi_3", "beta_hdi_97", "sign", "lambda_mean",
    }
    assert expected_cols.issubset(df.columns)
    assert len(df) == 4
    assert list(df["rank"]) == [1, 2, 3, 4]
    assert (df["p_abs_gt_delta"].between(0.0, 1.0)).all()

    # Strong signal on top, near-certain; noise predictors near zero.
    assert df.iloc[0]["predictor"] == "signal"
    assert df.iloc[0]["p_abs_gt_delta"] > 0.95
    assert df.iloc[0]["sign"] == "+"
    noise = df[df["predictor"].isin(["n1", "n2"])]
    assert (noise["p_abs_gt_delta"] < 0.1).all()
    # Direction is read from the posterior-mean sign.
    assert df[df["predictor"] == "moderate"]["sign"].item() == "-"


def test_horseshoe_ranking_without_lambda():
    """``lambda_mean`` is optional: a posterior lacking ``hs_lambda`` still ranks."""
    rng = np.random.default_rng(1)
    beta = rng.normal(0.0, 0.3, (2, 200, 3))
    idata = _posterior(beta, ["a", "b", "c"])
    df = horseshoe_ranking(idata)
    assert "lambda_mean" not in df.columns
    assert len(df) == 3 and list(df["rank"]) == [1, 2, 3]
