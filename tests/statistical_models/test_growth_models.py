# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit + smoke tests for the joint growth-curve models (LRP69/70, issue #187).

Covers the three pieces the models add: the time-invariant baseline-covariate
loader on :func:`load_wave_panel`, the pure ``growth_association_summary`` read-out
in :mod:`statistical_models.reporting`, and that :func:`build_growth_model` builds
(both layers) and draws a small prior predictive. Full posterior correctness is
validated by the end-to-end fits in ``scripts/fit_statistical_model.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.factories import build_growth_model
from language_reading_predictors.statistical_models.preprocessing import load_wave_panel
from language_reading_predictors.statistical_models.reporting import (
    growth_association_summary,
)

_OUTCOMES = ("R", "E", "T", "W", "L")


def _write_growth_csv(tmp_path, n_children: int = 20, seed: int = 5, drop_block=None):
    """Four-wave long CSV with the five growth measures + a t1-only ``blocks``.

    ``blocks`` is written only at wave 1 (as in the real data). ``drop_block`` is an
    optional subject id whose ``blocks`` is left missing everywhere, to exercise the
    completeness guard.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_children):
        sid = f"S{i:03d}"
        age0 = int(rng.integers(60, 110))
        block = int(rng.integers(0, 25))
        for t in (1, 2, 3, 4):
            row = {
                V.SUBJECT_ID: sid,
                V.TIME: t,
                V.AGE: age0 + 6 * (t - 1),
                V.ROWPVT: int(rng.integers(0, 100)),
                V.EOWPVT: int(rng.integers(0, 100)),
                V.TROG: int(rng.integers(0, 28)),
                V.EWRSWR: int(rng.integers(0, 65)),
                V.YARCLET: int(rng.integers(0, 33)),
                V.ATTEND: int(rng.integers(0, 10)),
                # blocks recorded at t1 only; NaN at later waves.
                V.BLOCKS: block if t == 1 else np.nan,
            }
            if drop_block is not None and sid == drop_block:
                row[V.BLOCKS] = np.nan
            rows.append(row)
    p = tmp_path / "rli.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# Baseline-covariate loader
# ---------------------------------------------------------------------------


def test_baseline_covariate_loads_standardised_and_aligned(tmp_path):
    p = _write_growth_csv(tmp_path, n_children=20)
    panel = load_wave_panel(
        path=p, outcomes=_OUTCOMES, baseline_covariates=("blocks",)
    )

    z = panel.baseline["blocks"]
    raw = panel.baseline_raw["blocks"]
    assert z.shape == (20,)
    assert raw.shape == (20,)
    # Standardised over children: mean ~0, sd ~1.
    assert abs(float(z.mean())) < 1e-9
    assert abs(float(z.std(ddof=1)) - 1.0) < 1e-9
    # Raw is the t1 block-design value aligned to sorted subject_ids; the scaler
    # inverts the standardised vector back to the raw values.
    src = pd.read_csv(p)
    expected = (
        src[src[V.TIME] == 1]
        .set_index(V.SUBJECT_ID)
        .loc[panel.subject_ids, V.BLOCKS]
        .to_numpy(float)
    )
    assert np.allclose(raw, expected)
    assert np.allclose(panel.baseline_scaler["blocks"].inverse(z), raw)


def test_baseline_covariate_missing_for_a_child_raises(tmp_path):
    p = _write_growth_csv(tmp_path, n_children=12, drop_block="S003")
    with pytest.raises(ValueError, match="blocks"):
        load_wave_panel(path=p, outcomes=_OUTCOMES, baseline_covariates=("blocks",))


def test_no_baseline_covariates_leaves_baseline_empty(tmp_path):
    p = _write_growth_csv(tmp_path, n_children=8)
    panel = load_wave_panel(path=p, outcomes=_OUTCOMES)
    assert panel.baseline == {}


# ---------------------------------------------------------------------------
# growth_association_summary (pure read-out)
# ---------------------------------------------------------------------------


def _growth_trace(gamma, delta, loading=None):
    """Wrap synthetic (chain, draw, outcome) coefficient arrays as ``.posterior``."""
    n_chain, n_draw, K = gamma.shape
    data = {
        "gamma": (("chain", "draw", "outcome"), gamma),
        "delta": (("chain", "draw", "outcome"), delta),
    }
    if loading is not None:
        data["loading"] = (("chain", "draw", "outcome"), loading)
    ds = xr.Dataset(
        data,
        coords={
            "chain": np.arange(n_chain),
            "draw": np.arange(n_draw),
            "outcome": ["R", "E", "T"][:K],
        },
    )
    return SimpleNamespace(posterior=ds)


def test_growth_association_summary_direction_bands_and_role():
    rng = np.random.default_rng(0)
    # outcome R strongly positive, E ~ null, T negative.
    means = np.array([0.30, 0.00, -0.20])
    gamma = rng.normal(means, 0.08, size=(2, 400, 3))
    delta = rng.normal(0.0, 0.05, size=(2, 400, 3))
    df = growth_association_summary(_growth_trace(gamma, delta), ci_prob=0.89)

    # gamma + delta only (beta / loading absent are skipped, not errored).
    assert set(df["coefficient"]) == {"gamma", "delta"}
    assert (df["role"] == "association").all()
    assert set(df[df["coefficient"] == "gamma"]["outcome"]) == {"R", "E", "T"}

    g = df[df["coefficient"] == "gamma"].set_index("outcome")
    # Direction is read from P(>0): R clearly positive, T clearly negative.
    assert g.loc["R", "prob_positive"] > 0.99
    assert g.loc["R", "favoured_direction"] == "positive"
    assert g.loc["T", "prob_positive"] < 0.02
    assert g.loc["T", "favoured_direction"] == "negative"
    # Fixed 50/89 bands are present and nested around the median (#177, revised
    # 2026-07-17: the 90% sensitivity band was retired).
    r = g.loc["R"]
    assert r["lo89"] <= r["lo50"] <= r["median"]
    assert r["median"] <= r["hi50"] <= r["hi89"]


def test_growth_association_summary_includes_loading_only_when_present():
    rng = np.random.default_rng(1)
    gamma = rng.normal(0.0, 0.1, size=(2, 100, 3))
    delta = rng.normal(0.0, 0.1, size=(2, 100, 3))
    # HalfNormal-like positive loadings.
    loading = np.abs(rng.normal(0.2, 0.05, size=(2, 100, 3)))

    without = growth_association_summary(_growth_trace(gamma, delta))
    assert "loading" not in set(without["coefficient"])

    with_factor = growth_association_summary(_growth_trace(gamma, delta, loading))
    load_rows = with_factor[with_factor["coefficient"] == "loading"]
    assert set(load_rows["outcome"]) == {"R", "E", "T"}
    assert (load_rows["prob_positive"] > 0.99).all()  # positive loadings


# ---------------------------------------------------------------------------
# Factory build (both layers)
# ---------------------------------------------------------------------------


def _n_observed(panel) -> int:
    return int(sum(int(panel.obs_mask[s].sum()) for s in panel.outcomes))


def test_growth_factory_core_builds_and_samples_prior(tmp_path):
    p = _write_growth_csv(tmp_path, n_children=20)
    panel = load_wave_panel(
        path=p, outcomes=_OUTCOMES, baseline_covariates=("blocks",)
    )
    built = build_growth_model(panel, use_shared_factor=False)
    rv = {v.name for v in built.model.free_RVs}
    det = {v.name for v in built.model.deterministics}
    assert {"gamma", "delta", "beta", "alpha", "sigma_slope", "sigma_intercept",
            "kappa"}.issubset(rv)
    assert {"intercept", "slope", "theta"}.issubset(det)
    # Core model has no shared-tempo factor.
    assert "loading" not in rv and "G_tempo" not in rv
    with built.model:
        pp = pm.sample_prior_predictive(draws=5, random_seed=1)
    y = pp.prior_predictive["y_obs"].values
    assert y.shape[-1] == _n_observed(panel)
    assert y.min() >= 0  # bounded counts


def test_growth_factory_shared_factor_adds_tempo_and_loadings(tmp_path):
    p = _write_growth_csv(tmp_path, n_children=18)
    panel = load_wave_panel(
        path=p, outcomes=_OUTCOMES, baseline_covariates=("blocks",)
    )
    built = build_growth_model(panel, use_shared_factor=True)
    rv = {v.name for v in built.model.free_RVs}
    assert {"G_tempo", "loading"}.issubset(rv)


def test_growth_factory_requires_loaded_baseline(tmp_path):
    p = _write_growth_csv(tmp_path, n_children=8)
    panel = load_wave_panel(path=p, outcomes=_OUTCOMES)  # no baseline_covariates
    with pytest.raises(KeyError, match="blocks"):
        build_growth_model(panel, baseline_covariate="blocks")


def test_growth_factory_age_ability_interaction_adds_terms(tmp_path):
    """The #228 item-10 branch adds the baseline-age main effect, its interaction
    with ability, and the standardised baseline-age data container — and only when
    opted in. The extra terms must also compose into a samplable model."""
    p = _write_growth_csv(tmp_path, n_children=20)
    panel = load_wave_panel(
        path=p, outcomes=_OUTCOMES, baseline_covariates=("blocks",)
    )
    built = build_growth_model(panel, age_ability_interaction=True)
    rv = {v.name for v in built.model.free_RVs}
    assert {"gamma_age", "gamma_int"}.issubset(rv)
    assert "age0_std" in set(built.model.named_vars)  # the pm.Data container

    # Opt-in: the default build carries none of the interaction terms.
    base = build_growth_model(panel)
    assert not ({"gamma_age", "gamma_int"} & {v.name for v in base.model.free_RVs})
    assert "age0_std" not in set(base.model.named_vars)

    # The interaction folds into the slope mean without shape errors.
    with built.model:
        pp = pm.sample_prior_predictive(draws=5, random_seed=2)
    y = pp.prior_predictive["y_obs"].values
    assert y.shape[-1] == _n_observed(panel)
    assert y.min() >= 0
