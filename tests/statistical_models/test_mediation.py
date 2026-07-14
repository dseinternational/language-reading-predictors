# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for the g-formula mediation decomposition (``mediation.decompose``).

These build the joint mediator + outcome model on synthetic data to obtain a
``MediationData`` aligned to the model, then run :func:`decompose` against a
small *synthetic* posterior (no MCMC sampling) — fast and deterministic. The
focus is the issue #85 guarantee: the g-formula adjusts for exactly the
confounder set the model was fitted with, sourced from ``MediationData`` rather
than a hardcoded ``{E, R}``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import xarray as xr

from language_reading_predictors.statistical_models.factories import (
    build_mediation_model,
)
from language_reading_predictors.statistical_models.mediation import (
    decompose,
    sensitivity_sweep,
)
from language_reading_predictors.statistical_models.preprocessing import (
    load_and_prepare,
)

from .test_factories import _write_synthetic

_QUANTITIES = {"total", "NDE", "NIE", "proportion_mediated"}
# Outcome- and mediator-leg coefficients the decomposition reads, minus the
# per-confounder ``b_<s>`` / ``a_<s>`` terms (added per test to match the
# fitted confounder set).
_OUTCOME_DRAWS = ["b0", "b_G", "b_M", "b_GM", "b_W", "b_A"]
_BB_MEDIATOR_DRAWS = ["a0", "a_G", "a_L", "a_A"]
_GAUSSIAN_MEDIATOR_DRAWS = ["a0", "a_G", "a_comp", "a_A"]


def _prepare(tmp_path, n_children: int = 15):
    p = _write_synthetic(tmp_path, n_children=n_children)
    return load_and_prepare(path=p, phase_mode="itt")


def _fake_trace(
    names, *, positive=(), values=None, chains: int = 2, draws: int = 20, seed: int = 0
):
    """A minimal posterior stand-in: an xarray ``Dataset`` of (chain, draw)
    coefficient draws exposed as ``.posterior`` — exactly what ``decompose``
    consumes (``trace.posterior[name].stack(...)``).

    ``values`` pins a coefficient to a near-constant value (tiny jitter for a
    non-degenerate posterior), so a deterministic scenario can be constructed.
    """
    rng = np.random.default_rng(seed)
    values = values or {}
    data = {}
    for name in names:
        if name in values:
            arr = float(values[name]) + rng.normal(0.0, 1e-3, size=(chains, draws))
        else:
            arr = rng.normal(0.0, 0.2, size=(chains, draws))
            if name in positive:
                arr = np.abs(arr) + 5.0  # kappa / sigma must be positive
        data[name] = (("chain", "draw"), arr)
    return SimpleNamespace(posterior=xr.Dataset(data))


def test_decompose_beta_binomial(tmp_path):
    """LRP59: single Beta-Binomial mediator, confounders {E, R}."""
    prep = _prepare(tmp_path)
    _, med = build_mediation_model(prep, confounder_symbols=("E", "R"))
    names = _OUTCOME_DRAWS + ["b_E", "b_R"] + _BB_MEDIATOR_DRAWS + ["a_E", "a_R", "kappa_M"]
    df = decompose(_fake_trace(names, positive=["kappa_M"]), med, n_replicates=4)
    assert set(df["quantity"]) == _QUANTITIES
    effects = df[df["quantity"].isin(["total", "NDE", "NIE"])]
    assert effects[["prob_mean", "words_mean"]].notna().all().all()


def test_decompose_offfloor_outcome(tmp_path):
    """#228 item 12 (LRP86, nonword N): an off-floor (Bernoulli) OUTCOME. The outcome
    leg drops the own-baseline b_W, so decompose must NOT read it, and it reports
    NIE/NDE on the off-floor risk-difference scale (n_trials_W = 1, so words_* ==
    prob_*)."""
    prep = _prepare(tmp_path)
    _, med = build_mediation_model(
        prep, confounder_symbols=("E", "R"), outcome_kind="bernoulli_offfloor"
    )
    assert med.off_floor is True
    assert med.n_trials_W == 1
    # Fake trace WITHOUT b_W (never created on the off-floor leg) and WITHOUT kappa_Y
    # (the Bernoulli has no dispersion) — so if decompose read b_W it would KeyError.
    names = (
        ["b0", "b_G", "b_M", "b_GM", "b_A", "b_E", "b_R"]
        + _BB_MEDIATOR_DRAWS
        + ["a_E", "a_R", "kappa_M"]
    )
    df = decompose(_fake_trace(names, positive=["kappa_M"]), med, n_replicates=4)
    assert set(df["quantity"]) == _QUANTITIES
    eff = df[df["quantity"].isin(["total", "NDE", "NIE"])]
    assert eff[["prob_mean", "words_mean"]].notna().all().all()
    # n_trials_W = 1 => the items ("words") columns collapse onto the risk difference.
    assert np.allclose(eff["prob_mean"].to_numpy(), eff["words_mean"].to_numpy())
    # Every row carries the off_floor flag so the report labels the scale.
    assert bool(df["off_floor"].iloc[0]) is True


def test_decompose_gaussian_composite(tmp_path):
    """LRP62: Gaussian route-composite mediator, confounders {E, R}."""
    prep = _prepare(tmp_path)
    _, med = build_mediation_model(
        prep,
        mediator_kind="gaussian_composite",
        route_symbols=("L", "B"),
        confounder_symbols=("E", "R"),
    )
    names = (
        _OUTCOME_DRAWS + ["b_E", "b_R"] + _GAUSSIAN_MEDIATOR_DRAWS + ["a_E", "a_R", "sigma_M"]
    )
    df = decompose(_fake_trace(names, positive=["sigma_M"]), med, n_replicates=4)
    assert set(df["quantity"]) == _QUANTITIES


def test_sensitivity_sweep(tmp_path):
    """#230: the NIE sensitivity sweep returns a delta-indexed NIE curve plus a
    summary whose flags are mutually exclusive (already-null / robust / tipping),
    reusing the g-formula via the ``b_m_shift`` lever."""
    prep = _prepare(tmp_path)
    _, med = build_mediation_model(prep, confounder_symbols=("E", "R"))
    names = _OUTCOME_DRAWS + ["b_E", "b_R"] + _BB_MEDIATOR_DRAWS + ["a_E", "a_R", "kappa_M"]
    trace = _fake_trace(names, positive=["kappa_M"], seed=1)
    sweep, summary = sensitivity_sweep(trace, med, n_deltas=6, n_replicates=4)
    assert sweep["delta"].iloc[0] == 0.0
    assert {"nie_median", "nie_lo", "nie_hi", "delta_frac_of_bM"} <= set(sweep.columns)
    assert {
        "tipping_delta", "tipping_frac_of_bM", "already_null_at_zero",
        "robust_over_full_sweep", "b_M_effective_mean",
    } <= set(summary)
    # Exactly one state holds: already-null at 0, robust over the whole sweep, or a
    # finite tipping point in between.
    finite_tip = not summary["already_null_at_zero"] and not summary[
        "robust_over_full_sweep"
    ]
    assert (
        int(summary["already_null_at_zero"])
        + int(summary["robust_over_full_sweep"])
        + int(finite_tip)
    ) == 1
    # Tie the flags to the value they summarise: a finite tipping_delta iff finite_tip.
    assert np.isfinite(summary["tipping_delta"]) == finite_tip


def test_sensitivity_sweep_attenuates_toward_null_for_both_signs(tmp_path):
    """#289 review: the sweep must shrink the effective slope toward 0 regardless of its
    sign. A fixed positive subtraction pushed a *negative* fitted slope further from 0 and
    silently reported ``robust_over_full_sweep=True``. Here treatment raises the mediator
    (a_G>0) and |b_M| is large, so the NIE is clearly non-null at delta=0 and must reach a
    finite tipping point (not "robust") for b_M>0 AND b_M<0."""
    prep = _prepare(tmp_path)
    _, med = build_mediation_model(prep, confounder_symbols=("E", "R"))
    names = _OUTCOME_DRAWS + ["b_E", "b_R"] + _BB_MEDIATOR_DRAWS + ["a_E", "a_R", "kappa_M"]
    for b_m in (3.0, -3.0):
        vals = {
            "b0": 0.0, "b_G": 0.0, "b_M": b_m, "b_GM": 0.0, "b_W": 0.0, "b_A": 0.0,
            "b_E": 0.0, "b_R": 0.0,
            "a0": 0.0, "a_G": 1.5, "a_L": 0.0, "a_A": 0.0, "a_E": 0.0, "a_R": 0.0,
        }
        trace = _fake_trace(names, positive=["kappa_M"], values=vals, seed=3)
        _, summary = sensitivity_sweep(trace, med, n_deltas=21, n_replicates=8)
        assert not summary["already_null_at_zero"], f"b_M={b_m}: NIE should be non-null at 0"
        assert not summary["robust_over_full_sweep"], (
            f"b_M={b_m}: sweep must attenuate toward 0 and find a tipping point, not 'robust'"
        )
        assert np.isfinite(summary["tipping_delta"])
        assert 0.0 < summary["tipping_delta"]
        # Fraction of the fitted slope is a positive magnitude whatever the slope's sign.
        assert summary["tipping_frac_of_bM"] > 0.0
        assert np.sign(summary["b_M_effective_mean"]) == np.sign(b_m)


def test_decompose_follows_fitted_confounder_set(tmp_path):
    """The core issue #85 guarantee: ``decompose`` adjusts for exactly the
    confounders carried on ``MediationData``. Fit with a single, *non-default*
    confounder (R only) and supply a posterior that contains only the R
    coefficients — under the old hardcoded ``{E, R}`` this would ``KeyError`` on
    the missing E coefficients / ``conf_logit['E']``."""
    prep = _prepare(tmp_path)
    _, med = build_mediation_model(prep, confounder_symbols=("R",))
    assert med.confounder_symbols == ("R",)
    assert set(med.conf_logit) == {"R"}
    names = _OUTCOME_DRAWS + ["b_R"] + _BB_MEDIATOR_DRAWS + ["a_R", "kappa_M"]
    df = decompose(_fake_trace(names, positive=["kappa_M"]), med, n_replicates=4)
    assert set(df["quantity"]) == _QUANTITIES
