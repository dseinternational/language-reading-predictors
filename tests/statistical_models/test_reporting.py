# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for :mod:`statistical_models.reporting`."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from scipy.special import expit

from language_reading_predictors.statistical_models.reporting import (
    evidence_label,
    offfloor_mover_table,
    proportion_at_zero_ppc,
    rope_summary,
    tau_moderation_summary,
    tau_summary_itt,
    tau_summary_offfloor,
)


def _trace(eta, tau, tau_i=None):
    """Wrap synthetic posterior arrays as an object exposing ``.posterior``.

    ``eta`` has shape (chain, draw, obs); ``tau`` (chain, draw); optional
    ``tau_i`` (chain, draw, obs). Using a plain ``Dataset`` keeps the test
    independent of the installed ArviZ version's ``from_dict`` behaviour.
    """
    n_chain, n_draw, n_obs = eta.shape
    data = {
        "eta": (("chain", "draw", "obs_id"), eta),
        "tau": (("chain", "draw"), tau),
    }
    if tau_i is not None:
        data["tau_i"] = (("chain", "draw", "obs_id"), tau_i)
    ds = xr.Dataset(
        data,
        coords={
            "chain": np.arange(n_chain),
            "draw": np.arange(n_draw),
            "obs_id": np.arange(n_obs),
        },
    )
    return SimpleNamespace(posterior=ds)


def _ame_by_loop(eta, delta, G):
    """Reference AME: per draw, average over obs of expit(eta0+delta)-expit(eta0)."""
    n_draw, n_obs = eta.shape[1], eta.shape[2]
    per_draw = []
    for d in range(n_draw):
        diffs = []
        for i in range(n_obs):
            d_i = delta[0, d, i] if delta.ndim == 3 else delta[0, d]
            eta0 = eta[0, d, i] - d_i * G[i]
            diffs.append(expit(eta0 + d_i) - expit(eta0))
        per_draw.append(np.mean(diffs))
    return float(np.mean(per_draw))


def test_evidence_label_round_odds_boundaries():
    # Boundaries are inclusive-below the next tier: [.75,.91)=suggestive, etc.
    assert evidence_label(0.60) == "inconclusive"
    assert evidence_label(0.75) == "suggestive"
    assert evidence_label(0.90) == "suggestive"
    assert evidence_label(0.91) == "moderate"
    assert evidence_label(0.96) == "moderate"
    assert evidence_label(0.97) == "strong"
    assert evidence_label(0.985) == "strong"
    assert evidence_label(0.99) == "very strong"
    assert evidence_label(0.999) == "very strong"


def test_rope_delta_registry():
    from language_reading_predictors.statistical_models.measures import (
        ROPE_DELTA,
        ROPE_DELTA_PROB,
        rope_delta,
    )

    assert rope_delta("L") == 2.0
    assert rope_delta("W") == 1.0
    assert set(ROPE_DELTA_PROB) == {"P", "N"}
    # Floored / not-yet-agreed outcomes have no items delta.
    for missing in ("P", "N", "F", "T"):
        assert missing not in ROPE_DELTA
        with pytest.raises(KeyError):
            rope_delta(missing)


def test_rope_summary_matches_reference():
    rng = np.random.default_rng(0)
    n_chain, n_draw, n_obs = 2, 400, 12
    eta = rng.normal(0.0, 1.0, (n_chain, n_draw, n_obs))
    tau = rng.normal(0.4, 0.2, (n_chain, n_draw))
    G = (rng.random(n_obs) > 0.5).astype(float)
    n_trials, delta = 30, 1.5

    out = rope_summary(_trace(eta, tau), G=G, n_trials=n_trials, delta=delta, ci_prob=0.9)

    # Reference: per-draw items average marginal effect (all-on vs all-off).
    tau_flat = tau.reshape(-1)  # (S,)
    eta_flat = eta.reshape(-1, n_obs)  # (S, n_obs), same sample order as the stack
    eta0 = eta_flat - tau_flat[:, None] * G[None, :]
    ame = (expit(eta0 + tau_flat[:, None]) - expit(eta0)).mean(axis=1)  # (S,)
    items = ame * n_trials

    assert out["items_median"] == pytest.approx(float(np.median(items)))
    assert out["pd"] == pytest.approx(float(np.mean(tau_flat > 0)))
    assert out["prob_benefit_ge_delta"] == pytest.approx(float(np.mean(items >= delta)))
    assert out["prob_in_rope"] == pytest.approx(float(np.mean(np.abs(items) <= delta)))
    assert out["prob_harm_ge_delta"] == pytest.approx(float(np.mean(items <= -delta)))
    assert out["delta_items"] == delta
    # Nested intervals are ordered.
    assert (
        out["tau_logit_lo"]
        <= out["tau_logit_lo50"]
        <= out["tau_logit_median"]
        <= out["tau_logit_hi50"]
        <= out["tau_logit_hi"]
    )
    assert out["direction_label"] == evidence_label(out["pd"])
    assert out["benefit_label"] == evidence_label(out["prob_benefit_ge_delta"])


def test_tau_summary_itt_constant_tau_average_marginal_effect():
    # 1 chain, 2 draws, 3 observations.
    eta = np.array([[[0.0, 1.0, -0.5], [0.2, -1.0, 0.3]]])
    tau = np.array([[0.4, 0.6]])
    G = np.array([1.0, 0.0, 1.0])

    out = tau_summary_itt(_trace(eta, tau), ci_prob=0.9, G=G)

    assert out["tau_prob_mean"] == pytest.approx(_ame_by_loop(eta, tau, G))
    assert out["tau_logit_mean"] == pytest.approx(float(np.mean(tau)))
    assert out["prob_tau_pos"] == pytest.approx(1.0)  # both tau draws > 0


def test_tau_summary_itt_operating_point_comes_from_full_eta():
    # Two constant etas (stand-ins for the cross-baseline / adjuster / GP terms
    # the old scalar baseline ignored) give different probability-scale effects
    # for the *same* tau, because expit is non-linear — confirming eta drives
    # the operating point rather than a single own-baseline mean.
    tau = np.array([[0.5]])
    G = np.array([1.0, 1.0])
    near_floor = tau_summary_itt(_trace(np.array([[[-2.0, -2.0]]]), tau), ci_prob=0.9, G=G)
    near_mid = tau_summary_itt(_trace(np.array([[[0.0, 0.0]]]), tau), ci_prob=0.9, G=G)
    assert near_floor["tau_prob_mean"] != pytest.approx(near_mid["tau_prob_mean"])
    # Logit-scale summary is the operating-point-invariant tau itself.
    assert near_floor["tau_logit_mean"] == pytest.approx(near_mid["tau_logit_mean"])


def test_tau_summary_itt_varying_tau_uses_tau_i():
    eta = np.array([[[0.1, -0.2, 0.4]]])
    tau = np.array([[0.5]])  # headline tau -> logit-scale summary
    tau_i = np.array([[[0.3, 0.7, 0.5]]])  # per-obs effect -> drives the AME
    G = np.array([1.0, 0.0, 1.0])

    out = tau_summary_itt(_trace(eta, tau, tau_i=tau_i), ci_prob=0.9, G=G)

    assert out["tau_prob_mean"] == pytest.approx(_ame_by_loop(eta, tau_i, G))
    assert out["tau_logit_mean"] == pytest.approx(0.5)


def test_tau_summary_itt_rejects_misaligned_G():
    eta = np.array([[[0.0, 1.0, -0.5]]])  # 3 observations
    tau = np.array([[0.4]])
    with pytest.raises(ValueError, match="aligned with the fitted subset"):
        tau_summary_itt(_trace(eta, tau), ci_prob=0.9, G=np.array([1.0, 0.0]))


def _posterior(**arrays):
    """Wrap (chain, draw) arrays as an object exposing ``.posterior``."""
    data = {k: (("chain", "draw"), v) for k, v in arrays.items()}
    any_v = next(iter(arrays.values()))
    ds = xr.Dataset(
        data,
        coords={
            "chain": np.arange(any_v.shape[0]),
            "draw": np.arange(any_v.shape[1]),
        },
    )
    return SimpleNamespace(posterior=ds)


def test_tau_summary_offfloor_delegates_to_tau_summary_itt():
    # The floor-rule PRIMARY reuses the marginal-effect machinery verbatim, so
    # expit(eta) reads as Pr(off floor) and the prob scale is the risk difference.
    eta = np.array([[[0.0, 1.0, -0.5], [0.2, -1.0, 0.3]]])
    tau = np.array([[0.4, 0.6]])
    G = np.array([1.0, 0.0, 1.0])
    trace = _trace(eta, tau)
    assert tau_summary_offfloor(trace, ci_prob=0.9, G=G) == tau_summary_itt(
        trace, ci_prob=0.9, G=G
    )


def test_offfloor_mover_table_arm_coding_and_counts():
    # Positive-benefit coding: G == 1 intervention, G == 0 control.
    # post: intervention {3, 0, NaN}, control {0, 5}.
    prepared = SimpleNamespace(
        post_counts={"P": np.array([3.0, 0.0, np.nan, 0.0, 5.0])},
        G=np.array([1, 1, 1, 0, 0]),
    )
    df = offfloor_mover_table(prepared, "P").set_index("arm")
    assert list(df.index) == ["intervention", "control"]
    assert df.loc["intervention", "n"] == 2  # NaN post excluded
    assert df.loc["intervention", "off_floor"] == 1
    assert df.loc["intervention", "at_floor"] == 1
    assert df.loc["intervention", "prop_off_floor"] == pytest.approx(0.5)
    assert df.loc["control", "n"] == 2
    assert df.loc["control", "off_floor"] == 1


def test_offfloor_mover_table_empty_arm_is_nan():
    prepared = SimpleNamespace(
        post_counts={"P": np.array([1.0, 0.0])}, G=np.array([1, 1])
    )
    df = offfloor_mover_table(prepared, "P").set_index("arm")
    assert df.loc["control", "n"] == 0
    assert np.isnan(df.loc["control", "prop_off_floor"])


def test_tau_moderation_summary_reports_present_coeffs():
    rng = np.random.default_rng(0)
    gint = rng.normal(0.3, 0.1, size=(1, 500))
    gmod = rng.normal(-0.2, 0.1, size=(1, 500))
    out = tau_moderation_summary(
        _posterior(gamma_tau_int=gint, gamma_tau_mod=gmod), ci_prob=0.9
    )
    assert out["gamma_tau_int_mean"] == pytest.approx(float(np.mean(gint)))
    assert out["gamma_tau_int_lo"] < out["gamma_tau_int_mean"] < out["gamma_tau_int_hi"]
    assert out["prob_gamma_tau_int_pos"] == pytest.approx(float(np.mean(gint > 0)))
    assert out["prob_gamma_tau_mod_pos"] == pytest.approx(float(np.mean(gmod > 0)))


def test_tau_moderation_summary_skips_absent_coeffs():
    out = tau_moderation_summary(
        _posterior(gamma_tau_mod=np.array([[0.1, 0.2, 0.3]])), ci_prob=0.9
    )
    assert "gamma_tau_mod_mean" in out
    assert not any(k.startswith("gamma_tau_int") for k in out)


def test_proportion_at_zero_ppc():
    prepared = SimpleNamespace(post_counts={"N": np.array([0.0, 0.0, 1.0, 3.0])})
    # Replicated y_post (chain, draw, obs): zero-fractions 3/4 and 1/4.
    yrep = np.array([[[0, 0, 0, 1], [0, 1, 2, 3]]], dtype=float)
    pp = xr.Dataset(
        {"y_post": (("chain", "draw", "obs_id"), yrep)},
        coords={"chain": [0], "draw": [0, 1], "obs_id": np.arange(4)},
    )
    out = proportion_at_zero_ppc(prepared, "N", SimpleNamespace(posterior_predictive=pp))
    assert out["obs_prop_at_zero"] == pytest.approx(0.5)  # 2 of 4 are zero
    assert out["ppc_mean_prop_at_zero"] == pytest.approx(0.5)  # mean(0.75, 0.25)
    assert out["ppc_p_value"] == pytest.approx(0.5)  # P(rep >= 0.5)
    assert out["rep"].shape == (2,)
