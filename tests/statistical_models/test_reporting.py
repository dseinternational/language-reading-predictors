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
    AssociationTerm,
    ConcurrentTerm,
    association_marginals,
    concurrent_marginals,
    eti_bands,
    evidence_label,
    favoured_direction,
    level_t2_marginal_effect,
    offfloor_mover_table,
    proportion_at_zero_ppc,
    rope_markdown,
    rope_sensitivity,
    rope_sensitivity_markdown,
    rope_summary,
    tau_moderation_summary,
    tau_summary_itt,
    tau_summary_offfloor,
    treatment_marginal_effect,
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


def _ame_median_by_loop(eta, delta, G):
    """Reference AME central estimate: per draw average the per-obs
    expit(eta0+delta)-expit(eta0), then take the posterior **median** over draws
    (the house convention reported by :func:`tau_summary_itt`)."""
    n_draw, n_obs = eta.shape[1], eta.shape[2]
    per_draw = []
    for d in range(n_draw):
        diffs = []
        for i in range(n_obs):
            d_i = delta[0, d, i] if delta.ndim == 3 else delta[0, d]
            eta0 = eta[0, d, i] - d_i * G[i]
            diffs.append(expit(eta0 + d_i) - expit(eta0))
        per_draw.append(np.mean(diffs))
    return float(np.median(per_draw))


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


def test_rope_delta_grid():
    from language_reading_predictors.statistical_models.measures import (
        ROPE_DELTA_PROB_GRID,
        rope_delta_grid,
    )

    # Word reading reports at δ = 1 and δ = 2 (education-lead decision, #144); the
    # grid is [δ, 2·δ] on the items scale.
    assert rope_delta_grid("W") == [1.0, 2.0]
    assert rope_delta_grid("L") == [2.0, 4.0]
    # Floored outcomes have no items grid; they use the risk-difference grid.
    with pytest.raises(KeyError):
        rope_delta_grid("P")
    assert ROPE_DELTA_PROB_GRID == (0.10, 0.15, 0.20)


def test_tau_summary_itt_retains_mean_secondary():
    # The median leads, but the posterior mean is kept as a secondary field (#144).
    rng = np.random.default_rng(3)
    n_obs = 10
    eta = rng.normal(0.0, 1.0, (2, 300, n_obs))
    tau = rng.normal(0.5, 0.3, (2, 300))
    G = (rng.random(n_obs) > 0.5).astype(float)
    out = tau_summary_itt(_trace(eta, tau), ci_prob=0.9, G=G)
    assert out["tau_logit_mean"] == pytest.approx(float(np.mean(tau)))
    # tau_prob_mean is the mean of the per-draw AME (companion to tau_prob_median).
    tau_flat = tau.reshape(-1)
    eta_flat = eta.reshape(-1, n_obs)
    eta0 = eta_flat - tau_flat[:, None] * G[None, :]
    ame = (expit(eta0 + tau_flat[:, None]) - expit(eta0)).mean(axis=1)
    assert out["tau_prob_mean"] == pytest.approx(float(np.mean(ame)))
    # The lead statistic is still the median.
    assert out["tau_logit_median"] == pytest.approx(float(np.median(tau)))


def test_rope_sensitivity_sweeps_deltas():
    rng = np.random.default_rng(1)
    n_chain, n_draw, n_obs = 2, 400, 12
    eta = rng.normal(0.0, 1.0, (n_chain, n_draw, n_obs))
    tau = rng.normal(0.4, 0.2, (n_chain, n_draw))
    G = (rng.random(n_obs) > 0.5).astype(float)
    n_trials = 30
    deltas = [1.0, 2.0, 3.0]

    sens = rope_sensitivity(_trace(eta, tau), G=G, n_trials=n_trials, deltas=deltas)

    # One row per delta, preserving order.
    assert list(sens["delta_items"]) == deltas
    # Reference items AME (same construction as test_rope_summary_matches_reference).
    tau_flat = tau.reshape(-1)
    eta_flat = eta.reshape(-1, n_obs)
    eta0 = eta_flat - tau_flat[:, None] * G[None, :]
    ame = (expit(eta0 + tau_flat[:, None]) - expit(eta0)).mean(axis=1)
    items = ame * n_trials
    for _, r in sens.iterrows():
        d = r["delta_items"]
        assert r["prob_benefit_ge_delta"] == pytest.approx(float(np.mean(items >= d)))
        assert r["prob_in_rope"] == pytest.approx(float(np.mean(np.abs(items) <= d)))
        assert r["prob_harm_ge_delta"] == pytest.approx(float(np.mean(items <= -d)))
        assert r["benefit_label"] == evidence_label(r["prob_benefit_ge_delta"])
    # P(benefit ≥ δ) is non-increasing as δ rises.
    pb = list(sens["prob_benefit_ge_delta"])
    assert pb == sorted(pb, reverse=True)


def test_rope_sensitivity_agrees_with_rope_summary_at_shared_delta():
    # The sweep and the headline card share the AME core, so they agree on
    # P(benefit ≥ δ) at a common δ (issue #144).
    rng = np.random.default_rng(2)
    eta = rng.normal(0.0, 1.0, (2, 300, 10))
    tau = rng.normal(0.3, 0.2, (2, 300))
    G = (rng.random(10) > 0.5).astype(float)
    card = rope_summary(_trace(eta, tau), G=G, n_trials=20, delta=1.0, ci_prob=0.9)
    sens = rope_sensitivity(_trace(eta, tau), G=G, n_trials=20, deltas=[1.0])
    assert sens.iloc[0]["prob_benefit_ge_delta"] == pytest.approx(
        card["prob_benefit_ge_delta"]
    )


def test_rope_sensitivity_markdown_renders():
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "delta_items": 1.0,
                "prob_benefit_ge_delta": 0.90,
                "prob_in_rope": 0.05,
                "prob_harm_ge_delta": 0.01,
                "benefit_label": "moderate",
            },
            {
                "delta_items": 2.0,
                "prob_benefit_ge_delta": 0.40,
                "prob_in_rope": 0.30,
                "prob_harm_ge_delta": 0.05,
                "benefit_label": "inconclusive",
            },
        ]
    )
    md = rope_sensitivity_markdown(df)
    assert "δ-sensitivity" in md
    assert "| 1 |" in md and "| 2 |" in md
    # Risk-difference scale reports δ in percentage points (0.10 -> 10 pp).
    rd = pd.DataFrame(
        [
            {
                "delta_items": 0.10,
                "prob_benefit_ge_delta": 0.7,
                "prob_in_rope": 0.2,
                "prob_harm_ge_delta": 0.02,
                "benefit_label": "suggestive",
            }
        ]
    )
    md_rd = rope_sensitivity_markdown(rd, is_risk_difference=True)
    assert "pp" in md_rd and "| 10 |" in md_rd


def test_tau_summary_itt_constant_tau_average_marginal_effect():
    # 1 chain, 2 draws, 3 observations.
    eta = np.array([[[0.0, 1.0, -0.5], [0.2, -1.0, 0.3]]])
    tau = np.array([[0.4, 0.6]])
    G = np.array([1.0, 0.0, 1.0])

    out = tau_summary_itt(_trace(eta, tau), ci_prob=0.9, G=G)

    assert out["tau_prob_median"] == pytest.approx(_ame_median_by_loop(eta, tau, G))
    assert out["tau_logit_median"] == pytest.approx(float(np.median(tau)))
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
    assert near_floor["tau_prob_median"] != pytest.approx(near_mid["tau_prob_median"])
    # Logit-scale summary is the operating-point-invariant tau itself.
    assert near_floor["tau_logit_median"] == pytest.approx(near_mid["tau_logit_median"])


def test_tau_summary_itt_varying_tau_uses_tau_i():
    eta = np.array([[[0.1, -0.2, 0.4]]])
    tau = np.array([[0.5]])  # headline tau -> logit-scale summary
    tau_i = np.array([[[0.3, 0.7, 0.5]]])  # per-obs effect -> drives the AME
    G = np.array([1.0, 0.0, 1.0])

    out = tau_summary_itt(_trace(eta, tau, tau_i=tau_i), ci_prob=0.9, G=G)

    assert out["tau_prob_median"] == pytest.approx(_ame_median_by_loop(eta, tau_i, G))
    assert out["tau_logit_median"] == pytest.approx(0.5)


def test_tau_summary_itt_rejects_misaligned_G():
    eta = np.array([[[0.0, 1.0, -0.5]]])  # 3 observations
    tau = np.array([[0.4]])
    with pytest.raises(ValueError, match="aligned with the fitted subset"):
        tau_summary_itt(_trace(eta, tau), ci_prob=0.9, G=np.array([1.0, 0.0]))


def test_tau_summary_itt_includes_hpdi_fields():
    # HPDI sensitivity fields accompany the equal-tailed fields on both scales,
    # are finite, and bracket the median (#170).
    rng = np.random.default_rng(0)
    eta = rng.normal(0.0, 1.0, (1, 4000, 5))
    tau = rng.normal(0.3, 0.2, (1, 4000))
    G = (rng.random(5) > 0.5).astype(float)

    out = tau_summary_itt(_trace(eta, tau), ci_prob=0.9, G=G)

    for lo, med, hi in (
        ("tau_logit_hpdi_lo", "tau_logit_median", "tau_logit_hpdi_hi"),
        ("tau_prob_hpdi_lo", "tau_prob_median", "tau_prob_hpdi_hi"),
    ):
        assert lo in out and hi in out
        assert np.isfinite(out[lo]) and np.isfinite(out[hi])
        assert out[lo] <= out[med] <= out[hi]


def test_tau_summary_itt_hpdi_differs_from_eti_on_skewed_posterior():
    # On a right-skewed logit-scale posterior the HPDI is the narrower interval and
    # its lower bound sits below the equal-tailed lower bound (mass piled near the
    # floor) — proving the HPDI fields are not aliases of the equal-tailed fields.
    rng = np.random.default_rng(1)
    tau = rng.gamma(shape=1.5, scale=0.3, size=(1, 40000))  # skewed, > 0
    eta = np.zeros((1, 40000, 3))
    G = np.array([1.0, 0.0, 1.0])

    out = tau_summary_itt(_trace(eta, tau), ci_prob=0.95, G=G)

    eti_width = out["tau_logit_hi"] - out["tau_logit_lo"]
    hpdi_width = out["tau_logit_hpdi_hi"] - out["tau_logit_hpdi_lo"]
    assert hpdi_width < eti_width
    assert out["tau_logit_hpdi_lo"] < out["tau_logit_lo"]
    assert not np.isclose(out["tau_logit_hpdi_lo"], out["tau_logit_lo"])


def test_rope_summary_includes_hpdi_fields():
    # rope_summary / _rope_card carry HPDI bounds for the logit effect and the
    # items scale alongside the equal-tailed fields (#170).
    rng = np.random.default_rng(2)
    eta = rng.normal(0.0, 1.0, (2, 500, 8))
    tau = rng.normal(0.4, 0.2, (2, 500))
    G = (rng.random(8) > 0.5).astype(float)

    out = rope_summary(_trace(eta, tau), G=G, n_trials=30, delta=1.5, ci_prob=0.9)

    for lo, hi in (
        ("tau_logit_hpdi_lo", "tau_logit_hpdi_hi"),
        ("items_hpdi_lo", "items_hpdi_hi"),
    ):
        assert lo in out and hi in out
        assert np.isfinite(out[lo]) and np.isfinite(out[hi])
        assert out[lo] <= out[hi]


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


def _trace_named(eta, **scalars):
    """Trace with ``eta`` (chain, draw, obs) plus named scalar (chain, draw) vars."""
    n_chain, n_draw, n_obs = eta.shape
    data = {"eta": (("chain", "draw", "obs_id"), eta)}
    for name, arr in scalars.items():
        data[name] = (("chain", "draw"), arr)
    ds = xr.Dataset(
        data,
        coords={
            "chain": np.arange(n_chain),
            "draw": np.arange(n_draw),
            "obs_id": np.arange(n_obs),
        },
    )
    return SimpleNamespace(posterior=ds)


def test_treatment_marginal_effect_folds_onto_core_and_reports_median():
    # The gain-family treatment AME is the same counterfactual as the ITT core,
    # parameterised by term=beta_trt / G=trt. Folding onto _itt_ame_draws must
    # reproduce the hand-rolled all-on-vs-all-off AME, and the point estimate is
    # now the MEDIAN (transformation-invariant), per #130.
    eta = np.array([[[0.0, 1.0, -0.5], [0.2, -1.0, 0.3], [0.4, 0.1, -0.2]]])
    beta = np.array([[0.4, 0.6, 0.5]])
    trt = np.array([1.0, 0.0, 1.0])
    n_trials = 20

    out = treatment_marginal_effect(
        _trace_named(eta, beta_trt=beta), trt=trt, n_trials=n_trials, ci_prob=0.9
    )

    b = beta.reshape(-1)  # (S,)
    e = eta.reshape(-1, 3)  # (S, n_obs)
    eta0 = e - b[:, None] * trt[None, :]
    ame = (expit(eta0 + b[:, None]) - expit(eta0)).mean(axis=1)  # (S,)
    assert out["trt_prob_median"] == pytest.approx(float(np.median(ame)))
    assert out["trt_items_median"] == pytest.approx(float(np.median(ame * n_trials)))
    assert out["prob_trt_pos"] == pytest.approx(float(np.mean(b > 0)))
    # Median keys replace the old _mean keys.
    assert "trt_items_mean" not in out and "trt_prob_mean" not in out


def test_treatment_marginal_effect_row_mask_restricts_to_subset():
    # #247 P2: the gain family passes a period-1 row_mask so the treatment marginal is
    # averaged over only the randomised (all-untreated-baseline) transition. The masked
    # AME must equal the hand-rolled AME over just the masked rows, differ from the
    # all-rows AME, and leave the logit-scale prob_trt_pos (a summary of the draws)
    # untouched.
    eta = np.array(
        [[[0.0, 1.0, -0.5, 0.3], [0.2, -1.0, 0.3, -0.4], [0.4, 0.1, -0.2, 0.6]]]
    )
    beta = np.array([[0.4, 0.6, 0.5]])
    trt = np.array([1.0, 0.0, 1.0, 0.0])
    n_trials = 20
    mask = np.array([True, True, False, False])  # the "period-1" rows

    trace = _trace_named(eta, beta_trt=beta)
    out = treatment_marginal_effect(trace, trt=trt, n_trials=n_trials, row_mask=mask)
    out_all = treatment_marginal_effect(trace, trt=trt, n_trials=n_trials)

    b = beta.reshape(-1)  # (S,)
    e = eta.reshape(-1, 4)  # (S, n_obs)
    eta0 = e - b[:, None] * trt[None, :]
    contrib = expit(eta0 + b[:, None]) - expit(eta0)  # (S, n_obs)
    ame_masked = contrib[:, mask].mean(axis=1)
    assert out["trt_prob_median"] == pytest.approx(float(np.median(ame_masked)))
    assert out["trt_items_median"] == pytest.approx(float(np.median(ame_masked * n_trials)))
    # The mask genuinely changes the estimate (guards against a no-op mask)…
    assert abs(out["trt_prob_median"] - out_all["trt_prob_median"]) > 1e-6
    # …but the logit-scale direction probability is unaffected (it summarises beta_trt).
    assert out["prob_trt_pos"] == pytest.approx(out_all["prob_trt_pos"])

    # An integer index array is an accepted alternative form and must agree with the
    # boolean mask selecting the same rows.
    out_idx = treatment_marginal_effect(
        trace, trt=trt, n_trials=n_trials, row_mask=np.array([0, 1])
    )
    assert out_idx["trt_prob_median"] == pytest.approx(out["trt_prob_median"])


def test_treatment_marginal_effect_row_mask_rejects_malformed():
    # #286 review: a 2-D, float, wrong-length, or out-of-range row_mask must fail loudly
    # rather than silently change the indexing semantics of the AME.
    eta = np.array([[[0.0, 1.0, -0.5, 0.3], [0.2, -1.0, 0.3, -0.4]]])
    beta = np.array([[0.4, 0.6]])
    trt = np.array([1.0, 0.0, 1.0, 0.0])
    trace = _trace_named(eta, beta_trt=beta)
    kw = dict(trt=trt, n_trials=20)
    with pytest.raises(ValueError, match="1-D"):
        treatment_marginal_effect(trace, row_mask=np.ones((2, 4), dtype=bool), **kw)
    with pytest.raises(ValueError, match="boolean row_mask has 3 entries"):
        treatment_marginal_effect(trace, row_mask=np.array([True, False, True]), **kw)
    with pytest.raises(ValueError, match="integer row_mask has indices"):
        treatment_marginal_effect(trace, row_mask=np.array([0, 4]), **kw)
    with pytest.raises(ValueError, match="boolean mask or integer"):
        treatment_marginal_effect(trace, row_mask=np.array([0.0, 1.0]), **kw)


def _trace_named_vec(eta, *, scalars=None, vectors=None):
    """Trace with ``eta`` (chain, draw, obs), named scalar and per-obs vector vars."""
    n_chain, n_draw, n_obs = eta.shape
    data = {"eta": (("chain", "draw", "obs_id"), eta)}
    for name, arr in (scalars or {}).items():
        data[name] = (("chain", "draw"), arr)
    for name, arr in (vectors or {}).items():
        data[name] = (("chain", "draw", "obs_id"), arr)
    ds = xr.Dataset(
        data,
        coords={
            "chain": np.arange(n_chain),
            "draw": np.arange(n_draw),
            "obs_id": np.arange(n_obs),
        },
    )
    return SimpleNamespace(posterior=ds)


def test_treatment_marginal_effect_nets_out_interactions():
    # Regression guard for the gain-family AME fix: the treatment contribution is
    # beta_trt + Σ_k gamma_int_trt_k · z_k, so the counterfactual must remove and
    # toggle the FULL per-row effect — not beta_trt alone. Passing the moderators
    # must reproduce the hand-rolled full-effect AME, and must differ from the
    # (wrong) beta_trt-only AME whenever the interaction has non-zero draws.
    eta = np.array([[[0.0, 1.0, -0.5, 0.3], [0.2, -1.0, 0.3, -0.4]]])
    beta = np.array([[0.4, 0.6]])
    gint = np.array([[0.5, -0.3]])  # gamma_int_trt_ability draws
    z_ability = np.array([1.2, -0.8, 0.4, -1.5])  # standardised moderator, per obs
    trt = np.array([1.0, 0.0, 1.0, 0.0])
    n_trials = 20
    trace = _trace_named_vec(
        eta, scalars={"beta_trt": beta, "gamma_int_trt_ability": gint}
    )

    out = treatment_marginal_effect(
        trace, trt=trt, n_trials=n_trials,
        moderators=[("gamma_int_trt_ability", z_ability)],
        ci_prob=0.9,
    )

    b = beta.reshape(-1)  # (S,)
    gi = gint.reshape(-1)  # (S,)
    e = eta.reshape(-1, 4)  # (S, n_obs)
    delta = b[:, None] + gi[:, None] * z_ability[None, :]  # (S, n_obs)
    eta0 = e - delta * trt[None, :]
    ame_full = (expit(eta0 + delta) - expit(eta0)).mean(axis=1)  # (S,)
    assert out["trt_prob_median"] == pytest.approx(float(np.median(ame_full)))
    assert out["trt_items_median"] == pytest.approx(float(np.median(ame_full * n_trials)))

    # The beta_trt-only AME (the pre-fix behaviour) is genuinely different here.
    eta0_wrong = e - b[:, None] * trt[None, :]
    ame_wrong = (expit(eta0_wrong + b[:, None]) - expit(eta0_wrong)).mean(axis=1)
    assert float(np.median(ame_full)) != pytest.approx(float(np.median(ame_wrong)))


def test_rope_summary_accepts_named_treatment_term():
    # rope_summary is reusable for the gain family by naming term=beta_trt; the
    # numbers must match an explicit reference and the default-term path.
    rng = np.random.default_rng(1)
    eta = rng.normal(0.0, 1.0, (2, 300, 8))
    beta = rng.normal(0.4, 0.2, (2, 300))
    G = (rng.random(8) > 0.5).astype(float)
    n_trials, delta = 25, 1.0

    out = rope_summary(
        _trace_named(eta, beta_trt=beta),
        G=G, n_trials=n_trials, delta=delta, ci_prob=0.9,
        term="beta_trt", varying_term="",
    )
    b = beta.reshape(-1)
    e = eta.reshape(-1, 8)
    eta0 = e - b[:, None] * G[None, :]
    items = (expit(eta0 + b[:, None]) - expit(eta0)).mean(axis=1) * n_trials
    assert out["items_median"] == pytest.approx(float(np.median(items)))
    assert out["pd"] == pytest.approx(float(np.mean(b > 0)))
    assert out["prob_benefit_ge_delta"] == pytest.approx(float(np.mean(items >= delta)))
    # Renaming the var to the default name reproduces the same card.
    same = rope_summary(
        _trace_named(eta, tau=beta), G=G, n_trials=n_trials, delta=delta, ci_prob=0.9
    )
    assert out["items_median"] == pytest.approx(same["items_median"])


def test_level_t2_marginal_effect_nets_group_ability_interaction():
    # Issue #271 item 5: the level t2 AME nets out the FULL group contribution
    # (t2 contrast + group×ability interaction) to recover the untreated baseline,
    # but adds back ONLY b_grp_time[t2] — the clean randomised effect at MEAN
    # ability. The time-invariant gamma_grp_ability is deliberately excluded from
    # the causal card. Build a trace with a per-timepoint b_grp_time vector and a
    # scalar gamma_grp_ability, then compare to a loop.
    n_chain, n_draw, n_obs, n_phase = 1, 4, 6, 4
    rng = np.random.default_rng(2)
    eta = rng.normal(0.0, 1.0, (n_chain, n_draw, n_obs))
    b_grp = rng.normal(0.3, 0.2, (n_chain, n_draw, n_phase))
    g_ab = rng.normal(-0.1, 0.1, (n_chain, n_draw))
    phase = np.array([0, 1, 2, 3, 1, 0])  # two rows at t2 (phase == 1)
    G = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    ability = np.array([0.5, -1.0, 0.2, 0.3, 0.8, -0.4])

    ds = xr.Dataset(
        {
            "eta": (("chain", "draw", "obs_id"), eta),
            "b_grp_time": (("chain", "draw", "phase"), b_grp),
            "gamma_grp_ability": (("chain", "draw"), g_ab),
        },
        coords={
            "chain": np.arange(n_chain), "draw": np.arange(n_draw),
            "obs_id": np.arange(n_obs), "phase": np.arange(n_phase),
        },
    )
    contrast, ame = level_t2_marginal_effect(
        SimpleNamespace(posterior=ds), phase=phase, G=G, ability=ability
    )

    b_flat = b_grp.reshape(-1, n_phase)  # (S, phase)
    g_flat = g_ab.reshape(-1)  # (S,)
    e_flat = eta.reshape(-1, n_obs)  # (S, obs)
    rows = np.where(phase == 1)[0]
    ref = []
    for s in range(n_draw):
        diffs = []
        for i in rows:
            d_i = b_flat[s, 1] + g_flat[s] * ability[i]  # full group contribution
            e0 = e_flat[s, i] - d_i * G[i]  # untreated baseline (nets the interaction)
            add_back = b_flat[s, 1]  # only b_grp_time[t2], at mean ability
            diffs.append(expit(e0 + add_back) - expit(e0))
        ref.append(np.mean(diffs))
    assert contrast == pytest.approx(b_flat[:, 1])  # logit contrast = b_grp_time[t2]
    assert ame == pytest.approx(np.array(ref))


def test_level_t2_marginal_effect_requires_t2_rows():
    eta = np.zeros((1, 2, 3))
    ds = xr.Dataset(
        {
            "eta": (("chain", "draw", "obs_id"), eta),
            "b_grp_time": (("chain", "draw", "phase"), np.zeros((1, 2, 4))),
        },
        coords={"chain": [0], "draw": [0, 1], "obs_id": np.arange(3), "phase": np.arange(4)},
    )
    with pytest.raises(ValueError, match="No rows at t2_phase"):
        level_t2_marginal_effect(
            SimpleNamespace(posterior=ds), phase=np.array([0, 2, 3]), G=np.ones(3)
        )


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


def test_eti_bands_nesting_and_quantiles():
    # The fixed 50/90/95 band convention (#177): nested equal-tailed intervals,
    # and the 95% bounds are the 2.5 / 97.5 quantiles.
    rng = np.random.default_rng(4)
    x = rng.normal(0.0, 1.0, 50000)
    b = eti_bands(x)
    assert set(b) == {"lo50", "hi50", "lo90", "hi90", "lo95", "hi95"}
    assert b["lo50"] > b["lo90"] > b["lo95"]
    assert b["hi50"] < b["hi90"] < b["hi95"]
    lo, hi = np.quantile(x, [0.025, 0.975])
    assert b["lo95"] == pytest.approx(float(lo))
    assert b["hi95"] == pytest.approx(float(hi))


def test_tau_summary_itt_exposes_50_90_95_bands():
    # 50% and 90% equal-tailed bands accompany the 95% headline (lo/hi) on both
    # scales, correctly nested; floored models inherit them via delegation (#177).
    rng = np.random.default_rng(5)
    eta = rng.normal(0.0, 1.0, (2, 800, 4))
    tau = rng.normal(0.3, 0.2, (2, 800))
    G = (rng.random(4) > 0.5).astype(float)
    out = tau_summary_itt(_trace(eta, tau), ci_prob=0.95, G=G)
    for scale in ("tau_logit", "tau_prob"):
        for k in ("lo50", "hi50", "lo90", "hi90"):
            assert f"{scale}_{k}" in out
        assert out[f"{scale}_lo50"] > out[f"{scale}_lo90"] > out[f"{scale}_lo"]
        assert out[f"{scale}_hi50"] < out[f"{scale}_hi90"] < out[f"{scale}_hi"]
    assert "tau_prob_lo90" in tau_summary_offfloor(_trace(eta, tau), ci_prob=0.95, G=G)


def test_rope_markdown_labels_central_50_interval():
    # rope_markdown names the 50% band explicitly and drops the ambiguous bare
    # "50% CrI" shorthand (#177).
    import pandas as pd

    rng = np.random.default_rng(6)
    eta = rng.normal(0.0, 1.0, (2, 500, 6))
    tau = rng.normal(0.4, 0.2, (2, 500))
    G = (rng.random(6) > 0.5).astype(float)
    rc = rope_summary(_trace(eta, tau), G=G, n_trials=20, delta=1.0, ci_prob=0.95)
    md = rope_markdown(pd.DataFrame([rc]), "word reading")
    assert "central 50% interval" in md
    assert "equal-tailed 95% credible interval" in md
    assert "50% CrI" not in md


def test_favoured_direction_orients_to_named_claim():
    # Labels the favoured direction, not the raw positive claim (#179).
    neg = favoured_direction(0.02)
    assert neg["favoured_direction"] == "negative"
    assert neg["favoured_direction_prob"] == pytest.approx(0.98)
    assert neg["favoured_direction_label"] != "inconclusive"  # strong evidence of harm
    pos = favoured_direction(0.985)
    assert pos["favoured_direction"] == "positive"
    assert pos["favoured_direction_prob"] == pytest.approx(0.985)
    # A near-50:50 posterior is inconclusive in either orientation.
    mid = favoured_direction(0.52)
    assert mid["favoured_direction"] == "positive"
    assert mid["favoured_direction_label"] == "inconclusive"


def test_tau_summary_itt_labels_harm_for_negative_effect():
    # Regression (#179): a clearly harmful effect must be labelled evidence of
    # harm via the favoured direction, not left "inconclusive" (which is correct
    # only for the benefit claim).
    rng = np.random.default_rng(11)
    eta = rng.normal(0.0, 1.0, (2, 800, 5))
    tau = rng.normal(-0.6, 0.15, (2, 800))  # mostly negative → harmful
    G = (rng.random(5) > 0.5).astype(float)
    out = tau_summary_itt(_trace(eta, tau), ci_prob=0.95, G=G)
    assert out["prob_tau_pos"] < 0.05
    assert out["direction_label"] == "inconclusive"  # for the "helps" claim
    assert out["favoured_direction"] == "negative"
    assert out["favoured_direction_prob"] > 0.95
    assert out["favoured_direction_label"] in {"strong", "very strong"}
    # offfloor inherits the favoured-direction fields via delegation
    assert "favoured_direction" in tau_summary_offfloor(_trace(eta, tau), ci_prob=0.95, G=G)


def test_rope_markdown_harm_wording_for_negative_effect():
    import pandas as pd

    rng = np.random.default_rng(12)
    eta = rng.normal(0.0, 1.0, (2, 600, 6))
    tau = rng.normal(-0.6, 0.15, (2, 600))
    G = (rng.random(6) > 0.5).astype(float)
    rc = rope_summary(_trace(eta, tau), G=G, n_trials=20, delta=1.0, ci_prob=0.95)
    md = rope_markdown(pd.DataFrame([rc]), "word reading")
    assert "is harmful" in md  # favoured-direction claim named
    assert "P(intervention helps)" in md  # probability shown first
    # the harm claim carries a strong label, not "inconclusive" (the magnitude
    # clause may still say inconclusive for the separate benefit-≥-δ claim)
    assert (
        "is harmful — *very strong evidence*" in md
        or "is harmful — *strong evidence*" in md
    )


# ---------------------------------------------------------------------------
# association_marginals — per-covariate items-scale association marginals (#310)
# ---------------------------------------------------------------------------


def _assoc_ame_ref(eta, coef, *, main_scale, dz, interactions=()):
    """Reference per-draw AME for a +Δz covariate perturbation, by explicit loop.

    ``eta`` (chain, draw, obs); ``coef`` (chain, draw); ``interactions`` a list of
    ``(gamma_int (chain, draw), z_partner (obs,))``. Mirrors the maths the helper
    vectorises: Δη_i = coef·(dz·main_scale) + Σ γ_int·z_partner_i·dz."""
    n_draw, n_obs = eta.shape[1], eta.shape[2]
    per_draw = []
    for d in range(n_draw):
        diffs = []
        for i in range(n_obs):
            de = coef[0, d] * dz * main_scale
            for gi, zp in interactions:
                de += gi[0, d] * zp[i] * dz
            diffs.append(expit(eta[0, d, i] + de) - expit(eta[0, d, i]))
        per_draw.append(np.mean(diffs))
    return np.array(per_draw)


def test_association_marginals_sign_and_scale_consistency():
    # Guard (#310): for a covariate with no interaction the items-scale AME must be
    # sign-consistent with the logit coefficient, prob_pos must equal P(coef > 0), and
    # items must be exactly n_trials × prob. Compared to an explicit-loop reference.
    rng = np.random.default_rng(7)
    n_obs, n_trials = 9, 24
    eta = rng.normal(0.0, 1.0, (1, 400, n_obs))
    gamma = rng.normal(0.5, 0.2, (1, 400))  # mostly-positive skill-baseline coefficient
    trace = _trace_named_vec(eta, scalars={"gamma_L": gamma})
    term = AssociationTerm(
        "L", "gamma_L", main_scale=1.5, interactions=(),
        n_items=40, mean_prop=0.4, sd_items=6.0,
    )

    df = association_marginals(trace, terms=[term], n_trials=n_trials, ci_prob=0.9)
    sd_row = df[df.scale == "+1 SD"].iloc[0]

    ref = _assoc_ame_ref(eta, gamma, main_scale=1.5, dz=1.0)
    assert sd_row["prob_median"] == pytest.approx(float(np.median(ref)))
    # Scale: items = n_trials × prob, exactly.
    assert sd_row["items_median"] == pytest.approx(n_trials * sd_row["prob_median"])
    # Sign consistency + prob_pos ties to the logit coefficient (no interaction, so the
    # per-draw AME sign is exactly the coefficient sign).
    assert np.sign(sd_row["items_median"]) == np.sign(float(np.median(gamma)))
    assert sd_row["prob_pos"] == pytest.approx(float(np.mean(gamma.reshape(-1) > 0)))
    assert sd_row["role"] == "association"


def test_association_marginals_per_k_items_row_for_bounded_count():
    # A bounded-count covariate (n_items + mean_prop set) gets a "+k items" companion
    # row evaluated at the mean proportion; its Δz maps the items increment into
    # standardised units and matches the loop reference.
    rng = np.random.default_rng(11)
    n_obs, n_trials, n_items, k, main_scale = 8, 30, 40, 5, 1.5
    eta = rng.normal(0.0, 1.0, (1, 300, n_obs))
    gamma = rng.normal(0.6, 0.2, (1, 300))
    trace = _trace_named_vec(eta, scalars={"gamma_L": gamma})
    p = 0.35
    term = AssociationTerm(
        "L", "gamma_L", main_scale=main_scale, interactions=(),
        n_items=n_items, mean_prop=p,
    )

    df = association_marginals(
        trace, terms=[term], n_trials=n_trials, k_items=k, ci_prob=0.9
    )
    assert set(df.scale) == {"+1 SD", f"+{k} items"}
    k_row = df[df.scale == f"+{k} items"].iloc[0]

    from scipy.special import logit as _logit

    dz = (_logit(p + k / n_items) - _logit(p)) / main_scale
    ref = _assoc_ame_ref(eta, gamma, main_scale=main_scale, dz=dz)
    assert k_row["prob_median"] == pytest.approx(float(np.median(ref)))
    assert k_row["items_median"] == pytest.approx(n_trials * k_row["prob_median"])


def test_association_marginals_nets_interaction_contribution():
    # A covariate that participates in an interaction: the +1 SD AME must include the
    # interaction's per-row contribution (γ_int · z_partner), matching the loop.
    rng = np.random.default_rng(13)
    n_obs, n_trials = 7, 20
    eta = rng.normal(0.0, 1.0, (1, 300, n_obs))
    gamma = rng.normal(0.3, 0.2, (1, 300))          # gamma_ability
    gint = rng.normal(-0.2, 0.15, (1, 300))         # gamma_int_trt_ability
    z_trt = (rng.random(n_obs) > 0.5).astype(float)  # partner = treatment indicator
    trace = _trace_named_vec(
        eta, scalars={"gamma_ability": gamma, "gamma_int_trt_ability": gint}
    )
    term = AssociationTerm(
        "ability", "gamma_ability", main_scale=1.0,
        interactions=(("gamma_int_trt_ability", z_trt),),
    )

    df = association_marginals(trace, terms=[term], n_trials=n_trials, ci_prob=0.9)
    sd_row = df[df.scale == "+1 SD"].iloc[0]

    ref = _assoc_ame_ref(
        eta, gamma, main_scale=1.0, dz=1.0,
        interactions=[(gint, z_trt)],
    )
    assert sd_row["prob_median"] == pytest.approx(float(np.median(ref)))


def test_association_marginals_off_floor_items_equal_prob():
    # Floor-rule path (#310): off-floor outcomes (P, N) pass n_trials=1, so the items
    # scale collapses to the off-floor probability delta — items == prob exactly, and
    # the off_floor flag is recorded. The own baseline is dropped upstream, so the term
    # list here is a skill covariate only (as the pipeline builds for an off-floor fit).
    rng = np.random.default_rng(17)
    eta = rng.normal(-0.3, 1.0, (1, 250, 8))
    gamma = rng.normal(0.4, 0.2, (1, 250))
    trace = _trace_named_vec(eta, scalars={"gamma_L": gamma})
    term = AssociationTerm("L", "gamma_L", main_scale=1.2, interactions=(),
                           n_items=40, mean_prop=0.3)

    df = association_marginals(
        trace, terms=[term], n_trials=1, off_floor=True, ci_prob=0.9
    )
    assert bool(df["off_floor"].iloc[0]) is True
    for _, r in df.iterrows():
        assert r["items_median"] == pytest.approx(r["prob_median"])
        assert r["items_lo"] == pytest.approx(r["prob_lo"])
        assert r["items_hi"] == pytest.approx(r["prob_hi"])


def test_association_marginals_row_mask_restricts_averaging_population():
    # The averaging population is configurable: row_mask=None averages over ALL rows
    # (the association default), a mask restricts it, and a malformed mask fails loudly.
    rng = np.random.default_rng(19)
    eta = rng.normal(0.0, 1.0, (1, 200, 6))
    gamma = rng.normal(0.5, 0.2, (1, 200))
    trace = _trace_named_vec(eta, scalars={"gamma_L": gamma})
    term = AssociationTerm("L", "gamma_L", main_scale=1.0, interactions=())

    mask = np.array([True, True, False, False, False, False])
    masked = association_marginals(trace, terms=[term], n_trials=10, row_mask=mask)
    ref = _assoc_ame_ref(eta[:, :, :2], gamma, main_scale=1.0, dz=1.0)
    assert masked[masked.scale == "+1 SD"].iloc[0]["prob_median"] == pytest.approx(
        float(np.median(ref))
    )
    with pytest.raises(ValueError, match="boolean row_mask has 3 entries"):
        association_marginals(
            trace, terms=[term], n_trials=10, row_mask=np.array([True, False, True])
        )


# ---------------------------------------------------------------------------
# concurrent_marginals — per-predictor items-scale marginals, no interactions (#312)
# ---------------------------------------------------------------------------


def _concurrent_ame_ref(eta, beta, *, dz):
    """Reference per-draw AME for a +Δz standardised perturbation (scalar shift).

    ``eta`` (chain, draw, obs); ``beta`` (chain, draw). The concurrent family has no
    interactions, so Δη is the scalar ``beta·dz`` broadcast over observations."""
    n_draw, n_obs = eta.shape[1], eta.shape[2]
    per_draw = []
    for d in range(n_draw):
        de = beta[0, d] * dz
        diffs = [expit(eta[0, d, i] + de) - expit(eta[0, d, i]) for i in range(n_obs)]
        per_draw.append(float(np.mean(diffs)))
    return np.asarray(per_draw)


def test_concurrent_marginals_sign_and_scale_consistency():
    # +1 SD row: items == n_trials × prob exactly, prob_pos ties to P(beta > 0), and
    # the AME is sign-consistent with the (interaction-free) coefficient.
    rng = np.random.default_rng(23)
    n_obs, n_trials = 9, 79
    eta = rng.normal(0.0, 1.0, (1, 400, n_obs))
    beta = rng.normal(0.5, 0.2, (1, 400))
    trace = _trace_named_vec(eta, scalars={"beta_L": beta})
    term = ConcurrentTerm("L", "beta_L", sd_logit=1.3, n_items=32, mean_prop=0.4, k_items=3)

    df = concurrent_marginals(trace, terms=[term], n_trials=n_trials, ci_prob=0.9)
    sd_row = df[df.scale == "+1 SD"].iloc[0]

    ref = _concurrent_ame_ref(eta, beta, dz=1.0)
    assert sd_row["prob_median"] == pytest.approx(float(np.median(ref)))
    assert sd_row["items_median"] == pytest.approx(n_trials * sd_row["prob_median"])
    assert np.sign(sd_row["items_median"]) == np.sign(float(np.median(beta)))
    assert sd_row["prob_pos"] == pytest.approx(float(np.mean(beta.reshape(-1) > 0)))
    assert sd_row["role"] == "association"


def test_concurrent_marginals_k_items_row_uses_sd_logit():
    # A bounded-count predictor gets a "+k items" row whose Δz divides the mean-point
    # logit shift by sd_logit (the standardisation the factory applied) — matching the
    # explicit-loop reference. This is the raw-logit/standardised bridge the pushforward
    # depends on; a wrong sd_logit here would silently mis-scale every items AME.
    from scipy.special import logit as _logit

    rng = np.random.default_rng(29)
    n_obs, n_trials, n_items, k, sd_logit, p = 8, 79, 32, 3, 1.4, 0.35
    eta = rng.normal(0.0, 1.0, (1, 300, n_obs))
    beta = rng.normal(0.6, 0.2, (1, 300))
    trace = _trace_named_vec(eta, scalars={"beta_L": beta})
    term = ConcurrentTerm(
        "L", "beta_L", sd_logit=sd_logit, n_items=n_items, mean_prop=p, k_items=k
    )

    df = concurrent_marginals(trace, terms=[term], n_trials=n_trials, ci_prob=0.9)
    assert set(df.scale) == {"+1 SD", f"+{k} items"}
    k_row = df[df.scale == f"+{k} items"].iloc[0]

    dz = (_logit(p + k / n_items) - _logit(p)) / sd_logit
    ref = _concurrent_ame_ref(eta, beta, dz=dz)
    assert k_row["prob_median"] == pytest.approx(float(np.median(ref)))
    assert k_row["items_median"] == pytest.approx(n_trials * k_row["prob_median"])


def test_concurrent_marginals_no_k_row_without_items_metadata():
    # A predictor with no bounded-count metadata (e.g. age) gets only the +1 SD row.
    rng = np.random.default_rng(31)
    eta = rng.normal(0.0, 1.0, (1, 200, 6))
    beta = rng.normal(0.2, 0.2, (1, 200))
    trace = _trace_named_vec(eta, scalars={"beta_age": beta})
    term = ConcurrentTerm("age", "beta_age", sd_logit=1.0)
    df = concurrent_marginals(trace, terms=[term], n_trials=79)
    assert list(df.scale) == ["+1 SD"]


def test_concurrent_marginals_k_row_caps_at_ceiling():
    # Near the ceiling a fixed +k that would push p past 1 is CAPPED to the largest
    # whole increment that still fits below the ceiling, and the row is relabelled to
    # the effective k (not silently clamped to a smaller-than-labelled shift).
    rng = np.random.default_rng(37)
    eta = rng.normal(0.0, 1.0, (1, 200, 8))
    beta = rng.normal(0.4, 0.2, (1, 200))
    trace = _trace_named_vec(eta, scalars={"beta_B": beta})
    # mean 0.88 on a 10-item scale: +5 → 1.38 (over ceiling); only +1 (→0.98) fits.
    term = ConcurrentTerm("B", "beta_B", sd_logit=1.2, n_items=10, mean_prop=0.88, k_items=5)
    df = concurrent_marginals(trace, terms=[term], n_trials=79)
    assert set(df.scale) == {"+1 SD", "+1 items"}  # capped from +5 to +1


def test_concurrent_marginals_k_row_skipped_at_full_ceiling():
    # If even +1 item exceeds the ceiling, no positive increment is representable, so
    # the +k row is skipped entirely (only the +1 SD row remains).
    rng = np.random.default_rng(41)
    eta = rng.normal(0.0, 1.0, (1, 150, 6))
    beta = rng.normal(0.3, 0.2, (1, 150))
    trace = _trace_named_vec(eta, scalars={"beta_B": beta})
    term = ConcurrentTerm("B", "beta_B", sd_logit=1.0, n_items=10, mean_prop=0.97, k_items=2)
    df = concurrent_marginals(trace, terms=[term], n_trials=79)
    assert list(df.scale) == ["+1 SD"]
