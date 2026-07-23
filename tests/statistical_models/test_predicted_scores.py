# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for :mod:`statistical_models.predicted_scores` (#316).

The most important test here is the drift guard: the module's counterfactual
arithmetic deliberately mirrors ``reporting._itt_ame_draws``, and the guard
asserts the two produce *identical* average-marginal-effect draws (and hence
that ``predicted_scores.csv`` agrees with ``treatment_marginal.csv`` /
``rope_summary.csv``) on traces exercising the varying term, treatment
interactions and a row mask.
"""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import xarray as xr
from scipy.special import expit

from language_reading_predictors.statistical_models.predicted_scores import (
    counterfactual_predictive_contrast,
    icon_array_counts,
    predicted_scores_table,
    write_predicted_scores_artifacts,
)
from language_reading_predictors.statistical_models.reporting import (
    _itt_ame_draws,
    treatment_marginal_effect,
)


def _trace(eta, tau, *, tau_i=None, kappa=None, extra=None):
    """Synthetic posterior as ``SimpleNamespace(posterior=Dataset)``.

    Follows the ``test_reporting.py`` convention: ``eta`` has shape
    ``(chain, draw, obs)``; scalars ``(chain, draw)``; ``extra`` maps names to
    ``(dims, values)`` for moderator coefficients and random-intercept nodes.
    """
    n_chain, n_draw, n_obs = eta.shape
    data = {
        "eta": (("chain", "draw", "obs_id"), eta),
        "tau": (("chain", "draw"), tau),
    }
    if tau_i is not None:
        data["tau_i"] = (("chain", "draw", "obs_id"), tau_i)
    if kappa is not None:
        data["kappa"] = (("chain", "draw"), kappa)
    for name, (dims, values) in (extra or {}).items():
        data[name] = (dims, values)
    coords = {
        "chain": np.arange(n_chain),
        "draw": np.arange(n_draw),
        "obs_id": np.arange(n_obs),
    }
    for name, (dims, values) in (extra or {}).items():
        if "child" in dims:
            coords["child"] = np.arange(np.asarray(values).shape[-1])
    ds = xr.Dataset(data, coords=coords)
    return SimpleNamespace(posterior=ds)


def _rng_trace(n_chain=2, n_draw=40, n_obs=12, *, seed=7, tau_i=False, kappa=True):
    rng = np.random.default_rng(seed)
    eta = rng.normal(0.0, 1.0, size=(n_chain, n_draw, n_obs))
    tau = rng.normal(0.4, 0.2, size=(n_chain, n_draw))
    ti = (
        tau[:, :, None] + rng.normal(0.0, 0.1, size=(n_chain, n_draw, n_obs))
        if tau_i
        else None
    )
    kp = rng.uniform(20.0, 60.0, size=(n_chain, n_draw)) if kappa else None
    return eta, tau, ti, kp


# ---------------------------------------------------------------------------
# Drift guard: the module's AME must equal reporting._itt_ame_draws exactly.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_tau_i", [False, True])
@pytest.mark.parametrize("use_mask", [False, True])
def test_ame_matches_itt_ame_draws(use_tau_i, use_mask):
    eta, tau, tau_i, kappa = _rng_trace(tau_i=use_tau_i)
    n_obs = eta.shape[2]
    rng = np.random.default_rng(11)
    mod_vec = rng.normal(0.0, 1.0, size=n_obs)
    gamma = rng.normal(0.2, 0.05, size=eta.shape[:2])
    trace = _trace(
        eta,
        tau,
        tau_i=tau_i,
        kappa=kappa,
        extra={"gamma_int": (("chain", "draw"), gamma)},
    )
    G = (np.arange(n_obs) % 2).astype(float)
    row_mask = (np.arange(n_obs) < 8) if use_mask else None
    moderators = [("gamma_int", mod_vec)]

    _, expected_ame = _itt_ame_draws(
        trace, G=G, moderators=moderators, row_mask=row_mask
    )
    contrast = counterfactual_predictive_contrast(
        trace,
        G=G,
        n_trials=32,
        term="tau",
        moderators=moderators,
        row_mask=row_mask,
        likelihood="bernoulli",
    )
    np.testing.assert_allclose(contrast.ame_prob, expected_ame, rtol=0, atol=1e-12)


def test_summary_median_matches_treatment_marginal():
    """predicted_scores.csv's AME row must agree with treatment_marginal.csv."""
    eta, tau, _, kappa = _rng_trace(seed=21)
    n_obs = eta.shape[2]
    trace = _trace(eta, tau, kappa=kappa)
    # Reuse the gain-family shape: an on-intervention indicator and a row mask.
    trt = (np.arange(n_obs) % 2).astype(float)
    row_mask = np.arange(n_obs) < 9
    n_trials = 54

    tme = treatment_marginal_effect(
        trace, trt=trt, n_trials=n_trials, term="tau", ci_prob=0.95, row_mask=row_mask
    )
    contrast = counterfactual_predictive_contrast(
        trace,
        G=trt,
        n_trials=n_trials,
        term="tau",
        varying_term="",
        row_mask=row_mask,
        likelihood="bernoulli",
    )
    table = predicted_scores_table(
        contrast,
        outcome_symbol="W",
        ci_prob=0.95,
        population="test",
        contrast_status="test",
    )
    ame = table[table.quantity == "average_marginal_effect"].iloc[0]
    assert ame["median"] == pytest.approx(tme["trt_prob_median"], abs=1e-12)
    assert ame["lo"] == pytest.approx(tme["trt_prob_lo"], abs=1e-12)
    assert ame["hi"] == pytest.approx(tme["trt_prob_hi"], abs=1e-12)


# ---------------------------------------------------------------------------
# Predictive simulation
# ---------------------------------------------------------------------------


def test_bernoulli_contrast_probabilities_are_analytic():
    eta = np.array([[[0.0, 1.0], [0.5, -0.5]]])  # (1, 2, 2)
    tau = np.array([[0.8, 0.8]])
    trace = _trace(eta, tau)
    G = np.array([1.0, 0.0])

    contrast = counterfactual_predictive_contrast(
        trace, G=G, n_trials=1, term="tau", varying_term="", likelihood="bernoulli"
    )
    eta0 = eta[0] - 0.8 * G[None, :]
    np.testing.assert_allclose(contrast.prob_control, expit(eta0).mean(axis=1))
    np.testing.assert_allclose(
        contrast.prob_intervention, expit(eta0 + 0.8).mean(axis=1)
    )
    assert contrast.score_control.size == 0  # no score simulation for the floor rule


def test_score_simulation_mean_tracks_expit_eta():
    """With a huge kappa the Beta collapses; score means ≈ n·E[expit(eta)]."""
    rng_eta = np.random.default_rng(3)
    eta = rng_eta.normal(0.3, 0.05, size=(2, 200, 10))
    tau = np.full((2, 200), 0.6)
    kappa = np.full((2, 200), 5e4)
    trace = _trace(eta, tau, kappa=kappa)
    G = (np.arange(10) % 2).astype(float)
    n_trials = 40

    contrast = counterfactual_predictive_contrast(
        trace,
        G=G,
        n_trials=n_trials,
        term="tau",
        varying_term="",
        rng=np.random.default_rng(5),
    )
    eta0 = eta - 0.6 * G[None, None, :]
    expected_c = n_trials * expit(eta0).mean()
    expected_t = n_trials * expit(eta0 + 0.6).mean()
    assert contrast.score_control.mean() == pytest.approx(expected_c, abs=0.35)
    assert contrast.score_intervention.mean() == pytest.approx(expected_t, abs=0.35)
    assert contrast.score_control.min() >= 0
    assert contrast.score_intervention.max() <= n_trials


def test_new_child_swaps_fitted_intercepts():
    """With sigma_child == 0 the fitted intercepts are removed and no noise added."""
    n_chain, n_draw, n_obs = 1, 300, 6
    eta_base = np.zeros((n_chain, n_draw, n_obs))
    u_child_values = np.linspace(-2.0, 2.0, n_obs)  # one child per row
    u_child = np.broadcast_to(u_child_values, (n_chain, n_draw, n_obs)).copy()
    eta = eta_base + u_child  # the stored eta includes the fitted intercepts
    tau = np.zeros((n_chain, n_draw))
    kappa = np.full((n_chain, n_draw), 5e4)
    trace = _trace(
        eta,
        tau,
        kappa=kappa,
        extra={
            "u_child": (("chain", "draw", "child"), u_child),
            "sigma_child": (("chain", "draw"), np.zeros((n_chain, n_draw))),
        },
    )
    contrast = counterfactual_predictive_contrast(
        trace,
        G=np.zeros(n_obs),
        n_trials=100,
        term="tau",
        varying_term="",
        child_effect_name="u_child",
        child_sd_name="sigma_child",
        child_idx=np.arange(n_obs),
        rng=np.random.default_rng(9),
    )
    # Fitted intercepts removed, sigma_child = 0 adds nothing: every simulated
    # child sits at eta = 0, so the predictive mean is n/2 regardless of the
    # (deliberately extreme) fitted u_child values.
    assert contrast.score_control.mean() == pytest.approx(50.0, abs=0.6)


def test_new_child_intercepts_vary_per_simulated_child():
    """Simulations reusing a posterior draw must get independent intercepts.

    One posterior draw with sigma_child = 1.5 and a huge kappa: if every
    simulated child shared that draw's single intercept, the predictive spread
    would be Binomial-only (SD ~ a few items on 200 trials); independent
    per-child intercepts push expit(u) across most of the score range.
    """
    n_chain, n_draw, n_obs = 1, 1, 4
    u_child = np.zeros((n_chain, n_draw, n_obs))
    eta = np.zeros((n_chain, n_draw, n_obs))
    tau = np.zeros((n_chain, n_draw))
    kappa = np.full((n_chain, n_draw), 5e4)
    trace = _trace(
        eta,
        tau,
        kappa=kappa,
        extra={
            "u_child": (("chain", "draw", "child"), u_child),
            "sigma_child": (("chain", "draw"), np.full((n_chain, n_draw), 1.5)),
        },
    )
    contrast = counterfactual_predictive_contrast(
        trace,
        G=np.zeros(n_obs),
        n_trials=200,
        term="tau",
        varying_term="",
        child_effect_name="u_child",
        child_sd_name="sigma_child",
        child_idx=np.arange(n_obs),
        rng=np.random.default_rng(17),
    )
    # Binomial-only spread would be ~7 items; N(0, 1.5) intercepts give ~55.
    assert contrast.score_control.std() > 30.0


def test_new_child_population_ame_integrates_the_intercept():
    """#391 finding 4: the new-child AME integrates the intercept out, so it differs
    from the observed-child (fitted-intercept) AME and matches the analytic integral.
    """
    n_chain, n_draw, n_obs = 1, 1, 6
    # Widely-spread fitted intercepts with a small population SD: eta = u_child
    # (eta_base 0), G = 0 so eta0 = eta. The conditional AME averages expit over the
    # spread fitted intercepts; the new-child AME concentrates near u = 0, so the two
    # diverge sharply — exactly the nonlinearity finding 4 is about.
    u_vals = np.array([-3.0, -1.8, -0.6, 0.6, 1.8, 3.0])
    u_child = u_vals.reshape(n_chain, n_draw, n_obs)
    eta = u_child.copy()
    d = 1.0
    tau = np.full((n_chain, n_draw), d)  # constant treatment contribution
    s = 0.3
    trace = _trace(
        eta,
        tau,
        extra={
            "u_child": (("chain", "draw", "child"), u_child),
            "sigma_child": (("chain", "draw"), np.full((n_chain, n_draw), s)),
        },
    )
    contrast = counterfactual_predictive_contrast(
        trace,
        G=np.zeros(n_obs),
        n_trials=1,
        term="tau",
        varying_term="",
        likelihood="bernoulli",  # floor-rule path: previously no new-child step
        child_effect_name="u_child",
        child_sd_name="sigma_child",
        child_idx=np.arange(n_obs),
    )
    # New-child: eta0_new = eta0 - u_child = 0, so p_control = E[expit(u)] and
    # p_intervention = E[expit(d + u)], u ~ N(0, s). Compare to a large MC integral.
    mc = np.random.default_rng(0).normal(0.0, s, size=2_000_000)
    mc_ame = float(expit(d + mc).mean() - expit(mc).mean())
    assert contrast.ame_prob_new_child.size == 1
    assert contrast.ame_prob_new_child[0] == pytest.approx(mc_ame, abs=2e-3)
    # The observed-child AME averages expit(u_r + d) - expit(u_r) over the spread
    # fitted intercepts, so it is materially different from the population value.
    assert abs(contrast.ame_prob[0] - contrast.ame_prob_new_child[0]) > 0.03


def test_bernoulli_table_emits_both_population_targets():
    """The floor-rule table now carries the new-child population effect alongside the
    observed-child conditional one, with distinct population labels."""
    n_chain, n_draw, n_obs = 1, 40, 6
    rng = np.random.default_rng(5)
    u_child = rng.normal(0.0, 1.0, size=(n_chain, n_draw, n_obs))
    eta = u_child + rng.normal(0.0, 0.3, size=(n_chain, n_draw, n_obs))
    tau = rng.normal(0.5, 0.1, size=(n_chain, n_draw))
    trace = _trace(
        eta,
        tau,
        extra={
            "u_child": (("chain", "draw", "child"), u_child),
            "sigma_child": (("chain", "draw"), np.abs(rng.normal(1.0, 0.1, (n_chain, n_draw)))),
        },
    )
    contrast = counterfactual_predictive_contrast(
        trace, G=(np.arange(n_obs) % 2).astype(float), n_trials=1, term="tau",
        varying_term="", likelihood="bernoulli", child_effect_name="u_child",
        child_sd_name="sigma_child", child_idx=np.arange(n_obs),
    )
    table = predicted_scores_table(
        contrast, outcome_symbol="P", ci_prob=0.89, population="ref", contrast_status="s",
    )
    assert list(table.quantity) == [
        "event_probability_control",
        "event_probability_intervention",
        "average_marginal_effect",
        "event_probability_control_new_child_population",
        "event_probability_intervention_new_child_population",
        "average_marginal_effect_new_child_population",
    ]
    observed = table[table.quantity == "average_marginal_effect"].iloc[0]
    newchild = table[table.quantity == "average_marginal_effect_new_child_population"].iloc[0]
    assert observed["intercept_basis"].startswith("observed-child")
    assert newchild["intercept_basis"].startswith("new-child")
    # `population` (covariate/subject provenance) is shared; the basis distinguishes.
    assert observed["population"] == newchild["population"] == "ref"


def test_child_re_requires_sd_and_index():
    eta, tau, _, kappa = _rng_trace(seed=31)
    trace = _trace(eta, tau, kappa=kappa)
    with pytest.raises(ValueError, match="child_sd_name and child_idx"):
        counterfactual_predictive_contrast(
            trace,
            G=np.zeros(eta.shape[2]),
            n_trials=10,
            term="tau",
            varying_term="",
            child_effect_name="u_child",
        )


# ---------------------------------------------------------------------------
# Icon array rounding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "triple",
    [
        (0.694, 0.305, 0.0007),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1 / 3, 1 / 3, 1 / 3),
        (0.005, 0.005, 0.005),  # remainder folded into the negligible band
        (0.499, 0.499, 0.002),
    ],
)
def test_icon_array_counts_sum_to_100(triple):
    counts = icon_array_counts(*triple)
    assert sum(counts) == 100
    assert all(c >= 0 for c in counts)


def test_icon_array_counts_fold_remainder_into_middle():
    # 60/20/10 leaves 10 dots between delta and the ROPE edge; they must show
    # as "smaller either way", never as benefit or harm.
    benefit, middle, harm = icon_array_counts(0.6, 0.2, 0.1)
    assert benefit == 60
    assert harm == 10
    assert middle == 30


def test_icon_array_counts_reject_bad_probabilities():
    with pytest.raises(ValueError):
        icon_array_counts(1.2, 0.0, 0.0)


def test_icon_array_counts_reject_sum_above_one():
    with pytest.raises(ValueError, match="sum"):
        icon_array_counts(0.7, 0.4, 0.1)


def test_icon_array_counts_tolerate_boundary_tie_overshoot():
    # rope_card's inclusive comparisons can double-count a draw exactly at
    # delta, pushing the sum a hair over 1; counts must still total 100.
    counts = icon_array_counts(0.6, 0.4 + 5e-8, 0.0)
    assert sum(counts) == 100


# ---------------------------------------------------------------------------
# Table schema and end-to-end writer
# ---------------------------------------------------------------------------


def test_table_schema_graded_and_binary():
    eta, tau, _, kappa = _rng_trace(seed=41)
    trace = _trace(eta, tau, kappa=kappa)
    G = (np.arange(eta.shape[2]) % 2).astype(float)

    graded = predicted_scores_table(
        counterfactual_predictive_contrast(
            trace, G=G, n_trials=32, term="tau", varying_term="",
            rng=np.random.default_rng(1),
        ),
        outcome_symbol="L",
        ci_prob=0.95,
        population="pop",
        contrast_status="status",
    )
    assert list(graded.quantity) == [
        "predicted_score_control",
        "predicted_score_intervention",
        "predicted_score_paired_difference",
        "average_marginal_effect",
    ]
    assert set(graded.columns) >= {
        "outcome", "quantity", "scale", "median", "lo", "hi", "lo50", "hi50",
        "n_trials", "population", "intercept_basis", "contrast_status",
    }
    # No child intercept was supplied, so there is a single population and no
    # separate new-child rows.
    assert (graded.intercept_basis == "single population; no child random intercept").all()

    binary = predicted_scores_table(
        counterfactual_predictive_contrast(
            trace, G=G, n_trials=1, term="tau", varying_term="",
            likelihood="bernoulli",
        ),
        outcome_symbol="P",
        ci_prob=0.95,
        population="pop",
        contrast_status="status",
    )
    assert list(binary.quantity) == [
        "event_probability_control",
        "event_probability_intervention",
        "average_marginal_effect",
    ]
    assert binary.iloc[-1].scale == "risk_difference"


def test_writer_emits_all_artifacts(tmp_path):
    eta, tau, _, kappa = _rng_trace(seed=51)
    trace = _trace(eta, tau, kappa=kappa)
    G = (np.arange(eta.shape[2]) % 2).astype(float)

    summary = write_predicted_scores_artifacts(
        str(tmp_path),
        trace,
        outcome_symbol="L",
        item_label="Letter-sound knowledge (YARC-LSK)",
        G=G,
        n_trials=32,
        term="tau",
        varying_term="",
        delta=2.0,
        random_seed=13,
    )
    assert (tmp_path / "predicted_scores.csv").exists()
    assert (tmp_path / "predicted_scores.png").exists()
    assert (tmp_path / "icon_array.png").exists()
    assert (tmp_path / "icon_array.csv").exists()
    assert len(summary) == 4

    binary_dir = tmp_path / "binary"
    binary_dir.mkdir()
    summary_b = write_predicted_scores_artifacts(
        str(binary_dir),
        trace,
        outcome_symbol="P",
        item_label="Phonetic spelling (SPPHON)",
        G=G,
        n_trials=1,
        term="tau",
        varying_term="",
        likelihood="bernoulli",
        delta=0.10,
        random_seed=13,
    )
    assert (binary_dir / "predicted_scores.png").exists()
    assert (binary_dir / "icon_array.png").exists()
    assert len(summary_b) == 3


def test_writer_split_emits_individual_files(tmp_path):
    """split=True writes the distribution and effect panels as separate files."""
    eta, tau, _, kappa = _rng_trace(seed=71)
    trace = _trace(eta, tau, kappa=kappa)
    G = (np.arange(eta.shape[2]) % 2).astype(float)

    write_predicted_scores_artifacts(
        str(tmp_path),
        trace,
        outcome_symbol="L",
        item_label="Letter-sound knowledge (LS)",
        G=G,
        n_trials=32,
        term="tau",
        varying_term="",
        delta=2.0,
        random_seed=13,
        split=True,
    )
    for stem in ("predicted_scores", "predicted_effect"):
        assert (tmp_path / f"{stem}.png").exists()
        assert (tmp_path / f"{stem}.svg").exists()
        assert (tmp_path / f"{stem}.csv").exists()


def test_writer_skips_icon_array_without_delta(tmp_path):
    eta, tau, _, kappa = _rng_trace(seed=61)
    trace = _trace(eta, tau, kappa=kappa)
    G = (np.arange(eta.shape[2]) % 2).astype(float)

    write_predicted_scores_artifacts(
        str(tmp_path),
        trace,
        outcome_symbol="F",
        item_label="Test",
        G=G,
        n_trials=20,
        term="tau",
        varying_term="",
        delta=None,
        random_seed=13,
    )
    assert (tmp_path / "predicted_scores.png").exists()
    assert not (tmp_path / "icon_array.png").exists()
