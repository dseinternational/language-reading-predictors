# SPDX-License-Identifier: AGPL-3.0-or-later

"""The joint contrast estimator under zero, positive and negative outcome dependence.

#588 acceptance criterion: *"Simulation or exact-refit tests cover zero, positive and
negative outcome dependence and demonstrate the behaviour of the chosen contrast
estimator."* The audit's finding 1 asked for the same evidence in different words —
validate the dependence construction under all three signs rather than asserting what
it does.

Two levels of evidence, because the claim has two halves:

1. **The estimator.** ``tau_difference_summary`` subtracts *per draw*, so the
   contrast's uncertainty is ``Var(A) + Var(B) - 2 Cov(A, B)`` and moves with the
   sign of the cross-outcome posterior covariance. Pairing draws from two separate
   fits instead — the working-independence assumption the joint models exist to
   avoid — silently substitutes the zero-covariance answer. Both are exact
   properties, so they are tested exactly on constructed posteriors.
2. **The fitted block.** A sampled fit on simulated data with a known within-child
   residual correlation, to show the LKJ block carries that dependence through to
   the declared contrast in the right direction when it is identified — which is
   what makes the near-nil covariance channel measured on the three registered
   pairs a statement about *those data* rather than about the estimator.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pymc as pm
import pytest
import xarray as xr

from language_reading_predictors.statistical_models.reporting import (
    tau_difference_summary,
)

CI_PROB = 0.89


def _joint_trace(tau: np.ndarray, eta0: np.ndarray, G: np.ndarray):
    """A factorised multi-outcome trace whose ``eta`` is built at the observed ``G``.

    Mirrors the shape the joint factory persists: ``tau`` indexed by outcome and
    ``eta`` the fitted linear predictor including the treatment contribution, which
    :func:`_joint_ame_draws` nets out per draw.
    """
    n_chain, n_draw, n_outcome = tau.shape
    n_obs = eta0.shape[0]
    eta = np.empty((n_chain, n_draw, n_obs, n_outcome))
    for k in range(n_outcome):
        eta[..., k] = eta0[:, k][None, None, :] + tau[..., k, None] * G
    posterior = xr.Dataset(
        {
            "tau": (("chain", "draw", "outcome"), tau),
            "eta": (("chain", "draw", "obs_id", "outcome"), eta),
        },
        coords={"outcome": ["A", "B"], "obs_id": np.arange(n_obs)},
    )
    constant = xr.Dataset(
        {
            "G": ("obs_id", G),
            "y_post_cell_row": ("cell", np.repeat(np.arange(n_obs), n_outcome)),
            "y_post_cell_outcome": ("cell", np.tile(np.arange(n_outcome), n_obs)),
        }
    )
    return SimpleNamespace(posterior=posterior, constant_data=constant)


def _correlated_tau(rho: float, *, n_draw: int = 4000, seed: int = 5) -> np.ndarray:
    """Two outcomes' ``tau`` draws with a controlled posterior correlation.

    Equal marginal SDs so the contrast's width depends on ``rho`` alone: with
    ``Var(A) = Var(B) = s^2`` the difference has SD ``s * sqrt(2 (1 - rho))``.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n_draw, 2))
    chol = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    draws = 0.4 + 0.15 * (base @ chol.T)
    return draws[None, ...]


def _contrast_width(rho: float, *, shuffle_second_outcome: bool = False) -> dict:
    tau = _correlated_tau(rho)
    if shuffle_second_outcome:
        # Destroy the within-draw pairing while leaving both marginals intact: this
        # is exactly what pairing draws from two separately sampled fits does.
        rng = np.random.default_rng(99)
        tau = tau.copy()
        tau[0, :, 1] = rng.permutation(tau[0, :, 1])
    G = np.tile([1.0, 0.0], 20)
    eta0 = np.zeros((G.size, 2))
    summary = tau_difference_summary(
        _joint_trace(tau, eta0, G), ["A", "B"], ("A", "B"), ci_prob=CI_PROB
    )
    return {
        "median": float(summary["diff_logit_median"]),
        "width": float(summary["diff_logit_hi"]) - float(summary["diff_logit_lo"]),
    }


def test_the_contrast_estimator_tracks_cross_outcome_dependence():
    """Positive dependence narrows the declared contrast, negative widens it.

    The sign rule three report templates cite, tested on the estimator that
    implements it. The *location* is untouched: differencing per draw is linear in
    each draw, so only the spread responds to the covariance.
    """
    negative = _contrast_width(-0.6)
    independent = _contrast_width(0.0)
    positive = _contrast_width(0.6)

    assert positive["width"] < independent["width"] < negative["width"]
    # Against the exact Gaussian relation: width ratios follow sqrt(1 - rho).
    assert positive["width"] / independent["width"] == pytest.approx(
        np.sqrt(0.4), rel=0.05
    )
    assert negative["width"] / independent["width"] == pytest.approx(
        np.sqrt(1.6), rel=0.05
    )
    for other in (independent, positive):
        assert other["median"] == pytest.approx(negative["median"], abs=0.02)


def test_pairing_draws_from_separate_fits_discards_the_dependence():
    """Why the contrast must come from one joint fit rather than two paired ones.

    Shuffling one outcome's draws leaves both marginals exactly as they were and
    returns the independent-case width, so a product-of-marginals pairing reports
    the zero-covariance answer whatever the true dependence is — too wide under
    positive dependence and too narrow under negative.
    """
    independent = _contrast_width(0.0)["width"]
    for rho in (-0.6, 0.6):
        paired = _contrast_width(rho)["width"]
        unpaired = _contrast_width(rho, shuffle_second_outcome=True)["width"]
        assert unpaired == pytest.approx(independent, rel=0.05)
        assert (unpaired > paired) is (rho > 0)


# ---------------------------------------------------------------------------
# The fitted block: simulate the dependence, sample, and read the contrast.
# ---------------------------------------------------------------------------

SIMULATED_RHOS = (-0.7, 0.0, 0.7)


def _simulate_joint(prepared, *, rho: float, sigma_u: float = 0.8, seed: int = 3):
    """Overwrite the post-scores with draws carrying a known within-child correlation.

    The standard normals are drawn once and only the Cholesky factor changes, so the
    three simulated datasets differ in their cross-outcome dependence rather than in
    their marginal variability.
    """
    rng = np.random.default_rng(seed)
    outcomes = ("W", "R")
    n_obs = prepared.n_obs
    chol = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    u = sigma_u * (rng.normal(size=(n_obs, 2)) @ chol.T)
    tau_true = {"W": 0.5, "R": 0.2}
    for k, symbol in enumerate(outcomes):
        n = prepared.n_trials[symbol]
        eta = -0.2 + tau_true[symbol] * np.asarray(prepared.G, dtype=float) + u[:, k]
        p = 1.0 / (1.0 + np.exp(-eta))
        prepared.post_counts[symbol] = rng.binomial(n, p).astype(float)
        prepared.pre_counts[symbol] = rng.binomial(n, p).astype(float)
    return prepared


@pytest.fixture(scope="module")
def simulated_dependence_fits(tmp_path_factory) -> dict[float, dict]:
    """One correlated joint fit per simulated dependence sign, shared by the tests.

    Coarse by design — 80 children, 500 draws — because the question is which way
    the construction points, not how precisely it estimates a correlation. The
    residual scale is deliberately large so the block *is* identified; on the three
    registered companions it is not, which is why their contrast intervals differ
    from their parents' through marginal uncertainty rather than covariance
    (2026-08-24 review of the joint audit).
    """
    from language_reading_predictors.statistical_models import reporting as _report
    from language_reading_predictors.statistical_models.factories import (
        build_joint_model,
    )
    from language_reading_predictors.statistical_models.preprocessing import (
        load_and_prepare,
    )

    from .test_factories import _write_synthetic

    root = tmp_path_factory.mktemp("joint-dependence")
    fits: dict[float, dict] = {}
    for rho in SIMULATED_RHOS:
        directory = root / f"rho{rho}"
        directory.mkdir()
        path = _write_synthetic(directory, n_children=80, seed=17)
        prepared = load_and_prepare(path=path, phase_mode="itt", outcomes=("W", "R"))
        _simulate_joint(prepared, rho=rho)
        built = build_joint_model(
            prepared,
            outcomes=("W", "R"),
            use_residual_correlation=True,
            use_cross_baselines=False,
        )
        with built.model:
            idata = pm.sample(
                draws=500,
                tune=500,
                chains=2,
                cores=1,
                target_accept=0.9,
                nuts_sampler="nutpie",
                random_seed=41,
                progressbar=False,
            )
        G = np.asarray(prepared.G, dtype=float)
        _, ame = _report._joint_ame_draws(idata, ["W", "R"], G=G)
        summary = _report.tau_difference_summary(
            idata, ["W", "R"], ("W", "R"), ci_prob=CI_PROB, G=G
        )
        fits[rho] = {
            "residual_correlation": idata.posterior["u_corr_pair"].values.ravel(),
            "ame_correlation": float(np.corrcoef(ame[0], ame[1])[0, 1]),
            "marginal_sds": (float(ame[0].std()), float(ame[1].std())),
            "contrast_width": float(summary["diff_prob_hi"])
            - float(summary["diff_prob_lo"]),
            "contrast_median": float(summary["diff_prob_median"]),
        }
    return fits


@pytest.mark.parametrize("rho", SIMULATED_RHOS)
def test_the_fitted_block_recovers_the_simulated_dependence(
    simulated_dependence_fits, rho
):
    """The LKJ block points the right way under each simulated sign."""
    corr = simulated_dependence_fits[rho]["residual_correlation"]
    positive = float(np.mean(corr > 0))
    if rho > 0:
        assert positive > 0.9
    elif rho < 0:
        assert positive < 0.1
    else:
        assert 0.1 < positive < 0.9


def test_the_declared_contrast_carries_the_fitted_dependence(
    simulated_dependence_fits,
):
    """The consequence for the released quantity, which is what finding 2 asks about.

    The simulated dependence reaches the two treatment effects — their posterior
    correlation orders with it — and the contrast's interval orders the other way, as
    ``Var(A - B) = V_A + V_B - 2 Cov(A, B)`` requires. The per-outcome marginal SDs
    barely move across the three fits, so here the width change is a covariance
    correction and nothing else: the complement of the registered pairs, where the
    marginals widen and the covariance channel is unmeasurable.
    """
    negative, independent, positive = (simulated_dependence_fits[r] for r in SIMULATED_RHOS)

    assert (
        negative["ame_correlation"]
        < independent["ame_correlation"]
        < positive["ame_correlation"]
    )
    assert independent["ame_correlation"] == pytest.approx(0.0, abs=0.1)
    assert positive["contrast_width"] < independent["contrast_width"]
    assert independent["contrast_width"] < negative["contrast_width"]
    marginal_sds = [sd for fit in simulated_dependence_fits.values() for sd in fit["marginal_sds"]]
    assert max(marginal_sds) / min(marginal_sds) < 1.15
