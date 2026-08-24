# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Deterministic statistical validation of the DiD arm-by-wave estimands.

The production model uses a Beta-Binomial likelihood, whose binomial limit has the
same arm-by-wave mean structure tested here. A saturated binomial model has a closed
form maximum-likelihood estimate in every arm-wave cell, so these tests can exercise
identification and misspecification without slow or potentially flaky NUTS sampling.

The tests deliberately distinguish three quantities:

* ``tau_t2``: the immediate-minus-waitlist arm gap at t2, identified by the original
  randomisation;
* ``arm_gap_t3``: the arm gap after both arms have received intervention but have
  different histories;
* ``delta_crossover = tau_t2 - arm_gap_t3``: the change in the arm gap, which mixes
  catch-up with carryover, cumulative exposure, block and maturation differences.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest


@dataclass(frozen=True)
class _HistoryDGP:
    """Additive logit-scale intervention history for the t2/t3 cells."""

    first_block: float
    retained_first_block: float
    second_block: float

    @property
    def arm_gap_t3(self) -> float:
        """Immediate history minus waitlist history at t3.

        At t3, the waitlist has received its first block, while the immediate arm has
        a retained first-block contribution plus its second-block contribution.
        """
        return self.retained_first_block + self.second_block - self.first_block

    @property
    def delta_crossover(self) -> float:
        return self.first_block - self.arm_gap_t3


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1.0 - p))


def _simulate_counts(
    arm_gaps: np.ndarray,
    *,
    seed: int,
    n_children_per_arm: int = 4_000,
    n_trials: int = 30,
    t3_subject_shift: np.ndarray | None = None,
) -> np.ndarray:
    """Simulate bounded counts, shaped ``(arm, wave, child)``.

    Arm 0 is waitlist and arm 1 is immediate intervention. The common wave effects
    are nuisance maturation terms; ``arm_gaps[w]`` is the conditional logit gap at
    wave ``w``. Optional t3 shifts represent latent child-specific maturation.
    """
    gaps = np.asarray(arm_gaps, dtype=float)
    if gaps.shape != (3,):
        raise ValueError(f"arm_gaps must have shape (3,), got {gaps.shape}")
    if t3_subject_shift is not None:
        t3_subject_shift = np.asarray(t3_subject_shift, dtype=float)
        expected = (2, n_children_per_arm)
        if t3_subject_shift.shape != expected:
            raise ValueError(
                f"t3_subject_shift must have shape {expected}, got "
                f"{t3_subject_shift.shape}"
            )

    alpha = -1.15
    wave_effect = np.asarray([0.0, 0.35, 0.65])
    eta = np.empty((2, 3, n_children_per_arm), dtype=float)
    for arm in (0, 1):
        eta[arm] = alpha + wave_effect[:, None] + arm * gaps[:, None]
    if t3_subject_shift is not None:
        eta[:, 2, :] += t3_subject_shift

    rng = np.random.default_rng(seed)
    return rng.binomial(n_trials, _expit(eta)).astype(np.int64)


def _fit_saturated_arm_wave(
    counts: np.ndarray, *, n_trials: int = 30
) -> dict[str, np.ndarray | float]:
    """Closed-form saturated-binomial MLE and derived arm-wave contrasts.

    The half-count correction is negligible at this sample size but prevents an
    infinite logit if a future stress scenario creates an all-zero cell.
    """
    values = np.asarray(counts)
    if values.ndim != 3 or values.shape[:2] != (2, 3):
        raise ValueError(f"counts must have shape (2, 3, n), got {values.shape}")
    successes = values.sum(axis=2, dtype=float)
    cell_trials = values.shape[2] * n_trials
    cell_probability = (successes + 0.5) / (cell_trials + 1.0)
    cell_logit = _logit(cell_probability)
    arm_gap = cell_logit[1] - cell_logit[0]
    return {
        "cell_probability": cell_probability,
        "arm_gap": arm_gap,
        "tau_t2": float(arm_gap[1]),
        "arm_gap_t3": float(arm_gap[2]),
        "delta_crossover": float(arm_gap[1] - arm_gap[2]),
    }


def test_arm_by_wave_recovers_t2_gap_without_forcing_t3_gap() -> None:
    """A correctly specified saturated model recovers two distinct arm gaps."""
    truth = np.asarray([0.0, 0.52, 0.11])
    fit = _fit_saturated_arm_wave(_simulate_counts(truth, seed=20260715))

    np.testing.assert_allclose(fit["arm_gap"], truth, atol=0.025)
    assert fit["tau_t2"] == pytest.approx(0.52, abs=0.025)
    assert fit["arm_gap_t3"] == pytest.approx(0.11, abs=0.025)
    assert fit["delta_crossover"] == pytest.approx(0.41, abs=0.035)
    # This is the restriction removed from the legacy common-current-treatment model.
    assert abs(fit["tau_t2"] - fit["arm_gap_t3"]) > 0.30


def test_history_effects_move_catchup_without_moving_randomised_t2() -> None:
    """Carryover, cumulative exposure and block effects belong to the t3 association.

    The exact same simulated t1/t2 observations are reused in every scenario. Only t3
    is regenerated, so any change in ``delta_crossover`` cannot be attributed to a
    changed randomised t2 contrast.
    """
    scenarios = {
        # Immediate at t3: no retained P1 contribution + a P2 block as effective as
        # the waitlist's first block. The arms have equal histories on this scale.
        "no_carryover_equal_blocks": _HistoryDGP(0.50, 0.00, 0.50),
        # Persistence plus an additional block leaves the immediate arm ahead.
        "cumulative_carryover": _HistoryDGP(0.50, 0.50, 0.25),
        # Partial retention and a weak/different second block leave it behind.
        "different_second_block": _HistoryDGP(0.50, 0.25, 0.05),
    }
    shared = _simulate_counts(np.asarray([0.0, 0.50, 0.0]), seed=1101)
    fitted: dict[str, dict[str, np.ndarray | float]] = {}

    for index, (name, dgp) in enumerate(scenarios.items()):
        scenario = shared.copy()
        t3 = _simulate_counts(
            np.asarray([0.0, 0.50, dgp.arm_gap_t3]), seed=2200 + index
        )
        scenario[:, 2, :] = t3[:, 2, :]
        fitted[name] = _fit_saturated_arm_wave(scenario)

        assert fitted[name]["tau_t2"] == fitted["no_carryover_equal_blocks"]["tau_t2"]
        assert fitted[name]["arm_gap_t3"] == pytest.approx(
            dgp.arm_gap_t3, abs=0.025
        )
        assert fitted[name]["delta_crossover"] == pytest.approx(
            dgp.delta_crossover, abs=0.035
        )

    catchup = {name: float(fit["delta_crossover"]) for name, fit in fitted.items()}
    assert max(catchup.values()) - min(catchup.values()) > 0.40


def test_heterogeneous_maturation_is_absorbed_by_post_crossover_gap() -> None:
    """A realised imbalance in latent maturation changes catch-up, not ``tau_t2``.

    Both scenarios contain the same overall 50:50 mixture of faster and slower
    maturers and exactly the same t1/t2 data. In the imbalanced scenario, more fast
    maturers happen to be in the immediate arm. The t3 arm gap therefore changes even
    though neither the randomised t2 effect nor either arm's intervention history did.
    """
    n_children = 4_000
    magnitude = 0.45
    shared = _simulate_counts(
        np.asarray([0.0, 0.50, 0.0]),
        seed=3301,
        n_children_per_arm=n_children,
    )

    alternating = np.resize(np.asarray([-magnitude, magnitude]), n_children)
    balanced_shift = np.stack([alternating, alternating])

    slow = np.full(3 * n_children // 4, -magnitude)
    fast = np.full(n_children // 4, magnitude)
    imbalanced_shift = np.stack(
        [np.concatenate([slow, fast]), np.concatenate([-fast, -slow])]
    )

    fitted = {}
    for index, (name, shift) in enumerate(
        (("balanced", balanced_shift), ("imbalanced", imbalanced_shift))
    ):
        scenario = shared.copy()
        t3 = _simulate_counts(
            np.asarray([0.0, 0.50, 0.0]),
            seed=4400 + index,
            n_children_per_arm=n_children,
            t3_subject_shift=shift,
        )
        scenario[:, 2, :] = t3[:, 2, :]
        fitted[name] = _fit_saturated_arm_wave(scenario)

    assert fitted["balanced"]["tau_t2"] == fitted["imbalanced"]["tau_t2"]
    assert abs(fitted["balanced"]["arm_gap_t3"]) < 0.03
    assert fitted["imbalanced"]["arm_gap_t3"] > 0.35
    assert (
        fitted["balanced"]["delta_crossover"]
        - fitted["imbalanced"]["delta_crossover"]
        > 0.35
    )


# --- production-likelihood parameter recovery (#390) ---------------------------


def _prepared_arm_wave_panel(
    *,
    n_children_per_arm: int,
    truth: dict[str, float | np.ndarray],
    n_trials: int,
    seed: int,
):
    """A PreparedData levels panel simulated from the production DGP.

    Children carry a Normal(0, sigma_child) random intercept; every row draws a
    Beta-Binomial count through the logit mean structure the factory fits. Ages
    are simulated independent of outcome, so the fitted precision term has a
    true coefficient of zero.
    """
    from language_reading_predictors.statistical_models.preprocessing import (
        PreparedData,
        standardise,
    )

    rng = np.random.default_rng(seed)
    n_children = 2 * n_children_per_arm
    child_g = np.repeat([1.0, 0.0], n_children_per_arm)
    u_child = rng.normal(0.0, truth["sigma_child"], size=n_children)
    ages = rng.uniform(60.0, 120.0, size=n_children)

    child_idx = np.repeat(np.arange(n_children), 3)
    phase = np.tile(np.arange(3), n_children)
    G = child_g[child_idx]
    wave_offset = np.asarray(truth["wave_offset"], dtype=float)
    arm_gap = np.asarray(truth["arm_gap"], dtype=float)
    eta = (
        truth["alpha"]
        + wave_offset[phase]
        + u_child[child_idx]
        + arm_gap[phase] * G
    )
    p = _expit(eta)
    kappa = float(truth["kappa"])
    theta = rng.beta(p * kappa, (1.0 - p) * kappa)
    y = rng.binomial(n_trials, theta).astype(float)

    a_months = ages[child_idx] + 5.0 * phase
    a_std, age_scaler = standardise(a_months)
    prepared = PreparedData(
        subject_ids=np.asarray([f"c{i:03d}" for i in child_idx]),
        child_idx=child_idx.astype(np.int64),
        phase=phase.astype(np.int64),
        G=G,
        A_months=a_months,
        A_std=a_std,
        age_scaler=age_scaler,
        pre_logit={},
        post_counts={"W": y},
        n_trials={"W": n_trials},
        n_obs=int(y.size),
        n_children=n_children,
        n_phases=3,
        dropped_rows=0,
        phase_mode="levels",
    )
    # A single realisation's fitted model can only see the *realised* spread of
    # its 2n random intercepts, not the super-population sigma, so the recovery
    # assertion for sigma_child targets this quantity.
    return prepared, float(u_child.std(ddof=1))


def test_production_beta_binomial_random_intercept_recovers_truth() -> None:
    """#390: one small real-sampler recovery test of the hierarchical production
    likelihood — Beta-Binomial + non-centred child random intercept through
    ``build_did_model`` itself — complementing the closed-form saturated-binomial
    checks above, which cannot see the random-intercept or dispersion parts."""
    import pymc as pm

    from language_reading_predictors.statistical_models.factories import (
        build_did_model,
    )

    truth = {
        "alpha": -0.4,
        "wave_offset": np.asarray([0.0, 0.35, 0.55]),
        "arm_gap": np.asarray([0.0, 0.60, 0.20]),
        "sigma_child": 0.45,
        "kappa": 60.0,
    }
    prepared, realised_sigma_child = _prepared_arm_wave_panel(
        n_children_per_arm=30, truth=truth, n_trials=30, seed=20260807
    )
    built = build_did_model(prepared, outcome_symbol="W")
    with built.model:
        trace = pm.sample(
            draws=500,
            tune=500,
            chains=2,
            cores=2,
            target_accept=0.9,
            nuts_sampler="nutpie",
            return_inferencedata=True,
            random_seed=20260807,
            progressbar=False,
        )

    posterior = trace.posterior

    def _draws(name: str) -> np.ndarray:
        return posterior[name].values.ravel()

    def _covers(name: str, value: float) -> bool:
        d = _draws(name)
        lo, hi = np.quantile(d, (0.03, 0.97))
        return bool(lo <= value <= hi)

    tau = _draws("tau_t2")
    gap_t3 = _draws("arm_gap_t3")
    assert abs(float(tau.mean()) - 0.60) < 0.30
    assert _covers("tau_t2", 0.60)
    assert abs(float(gap_t3.mean()) - 0.20) < 0.30
    assert _covers("arm_gap_t3", 0.20)
    # The two gaps are genuinely distinct in the recovered posterior — the
    # restriction the legacy common-current-treatment model imposed.
    assert float((tau - gap_t3).mean()) > 0.10
    # Hierarchy and dispersion: the parts the saturated closed form cannot see.
    assert _covers("sigma_child", realised_sigma_child)
    assert 0.2 < float(_draws("sigma_child").mean()) < 0.8
    assert _covers("kappa", truth["kappa"])
    # The simulated ages are outcome-independent, so the precision term is null.
    assert abs(float(_draws("gamma_A").mean())) < 0.15


def test_production_recovery_under_material_baseline_imbalance() -> None:
    """#576 finding 4: recovery when the realised t1 arm gap is **not** zero.

    The recovery test above simulates a zero baseline gap, so it cannot see what the
    family's estimand sign-off is about. ``tau_t2`` is the covariate-adjusted t2
    arm-gap *level*, not the differenced ``tau_t2 - arm_gap_t1``, and the baseline
    adjustment it does carry is soft: a realised imbalance is allocated between the
    tightly regularised ``arm_gap_t1`` (Normal(0, 0.3)) and the arm-mean of the
    shared child intercepts. Under a zero true baseline gap every parameterisation
    agrees, so the distinction is untested precisely where it matters.

    Here the truth carries a material +0.45-logit t1 gap. Three things must hold:
    the level ``tau_t2`` recovers the *level* (0.60), not the change; the balance
    term recovers the baseline gap rather than being shrunk to nothing by its tight
    prior; and the derived change ``tau_t2 - arm_gap_t1`` recovers the true change
    (0.15). Together those pin the estimand the family publishes, so a future
    re-parameterisation onto a t1-referenced gap change cannot land silently.
    """
    import pymc as pm

    from language_reading_predictors.statistical_models.factories import (
        build_did_model,
    )

    truth = {
        "alpha": -0.4,
        "wave_offset": np.asarray([0.0, 0.35, 0.55]),
        # A material pre-randomisation imbalance in the immediate arm's favour.
        "arm_gap": np.asarray([0.45, 0.60, 0.20]),
        "sigma_child": 0.45,
        "kappa": 60.0,
    }
    prepared, _realised_sigma_child = _prepared_arm_wave_panel(
        n_children_per_arm=60, truth=truth, n_trials=30, seed=20260824
    )
    built = build_did_model(prepared, outcome_symbol="W")
    with built.model:
        trace = pm.sample(
            draws=500,
            tune=500,
            chains=2,
            cores=2,
            target_accept=0.9,
            nuts_sampler="nutpie",
            return_inferencedata=True,
            random_seed=20260824,
            progressbar=False,
        )

    posterior = trace.posterior

    def _draws(name: str) -> np.ndarray:
        return posterior[name].values.ravel()

    def _covers(name: str, value: float) -> bool:
        d = _draws(name)
        lo, hi = np.quantile(d, (0.03, 0.97))
        return bool(lo <= value <= hi)

    # 1. The published contrast is the t2 arm-gap LEVEL.
    assert _covers("tau_t2", 0.60)
    assert abs(float(_draws("tau_t2").mean()) - 0.60) < 0.30
    # It is emphatically not the gap *change*: 0.15 must sit outside the interval,
    # or the two estimands would be indistinguishable and this test vacuous.
    assert not _covers("tau_t2", 0.15)

    # 2. The tight balance prior still lets the realised imbalance be recovered
    #    rather than shrinking it into the child intercepts.
    assert _covers("arm_gap_t1", 0.45)

    # 3. The derived t1-referenced change recovers the true change.
    change = _draws("tau_t2") - _draws("arm_gap_t1")
    lo, hi = np.quantile(change, (0.03, 0.97))
    assert lo <= 0.15 <= hi


def test_wide_baseline_allocation_priors_do_not_move_the_t2_level() -> None:
    """#576 finding 4: the registered allocation sensitivity (LRPDID104) is honest.

    Widening ``arm_gap_t1`` and ``sigma_child`` changes *how* a realised baseline
    imbalance is allocated. It must not change what ``tau_t2`` estimates. Fitting the
    same simulated panel under both prior settings and requiring both posteriors to
    cover the true t2 gap level is what makes the companion a sensitivity rather than
    a second, differently-defined model.
    """
    import pymc as pm

    from language_reading_predictors.statistical_models.factories import (
        build_did_model,
    )

    truth = {
        "alpha": -0.4,
        "wave_offset": np.asarray([0.0, 0.35, 0.55]),
        "arm_gap": np.asarray([0.45, 0.60, 0.20]),
        "sigma_child": 0.45,
        "kappa": 60.0,
    }
    prepared, _ = _prepared_arm_wave_panel(
        n_children_per_arm=60, truth=truth, n_trials=30, seed=20260824
    )
    covered = {}
    for label, kwargs in (
        ("default", {}),
        (
            "wide",
            {"arm_gap_t1_prior_sigma": 1.0, "sigma_child_prior_sigma": 1.0},
        ),
    ):
        built = build_did_model(prepared, outcome_symbol="W", **kwargs)
        with built.model:
            trace = pm.sample(
                draws=400,
                tune=400,
                chains=2,
                cores=2,
                target_accept=0.9,
                nuts_sampler="nutpie",
                return_inferencedata=True,
                random_seed=20260824,
                progressbar=False,
            )
        tau = trace.posterior["tau_t2"].values.ravel()
        lo, hi = np.quantile(tau, (0.03, 0.97))
        covered[label] = (bool(lo <= 0.60 <= hi), float(tau.mean()))
    assert covered["default"][0], covered
    assert covered["wide"][0], covered
    # And the two point estimates agree to well within their own uncertainty.
    assert abs(covered["default"][1] - covered["wide"][1]) < 0.15, covered
