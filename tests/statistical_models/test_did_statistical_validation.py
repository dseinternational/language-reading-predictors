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
