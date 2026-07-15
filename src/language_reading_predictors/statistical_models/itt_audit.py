# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Analysis-set and attrition diagnostics for the RLI randomised comparison.

The modelling data begin at the 54-child t1 analytic cohort, whereas the trial
randomised 57 children (29 immediate intervention, 28 wait-list control).  These
helpers keep that distinction visible in every ITT artefact and provide a simple,
model-free extreme-case bound for the marginal post-score contrast.

The bound is deliberately not an imputation model: each missing randomised
participant is assigned either the minimum or maximum possible score.  It therefore
answers whether a headline direction survives the full range allowed by the test,
without pretending the absent outcomes or covariates are recoverable from these data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from language_reading_predictors.statistical_models.preprocessing import PreparedData


INTERVENTION_G = 1
CONTROL_G = 0
ARM_LABELS = {INTERVENTION_G: "intervention", CONTROL_G: "control"}

# Burgoyne et al. (2012) CONSORT flow: 29 immediate, 28 waiting control were
# randomised; the repository's t1 data contain 28 and 26 respectively.
RLI_RANDOMISED_BY_G = {INTERVENTION_G: 29, CONTROL_G: 28}
RLI_AVAILABLE_T1_BY_G = {INTERVENTION_G: 28, CONTROL_G: 26}


def analysis_set_table(
    prepared: PreparedData,
    *,
    outcome_symbol: str | None = None,
) -> pd.DataFrame:
    """Return randomised, t1-available and fitted counts by trial arm.

    ``prepared`` may already be a complete-case or subgroup restriction.  When an
    ``outcome_symbol`` is supplied, a row counts as fitted only when its post-score is
    finite.  The table therefore works for ordinary ITTs, covariate-restricted
    robustness fits and the baseline-floor subgroup.
    """

    G = np.asarray(prepared.G, dtype=int)
    if outcome_symbol is None:
        keep = np.ones(G.shape[0], dtype=bool)
    else:
        if outcome_symbol not in prepared.post_counts:
            raise KeyError(f"Outcome {outcome_symbol!r} is absent from prepared data")
        keep = np.isfinite(np.asarray(prepared.post_counts[outcome_symbol], dtype=float))

    rows: list[dict[str, int | str]] = []
    for g in (INTERVENTION_G, CONTROL_G):
        randomised_n = RLI_RANDOMISED_BY_G[g]
        available_n = RLI_AVAILABLE_T1_BY_G[g]
        fitted_n = int(np.sum((G == g) & keep))
        if fitted_n > available_n:
            raise ValueError(
                f"Fitted {ARM_LABELS[g]} count {fitted_n} exceeds the archived "
                f"t1-available count {available_n}."
            )
        rows.append(
            {
                "arm": ARM_LABELS[g],
                "G": g,
                "randomised_n": randomised_n,
                "available_t1_n": available_n,
                "fitted_n": fitted_n,
                "absent_from_archive_n": randomised_n - available_n,
                "not_in_fitted_analysis_n": randomised_n - fitted_n,
                "excluded_after_archive_n": available_n - fitted_n,
            }
        )
    return pd.DataFrame(rows)


def randomised_postscore_bounds(
    prepared: PreparedData,
    outcome_symbol: str,
) -> pd.DataFrame:
    """Worst-case full-randomised bounds for the marginal t2 score contrast.

    Scores are normalised to the proportion-correct scale ``[0, 1]``.  For each
    randomised participant without an observed fitted post-score, the lower scenario
    assigns 0 and the upper scenario assigns 1.  The resulting intervention-minus-
    control interval is a transparent attrition benchmark, not the covariate-adjusted
    Bayesian estimand and not a missing-at-random analysis.
    """

    if outcome_symbol not in prepared.post_counts:
        raise KeyError(f"Outcome {outcome_symbol!r} is absent from prepared data")
    n_trials = int(prepared.n_trials[outcome_symbol])
    if n_trials <= 0:
        raise ValueError(f"n_trials must be positive for {outcome_symbol!r}")

    G = np.asarray(prepared.G, dtype=int)
    post = np.asarray(prepared.post_counts[outcome_symbol], dtype=float)
    finite = np.isfinite(post)
    if np.any((post[finite] < 0) | (post[finite] > n_trials)):
        raise ValueError(f"Observed {outcome_symbol!r} scores lie outside [0, {n_trials}]")

    arm: dict[int, dict[str, float | int]] = {}
    for g in (INTERVENTION_G, CONTROL_G):
        observed = post[(G == g) & finite] / n_trials
        observed_n = int(observed.size)
        randomised_n = RLI_RANDOMISED_BY_G[g]
        missing_n = randomised_n - observed_n
        if missing_n < 0:
            raise ValueError(
                f"Observed {ARM_LABELS[g]} count {observed_n} exceeds randomised "
                f"count {randomised_n}."
            )
        observed_sum = float(np.sum(observed))
        arm[g] = {
            "observed_n": observed_n,
            "missing_n": missing_n,
            "observed_mean": float(np.mean(observed)) if observed_n else float("nan"),
            "full_mean_min": observed_sum / randomised_n,
            "full_mean_max": (observed_sum + missing_n) / randomised_n,
        }

    intervention = arm[INTERVENTION_G]
    control = arm[CONTROL_G]
    observed_diff = float(intervention["observed_mean"]) - float(control["observed_mean"])
    lower = float(intervention["full_mean_min"]) - float(control["full_mean_max"])
    upper = float(intervention["full_mean_max"]) - float(control["full_mean_min"])

    return pd.DataFrame(
        [
            {
                "outcome": outcome_symbol,
                "scale": "proportion_correct",
                "observed_intervention_n": int(intervention["observed_n"]),
                "observed_control_n": int(control["observed_n"]),
                "missing_intervention_n": int(intervention["missing_n"]),
                "missing_control_n": int(control["missing_n"]),
                "observed_mean_difference": observed_diff,
                "worst_case_lower": lower,
                "worst_case_upper": upper,
                "observed_items_difference": observed_diff * n_trials,
                "worst_case_items_lower": lower * n_trials,
                "worst_case_items_upper": upper * n_trials,
                "n_trials": n_trials,
                "interpretation": (
                    "Extreme-case full-randomised marginal post-score bound; not the "
                    "covariate-adjusted Bayesian estimand."
                ),
            }
        ]
    )
