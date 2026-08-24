# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP305 - dispersion prior-family sensitivity for L -> R (n = 170) (#605).

The high-denominator half of the #605 dispersion pair. Companion to **LRP97**,
identical in rows, outcome, baseline, adjustment set and coefficient priors,
differing only in that the Beta-Binomial concentration is put on the **dispersion
scale**, ``1 / sqrt(kappa) ~ HalfNormal(0.25)``, instead of the family's shared
``kappa ~ HalfNormal(50)``.

**Why this outcome.** Receptive vocabulary has the family's largest denominator
(170 items), where the registered prior bites hardest: at its own median
(``kappa`` about 33.7) it implies a **5.87x** variance inflation over Binomial, and
coming within 10% of Binomial would need ``kappa > 1689`` - a region
``HalfNormal(50)`` gives effectively zero mass. Four registered mechanism fits sit at
this denominator; if the prior is binding anywhere in the family it is here.

Read it exactly as LRP304: does the family's declared items-scale headline contrast
move when the model is *allowed* to conclude there is no extra-Binomial dispersion,
and did the data move ``kappa`` at all? ``dispersion_summary.csv`` publishes the
posterior concentration against its prior and the implied variance-inflation factor.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.mechanism import (
    MechanismModelSettings,
)
from language_reading_predictors.statistical_models.pipelines.mechanism import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-305",
    kind="mechanism",
    title="Dispersion prior sensitivity: letter sounds (L) -> receptive vocabulary (R)",
    outcome_symbol="R",
    mechanism_symbol="L",
    adjustment=["G", "A", "R_pre"],
    model_settings=MechanismModelSettings(
        adjust_baseline_symbol="R",
        outcomes=("R", "L"),
        adjust_for=("hs", "hs_missing", "attend", "deapp_c", "deapp_c_missing"),
        linear_mechanism=True,
        kappa_prior_family="halfnormal_inverse_sqrt",
        use_age_gp=False,
        use_subject_random_intercept=True,
    ),
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
