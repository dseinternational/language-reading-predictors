# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPDID105 - dispersion-prior sensitivity for LRPDID10 (basic concepts, F).

This is the low-denominator member of the DiD dispersion check. LRPDID106
checks a high-denominator outcome.

The reference prior ``kappa ~ HalfNormal(50)`` has support at every positive
concentration, but gives the near-Binomial region very little probability for
long tests. Variance inflation is ``(n + kappa) / (1 + kappa)``. At ``n = 170``,
the prior median concentration implies about 5.9 times Binomial variance. At
smaller denominators the same concentration gives less extra variation.

This companion uses ``1 / sqrt(kappa) ~ HalfNormal(0.25)``, which gives the
near-Binomial region appreciable prior support. ``kappa`` remains a derived
quantity and the other settings are unchanged. Compare arm-gap and dispersion
summaries with the parent to assess sensitivity. The two outcome checks differ
in more than their denominators, so their comparison cannot isolate a
denominator effect. No term's causal status changes.
"""

from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
)
from language_reading_predictors.statistical_models.did import DiDModelSettings
from language_reading_predictors.statistical_models.pipelines.did import fit_did

SPEC = ModelSpec(
    model_id="lrp-rli-did-105",
    kind="did",
    title=("Dispersion-prior sensitivity for the basic-concepts arm-by-wave contrasts (CELF) (F)"),
    outcome_symbol="F",
    family="did",
    design="waitlist-crossover arm-by-wave levels, inverse-sqrt dispersion prior",
    estimand_type="mixed",
    causal_status="t2 randomised; t3 a randomised treatment-schedule contrast",
    model_settings=DiDModelSettings(
        # Identical to LRPDID10 except kappa_prior_family.
        outcomes=("F",),
        waves=(0, 1, 2),
        use_child_re=True,
        use_age=True,
        dose=False,
        kappa_prior_family="halfnormal_inverse_sqrt",
    ),
)


def fit(config: str = "dev") -> StatisticalFitContext:
    return fit_did(SPEC, config=config)
