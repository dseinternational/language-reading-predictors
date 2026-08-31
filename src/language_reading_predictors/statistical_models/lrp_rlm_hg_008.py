# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""RLMHG08 - historical BAS number-skills growth, waves 1-4 (Byrne et al., #164/#338).

A descriptive group-by-wave growth model for BAS number skills (``basnum``) over the four annual waves of the Byrne, MacDonald & Buckley reading-language-memory
study (``study_id="rlm"``), the per-measure sibling of ``lrp-rlm-hg-001`` (word
reading). It runs through the shared historical-growth pipeline
(:func:`pipelines.historical_growth.fit_historical_growth`), so it uses the same sampler,
convergence gate, output layout and report conventions.

**Extended follow-up window (#338).** The complete-case core stays the paper's
waves 1-3 (the Table 2 audit subset); wave 4 enters as an **extension wave**:
kept children contribute wherever the measure was observed, so the wave-4
cells are an attrition-selected follow-up tail with their own per-cell ``n``.
Wave 4 carries all three groups (the between-group window is waves 1-4);
``basnum`` was not assessed at wave 5. Interval growth is summarised on the
children observed at both endpoint waves, and the random-effect scales
(``sigma_subject``, ``kappa``) are indexed by group (follow-up-plan
decision 7).

**Descriptive natural-history evidence, not an intervention effect:** ``readgrp``
(Down syndrome / average / reading-matched) is a cohort factor with no causal
warrant. Every quantity is a descriptive group-by-wave expected score or growth
rate. The audit baseline is the paper's Table 2 complete-case reproduction for
this measure (computed straight from the complete-case panel).

**Primary-source identity; provisional ceiling.** Byrne et al. (2002) state that
raw scores were analysed and label this measure BAS number skills. Their published
Table 3 means reproduce exactly from the prepared extract, confirming the stored
score's source identity. The paper does not state the administered form or maximum,
so ``n_trials=60`` remains the observed-maximum operational denominator and
``n_trials_confirmed=False``. The fit-time publication contract therefore withholds
scientific findings until the 1983 BAS manual or cohort test record confirms the
bounded-score ceiling.
Per-measure prior calibration is left to the prior-critical-review
follow-up; the shared defaults are used here and the prior-predictive check will
flag any miscalibration.
"""

from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
)
from language_reading_predictors.statistical_models.historical_growth import (
    HistoricalGrowthModelSettings,
)
from language_reading_predictors.statistical_models.pipelines.historical_growth import fit_historical_growth

SPEC = ModelSpec(
    model_id="lrp-rlm-hg-008",
    kind="historical_growth",
    title="Historical BAS number-skills growth, waves 1-4 (Byrne et al.)",
    outcome_symbol="basnum",
    study_id="rlm",
    family="historical_growth",
    design="historical_cohort",
    estimand_type="descriptive",
    causal_status="none",
    dataset_ref="rlm:reading_language_memory_data_long",
    audit_baseline="table2_complete_case_summary",
    model_settings=HistoricalGrowthModelSettings(
        measure="basnum",
        waves=(1, 2, 3),
        extension_waves=(4,),
        eta_prior_sigma=1.5,
        # Widened 0.5 -> 1.0 (#383): DS sigma_subject posteriors sat at/beyond
        # the HalfNormal(0.5) 99th percentile — see lrp_rlm_hg_001.py.
        sigma_subject_prior_sigma=1.0,
        # Dispersion-scale prior (2026-08-21 review, finding 8): the previous
        # HalfNormal(50) on kappa itself gave the near-Binomial limit a prior
        # probability of 0.001 at these denominators, and 20 of 27 fitted cells
        # had a kappa posterior no narrower than its prior. 1/sqrt(kappa) ~
        # HalfNormal(0.25) preserves the old prior's median variance inflation
        # at every denominator while letting "no extra-Binomial dispersion" be
        # an ordinary outcome. See priors.inv_sqrt_kappa_prior.
        dispersion_prior_sigma=0.25,
    ),
)


def fit(config: str = "dev") -> StatisticalFitContext:
    return fit_historical_growth(SPEC, config=config)
