# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""RLMHG01 - historical BAS word-reading growth, waves 1-4 + DS wave 5 (Byrne et al., #165/#338).

The first non-RLI, package-level statistical model: a descriptive group-by-wave
growth model for BAS word reading over waves 1-4 (plus the Down-syndrome-only
wave-5 tail) of the Byrne, MacDonald & Buckley reading-language-memory study
(``study_id="rlm"``). It runs through the shared statistical-model pipeline
(:func:`pipelines.historical_growth.fit_historical_growth`) so it uses the same sampler,
convergence gate, output layout and report conventions as the intervention
models.

**Extended follow-up window (#338).** The complete-case core stays the paper's
waves 1-3 (the Table 2 audit subset); waves 4 and 5 enter as **extension
waves**: kept children contribute wherever the measure was observed, so the
later cells are an attrition-selected follow-up tail with their own per-cell
``n``. Wave 4 carries all three groups (the between-group window is waves
1-4); wave 5 exists only for the Down syndrome group. Interval growth is
summarised on the children observed at both endpoint waves, and the
random-effect scales (``sigma_subject``, ``kappa``) are indexed by group
(follow-up-plan decision 7).

This is **descriptive natural-history evidence, not an intervention effect**:
``readgrp`` (Down syndrome / average / reading-matched) is a cohort factor with no
causal warrant. The audit baseline is the paper's Table 2 complete-case
reproduction (identical to the complete-case core subset the model is fit on).
Supersedes the standalone ``scripts/fit_historical_growth_model.py`` prototype
(#163).
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.historical_growth import (
    HistoricalGrowthModelSettings,
)
from language_reading_predictors.statistical_models.pipelines.historical_growth import fit_historical_growth

SPEC = ModelSpec(
    model_id="lrp-rlm-hg-001",
    kind="historical_growth",
    title="Historical BAS word-reading growth, waves 1-4 + DS wave 5 (Byrne et al.)",
    outcome_symbol="basread",
    study_id="rlm",
    family="historical_growth",
    design="historical_cohort",
    estimand_type="descriptive",
    causal_status="none",
    dataset_ref="rlm:reading_language_memory_data_long",
    audit_baseline="table2_complete_case_summary",
    model_settings=HistoricalGrowthModelSettings(
        measure="basread",
        waves=(1, 2, 3),
        extension_waves=(4, 5),
        eta_prior_sigma=1.5,
        # Widened 0.5 -> 1.0 (#383, prior-critical-review 2026-07-21): under
        # HalfNormal(0.5) the fitted Down-syndrome sigma_subject posteriors ran
        # 1.25-1.39 across the verbal/reading measures — at/beyond the prior's
        # 99th percentile (1.29) — a genuine prior-data conflict that mildly
        # biased the reported between-child spread downward. HalfNormal(1.0)
        # covers the fitted range (99th pct 2.58) while staying weakly
        # informative for the low-heterogeneity measures. This reverses the
        # 2026-07-07 review's 1.0 -> 0.5 reconciliation on the later review's
        # evidence. The eta[group, wave] grid stays unanchored at 1.5: the
        # pushforward shows it is well-calibrated on the full-range basread
        # scale (not a low-occupancy outcome), so anchoring it is not warranted.
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


def fit(config: str = "dev"):
    return fit_historical_growth(SPEC, config=config)
