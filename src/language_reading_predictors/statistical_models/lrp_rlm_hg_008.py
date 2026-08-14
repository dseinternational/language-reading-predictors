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

**Unresolved score definition and provisional ceiling.** The pipeline operationally
treats the stored 0-60 integer score as a bounded count with ``n_trials=60`` so
diagnostic fits remain possible, but 60 is only the prepared extract's observed
maximum. Published descriptions of the first-edition BAS Basic Number Skills Forms
B and C each give 34 items and a maximum raw score of 34, so the stored value cannot
be assumed to be a standard item-correct raw score. Both
``score_definition_confirmed`` and ``n_trials_confirmed`` are false in
``datasets.RLM_MEASURES``; the fit-time publication contract therefore withholds
scientific findings until the administered form and scoring transformation are
reconciled.
Per-measure prior calibration is left to the prior-critical-review
follow-up; the shared defaults are used here and the prior-predictive check will
flag any miscalibration.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
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
        kappa_prior_sigma=50.0,
    ),
)


def fit(config: str = "dev"):
    return fit_historical_growth(SPEC, config=config)
