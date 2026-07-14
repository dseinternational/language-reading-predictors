# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""RLMHG02 - historical BAS spelling growth, waves 1-3 (Byrne et al., #164).

A descriptive group-by-wave growth model for BAS spelling over the first three
waves of the Byrne, MacDonald & Buckley reading-language-memory study
(``study_id="rlm"``), the per-measure sibling of ``lrp-rlm-hg-001`` (word
reading). It runs through the shared historical-growth pipeline
(:func:`pipeline.fit_historical_growth`), so it uses the same sampler,
convergence gate, output layout and report conventions.

**Descriptive natural-history evidence, not an intervention effect:** ``readgrp``
(Down syndrome / average / reading-matched) is a cohort factor with no causal
warrant. Every quantity is a descriptive group-by-wave expected score or growth
rate. The audit baseline is the paper's Table 2 complete-case reproduction for
this measure (computed straight from the complete-case panel).

**Provisional ceiling.** The Beta-Binomial denominator (``n_trials=18``) is the
*observed maximum* in the prepared extract, not a manual-confirmed instrument
ceiling (``n_trials_confirmed=False`` in ``datasets.RLM_MEASURES``); confirming it
is a data-owner decision (``notes/202607021052-issue-164-byrne-followup-plan.md``,
decision 3). Per-measure prior calibration (e.g. ``eta`` anchoring for
lower-occupancy measures) is left to the prior-critical-review follow-up; the
shared defaults are used here and the prior-predictive check will flag any
miscalibration.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_historical_growth

SPEC = ModelSpec(
    model_id="lrp-rlm-hg-002",
    kind="historical_growth",
    title="Historical BAS spelling growth, waves 1-3 (Byrne et al.)",
    outcome_symbol="basspel",
    study_id="rlm",
    family="historical_growth",
    design="historical_cohort",
    estimand_type="descriptive",
    causal_status="none",
    dataset_ref="rlm:reading_language_memory_data_long",
    audit_baseline="table2_complete_case_summary",
    extra={
        "study_id": "rlm",
        "measure": "basspel",
        "waves": (1, 2, 3),
        "eta_prior_sigma": 1.5,
        "sigma_subject_prior_sigma": 0.5,
        "kappa_prior_sigma": 50.0,
    },
)


def fit(config: str = "dev"):
    return fit_historical_growth(SPEC, config=config)
