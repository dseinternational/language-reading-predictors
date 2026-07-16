# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""RLMHG09 - historical BAS matrices growth, waves 3-4 + DS wave 5 (Byrne et al., #338).

A descriptive group-by-wave growth model for BAS matrices / non-verbal
reasoning (``basmat``) in the Byrne, MacDonald & Buckley
reading-language-memory study (``study_id="rlm"``), the per-measure sibling of
``lrp-rlm-hg-001`` (word reading). ``basmat`` was introduced at **wave 3** (no
wave-1/2 baseline exists), so this model runs on its own later-wave window
rather than the suite's waves 1-3 core: the complete-case core is **waves
3-4** (children with matrices observed at both), and wave 5 enters as an
extension wave for the Down syndrome group (the only group followed there).
It runs through the shared historical-growth pipeline
(:func:`pipeline.fit_historical_growth`), so it uses the same sampler,
convergence gate, output layout and report conventions.

Because the window starts at wave 3, the observed baseline here is **not** a
paper Table 2 reproduction (Table 2 covers the first three waves); it is the
same complete-case observed-cell audit computed from the prepared extract.
Interval growth is summarised on the children observed at both endpoint
waves, and the random-effect scales (``sigma_subject``, ``kappa``) are
indexed by group (follow-up-plan decision 7).

**Descriptive natural-history evidence, not an intervention effect:** ``readgrp``
(Down syndrome / average / reading-matched) is a cohort factor with no causal
warrant. Every quantity is a descriptive group-by-wave expected score or growth
rate.

**Confirmed ceiling (#338).** The Beta-Binomial denominator (``n_trials=28``)
is the instrument's confirmed maximum (``n_trials_confirmed=True`` in
``datasets.RLM_MEASURES``): BAS Matrices (first edition) has 28 items (CLOSER cognitive-measures guide; Parsons, 2014); Laws et al. (1995) used Raven's CPM (36 items) as this cohort's matrices measure, so the sign-off records the caveat that this column is BAS Matrices. Researched and signed off by the data owner
(2026-07-16). Per-measure prior calibration is left to the prior-critical-review
follow-up; the shared defaults are used here and the prior-predictive check will
flag any miscalibration.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_historical_growth

SPEC = ModelSpec(
    model_id="lrp-rlm-hg-009",
    kind="historical_growth",
    title="Historical BAS matrices growth, waves 3-4 + DS wave 5 (Byrne et al.)",
    outcome_symbol="basmat",
    study_id="rlm",
    family="historical_growth",
    design="historical_cohort",
    estimand_type="descriptive",
    causal_status="none",
    dataset_ref="rlm:reading_language_memory_data_long",
    audit_baseline="complete_case_summary",
    extra={
        "study_id": "rlm",
        "measure": "basmat",
        "waves": (3, 4),
        "extension_waves": (5,),
        "eta_prior_sigma": 1.5,
        "sigma_subject_prior_sigma": 0.5,
        "kappa_prior_sigma": 50.0,
    },
)


def fit(config: str = "dev"):
    return fit_historical_growth(SPEC, config=config)
