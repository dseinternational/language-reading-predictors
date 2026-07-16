# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""RLMHG01 - historical BAS word-reading growth, waves 1-4 + DS wave 5 (Byrne et al., #165/#338).

The first non-RLI, package-level statistical model: a descriptive group-by-wave
growth model for BAS word reading over waves 1-4 (plus the Down-syndrome-only
wave-5 tail) of the Byrne, MacDonald & Buckley reading-language-memory study
(``study_id="rlm"``). It runs through the shared statistical-model pipeline
(:func:`pipeline.fit_historical_growth`) so it uses the same sampler,
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
from language_reading_predictors.statistical_models.pipeline import fit_historical_growth

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
    extra={
        "study_id": "rlm",
        "measure": "basread",
        "waves": (1, 2, 3),
        "extension_waves": (4, 5),
        "eta_prior_sigma": 1.5,
        # Reconciled 1.0 -> 0.5 to match the shared random-effect SD scale
        # (sigma_child ~ HalfNormal(0.5)) — prior-critical-review 2026-07-07,
        # recommendation 3. The eta[group, wave] grid stays unanchored at 1.5:
        # the pushforward shows it is well-calibrated on the full-range basread
        # scale (not a low-occupancy outcome), so anchoring it is not warranted.
        "sigma_subject_prior_sigma": 0.5,
        "kappa_prior_sigma": 50.0,
    },
)


def fit(config: str = "dev"):
    return fit_historical_growth(SPEC, config=config)
