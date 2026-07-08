# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""RLMHG01 - historical BAS word-reading growth, waves 1-3 (Byrne et al., #165).

The first non-RLI, package-level statistical model: a descriptive group-by-wave
growth model for BAS word reading over the first three waves of the Byrne,
MacDonald & Buckley reading-language-memory study (``study_id="rlm"``). It runs
through the shared statistical-model pipeline (:func:`pipeline.fit_historical_growth`)
so it uses the same sampler, convergence gate, output layout and report
conventions as the intervention models.

This is **descriptive natural-history evidence, not an intervention effect**:
``readgrp`` (Down syndrome / average / reading-matched) is a cohort factor with no
causal warrant. The audit baseline is the paper's Table 2 complete-case
reproduction (identical to the complete-case subset the model is fit on).
Supersedes the standalone ``scripts/fit_historical_growth_model.py`` prototype
(#163).
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_historical_growth

SPEC = ModelSpec(
    model_id="lrp-rlm-hg-001",
    kind="historical_growth",
    title="Historical BAS word-reading growth, waves 1-3 (Byrne et al.)",
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
