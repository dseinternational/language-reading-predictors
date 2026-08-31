# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""RLMJC02 - within-child reading-language-memory coupling (#409 C2(ii)).

The balanced-core companion to ``lrp-rlm-jc-001``. BAS word reading, BPVS
receptive vocabulary and BAS recall of digits are fitted jointly over waves
1-3 for the 71 children observed on every measure at every wave. The model
keeps the correlated stable child offsets from RLMJC01 and adds a second
correlated logistic-normal deviation for each child-wave row. Double-centering
makes the latter average to zero within child and within group-by-wave cell, so
the target correlation asks whether a wave when a child is above their own
stable level on one measure is also an above-level wave on another measure.

The core-only window is deliberate. Adding waves 4-5 would give some children
more influence and mix the common three-group period with an attrition-selected
tail whose final wave contains only the Down syndrome group. The correlation
matrices are shared across groups as a parsimony assumption at this sample
size. The counts use a Binomial likelihood because the wave-specific latent
deviation already supplies extra-Binomial variation; retaining a separate
Beta-Binomial concentration made the correlation prior-dominated in the
development probe. Prior sensitivity remains required before interpretation.

Everything is descriptive. ``readgrp`` is an observational cohort factor;
within-child temporal co-movement does not identify direction or causation.
"""

from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
)
from language_reading_predictors.statistical_models.historical_joint import (
    HistoricalJointModelSettings,
)
from language_reading_predictors.statistical_models.pipelines.historical_joint import fit_rlm_joint_growth

SPEC = ModelSpec(
    model_id="lrp-rlm-jc-002",
    kind="historical_joint",
    title=(
        "Byrne within-child joint coupling: word reading, receptive vocabulary "
        "and digit recall, balanced waves 1-3"
    ),
    outcome_symbol=None,
    study_id="rlm",
    family="historical_joint",
    design="historical_cohort",
    estimand_type="descriptive",
    causal_status="none",
    dataset_ref="rlm:reading_language_memory_data_long",
    audit_baseline="complete_case_summary",
    model_settings=HistoricalJointModelSettings(
        measures=("basread", "bpvs", "basdig"),
        waves=(1, 2, 3),
        extension_waves=(),
        eta_prior_sigma=1.5,
        sigma_subject_prior_sigma=1.0,
        lkj_eta=2.0,
        within_correlation=True,
        sigma_within_prior_sigma=0.5,
        within_lkj_eta=2.0,
    ),
)


def fit(config: str = "dev") -> StatisticalFitContext:
    return fit_rlm_joint_growth(SPEC, config=config)
