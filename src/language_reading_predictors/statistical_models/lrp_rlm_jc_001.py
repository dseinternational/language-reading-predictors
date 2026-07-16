# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""RLMJC01 - Byrne joint correlated growth: reading, vocabulary, memory (#338 Phase B).

The joint multivariate extension of the per-measure historical-growth suite
(``lrp-rlm-hg-001/004/006``): BAS word reading, BPVS receptive vocabulary and
BAS recall of digits fitted **together**, each with its own supported-cell
group-by-wave grid and group-indexed random-effect scales, and the per-child
stable offsets correlated across measures through an LKJ prior. The headline
is the 3x3 **between-child correlation matrix of stable levels** - the Byrne
reading-language-memory coupling question in its most parsimonious
between-child form, and the trajectory-level companion of the wave-3
measurement model (``lrp-rlm-mm-001``).

Window: the suite's complete-case waves 1-3 core across all three measures
(n = 71) plus the #338 extension waves (wave 4 all groups, wave 5 Down
syndrome only) where a kept child has **all three** measures observed. The
correlation matrix is shared across the three reading groups (a stated
assumption at this n); the subject-intercept SDs and overdispersion are
group-indexed per the 2026-07-16 heterogeneity decision. PSIS-LOO is not
computed - the model has one likelihood node per measure, so a single
pointwise LOO is not defined for it.

**Descriptive natural-history evidence, not an intervention effect:**
``readgrp`` is a cohort factor with no causal warrant, and a between-child
correlation of stable levels says nothing about within-child coupling or
direction - the reciprocal question is Phase C's lagged model, gated on the
lagged Byrne DAG.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_rlm_joint_growth

SPEC = ModelSpec(
    model_id="lrp-rlm-jc-001",
    kind="historical_joint",
    title=(
        "Byrne joint correlated growth: word reading, receptive vocabulary "
        "and digit recall, waves 1-4 + DS wave 5"
    ),
    outcome_symbol=None,
    study_id="rlm",
    family="historical_joint",
    design="historical_cohort",
    estimand_type="descriptive",
    causal_status="none",
    dataset_ref="rlm:reading_language_memory_data_long",
    audit_baseline="complete_case_summary",
    extra={
        "study_id": "rlm",
        "measures": ("basread", "bpvs", "basdig"),
        "waves": (1, 2, 3),
        "extension_waves": (4, 5),
        "eta_prior_sigma": 1.5,
        "sigma_subject_prior_sigma": 0.5,
        "kappa_prior_sigma": 50.0,
        "lkj_eta": 2.0,
    },
)


def fit(config: str = "dev"):
    return fit_rlm_joint_growth(SPEC, config=config)
