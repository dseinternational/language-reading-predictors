# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPAL07 - aligned-40-week per-protocol single gain for basic concept knowledge (F).

Per-protocol onset-aligned Beta-Binomial ANCOVA of the aligned post-score on its
own onset baseline, age-at-onset, cognitive ability and the cohort indicator. Both
arms are aligned by intervention onset (immediate t1->t3, wait-list t2->t4); one
row per child, no child random intercept. The cohort term is NOT randomised
(confounded by age-at-onset and cohort/timing), so under this per-protocol design
no coefficient is a clean treatment effect -- all are associations.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_aligned

SPEC = ModelSpec(
    model_id="lrpal07",
    kind="aligned",
    title="Aligned-40-week per-protocol gain for basic concept knowledge (F)",
    outcome_symbol="F",
    extra={
        "ability_covariate": V.BLOCKS,
        "use_cohort": True,
        "use_dose": False,
    },
)


def fit(config: str = "dev"):
    return fit_aligned(SPEC, config=config)
