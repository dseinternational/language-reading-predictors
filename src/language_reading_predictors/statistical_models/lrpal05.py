# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPAL05 - aligned-40-week per-protocol single gain for phonetic spelling (P).

Per-protocol onset-aligned Beta-Binomial ANCOVA of the aligned post-score on its
own onset baseline, age-at-onset, cognitive ability and the cohort indicator. Both
arms are aligned by intervention onset (immediate t1->t3, wait-list t2->t4); one
row per child, no child random intercept. The cohort term is NOT randomised
(confounded by age-at-onset and cohort/timing), so under this per-protocol design
no coefficient is a clean treatment effect -- all are associations.

Phonetic spelling is heavily floored, so this uses the floor rule (``likelihood="bernoulli_offfloor"``): a Bernoulli on the off-floor indicator (aligned post > 0), with no kappa and the cohort marginal an off-floor risk difference.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_aligned

SPEC = ModelSpec(
    model_id="lrpal05",
    kind="aligned",
    title="Aligned-40-week per-protocol gain for phonetic spelling (P)",
    outcome_symbol="P",
    extra={
        "ability_covariate": V.BLOCKS,
        "use_cohort": True,
        "use_dose": False,
        "likelihood": "bernoulli_offfloor",
    },
)


def fit(config: str = "dev"):
    return fit_aligned(SPEC, config=config)
