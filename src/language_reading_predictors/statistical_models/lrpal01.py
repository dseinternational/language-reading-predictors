# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPAL01 - aligned-40-week per-protocol single gain for word reading (W).

A cross-sectional Beta-Binomial ANCOVA of the onset-aligned post-score on its
own onset baseline, age-at-onset and cognitive ability (block design), plus the
cohort indicator (immediate vs wait-list). Both arms are aligned by intervention
onset across ~two periods (~the 40-week RLI program): the immediate arm onsets at
t1 (window t1->t3) and the wait-list arm at t2 (window t2->t4). One row per child,
so there is no child random intercept.

The cohort term is **not** randomised -- it contrasts the arms at their own
aligned endpoints, confounded by age-at-onset and cohort/timing -- so under this
per-protocol design no coefficient is a clean treatment effect; all are
associations (see the LRPAL design note). The dose variant (cumulative sessions)
is a separate sensitivity model, not part of this primary adjustment set.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_aligned

SPEC = ModelSpec(
    model_id="lrpal01",
    kind="aligned",
    title="Aligned-40-week per-protocol gain for word reading (W)",
    outcome_symbol="W",
    extra={
        "ability_covariate": V.BLOCKS,
        "use_cohort": True,
        "use_dose": False,
    },
)


def fit(config: str = "dev"):
    return fit_aligned(SPEC, config=config)
