# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP58 - Mechanism model: letter-sound knowledge (L) -> word reading (W).

Adjustment set (DAG-derived):

- G, A: standard confounders.
- E: confounds L and W (E -> L and E -> W). Must be conditioned on.
- R: confounds L through E (R -> E -> L) and affects W. Conditioning on R
  closes the backdoor via E.
- W_pre: baseline word reading.

B (phoneme blending) is a descendant of L in the literacy-development
cascade and must NOT be conditioned on (conditioning on a descendant of L
would introduce collider bias and attenuate f^L). F and P are also excluded
for the same reason. This decision is documented in the accompanying report.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp58",
    kind="mechanism",
    title="Mechanism model: letter-sound knowledge (L) -> word reading (W)",
    outcome_symbol="W",
    mechanism_symbol="L",
    adjustment=["G", "A", "E", "R", "W_pre"],
    # Age GP off, subject random intercept on — see
    # notes/202604181700-lrp55-age-gp-drop.md and
    # notes/202604181800-mechanism-random-intercepts.md.
    extra={
        "adjust_baseline_symbol": "W",
        "use_age_gp": False,
        "phase_specific_mechanism": False,
        "use_subject_random_intercept": True,
    },
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
