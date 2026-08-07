# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPGF05m - treatment-moderation variant of LRPGF05 (phonetic spelling (P), off-floor).

Explicitly associational moderation companion (#391 finding 3 decision, 2026-07-22):
identical to LRPGF05 except that it retains the pre-specified treatment
interactions (trt x ability, trt x own, where own is the binary off-floor-at-pre indicator) that the interaction-free causal
headline dropped. Those interactions are estimated on ALL stacked periods — including
post-crossover rows with no untreated comparison — so the interaction-aware period-1
marginal (``beta_trt`` with the fitted interactions netted out, the #391 finding 1
netting) is model-dependent and partly informed by post-crossover data. No term here
is a causal headline — read the randomised effect from LRPGF05 — and the fit
is skipped by the robustness release gate for exactly that reason
(``release.gate_applies``). Moderation on ~25 period-1 control children is not
seriously estimable; these posteriors bound the moderation questions, they do not
settle them.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_gain_factors

SPEC = ModelSpec(
    model_id="lrp-rli-gf-205",
    kind="gain_factors",
    title="Treatment-moderation variant: gains in phonetic spelling (P), off-floor",
    outcome_symbol="P",
    extra={
        "skill_symbols": ('L', 'B'),
        "ability_covariate": V.BLOCKS,
        "adjust_for": ('erbto', 'erbto_missing'),
        "interactions": (("trt", "ability"), ("trt", "own"), ("age", "ability")),
        "treated_only": False,
        "likelihood": "bernoulli_offfloor",
        "moderation_variant": True,
    },
)


def fit(config: str = "dev"):
    return fit_gain_factors(SPEC, config=config)
