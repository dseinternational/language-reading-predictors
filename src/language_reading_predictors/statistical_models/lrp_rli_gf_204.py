# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPGF04m - treatment-moderation variant of LRPGF04 (letter sounds (L)).

Explicitly associational moderation companion (#391 finding 3 decision, 2026-07-22):
identical to LRPGF04 except that it retains the pre-specified treatment
interactions (trt x ability, trt x own) that the interaction-free causal
headline dropped. Those interactions are estimated on ALL stacked periods — including
post-crossover rows with no untreated comparison — so the interaction-aware period-1
marginal (``beta_trt`` with the fitted interactions netted out, the #391 finding 1
netting) is model-dependent and partly informed by post-crossover data. No term here
is a causal headline — read the randomised effect from LRPGF04 — and the fit
is skipped by the robustness release gate for exactly that reason
(``release.gate_applies``). Moderation on ~25 period-1 control children is not
seriously estimable; these posteriors bound the moderation questions, they do not
settle them.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.gain_factors import (
    GainFactorsModelSettings,
)
from language_reading_predictors.statistical_models.pipelines.gain_factors import fit_gain_factors

SPEC = ModelSpec(
    model_id="lrp-rli-gf-204",
    kind="gain_factors",
    title="Treatment-moderation variant: gains in letter sounds (L)",
    outcome_symbol="L",
    model_settings=GainFactorsModelSettings(
        skill_symbols=(),
        ability_covariate=V.BLOCKS,
        adjust_for=('hs', 'hs_missing', 'deapp_c', 'deapp_c_missing'),
        interactions=(("trt", "ability"), ("trt", "own"), ("age", "ability")),
        treated_only=False,
        moderation_variant=True,
    ),
)


def fit(config: str = "dev"):
    return fit_gain_factors(SPEC, config=config)
