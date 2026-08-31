# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPGF01m - treatment-moderation variant of LRPGF01 (word reading (W)).

Explicitly associational moderation companion (#391 finding 3 decision, 2026-07-22):
identical to LRPGF01 except that it retains the pre-specified treatment
interactions (trt x ability, trt x own) that the interaction-free causal
headline dropped. Those interactions are estimated on ALL stacked periods — including
post-crossover rows with no untreated comparison — so the interaction-aware period-1
marginal (``beta_trt`` with the fitted interactions netted out, the #391 finding 1
netting) is model-dependent and partly informed by post-crossover data. No term here
is a causal headline — read the randomised effect from LRPGF01 — and the fit
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
    model_id="lrp-rli-gf-201",
    kind="gain_factors",
    title="Treatment-moderation variant: gains in word reading (W)",
    outcome_symbol="W",
    model_settings=GainFactorsModelSettings(
        skill_symbols=('TR', 'TE', 'R', 'E', 'L', 'N', 'B'),
        ability_covariate=V.BLOCKS,
        adjust_for=(),
        interactions=(("trt", "ability"), ("trt", "own"), ("age", "ability")),
        treated_only=False,
        moderation_variant=True,
    ),
)


def fit(config: str = "dev"):
    return fit_gain_factors(SPEC, config=config)
