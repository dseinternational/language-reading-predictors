# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Documentation edited by a LLM-based AI tool (Codex/GPT-6).

"""Phase-0 word-reading decomposition using a composite phonics mediator.

The registered mediator combines letter-sound knowledge and phoneme blending
with equal weights. Both legs condition on baseline word reading, the baseline
composite and the declared covariates. The g-formula integrates over the fitted
normal mediator to estimate the indirect and residual components in the
available-case sample. Same-wave measurement, latent general ability and
treatment-induced confounding by teaching dose limit causal interpretation.
"""

from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
)
from language_reading_predictors.statistical_models.mediation_settings import (
    MediationModelSettings,
)
from language_reading_predictors.statistical_models.pipelines.mediation import fit_mediation

SPEC = ModelSpec(
    model_id="lrp-rli-med-062",
    kind="mediation",
    title=(
        "Reading-route decomposition: how much of the word-reading (W) gain is "
        "code-based-route-compatible (letter-sound + blending) vs lexical?"
    ),
    outcome_symbol="W",
    mechanism_symbol=None,  # the mediator is a composite, not a single measure
    adjustment=[
        # E/R retained — settled by the time-indexed d-separation (#264;
        # notes/202607142340-lrp264-mediation-adjustment-dsep.md); the revised-DAG
        # common causes HS/RW/SP added by the missing-indicator method (#246).
        "G",
        "A",
        "E",
        "R",
        "W_pre",
        "hs",
        "hs_missing",
        "erbto",
        "erbto_missing",
        "deapp_c",
        "deapp_c_missing",
    ],
    model_settings=MediationModelSettings(
        mediator_kind="gaussian_composite",
        # Code-based-route composite. P dropped per the measurement-sensitivity audit.
        route_symbols=("L", "B"),
    ),
)


def fit(config: str = "dev") -> StatisticalFitContext:
    return fit_mediation(SPEC, config=config)
