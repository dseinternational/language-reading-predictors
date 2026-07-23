# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""RLMMM01 - Byrne correlated domain-factor measurement model, wave 3 (#338 Phase B).

The Byrne analogue of ``lrp-rli-mm-001``, **measurement-only** (2026-07-16
sign-off): correlated unit-variance domain factors over the wave-3 nine-measure
battery - reading {``basread``, ``basspel``, ``woco``}, language {``bpvs``,
``trog``}, memory {``basdig``}, ability {``bassim``, ``basmat``, ``basnum``} -
with the per-child factor scores marginalised out of the Gaussian likelihood
(the mm-001 funnel fix). The deliverable is the loadings/communalities table
and the domain-factor correlation matrix: the modern analogue of the paper's
correlation tables, and the construct-structure summary that precedes any
coupling model (Phase C). No structural leg - the factor -> gain question
belongs to Phase D (``lrp-rlm-adj-001`` / ``lrp-rlm-hs-001``).

**Wave 3** is the only wave carrying the full ability triad (``basmat`` is
wave-3+), n = 75 complete-case across the battery. **Memory is a single
indicator** (``basdig``; the paper's visual-recall measures are not in the
prepared extract), so its loading and residual are fixed by an assumed
reliability of 0.8 (``lambda = sqrt(0.8)``): correlations involving the memory
factor scale with that assumption and the report says so. Loadings are shared
across the three reading groups - a pooled measurement model with the
measurement-invariance assumption stated, not tested, at this n. Descriptive
associations only; nothing causal exists in this cohort.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_rlm_corr_factor

SPEC = ModelSpec(
    model_id="lrp-rlm-mm-001",
    kind="corr_factor",
    title=(
        "Byrne correlated domain-factor measurement model, wave 3 "
        "(reading / language / memory / ability)"
    ),
    outcome_symbol=None,
    study_id="rlm",
    family="corr_factor",
    design="historical_cohort",
    estimand_type="descriptive",
    causal_status="none",
    dataset_ref="rlm:reading_language_memory_data_long",
    extra={
        "study_id": "rlm",
        "wave": 3,
        "domains": {
            "reading": ("basread", "basspel", "woco"),
            "language": ("bpvs", "trog"),
            "memory": ("basdig",),
            "ability": ("bassim", "basmat", "basnum"),
        },
        "single_indicator_reliability": 0.8,
        "lkj_eta": 2.0,
        # Communality parameterisation (#409 item B, the gate rescue): the free
        # parameter is each indicator's communality c ~ Beta(comm_alpha, comm_beta),
        # with lambda = sqrt(c) and sigma = sqrt(1 - c) so lambda**2 + sigma**2 = 1
        # (the unit variance standardised indicators imply). This removes the
        # over-parameterised lambda-sigma ridge / Heywood corner that gate-failed the
        # earlier free HalfNormal build (143 divergences, R-hat 1.03). Beta(2, 2) is a
        # weakly-informative communality prior centred at 0.5 — flagged for review.
        "comm_alpha": 2.0,
        "comm_beta": 2.0,
        # target_accept stays at 0.99 (as the RLI corr-factor fits use). The wave-3
        # domains are near-collinear (factor correlations 0.81-0.95), so the LKJ
        # correlation matrix sits against the positive-definite boundary and leaves
        # residual boundary divergences even after the communality reparameterisation
        # removed the loading-residual ridge (143 -> 72). Pushing target_accept to
        # 0.999 cut those to ~12 but starved the chains (min ESS 749 -> 99, max R-hat
        # 1.006 -> 1.045), so 0.99 is the better operating point: the deliverables
        # (loadings, communalities, factor correlations) converge cleanly (R-hat
        # <= 1.01, ESS >= 400) and only the intrinsic near-singular-correlation
        # divergences remain. Whether that structure should instead be a higher-order
        # / single general factor is a modelling-structure question flagged for review.
        "target_accept": 0.99,
    },
)


def fit(config: str = "dev"):
    return fit_rlm_corr_factor(SPEC, config=config)
