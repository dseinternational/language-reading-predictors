# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPMM01 - correlated-domain-factor measurement model (vocabulary / code / grammar).

The DAG-sanctioned successor to the closed LRP66 (#97). Instead of a single latent
general ability ``g``, it fits **correlated domain factors** - vocabulary {R, E},
code {L, B}, grammar {F, T} - over the standardised T1 skill indicators, with an
LKJ prior on the factor correlation matrix and factor variances fixed to 1. The
indicator residual variance is free, so a loading is a coefficient on the
unit-variance factor; the report carries the indicator-factor **correlation**
(``lambda / sqrt(lambda**2 + sigma**2)`` = ``sqrt(communality)``) and the
**communality** ``lambda**2 / (lambda**2 + sigma**2)`` alongside it. A structural
Beta-Binomial leg regresses word-reading gain (W post | W T1) on the latent
factors (plus non-verbal MA and age), giving measurement-error-corrected
factor->gain slopes.

This is the **identification-neutral but better-fitting** measurement match the
locked DAG defers to (#115): a single ``g`` is the wrong granularity for the
observed same-construct clustering. It is a **measurement / triangulation** model,
**not** causal - per ID-2 every factor->gain slope is a latent-ability-confounded
**adjusted association**, and the randomised causal claim continues to live in the
ITT suite (``lrp-rli-itt-010`` for word reading) and the ``gain_factors`` randomised term.

At n ~ 51 a latent model is fragile and prior-dependent; the wide intervals are
the honest result, exactly as the closed LRP66 reported. The unique contribution
over the merged ``gain_factors`` family is the explicit **loadings / communality**
table and the partial measurement-error correction of the skill slopes.

See ``notes/202606291700-correlated-domain-factor-measurement-model.md``.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_correlated_factor

SPEC = ModelSpec(
    model_id="lrp-rli-mm-001",
    kind="corr_factor",
    title=(
        "Correlated-domain-factor measurement model "
        "(vocabulary / code / grammar) - reading-gain structural leg"
    ),
    outcome_symbol="W",
    extra={
        "domains": {
            "vocabulary": ("R", "E"),
            "code": ("L", "B"),
            "grammar": ("F", "T"),
        },
        "structural_covariates": ("blocks",),
        "use_age": True,
    },
)


def fit(config: str = "dev"):
    return fit_correlated_factor(SPEC, config=config)
