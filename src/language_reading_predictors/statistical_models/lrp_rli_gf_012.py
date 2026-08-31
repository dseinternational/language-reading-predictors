# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPGF12 - gain factors for taught receptive vocabulary (TR), with broad vocabulary associates.

#421 Tier 1: extends ``gf-009`` (taught-receptive-vocabulary gains) with the standardised
transfer-vocabulary skills as adjusted associations. ``gf-009`` deliberately carries
**no** ``skill_symbols`` (under the revised DAG the standardised measures RV/EV sit
*downstream* of taught vocabulary, ``TR -> RV``), but the letter-sound -> word-reading
review found broad **receptive** vocabulary to be the single clearest predictor of
taught-word learning anywhere in the suite. This model surfaces that association
explicitly by adding ``skill_symbols = ("R", "E")`` alongside the randomised term and the
trait adjusters. (The review's headline figure came from a probe that also carried ``L``;
this registered model fits ``R``/``E`` only, so that number is motivation, not a
prediction for this fit.)

The ``R``/``E`` terms are **adjusted associations, not DAG-parent adjustments** - they
are entered to describe the review's finding, and (like every non-randomised term here)
are latent-GA-confounded. Only the randomised on-intervention term is causal, as its
period-1 average marginal effect; everything else is descriptive. Report median + inner
50% + outer 89% credible interval + P(>0) with that caveat.

The causal headline is interaction-free (#391 finding 3 decision, 2026-07-22):
the trt x ability / trt x own moderation questions for TR are carried by the
associational variant LRPGF09m (anchored to the per-outcome primary; this alternate
adjustment set declares none), and only the age x ability precision interaction
remains here.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
)
from language_reading_predictors.statistical_models.gain_factors import (
    GainFactorsModelSettings,
)
from language_reading_predictors.statistical_models.pipelines.gain_factors import fit_gain_factors

SPEC = ModelSpec(
    model_id="lrp-rli-gf-012",
    kind="gain_factors",
    title="Factors associated with gains in taught receptive vocabulary (TR), with broad vocabulary associates",
    outcome_symbol="TR",
    model_settings=GainFactorsModelSettings(
        # The review's finding: broad receptive (and expressive) vocabulary as
        # associates of taught-receptive-vocabulary gains. Adjusted associations.
        skill_symbols=("R", "E"),
        # R/E sit *downstream* of taught vocabulary under the revised DAG
        # (TR -> RV), so they are descriptive associates, not DAG-parent
        # adjusters — the role keeps config.json and the recipe from
        # mislabelling the adjustment rationale (#575 finding 9).
        descriptive_skills=("R", "E"),
        ability_covariate=V.BLOCKS,
        # TR's non-measure confounders (matches gf-009): hearing + phonological memory.
        adjust_for=("hs", "hs_missing", "erbto", "erbto_missing"),
        interactions=(("age", "ability"),),
        treated_only=False,
    ),
)


def fit(config: str = "dev") -> StatisticalFitContext:
    return fit_gain_factors(SPEC, config=config)
