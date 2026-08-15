# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""RLMGC01: baseline verbal reasoning and the word-reading trajectory (#409 D4).

The model uses the Byrne paper's waves 1-3 and the confirmed measures BAS word
reading (trajectory outcome) and wave-1 BAS similarities (baseline verbal-reasoning
proxy). Children need the baseline proxy and word reading at two or more waves.

Reading-group-specific nuisance intercepts and slopes absorb the three historical
cohorts' different trajectories. ``gamma`` is the shared within-group association
between a 1-SD higher baseline similarity score and the logit reading growth rate;
``delta`` is the association with reading level at the pooled mean age. Neither is
causal. Latent general ability remains incompletely measured, and the reading-matched
group was selected on the outcome itself.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.growth import GrowthModelSettings
from language_reading_predictors.statistical_models.pipelines.growth import fit_growth

SPEC = ModelSpec(
    model_id="lrp-rlm-gc-001",
    kind="growth",
    title=(
        "Does baseline verbal reasoning predict the BAS word-reading trajectory? "
        "(Byrne waves 1-3, reading-group-adjusted)"
    ),
    outcome_symbol=None,
    model_settings=GrowthModelSettings(
        outcomes=("basread",),
        baseline_covariate="bassim",
        use_shared_factor=False,
        use_random_slope=False,
        age_ability_interaction=False,
        waves=(1, 2, 3),
        baseline_scale="logit_safe",
        min_outcome_waves=2,
        adjust_for_group=True,
    ),
    study_id="rlm",
    family="growth",
    design="historical_cohort_longitudinal",
    estimand_type="association",
    causal_status="none",
)


def fit(config: str = "dev"):
    """Fit the registered Byrne reading-trajectory model."""
    return fit_growth(SPEC, config=config)
