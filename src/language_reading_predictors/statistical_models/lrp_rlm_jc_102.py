# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""RLMJC102 - wider within-scale prior sensitivity for ``lrp-rlm-jc-002`` (#588).

The registered alternative-prior companion the ``jc-002`` report has always
promised and the repository never provided. ``jc-002``'s report says a
``HalfNormal(1.0)`` sensitivity "checks whether the correlation conclusions
depend on that regularisation", but no registered companion, pipeline branch or
sub-fit ever set ``sigma_within_prior_sigma`` to 1.0, and no trace or provenance
artefact recorded such a fit (2026-08-23 joint audit, finding 5). This module
makes the promise reproducible: it is ``jc-002`` with that one setting changed.

**Why the prior matters here rather than being routine.** ``sigma_within`` - the
wave-specific latent deviation's scale, and the name it carries in the fitted
model and in every artefact - decides whether each measure passes the
resolvability rule (posterior support above the 0.05-logit identifiability
threshold), and therefore whether each correlation pair is interpretable at all.
A prior that regularises ``sigma_within`` toward zero can withhold a pair; one that
does not can admit it. That is a conclusion-level dependence, not a nuisance.

**Power scaling is not a substitute.** ``psense`` perturbs the prior locally
around the fitted posterior and answers a different question from an
independently sampled fit under a different prior. Compare the two fits'
posterior medians, 50% and 89% intervals, sign probabilities and resolvability
classifications; do **not** pair draws by draw number - the chains are unrelated.

Everything is descriptive, exactly as in the parent. ``readgrp`` is an
observational cohort factor and within-child temporal co-movement identifies no
direction or causation. Until this fit has been run and has passed its own
convergence gate, the parent's prior-robustness claim is preliminary.
"""

from dataclasses import replace

from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
)
from language_reading_predictors.statistical_models.lrp_rlm_jc_002 import SPEC as _PARENT
from language_reading_predictors.statistical_models.historical_joint import (
    HistoricalJointModelSettings,
)
from language_reading_predictors.statistical_models.pipelines.historical_joint import (
    fit_rlm_joint_growth,
)

# ``replace`` on the parent's frozen settings guarantees the measures, waves,
# window, likelihood and every other prior cannot drift apart from the fit this
# exists to check: the only difference is the within-scale prior.
_PARENT_SETTINGS = _PARENT.model_settings
# Not an ``assert``: this companion is defined by reusing its parent's
# settings object, and ``-O`` would remove the one check that the parent
# still declares them typed (#637).
if not isinstance(_PARENT_SETTINGS, HistoricalJointModelSettings):
    raise TypeError(
        f"{_PARENT.model_id} must declare HistoricalJointModelSettings for this "
        f"companion to reuse; got {type(_PARENT_SETTINGS).__name__}"
    )

SPEC = ModelSpec(
    model_id="lrp-rlm-jc-102",
    kind="historical_joint",
    title=(
        "Byrne within-child joint coupling: HalfNormal(1.0) within-scale prior "
        "sensitivity companion of lrp-rlm-jc-002"
    ),
    outcome_symbol=None,
    study_id="rlm",
    family="historical_joint",
    design="historical_cohort",
    estimand_type="descriptive",
    causal_status="none",
    dataset_ref="rlm:reading_language_memory_data_long",
    audit_baseline="complete_case_summary",
    model_settings=replace(
        _PARENT_SETTINGS,
        sigma_within_prior_sigma=1.0,
    ),
)


def fit(config: str = "dev") -> StatisticalFitContext:
    return fit_rlm_joint_growth(SPEC, config=config)
