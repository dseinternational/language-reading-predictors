# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""MED-186 - interventional analogue of the L -> N off-floor decomposition.

This is the parent+100 companion to MED-086 (#323). It fits the same joint
letter-sound mediator and Bernoulli off-floor nonword-reading outcome model, on
the same Phase-0 rows and with the same adjustment set, but labels the simulated
decomposition as the interventional direct and indirect effects (IDE/IIE).

The interventional estimand avoids the cross-world counterfactual required by
natural direct and indirect effects, so the treatment-induced common cause
``IS`` (sessions) is not a recanting-witness obstruction to *defining* the
target. It does not make the target identifiable here (#585 finding 2):
identifying an interventional effect under an exposure-induced mediator-outcome
confounder requires that confounder to be measured and integrated over -- within
its treated distribution in the outcome leg and its control distribution in the
mediator leg. ``IS`` is in neither leg, so this functional identifies the
interventional estimand only under the same no-exposure-induced-confounding
condition the natural version needs, and latent general ability still confounds
the mediator-outcome relationship either way. Read the
IDE/IIE as a weaker-assumption, model-based decomposition, not a causal route.

In this fully parametric implementation the mediator is drawn from the same
fitted covariate-conditional law for both estimand classes. MED-186's IIE should
therefore match MED-086's NIE up to posterior-sampling and g-formula Monte Carlo
error. That agreement is an implementation check, not evidence that shared dose
causes no distortion.
"""

from dataclasses import replace

from language_reading_predictors.statistical_models.lrp_rli_med_086 import (
    SPEC as PARENT_SPEC,
)
from language_reading_predictors.statistical_models.mediation_settings import (
    MediationModelSettings,
)
from language_reading_predictors.statistical_models.pipelines.mediation import fit_mediation
from language_reading_predictors.statistical_models.context import (
    StatisticalFitContext,
)

PARENT_SETTINGS = PARENT_SPEC.model_settings
# Not an ``assert``: this companion is defined by reusing its parent's settings
# object, and ``-O`` would remove the one check that the parent still declares
# them typed (#637 stage 4).
if not isinstance(PARENT_SETTINGS, MediationModelSettings):
    raise TypeError(
        f"{PARENT_SPEC.model_id} must declare MediationModelSettings for this "
        f"companion to reuse; got {type(PARENT_SETTINGS).__name__}"
    )

SPEC = replace(
    PARENT_SPEC,
    model_id="lrp-rli-med-186",
    title=(
        "Interventional-effects decomposition: nonword reading (N, off-floor) "
        "via letter-sound knowledge (L) - companion to MED-086"
    ),
    adjustment=list(PARENT_SPEC.adjustment),
    model_settings=replace(
        PARENT_SETTINGS,
        estimand="interventional",
        companion_of=PARENT_SPEC.model_id,
    ),
)


def fit(config: str = "dev") -> StatisticalFitContext:
    return fit_mediation(SPEC, config=config)
