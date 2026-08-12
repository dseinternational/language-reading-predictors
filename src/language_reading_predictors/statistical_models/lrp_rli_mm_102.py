# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPMM102 - focal-slope prior-sensitivity companion to the LRPMM02 EiV mechanism.

LRPMM02's structural leg regresses word-reading gain on the **latent code factor**
with the mech-058 observed adjustment set, and its focal slope ``beta_code``
carries the association-scale ``Normal(0, 0.3)`` prior every adjusted-association
slope in the suite takes. The prior-critical review
(`notes/202607211500-prior-critical-review.md`, #382 recommendation 1) flagged
that as the one place a headline prior is plausibly **under-scaled**: the
code->word slope is the suite's documented primary mechanism, and the linear
mechanism factory prices that role at ``Normal(0, 1)``. Whether ``N(0, 0.3)``
attenuates the EiV estimate was untested — the earlier companion (LRPMM101)
varied only the measurement-layer priors, never the structural slope.

LRPMM102 is **identical to LRPMM02 except** ``beta_code``, which takes
``Normal(0, 1)`` via the factory's ``focal_slope_sigma`` knob. Everything else —
the communality-scale measurement layer (#383), the mech-058 covariate set on
the association scale, the data, and ``target_accept = 0.999`` — matches, so any
difference in the code->word slope is attributable to the focal prior alone.
``beta_G`` and the precision/covariate slopes (``beta_age``, hearing, speech,
phonological memory and their missingness indicators) deliberately keep
``Normal(0, 0.3)``: recommendation 1 explicitly keeps the arm covariate on the
association scale (it is an adjustment term, not the mechanism), and widening
the whole set would confound the answer.

Read the comparison the way the review framed it: if the LRPMM02 posterior is
genuinely data-dominated, the slope here should match it to within Monte-Carlo
error; a materially larger slope under the wider prior would mean the reported
estimate is prior-attenuated and the association scale is doing quiet work.

Same caveats as LRPMM02: a **measurement / triangulation** model, not causal. Per
ID-2 the code->W slope is a latent-ability-confounded **adjusted association**,
and ``beta_G`` is an adjusted-association covariate, not the available-case
modified ITT estimate.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.corr_factor import (
    CorrFactorModelSettings,
)
from language_reading_predictors.statistical_models.pipeline import fit_correlated_factor

SPEC = ModelSpec(
    model_id="lrp-rli-mm-102",
    kind="corr_factor",
    title=(
        "Focal-slope prior sensitivity for the errors-in-variables code->word "
        "mechanism (beta_code at Normal(0, 1))"
    ),
    outcome_symbol="W",
    model_settings=CorrFactorModelSettings(
        domains=(
            ("vocabulary", ("R", "E")),
            ("code", ("L", "B")),
            ("grammar", ("F", "T")),
        ),
        structural_factors=("code",),
        use_group=True,
        use_age=True,
        structural_covariates=(
            "hs",
            "hs_missing",
            "deapp_c",
            "deapp_c_missing",
            "erbto",
            "erbto_missing",
        ),
        focal_slope_sigma=1.0,
    ),
    extra={
        # Identical to LRPMM02 in every respect except focal_slope_sigma.
        # The single free variable: the focal beta_code prior moves from the
        # association-scale N(0, 0.3) to the primary-mechanism N(0, 1); beta_G
        # deliberately stays at the association scale (recommendation 1).
        "target_accept": 0.999,
    },
)


def fit(config: str = "dev"):
    return fit_correlated_factor(SPEC, config=config)
