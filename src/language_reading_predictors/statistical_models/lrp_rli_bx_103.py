# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPBX103 - wide-delta prior-sensitivity companion to LRPBX03 (UE2).

The distal-tier tightening (#141: broad standardised-transfer outcomes take
``tau ~ Normal(0, 0.3)`` rather than the proximal 0.5) also prices the block-2
not-taught expressive comparator's focal exposure effect: LRPBX03's ``delta``
runs at ``Normal(0, 0.3)`` because UE2 is distal. The prior-critical review
(`notes/202607211500-prior-critical-review.md`, #382 recommendation 4) asks for
a single ``Normal(0, 0.5)`` check on the distal tier to confirm that
tightening is not attenuating a real transfer effect. The ITT distal outcomes
(R/E/UR/UE/T/F) are certified by the standard 44-cell treatment-prior sweep,
and the aligned family's ``beta_cohort`` was never tiered (it is an
adjusted-association cohort contrast held at 0.5), which leaves the
block-exposure UE2 comparator as the one distal focal term without sensitivity
evidence.

LRPBX103 is **identical to LRPBX03 except** ``delta``, which takes
``Normal(0, 0.5)`` via the factory's ``delta_prior_sigma`` override. Everything
else — the staggered block-active exposure identification, the ability
covariate, the observed adjustment set, and the child random intercept —
matches, so any difference in the exposure effect is attributable to the tier
prior alone.

Same reading rules as LRPBX03: block 2 has no t1 baseline and no randomised
contrast, so ``delta`` is an **adjusted association** whichever prior it takes;
the not-taught comparators are expected near zero, and a null here is the
design behaving, not a power failure.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.block_exposure import (
    BlockExposureModelSettings,
)
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_block_exposure

SPEC = ModelSpec(
    model_id="lrp-rli-bx-103",
    kind="block_exposure",
    title=(
        "Wide-delta prior sensitivity for block-2 not-taught expressive "
        "vocabulary (UE2)"
    ),
    outcome_symbol="UE2",
    model_settings=BlockExposureModelSettings(
        # Identical to LRPBX03 in every respect except delta_prior_sigma.
        ability_covariate=V.BLOCKS,
        adjust_for=("hs", "hs_missing", "deapp_c", "deapp_c_missing", "erbto", "erbto_missing"),
        use_child_re=True,
        # The single free variable (#382 rec 4): the focal exposure effect's
        # prior widens from the distal-tier Normal(0, 0.3) to Normal(0, 0.5).
        delta_prior_sigma=0.5,
    ),
)


def fit(config: str = "dev"):
    return fit_block_exposure(SPEC, config=config)
