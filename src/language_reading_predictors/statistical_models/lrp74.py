# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP74 - ITT analysis of taught expressive vocabulary (block 1).

The headline intervention-fidelity ("positive control") analysis: does randomised
group assignment raise the directly-taught *expressive* vocabulary (the Block 1
target words, ``b1extau``) in the randomised window (t1 -> t2)? The published
trial (Burgoyne et al., 2012) found a significant gain on exactly this measure
while the standardised expressive-vocabulary test (LRP54) did not move - so this
is the contrast that lets the suite speak to the trial's vocabulary result.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_itt

SPEC = ModelSpec(
    model_id="lrp74",
    kind="itt",
    title="ITT effect of group assignment on taught expressive vocabulary (block 1)",
    outcome_symbol="TE",
    # ``TE`` (b1extau) is outside ITT_OUTCOMES, so the prepared outcome set and
    # the cross-baseline set are given explicitly. The outcome is conditioned on
    # its own taught-vocabulary baseline (gamma_own) plus the standardised
    # expressive-vocabulary baseline ``E`` - so tau reads as the effect on the
    # directly-taught words over and above general expressive-vocabulary level -
    # rather than on all eight ITT baselines (parsimony, and no extra dropped
    # cases, at n~54). HSGPs off by default, matching LRP52-LRP54.
    extra={
        "outcomes": ("TE", "E"),
        "cross_symbols": ("E",),
        "use_age_gp": False,
        "use_own_baseline_gp": False,
        "use_varying_tau": False,
    },
)


def fit(config: str = "dev"):
    return fit_itt(SPEC, config=config)
