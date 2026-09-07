# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP308 - composite-ability Tier-1 panel: letter sounds (L) -> expressive vocabulary (E).

One-knob companion to **LRP198**, identical in rows, outcome, baseline,
adjustment set and priors, differing in exactly one thing: the measured-ability
adjuster is the two-subtest composite ``objass_c`` (WPPSI-III Block Design +
Object Assembly) rather than Block Design alone.

**Why.** The ability-adjusted panel LRP196-201 is the evidence for "the
letter-sound association is not merely measured ability", and its obvious
rebuttal is that it adjusts for one noisy subtest. The two subtests correlate at
0.664 over the 54 analysed children, so Block Design alone is roughly a
0.66-reliable measure of what they share while the sum is roughly 0.80-reliable.
This panel re-reads the same comparison with the more reliable measure, so the
rebuttal has a fitted answer rather than an argument.

**Read.** Against LRP198, not against the unadjusted LRP098. A slope that
holds up here having held up there is adjusted for the shared visuospatial
variance about as well as this battery permits. A slope that shrinks here but not
there was carrying subtest-specific noise. A screening regression over the fitted
rows put the difference at well under a tenth of a standard error for the
letter-sound slope, so **agreement is the expected result** and is the point: it
closes a line of criticism rather than reporting a new effect.

**Ceilings.** A better-measured proxy is still a proxy. Both subtests are
perceptual-organisation tasks, so what they share is a **narrow visuospatial
factor** - the domain of relative strength in the Down syndrome profile - and not
the latent general ability ``GA`` of the causal diagram, which stays unmeasured
and structurally unblockable. Adjusting for the composite therefore does not move
``beta_mech`` any closer to a causal quantity: it remains an **adjusted
association**. Role in the panel: an oral-language negative control.
"""

from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
)
from language_reading_predictors.statistical_models.mechanism import (
    MechanismModelSettings,
)
from language_reading_predictors.statistical_models.pipelines.mechanism import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-308",
    kind="mechanism",
    title="Composite-ability Tier-1 panel: letter sounds (L) -> expressive vocabulary (E)",
    outcome_symbol="E",
    mechanism_symbol="L",
    adjustment=["G", "A", "E_pre"],
    model_settings=MechanismModelSettings(
        adjust_baseline_symbol="E",
        outcomes=("E", "L"),
        adjust_for=("hs", "hs_missing", "attend", "deapp_c", "deapp_c_missing"),
        ability_covariate="objass_c",
        linear_mechanism=True,
        use_age_gp=False,
        phase_specific_mechanism=False,
        use_subject_random_intercept=True,
    ),
)


def fit(config: str = "dev") -> StatisticalFitContext:
    return fit_mechanism(SPEC, config=config)
