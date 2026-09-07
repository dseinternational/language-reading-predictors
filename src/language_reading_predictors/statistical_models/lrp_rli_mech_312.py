# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP312 - composite-ability mechanism curve: letter sounds (L) -> word reading (W).

One-knob companion to **LRP258**, identical in rows, outcome, baseline,
adjustment set and priors, differing in exactly one thing: the measured-ability
adjuster is the two-subtest composite ``objass_c`` (WPPSI-III Block Design +
Object Assembly) rather than Block Design alone. Where the LRP306-311 panel does
this for the linear Tier-1 slopes, this does it for the family's **headline
estimand** - the flexible letter-sound curve of LRP58, whose interquartile
contrast is what the mechanism family publishes.

**Why.** LRP258 is the fit that answers "what does the curve look like once
measured ability is partialled out", and the answer on record is that it barely
moves: on the family's declared interquartile estimand the curve reads +2.7
items (89% +0.6 to +4.6) against +2.8 (+0.8 to +4.6) for the unadjusted LRP58.
That is a load-bearing robustness claim for the project's most-cited
association, and it currently rests on one noisy subtest — the mechanism
findings note makes the same point, that adjusting for a mismeasured confounder
removes only part of its influence. The two subtests correlate at 0.664, so
Block Design alone is roughly a 0.66-reliable measure of what they share while
the sum is roughly 0.80-reliable.

**Read.** Against LRP258, not against the unadjusted LRP58. The comparison to
make is whether the curve's declared interquartile contrast holds at the higher
reliability; a screening regression over the fitted rows expects it to, so
agreement is the result that closes the criticism. Because the curve is an HSGP
term, check the convergence gate before reading anything: LRP258's own history
shows this geometry is the demanding part of the family, and the ability term is
not what makes it hard.

**Ceilings.** A better-measured proxy is still a proxy. Both subtests are
perceptual-organisation tasks, so what they share is a **narrow visuospatial
factor** - the domain of relative strength in the Down syndrome profile - and not
the latent general ability ``GA`` of the causal diagram, which stays unmeasured
and structurally unblockable. The curve remains an **adjusted association** and
no adjustment available here makes it a causal skill-to-skill effect.
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
    model_id="lrp-rli-mech-312",
    kind="mechanism",
    title="Composite-ability mechanism curve: letter sounds (L) -> word reading (W)",
    outcome_symbol="W",
    mechanism_symbol="L",
    adjustment=["G", "A", "W_pre"],
    model_settings=MechanismModelSettings(
        outcomes=("W", "L"),
        adjust_baseline_symbol="W",
        adjust_for=("hs", "hs_missing", "attend", "deapp_c", "deapp_c_missing"),
        ability_covariate="objass_c",
        use_age_gp=False,
        phase_specific_mechanism=False,
        use_subject_random_intercept=True,
    ),
)


def fit(config: str = "dev") -> StatisticalFitContext:
    return fit_mechanism(SPEC, config=config)
