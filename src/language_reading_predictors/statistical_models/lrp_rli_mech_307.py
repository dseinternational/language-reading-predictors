# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP307 - composite-ability Tier-1 panel: letter sounds (L) -> receptive vocabulary (R).

One-knob companion to **LRP197**, identical in rows, outcome, baseline,
adjustment set and priors, differing in exactly one thing: the measured-ability
adjuster is the two-subtest composite ``objass_c`` (WPPSI-III Block Design +
Object Assembly) rather than Block Design alone.

**Why.** The Block Design adjustment uses one noisy subtest. The two
subtests correlate at 0.664 over the 54 analysed children, but that correlation
does not by itself establish either subtest's reliability or the composite's
reliability. The composite offers another measured-ability proxy for a
sensitivity comparison.

**Read.** Compare with LRP197, which uses Block Design on the same
rows. Agreement or a shift describes sensitivity to the ability proxy. Neither
establishes that latent ability has been controlled, that subtest noise caused
a shift, or that a criticism of residual confounding is resolved.

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
    model_id="lrp-rli-mech-307",
    kind="mechanism",
    title="Composite-ability Tier-1 panel: letter sounds (L) -> receptive vocabulary (R)",
    outcome_symbol="R",
    mechanism_symbol="L",
    adjustment=["G", "A", "R_pre"],
    model_settings=MechanismModelSettings(
        adjust_baseline_symbol="R",
        outcomes=("R", "L"),
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
