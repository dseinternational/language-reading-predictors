# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP301 - between/within (Mundlak) split of the L -> W linear slope (#603).

Companion to **LRP101**, identical in rows, outcome, baseline, adjustment set and
priors, differing in exactly one thing: the standardised letter-sound exposure is
split into each child's **fitted-row average** and their **deviation from it**, so
the model fits ``beta_between`` and ``beta_within`` instead of one ``beta_mech``.

**Why.** Every other mechanism fit reports one exposure coefficient estimated over a
child random intercept. That coefficient is a precision-weighted *blend* of two
associations that answer different questions:

- **between children** - do children who generally know more letter sounds generally
  read more words?
- **within a child** - when a child's own letter-sound score moves, does their word
  reading move with it?

The random intercept does not separate them. It models repeated-measures dependence
under an independence assumption, is not permitted to correlate with the exposure,
and does not decompose the exposure into a child mean and a deviation from it - so
the weights are set by the variance ratio rather than by the question. On these data
the two are far apart: the ``pooled_levels`` family, which already runs this split,
measures r = **0.81 between** against **0.45 within** for letter sounds and word
reading on the logit scale (0.70 against 0.51 on raw counts). A blend of 0.81 and
0.45 is not a good answer to either question.

The developmental reading of this family - "children who know more letter sounds read
more words, and here is whether that relationship bends" - is naturally a
**within-child** claim, and that is the component the pooled parameterisation is least
able to isolate.

**What the split does and does not buy.** ``beta_within`` removes *stable*
between-child confounding, including the stable part of latent general ability. It
does **not** make the exposure temporally prior to the outcome (both are still
measured at the same wave), and it does not remove time-varying confounding or
reverse causation. The result is a better-posed association, not an identified
effect; ``beta_between`` is a frankly cross-sectional comparison and is confounded
by every stable child characteristic.

Linear only by design: a between/within split of a nonparametric curve is a larger
design question, and the linear split is the one that answers the estimand question
cleanly (#603 scope). The family's headline natural-scale contrast on this fit is
therefore the **within-child** items contrast - moving one wave's exposure while
holding that child's study average fixed - because the between term cancels from any
exposure contrast.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.mechanism import (
    MechanismModelSettings,
)
from language_reading_predictors.statistical_models.pipelines.mechanism import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-301",
    kind="mechanism",
    title="Between/within split: letter sounds (L) -> word reading (W)",
    outcome_symbol="W",
    mechanism_symbol="L",
    adjustment=["G", "A", "W_pre"],
    model_settings=MechanismModelSettings(
        adjust_baseline_symbol="W",
        outcomes=("W", "L"),
        adjust_for=("hs", "hs_missing", "attend", "deapp_c", "deapp_c_missing"),
        linear_mechanism=True,
        decompose_between_within=True,
        use_age_gp=False,
        use_subject_random_intercept=True,
    ),
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
