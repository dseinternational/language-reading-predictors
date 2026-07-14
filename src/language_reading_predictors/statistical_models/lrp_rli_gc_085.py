# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP85 - growth curves with an age x ability interaction on the growth rate (#228 item 10).

The dedicated growth-curve vehicle for the gain-factor finding that the
**age x ability** interaction is positive in 6 of 8 outcomes -- very strong for
basic concepts (+0.20, P = 0.99) and grammar (+0.14, P = 1.00) -- i.e.
*older-and-more-able children progress more than age and ability predict
separately*. It extends the independent-core LRP69 (:func:`factories.build_growth_model`)
with a child-level **baseline (t1) age** moderator on the growth slope::

    slope[i,k] = beta_k + gamma_k * z(blocks_i)          # ability -> rate (as LRP69)
                        + gamma_age_k * age0_i            # baseline age -> rate
                        + gamma_int_k * (age0_i * z(blocks_i))   # <-- headline
                        + sigma1_k * z1[i,k]

where ``age0_i`` is the child's first-wave age standardised **across children**
(distinct from the within-child ``age_std`` time axis that the slope multiplies).
``gamma_int_k`` is the headline: positive = a child who is both older at baseline
and more able grows faster on measure k than the two main effects predict, bringing
the gain factors' ``gamma_int_A_ability`` onto the latent growth rate.

**Still an adjusted, GA-confounded association, never causal.** Block design is an
off-DAG ability proxy and latent general ability is the unobserved common cause; the
child random intercept only partially adjusts (``dag/dag-language-reading.dagitty``,
``METHODS.md``). Baseline age is likewise not randomised. Read ``gamma_int_k`` /
``gamma_age_k`` / ``gamma_k`` as adjusted associations. Exploratory at n~54:
intervals are wide; the deliverable is the *direction* of ``gamma_int_k`` per
measure. The growth family models the five verbal/reading trajectories
(``R``/``E``/``T``/``W``/``L``); concepts (F) and blending (B) — where the gain
factors' age x ability signal was strongest — are **not** growth-curve outcomes, so
this tests the interaction only on those five and cannot directly check the F/B
signals. Expect a mostly-positive pattern echoing the gain factors' 6-of-8
direction. Independent-core (no shared growth-tempo factor); a factor companion
could follow, as LRP70 does for LRP69.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_growth

SPEC = ModelSpec(
    model_id="lrp-rli-gc-085",
    kind="growth",
    title=(
        "Growth curves with an age x ability interaction on the growth rate: "
        "do older-and-more-able children progress more than age and ability "
        "predict separately?"
    ),
    outcome_symbol=None,
    extra={
        "outcomes": ["R", "E", "T", "W", "L"],
        "baseline_covariate": "blocks",
        "use_shared_factor": False,
        # LRP85 layer: baseline-age main effect + age0 x ability interaction on the slope.
        "age_ability_interaction": True,
    },
    study_id="rli",
    family="growth",
    design="observational_longitudinal",
    estimand_type="descriptive",
    causal_status="adjusted",
)


def fit(config: str = "dev"):
    return fit_growth(SPEC, config=config)
