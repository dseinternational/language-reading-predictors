# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP69 - joint multivariate growth curves: does baseline non-verbal ability predict verbal/reading trajectory shape?

The descriptive, longitudinal answer to issue #187 (Q5). A joint latent
growth-curve model over the four RLI waves for the five verbal / reading measures
- receptive vocabulary (``R`` = ROWPVT), expressive vocabulary (``E`` = EOWPVT),
receptive grammar (``T`` = TROG-2), word reading (``W`` = EWRSWR) and letter-sound
knowledge (``L`` = YARC-LSK) - each on the logit scale via a masked Beta-Binomial
(see :func:`factories.build_growth_model`)::

    theta[i,t,k]   = intercept[i,k] + slope[i,k] * age_std[i,t]
    intercept[i,k] = alpha_k + delta_k * z(blocks_i) + sigma0_k * z0[i,k]
    slope[i,k]     = beta_k  + gamma_k * z(blocks_i) + sigma1_k * z1[i,k]

**Baseline non-verbal ability** (``blocks``, the t1-only WPPSI Block Design score,
complete for all 54 children) enters as a per-child, standardised predictor of
trajectory shape: ``gamma_k`` on the growth *rate* is the headline Q5 estimand -
"does a child's baseline non-verbal ability predict how fast measure k grows?" -
and ``delta_k`` is the effect on the *level at the sample-mean (mid-study) age* —
``age_std`` is standardised over all child-wave cells, so the intercept is the
level at the pooled mean age, NOT at study entry. The entry-level association is
``delta_k + gamma_k * E[age_std at t1]`` (and ``E[age_std at t1] < 0``).

**This is an adjusted, GA-confounded association, never causal.** Per the locked
DAG (``dag/dag-language-reading.dagitty``) block design is an
off-DAG ability proxy and latent general ability is the unobserved common cause;
the child random intercept only *partially* adjusts (``METHODS.md`` § "Causal
interpretation and its limits"). Report ``gamma_k`` / ``delta_k`` as adjusted
associations, never "non-verbal ability drives growth".

The child-level random intercept and slope are independent per measure - the
within-measure intercept-slope correlation is deliberately omitted at n~54,
mirroring the joint ITT model's disabled LKJ residual correlation. LRP70 is the
companion that adds a shared growth-tempo factor (the genuinely joint layer);
``LOO(LRP69 vs LRP70)`` shows whether that extra structure earns its keep.

Exploratory at n~54: intervals are wide. The deliverable is the *direction* and
rough magnitude of ``gamma_k`` per measure, framed as an adjusted association.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_growth

SPEC = ModelSpec(
    model_id="lrp-rli-gc-069",
    kind="growth",
    title=(
        "Joint multivariate growth curves: does baseline non-verbal ability "
        "predict verbal/reading trajectory shape? (independent-core)"
    ),
    outcome_symbol=None,
    extra={
        # Verbal / reading measures whose trajectories are characterised.
        "outcomes": ["R", "E", "T", "W", "L"],
        # Time-invariant baseline covariate entering as a predictor of shape.
        "baseline_covariate": "blocks",
        # Core model: no shared growth-tempo factor (see LRP70 for that layer).
        "use_shared_factor": False,
    },
    study_id="rli",
    family="growth",
    design="observational_longitudinal",
    estimand_type="descriptive",
    causal_status="adjusted",
)


def fit(config: str = "dev"):
    return fit_growth(SPEC, config=config)
