# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP70 - joint growth curves with a shared growth-tempo factor (companion to LRP69).

The genuinely *joint* layer of the issue #187 (Q5) trajectory model. It extends
LRP69 with a rank-1 shared child-level **growth-tempo factor** ``G_i ~ Normal(0, 1)``
that loads (positively, for identification) on every measure's slope::

    slope[i,k] = beta_k + gamma_k * z(blocks_i) + loading_k * G_i + sigma1_k * z1[i,k]

This asks two questions the independent-core LRP69 cannot:

1. **Do the measures grow together?** ``loading_k`` is how strongly measure k's
   growth rate tracks a common developmental tempo. Large, similar loadings mean a
   child who grows fast on one measure tends to grow fast on all.
2. **Does baseline non-verbal ability predict the common tempo?** Read out
   post-hoc as the per-draw correlation between each child's latent tempo ``G_i``
   and their standardised block-design score. ``G_tempo`` is independent a priori,
   but the posterior can still correlate ``G`` and ``blocks`` through the
   likelihood, so this is a descriptive post-hoc read-out, not a guaranteed
   orthogonal decomposition. ``gamma_k`` remains the per-measure block-design ->
   slope association, now *incremental* to the tempo.

**Still an adjusted, GA-confounded association, never causal** - the same DAG
caveats as LRP69 apply (``notes/202606231600-dag-revision-consolidated.md``,
``METHODS.md``). The shared factor is a parsimonious rank-1 stand-in for
cross-measure slope covariation, **not** an identified general-ability construct.

``LOO(LRP69 vs LRP70)`` is the test of whether the shared tempo earns its keep at
n~54; if LRP70 will not sample cleanly, LRP69 (independent-core) is the primary
deliverable and the factor is reported as not estimable at this sample size (as the
RI-CLPM companion was for LRP67).
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_growth

SPEC = ModelSpec(
    model_id="lrp-rli-gc-070",
    kind="growth",
    title=(
        "Joint multivariate growth curves with a shared growth-tempo factor: "
        "do verbal/reading trajectories grow together, and does baseline "
        "non-verbal ability predict the common tempo?"
    ),
    outcome_symbol=None,
    extra={
        "outcomes": ["R", "E", "T", "W", "L"],
        "baseline_covariate": "blocks",
        # Factor layer: add the rank-1 shared growth-tempo factor G_i.
        "use_shared_factor": True,
    },
    study_id="rli",
    family="growth",
    design="observational_longitudinal",
    estimand_type="descriptive",
    causal_status="adjusted",
)


def fit(config: str = "dev"):
    return fit_growth(SPEC, config=config)
