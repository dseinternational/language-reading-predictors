# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPITT12 - Factorised multi-outcome ITT model (migrates LRP55).

One PyMC graph over the single-outcome treatment effects, acting as a consistency
check on the single-outcome LRPITT models. With residual correlation disabled,
the likelihood and priors factorise by outcome: it is a product of marginal
models, not a dependence-aware joint posterior. Cross-outcome comparisons are
therefore reported on a common probability scale with a paired-dependence caveat.

Outcomes are the ten **baseline-bearing** LRPITT measures. Nonword (N) is excluded:
it is post-only and its degenerate baseline cannot enter the joint's dense
pre-score matrix without injecting NaNs or polluting the cross-baseline block — its
effect is read from the single-outcome off-floor model LRPITT11. Phonetic spelling
(P) **is** included, as a graded outcome (its floored baseline simply shrinks
gamma_own[P]); its binary off-floor exploratory analysis is in LRPITT09. LKJ residual
correlation is off by default (prior-dominated at 8 outcomes, worse at 10; keep the
flag for a sensitivity fit). Sign convention: positive tau => intervention helps.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_joint

# The ten baseline-bearing suite outcomes (LRPITT_OUTCOMES minus post-only N).
LRPITT12_OUTCOMES: tuple[str, ...] = (
    "TR", "TE", "UR", "UE", "R", "E", "L", "B", "P", "W",
)

SPEC = ModelSpec(
    model_id="lrp-rli-itt-012",
    kind="joint",
    title="Joint ITT model over the LRPITT suite outcomes (TR, TE, UR, UE, R, E, L, B, P, W)",
    extra={
        "outcomes": LRPITT12_OUTCOMES,
        # DAG-faithful, mirroring the single-outcome suite's *structure*: own
        # baseline + linear age as precision terms, no cross-baselines. NOTE the
        # joint model keeps the common Normal(0, 0.5) tau prior for every outcome;
        # it does NOT apply the single-outcome suite's distal tier
        # (Normal(0, 0.3) for measures.DISTAL_OUTCOMES = {R, E, T, F, UR, UE}; see
        # PRIORS.md). So the joint tau_k track the single-outcome tau_k closely on
        # the proximal outcomes, but a distal tau_k may differ from its tiered
        # single-outcome counterpart by a prior-driven amount. LKJ residual + age
        # GP off.
        "use_cross_baselines": False,
        "use_age_linear": True,
        "use_age_gp": False,
        "use_residual_correlation": False,
        "joint_structure": "factorised_outcome_marginals",
        "loo_unit": "child",
    },
)


def fit(config: str = "dev"):
    return fit_joint(SPEC, config=config)
