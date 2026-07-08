# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPITT12 - Joint ITT model over the suite outcomes (migrates LRP55).

One posterior over the single-outcome treatment effects, enabling cross-outcome
contrasts ("is tau_L more positive than tau_W?") and acting as a consistency check
on the single-outcome LRPITT models.

Outcomes are the ten **baseline-bearing** LRPITT measures. Nonword (N) is excluded:
it is post-only and its degenerate baseline cannot enter the joint's dense
pre-score matrix without injecting NaNs or polluting the cross-baseline block — its
effect is read from the single-outcome off-floor model LRPITT11. Phonetic spelling
(P) **is** included, as a graded outcome (its floored baseline simply shrinks
gamma_own[P]); its binary off-floor primary is in LRPITT09. LKJ residual
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
    },
)


def fit(config: str = "dev"):
    return fit_joint(SPEC, config=config)
