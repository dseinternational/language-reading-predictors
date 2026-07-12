# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP57 - Mechanism model: expressive vocabulary (E) -> word reading (W).

Revised-DAG (2026-07-10, #245) adjustment set. Observed parents of EV are
{A, HS, TR, TE, RV(R), RW, SP}, so:

- G, A: group (beta_G) and age (linear gamma_A).
- TR, TE, R: taught receptive/expressive and standardised receptive vocabulary -
  parents of EV that also cause WR (measure confounders).
- HS (hs / hs_missing), RW (erbto), SP (deapp_c): covariate adjusters newly
  required by the revised DAG.
- W_pre: autoregressive baseline.

F, L, P, B are descendants of E or lie on post-E mediation paths and are
excluded.

GA (latent general ability) is NOT adjusted for, and the child random intercept does
**not** stand in for it: a zero-mean, predictor-independent random effect captures
stable residual heterogeneity and repeated-measure dependence, but cannot block
``E <- GA -> W`` or isolate a within-child association. That would need a
within/between decomposition, Mundlak terms, child fixed effects or an RI-CLPM, none
of which is fitted here. **Residual confounding by GA remains**, and f^E is an
**adjusted association**, not a causal effect.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-057",
    kind="mechanism",
    title="Mechanism model: expressive vocabulary (E) -> word reading (W)",
    outcome_symbol="W",
    mechanism_symbol="E",
    adjustment=["G", "A", "TR", "TE", "R", "W_pre"],
    # Age GP off (age enters linearly), subject random intercept on.
    extra={
        "outcomes": ("W", "E", "R", "TR", "TE"),
        "adjust_baseline_symbol": "W",
        "adjust_for": (
            "hs", "hs_missing", "erbto", "erbto_missing", "deapp_c", "deapp_c_missing",
        ),
        "use_age_gp": False,
        "phase_specific_mechanism": False,
        "use_subject_random_intercept": True,
        # LINEAR mechanism, not the HSGP curve — see the matching note in LRP56. The
        # nonparametric f_mech(E) curve does not converge at reporting tier (~200
        # divergences, unmoved by target_accept 0.99); the DAG-required adjusters
        # are kept and the mechanism enters linearly instead, which converges
        # cleanly (0 divergences), per the #258 review. The estimand is the LINEAR
        # E -> W adjusted association.
        "linear_mechanism": True,
    },
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
