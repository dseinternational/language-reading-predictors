# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP58 - Mechanism model: letter-sound knowledge (L) -> word reading (W).

Revised-DAG (2026-07-10, #245) adjustment set. The revision drops EV->LS (SP is
now the speech->code cause), so the observed parents of LS are {A, HS, IG, IS, SP}:

- G, A: group (beta_G) and age (linear gamma_A).
- HS (hs / hs_missing): hearing, a new common cause of letter sounds.
- IS (attend): intervention sessions - a common cause of L and W.
- SP (deapp_c): speech production - now a cause of letter-sound knowledge.
- W_pre: autoregressive baseline.

The old E / R adjusters are REMOVED: under the revised DAG neither expressive nor
receptive vocabulary is a parent of LS, and conditioning on EV (a common effect of
SP and RW, which both also reach WR) would open a collider path. B (blending) is a
descendant of L and stays excluded, as do F and P.

GA (latent general ability) is NOT adjusted for, and the child random intercept does
**not** stand in for it. Under the usual random-effects independence assumption
``u_child`` captures stable residual between-child heterogeneity and the
repeated-measure dependence, but it does not block ``L <- GA -> W`` and does not
isolate a within-child exposure association: an omitted trait correlated with the
measured predictors is exactly what a zero-mean, predictor-independent random effect
cannot remove. Blocking that path would need a within/between decomposition,
correlated random effects (Mundlak terms), child fixed effects, or a genuine RI-CLPM.
None is fitted here.

So **residual confounding by GA remains**, and ``f^L`` is an **adjusted association**,
not a causal effect. The randomised causal claim lives in the ITT suite.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-058",
    kind="mechanism",
    title="Mechanism model: letter-sound knowledge (L) -> word reading (W)",
    outcome_symbol="W",
    mechanism_symbol="L",
    adjustment=["G", "A", "W_pre"],
    # Age GP off (age enters linearly), subject random intercept on.
    extra={
        "outcomes": ("W", "L"),
        "adjust_baseline_symbol": "W",
        "adjust_for": ("hs", "hs_missing", "attend", "deapp_c", "deapp_c_missing"),
        "use_age_gp": False,
        "phase_specific_mechanism": False,
        "use_subject_random_intercept": True,
        # A few boundary divergences survive at the reporting preset's 0.95 on an
        # otherwise-healthy posterior (R-hat 1.0, min ESS ~2400). Lift target_accept
        # for smaller steps near the boundary — the same legitimate response the
        # mm-001 fit uses, touching no adjusters. The HSGP mechanism curve is kept
        # (unlike LRP56/57, the letter-sound mechanism converges as a curve).
        "target_accept": 0.999,
    },
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
