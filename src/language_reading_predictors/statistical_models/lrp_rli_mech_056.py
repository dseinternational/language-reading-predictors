# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP56 - Mechanism model: receptive vocabulary (R) -> word reading (W).

Revised-DAG (2026-07-10, #245) adjustment set for the R_post -> W_post backdoor.
The observed parents of RV are {A, HS, RW, TR}, so:

- TR: taught receptive vocabulary (parent of RV, cause of WR) - measure confounder.
- HS: hearing status (hs / hs_missing) - covariate adjuster (new HS edges).
- RW: phonological memory (erbto) - covariate adjuster (new RW edges).
- A (age): linear ``gamma_A`` term. G (group) is always controlled via beta_G;
  W_pre is the autoregressive baseline.

E is a *descendant* of R (R -> E) that also affects W. Conditioning on E
would block the legitimate indirect path R -> E -> W and bias f^R toward
zero. E is therefore deliberately NOT in the adjustment set. GA (general ability)
is latent and unadjustable - the child random intercept proxies its time-invariant
part, so f^R stays an adjusted association, not a causal effect.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-056",
    kind="mechanism",
    title="Mechanism model: receptive vocabulary (R) -> word reading (W)",
    outcome_symbol="W",
    mechanism_symbol="R",
    adjustment=["G", "A", "TR", "W_pre"],
    # Age enters as a linear gamma_A term (A is a declared confounder). The subject
    # random intercept (on by default) handles the 157 non-independent rows (up to 3
    # phases × 53 children) and proxies the time-invariant part of latent ability.
    extra={
        # Load the exposure (R), outcome (W) and the TR measure confounder so TR's
        # baseline is available; the complete-case mask ignores unused measures.
        "outcomes": ("W", "R", "TR"),
        "adjust_baseline_symbol": "W",
        "adjust_for": ("hs", "hs_missing", "erbto", "erbto_missing"),
        "use_age_gp": False,
        "phase_specific_mechanism": False,
        "use_subject_random_intercept": True,
        # LINEAR mechanism, not the HSGP curve. At reporting tier the nonparametric
        # f_mech(R) curve does not converge here (~300 divergences on an otherwise
        # healthy posterior — R-hat 1.0, min ESS ~2600 — and raising target_accept
        # to 0.99 barely moves it, so it is HSGP geometry, not boundary steps). The
        # revised DAG adjusters are DAG-required and are NOT dropped to buy
        # convergence (#258 review). Instead the mechanism enters linearly, which
        # converges cleanly (0 divergences), following the review's own "consider a
        # linear mechanism" option. The estimand is thereby the LINEAR R -> W
        # adjusted association (a single slope), not the nonparametric shape;
        # recovering the curve would need a reparameterised GP and is left as
        # follow-up. LRP58 (L -> W) keeps its HSGP curve — only the vocabulary-
        # predictor mechanisms show this pathology.
        "linear_mechanism": True,
    },
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
