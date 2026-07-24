# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPJM01 - joint bivariate mechanism: letter sounds -> {word reading, nonword decoding}.

#421 Tier 3 (letter-sound -> word-reading review note #424; decoding-specificity spec
202607172330 line 46). The review's decoding-specificity finding -- letter sounds predict
*pure decoding* (nonword reading, N) far more strongly than *mixed* word reading (W),
the signature that letter-sound knowledge is genuinely converted into decoding -- was
until now only a **product-of-marginals sensitivity**: the contrast delta = beta(LS->N) -
beta(LS->W) was assembled by pairing draws from two *separate* mechanism fits
(``mech-096`` and ``mech-101``) under a working-independence assumption, so its interval
was not an identified posterior (PR #359 review).

This model makes that contrast **identified**. It stacks both outcomes on the *same*
standardised letter-sound exposure with a per-outcome linear slope, while a **single**
child random intercept -- the best available latent-general-ability proxy -- is
partialled from **both** slopes. ``delta_ls_decoding = beta_mech[N] - beta_mech[W]`` is
then a within-model deterministic with the true joint-posterior cross-outcome covariance
baked in. Each outcome keeps its own autoregressive baseline (``W_pre`` / ``N_pre``) and
its own Beta-Binomial denominator (79 vs 6 items), so the floored 6-item nonword count is
never pooled with word reading.

**Adjustment set** = the ``mech-058`` / ``mech-101`` letter-sound -> word-reading
confounders {G, A, HS, IS(``attend``), SP(``deapp_c``)} + each outcome's own baseline,
per outcome. Every slope is an **adjusted association**, latent general ability only
partially proxied by the shared intercept -- never causal. The contrast is a
Campbell-Fiske convergent/discriminant argument (a pure-GA account gives no reason for LS
to predict N more than W), not an identification of a causal decoding effect.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_joint_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-jm-001",
    kind="joint_mechanism",
    title=(
        "Joint bivariate mechanism: letter sounds -> {word reading (W), "
        "nonword decoding (N)} with an identified decoding-specificity contrast"
    ),
    mechanism_symbol="L",
    adjustment=["G", "A"],
    extra={
        "outcome_symbols": ("W", "N"),
        "contrast": ("N", "W"),
        "adjust_for": ("hs", "hs_missing", "attend", "deapp_c", "deapp_c_missing"),
        "confounder_symbols": ("G", "A"),
    },
)


def fit(config: str = "dev"):
    return fit_joint_mechanism(SPEC, config=config)
