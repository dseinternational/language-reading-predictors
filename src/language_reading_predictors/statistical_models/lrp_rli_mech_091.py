# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP91 - GP knee-test: phoneme blending / phonological awareness (B) -> word reading (W).

A NEW mechanism model (no prior blending -> reading fit existed) built to TEST whether
phoneme blending shows a "knee" - a level of blending skill beyond which it is
associated with a more marked difference in word reading - the way LRP58 found for
letter sounds. The mechanism enters as an HSGP curve on the logit-safe transform of the
blending post-score; target_accept is 0.999 (per LRP58). Blending is a small bounded
count (n = 10), so the curve is demanding at this sample size; a divergence is itself an
honest result and would leave the knee untestable.

Adjustment set (revised DAG, 2026-07-10). Derived by a backdoor d-separation search with
the latent GA held (the criterion that reproduces LRP56/58 and dose-077). In the DAG,
PA (blending) has parents {A, GA, HS, IG, IS, TE, EV, SP, LS, RW}; the minimal observed
set that blocks every backdoor to WR is {A, HS, IG, IS, LS, TE, EV, SP, RW}. Crucially
this includes LETTER SOUNDS (LS = L): LS -> PA and LS -> WR make it a PA <- LS -> WR
confounder, so a blending -> reading association must be read *net of* letter-sound
knowledge. Measure confounders L, TE, E enter on their logit scale via ``adjustment``;
the continuous confounders HS (hs), IS (attend), SP (deapp_c) and RW (erbto) enter via
``adjust_for``; group G(=IG) is the always-in precision term and W_pre the
autoregressive baseline. The causal paths PA -> WR and PA -> NW -> WR are preserved
(NW, PS are descendants of PA and are never adjusted).

Residual confounding by latent general ability (GA) remains, so f^B is an ADJUSTED
ASSOCIATION, not a causal effect. The randomised causal claim lives in the ITT suite.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-091",
    kind="mechanism",
    title="GP knee-test: phoneme blending (B) -> word reading (W)",
    outcome_symbol="W",
    mechanism_symbol="B",
    adjustment=["G", "A", "L", "TE", "E", "W_pre"],
    extra={
        "outcomes": ("W", "B", "L", "TE", "E"),
        "adjust_baseline_symbol": "W",
        "adjust_for": (
            "hs", "hs_missing", "attend", "deapp_c", "deapp_c_missing",
            "erbto", "erbto_missing",
        ),
        "use_age_gp": False,
        "phase_specific_mechanism": False,
        "use_subject_random_intercept": True,
        # HSGP mechanism curve ON (knee-test); target_accept 0.999 per LRP58.
        "target_accept": 0.999,
    },
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
