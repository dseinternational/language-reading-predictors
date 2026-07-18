# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP188 - GP knee-test: taught receptive vocabulary (TR) -> word reading (W).

Nonparametric (HSGP) re-attempt of the LRP88 mechanism, built to TEST whether the
TR -> W association has a "knee" the way LRP58 found for letter sounds. LRP88 was forced
to a LINEAR mechanism on the vocabulary-exposure precedent (LRP56/57), so its estimand
is a single slope, not a shape. This variant switches the HSGP curve back on and lifts
target_accept to 0.999 (per LRP58); a persisting divergence is itself the honest result
and leaves the knee untestable for this exposure.

Adjustment set (revised DAG, 2026-07-10). Re-derived by a backdoor d-separation search
with the latent GA held: TR's observed confounders that also reach WR are
{A, HS, IG, IS, RW}. NB this ADDS intervention sessions (IS = attend) relative to LRP88,
which omitted it: IS -> TR and IS -> WR make sessions a genuine confounder of TR -> W
(the same adjuster LRP58 already carries for L -> W). Group G(=IG) is the always-in
precision term and W_pre the autoregressive baseline.

Residual confounding by latent general ability (GA) remains, so f^TR is an ADJUSTED
ASSOCIATION, not a causal effect. The randomised causal claim lives in the ITT suite.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-188",
    kind="mechanism",
    title="GP knee-test: taught receptive vocabulary (TR) -> word reading (W)",
    outcome_symbol="W",
    mechanism_symbol="TR",
    adjustment=["G", "A", "W_pre"],
    extra={
        "outcomes": ("W", "TR"),
        "adjust_baseline_symbol": "W",
        "adjust_for": ("hs", "hs_missing", "attend", "erbto", "erbto_missing"),
        "use_age_gp": False,
        "phase_specific_mechanism": False,
        "use_subject_random_intercept": True,
        # HSGP mechanism curve ON (knee-test); target_accept 0.999 per LRP58.
        "target_accept": 0.999,
    },
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
