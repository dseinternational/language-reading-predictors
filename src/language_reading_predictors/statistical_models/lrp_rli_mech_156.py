# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP156 - GP knee-test: receptive vocabulary (R) -> word reading (W).

Nonparametric (HSGP) re-attempt of the LRP56 mechanism, built to TEST whether the
R -> W association has a "knee" - a level of receptive vocabulary beyond which it is
associated with a more marked difference in word reading - the way LRP58 found for
letter sounds. LRP56 itself was forced to a LINEAR mechanism because the HSGP curve
diverged (~300 divergences, unmoved by target_accept 0.99), so its estimand is a
single slope, not a shape, and no knee could be looked for. This variant switches the
HSGP curve back on and lifts target_accept to 0.999 (the setting that stabilises the
letter-sound curve in LRP58) so the curve gets a genuine chance to converge. If it
still diverges, that non-convergence is itself the honest result and the knee stays
untestable for this exposure.

Adjustment set (revised DAG, 2026-07-10). Re-derived by a backdoor d-separation search
with the latent GA held (the criterion that reproduces LRP56/58 and dose-077 exactly):
R's observed confounders that also reach WR are {A, HS, TR, RW}; group G(=IG) is the
always-in precision term and W_pre the autoregressive baseline. Identical to LRP56 -
only the mechanism's functional form changes.

Residual confounding by latent general ability (GA) remains, so f^R is an ADJUSTED
ASSOCIATION, not a causal effect. The randomised causal claim lives in the ITT suite.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-156",
    kind="mechanism",
    title="GP knee-test: receptive vocabulary (R) -> word reading (W)",
    outcome_symbol="W",
    mechanism_symbol="R",
    adjustment=["G", "A", "TR", "W_pre"],
    extra={
        "outcomes": ("W", "R", "TR"),
        "adjust_baseline_symbol": "W",
        "adjust_for": ("hs", "hs_missing", "erbto", "erbto_missing"),
        "use_age_gp": False,
        "phase_specific_mechanism": False,
        "use_subject_random_intercept": True,
        # HSGP mechanism curve ON (knee-test); target_accept 0.999 per LRP58.
        "target_accept": 0.999,
    },
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
