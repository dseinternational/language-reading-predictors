# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP157 - GP knee-test: expressive vocabulary (E) -> word reading (W).

Nonparametric (HSGP) re-attempt of the LRP57 mechanism, built to TEST whether the
E -> W association has a "knee" the way LRP58 found for letter sounds. LRP57 was forced
to a LINEAR mechanism because the HSGP curve diverged (~200 divergences, unmoved by
target_accept 0.99), so its estimand is a single slope, not a shape. This variant
switches the HSGP curve back on and lifts target_accept to 0.999 (per LRP58) so the
curve gets a genuine chance to converge; a persisting divergence is itself the honest
result and leaves the knee untestable for this exposure.

Adjustment set (revised DAG, 2026-07-10). Re-derived by a backdoor d-separation search
with the latent GA held (the criterion that reproduces LRP56/58 and dose-077): E's
observed confounders that also reach WR are {A, HS, R, RW, SP, TE, TR}; group G(=IG) is
the always-in precision term and W_pre the autoregressive baseline. Identical to LRP57 -
only the mechanism's functional form changes.

Residual confounding by latent general ability (GA) remains, so f^E is an ADJUSTED
ASSOCIATION, not a causal effect. The randomised causal claim lives in the ITT suite.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-157",
    kind="mechanism",
    title="GP knee-test: expressive vocabulary (E) -> word reading (W)",
    outcome_symbol="W",
    mechanism_symbol="E",
    adjustment=["G", "A", "TR", "TE", "R", "W_pre"],
    extra={
        "outcomes": ("W", "E", "R", "TR", "TE"),
        "adjust_baseline_symbol": "W",
        "adjust_for": (
            "hs", "hs_missing", "erbto", "erbto_missing", "deapp_c", "deapp_c_missing",
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
