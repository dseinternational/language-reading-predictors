# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LCSM-082 - blending <-> word reading reciprocal dominance (exploratory).

The cross-lagged dominance question the lagged DAG makes first-class (#250):
does prior **reading** predict **blending** change more strongly than prior
blending predicts reading change? Design:
``notes/202607141030-time-lagged-model-designs.md``.

Model (the generalised LCSM scaffold): processes ``W`` / ``B`` / ``L``, with
reciprocal couplings and letter-sounds retained as the one cheap measured
confounder both directions share::

    Delta_W = a_W[arm, w] + b_W * x_W + g_B_W * x_B + g_L_W * x_L + d_age * age
            + (hs / rw / sp covariate block)
    Delta_B = a_B[arm, w] + b_B * x_B + g_W_B * x_W + g_L_B * x_L + d_age * age
            + (same covariate block)
    Delta_L = a_L[arm, w] + b_L * x_L + d_age * age

The headline readout is the **SD-standardised dominance contrast**
(``dominance_summary.csv``): per posterior draw each coupling is standardised
by the model's own latent scales (``g* = g * sd(prior source levels) /
sd(target changes)``) and ``|g*_W->B| - |g*_B->W|`` is reported with its
posterior probability.

**Exploratory, association-only, both directions.** The verified minimal
backdoor set for either direction needs ~13 nodes including floored nonword
and two-waves-back reading, so no fittable version of this model approaches
the measured blocking set (unlike LCSM-081) - a hypothesis-generator for a
future cohort, not an evidence claim. Fit only after LCSM-081 clears its
diagnostics gate. The mild late blending ceiling (14-21% at 10/10 by t3-t4)
is absorbed by the Beta-Binomial.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_lcsm

SPEC = ModelSpec(
    model_id="lrp-rli-lcsm-082",
    kind="lcsm",
    title=(
        "Reciprocal dominance: blending (B) <-> word reading (W) lagged "
        "cross-couplings with letter-sounds (L) adjustment"
    ),
    outcome_symbol="W",
    extra={
        "outcomes": ["W", "B", "L"],
        # Reciprocal pair plus the shared letter-sound confounder couplings.
        "couplings": {"W": ["B", "L"], "B": ["W", "L"]},
        # The SD-standardised dominance readout compares these two directions.
        "dominance_pair": ["W", "B"],
        "arm_window_intercepts": True,
        # B (= PA) is an HS/SP/RW child in the lagged graph; the block is shared
        # across both target equations (parameter-sparing default at n~54).
        "covariate_block": [
            "hs", "hs_missing",
            "erbto", "erbto_missing",
            "deapp_c", "deapp_c_missing",
        ],
        "covariate_targets": ["W", "B"],
        "coupling_prior_sigma": 0.3,
        "use_process_noise": True,
        "shared_process_noise": False,
    },
)


def fit(config: str = "dev"):
    return fit_lcsm(SPEC, config=config)
