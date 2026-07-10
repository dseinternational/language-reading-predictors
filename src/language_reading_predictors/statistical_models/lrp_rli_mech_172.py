# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP72base - no-interaction baseline for LRP72 (L + blending main effects).

Fits the same outcome / data / linear mechanism as LRP72, but with phoneme
blending (B) entered as a plain main-effect confounder and **no moderation
term** (no `gamma_int`). A PSIS-LOO comparison of LRP72 against this baseline
therefore isolates the predictive contribution of the blending × letter-sound
**interaction** — the "does adding the interaction improve fit?" question.

B enters here on the raw logit scale (`gamma_B · logit B_post`) rather than the
standardised `z(B)` used by LRP72's moderator main effect; the two span the same
1-D column, so the maximised fit / LOO is identical for the main-effect part and
the comparison is a clean nested test of the interaction. See `lrp-rli-mech-072.py`.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-172",
    kind="mechanism",
    title=(
        "Code-based route baseline (no interaction): letter-sound (L) + blending (B) "
        "main effects -> decoding (nonword)"
    ),
    outcome_symbol="N",
    mechanism_symbol="L",
    # B added as a plain confounder (main effect only); N_pre is the baseline. The
    # revised-DAG L backdoor adjusters (HS, attend, SP; #245) enter via adjust_for.
    adjustment=["G", "A", "N_pre", "B"],
    extra={
        "adjust_baseline_symbol": "N",
        "outcomes": ["L", "B", "N"],
        "adjust_for": ("hs", "hs_missing", "attend", "deapp_c", "deapp_c_missing"),
        "linear_mechanism": True,
        "use_age_gp": False,
        "phase_specific_mechanism": False,
        "use_subject_random_intercept": True,
        # no moderator_symbol -> no gamma_mod / gamma_int
    },
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
