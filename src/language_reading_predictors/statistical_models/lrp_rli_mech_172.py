# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP72base - no-interaction baseline for LRP72 (L + blending main effects).

Fits the same outcome / data / linear mechanism as LRP72, but with phoneme
blending (B) entered as the moderator **main effect only** and `include_interaction=False`,
so there is **no** `gamma_int` (L × B) term. Because the two models then differ by
*exactly* the interaction — B enters through the identical `gamma_mod · z(B)`
term with the identical prior in both — a PSIS-LOO comparison of LRP72 against
this baseline is a clean **nested** test of whether the blending × letter-sound
interaction improves predictive fit.

This mirrors the LRP73/LRP73base exact-nesting construction. Earlier revisions
entered B here on the raw logit scale (`gamma_B · logit B_post`) as a plain
confounder, which is *not* prior-invariant to LRP72's standardised `z(B)`: under
proper priors part of any ELPD difference would then be attributable to the
different effective prior on the B main effect, not the interaction (issue #270
item 3). Keeping `moderator_symbol="B"` in both removes that leak. See
`lrp-rli-mech-072.py`.
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
    # B enters as the moderator MAIN EFFECT (gamma_mod·z(B)), identical to LRP72,
    # with include_interaction=False dropping only gamma_int -> exact nesting.
    # N_pre is the baseline; the revised-DAG L backdoor adjusters (HS, attend, SP;
    # #245) enter via adjust_for.
    adjustment=["G", "A", "N_pre"],
    extra={
        "adjust_baseline_symbol": "N",
        "outcomes": ["L", "B", "N"],
        "adjust_for": ("hs", "hs_missing", "attend", "deapp_c", "deapp_c_missing"),
        "moderator_symbol": "B",
        "include_interaction": False,
        "linear_mechanism": True,
        "use_age_gp": False,
        "phase_specific_mechanism": False,
        "use_subject_random_intercept": True,
    },
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
