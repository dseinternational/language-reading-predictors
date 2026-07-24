# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP103 - Mechanism model: speech production (SP) -> nonword decoding (N).

#421 Tier 1: the suite's first speech-production-exposure mechanism. Under the revised
DAG speech production (SP, ``deapp_c``) is a parent of both letter sounds and nonword
decoding (``SP -> { TE EV LS PA NW }``), and the letter-sound -> word-reading review
found it a consistent discriminator of word reading among children alike on letter
sounds (+0.19 to +0.28 across waves), yet no SP-exposure mechanism model existed. This
fits the adjusted SP -> nonword-decoding dose-response across the three phase
transitions.

**Covariate exposure.** SP is the speech-accuracy covariate ``deapp_c`` (no bounded-count
denominator), so it enters as a standardised continuous covariate
(``mechanism_is_covariate``); ``beta_mech`` is the association per +1 SD. ``require_observed``
drops the mean-imputed rows. **Linear, not HSGP:** as ``mech-102``, nonword reading is
heavily floored, so a nonparametric curve is not identifiable; the outcome is the graded
Beta-Binomial on the nonword count with its own baseline ``N_pre``.

**Adjustment set.** SP's measured parents under the revised DAG are age and hearing (HS)
- the same trait roots that confound the SP -> N backdoor - so with the group nuisance
and the ``N`` baseline that is the adjustment set. Every coefficient is a
latent-GA-confounded **adjusted association**, never causal.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-103",
    kind="mechanism",
    title="Mechanism model: speech production (SP, deapp_c) -> nonword decoding (N)",
    outcome_symbol="N",
    mechanism_symbol="deapp_c",
    adjustment=["G", "A", "N_pre"],
    extra={
        # Only the outcome (N) is a bounded-count measure; the exposure is the deapp_c
        # covariate, so the measure complete-case mask is N alone.
        "outcomes": ("N",),
        "adjust_baseline_symbol": "N",
        # SP's measured parents (age via gamma_A + hearing) confound the SP -> N backdoor.
        "adjust_for": ("hs", "hs_missing"),
        # The exposure must be genuinely observed - drop mean-imputed rows.
        "require_observed": ("deapp_c",),
        # Standardised-covariate exposure; linear (floored N outcome).
        "mechanism_is_covariate": True,
        "linear_mechanism": True,
        "use_age_gp": False,
        "phase_specific_mechanism": False,
        "use_subject_random_intercept": True,
    },
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
