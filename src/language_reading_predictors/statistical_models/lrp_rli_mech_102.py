# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP102 - Mechanism model: phonological memory (RW) -> nonword decoding (N).

#421 Tier 1: the alphabetic-route counterpart of ``mech-090`` (RW -> word reading). The
letter-sound -> word-reading review found nonword repetition (the phonological
short-term-memory component of the word-repetition task, RW) the sharpest discriminator
of word reading among children alike on letter sounds. ``mech-090`` already tests
whether RW feeds *word reading*; nothing tested whether it feeds the **alphabetic route**
specifically. This fits the adjusted RW -> nonword-decoding dose-response across the
three phase transitions - "a higher word/nonword repetition score is associated with
more nonwords read".

**Covariate exposure (as mech-090).** RW is the ERB total (``erbto``), which has no
recorded test maximum, so it enters as a standardised continuous covariate
(``mechanism_is_covariate``); ``beta_mech`` is the association per +1 SD. ``require_observed``
drops the mean-imputed rows. **Linear, not HSGP:** nonword reading is a 6-item outcome
floored at zero for ≈72/64/52/40% of children at t1-t4, where a nonparametric
dose-response is not identifiable (the same reasoning as ``mech-072``); the outcome is
still the graded Beta-Binomial on the nonword count, conditioning on its own baseline
``N_pre``.

**Adjustment set (transfers from mech-090).** HS is RW's only measured parent under the
revised DAG; with age (linear ``gamma_A``), the group nuisance and the ``N`` baseline,
that is the adjustment set. Every coefficient is a latent-GA-confounded **adjusted
association**, never causal.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.mechanism import (
    MechanismModelSettings,
)
from language_reading_predictors.statistical_models.pipelines.mechanism import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-102",
    kind="mechanism",
    title=(
        "Mechanism model: phonological memory (RW, word/nonword repetition) -> "
        "nonword decoding (N)"
    ),
    outcome_symbol="N",
    mechanism_symbol="erbto",
    adjustment=["G", "A", "N_pre"],
    model_settings=MechanismModelSettings(
        # Only the outcome (N) is a bounded-count measure; the exposure is the erbto
        # covariate, so the measure complete-case mask is N alone.
        outcomes=("N",),
        adjust_baseline_symbol="N",
        # HS is RW's only measured parent under the revised DAG (as mech-090).
        adjust_for=("hs", "hs_missing"),
        # The exposure must be genuinely observed - drop mean-imputed rows.
        require_observed=("erbto",),
        # Standardised-covariate exposure (no fabricated ERB denominator); linear.
        mechanism_is_covariate=True,
        linear_mechanism=True,
        use_age_gp=False,
        phase_specific_mechanism=False,
        use_subject_random_intercept=True,
    ),
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
