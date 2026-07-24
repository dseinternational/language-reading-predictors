# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP61base - no-interaction baseline for LRP61 (L + blending main effects -> W).

Fits the same outcome / data / HSGP mechanism as LRP61, but with phoneme blending (B)
entered as the moderator **main effect only** and ``include_interaction=False``, so
there is **no** ``gamma_int`` (L x B) term. Because the two models then differ by
*exactly* the interaction - B enters through the identical ``gamma_mod * z(B)`` term
with the identical prior in both, and the HSGP ``f_mech`` on letter sounds is unchanged -
a PSIS-LOO comparison of LRP61 against this baseline is a clean **nested** test of
whether the blending x letter-sound interaction improves predictive fit on word reading.

This mirrors the LRP72/LRP72base exact-nesting construction, but keeps the HSGP
mechanism of LRP61 (LRP72/72base are linear because decoding is heavily floored);
keeping ``moderator_symbol="B"`` in both removes any prior-scale leak into the ELPD
difference (issue #270 item 3).

As in LRP61, B = PA is a DAG-descendant of the exposure L (``LS -> PA``), so conditioning
on B makes ``beta_mech`` a controlled-direct (not total) L -> W effect; this baseline
shares that estimand, so the nested LOO comparison is like-for-like. See
``notes/202607172000-adjustment-set-review-full-suite.md``.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-161",
    kind="mechanism",
    title=(
        "Joint-readiness baseline (no interaction): letter sounds (L) + phoneme "
        "blending (B) main effects -> word reading (W)"
    ),
    outcome_symbol="W",
    mechanism_symbol="L",
    adjustment=["G", "A", "W_pre"],
    extra={
        "outcomes": ("W", "L", "B"),
        "adjust_baseline_symbol": "W",
        "adjust_for": ("hs", "hs_missing", "attend", "deapp_c", "deapp_c_missing"),
        "use_age_gp": False,
        "phase_specific_mechanism": False,
        "use_subject_random_intercept": True,
        "moderator_symbol": "B",
        "include_interaction": False,
        "target_accept": 0.999,
    },
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
