# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP63base - no-interaction baseline for LRP63 (L + nonword-decoding main effects -> W).

Fits the same outcome / data / HSGP mechanism as LRP63, but with nonword decoding (N)
entered as the moderator **main effect only** and ``include_interaction=False``, so
there is **no** ``gamma_int`` (L x N) term. The two models then differ by *exactly* the
interaction - N enters through the identical ``gamma_mod * z(N)`` term with the identical
prior in both, and the HSGP ``f_mech`` on letter sounds is unchanged - so a PSIS-LOO
comparison of LRP63 against this baseline is a clean **nested** test of whether the
decoding x letter-sound interaction improves predictive fit on word reading.

This mirrors the LRP61/LRP61base and LRP72/LRP72base exact-nesting construction, keeping
the HSGP mechanism of LRP63 and ``moderator_symbol="N"`` in both to remove any prior-scale
leak into the ELPD difference (issue #270 item 3).

As in LRP63, N is a DAG-descendant of the exposure L (``LS -> ... -> NW``), so conditioning
on N makes ``beta_mech`` a controlled-direct (not total) L -> W effect; this baseline
shares that estimand, so the nested LOO comparison is like-for-like. N is ~57% floored,
so the interaction it tests is weakly powered - the LOO contrast is suggestive at best.
See ``notes/202607172000-adjustment-set-review-full-suite.md``.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-163",
    kind="mechanism",
    title=(
        "Joint-readiness baseline (no interaction): letter sounds (L) + nonword "
        "decoding (N) main effects -> word reading (W)"
    ),
    outcome_symbol="W",
    mechanism_symbol="L",
    adjustment=["G", "A", "W_pre"],
    extra={
        "outcomes": ("W", "L", "N"),
        "adjust_baseline_symbol": "W",
        "adjust_for": ("hs", "hs_missing", "attend", "deapp_c", "deapp_c_missing"),
        "use_age_gp": False,
        "phase_specific_mechanism": False,
        "use_subject_random_intercept": True,
        "moderator_symbol": "N",
        "include_interaction": False,
        "target_accept": 0.999,
    },
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
