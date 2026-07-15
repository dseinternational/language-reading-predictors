# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPDID07base - pooled session-dose response on letter-sound knowledge (L) - LRPDID07 comparator.

The no-period-variation comparator to LRPDID07: the same waitlist-crossover
transition dose model on letter sounds (L), but with a **single
pooled** dose slope ``beta_dose`` instead of partial-pooled per-period slopes.
This is the letter-sound analogue of LRPDID06 (the pooled word-reading dose model).

LRPDID07 (period-varying) and LRPDID07base (pooled) are a nested pair on the same
outcome and rows, so the PSIS-LOO comparison between them in
``compare_statistical_models.py`` is a clean test of whether the letter-sound
dose-gain slope varies by period (#135).

The model separates randomised arm/history, treatment presence and session
intensity; both periods adjust for the shared pre-randomisation t1 outcome and age.
The session slope is an observational intensive-margin association, potentially
confounded by general ability and attendance selection. Cumulative sessions are
not conditioned on.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_did

SPEC = ModelSpec(
    model_id="lrp-rli-did-107",
    kind="did",
    title=(
        "Waitlist-crossover session-dose association (pooled slope) on "
        "letter-sound knowledge (L) - no-period-variation comparator to LRPDID07"
    ),
    outcome_symbol="L",
    family="did",
    design="waitlist-crossover transition dose intensive margin",
    estimand_type="association",
    causal_status="none for session-dose coefficient",
    extra={
        "outcomes": ("L",),
        "periods": (0, 1),
        "use_child_re": True,
        "use_age": True,
        "dose": True,
        "period_varying_dose": False,
    },
)


def fit(config: str = "dev"):
    return fit_did(SPEC, config=config)
