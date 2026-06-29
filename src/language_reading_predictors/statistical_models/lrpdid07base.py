# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPDID07base - pooled session-dose response on letter-sound knowledge (L) - LRPDID07 comparator.

The no-period-variation comparator to LRPDID07: identical waitlist-crossover /
difference-in-differences dose-response on letter sounds (L), but with a **single
pooled** dose slope ``beta_dose`` instead of partial-pooled per-period slopes.
This is the letter-sound analogue of LRPDID06 (the pooled word-reading dose model).

LRPDID07 (period-varying) and LRPDID07base (pooled) are a nested pair on the same
outcome and rows, so the PSIS-LOO comparison between them in
``compare_statistical_models.py`` is a clean test of whether the letter-sound
dose-gain slope varies by period (#135).

DAG note (#115): the per-period session count is the **exposure** ``IS`` - an
ID-3 observational dose-response, confounded by GA -> IS, reported as an *adjusted
association*; ``attend_cumul`` (the ``IS`` collider) is never conditioned on. Only
the randomised contrast (``lrpitt07`` / ``lrpdid02``) is causal.

Sign convention: positive => intervention helps.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_did

SPEC = ModelSpec(
    model_id="lrpdid07base",
    kind="did",
    title=(
        "Waitlist-crossover (DiD) dose-response (pooled dose slope) on "
        "letter-sound knowledge (L) - no-period-variation comparator to LRPDID07"
    ),
    outcome_symbol="L",
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
