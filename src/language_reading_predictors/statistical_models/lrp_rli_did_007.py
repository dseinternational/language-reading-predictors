# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPDID07 - period-resolved session-dose response on letter-sound knowledge (YARC-LSK) (L).

The letter-sound analogue of LRPDID06's word-reading dose association, but with the
session-dose slope **resolved by period** (#135). The transition model separates
randomised arm/history, current treatment presence and session intensity, and
adjusts both periods for the shared pre-randomisation t1 outcome and t1 age.
Treated-row-standardised sessions enter with partial-pooled per-period slopes
``beta_dose_phase[p] = mu_dose + sigma_dose * z_p``.

The live question is whether the dose-gain slope varies by period; it is answered
by a **nested PSIS-LOO** of this model against its pooled-dose comparator,
LRPDID07base, in ``compare_statistical_models.py``. The word-reading analogue
(LRP77) found the dose slope **constant across periods** (LOO preferred the pooled
model), so the most likely outcome here is "no period variation at this n" - a
useful, publishable negative result.

The session slopes are observational intensive-margin associations, potentially
confounded by general ability and attendance selection. Cumulative sessions are
not conditioned on. The causal intervention contrast is reported by the ITT and
binary arm-by-wave models, not by these dose coefficients.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_did

SPEC = ModelSpec(
    model_id="lrp-rli-did-007",
    kind="did",
    title=(
        "Waitlist-crossover period-resolved session-dose association on "
        "letter-sound knowledge (YARC-LSK) (L)"
    ),
    outcome_symbol="L",
    family="did",
    design="waitlist-crossover transition dose intensive margin",
    estimand_type="association",
    causal_status="none for session-dose coefficients",
    extra={
        "outcomes": ("L",),
        "periods": (0, 1),
        "use_child_re": True,
        "use_age": True,
        "dose": True,
        "period_varying_dose": True,
    },
)


def fit(config: str = "dev"):
    return fit_did(SPEC, config=config)
