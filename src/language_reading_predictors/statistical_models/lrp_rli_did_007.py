# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPDID07 - period-resolved session-dose response on letter-sound knowledge (YARC-LSK) (L).

The letter-sound analogue of LRPDID06's word-reading dose-response, but with the
session-dose slope **resolved by period** (#135). It reuses LRPDID02's
identification - the phase-stacked waitlist-crossover frame with each child as
their own control and the immediate arm anchoring the time/maturation trend - and
replaces the binary treated indicator with the standardised per-period
intervention-session count, entered with **partial-pooled per-period slopes**
``beta_dose_phase[p] = mu_dose + sigma_dose * z_p``.

The live question is whether the dose-gain slope varies by period; it is answered
by a **nested PSIS-LOO** of this model against its pooled-dose comparator,
LRPDID07base, in ``compare_statistical_models.py``. The word-reading analogue
(LRP77) found the dose slope **constant across periods** (LOO preferred the pooled
model), so the most likely outcome here is "no period variation at this n" - a
useful, publishable negative result.

DAG note (#115). The per-period session count is the **exposure** ``IS``, not a
conditioning variable: this is the ID-3 observational dose-response, **confounded
by GA -> IS** and reported as an *adjusted association*, never "more sessions cause
more gain". Crucially the DiD machinery adjusts only ``{IG, A}`` and never
conditions on cumulative sessions (``attend_cumul``, the ``IS`` collider), so it
sidesteps the conditioning that closed PR #108. Only the randomised contrast (the
ITT ``lrp-rli-itt-007`` / the DiD arm term ``lrp-rli-did-002``) is causal.

Sign convention: positive => intervention helps.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_did

SPEC = ModelSpec(
    model_id="lrp-rli-did-007",
    kind="did",
    title=(
        "Waitlist-crossover (DiD) period-resolved session-dose response on "
        "letter-sound knowledge (YARC-LSK) (L)"
    ),
    outcome_symbol="L",
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
