# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP158 - complete-case comparator for LRP58 (letter-sound L -> word reading W).

The revised-DAG confounders enter LRP58 by the **missing-indicator method**: hearing
(HS), speech production (SP = ``deapp_c``) and phonological memory (RW = ``erbto``)
are filled to their column mean and a ``{col}_missing`` flag carries the unknown
group as its own adjustment level. That keeps every child, which matters at n ~ 54.

But it is not a free lunch, and the #258 review is right to press on it: mean
imputation plus an indicator **preserves rows without guaranteeing confounding
control**. It assumes the imputed group's confounder effect is captured by a single
intercept shift — i.e. that within the "unknown" stratum the confounder is unrelated
to both exposure and outcome once the flag is in the model. That is an assumption,
not a consequence of the method, and it is doing real work here because the
missingness is not trivial:

- ``hearing_c`` missing for **10 of 54 children at every wave** (~19%);
- ``deapp_c`` ~4% of rows; ``erbto`` ~6.5% of rows;
- only **35 of 54 children** are complete on all three at every wave.

LRP158 is therefore the honest comparator: **identical to LRP58 in every respect**
except that the mean-imputed rows are dropped (``require_observed``), so every
confounder is genuinely observed. The missingness indicators then go constant and are
dropped, so no vacuous coefficient is estimated.

**How to read it.** If the mechanism slope agrees with LRP58's, the imputation is not
driving the result and LRP58 stands as the primary (higher-powered) fit. If they
diverge, the imputation *is* load-bearing and neither fit should be reported without
the other. The comparator is smaller and so has wider intervals **by construction** —
that is the price of the restriction, not a finding.

Same caveats as LRP58: latent general ability is **not** adjusted for and the child
random intercept does not stand in for it, so ``f^L`` is an **adjusted association**,
never a causal effect.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-158",
    kind="mechanism",
    title=(
        "Mechanism model: letter-sound (L) -> word reading (W) - "
        "complete-case comparator (no imputed confounders)"
    ),
    outcome_symbol="W",
    mechanism_symbol="L",
    adjustment=["G", "A", "W_pre"],
    extra={
        # Identical to LRP58 ...
        "adjust_for": ("hs", "hs_missing", "attend", "deapp_c", "deapp_c_missing"),
        "use_age_gp": False,
        "use_subject_random_intercept": True,
        # ... except that the imputed rows are dropped, so HS and SP are observed.
        "require_observed": ("hs", "deapp_c"),
    },
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
