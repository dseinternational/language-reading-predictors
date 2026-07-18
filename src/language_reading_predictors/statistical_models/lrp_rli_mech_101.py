# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP101 - linear letter-sound (L) -> word reading (W) slope (contrast/panel anchor).

Tier-1 decoding-specificity mini-suite (see
``notes/202607172330-tier1-decoding-specificity-spec.md``). This is the **linear
single-slope** counterpart to LRP58, which fits the nonparametric HSGP *shape* of
the letter-sound -> word-reading relationship. The Tier-1 contrast (1A) and
negative-control forest (1B) need every outcome on the *same* parameterisation - a
single ``beta_mech . z(logit L)`` slope, logit per SD of the exposure - so the
cross-outcome comparison is like-for-like. LRP58 stays the deliverable for the
*curve*; LRP101 supplies the comparable slope.

Roles:

- **1A convergent-discriminant contrast** - the ``W`` partner to LRP96's ``L -> N``
  slope: Delta = beta(L->N) - beta(L->W). Letter sounds should feed pure decoding (N)
  at least as strongly as the sight-readable word channel (W).
- **1B negative-control panel** - the positive-control anchor: the ``L`` slope should
  be clearly positive here (and on N / spelling) but ~0 on the oral-language
  negative-control outcomes R/E/T/F (LRP97-100).

Adjustment {G, A, HS, IS, SP} + ``W_pre`` - identical to LRP58/LRP96 (revised-DAG
parents of L, #245), so the whole Tier-1 panel shares one conditioning set. GA is
unblockable; ``beta_mech`` is an **adjusted association**, not a causal effect.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-101",
    kind="mechanism",
    title="Linear letter-sound (L) -> word reading (W) slope (Tier-1 contrast anchor)",
    outcome_symbol="W",
    mechanism_symbol="L",
    adjustment=["G", "A", "W_pre"],
    extra={
        "adjust_baseline_symbol": "W",
        "outcomes": ("W", "L"),
        "adjust_for": ("hs", "hs_missing", "attend", "deapp_c", "deapp_c_missing"),
        "linear_mechanism": True,
        "use_age_gp": False,
        "phase_specific_mechanism": False,
        "use_subject_random_intercept": True,
    },
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
