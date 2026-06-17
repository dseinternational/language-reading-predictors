# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP72 - the phonics route: letter-sound (L) × phoneme blending (B) -> decoding.

Second model in the interaction family (after LRP71). Tests the dual-route /
triangle-model hypothesis that the two phonological prerequisites — phoneme
**blending** (B) and **letter-sound** knowledge (L) — combine **super-additively**
to bring word **decoding** online: you need both, and each is worth more when the
other is present. Decoding is measured by `nonword` (the six monster nonwords),
symbol `N`. This is the phonological-route sub-mechanism feeding the established
letter-sound -> word-reading effect (LRP58).

Form (mirrors LRP71, mechanism = L, moderator = B):

    eta = ... + beta_mech·z(logit L) + gamma_mod·z(B) + gamma_int·z(logit L)·z(B)

`gamma_int > 0` means letter-sound converts to decoding *more* strongly for
higher-blending children — i.e. the prerequisites combine. The framing is
symmetric; "mechanism = L, moderator = B" is just the factory's vocabulary.

Design choices:

- **Linear mechanism** (`linear_mechanism=True`): decoding is a low, 57%-floored
  0-6 count, so a full HSGP dose-response on logit(L) is not identifiable. The
  mechanism enters as a single linear slope `beta_mech·z(logit L)`.
- **Minimal outcomes** (`outcomes=["L","B","N"]`): only the symbols this model
  uses, so `drop_missing_pre` does not discard rows for unrelated measures.
- **Adjustment set {G, A, N_pre}** — group, age, and baseline decoding
  (autoregressive). **Word reading `W` is deliberately excluded**: it is a
  sibling/descendant of decoding, so conditioning on it is over-control / a
  collider risk (same error class as the celf/LRP70 issue). Vocabulary/concepts
  are not plausible confounders of the phonics -> decoding path. L and B are
  treated as parallel prerequisites; the interaction estimand does not require
  resolving whether L precedes B. Documented in the report.

Data caveat: `nonword` is 57% floored (median 0) but informative and moving
(28% -> 60% of children decode >=1 nonword across t1..t4); predictors (L, B) are
each only ~2% floored. The floor reduces power, so a credible positive
`gamma_int` would be the cleanest confirmation of the phonological route, but the
floor + n~54 may keep it suggestive.

`lrp72base.py` is the no-interaction companion (L + B main effects, no L×B) for
the PSIS-LOO comparison that isolates the interaction's predictive value.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp72",
    kind="mechanism",
    title=(
        "Phonics route: letter-sound (L) moderated by phoneme blending (B) "
        "-> decoding (nonword)"
    ),
    outcome_symbol="N",
    mechanism_symbol="L",
    adjustment=["G", "A", "N_pre"],
    extra={
        "adjust_baseline_symbol": "N",
        "outcomes": ["L", "B", "N"],
        "moderator_symbol": "B",
        "linear_mechanism": True,
        "use_age_gp": False,
        "phase_specific_mechanism": False,
        "use_subject_random_intercept": True,
    },
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
