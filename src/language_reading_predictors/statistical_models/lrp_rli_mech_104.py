# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP104 - does the letter-sound -> word-reading effect depend on phonological memory?

#421 Tier 2 (letter-sound -> word-reading review note, #424): the direct,
uncertainty-quantified version of the review's Q3 discrepancy finding. Q3 showed that,
among children who know their letter sounds but read few words, the sharpest
distinguishing feature is weak phonological memory / decoding - but it rested on a
15-vs-15 descriptive split plus one-at-a-time partials. This model asks the same
question as an interaction:

    eta = ... + f_mech(logit L) + gamma_mod·z(RW) + gamma_int·z(logit L)·z(RW)

``gamma_int > 0`` would mean letter sounds convert to word reading **more** strongly for
children with better phonological memory (word/nonword repetition, RW = ``erbto``) -
exactly the Q3 hypothesis. W is not floored, so the nonparametric ``f_mech`` HSGP is kept.

**Covariate moderator.** RW is the ERB total (``erbto``), a covariate rather than a
``MEASURES`` symbol, so it enters as ``moderator_is_covariate=True`` - the ``mech-073``
path, except that (unlike intrinsic age) a covariate moderator must be *loaded*; the
pipeline now does so. It also complete-cases on the moderator (``require_observed``):
``erbto`` is mean-imputed for the children missing it, and moderating an interaction by
a sample-mean-filled effect modifier is not meaningful, so those rows are dropped rather
than read as sitting at average phonological memory. Confounder set is the
``mech-058``/``mech-073`` letter-sound ->
word-reading set {G, A, HS, IS(attend), SP} + ``W_pre``; RW is not in it (it is the
moderator, entered via ``gamma_mod``/``gamma_int``, never ``adjust_for``).

**Read like mech-071/072/073.** Every coefficient is an **adjusted association**, latent
general ability unblocked; ``gamma_int`` is descriptive, never causal. The prior
interaction models repeatedly found an apparent moderation that was really a
between-child ability confound and collapsed under adjustment + subject random effects,
so the honest expectation is the same here. ``lrp-rli-mech-204`` is the no-interaction
companion for the nested PSIS-LOO test; the within-child check
(``scripts/within_child_interaction_check.py``) is the key diagnostic.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-104",
    kind="mechanism",
    title=(
        "Mechanism model: letter-sound (L) -> word reading (W), "
        "moderated by phonological memory (RW)"
    ),
    outcome_symbol="W",
    mechanism_symbol="L",
    adjustment=["G", "A", "W_pre"],
    extra={
        "outcomes": ("W", "L"),
        "adjust_baseline_symbol": "W",
        "adjust_for": ("hs", "hs_missing", "attend", "deapp_c", "deapp_c_missing"),
        "moderator_symbol": "erbto",
        "moderator_is_covariate": True,
        # Drop the mean-imputed erbto rows: do not moderate by an average-filled
        # effect modifier (loads erbto_missing for the loader's filter).
        "require_observed": ("erbto",),
        "include_interaction": True,
        "use_age_gp": False,
        "phase_specific_mechanism": False,
        "use_subject_random_intercept": True,
    },
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
