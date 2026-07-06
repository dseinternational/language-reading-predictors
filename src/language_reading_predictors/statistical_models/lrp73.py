# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP73 - is the letter-sound -> word-reading effect age-dependent?

Third model in the interaction family. The SHAP screen's strongest *gain-side*
signal was age, which moderated the skill->gain relationships. LRP73 tests the
top one on the reading *level/mechanism* (better powered): does the letter-sound
(L) -> word reading (W) effect depend on **age** (a developmental-readiness
question)?

Form (LRP71/72 moderated-mechanism machinery, moderator = age as a continuous
covariate):

    eta = ... + f_mech(logit L) + gamma_mod·z(age) + gamma_int·z(logit L)·z(age)

`gamma_int > 0` would mean the code-based route converts to reading *more* strongly at older
ages. W is not floored, so the nonparametric `f_mech` HSGP is kept (unlike LRP72).

Adjustment set: LRP58's {G, A, E, R, W_pre}. NOTE: the mechanism factory now
conditions on age **linearly by default** — whenever `A` is a declared confounder
and the age GP is off (`use_age_gp=False`), a `gamma_A·z(age)` term is added — so
LRP58 (and the rest of the mechanism family) *does* adjust for the age confounder
as fit. LRP73 is the one exception: because age is its *moderator*
(`moderator_is_covariate`), the linear `gamma_A` term is skipped (it would be
collinear with the moderator main effect) and the age adjustment is realised
instead by the moderator main effect `gamma_mod·z(age)`, alongside the
`gamma_int·z(logit L)·z(age)` interaction being tested. Documented in the report.

**Sharp prior expectation.** The two prior interaction models both showed an
apparent interaction that was really a *between-child ability confound*, which
collapsed once subject random effects + adjustment were in (LRP71 code-based×vocab
null; LRP72 blending×letter-sound largely independent). Age is *even more* a
between-child variable — most age variation is across children, not within a
child's four waves — so LRP73 is the most confound-prone of the three. The honest
expectation is that the age moderation may also be a confound. A third
confound-driven null would make the interaction-phase conclusion robust:
contributors to reading combine **additively**; the raw moderations are
between-child ability confounds. The within-child check
(`scripts/within_child_interaction_check.py`) is the key diagnostic; see the note.

`lrp73base.py` is the no-interaction companion (same f_mech + age main effect, no
L×age) for the nested PSIS-LOO test.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp73",
    kind="mechanism",
    title="Mechanism model: letter-sound (L) -> word reading (W), moderated by age",
    outcome_symbol="W",
    mechanism_symbol="L",
    adjustment=["G", "A", "E", "R", "W_pre"],
    extra={
        "adjust_baseline_symbol": "W",
        "moderator_symbol": "A",
        "moderator_is_covariate": True,
        "include_interaction": True,
        "use_age_gp": False,
        "phase_specific_mechanism": False,
        "use_subject_random_intercept": True,
    },
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
