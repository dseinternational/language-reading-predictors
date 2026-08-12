# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP60 - sequential code route: letter sounds (L) -> nonword decoding (N) -> word reading (W).

#421 Tier 3 (letter-sound -> word-reading review note #424; decoding-specificity spec
202607172330 §6). *Registered as med-060, not the note's proposed med-081:* the bare
legacy alias ``lrp81`` is already the live ``lcsm-081``, so 081 was never free; 060 is the
lowest free bare alias in the mediation range. The chained companion to ``med-075`` (L -> blending B -> W), but through
the **alphabetic route's own fingerprint**, nonword decoding N. It was specced then
**withdrawn at build time** because ``build_two_mediator_model`` required each mediator's
autoregressive baseline, and N is post-only with a ~72%-floored t1 score. This model uses
the new **off-floor second-mediator leg**: N enters as a Bernoulli off-floor indicator
(P(N > 0)) with no autoregressive baseline, so the chained ``L -> N -> W`` g-formula
decomposition finally builds.

**What it estimates.** With ``chain=True`` the N leg regresses on post-L (the ``L -> N``
coupling ``aN_L``), and the g-formula draws the off-floor N indicator conditional on the
*simulated* L, so the joint indirect effect through the ``{L, N}`` block carries the
``L -> N -> W`` sub-path. The robust headline is the **joint indirect effect through
{L, N}** and the ``L -> N`` coupling; the per-path NIE_L / NIE_N split is exploratory and
ordering-dependent, and — because N is heavily floored — **floor-limited and low-power**
(exactly the caveat that led to the original withdrawal).

**Estimand / caveats.** The outcome W stays a graded Beta-Binomial; only the N *mediator*
is off-floor. Every path is an **adjusted association / g-formula decomposition under the
stated cross-world assumptions**, never an identified natural effect (the ``IG -> IS ->
{L, N, W}`` dose witness is untouched). N as an off-floor indicator means the N-path is
"the effect that flows through *crossing the decoding floor*", not through graded decoding
skill.

.. note::
   Adjustment set (``{E, R}`` + hearing/speech/phonological-memory traits, mirroring
   ``med-075``) and the off-floor-mediator NIE definition are flagged for Frank's
   methodological sign-off (the exposure/estimand is his call; this model ships the
   machinery + a fittable specification).
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.mediation_settings import (
    MediationMultiModelSettings,
)
from language_reading_predictors.statistical_models.pipeline import fit_mediation_multi

SPEC = ModelSpec(
    model_id="lrp-rli-med-060",
    kind="mediation_multi",
    title=(
        "Sequential code route: does the word-reading (W) gain run letter sounds (L) "
        "-> nonword decoding (N, off-floor) -> reading?"
    ),
    outcome_symbol="W",
    mechanism_symbol=None,  # two mediators; named in extra["mediators"]
    adjustment=[
        "G", "A", "E", "R", "W_pre", "L_t1",
        "hs", "hs_missing", "deapp_c", "deapp_c_missing", "erbto", "erbto_missing",
    ],
    model_settings=MediationMultiModelSettings(
        mediators=("L", "N"),
        order=("L", "N"),
        chain=True,  # add the L -> N edge; draw N conditional on simulated L
        second_mediator_offfloor=True,  # N is post-only / ~72% floored -> Bernoulli leg
        outcomes=("W", "L", "N"),  # load N (floored, outside the default ITT set)
    ),
)


def fit(config: str = "dev"):
    return fit_mediation_multi(SPEC, config=config)
