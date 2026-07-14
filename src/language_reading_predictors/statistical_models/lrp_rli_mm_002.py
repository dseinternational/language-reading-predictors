# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPMM02 - errors-in-variables code->word-reading mechanism (latent code factor).

The measurement-error-corrected counterpart of the L->W mechanism (mech-058), for
#228 item 14. ``METHODS.md`` flags that the mechanism models do not separately model
measurement error, so mech-058's HSGP association between the *observed* letter-sound
logit and word reading is attenuated by the noise in the letter-sound test. LRPMM02
corrects for that attenuation by regressing word reading on a **latent code factor**
(built from letter-sounds L + blending B), where the observed-indicator noise loads
onto the residuals, not the structural slope.

**Identification.** A standalone one-factor / two-indicator code model (L + B alone)
is under-identified. So LRPMM02 keeps LRPMM01's full, identified three-factor
measurement model (code {L, B} / vocabulary {R, E} / grammar {F, T}) - the cross-factor
correlations and the other indicators supply the covariances that pin the code factor's
loadings and residuals - but the **structural leg regresses word reading on the code
factor only** (``structural_factors=("code",)``), isolating the code->reading slope.

**Estimand vs LRPMM01 — a deliberate bracketing pair.** LRPMM01 adjusts the code
slope for the co-varying **latent** vocabulary and grammar factors (its "unique code
contribution among constructs"). LRPMM02 instead adjusts for an **observed** confounder
set - the randomised arm G (``use_group``), age, own word-reading baseline, hearing
(HS), speech production (SP) and phonological memory (RW = ``erbto``). RW is required
here (beyond the L-only mech-058 mirror) because the exposure is a code factor on **L
and B**, and blending (PA) has parents letter sounds does not - ``RW -> PA``, plus
``TE -> PA`` and ``EV -> PA`` - so anything with an arrow into *either* indicator's
true score confounds the factor->W slope. The two models are a **bracketing pair**:
mm-001 = latent-adjusted code slope, mm-002 = observed-set (mech-058-style) code
slope, and reading them together brackets the estimand. mm-002's named
**residual-confounding direction is vocabulary via the blending indicator** (``TE ->
PA`` / ``EV -> PA`` are left in the residuals, since conditioning on *observed* noisy
vocabulary inside an errors-in-variables model is incoherent - the fully-latent
version of that adjustment is exactly mm-001's structural leg). The report also shows
L's and B's **communality** (how well each measures latent code).

**Caveats.** Still ID-2: ``beta_code`` is a latent-ability-confounded **adjusted
association**, never "code drives reading" (the randomised claim lives in the ITT
suite). Frame difference from mech-058: this is between-child (``phase_mode="span"``,
t1 baselines -> t4 word reading), not mech-058's phase-stacked within-transition frame,
so it is a close companion, not an identical re-fit. The head-to-head attenuation
comparison is therefore against **mech-058's observed-L slope** (same adjustment set,
error-uncorrected), *not* mm-001 (which differs by adjustment set - latent vocabulary,
grammar and non-verbal ability - not by error handling). Intervention dose (``attend``)
is **omitted** for a temporal reason: the exposure is the **t1** code factor and
sessions begin after t1, so there is no arrow into this pre-treatment exposure - IS is
simply not a confounder here (it is *not* "a collider"). Had the exposure been
post-treatment, the #309/#311 pattern would apply instead: IS is a treatment-affected
common cause and conditioning on it would open ``IG -> IS <- GA -> W``. Wide intervals
at n ~ 51.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_correlated_factor

SPEC = ModelSpec(
    model_id="lrp-rli-mm-002",
    kind="corr_factor",
    title=(
        "Errors-in-variables code->word-reading mechanism "
        "(latent code factor, mech-058 adjustment)"
    ),
    outcome_symbol="W",
    extra={
        # Full three-factor measurement model kept for identification of the code factor.
        "domains": {
            "vocabulary": ("R", "E"),
            "code": ("L", "B"),
            "grammar": ("F", "T"),
        },
        # Structural leg on the code factor only (the mech-058-comparable slope).
        "structural_factors": ("code",),
        # mech-058 set on the span frame PLUS RW: G (use_group), age (use_age), own W
        # baseline (gamma_own), hearing + speech, and phonological memory (erbto).
        # RW is added because the exposure is a code factor on L AND B, and blending
        # (PA) has parents letter sounds lacks — RW -> PA — so the L-only mech-058
        # mirror is insufficient once B joins the exposure (signed off 2026-07-14).
        # No `blocks` (mech-058 does not adjust for ability); no `attend` (the t1 code
        # exposure precedes sessions, so IS is not a confounder here — see docstring).
        "use_group": True,
        "use_age": True,
        "structural_covariates": (
            "hs", "hs_missing", "deapp_c", "deapp_c_missing",
            "erbto", "erbto_missing",
        ),
        # Same small-n latent-factor geometry fix as LRPMM01.
        "target_accept": 0.999,
    },
)


def fit(config: str = "dev"):
    return fit_correlated_factor(SPEC, config=config)
