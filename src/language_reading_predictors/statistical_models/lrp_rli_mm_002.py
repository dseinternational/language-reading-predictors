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

**Estimand vs LRPMM01.** LRPMM01 adjusts the code slope for the co-varying latent
vocabulary and grammar factors (its "unique code contribution among constructs").
LRPMM02 instead adjusts for the **mech-058 confounder set** - the randomised arm G
(``use_group``), age, own word-reading baseline, hearing (HS) and speech production
(SP) - so its ``beta_code`` is the errors-in-variables analogue of the mech-058
estimand, directly comparable to mech-058's attenuated observed-L association. The
report also shows L's and B's **communality** (how well each measures latent code).

**Caveats.** Still ID-2: ``beta_code`` is a latent-ability-confounded **adjusted
association**, never "code drives reading" (the randomised claim lives in the ITT
suite). Frame difference from mech-058: this is between-child (``phase_mode="span"``,
t1 baselines -> t4 word reading), not mech-058's phase-stacked within-transition frame,
so it is a close companion, not an identical re-fit. Intervention dose (``attend``,
a per-period collider with no clean between-child span analogue - t1 attendance is 0)
is deliberately **omitted** from the adjustment. Wide intervals at n ~ 51.
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
        # mech-058 adjustment set on the span frame: G (use_group), age (use_age),
        # own W baseline (gamma_own), hearing + speech (below). No `blocks` (mech-058
        # does not adjust for ability); no `attend` (per-period collider, no clean
        # span analogue).
        "use_group": True,
        "use_age": True,
        "structural_covariates": (
            "hs", "hs_missing", "deapp_c", "deapp_c_missing",
        ),
        # Same small-n latent-factor geometry fix as LRPMM01.
        "target_accept": 0.999,
    },
)


def fit(config: str = "dev"):
    return fit_correlated_factor(SPEC, config=config)
