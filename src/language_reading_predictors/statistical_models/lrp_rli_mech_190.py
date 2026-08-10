# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP190 - GP knee-test: phoneme blending / phonological awareness (B) -> word reading (W).

A NEW mechanism model (no prior blending -> reading fit existed) built to TEST whether
phoneme blending shows a "knee" - a level of blending skill beyond which it is
associated with a more marked difference in word reading - the way LRP58 found for
letter sounds. The mechanism enters as an HSGP curve on the logit-safe transform of the
blending post-score; target_accept is 0.999 (per LRP58). Blending is a small bounded
count (n = 10), so the curve is demanding at this sample size.

Reparameterised for the thin blending support (#430). At the shared f_mech defaults
(m = 10, lengthscale ``InverseGamma(5, 5)``) the curve diverged 31 times at reporting
tier - a boundary-geometry funnel, not a mixing failure (R-hat / ESS / BFMI all
passed). Fewer basis functions (``mech_hsgp_m = 6``) and a tighter lengthscale prior
(``mech_lengthscale_tight`` -> ``InverseGamma(8, 8)``, the short-lengthscale tail
thinned, mode essentially unchanged) clear it to zero divergences without forcing the
curve flat. The fitted curve is then flat and wide (posterior-mean amplitude ~0.15 on
the logit scale against an ~0.45 89% band, near-zero slope in both halves): the honest
knee-test answer is that **no knee is resolved** for blending at this sample size and
measurement support - unlike the letter-sound knee LRP58 resolves - not that a knee is
absent. Read the curve as "shape unresolved", with the same bounded-scale / logit-link
caveat LRP58 carries.

Adjustment set (revised DAG, 2026-07-10). Derived by a backdoor d-separation search with
the latent GA held (the criterion that reproduces LRP56/58 and dose-077). In the DAG,
PA (blending) has parents {A, GA, HS, IG, IS, TE, EV, SP, LS, RW}; the minimal observed
set that blocks every backdoor to WR is {A, HS, IG, IS, LS, TE, EV, SP, RW}. Crucially
this includes LETTER SOUNDS (LS = L): LS -> PA and LS -> WR make it a PA <- LS -> WR
confounder, so a blending -> reading association must be read *net of* letter-sound
knowledge. Measure confounders L, TE, E enter on their logit scale via ``adjustment``;
the continuous confounders HS (hs), IS (attend), SP (deapp_c) and RW (erbto) enter via
``adjust_for``; group G(=IG) is the always-in precision term and W_pre the
autoregressive baseline. The causal paths PA -> WR and PA -> NW -> WR are preserved
(NW, PS are descendants of PA and are never adjusted).

Residual confounding by latent general ability (GA) remains, so f^B is an ADJUSTED
ASSOCIATION, not a causal effect. The corresponding randomised-arm result is an
available-case modified ITT estimate in the ITT suite.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-190",
    kind="mechanism",
    title="GP knee-test: phoneme blending (B) -> word reading (W)",
    outcome_symbol="W",
    mechanism_symbol="B",
    adjustment=["G", "A", "L", "TE", "E", "W_pre"],
    extra={
        "outcomes": ("W", "B", "L", "TE", "E"),
        "adjust_baseline_symbol": "W",
        "adjust_for": (
            "hs", "hs_missing", "attend", "deapp_c", "deapp_c_missing",
            "erbto", "erbto_missing",
        ),
        "use_age_gp": False,
        "phase_specific_mechanism": False,
        "use_subject_random_intercept": True,
        # HSGP mechanism curve ON (knee-test); target_accept 0.999 per LRP58.
        "target_accept": 0.999,
        # Thin-support HSGP reparameterisation (#430). Blending is a 10-item task with
        # a chance floor ~3.3 and ~19% at ceiling, so the shared f_mech defaults
        # (m=10, ell InverseGamma(5,5)) leave a wiggly, weakly-identified curve that
        # diverged 31 times at reporting tier. Fewer basis functions + a tighter
        # lengthscale (InverseGamma(8,8), short-tail thinned, mode ~unchanged) clear
        # the boundary geometry (0 divergences) without forcing the curve flat.
        "mech_hsgp_m": 6,
        "mech_lengthscale_tight": True,
    },
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
