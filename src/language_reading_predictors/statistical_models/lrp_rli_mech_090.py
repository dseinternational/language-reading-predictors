# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP90 - Mechanism model: phonological memory (RW, word/nonword repetition) ->
word reading (W).

#311 (descriptive-association workstream #314): the phonological-memory
dose-response into word reading - "a higher word/nonword repetition score is
associated with more words read" - as an adjusted association across all three
phase transitions. Completes the #311 trio after LRP88 (TR -> W) and LRP89
(TE -> W).

**Covariate exposure (route (b), notes/202607142100).** The exposure is the ERB
total (``erbto``), an integer score whose documented test maximum is recorded
nowhere in the repo - registering it as a bounded-count ``Measure`` would
fabricate a denominator, and the Beta-Binomial logit machinery is exactly wrong
if the true maximum differs. It therefore enters as a standardised continuous
covariate (``mechanism_is_covariate``): ``beta_mech`` is the adjusted association
per +1 SD of the observed ERB total (the raw-units SD is recorded in
``config.json`` as ``mechanism_exposure_sd_raw``). ``require_observed`` drops the
mean-imputed rows: imputation plus a missingness indicator is an *adjuster*
policy, not acceptable for the exposure itself.

Revised-DAG (2026-07-10) adjustment set for the RW_post -> W_post backdoor. The
parents of RW are {A, GA, HS} - and nothing else:

- A (age): linear ``gamma_A`` term. G (group) is always controlled via beta_G;
  W_pre is the autoregressive baseline.
- HS: hearing status (hs / hs_missing) - covariate adjuster; RW's only measured
  parent. Blocks every ``RW <- HS -> {TR RV TE EV SP PA LS} -> W`` route at its
  root.
- GA (general ability) is latent and unadjustable - the child random intercept
  proxies its time-invariant part, so the slope stays an adjusted association,
  never a causal effect.

**No session-dose backdoor** - unlike LRP88/LRP89. The DAG has no ``IG -> RW``
or ``IS -> RW`` edge: under the locked structure the intervention does not move
phonological memory, so session dose is not a common cause of exposure and
outcome and the IS caveat those models carry does not apply here. RW's variation
is not intervention-generated.

TR, TE, RV, EV, PA, NW and PS are *descendants* of RW (``RW -> {TE EV TR RV PA
NW PS}``) that also affect W: conditioning on any of them would block legitimate
indirect paths (e.g. ``RW -> PA -> NW -> W``) and bias the slope toward the
direct-only component, so they are deliberately NOT in the adjustment set. SP is
neither a parent nor a descendant of RW and is not needed.

Linear mechanism: required for a covariate exposure (the HSGP curve, its priors
and the readiness-threshold post-processing all assume a bounded-count logit
input). The estimand is the LINEAR RW -> W adjusted association (a single slope,
not a shape). ``erbto`` is also measured with error; non-negligible error
attenuates the slope toward zero.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-090",
    kind="mechanism",
    title=(
        "Mechanism model: phonological memory (RW, word/nonword repetition) -> "
        "word reading (W)"
    ),
    outcome_symbol="W",
    mechanism_symbol="erbto",
    adjustment=["G", "A", "W_pre"],
    # Age enters as a linear gamma_A term; the subject random intercept handles the
    # non-independent rows (up to 3 phases x 53 children) and proxies the
    # time-invariant part of latent ability.
    extra={
        # Only the outcome (W) is a bounded-count measure here; the exposure is the
        # erbto covariate, so the measure complete-case mask is W alone.
        "outcomes": ("W",),
        "adjust_baseline_symbol": "W",
        # HS is RW's only measured parent under the revised DAG; SP/RW-adjusters
        # from LRP88/89 do not apply (SP is not a parent of RW, and erbto IS the
        # exposure).
        "adjust_for": ("hs", "hs_missing"),
        # The exposure must be genuinely observed - drop the mean-imputed rows
        # rather than carrying an imputed exposure with a missingness flag.
        "require_observed": ("erbto",),
        # Route (b) of #311: standardised-covariate exposure (no fabricated
        # denominator for the ERB total). Requires the linear mechanism.
        "mechanism_is_covariate": True,
        "linear_mechanism": True,
        "use_age_gp": False,
        "phase_specific_mechanism": False,
        "use_subject_random_intercept": True,
    },
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
