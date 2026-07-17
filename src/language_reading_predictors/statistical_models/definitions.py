# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Lightweight registry of the Bayesian statistical models.

A pure-data description of every fitted model — id, kind, outcome, family and
editorial status — with **no heavy imports** (no PyMC, no factories), so it can be
imported cheaply by the report (to code-generate a model register) and by any
tooling that needs the catalogue without paying to import the model-building code.

This mirrors the vocabulary-growth report's ``vocab_growth.models.definitions``. It
deliberately duplicates the lightweight metadata that otherwise lives on each
module's ``SPEC`` (the ``ModelSpec`` in ``context.py``); the test
``tests/test_model_definitions.py`` guards the two against drift.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from language_reading_predictors import model_ids


class Status(str, Enum):
    """Editorial status of a model in the report register (the status taxonomy)."""

    MODEL_OF_RECORD = "Model of record"
    JOINT = "Joint"
    ROBUSTNESS = "Robustness"
    ASSOCIATION = "Association"
    COMPANION = "Companion"


#: Outcome symbol -> human label. The suite models a set of bounded-count measures.
OUTCOMES: dict[str, str] = {
    "TR": "taught receptive vocabulary",
    "TE": "taught expressive vocabulary",
    "UR": "not-taught receptive vocabulary",
    "UE": "not-taught expressive vocabulary",
    "R": "receptive vocabulary",
    "E": "expressive vocabulary",
    "L": "letter-sound knowledge",
    "B": "phoneme blending",
    "P": "phonetic spelling",
    "W": "word reading",
    "N": "nonword reading",
    "F": "CELF basic concepts",
    "T": "TROG receptive grammar",
    "TE2": "taught expressive vocabulary, block 2",
    "TR2": "taught receptive vocabulary, block 2",
    "UE2": "not-taught expressive vocabulary, block 2",
    "UR2": "not-taught receptive vocabulary, block 2",
}

#: Outcomes that take the post-hoc reanalysis floor rule (a binary off-floor estimand).
FLOORED: frozenset[str] = frozenset({"P", "N"})

#: Valid model kinds (the statistical-model families).
KINDS: frozenset[str] = frozenset(
    {
        "itt",
        "joint",
        "mechanism",
        "mediation",
        "did",
        "gain_factors",
        "level_factors",
        "aligned",
        "adjusted",
        "corr_factor",
        "dose_response",
        "lcsm",
        "mediation_multi",
        "horseshoe",
        "growth",
        "historical_growth",
        "historical_joint",
        "survival",
        "block_exposure",
        "concurrent",
        "long_corr_factor",
    }
)


@dataclass(frozen=True)
class ModelDefinition:
    """Lightweight, pure-data description of one model."""

    model_id: str
    kind: str
    family: str
    status: Status
    outcome: str | None = None
    role: str = ""
    base: str | None = None
    """The model this one varies / derives from, if any (e.g. an adjustment or a
    companion baseline)."""

    @property
    def outcome_label(self) -> str | None:
        return OUTCOMES.get(self.outcome) if self.outcome else None

    @property
    def floored(self) -> bool:
        return self.outcome in FLOORED


def _d(*args, **kwargs) -> ModelDefinition:
    # Entries are written with legacy ids for brevity; canonicalise to the #168
    # scheme here (the one choke point through which literal *and* generated ids
    # pass), so the registry keys match each module's canonical ``SPEC.model_id``.
    # ``base`` shares the entry's family/kind, so the same ``kind`` canonicalises it.
    d = ModelDefinition(*args, **kwargs)
    return replace(
        d,
        model_id=model_ids.to_canonical(d.model_id, kind=d.kind),
        base=model_ids.to_canonical(d.base, kind=d.kind) if d.base else None,
    )


# --- ITT suite: one outcome each, the randomised models of record -----------------
_ITT = [
    _d("lrpitt01", "itt", "ITT suite", Status.MODEL_OF_RECORD, "TR", "randomised ITT effect"),
    _d("lrpitt02", "itt", "ITT suite", Status.MODEL_OF_RECORD, "TE", "randomised ITT effect"),
    _d("lrpitt03", "itt", "ITT suite", Status.MODEL_OF_RECORD, "UR", "randomised ITT effect"),
    _d("lrpitt04", "itt", "ITT suite", Status.MODEL_OF_RECORD, "UE", "randomised ITT effect"),
    _d("lrpitt05", "itt", "ITT suite", Status.MODEL_OF_RECORD, "R", "randomised ITT effect"),
    _d("lrpitt06", "itt", "ITT suite", Status.MODEL_OF_RECORD, "E", "randomised ITT effect"),
    _d("lrpitt07", "itt", "ITT suite", Status.MODEL_OF_RECORD, "L", "randomised ITT effect"),
    _d("lrpitt08", "itt", "ITT suite", Status.MODEL_OF_RECORD, "B", "randomised ITT effect"),
    _d("lrpitt09", "itt", "ITT suite", Status.MODEL_OF_RECORD, "P", "floor-rule branch"),
    _d("lrpitt10", "itt", "ITT suite", Status.MODEL_OF_RECORD, "W", "randomised ITT effect (primary outcome)"),
    _d("lrpitt11", "itt", "ITT suite", Status.MODEL_OF_RECORD, "N", "floor-rule branch"),
]

# --- Joint + generalisation contrasts ---------------------------------------------
_JOINT = [
    _d("lrpitt12", "joint", "Joint", Status.JOINT, None, "cross-outcome consistency + contrasts"),
    _d("lrpitt15", "joint", "Generalisation", Status.ROBUSTNESS, None, "taught vs not-taught (expressive)"),
    _d("lrpitt15b", "joint", "Generalisation", Status.ROBUSTNESS, None, "taught vs not-taught (receptive)"),
    _d("lrpitt16", "joint", "Modality contrast", Status.ROBUSTNESS, None, "taught expressive vs taught receptive"),
]

# --- SES adjustment + matched complete-case comparators ---------------------------
_SES = [
    _d("lrpitt13", "itt", "SES + comparators", Status.ROBUSTNESS, "W", "SES-adjusted", base="lrpitt10"),
    _d("lrpitt13b", "itt", "SES + comparators", Status.ROBUSTNESS, "L", "SES-adjusted", base="lrpitt07"),
    _d("lrpitt14", "itt", "SES + comparators", Status.ROBUSTNESS, "W", "matched complete-case comparator", base="lrpitt13"),
    _d("lrpitt14b", "itt", "SES + comparators", Status.ROBUSTNESS, "L", "matched complete-case comparator", base="lrpitt13b"),
]

# --- General-ability (block-design) robustness ------------------------------------
_ABIL = [
    _d("lrpitt17", "itt", "General-ability adjustment", Status.ROBUSTNESS, "TR", "block-design adjustment", base="lrpitt01"),
    _d("lrpitt18", "itt", "General-ability adjustment", Status.ROBUSTNESS, "TE", "block-design adjustment", base="lrpitt02"),
    _d("lrpitt19", "itt", "General-ability adjustment", Status.ROBUSTNESS, "UR", "block-design adjustment", base="lrpitt03"),
    _d("lrpitt20", "itt", "General-ability adjustment", Status.ROBUSTNESS, "UE", "block-design adjustment", base="lrpitt04"),
    _d("lrpitt21", "itt", "General-ability adjustment", Status.ROBUSTNESS, "R", "block-design adjustment", base="lrpitt05"),
    _d("lrpitt22", "itt", "General-ability adjustment", Status.ROBUSTNESS, "E", "block-design adjustment", base="lrpitt06"),
    _d("lrpitt23", "itt", "General-ability adjustment", Status.ROBUSTNESS, "L", "block-design adjustment", base="lrpitt07"),
    _d("lrpitt24", "itt", "General-ability adjustment", Status.ROBUSTNESS, "W", "block-design adjustment", base="lrpitt10"),
]

# --- Waitlist-crossover arm-by-wave models ----------------------------------------
_DID = [
    _d("lrpdid01", "did", "Arm-by-wave crossover", Status.ROBUSTNESS, "W", "randomised t2 contrast plus post-crossover arm gaps", base="lrpitt10"),
    _d("lrpdid02", "did", "Arm-by-wave crossover", Status.ROBUSTNESS, "L", "randomised t2 contrast plus post-crossover arm gaps", base="lrpitt07"),
    _d("lrpdid03", "did", "Arm-by-wave crossover", Status.ROBUSTNESS, "B", "randomised t2 contrast plus post-crossover arm gaps", base="lrpitt08"),
    _d("lrpdid04", "did", "Arm-by-wave crossover", Status.ROBUSTNESS, "TE", "randomised t2 contrast plus post-crossover arm gaps", base="lrpitt02"),
    _d("lrpdid05", "did", "Arm-by-wave crossover", Status.ROBUSTNESS, "R", "randomised t2 contrast plus post-crossover arm gaps", base="lrpitt05"),
    _d("lrpdid06", "did", "Crossover dose association", Status.ASSOCIATION, "W", "pooled treated-centred session association", base="lrpdid01"),
    _d("lrpdid07", "did", "Crossover dose association", Status.ASSOCIATION, "L", "period-resolved treated-centred session association", base="lrpdid02"),
    _d("lrpdid07base", "did", "Crossover dose association", Status.COMPANION, "L", "pooled treated-centred session comparator", base="lrpdid07"),
    # Waitlist-crossover extensions (#226). The floored P/N models fit off-floor
    # prevalence at t1/t2/t3; their clean t2 arm gaps complement, but do not duplicate,
    # the ITT siblings' baseline-floor-risk transition estimands.
    _d("lrpdid08", "did", "Arm-by-wave crossover", Status.ROBUSTNESS, "TR", "randomised t2 contrast plus post-crossover arm gaps", base="lrpitt01"),
    _d("lrpdid09", "did", "Arm-by-wave crossover", Status.ROBUSTNESS, "E", "randomised t2 contrast plus post-crossover arm gaps", base="lrpitt06"),
    _d("lrpdid10", "did", "Arm-by-wave crossover", Status.ROBUSTNESS, "F", "randomised t2 contrast plus post-crossover arm gaps", base="lrpitt25"),
    _d("lrpdid11", "did", "Arm-by-wave crossover", Status.ROBUSTNESS, "P", "off-floor prevalence by arm and wave", base="lrpitt09"),
    _d("lrpdid12", "did", "Arm-by-wave crossover", Status.ROBUSTNESS, "N", "off-floor prevalence by arm and wave", base="lrpitt11"),
    # Exploratory unexplained variation in the waitlist arm's t3 catch-up. This is
    # not a random treatment-effect slope and cannot classify causal responders.
    _d("lrpdid13", "did", "Arm-by-wave crossover", Status.ASSOCIATION, "W", "exploratory waitlist t3 catch-up heterogeneity", base="lrpdid01"),
]

# --- Mechanism / moderation / mediation (adjusted associations) -------------------
_MECH = [
    _d("lrp56", "mechanism", "Mechanism", Status.ASSOCIATION, "W", "receptive vocabulary -> word reading"),
    _d("lrp57", "mechanism", "Mechanism", Status.ASSOCIATION, "W", "expressive vocabulary -> word reading"),
    _d("lrp58", "mechanism", "Mechanism", Status.ASSOCIATION, "W", "letter sounds -> word reading"),
    _d("lrp158", "mechanism", "Mechanism", Status.COMPANION, "W", "complete-case comparator (no imputed confounders)", base="lrp58"),
    _d("lrp71", "mechanism", "Moderation", Status.ASSOCIATION, "W", "expressive-vocabulary moderation"),
    _d("lrp72", "mechanism", "Moderation", Status.ASSOCIATION, "N", "phonics route (letter-sound x blending -> decoding)"),
    _d("lrp72base", "mechanism", "Moderation", Status.COMPANION, "N", "no-interaction baseline", base="lrp72"),
    _d("lrp73", "mechanism", "Moderation", Status.ASSOCIATION, "W", "age-moderated letter-sound -> word reading"),
    _d("lrp73base", "mechanism", "Moderation", Status.COMPANION, "W", "no-interaction baseline", base="lrp73"),
    # Taught-vocabulary dose-response (#311, descriptive-association workstream #314).
    _d("lrp88", "mechanism", "Mechanism", Status.ASSOCIATION, "W", "taught receptive vocabulary -> word reading"),
    _d("lrp89", "mechanism", "Mechanism", Status.ASSOCIATION, "W", "taught expressive vocabulary -> word reading"),
    _d("lrp90", "mechanism", "Mechanism", Status.ASSOCIATION, "W", "phonological memory (word/nonword repetition) -> word reading"),
    # GP knee-test variants (do the vocabulary / blending / dose curves have a knee,
    # as letter sounds do?). Re-attempt the HSGP curve the linear models could not fit.
    _d("lrp156", "mechanism", "Mechanism", Status.ASSOCIATION, "W", "receptive vocabulary -> word reading (GP knee-test)", base="lrp56"),
    _d("lrp157", "mechanism", "Mechanism", Status.ASSOCIATION, "W", "expressive vocabulary -> word reading (GP knee-test)", base="lrp57"),
    _d("lrp188", "mechanism", "Mechanism", Status.ASSOCIATION, "W", "taught receptive vocabulary -> word reading (GP knee-test)", base="lrp88"),
    _d("lrp189", "mechanism", "Mechanism", Status.ASSOCIATION, "W", "taught expressive vocabulary -> word reading (GP knee-test)", base="lrp89"),
    _d("lrp91", "mechanism", "Mechanism", Status.ASSOCIATION, "W", "phoneme blending -> word reading (GP knee-test)"),
    _d("lrp92", "mechanism", "Mechanism", Status.ASSOCIATION, "W", "intervention sessions -> word reading (GP dose knee-test)"),
    # Joint-readiness: letter sounds x vocabulary interaction (do both need to be high?).
    # Companions to LRP71 (L x E); one per vocabulary measure.
    _d("lrp93", "mechanism", "Moderation", Status.ASSOCIATION, "W", "letter sounds x receptive-vocabulary interaction"),
    _d("lrp94", "mechanism", "Moderation", Status.ASSOCIATION, "W", "letter sounds x taught-receptive-vocabulary interaction"),
    _d("lrp95", "mechanism", "Moderation", Status.ASSOCIATION, "W", "letter sounds x taught-expressive-vocabulary interaction"),
    # Tier-1 decoding-specificity mini-suite (notes/202607172330-tier1-decoding-specificity-spec.md):
    # matched *linear* letter-sound slopes for the L->N vs L->W convergent-discriminant
    # contrast (1A) and the negative-control-outcome panel (1B). All linear_mechanism so
    # the cross-outcome forest/contrast is like-for-like.
    _d("lrp96", "mechanism", "Mechanism", Status.ASSOCIATION, "N", "decoding channel: letter sounds -> nonword decoding (1A contrast vs lrp101)"),
    _d("lrp101", "mechanism", "Mechanism", Status.ASSOCIATION, "W", "linear letter sounds -> word reading (Tier-1 contrast/panel anchor; linear counterpart of the HSGP lrp58)"),
    _d("lrp97", "mechanism", "Mechanism", Status.ASSOCIATION, "R", "negative-control outcome: letter sounds -> receptive vocabulary"),
    _d("lrp98", "mechanism", "Mechanism", Status.ASSOCIATION, "E", "negative-control outcome: letter sounds -> expressive vocabulary"),
    _d("lrp99", "mechanism", "Mechanism", Status.ASSOCIATION, "T", "negative-control outcome: letter sounds -> receptive grammar"),
    _d("lrp100", "mechanism", "Mechanism", Status.ASSOCIATION, "F", "negative-control outcome: letter sounds -> basic concepts"),
    _d("lrp59", "mediation", "Mediation", Status.ASSOCIATION, "W", "g-formula via letter sounds"),
    _d("lrp68", "mediation", "Mediation", Status.ASSOCIATION, "W", "g-formula via taught-expressive vocabulary"),
    _d("lrp80", "mediation", "Mediation", Status.ASSOCIATION, "W", "g-formula via taught-receptive vocabulary (TE companion)"),
    _d("lrp74", "mediation", "Mediation", Status.ASSOCIATION, "W", "g-formula via nonword decoding (floor-limited)"),
    _d("lrp62", "mediation", "Mediation", Status.ASSOCIATION, "W", "reading-route composite mediation"),
    _d("lrp64", "mediation_multi", "Mediation", Status.ASSOCIATION, "W", "two-mediator decomposition (letter sounds vs expressive vocabulary)"),
    _d("lrp66", "mediation_multi", "Mediation", Status.ASSOCIATION, "W", "two-mediator decomposition (letter sounds vs phoneme blending)"),
    _d("lrp75", "mediation_multi", "Mediation", Status.ASSOCIATION, "W", "sequential code route (letter sounds -> blending -> reading)"),
    _d("lrp76", "mediation", "Mediation", Status.ASSOCIATION, "W", "longitudinal-ordering (letter sounds t2 -> reading t4)"),
    _d("lrp176", "mediation", "Mediation", Status.ASSOCIATION, "L", "reverse longitudinal-ordering (word reading t2 -> letter sounds t4); WR->LS direction contrast to lrp76", base="lrp76"),
    _d("lrp276", "mediation", "Mediation", Status.ASSOCIATION, "L", "reverse WR->LS with t3 outcome (less-ceilinged sensitivity to lrp176)", base="lrp176"),
    _d("lrp78", "mediation", "Mediation", Status.ASSOCIATION, "W", "interventional-effects decomposition via letter sounds"),
    _d("lrp79", "mediation", "Mediation", Status.ASSOCIATION, "W", "negative-control mediator (grammar; calibrates GA confounding)"),
    # Code route beyond word reading (#228 item 12): the purest decoding outcome
    # (nonword N, off-floor) and the downstream chain link (blending B), both via L.
    _d("lrp86", "mediation", "Mediation", Status.ASSOCIATION, "N", "g-formula via letter sounds (nonword, off-floor risk difference)"),
    _d("lrp87", "mediation", "Mediation", Status.ASSOCIATION, "B", "g-formula via letter sounds (phoneme blending)"),
    # Interventional-estimand companions (#323), using the suite's parent+100
    # convention. Same fitted models/adjustment sets as MED-086/087; IDE/IIE
    # replace the cross-world NDE/NIE interpretation.
    _d("lrp186", "mediation", "Mediation", Status.COMPANION, "N", "interventional-effects analogue via letter sounds (nonword, off-floor risk difference)", base="lrp86"),
    _d("lrp187", "mediation", "Mediation", Status.COMPANION, "B", "interventional-effects analogue via letter sounds (phoneme blending)", base="lrp87"),
    # Period-stacked companion (#229 recommendation 2): the LRP59 design on the
    # gain-factor scaffold, exposure = per-period on-intervention (ignorability).
    _d("lrp92", "mediation", "Mediation", Status.ASSOCIATION, "W", "period-stacked g-formula via letter sounds (gain-factor scaffold, per-period on-intervention exposure)", base="lrp59"),
]

# --- DAG-focused associations, dose-response, latent structure and cross-checks ----
_STRUCT = [
    _d("lrp65", "adjusted", "Adjusted association", Status.ASSOCIATION, "W", "between-child independent baseline predictors of word-reading gain"),
    _d("lrp67", "lcsm", "Latent change score", Status.ASSOCIATION, "W", "coupled letter-sounds and vocabulary predicting reading change"),
    # Lagged reverse-coupling suite on the time-lagged DAG (#250; design
    # notes/202607141030-time-lagged-model-designs.md).
    _d("lrp81", "lcsm", "Latent change score", Status.ASSOCIATION, "TE", "lagged reverse coupling: prior word reading predicting taught-vocabulary change (W->TE, W->TR)"),
    _d("lrp181", "lcsm", "Latent change score", Status.COMPANION, "TE", "no-reverse-coupling LOO comparator", base="lrp81"),
    _d("lrp82", "lcsm", "Latent change score", Status.ASSOCIATION, "W", "reciprocal dominance: blending <-> word reading lagged cross-couplings (exploratory)"),
    # Change-on-change extension (#229 spec 2, notes/202607131530-lrp229-lcsm-change-change-spec.md).
    _d("lrp91", "lcsm", "Latent change score", Status.ASSOCIATION, "W", "lagged change-on-change: prior letter-sound / vocabulary change predicting reading change (exploratory)", base="lrp67"),
    _d("lrp69", "growth", "Growth curve", Status.ASSOCIATION, None, "joint multivariate growth curves: baseline non-verbal ability predicting trajectory shape (independent-core)"),
    _d("lrp70", "growth", "Growth curve", Status.ASSOCIATION, None, "joint multivariate growth curves with a shared growth-tempo factor", base="lrp69"),
    _d("lrp85", "growth", "Growth curve", Status.ASSOCIATION, None, "age x ability interaction on growth rate (older-and-more-able progress more)", base="lrp69"),
    _d("lrp77", "dose_response", "Dose-response", Status.ASSOCIATION, "W", "period-resolved intervention dose -> word reading"),
    _d("lrp77a", "dose_response", "Dose-response", Status.ROBUSTNESS, "W", "ability-adjusted sensitivity", base="lrp77"),
    _d("lrp77base", "dose_response", "Dose-response", Status.COMPANION, "W", "pooled-dose-slope comparator", base="lrp77"),
    # Dose-response coverage for the two largest ITT effects, L and B (#228 item 2);
    # the W family covers only word reading. Same observational IS->outcome estimand.
    _d("lrp83", "dose_response", "Dose-response", Status.ASSOCIATION, "L", "period-resolved intervention dose -> letter sounds"),
    _d("lrp84", "dose_response", "Dose-response", Status.ASSOCIATION, "B", "period-resolved intervention dose -> phoneme blending"),
    _d("lrpmm01", "corr_factor", "Measurement model", Status.ASSOCIATION, "W", "correlated-domain-factor measurement model (vocabulary / code / grammar)"),
    _d("lrpmm101", "corr_factor", "Measurement model", Status.ASSOCIATION, "W", "prior sensitivity for LRPMM01 (recalibrated loading / residual priors)", base="lrpmm01"),
    _d("lrpmm02", "corr_factor", "Measurement model", Status.ASSOCIATION, "W", "errors-in-variables code->word-reading mechanism slope (latent code factor, mech-058 adjustment)", base="lrpmm01"),
    _d("lrphs01", "horseshoe", "Horseshoe ranking", Status.ASSOCIATION, "W", "regularised-horseshoe ranking cross-check (word-reading gain)"),
    _d("lrphs02", "horseshoe", "Horseshoe ranking", Status.ASSOCIATION, "W", "regularised-horseshoe ranking cross-check (word-reading level)"),
    # Ranking cross-check for the flagship letter-sound outcome L (#228 item 3);
    # hs-001/002 cover word reading only. Cross-checked vs GB gbg-009 / gbl-009.
    _d("lrphs03", "horseshoe", "Horseshoe ranking", Status.ASSOCIATION, "L", "regularised-horseshoe ranking cross-check (letter-sound gain)"),
    _d("lrphs04", "horseshoe", "Horseshoe ranking", Status.ASSOCIATION, "L", "regularised-horseshoe ranking cross-check (letter-sound level)"),
]

# --- Factor families (gain / level) and onset-aligned per-protocol ----------------
_FACTOR_OUTCOMES = ["W", "R", "E", "L", "P", "B", "F", "T"]  # numbering 01..08

_GAIN = [
    _d(f"lrpgf{i:02d}", "gain_factors", "Gain factors", Status.ASSOCIATION, o,
       "ANCOVA gain; only the on-intervention term is causal")
    for i, o in enumerate(_FACTOR_OUTCOMES, 1)
]
_GAINB = [
    _d(f"lrpgf{i:02d}b", "gain_factors", "Gain factors", Status.COMPANION, o,
       "treated-only companion", base=f"lrpgf{i:02d}")
    for i, o in enumerate(_FACTOR_OUTCOMES, 1)
]
_LEVEL = [
    _d(f"lrplf{i:02d}", "level_factors", "Level factors", Status.ASSOCIATION, o,
       "levels view; only the t2 group contrast is randomised")
    for i, o in enumerate(_FACTOR_OUTCOMES, 1)
]
_ALIGNED = [
    _d(f"lrpal{i:02d}", "aligned", "Aligned per-protocol", Status.ASSOCIATION, o,
       "onset-aligned; no term is causal (age-at-onset confound)")
    for i, o in enumerate(_FACTOR_OUTCOMES, 1)
]
_ALIGNED.append(
    _d("lrpal01d", "aligned", "Aligned per-protocol", Status.ASSOCIATION, "W",
       "cumulative-session dose sensitivity (collider)", base="lrpal01")
)

# Taught-vocabulary factor models (#224, carried forward on the revised DAG in #247).
# Added explicitly rather than via ``_FACTOR_OUTCOMES`` so only the stock gain/level
# models are generated for TR/TE - the treated-only (``…b``) and onset-aligned families
# are out of scope.
_GAIN += [
    _d("lrpgf09", "gain_factors", "Gain factors", Status.ASSOCIATION, "TR",
       "ANCOVA gain; only the on-intervention term is causal"),
    _d("lrpgf10", "gain_factors", "Gain factors", Status.ASSOCIATION, "TE",
       "ANCOVA gain; only the on-intervention term is causal"),
]
_LEVEL += [
    _d("lrplf09", "level_factors", "Level factors", Status.ASSOCIATION, "TR",
       "levels view; only the t2 group contrast is randomised"),
    _d("lrplf10", "level_factors", "Level factors", Status.ASSOCIATION, "TE",
       "levels view; only the t2 group contrast is randomised"),
]

# Nonword-reading factor models (#225, carried forward on the revised DAG in #247).
# Off-floor Bernoulli likelihood like gf-005/lf-005 (phonetic spelling); added
# explicitly, same rationale as the taught-vocabulary entries above.
_GAIN += [
    _d("lrpgf11", "gain_factors", "Gain factors", Status.ASSOCIATION, "N",
       "ANCOVA gain; only the on-intervention term is causal"),
]
_LEVEL += [
    _d("lrplf11", "level_factors", "Level factors", Status.ASSOCIATION, "N",
       "levels view; only the t2 group contrast is randomised"),
]

# Suite-gap Tier-1 additions (#228): standalone ITTs for the two outcomes that had
# only factor/aligned models (F basic concepts, T receptive grammar), and a site
# (area) robustness check on the flagship word-reading (W) and letter-sound (L) ITTs.
_ITT_TIER1 = [
    _d("lrpitt25", "itt", "ITT suite", Status.MODEL_OF_RECORD, "F", "randomised ITT effect"),
    _d("lrpitt26", "itt", "ITT suite", Status.MODEL_OF_RECORD, "T", "randomised ITT effect"),
    _d("lrpitt27", "itt", "Site robustness", Status.ROBUSTNESS, "W", "site (area) adjustment", base="lrpitt10"),
    _d("lrpitt28", "itt", "Site robustness", Status.ROBUSTNESS, "L", "site (area) adjustment", base="lrpitt07"),
]

# Time-to-off-floor survival family (#230 §5): the four-wave generalisation of the
# floored P/N off-floor rule (siblings LRPITT09/11) — a discrete-time hazard for *when*
# a floor-sitter comes off the floor. Prognostic (both arms treated by t4), so the
# treatment hazard shift is an association anchored on the immediate arm's randomised
# window, not a randomised effect of record.
_SURV = [
    _d("lrpsurv09", "survival", "Floor-sitter survival", Status.ASSOCIATION, "P", "time-to-off-floor discrete-time hazard", base="lrpitt09"),
    _d("lrpsurv11", "survival", "Floor-sitter survival", Status.ASSOCIATION, "N", "time-to-off-floor discrete-time hazard", base="lrpitt11"),
]

# Block-2 taught-vocabulary block-active exposure family (#228 item 5): block 2 has no
# t1 baseline and no randomised contrast, so a staggered-adoption exposure association
# (immediate arm taught block 2 in phase 2 while the wait-list is still on block 1;
# wait-list reaches block 2 in phase 3). TE2 is the informative (expressive) outcome;
# TR2 is near-ceiling; UE2/UR2 are the not-taught specificity comparators.
_BX = [
    _d("lrpbx01", "block_exposure", "Block-2 exposure", Status.ASSOCIATION, "TE2", "staggered block-2 exposure; association (parallel trends)"),
    _d("lrpbx02", "block_exposure", "Block-2 exposure", Status.ASSOCIATION, "TR2", "staggered block-2 exposure; association (parallel trends)"),
    _d("lrpbx03", "block_exposure", "Block-2 exposure", Status.ASSOCIATION, "UE2", "not-taught comparator (specificity)"),
    _d("lrpbx04", "block_exposure", "Block-2 exposure", Status.ASSOCIATION, "UR2", "not-taught comparator (specificity)"),
]

# --- Concurrent conditional associations (#312, descriptive-association workstream #314)
# Per-wave mutually-adjusted associations between contemporaneous skill levels. Every
# term is an association (post-treatment conditioning is intentional; no causal claim).
_CA = [
    _d("lrpca01", "concurrent", "Concurrent associations", Status.ASSOCIATION, "W", "per-wave conditional associations of concurrent skills with word reading"),
    _d("lrpca02", "concurrent", "Concurrent associations", Status.ASSOCIATION, "L", "per-wave conditional associations of concurrent skills with letter sounds"),
    _d("lrpca03", "concurrent", "Concurrent associations", Status.ASSOCIATION, "TR", "per-wave conditional associations of concurrent skills with taught receptive vocabulary"),
    _d("lrpca04", "concurrent", "Concurrent associations", Status.ASSOCIATION, "TE", "per-wave conditional associations of concurrent skills with taught expressive vocabulary"),
    _d("lrpca05", "concurrent", "Concurrent associations", Status.ASSOCIATION, "R", "per-wave conditional associations of concurrent skills with standardised receptive vocabulary (ROWPVT)"),
    _d("lrpca06", "concurrent", "Concurrent associations", Status.ASSOCIATION, "E", "per-wave conditional associations of concurrent skills with standardised expressive vocabulary (EOWPVT)"),
]

# --- Longitudinal correlated-domain-factor model (#313, descriptive-association #314)
# Per-wave latent skill correlations (vocabulary {R,E,TR,TE} / code {L,B} / grammar
# {F,T}) over the four-wave panel: a measurement-error-aware companion to the
# concurrent regression family (_CA), with symmetric correlations and directional
# conditional slopes. Every quantity is a descriptive association.
_LCF = [
    _d("lrplcf01", "long_corr_factor", "Measurement model", Status.ASSOCIATION, None, "longitudinal correlated-domain-factor model (per-wave latent skill correlations)"),
]


#: The register: every fitted model, keyed by id. Must match the fit script's MODELS.
MODEL_REGISTRY: dict[str, ModelDefinition] = {
    d.model_id: d
    for d in (*_ITT, *_JOINT, *_SES, *_ABIL, *_DID, *_MECH, *_STRUCT, *_GAIN, *_GAINB, *_LEVEL, *_ALIGNED, *_ITT_TIER1, *_SURV, *_BX, *_CA, *_LCF)
}


# --- Provenance: deleted predecessors and reserved / deferred models --------------
# Not built (no module); recorded for the supersession map and roadmap only.
#
# These are *pre-#168* numeric ids. Some numbers were later reused by unrelated
# canonical models — 70 by the growth curve ``lrp-rli-gc-070``, and 74/75/76 by the
# mediation models ``lrp-rli-med-074/075/076`` — so a bare "LRP74" here is NOT the
# live ``lrp-rli-med-074``. To keep that unambiguous the reused keys carry a
# ``[pre-#168 …]`` qualifier so they no longer read as, or match, a live legacy
# alias; ``provenance_alias_collisions()`` guards against any future bare-id reuse.
SUPERSEDED: dict[str, str] = {
    "LRP52": "lrpitt10",   # word reading
    "LRP53": "lrpitt05",   # receptive vocabulary
    "LRP54": "lrpitt06",   # expressive vocabulary
    "LRP74 [pre-#168 taught-expressive ITT]": "lrpitt02",  # ≠ live lrp-rli-med-074
    "LRP75 [pre-#168 taught-receptive ITT]": "lrpitt01",   # ≠ live lrp-rli-med-075
    "LRP55": "lrpitt12",   # joint
    "LRP60": "lrpitt13",   # SES (word reading)
    "LRP60a": "lrpitt14",  # SES matched comparator
    "LRP76 [pre-#168 generalisation contrast]": "lrpitt15",  # ≠ live lrp-rli-med-076
}

RESERVED: dict[str, str] = {
    "LRP16": "descriptive developmental-trajectory model (the trajectory question is the companion vocabulary-growth report)",
    "LRP70 [pre-#168 reserved]": "CELF-moderated mechanism, deferred pending a DAG review of conditioning on a descendant of letter-sound knowledge (the number 70 is now the live growth curve lrp-rli-gc-070)",
}


def provenance_alias_collisions() -> list[str]:
    """Provenance keys that are also a *live* model's legacy alias (should be empty).

    A bare ``LRPnn``/``LRPnnx`` key in ``SUPERSEDED`` / ``RESERVED`` that equals a
    live model's legacy alias is ambiguous: the same string resolves to two
    different models. Qualified keys (e.g. ``"LRP74 [pre-#168 …]"``) do not match
    the bare pattern and are therefore skipped. ``tests`` asserts this is empty so
    a future entry cannot silently collide (issue #273).
    """
    import re

    bare = re.compile(r"^lrp\d+[a-z]?$")
    live_aliases = {model_ids.to_legacy(mid).lower() for mid in MODEL_REGISTRY}
    return [
        key
        for key in (*SUPERSEDED, *RESERVED)
        if bare.match(key.lower()) and key.lower() in live_aliases
    ]


def models_by_status(status: Status) -> list[ModelDefinition]:
    """All registered models with a given status, in registry order."""
    return [d for d in MODEL_REGISTRY.values() if d.status is status]
