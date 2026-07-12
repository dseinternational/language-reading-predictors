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
}

#: Outcomes that take the pre-specified floor rule (a binary off-floor estimand).
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
    _d("lrpitt16", "joint", "Generalisation", Status.ROBUSTNESS, None, "modality contrast: taught expressive vs taught receptive"),
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

# --- Waitlist-crossover / difference-in-differences -------------------------------
_DID = [
    _d("lrpdid01", "did", "Within-person DiD", Status.ROBUSTNESS, "W", "waitlist-crossover replication", base="lrpitt10"),
    _d("lrpdid02", "did", "Within-person DiD", Status.ROBUSTNESS, "L", "waitlist-crossover replication", base="lrpitt07"),
    _d("lrpdid03", "did", "Within-person DiD", Status.ROBUSTNESS, "B", "waitlist-crossover replication", base="lrpitt08"),
    _d("lrpdid04", "did", "Within-person DiD", Status.ROBUSTNESS, "TE", "waitlist-crossover replication", base="lrpitt02"),
    _d("lrpdid05", "did", "Within-person DiD", Status.ROBUSTNESS, "R", "waitlist-crossover replication", base="lrpitt05"),
    _d("lrpdid06", "did", "Within-person DiD", Status.ROBUSTNESS, "W", "dose-response sensitivity (sessions)", base="lrpdid01"),
    _d("lrpdid07", "did", "Within-person DiD", Status.ROBUSTNESS, "L", "session-dose response (period-resolved)", base="lrpdid02"),
    _d("lrpdid07base", "did", "Within-person DiD", Status.COMPANION, "L", "pooled-dose comparator", base="lrpdid07"),
    # Waitlist-crossover extensions (#226). TR/E replicate their ITT siblings;
    # F (basic concepts) has no ITT sibling (outside the eight standardised
    # outcomes). The floored P/N DiDs (011/012) take the off-floor floor rule -
    # a Bernoulli on the off-floor indicator, so delta is the within-person effect
    # on the log-odds of coming off the floor - mirroring siblings LRPITT09/11.
    _d("lrpdid08", "did", "Within-person DiD", Status.ROBUSTNESS, "TR", "waitlist-crossover replication", base="lrpitt01"),
    _d("lrpdid09", "did", "Within-person DiD", Status.ROBUSTNESS, "E", "waitlist-crossover replication", base="lrpitt06"),
    _d("lrpdid10", "did", "Within-person DiD", Status.ROBUSTNESS, "F", "waitlist-crossover replication", base="lrpitt25"),
    _d("lrpdid11", "did", "Within-person DiD", Status.ROBUSTNESS, "P", "waitlist-crossover replication (off-floor)", base="lrpitt09"),
    _d("lrpdid12", "did", "Within-person DiD", Status.ROBUSTNESS, "N", "waitlist-crossover replication (off-floor)", base="lrpitt11"),
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
    _d("lrp59", "mediation", "Mediation", Status.ASSOCIATION, "W", "g-formula via letter sounds"),
    _d("lrp68", "mediation", "Mediation", Status.ASSOCIATION, "W", "g-formula via taught-expressive vocabulary"),
    _d("lrp80", "mediation", "Mediation", Status.ASSOCIATION, "W", "g-formula via taught-receptive vocabulary (TE companion)"),
    _d("lrp74", "mediation", "Mediation", Status.ASSOCIATION, "W", "g-formula via nonword decoding (floor-limited)"),
    _d("lrp62", "mediation", "Mediation", Status.ASSOCIATION, "W", "reading-route composite mediation"),
    _d("lrp64", "mediation_multi", "Mediation", Status.ASSOCIATION, "W", "two-mediator decomposition (letter sounds vs expressive vocabulary)"),
    _d("lrp66", "mediation_multi", "Mediation", Status.ASSOCIATION, "W", "two-mediator decomposition (letter sounds vs phoneme blending)"),
    _d("lrp75", "mediation_multi", "Mediation", Status.ASSOCIATION, "W", "sequential code route (letter sounds -> blending -> reading)"),
    _d("lrp76", "mediation", "Mediation", Status.ASSOCIATION, "W", "longitudinal-ordering (letter sounds t2 -> reading t4)"),
    _d("lrp78", "mediation", "Mediation", Status.ASSOCIATION, "W", "interventional-effects decomposition via letter sounds"),
    _d("lrp79", "mediation", "Mediation", Status.ASSOCIATION, "W", "negative-control mediator (grammar; calibrates GA confounding)"),
]

# --- DAG-focused associations, dose-response, latent structure and cross-checks ----
_STRUCT = [
    _d("lrp65", "adjusted", "Adjusted association", Status.ASSOCIATION, "W", "between-child independent baseline predictors of word-reading gain"),
    _d("lrp67", "lcsm", "Latent change score", Status.ASSOCIATION, "W", "coupled letter-sounds and vocabulary predicting reading change"),
    _d("lrp69", "growth", "Growth curve", Status.ASSOCIATION, None, "joint multivariate growth curves: baseline non-verbal ability predicting trajectory shape (independent-core)"),
    _d("lrp70", "growth", "Growth curve", Status.ASSOCIATION, None, "joint multivariate growth curves with a shared growth-tempo factor", base="lrp69"),
    _d("lrp77", "dose_response", "Dose-response", Status.ASSOCIATION, "W", "period-resolved intervention dose -> word reading"),
    _d("lrp77a", "dose_response", "Dose-response", Status.ROBUSTNESS, "W", "ability-adjusted sensitivity", base="lrp77"),
    _d("lrp77base", "dose_response", "Dose-response", Status.COMPANION, "W", "pooled-dose-slope comparator", base="lrp77"),
    _d("lrpmm01", "corr_factor", "Measurement model", Status.ASSOCIATION, "W", "correlated-domain-factor measurement model (vocabulary / code / grammar)"),
    _d("lrpmm101", "corr_factor", "Measurement model", Status.ASSOCIATION, "W", "prior sensitivity for LRPMM01 (recalibrated loading / residual priors)", base="lrpmm01"),
    _d("lrphs01", "horseshoe", "Horseshoe ranking", Status.ASSOCIATION, "W", "regularised-horseshoe ranking cross-check (word-reading gain)"),
    _d("lrphs02", "horseshoe", "Horseshoe ranking", Status.ASSOCIATION, "W", "regularised-horseshoe ranking cross-check (word-reading level)"),
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

# Suite-gap Tier-1 additions (#228): standalone ITTs for the two outcomes that had
# only factor/aligned models (F basic concepts, T receptive grammar), and a site
# (area) robustness check on the flagship word-reading (W) and letter-sound (L) ITTs.
_ITT_TIER1 = [
    _d("lrpitt25", "itt", "ITT suite", Status.MODEL_OF_RECORD, "F", "randomised ITT effect"),
    _d("lrpitt26", "itt", "ITT suite", Status.MODEL_OF_RECORD, "T", "randomised ITT effect"),
    _d("lrpitt27", "itt", "Site robustness", Status.ROBUSTNESS, "W", "site (area) adjustment", base="lrpitt10"),
    _d("lrpitt28", "itt", "Site robustness", Status.ROBUSTNESS, "L", "site (area) adjustment", base="lrpitt07"),
]


#: The register: every fitted model, keyed by id. Must match the fit script's MODELS.
MODEL_REGISTRY: dict[str, ModelDefinition] = {
    d.model_id: d
    for d in (*_ITT, *_JOINT, *_SES, *_ABIL, *_DID, *_MECH, *_STRUCT, *_GAIN, *_GAINB, *_LEVEL, *_ALIGNED, *_ITT_TIER1)
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
