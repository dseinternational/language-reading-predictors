# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Lightweight registry of the Bayesian statistical models.

A pure-data description of every fitted model — id, kind, outcome, family and
editorial status — with **no heavy imports** (no PyMC, no factories), so it can be
imported cheaply by the report (to code-generate the model register in
``docs/report/appendices/appendix-b-model-catalogue.qmd``) and by any tooling that
needs the catalogue without paying to import the model-building code.

This mirrors the vocabulary-growth report's ``vocab_growth.models.definitions``. It
deliberately duplicates the lightweight metadata that otherwise lives on each
module's ``SPEC`` (the ``ModelSpec`` in ``context.py``); the test
``tests/test_model_definitions.py`` guards the two against drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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

#: Valid model kinds (the eight statistical-model families).
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
    return ModelDefinition(*args, **kwargs)


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
]

# --- Mechanism / moderation / mediation (adjusted associations) -------------------
_MECH = [
    _d("lrp56", "mechanism", "Mechanism", Status.ASSOCIATION, "W", "receptive vocabulary -> word reading"),
    _d("lrp57", "mechanism", "Mechanism", Status.ASSOCIATION, "W", "expressive vocabulary -> word reading"),
    _d("lrp58", "mechanism", "Mechanism", Status.ASSOCIATION, "W", "letter sounds -> word reading"),
    _d("lrp71", "mechanism", "Moderation", Status.ASSOCIATION, "W", "expressive-vocabulary moderation"),
    _d("lrp72", "mechanism", "Moderation", Status.ASSOCIATION, "N", "phonics route (letter-sound x blending -> decoding)"),
    _d("lrp72base", "mechanism", "Moderation", Status.COMPANION, "N", "no-interaction baseline", base="lrp72"),
    _d("lrp73", "mechanism", "Moderation", Status.ASSOCIATION, "W", "age-moderated letter-sound -> word reading"),
    _d("lrp73base", "mechanism", "Moderation", Status.COMPANION, "W", "no-interaction baseline", base="lrp73"),
    _d("lrp59", "mediation", "Mediation", Status.ASSOCIATION, "W", "g-formula via letter sounds"),
    _d("lrp62", "mediation", "Mediation", Status.ASSOCIATION, "W", "reading-route composite mediation"),
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


#: The register: every fitted model, keyed by id. Must match the fit script's MODELS.
MODEL_REGISTRY: dict[str, ModelDefinition] = {
    d.model_id: d
    for d in (*_ITT, *_JOINT, *_SES, *_ABIL, *_DID, *_MECH, *_GAIN, *_GAINB, *_LEVEL, *_ALIGNED)
}


# --- Provenance: deleted predecessors and reserved / deferred models --------------
# Not built (no module); recorded for the supersession map and roadmap only.
SUPERSEDED: dict[str, str] = {
    "LRP52": "lrpitt10",   # word reading
    "LRP53": "lrpitt05",   # receptive vocabulary
    "LRP54": "lrpitt06",   # expressive vocabulary
    "LRP74": "lrpitt02",   # taught expressive
    "LRP75": "lrpitt01",   # taught receptive
    "LRP55": "lrpitt12",   # joint
    "LRP60": "lrpitt13",   # SES (word reading)
    "LRP60a": "lrpitt14",  # SES matched comparator
    "LRP76": "lrpitt15",   # generalisation contrast
}

RESERVED: dict[str, str] = {
    "LRP16": "descriptive developmental-trajectory model (the trajectory question is the companion vocabulary-growth report)",
    "LRP70": "CELF-moderated mechanism, deferred pending a DAG review of conditioning on a descendant of letter-sound knowledge",
}


def models_by_status(status: Status) -> list[ModelDefinition]:
    """All registered models with a given status, in registry order."""
    return [d for d in MODEL_REGISTRY.values() if d.status is status]
