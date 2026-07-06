# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Canonical model-ID scheme + legacy <-> canonical resolver (issue #168, Phase 1).

The non-destructive compatibility layer of the model-ID migration
(``notes/202607021145-issue-168-model-id-migration-plan.md``). It defines the
canonical ``<project>-<study>-<family>-<nnn>`` scheme and pure, reversible
conversions to and from the current (legacy) ids — **without renaming anything**.
Both fit CLIs accept either form; the metadata accessors on ``ModelSpec`` /
``ModelConfig`` expose the canonical id; Phase 2 (the mechanical rename) can then
land incrementally behind this layer.

Three representations of the same id (numbers zero-padded to 3 digits in canonical
form; the legacy ids use 2 digits, which the transform accounts for):

===============  =================  =================
Use              Form               Example
===============  =================  =================
Display / docs   UPPER, hyphen      ``LRP-RLI-ITT-010``
CLI / paths      lower, hyphen      ``lrp-rli-itt-010``
Python module    lower, underscore  ``lrp_rli_itt_010``
===============  =================  =================

Deliberately dependency-free (stdlib only): the family of a *bare* ``lrp##`` model
(e.g. ``lrp65``) is not visible in its id, so the caller supplies its ``kind``
(``ModelSpec`` has it; the GB models embed ``gbg``/``gbl`` so need none). This keeps
the module importable without the PyMC / model-building stack.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PROJECT_CODE = "lrp"
"""The only project today; part of every canonical id."""

#: ``ModelSpec.kind`` -> canonical family code. Joint-outcome ITT models stay in the
#: ITT family (plan D-4; the joint structure is carried as ``variant_role``); both
#: mediation kinds share ``med``. ``growth`` (LRP69/70) and ``historical_growth``
#: (the RLM cohort's ``rlmhg``) were added after the plan note.
FAMILY_BY_KIND: dict[str, str] = {
    "itt": "itt",
    "joint": "itt",
    "gain_factors": "gf",
    "level_factors": "lf",
    "aligned": "al",
    "did": "did",
    "horseshoe": "hs",
    "corr_factor": "mm",
    "mechanism": "mch",
    "mediation": "med",
    "mediation_multi": "med",
    "adjusted": "adj",
    "lcsm": "lcs",
    "dose_response": "dr",
    "growth": "gc",
    "historical_growth": "hg",
}

#: Family codes that are embedded in the legacy id itself (so no ``kind`` is needed
#: to parse them), in the order the legacy id spells them.
_EMBEDDED_FAMILIES: tuple[str, ...] = (
    "itt", "gf", "lf", "al", "did", "hs", "mm", "gbg", "gbl", "hg",
)

#: Legacy prefix -> (project, study). ``rlm`` is a *study* under the ``lrp`` project.
_STUDY_BY_PREFIX: dict[str, str] = {"lrp": "rli", "rlm": "rlm"}
_PREFIX_BY_STUDY: dict[str, str] = {"rli": "lrp", "rlm": "rlm"}

#: Controlled variant-suffix vocabulary -> the ``variant_role`` it denotes. A
#: word suffix (``base``) is hyphenated in canonical form (``...-007-base``); a
#: single-letter suffix attaches (``...-001b``).
VARIANT_ROLE_BY_SUFFIX: dict[str, str] = {
    "b": "companion",
    "base": "comparator",
    "d": "dose_sensitivity",
    "a": "alternate",
}
_WORD_SUFFIXES: frozenset[str] = frozenset({"base"})

_LEGACY_RE = re.compile(
    r"^(?P<prefix>lrp|rlm)"
    r"(?P<fam>itt|gf|lf|al|did|hs|mm|gbg|gbl|hg)?"
    r"(?P<num>\d+)"
    r"(?P<suffix>base|[abd])?$"
)
_CANONICAL_RE = re.compile(
    r"^(?P<project>lrp)-(?P<study>rli|rlm)-(?P<family>[a-z]+)-"
    r"(?P<num>\d+)(?:-(?P<word_suffix>base)|(?P<letter_suffix>[abd]))?$"
)


class ModelIdError(ValueError):
    """Raised when a model id cannot be parsed or a bare id lacks its ``kind``."""


@dataclass(frozen=True)
class ModelId:
    """A parsed canonical model id and its three rendered forms."""

    project: str
    study: str
    family: str
    number: int
    suffix: str | None = None

    @property
    def _tail(self) -> str:
        n = f"{self.number:03d}"
        if self.suffix in _WORD_SUFFIXES:
            return f"{n}-{self.suffix}"
        if self.suffix:
            return f"{n}{self.suffix}"
        return n

    @property
    def cli(self) -> str:
        """Lower-hyphen form used on the CLI and in output paths."""
        return f"{self.project}-{self.study}-{self.family}-{self._tail}"

    @property
    def display(self) -> str:
        """UPPER-hyphen form for docs / report titles."""
        return self.cli.upper()

    @property
    def module(self) -> str:
        """Lower-underscore form for the eventual Python module name (Phase 2)."""
        return self.cli.replace("-", "_")

    @property
    def variant_role(self) -> str | None:
        """The variant role denoted by the suffix, if any."""
        return VARIANT_ROLE_BY_SUFFIX.get(self.suffix) if self.suffix else None

    @property
    def legacy(self) -> str:
        """The legacy id this canonical id maps back to (2-digit number)."""
        n = f"{self.number:02d}"
        suffix = self.suffix or ""
        if self.family in _EMBEDDED_FAMILIES:
            prefix = _PREFIX_BY_STUDY[self.study]
            return f"{prefix}{self.family}{n}{suffix}"
        # Bare families (mch/med/adj/lcs/dr/gc): the family is not spelled in the
        # legacy id, only the number is (e.g. lrp65, lrp77base).
        return f"{_PREFIX_BY_STUDY[self.study]}{n}{suffix}"


def parse_legacy(legacy_id: str, *, kind: str | None = None, study: str | None = None) -> ModelId:
    """Parse a legacy id (``lrpitt10``, ``lrp65``, ``lrpgbg12``, ``rlmhg01`` ...).

    ``kind`` is required only for *bare* ``lrp##`` models whose family is not spelled
    in the id (``lrp65`` -> needs ``kind="adjusted"``); ids that embed their family
    (``lrpitt``/``lrpgf``/``lrpgbg`` ...) ignore it. ``study`` overrides the study
    otherwise inferred from the prefix (``lrp`` -> ``rli``, ``rlm`` -> ``rlm``).
    """
    m = _LEGACY_RE.match(legacy_id)
    if m is None:
        raise ModelIdError(f"Unrecognised legacy model id: {legacy_id!r}")
    embedded = m.group("fam")
    study_code = study or _STUDY_BY_PREFIX[m.group("prefix")]
    if embedded is not None:
        family = embedded
    elif kind is not None:
        if kind not in FAMILY_BY_KIND:
            raise ModelIdError(f"No family code for kind {kind!r} (id {legacy_id!r})")
        family = FAMILY_BY_KIND[kind]
    else:
        raise ModelIdError(
            f"Bare id {legacy_id!r} needs its kind to resolve a family code."
        )
    return ModelId(
        project=PROJECT_CODE,
        study=study_code,
        family=family,
        number=int(m.group("num")),
        suffix=m.group("suffix"),
    )


def to_canonical(
    legacy_id: str,
    *,
    kind: str | None = None,
    study: str | None = None,
    form: str = "cli",
) -> str:
    """Canonical id for a legacy id, in ``form`` = ``"cli"`` / ``"display"`` / ``"module"``."""
    mid = parse_legacy(legacy_id, kind=kind, study=study)
    try:
        return getattr(mid, form)
    except AttributeError as exc:  # pragma: no cover - guard against a typo'd form
        raise ModelIdError(f"Unknown id form {form!r}") from exc


def parse_canonical(canonical_id: str) -> ModelId:
    """Parse any canonical form (display / cli / module) into a :class:`ModelId`."""
    normalised = canonical_id.strip().lower().replace("_", "-")
    m = _CANONICAL_RE.match(normalised)
    if m is None:
        raise ModelIdError(f"Unrecognised canonical model id: {canonical_id!r}")
    return ModelId(
        project=m.group("project"),
        study=m.group("study"),
        family=m.group("family"),
        number=int(m.group("num")),
        suffix=m.group("word_suffix") or m.group("letter_suffix"),
    )


def to_legacy(canonical_id: str) -> str:
    """Legacy id for any canonical form (the reverse of :func:`to_canonical`)."""
    return parse_canonical(canonical_id).legacy


def looks_canonical(model_id: str) -> bool:
    """True if ``model_id`` is written in a canonical form (has the study segment)."""
    return _CANONICAL_RE.match(model_id.strip().lower().replace("_", "-")) is not None


def resolve_to_legacy(model_id: str) -> str:
    """Normalise a user-supplied id to its legacy form for registry lookup.

    A canonical id (any form) is mapped back to its legacy id; anything else
    (already-legacy, or unrecognised) is returned unchanged so the caller's own
    "unknown model" handling still fires.
    """
    if looks_canonical(model_id):
        try:
            return to_legacy(model_id)
        except ModelIdError:
            return model_id
    return model_id
