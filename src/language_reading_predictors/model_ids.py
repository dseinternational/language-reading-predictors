# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Canonical model-ID scheme + legacy <-> canonical resolver (issue #168).

Defines the canonical ``<project>-<study>-<family>-<nnn>`` scheme and pure,
reversible conversions to and from the current (legacy) ids. Phase 1 shipped this
as a non-destructive compatibility layer; Phase 2 renames the module files, docs
folders, and output paths to the canonical form, keying the registry on the
canonical id and resolving legacy ids *forward* to it (the reverse of Phase 1) so
old ids keep working. Both fit CLIs accept either form.

Three representations of the same id (numbers zero-padded to 3 digits in canonical
form; the legacy ids use 2 digits, which the transform accounts for):

===============  =================  =================
Use              Form               Example
===============  =================  =================
Display / docs   UPPER, hyphen      ``LRP-RLI-ITT-010``
CLI / paths      lower, hyphen      ``lrp-rli-itt-010``
Python module    lower, underscore  ``lrp_rli_itt_010``
===============  =================  =================

**Variants (parent+100, issue #168 Frank sign-off).** A variant model
(``lrpgf01b``, ``lrp77base`` ...) renumbers to *parent + 100* rather than keeping a
suffix: ``lrpgf01b -> lrp-rli-gf-101``. This yields clean, suffix-free canonical
ids, but the renumber is **lossy** (the number alone cannot say it was variant
``b``) and, for the one parent with two variants (``lrp77``), naive parent+100 would
collide at ``177``. Both are handled by an explicit :data:`_VARIANT_RENUMBER`
table, so the legacy <-> canonical round-trip stays exact.

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
    "mechanism": "mech",
    "mediation": "med",
    "mediation_multi": "med",
    "adjusted": "adj",
    "lcsm": "lcsm",
    "dose_response": "dose",
    "growth": "gc",
    "historical_growth": "hg",
    "survival": "surv",
}

#: Family codes that are embedded in the legacy id itself (so no ``kind`` is needed
#: to parse them), in the order the legacy id spells them.
_EMBEDDED_FAMILIES: tuple[str, ...] = (
    "itt", "gf", "lf", "al", "did", "hs", "mm", "gbg", "gbl", "hg", "surv",
)

#: Legacy prefix -> (project, study). ``rlm`` is a *study* under the ``lrp`` project.
_STUDY_BY_PREFIX: dict[str, str] = {"lrp": "rli", "rlm": "rlm"}
_PREFIX_BY_STUDY: dict[str, str] = {"rli": "lrp", "rlm": "rlm"}

#: Controlled variant-suffix vocabulary -> the ``variant_role`` it denotes. The
#: suffix is not spelled in the canonical id (it is folded into the parent+100
#: number); it is recovered from the legacy id for ``variant_role`` metadata.
VARIANT_ROLE_BY_SUFFIX: dict[str, str] = {
    "b": "companion",
    "base": "comparator",
    "d": "dose_sensitivity",
    "a": "alternate",
}
_LETTER_SUFFIXES: frozenset[str] = frozenset(
    s for s in VARIANT_ROLE_BY_SUFFIX if len(s) == 1
)
_WORD_SUFFIXES: frozenset[str] = frozenset(
    s for s in VARIANT_ROLE_BY_SUFFIX if len(s) > 1
)

#: Explicit legacy-variant -> (family code, canonical parent+100 number). Explicit
#: because parent+100 is lossy (the suffix is unrecoverable from the number) and the
#: two ``lrp77`` variants would otherwise collide at 177; ``lrp77a=177`` /
#: ``lrp77base=277`` follows alphabetical suffix order (issue #168 sign-off).
_VARIANT_RENUMBER: dict[str, tuple[str, int]] = {
    "lrpgf01b": ("gf", 101),
    "lrpgf02b": ("gf", 102),
    "lrpgf03b": ("gf", 103),
    "lrpgf04b": ("gf", 104),
    "lrpgf05b": ("gf", 105),
    "lrpgf06b": ("gf", 106),
    "lrpgf07b": ("gf", 107),
    "lrpgf08b": ("gf", 108),
    "lrpitt13b": ("itt", 113),
    "lrpitt14b": ("itt", 114),
    "lrpitt15b": ("itt", 115),
    "lrpal01d": ("al", 101),
    "lrpdid07base": ("did", 107),
    "lrp72base": ("mech", 172),
    "lrp73base": ("mech", 173),
    "lrp77a": ("dose", 177),
    "lrp77base": ("dose", 277),
}
#: Inverse of :data:`_VARIANT_RENUMBER`: (family, canonical number) -> legacy id.
_VARIANT_LEGACY_ID: dict[tuple[str, int], str] = {
    (fam, num): legacy for legacy, (fam, num) in _VARIANT_RENUMBER.items()
}

#: Every valid canonical family code: the kind-mapped families plus the GB
#: gain/level codes. ``_CANONICAL_RE`` is restricted to these so an unknown family
#: (a typo like ``lrp-rli-zzz-010``) is not treated as canonical — and therefore
#: not silently remapped to a legacy id, which would risk running the wrong model.
_ALL_FAMILY_CODES: frozenset[str] = frozenset(FAMILY_BY_KIND.values()) | {"gbg", "gbl"}


def _alt(options) -> str:
    """Regex alternation, longest option first so no shorter code shadows a longer one."""
    return "|".join(sorted(options, key=lambda s: (-len(s), s)))


# The legacy regex derives its family + suffix sub-patterns from the declared maps
# above, so the maps and the parser cannot drift. The canonical regex has no suffix
# group: under parent+100 every canonical id is ``project-study-family-nnn``.
_LETTER_SUFFIX_CLASS = "".join(sorted(_LETTER_SUFFIXES))

_LEGACY_RE = re.compile(
    r"^(?P<prefix>lrp|rlm)"
    rf"(?P<fam>{_alt(_EMBEDDED_FAMILIES)})?"
    r"(?P<num>\d+)"
    rf"(?P<suffix>{_alt(_WORD_SUFFIXES)}|[{_LETTER_SUFFIX_CLASS}])?$"
)
_CANONICAL_RE = re.compile(
    r"^(?P<project>lrp)-(?P<study>rli|rlm)-"
    rf"(?P<family>{_alt(_ALL_FAMILY_CODES)})-(?P<num>\d+)$"
)


class ModelIdError(ValueError):
    """Raised when a model id cannot be parsed or a bare id lacks its ``kind``."""


@dataclass(frozen=True)
class ModelId:
    """A parsed canonical model id and its three rendered forms.

    ``number`` is the *canonical* number (parent+100 for variants); ``suffix`` holds
    the original legacy suffix as metadata only (it drives ``variant_role`` and the
    exact legacy round-trip, and is **not** rendered into the canonical forms).
    """

    project: str
    study: str
    family: str
    number: int
    suffix: str | None = None

    @property
    def _tail(self) -> str:
        # parent+100 folds any variant suffix into the number, so the canonical
        # tail is always just the zero-padded number.
        return f"{self.number:03d}"

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
        """Lower-underscore form for the Python module name."""
        return self.cli.replace("-", "_")

    @property
    def variant_role(self) -> str | None:
        """The variant role denoted by the (legacy) suffix, if any."""
        return VARIANT_ROLE_BY_SUFFIX.get(self.suffix) if self.suffix else None

    @property
    def legacy(self) -> str:
        """The legacy id this canonical id maps back to (2-digit number)."""
        if self.suffix is not None:
            # Variant: recover the exact legacy id from the renumber table (the
            # number alone is lossy, so a table lookup is the reverse of truth).
            return _VARIANT_LEGACY_ID[(self.family, self.number)]
        n = f"{self.number:02d}"
        if self.family in _EMBEDDED_FAMILIES:
            return f"{_PREFIX_BY_STUDY[self.study]}{self.family}{n}"
        # Bare families (mech/med/adj/lcsm/dose/gc): the family is not spelled in the
        # legacy id, only the number is (e.g. lrp65).
        return f"{_PREFIX_BY_STUDY[self.study]}{n}"


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
    suffix = m.group("suffix")
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
    if suffix is not None:
        # Variant: take the canonical parent+100 number from the explicit table.
        entry = _VARIANT_RENUMBER.get(legacy_id.lower())
        if entry is None:
            raise ModelIdError(
                f"Variant id {legacy_id!r} is not in the parent+100 renumber table."
            )
        family, number = entry
    else:
        number = int(m.group("num"))
    return ModelId(
        project=PROJECT_CODE,
        study=study_code,
        family=family,
        number=number,
        suffix=suffix,
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
    family = m.group("family")
    number = int(m.group("num"))
    # A number in a variant's parent+100 slot recovers its original suffix (for
    # variant_role + the exact legacy round-trip); everything else is a base model.
    legacy = _VARIANT_LEGACY_ID.get((family, number))
    suffix = _LEGACY_RE.match(legacy).group("suffix") if legacy is not None else None
    return ModelId(
        project=m.group("project"),
        study=m.group("study"),
        family=family,
        number=number,
        suffix=suffix,
    )


def to_legacy(canonical_id: str) -> str:
    """Legacy id for any canonical form (the reverse of :func:`to_canonical`)."""
    return parse_canonical(canonical_id).legacy


def looks_canonical(model_id: str) -> bool:
    """True if ``model_id`` is written in a canonical form (has the study segment)."""
    return _CANONICAL_RE.match(model_id.strip().lower().replace("_", "-")) is not None


def resolve_to_canonical(
    model_id: str, *, kind: str | None = None, study: str | None = None
) -> str:
    """Normalise any id (legacy or canonical, any form) to its canonical CLI id.

    The Phase-2 forward resolver: the registry is keyed on the canonical id, so a
    legacy id supplied on the CLI is mapped *forward* to it. A bare legacy id
    (``lrp65``) needs its ``kind``; callers without it should resolve via the
    registry's own legacy-alias index instead. Anything unrecognised is returned
    unchanged so the caller's own "unknown model" handling still fires.
    """
    stripped = model_id.strip()
    if looks_canonical(stripped):
        return parse_canonical(stripped).cli
    try:
        return parse_legacy(stripped, kind=kind, study=study).cli
    except ModelIdError:
        return model_id


def resolve_to_legacy(model_id: str) -> str:
    """Normalise a user-supplied id to its legacy form.

    A canonical id (any form) is mapped back to its legacy id; anything else
    (already-legacy, or unrecognised) is returned unchanged. Retained for
    back-compatibility and for callers that still key on legacy ids.
    """
    if looks_canonical(model_id):
        try:
            return to_legacy(model_id)
        except ModelIdError:
            return model_id
    return model_id
