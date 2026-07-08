# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for the canonical model-ID resolver (issue #168).

The whole-inventory checks are **import-free**: the model set is derived from the
module filenames — all in canonical underscore form since Phase 2, for both the stat
models (``lrp_rli_itt_001``) and the GB models (``lrp_rli_gbg_012``) — so they run
without the PyMC / GB stack. The ``kind`` for a bare id comes from the lightweight
``definitions.MODEL_REGISTRY``. Mirrors ``tests/test_model_definitions.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from language_reading_predictors import model_ids as M
from language_reading_predictors.statistical_models import definitions as D

_STAT_DIR = Path(D.__file__).resolve().parent
_GB_DIR = _STAT_DIR.parent / "models"

# (legacy id, kind, expected canonical CLI form) — one per family + each variant
# kind. Variants renumber to parent+100 (issue #168): lrpgf01b -> gf-101, and the
# two lrp77 variants split a=177 / base=277.
_CASES: list[tuple[str, str | None, str]] = [
    ("lrpitt10", "itt", "lrp-rli-itt-010"),
    ("lrpitt01", "itt", "lrp-rli-itt-001"),
    ("lrpitt15b", "joint", "lrp-rli-itt-115"),
    ("lrpgf01b", "gain_factors", "lrp-rli-gf-101"),
    ("lrplf08", "level_factors", "lrp-rli-lf-008"),
    ("lrpdid07base", "did", "lrp-rli-did-107"),
    ("lrpal01d", "aligned", "lrp-rli-al-101"),
    ("lrphs02", "horseshoe", "lrp-rli-hs-002"),
    ("lrpmm01", "corr_factor", "lrp-rli-mm-001"),
    ("lrp56", "mechanism", "lrp-rli-mech-056"),
    ("lrp64", "mediation_multi", "lrp-rli-med-064"),
    ("lrp65", "adjusted", "lrp-rli-adj-065"),
    ("lrp67", "lcsm", "lrp-rli-lcsm-067"),
    ("lrp77base", "dose_response", "lrp-rli-dose-277"),
    ("lrp77a", "dose_response", "lrp-rli-dose-177"),
    ("lrp69", "growth", "lrp-rli-gc-069"),
    ("lrpgbg12", None, "lrp-rli-gbg-012"),
    ("lrpgbl28", None, "lrp-rli-gbl-028"),
    ("rlmhg01", "historical_growth", "lrp-rlm-hg-001"),
]


def _kind_by_canonical() -> dict[str, str]:
    # MODEL_REGISTRY is canonical-keyed since #168 Phase 2.
    return {mid: d.kind for mid, d in D.MODEL_REGISTRY.items()}


def _model_canonical_ids() -> list[str]:
    """Every model's canonical CLI id, derived import-free from the module files.

    All modules are named in canonical underscore form since #168 Phase 2 — the stat
    models (``lrp_rli_itt_001`` -> ``lrp-rli-itt-001``) and, since the GB rename, the
    GB models too (``lrp_rli_gbg_012`` -> ``lrp-rli-gbg-012``) — so the canonical CLI
    id is just the stem with underscores swapped for hyphens.
    """
    stat = {p.stem.replace("_", "-") for p in _STAT_DIR.glob("lrp_rli_*.py")}
    stat |= {p.stem.replace("_", "-") for p in _STAT_DIR.glob("lrp_rlm_*.py")}
    gb = {p.stem.replace("_", "-") for p in _GB_DIR.glob("lrp_rli_gb*.py")}  # gbg/gbl
    return sorted(stat | gb)


@pytest.mark.parametrize("legacy, kind, canonical", _CASES)
def test_roundtrip_and_three_forms(legacy: str, kind: str | None, canonical: str) -> None:
    mid = M.parse_legacy(legacy, kind=kind)
    assert mid.cli == canonical
    assert mid.display == canonical.upper()
    assert mid.module == canonical.replace("-", "_")
    # legacy -> canonical (any form) -> legacy is an identity.
    assert M.to_legacy(mid.cli) == legacy
    assert M.to_legacy(mid.display) == legacy
    assert M.to_legacy(mid.module) == legacy


def test_resolve_to_legacy_accepts_every_form() -> None:
    for form in ("lrp-rli-itt-010", "LRP-RLI-ITT-010", "lrp_rli_itt_010"):
        assert M.resolve_to_legacy(form) == "lrpitt10"
    # A legacy id, the ``all`` sentinel and an unknown string pass through unchanged
    # so the CLIs' own resolution / error handling still fires.
    assert M.resolve_to_legacy("lrpitt10") == "lrpitt10"
    assert M.resolve_to_legacy("all") == "all"
    assert M.resolve_to_legacy("bogus") == "bogus"


def test_resolve_to_canonical_maps_forward() -> None:
    # Phase-2 forward direction: any id form -> canonical CLI id (the registry key).
    for form in ("lrpitt10", "lrp-rli-itt-010", "LRP-RLI-ITT-010", "lrp_rli_itt_010"):
        assert M.resolve_to_canonical(form) == "lrp-rli-itt-010"
    # An embedded-family variant needs no kind; parent+100 applies.
    assert M.resolve_to_canonical("lrpgf01b") == "lrp-rli-gf-101"
    assert M.resolve_to_canonical("lrp77a", kind="dose_response") == "lrp-rli-dose-177"
    # A bare legacy id needs its kind; without it, it passes through unchanged so the
    # caller's registry-alias index (which knows each model's kind) can resolve it.
    assert M.resolve_to_canonical("lrp65", kind="adjusted") == "lrp-rli-adj-065"
    assert M.resolve_to_canonical("lrp65") == "lrp65"
    # The ``all`` sentinel and unknown strings pass through unchanged.
    assert M.resolve_to_canonical("all") == "all"
    assert M.resolve_to_canonical("bogus") == "bogus"


def test_variant_role_from_suffix() -> None:
    assert M.parse_legacy("lrpgf01b", kind="gain_factors").variant_role == "companion"
    assert M.parse_legacy("lrp77base", kind="dose_response").variant_role == "comparator"
    assert M.parse_legacy("lrpal01d", kind="aligned").variant_role == "dose_sensitivity"
    assert M.parse_legacy("lrp77a", kind="dose_response").variant_role == "alternate"
    assert M.parse_legacy("lrpitt10", kind="itt").variant_role is None


def test_bare_id_requires_kind() -> None:
    with pytest.raises(M.ModelIdError):
        M.parse_legacy("lrp65")  # bare id, family not in the id
    assert M.to_canonical("lrp65", kind="adjusted") == "lrp-rli-adj-065"


def test_unrecognised_ids_raise() -> None:
    with pytest.raises(M.ModelIdError):
        M.parse_legacy("not-a-model")
    with pytest.raises(M.ModelIdError):
        M.parse_canonical("lrpitt10")  # legacy form is not canonical


def test_every_kind_has_a_family_code() -> None:
    # A new statistical-model kind must be given a family code or this fails.
    for kind in D.KINDS:
        assert kind in M.FAMILY_BY_KIND, f"kind {kind!r} has no family code"


def test_every_model_resolves_and_roundtrips() -> None:
    kind_by_canon = _kind_by_canonical()
    ids = _model_canonical_ids()
    assert len(ids) > 130  # sanity: the full inventory, not an empty glob
    for canonical in ids:
        # canonical -> legacy needs no kind (the family is embedded); legacy ->
        # canonical needs a kind only for bare ids, taken from the registry.
        legacy = M.to_legacy(canonical)
        kind = kind_by_canon.get(canonical)
        back = M.to_canonical(legacy, kind=kind) if kind else M.to_canonical(legacy)
        assert back == canonical, (canonical, legacy, back)


def test_no_duplicate_canonical_ids() -> None:
    ids = _model_canonical_ids()
    assert len(ids) == len(set(ids)), "canonical ids collide across the inventory"


def test_unknown_canonical_family_is_not_treated_as_canonical() -> None:
    # A typo'd / unknown family code (canonical-looking but not a real family) must
    # NOT be silently remapped to a legacy id — that would risk a CLI running the
    # wrong model. It fails to parse and passes through unchanged, so the CLI's own
    # "unknown model" handling fires.
    assert not M.looks_canonical("lrp-rli-zzz-010")
    assert M.resolve_to_legacy("lrp-rli-zzz-010") == "lrp-rli-zzz-010"
    with pytest.raises(M.ModelIdError):
        M.parse_canonical("lrp-rli-zzz-010")


def test_resolution_is_case_insensitive_after_lowering() -> None:
    # The CLIs lower-case the resolved id before the (lower-case) registry lookup,
    # so an upper-case legacy id, an upper-case canonical id and the ``all`` sentinel
    # all resolve. Mirrors how both fit_model.py and fit_statistical_model.py resolve.
    assert M.resolve_to_legacy("LRP-RLI-ITT-010").lower() == "lrpitt10"
    assert M.resolve_to_legacy("LRPITT10").lower() == "lrpitt10"
    assert M.resolve_to_legacy("ALL").lower() == "all"
