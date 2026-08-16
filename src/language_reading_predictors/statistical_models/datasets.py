# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Dataset and per-study measure metadata (issue #165).

The intervention study (RLI) is described implicitly by :mod:`data_variables`
(``Variables``) and the global :data:`measures.MEASURES` catalogue. As soon as a
second dataset enters the package (the Byrne, MacDonald & Buckley
reading-language-memory study, ``study_id="rlm"``), those single-study
assumptions need to become explicit *metadata* rather than being hard-coded into
loaders and models.

This module is a **leaf** (it imports only :mod:`environment`) so it can be
imported from ``preprocessing`` / ``factories`` / ``pipeline`` without a cycle.
It defines:

- :class:`StudyMeasure` - a per-study bounded-count measure (kept **separate**
  from the RLI symbol namespace so a study-local symbol such as ``trog`` never
  collides with ``Variables.TROG`` / global ``MEASURES``);
- :class:`DatasetSpec` - where a study's long-format CSV lives and how its
  subject / wave / group columns are named;
- the Byrne catalogue (:data:`RLM_MEASURES`) and dataset (:data:`RLM_DATASET`).

Consolidating the RLI ``MEASURES`` into this same abstraction is deliberately
out of scope for now (see #165) - this layer sits *alongside* the existing
global catalogue rather than replacing it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from language_reading_predictors.statistical_models import environment as _env


@dataclass(frozen=True)
class StudyMeasure:
    """A bounded-count measure within a single study.

    ``symbol`` is a study-local key (e.g. ``"basread"``); ``column`` is the CSV
    column; ``n_trials`` is the Beta-Binomial denominator (the test ceiling).
    ``n_trials_confirmed`` records whether that ceiling is the instrument's true
    maximum (as opposed to an observed-max placeholder), mirroring the same flag
    on :class:`measures.Measure`. ``instrument_identity_confirmed`` is separate:
    a denominator can be documented for a named scale while the source column's
    actual scale identity remains unresolved. ``available_waves`` records the
    source-field window when it is narrower than a model family's default.
    """

    symbol: str
    column: str
    n_trials: int
    label: str
    n_trials_confirmed: bool = False
    instrument_identity_confirmed: bool = False
    instrument_identity_note: str = ""
    available_waves: tuple[int, ...] = ()


@dataclass(frozen=True)
class DatasetSpec:
    """Where a study's long-format data lives and how its key columns are named.

    A neutral description of a longitudinal dataset, deliberately free of any
    intervention semantics (no treatment / randomised-phase fields): those belong
    to the RLI ``PreparedData`` path, not to descriptive historical cohorts.
    Source-lineage confirmation is nevertheless publication-relevant metadata and
    is snapshotted into each fit's release contract.
    """

    study_id: str
    label: str
    path: Path
    subject_col: str = "subject_id"
    wave_col: str = "time"
    group_col: str = "readgrp"
    group_labels: dict[int, str] = field(default_factory=dict)
    design: str = "historical_cohort"
    source: str = ""
    source_provenance_confirmed: bool = False
    source_provenance_note: str = ""
    source_provenance_manifest: str = ""


# --- Byrne, MacDonald & Buckley reading-language-memory study (study_id="rlm") ---
#
# NOTE: the group labels are duplicated in ``scripts/replicate_reading_language_memory.py``
# (the standalone audit tool that predates this package layer); consolidating the
# script onto this catalogue is a follow-up (part of the #165 multi-study registry
# work).
RLM_GROUP_LABELS: dict[int, str] = {
    1: "Down syndrome",
    2: "Average readers",
    3: "Reading-matched",
}

# Instrument ceilings researched against published sources and signed off by the
# data owner on 2026-07-16 (#338; see the dated decisions note). The battery,
# editions and raw-score use are confirmed by the cohort's primary paper (Byrne,
# MacDonald & Buckley, 2002, DOI 10.1348/00070990260377497): BAS first edition
# (Elliott, 1983), TROG (Bishop, 1983), BPVS (Dunn, 1982), and WORD (Rust,
# Golombok & Trickey, 1993). Confirmed ceilings (``n_trials_confirmed=True``):
#
# - ``basread``  90 - BAS Word Reading has 90 words (Beech 2004, Reading
#   Psychology; the previous 87 was the observed extract maximum mislabelled as
#   confirmed).
# - ``trog``     20 - 80 items in 20 blocks of 4, scored as blocks passed
#   (Bishop's original TROG manual, OSF). The extract reaches this ceiling.
# - ``basdig``   34 - BAS Recall of Digits, 34 items (CLOSER cognitive-measures
#   guide; Parsons 2014). The extract reaches this ceiling.
# - ``bassim``   21 - BAS Similarities, 21 items (CLOSER guide; Parsons 2014).
# - ``basmat``   28 - BAS Matrices, 28 items (CLOSER guide; Parsons 2014). The
#   checksum-pinned cohort source names its later-wave fields BASMAT3-BASMAT5.
#   The Raven CPM comparison previously cited as an identity caveat came from a
#   separate 14-child memory-training cohort (Laws et al. 1995), not this study.
# - ``bpvs``     32 - BPVS Short Form, 32 items (Ripley & Yuill 2005): the
#   observed maximum of 29 across ages to 11+ is only consistent with the
#   short form, not the long form.
#
# Still **provisional** (observed extract maximum; ``n_trials_confirmed=False``),
# pending the instrument manuals (follow-up-plan decision 3):
#
# - ``basspel``  18 - Byrne et al. (2002) explicitly identify Spelling among
#   the administered 1983 BAS subtests and analyse raw scores. The paper does
#   not state the item count, so 18 remains an observed-maximum placeholder.
# - ``basnum``   60 - Byrne et al. (2002) label the values BAS number-skills
#   raw scores; its Table 3 values reproduce exactly from the prepared extract.
#   The paper does not state the administered form or maximum, so 60 remains an
#   observed-maximum placeholder rather than a confirmed ceiling.
# - ``woco``     31 - WORD Reading Comprehension item count unverified (the
#   parent WIAT subtest is commonly described as 38 items).
#
# ``basmat`` is wave-3+ only (no wave-1 baseline): registered for the #338
# Phase A window extension and fitted on its own later-wave window
# (``lrp-rlm-hg-009``).
RLM_MEASURES: dict[str, StudyMeasure] = {
    "basread": StudyMeasure(
        symbol="basread",
        column="basread",
        n_trials=90,
        label="BAS word reading",
        n_trials_confirmed=True,
        instrument_identity_confirmed=True,
        available_waves=(1, 2, 3, 4, 5),
    ),
    "basspel": StudyMeasure(
        symbol="basspel",
        column="basspel",
        n_trials=18,
        label="BAS spelling",
        n_trials_confirmed=False,
        instrument_identity_confirmed=True,
        available_waves=(1, 2, 3, 4, 5),
    ),
    "woco": StudyMeasure(
        symbol="woco",
        column="woco",
        n_trials=31,
        label="WORD reading comprehension",
        n_trials_confirmed=False,
        instrument_identity_confirmed=True,
        available_waves=(1, 2, 3, 4, 5),
    ),
    "bpvs": StudyMeasure(
        symbol="bpvs",
        column="bpvs",
        n_trials=32,
        label="BPVS receptive vocabulary",
        n_trials_confirmed=True,
        instrument_identity_confirmed=True,
        available_waves=(1, 2, 3, 4, 5),
    ),
    "trog": StudyMeasure(
        symbol="trog",
        column="trog",
        n_trials=20,
        label="TROG receptive grammar",
        n_trials_confirmed=True,
        instrument_identity_confirmed=True,
        available_waves=(1, 2, 3, 4, 5),
    ),
    "basdig": StudyMeasure(
        symbol="basdig",
        column="basdig",
        n_trials=34,
        label="BAS recall of digits",
        n_trials_confirmed=True,
        instrument_identity_confirmed=True,
        available_waves=(1, 2, 3, 4, 5),
    ),
    "bassim": StudyMeasure(
        symbol="bassim",
        column="bassim",
        n_trials=21,
        label="BAS similarities/verbal reasoning",
        n_trials_confirmed=True,
        instrument_identity_confirmed=True,
        available_waves=(1, 2, 3, 4, 5),
    ),
    "basnum": StudyMeasure(
        symbol="basnum",
        column="basnum",
        n_trials=60,
        label="BAS number skills",
        n_trials_confirmed=False,
        instrument_identity_confirmed=True,
        available_waves=(1, 2, 3, 4),
    ),
    "basmat": StudyMeasure(
        symbol="basmat",
        column="basmat",
        n_trials=28,
        label="BAS matrices/non-verbal reasoning",
        n_trials_confirmed=True,
        instrument_identity_confirmed=True,
        instrument_identity_note=(
            "Confirmed from the checksum-pinned cohort source's native BASMAT3-BASMAT5 "
            "field names and the source repository's BAS Matrices definition. The "
            "2002 paper does not report this later-wave measure."
        ),
        available_waves=(3, 4, 5),
    ),
}

RLM_DATASET = DatasetSpec(
    study_id="rlm",
    label="Byrne, MacDonald & Buckley reading-language-memory study",
    path=(
        Path(_env.DATA_DIR)
        / "reading-language-memory"
        / "reading_language_memory_data_long.csv"
    ),
    subject_col="subject_id",
    wave_col="time",
    group_col="readgrp",
    group_labels=RLM_GROUP_LABELS,
    design="historical_cohort",
    source="Byrne, MacDonald & Buckley (2002)",
    source_provenance_confirmed=True,
    source_provenance_note=(
        "Reconciled 2026-08-16: the prepared 97-participant extract matches all "
        "97 cases and 52 shared non-identifying fields in the checksum-pinned SPSS "
        "source. The historical 96-row CSV omitted one Down-syndrome participant; "
        "the repaired derivative now matches all 97 source assessment rows."
    ),
    source_provenance_manifest=(
        "data/reading-language-memory/source_provenance.json"
    ),
)


# Registry so pipelines can resolve ``ModelSpec.study_id`` -> (dataset, measures)
# without a hand-maintained import list per study.
_DATASETS: dict[str, tuple[DatasetSpec, dict[str, StudyMeasure]]] = {
    "rlm": (RLM_DATASET, RLM_MEASURES),
}


def resolve_dataset(study_id: str) -> tuple[DatasetSpec, dict[str, StudyMeasure]]:
    """Return the ``(DatasetSpec, measures)`` pair for a study id."""
    if study_id not in _DATASETS:
        raise KeyError(
            f"Unknown study_id {study_id!r}; known: {sorted(_DATASETS)}"
        )
    return _DATASETS[study_id]


def publication_input_contract(
    study_id: str, measure_symbols: tuple[str, ...]
) -> dict[str, object]:
    """Return the publication-relevant input snapshot for one fitted study.

    The release evaluator must read the evidence that belonged to the fit, not the
    current mutable catalogue.  ``write_run_metadata`` therefore persists this
    contract in ``config.json`` and stored fits can be re-decided without silently
    inheriting a later manual or provenance sign-off.
    """

    dataset, measures = resolve_dataset(study_id)
    selected = tuple(dict.fromkeys(measure_symbols))
    unknown = sorted(set(selected) - set(measures))
    if unknown:
        raise KeyError(
            f"Unknown {study_id!r} measure symbol(s) in publication contract: "
            f"{', '.join(unknown)}"
        )

    blockers: list[str] = []
    if not dataset.source_provenance_confirmed:
        detail = dataset.source_provenance_note or "source provenance is unverified"
        blockers.append(f"dataset source provenance is unresolved: {detail}")

    measure_records: dict[str, dict[str, object]] = {}
    for symbol in selected:
        measure = measures[symbol]
        record: dict[str, object] = {
            "label": measure.label,
            "n_trials": measure.n_trials,
            "n_trials_confirmed": measure.n_trials_confirmed,
            "instrument_identity_confirmed": measure.instrument_identity_confirmed,
            "available_waves": list(measure.available_waves),
        }
        if measure.instrument_identity_note:
            record["instrument_identity_note"] = measure.instrument_identity_note
        measure_records[symbol] = record
        if not measure.n_trials_confirmed:
            blockers.append(
                f"{symbol}: the bounded-count denominator is not confirmed "
                "against the instrument"
            )
        if not measure.instrument_identity_confirmed:
            detail = (
                measure.instrument_identity_note
                or "the source instrument identity is unverified"
            )
            blockers.append(f"{symbol}: instrument identity is unresolved: {detail}")

    if not selected:
        blockers.append("no fitted study measures were recorded")

    return {
        "schema_version": 1,
        "study_id": study_id,
        "publication_ready": not blockers,
        "dataset": {
            "source": dataset.source,
            "source_provenance_confirmed": dataset.source_provenance_confirmed,
            "source_provenance_note": dataset.source_provenance_note,
            "source_provenance_manifest": dataset.source_provenance_manifest,
        },
        "measures": measure_records,
        "blockers": blockers,
    }
