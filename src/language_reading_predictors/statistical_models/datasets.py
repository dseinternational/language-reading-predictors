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
    on :class:`measures.Measure`.
    """

    symbol: str
    column: str
    n_trials: int
    label: str
    n_trials_confirmed: bool = False


@dataclass(frozen=True)
class DatasetSpec:
    """Where a study's long-format data lives and how its key columns are named.

    A neutral description of a longitudinal dataset, deliberately free of any
    intervention semantics (no treatment / randomised-phase fields): those belong
    to the RLI ``PreparedData`` path, not to descriptive historical cohorts.
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

# ``basread`` uses its confirmed BAS word-reading ceiling (87, matching the
# observed maximum in the prepared extract). The other seven Byrne Table 2
# measures are registered for the per-measure historical-growth sweep (#164
# Phase A) with a **provisional** ceiling equal to the observed maximum in the
# extract and ``n_trials_confirmed=False``: the Beta-Binomial denominator is a
# placeholder until each instrument's true maximum is confirmed against its
# manual (the data-owner decision in
# ``notes/202607021052-issue-164-byrne-followup-plan.md``, decision 3). This
# mirrors how ``basread`` itself began (observed-max placeholder) and is why the
# per-measure reports flag their ceiling as unconfirmed. ``basmat`` stays
# unregistered here: it is wave-3+ only (no wave-1 baseline), so it needs its own
# wave range and is handled separately.
RLM_MEASURES: dict[str, StudyMeasure] = {
    "basread": StudyMeasure(
        symbol="basread",
        column="basread",
        n_trials=87,
        label="BAS word reading",
        n_trials_confirmed=True,
    ),
    # --- #164 Phase A: provisional observed-max ceilings (n_trials_confirmed=False) ---
    "basspel": StudyMeasure(
        symbol="basspel",
        column="basspel",
        n_trials=18,
        label="BAS spelling",
        n_trials_confirmed=False,
    ),
    "woco": StudyMeasure(
        symbol="woco",
        column="woco",
        n_trials=31,
        label="WORD reading comprehension",
        n_trials_confirmed=False,
    ),
    "bpvs": StudyMeasure(
        symbol="bpvs",
        column="bpvs",
        n_trials=29,
        label="BPVS receptive vocabulary",
        n_trials_confirmed=False,
    ),
    "trog": StudyMeasure(
        symbol="trog",
        column="trog",
        n_trials=20,
        label="TROG receptive grammar",
        n_trials_confirmed=False,
    ),
    "basdig": StudyMeasure(
        symbol="basdig",
        column="basdig",
        n_trials=34,
        label="BAS recall of digits",
        n_trials_confirmed=False,
    ),
    "bassim": StudyMeasure(
        symbol="bassim",
        column="bassim",
        n_trials=18,
        label="BAS similarities/verbal reasoning",
        n_trials_confirmed=False,
    ),
    "basnum": StudyMeasure(
        symbol="basnum",
        column="basnum",
        n_trials=60,
        label="BAS number skills",
        n_trials_confirmed=False,
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
)


# Registry so pipelines can resolve ``spec.extra["study_id"]`` -> (dataset, measures)
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
