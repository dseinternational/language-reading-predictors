# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Canonical definitions of the bounded-count measures used in the statistical
models (the eight standardised ITT outcomes plus the taught-vocabulary block
family targeted by LRP74-LRP76).

Each measure has a short symbol (W, R, E, ...) used throughout the modelling
code, a column name in ``rli_data_long.csv``, and a test maximum ``n_trials``
used as the binomial denominator.

``n_trials`` is the Beta-Binomial denominator, so it scales every
probability-scale effect; each value is confirmed against a documented test
ceiling. The standardised measures use their test-manual maxima (e.g.
YARC-LSK = 32 items, ROWPVT = 170); the two study-specific composites use
Burgoyne et al. (2012), Table 3 — word reading (EWR + SWR) = 79 and phonetic
spelling = 92. ``load_and_prepare`` guards at runtime that no observed count
exceeds its ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass

from language_reading_predictors.data_variables import Variables as V


@dataclass(frozen=True)
class Measure:
    symbol: str
    """Short symbol used in model equations (W, R, E, L, P, B, F, T, N)."""
    column: str
    """Column name in ``rli_data_long.csv``."""
    n_trials: int
    """Binomial denominator (test maximum)."""
    label: str
    """Human-readable label for reports."""
    n_trials_confirmed: bool
    """Whether ``n_trials`` matches a documented test ceiling."""


MEASURES: dict[str, Measure] = {
    # Word reading composite (YARC Early Word Recognition + Single Word
    # Reading). Test maximum 79 (Burgoyne et al. 2012, Table 3, "Single-word
    # reading (79)"); observed max 64.
    "W": Measure("W", V.EWRSWR, 79, "Word reading (EWRSWR)", n_trials_confirmed=True),
    # ROWPVT (Brownell 2000): 170 items. Observed max 82.
    "R": Measure("R", V.ROWPVT, 170, "Receptive vocabulary (ROWPVT)", n_trials_confirmed=True),
    # EOWPVT-4: 170 items. Observed max 77.
    "E": Measure("E", V.EOWPVT, 170, "Expressive vocabulary (EOWPVT)", n_trials_confirmed=True),
    # YARC letter-sound knowledge: 32 items. Observed max 32.
    "L": Measure("L", V.YARCLET, 32, "Letter-sound knowledge (YARC-LSK)", n_trials_confirmed=True),
    # Phonetic spelling (phoneme-level count across 10 words with variable
    # item-stopping). Test maximum 92 (Burgoyne et al. 2012, Table 3,
    # "Phonetic spelling (92)"); observed max 92.
    "P": Measure("P", V.SPPHON, 92, "Phonetic spelling (SPPHON)", n_trials_confirmed=True),
    # Blending: 10 items. Observed max 10.
    "B": Measure("B", V.BLENDING, 10, "Phoneme blending", n_trials_confirmed=True),
    # CELF Preschool-2 basic concepts: 18 items. Observed max 18.
    "F": Measure("F", V.CELF, 18, "Basic concept knowledge (CELF)", n_trials_confirmed=True),
    # TROG-2: 8 blocks of 4 = 32 items. Observed max 27.
    "T": Measure("T", V.TROG, 32, "Receptive grammar (TROG-2)", n_trials_confirmed=True),
    # Nonword reading: 6 items. Post-only; not used as a baseline.
    "N": Measure("N", V.NONWORD, 6, "Nonword reading", n_trials_confirmed=True),
    # --- Taught-vocabulary block tests (intervention-fidelity outcomes) -------
    # Bespoke tests of the words explicitly taught in the intervention (Block 1,
    # weeks 1-20), tested both ways and split into the directly-taught target
    # words and a not-taught comparison set. Block 1 is taught in phase 1, so its
    # baseline is t1 and its randomised post-score is t2 - the ITT window. (Block
    # 2 is introduced in phase 2 and has no t1 baseline, so it is not modelled
    # here.) See ``docs/models/lrp74`` and Burgoyne et al. (2012), Table 3.
    #
    # Taught tests: "Six words of each type (nouns, adverbs, adjectives,
    # prepositions)" = 24 items; the paper tabulates the maximum as (24).
    "TE": Measure(
        "TE", V.B1EXTAU, 24, "Taught expressive vocabulary, block 1 (b1extau)",
        n_trials_confirmed=True,
    ),
    "TR": Measure(
        "TR", V.B1RETAU, 24, "Taught receptive vocabulary, block 1 (b1retau)",
        n_trials_confirmed=True,
    ),
    # Not-taught comparison sets. Burgoyne et al. (2012), Table 3, tabulates only
    # the 24-item taught tests; the not-taught set's item count is not documented.
    # Observed maximum is 12 for both modalities (consistent with a half-size
    # 3-words-x-4-types control set), so 12 is used as the denominator and flagged
    # unconfirmed - probability-scale summaries for these outcomes are therefore
    # approximate pending the data dictionary.
    "UE": Measure(
        "UE", V.B1EXNT, 12, "Not-taught expressive vocabulary, block 1 (b1exnt)",
        n_trials_confirmed=False,
    ),
    "UR": Measure(
        "UR", V.B1RENT, 12, "Not-taught receptive vocabulary, block 1 (b1rent)",
        n_trials_confirmed=False,
    ),
}


ITT_OUTCOMES: tuple[str, ...] = ("W", "R", "E", "L", "P", "B", "F", "T")
"""Eight outcomes used in LRP52-LRP55 (all bounded counts with pre- and post- values).

Deliberately excludes the taught-vocabulary block measures (``TE``/``TR``/
``UE``/``UR``): they are a separate intervention-fidelity family targeted by
LRP74-LRP76 and are passed explicitly via ``ModelSpec.extra["outcomes"]`` so the
eight-outcome joint model (LRP55) stays stable.
"""


TAUGHT_BLOCK1_OUTCOMES: tuple[str, ...] = ("TE", "TR", "UE", "UR")
"""Block-1 taught-vocabulary family: taught/not-taught x expressive/receptive."""


def unconfirmed_ceilings() -> list[str]:
    """Return the symbols of measures whose ``n_trials`` is not documented."""
    return [m.symbol for m in MEASURES.values() if not m.n_trials_confirmed]
