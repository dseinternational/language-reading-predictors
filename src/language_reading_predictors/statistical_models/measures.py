# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Canonical definitions of the bounded-count measures used in LRP52-LRP58.

Each measure has a short symbol (W, R, E, ...) used throughout the modelling
code, a column name in ``rli_data_long.csv``, and a test maximum ``n_trials``
used as the binomial denominator.

All ``n_trials`` values are preliminary and flagged for review. Where the
nominal test ceiling is known (e.g. YARC-LSK = 32 items), that value is used.
Where it is not (e.g. SPPHON, a phoneme-level count with variable
item-stopping), a conservative upper bound above the observed maximum is used
and called out explicitly. Update this file after confirming values with the
data dictionary or study team.
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
    # Word reading composite (YARC EWR + SWR). Observed max 64; conservative
    # ceiling 90 pending confirmation.
    "W": Measure("W", V.EWRSWR, 90, "Word reading (EWRSWR)", n_trials_confirmed=False),
    # ROWPVT-4: 190 items. Observed max 82.
    "R": Measure("R", V.ROWPVT, 190, "Receptive vocabulary (ROWPVT)", n_trials_confirmed=True),
    # EOWPVT-4: 170 items. Observed max 77.
    "E": Measure("E", V.EOWPVT, 170, "Expressive vocabulary (EOWPVT)", n_trials_confirmed=True),
    # YARC letter-sound knowledge: 32 items. Observed max 32.
    "L": Measure("L", V.YARCLET, 32, "Letter-sound knowledge (YARC-LSK)", n_trials_confirmed=True),
    # Phonetic spelling (phoneme-level count across 10 words with
    # variable-stopping). Observed max 92; ceiling 100 pending confirmation.
    "P": Measure("P", V.SPPHON, 100, "Phonetic spelling (SPPHON)", n_trials_confirmed=False),
    # Blending: 10 items. Observed max 10.
    "B": Measure("B", V.BLENDING, 10, "Phoneme blending", n_trials_confirmed=True),
    # CELF Preschool-2 basic concepts: 18 items. Observed max 18.
    "F": Measure("F", V.CELF, 18, "Basic concept knowledge (CELF)", n_trials_confirmed=True),
    # TROG-2: 8 blocks of 4 = 32 items. Observed max 27.
    "T": Measure("T", V.TROG, 32, "Receptive grammar (TROG-2)", n_trials_confirmed=True),
    # Nonword reading: 6 items. Post-only; not used as a baseline.
    "N": Measure("N", V.NONWORD, 6, "Nonword reading", n_trials_confirmed=True),
}


ITT_OUTCOMES: tuple[str, ...] = ("W", "R", "E", "L", "P", "B", "F", "T")
"""Eight outcomes used in LRP52-LRP55 (all bounded counts with pre- and post- values)."""


def unconfirmed_ceilings() -> list[str]:
    """Return the symbols of measures whose ``n_trials`` is not documented."""
    return [m.symbol for m in MEASURES.values() if not m.n_trials_confirmed]
