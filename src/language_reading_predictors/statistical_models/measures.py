# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Canonical definitions of the bounded-count measures used in the statistical
models (the eight standardised ITT outcomes plus the taught-vocabulary block
family modelled by the LRPITT taught/not-taught models, LRPITT01-04/15/15b).

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
    # here.) See ``docs/models/lrpitt02`` and Burgoyne et al. (2012), Table 3.
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
"""The eight standardised ITT outcomes (all bounded counts with pre- and post- values).

Used as the cross-baseline default in :func:`factories.build_itt_model` and the
default outcome set of :func:`factories.build_joint_model`. Deliberately excludes
the taught-vocabulary block measures (``TE``/``TR``/``UE``/``UR``) and nonword
(``N``); the LRPITT suite passes its own outcome set explicitly via
``ModelSpec.extra["outcomes"]`` (see :data:`LRPITT_OUTCOMES`).
"""


TAUGHT_BLOCK1_OUTCOMES: tuple[str, ...] = ("TE", "TR", "UE", "UR")
"""Block-1 taught-vocabulary family: taught/not-taught x expressive/receptive."""


# --- Treatment-effect prior tier (issue #141) --------------------------------
# The randomised treatment-effect prior tau is tiered by how *proximal* the
# outcome is to what the intervention directly teaches. A single logit prior is
# not scale-invariant: Normal(0, 0.5) implies a plausible ~4-item swing on letter
# sounds but a ~35-item 95% swing on the 170-item ROWPVT/EOWPVT vocabulary tests
# (notes/202607011600-issue-141-prior-audit.md). The distal, broad-transfer
# outcomes are exactly where large effects are least expected (Burgoyne 2012's
# proximal-only pattern; Donolato et al. 2023's weak receptive-vocabulary
# transfer), so they take a tighter tau prior; the proximal directly-taught /
# decoding outcomes keep the wider default.
DISTAL_OUTCOMES: frozenset[str] = frozenset({"R", "E", "T", "F", "UR", "UE"})
"""Broad standardised-transfer outcomes taking the tighter (distal) tau prior:
receptive/expressive vocabulary (R, E), grammar (T), basic concepts (F), and the
not-taught vocabulary comparison sets (UR, UE). Everything else — the directly
taught vocabulary (TR, TE), decoding (L, W, B) and the floored P/N — is proximal
and keeps the wider default. Membership is a substantive judgement owned by the
education/research team, not a statistical one."""


def is_distal(symbol: str | None) -> bool:
    """Whether ``symbol`` takes the tighter distal treatment-effect prior."""
    return symbol in DISTAL_OUTCOMES


LRPITT_OUTCOMES: tuple[str, ...] = (
    "TR", "TE", "UR", "UE", "R", "E", "L", "B", "P", "W", "N",
)
"""The eleven RCT-phase outcomes of the uniform DAG-faithful ITT suite (#119),
in LRPITT01-LRPITT11 order: taught/not-taught receptive & expressive vocabulary,
standardised receptive & expressive vocabulary, letter sounds, blending,
phonetic spelling, word reading, and nonword reading.

This is a *reference ordering* for the suite (the forest plot, the joint model's
outcome set, docs); each single-outcome model still loads only its own symbol
(plus any cross/moderator symbol) via ``ModelSpec.extra["outcomes"]`` so the
shared complete-case mask never drops rows for measures the model ignores. In
particular ``N`` (nonword) is post-only and floored, so it must not be co-loaded
with the other outcomes (see ``floor`` and ``preprocessing.load_and_prepare``).
"""


# --- ROPE half-widths (minimally-important difference, delta) -----------------
# Per-outcome delta on the *items* scale, adopted 2026-06-26
# (notes/202606261304-evidence-strength-and-rope-reporting.md): delta = half a
# period's natural maturation gain, floored at 1 item and rounded. Confirmed by the
# education lead 2026-07-01 (issue #144): the rule and W = 1 stand; UR/UE remain
# denominator-provisional (scale length unconfirmed — see ``unconfirmed_ceilings``).
# Consumed by ``reporting.rope_summary`` to report
# ``P(benefit >= delta)``. F/T (concepts/grammar) are outside the ITT suite and not
# yet agreed, so they are deliberately absent (a look-up raises).
ROPE_DELTA: dict[str, float] = {
    "L": 2.0,
    "W": 1.0,
    "R": 2.0,
    "E": 2.0,
    "TR": 1.0,
    "TE": 1.0,
    "UR": 1.0,
    "UE": 1.0,
    "B": 1.0,
}

# Floored outcomes: the ITT estimand is the probability of coming off the floor, so
# delta is a *risk difference* in P(off-floor), not an items count. Confirmed at 0.10
# (10 pp) by the education lead 2026-07-01 (issue #144): given this group's evident
# difficulty with these measures, 10 pp stands as the primary threshold.
ROPE_DELTA_PROB: dict[str, float] = {
    "P": 0.10,
    "N": 0.10,
}

# Risk-difference δ grid for the floored-outcome sensitivity table (issue #144): the
# adopted 10 pp plus 15 and 20 pp stricter checks, shown for every floored outcome.
ROPE_DELTA_PROB_GRID: tuple[float, ...] = (0.10, 0.15, 0.20)


def rope_delta(symbol: str) -> float:
    """Items-scale ROPE half-width (minimally-important difference) for an outcome.

    Raises ``KeyError`` for outcomes whose items delta is not agreed (F/T) or which
    use a probability-scale delta instead (P/N, see :data:`ROPE_DELTA_PROB`).
    """
    if symbol not in ROPE_DELTA:
        raise KeyError(
            f"No items-scale ROPE delta set for {symbol!r}; floored outcomes use "
            "ROPE_DELTA_PROB and F/T are not yet agreed."
        )
    return ROPE_DELTA[symbol]


def rope_delta_grid(symbol: str) -> list[float]:
    """δ-sensitivity grid (items scale): the adopted δ and a stricter 2·δ (issue #144).

    The adopted δ is *half* a period's natural-maturation gain, so 2·δ is ≈ a full
    period's gain: the pair brackets the meaningful-benefit claim between "half" and
    "a full" period of typical growth. Word reading (δ = 1) therefore reports at
    δ = 1 and δ = 2, exactly as the education lead requested (2026-07-01). Raises for
    outcomes with no items-scale δ (floored P/N use ``ROPE_DELTA_PROB_GRID``).
    """
    d = rope_delta(symbol)
    return [d, 2.0 * d]


def unconfirmed_ceilings() -> list[str]:
    """Return the symbols of measures whose ``n_trials`` is not documented."""
    return [m.symbol for m in MEASURES.values() if not m.n_trials_confirmed]
