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
    # Nonword reading: 6 items. It has t1 data, but ~72% of children are at the
    # floor at baseline (a near-degenerate pre), so it is not co-loaded as an
    # autoregressive baseline — doing so would drop rows to complete cases for
    # little signal.
    "N": Measure("N", V.NONWORD, 6, "Nonword reading", n_trials_confirmed=True),
    # --- Taught-vocabulary block tests (intervention-fidelity outcomes) -------
    # Bespoke tests of the words explicitly taught in the intervention (Block 1,
    # weeks 1-20), tested both ways and split into the directly-taught target
    # words and a not-taught comparison set. Block 1 is taught in phase 1, so its
    # baseline is t1 and its randomised post-score is t2 - the ITT window. (Block
    # 2 is introduced in phase 2 and has no t1 baseline, so it carries no
    # randomised contrast; it is modelled instead by the block-exposure family
    # ``bx`` on its staggered teaching - see ``TAUGHT_BLOCK2_OUTCOMES`` below and
    # ``factories.build_block_exposure_model``.) See ``docs/models/lrp-rli-itt-002``
    # and Burgoyne et al. (2012), Table 3.
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
    # Not-taught comparison sets. Confirmed against the RLI assessment word list
    # (#214): each block's word set is 9 words x 4 types (nouns, verbs,
    # adjectives, prepositions) = 36, split 6 taught + 3 not-taught per type, so
    # 24 taught and 12 not-taught in each modality. (Burgoyne et al. 2012, Table 3
    # tabulated only the 24-item taught tests, leaving the not-taught count
    # undocumented until the word list resolved it.) 12 is the denominator for
    # both the expressive (b1exnt) and receptive (b1rent) not-taught tests.
    "UE": Measure(
        "UE", V.B1EXNT, 12, "Not-taught expressive vocabulary, block 1 (b1exnt)",
        n_trials_confirmed=True,
    ),
    "UR": Measure(
        "UR", V.B1RENT, 12, "Not-taught receptive vocabulary, block 1 (b1rent)",
        n_trials_confirmed=True,
    ),
    # --- Block-2 taught-vocabulary tests (block-exposure family `bx`) ----------
    # The second taught word set (Block 2, weeks 21-40), same instrument as block
    # 1. It has no t1 baseline (introduced in phase 2) and no randomised contrast,
    # so the `bx` family estimates a staggered block-active exposure association
    # (immediate arm taught block 2 in phase 2 while the waitlist is still on
    # block 1; waitlist reaches block 2 in phase 3). Denominators MIRROR block 1
    # under the #214 per-block word-list logic (each block: 9 words x 4 types = 36,
    # split 6 taught + 3 not-taught per type -> 24 taught / 12 not-taught per
    # modality), and the observed maxima are consistent (b2extau 21, b2retau 24,
    # b2exnt 10, b2rent 12 excluding one corrupt cell). Confirmed against the
    # block-2 word list (2026-07-14): 24 taught / 12 not-taught per modality, exactly
    # as block 1, so ``n_trials_confirmed=True``. (The one corrupt b2rent cell > 12 is
    # a separate source-data fix, handled by the UR2 loader drop.)
    "TE2": Measure(
        "TE2", V.B2EXTAU, 24, "Taught expressive vocabulary, block 2 (b2extau)",
        n_trials_confirmed=True,
    ),
    "TR2": Measure(
        "TR2", V.B2RETAU, 24, "Taught receptive vocabulary, block 2 (b2retau)",
        n_trials_confirmed=True,
    ),
    "UE2": Measure(
        "UE2", V.B2EXNT, 12, "Not-taught expressive vocabulary, block 2 (b2exnt)",
        n_trials_confirmed=True,
    ),
    "UR2": Measure(
        "UR2", V.B2RENT, 12, "Not-taught receptive vocabulary, block 2 (b2rent)",
        n_trials_confirmed=True,
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


TAUGHT_BLOCK2_OUTCOMES: tuple[str, ...] = ("TE2", "TR2", "UE2", "UR2")
"""Block-2 taught-vocabulary family (block-exposure `bx`): taught/not-taught x
expressive/receptive. Modelled as a staggered block-active exposure association
(no randomised contrast; block 2 has no t1 baseline) — see
:func:`factories.build_block_exposure_model`."""


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
DISTAL_OUTCOMES: frozenset[str] = frozenset({"R", "E", "T", "F", "UR", "UE", "UR2", "UE2"})
"""Broad standardised-transfer outcomes taking the tighter (distal) tau prior:
receptive/expressive vocabulary (R, E), grammar (T), basic concepts (F), and the
not-taught vocabulary comparison sets (UR, UE and their block-2 counterparts
UR2, UE2). Everything else — the directly taught vocabulary (TR, TE, TR2, TE2),
decoding (L, W, B) and the floored P/N — is proximal and keeps the wider default.
Membership is a substantive judgement owned by the education/research team, not a
statistical one."""


def is_distal(symbol: str | None) -> bool:
    """Whether ``symbol`` takes the tighter distal treatment-effect prior."""
    return symbol in DISTAL_OUTCOMES


# --- Revised-DAG hearing-status adjacency (2026-07-10, #233/#244) -------------
# The revised graph (dag/dag-language-reading.dagitty) adds hearing status as a
# common cause: HS -> { TR RV TE EV SP RW PA LS }. HS is therefore an upstream
# confounder for any observational estimand whose exposure/outcome is one of these
# nodes, and enters those models' DAG-derived adjustment sets (wired as the
# ``hs`` / ``hs_missing`` covariates - see preprocessing.add_hearing_status; the
# per-family application is issues #245-247).
#
# Split across two namespaces to avoid silent membership bugs: the DAG-node set
# records the graph verbatim, while the measure-symbol set is what suite code
# (which works in symbols) should test outcome symbols against. SP (speech) and
# RW (phonological memory) are DAG-only nodes with no registered outcome measure,
# so they map to no symbol.
HS_CHILDREN_DAG_NODES: frozenset[str] = frozenset(
    {"TR", "RV", "TE", "EV", "SP", "RW", "PA", "LS"}
)
"""DAG children of hearing status (`HS`) under the 2026-07-10 revision, in DAG-node
names (RV/EV/PA/LS, *not* this module's measure symbols). For symbol-keyed
membership use :data:`HS_CHILD_SYMBOLS`."""

HS_CHILD_SYMBOLS: frozenset[str] = frozenset({"TR", "TE", "R", "E", "L", "B"})
"""The registered-outcome children of `HS` in this module's measure symbols
(RV->R, EV->E, PA->B blending, LS->L; TR/TE unchanged) — the set suite code should
test outcome symbols against when deciding whether `HS` enters an adjustment set.
SP/RW are DAG-only nodes (no outcome measure) and so are absent here."""


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
# education lead 2026-07-01 (issue #144): the rule and W = 1 stand. UR/UE scale
# length is now confirmed at 12 (#214), so their items δ is no longer provisional.
# Consumed by ``reporting.rope_summary`` to report ``P(benefit >= delta)``.
# F/T (basic concepts / receptive grammar) were initially deferred as outside the ITT
# suite; adopted 2026-07-20 at δ = 1 item each by the same ½-natural-maturation rule
# (their wait-list t1->t2 gains, ≈0 and ≈1 item, both floor to 1 item). Rule-derived
# and pending education-lead ratification like the others (#144).
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
    # Basic concepts (F, CELF) and receptive grammar (T, TROG): both floor to δ = 1
    # item under the ½-natural-maturation rule (adopted 2026-07-20, pending #144
    # ratification); previously deferred as outside the ITT suite.
    "F": 1.0,
    "T": 1.0,
    # Block-2 taught-vocabulary family (block-exposure `bx`), same items-scale δ
    # as their block-1 counterparts.
    "TE2": 1.0,
    "TR2": 1.0,
    "UE2": 1.0,
    "UR2": 1.0,
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

    Raises ``KeyError`` for outcomes that use a probability-scale delta instead
    (floored P/N, see :data:`ROPE_DELTA_PROB`).
    """
    if symbol not in ROPE_DELTA:
        raise KeyError(
            f"No items-scale ROPE delta set for {symbol!r}; floored outcomes (P/N) "
            "use ROPE_DELTA_PROB instead."
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
