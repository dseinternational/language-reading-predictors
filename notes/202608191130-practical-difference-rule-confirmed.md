> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Practical-difference thresholds: the half-natural-maturation rule confirmed as the standing rule

**Date:** 2026-08-19. **Decision:** the project's practical-difference thresholds (δ, the minimally-important difference behind every `P(benefit ≥ δ)` statement) are confirmed as **half the natural-maturation gain** — the rule adopted on 2026-06-26 (`notes/202606261304-evidence-strength-and-rope-reporting.md`) and signed off for the original suite on 2026-07-01 (issue #144) — and it now stands as the rule for **any outcome, prospectively**, not only as the derivation used for the outcomes already fitted.

## The rule, stated once

For a bounded-count outcome, δ is **half the waiting-list arm's mean gain over the untreated first period (timepoint 1 to timepoint 2)**, on the instrument's natural scale, **rounded to whole items and floored at one item** — you cannot resolve less than one item, and "the intervention adds at least half of what a child would gain naturally in a period" is a reading a practitioner can use. Two conventions travel with it:

- an outcome whose encoding rescales the instrument (expressive information is doubled to a count of half-marks so that its half-mark scoring fits a whole-number likelihood) takes its δ on the **natural marks** and expresses the same real magnitude on the rescaled scale — 1 mark of 40 = 2 half-marks of 80 — so the two encodings never judge the same effect against different bars;
- an outcome with **no untreated baseline period** (the block-2 taught and not-taught vocabulary tests, introduced after the crossover) inherits the δ of its block-1 counterpart, the same instrument on the same word-list design.

The floored outcomes (phonetic spelling, nonword reading) are unaffected: their estimand is the probability of coming off the floor and their δ is the separately signed-off risk difference of 0.10 (2026-07-01).

## The rule reproduces every adopted value

Waiting-list arm, mean timepoint-1 → timepoint-2 gain (n = 26, word reading 25, expressive grammar 25), from `data/rli_data_long.csv`:

| Outcome                               | Wait-arm gain | Half | Rule δ    | Adopted δ                |
| ------------------------------------- | ------------: | ---: | --------- | ------------------------ |
| Letter sounds (L, /32)                |          3.23 | 1.62 | 2         | 2                        |
| Word reading (W, /79)                 |          2.04 | 1.02 | 1         | 1                        |
| Receptive vocabulary (R, /170)        |          3.04 | 1.52 | 2         | 2                        |
| Expressive vocabulary (E, /170)       |          4.31 | 2.16 | 2         | 2                        |
| Taught receptive (TR, /24)            |          2.12 | 1.06 | 1         | 1                        |
| Taught expressive (TE, /24)           |          1.77 | 0.89 | 1 (floor) | 1                        |
| Not-taught receptive (UR, /12)        |          0.31 | 0.16 | 1 (floor) | 1                        |
| Not-taught expressive (UE, /12)       |          0.54 | 0.27 | 1 (floor) | 1                        |
| Blending (B, /10)                     |          0.04 | 0.02 | 1 (floor) | 1                        |
| Basic concepts (F, /18)               |          0.00 | 0.00 | 1 (floor) | 1                        |
| Receptive grammar (T, /32)            |          1.00 | 0.50 | 1 (floor) | 1                        |
| Expressive grammar (EG, /37)          |          1.24 | 0.62 | 1         | 1                        |
| Expressive information (EI40, /40)    |          2.98 | 1.49 | 1         | 1 (= 2 half-marks of 80) |
| Block-2 vocabulary (TE2/TR2, UE2/UR2) |             — |    — | inherited | 1 each                   |

Expressive information is the only borderline: 1.49 rounds to 1, but 2 marks was equally rule-consistent and was escalated rather than settled by the analysis (`notes/202608182015-apt-delta-threshold-ratification.md`). Every other value follows from the rule without judgement.

## What changes

- The **F and T** thresholds, rule-derived on 2026-07-20 and annotated in `measures.py` as "pending education-lead ratification like the others (#144)", are ratified by this confirmation of the rule; that annotation is updated.
- For **future analysis rounds** the rule is **pre-specified**: a new outcome's δ is derived from the waiting-list arm's untreated period-1 gain by the rule above and recorded in `measures.ROPE_DELTA` before the outcome is fitted. The open suggestion in `notes/202608182200-findings-by-question.md` (question 8, "pre-specify the practical-difference thresholds … the half-natural-maturation rule … could simply be adopted prospectively") is closed by this decision.
- **Nothing about the August 2026 run changes.** Its thresholds were derived after results existed, and the notes continue to say so; confirming the rule settles what future thresholds will be, not when this run's were chosen. The mandatory δ-sensitivity tables (each outcome at δ and 2·δ; floored outcomes at 10 / 15 / 20 percentage points) remain the reader's guard against any single threshold.

## One property worth remembering

Tying δ to each test's own pace of natural growth deliberately sets a **lower bar on tests where children progress slowly** — the 2026-06-26 note records this as a feature of the choice, not an accident. It is why letter sounds (δ = 2) and word reading (δ = 1) read as comparably meaningful; under an absolute, spread-based rule word reading would read as clearly weaker. The rule encodes "meaningful relative to the domain's own pace", and that is now the project's standing position.
