# Findings — exploratory families: block_exposure & survival

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8). Numbers from the `reporting`-config refit of 2026-07-16 (see [process note](202607161130-full-statistical-refit.md)). Preliminary. These are two small exploratory families grouped in one note.

## block_exposure (`bx-001…004`)

**What it asks:** the vocabulary teaching ran in two blocks (block 1 then block 2). These models ask whether a block-2 measure is higher _when block-2 teaching is active_ than when block-1 teaching was active — a **parallel-trends association** comparing the two exposure windows. Block exposure was **not randomised**, so these are associations, not effects.

**Gate:** all 4 **passed**.

| Model  | Measure                                | When block-2 teaching active, difference | Evidence              |
| ------ | -------------------------------------- | ---------------------------------------- | --------------------- |
| bx-001 | Taught expressive vocab, block 2 (TE2) | +0.7 items (−0.7 to +2.2)                | suggestive (positive) |
| bx-002 | Taught receptive vocab, block 2 (TR2)  | −0.7 items (−2.1 to +0.7)                | suggestive (negative) |
| bx-003 | Not-taught expressive, block 2 (UE2)   | −0.3 items (−1.1 to +0.5)                | inconclusive          |
| bx-004 | Not-taught receptive, block 2 (UR2)    | +0.2 items (−0.7 to +1.0)                | inconclusive          |

**Story:** the block-active exposure signals are small and mostly inconclusive; there is at most weak, suggestive movement on the directly-taught block-2 expressive measure. This is a fine-grained exploratory check that does not overturn anything in the main analyses.

## survival (`surv-009`, `surv-011`)

**What it asks:** for the heavily-floored measures (phonetic spelling P, nonword reading N), _when_ does a child first "come off the floor"? A discrete-time survival model estimates the hazard of moving off zero in each interval, with a treated-vs-not hazard term. Because **both arms are treated by the final wave**, this term is a **prognostic association over all waves, not a randomised effect**.

**Gate:** both **passed** cleanly.

| Model    | Measure               | Hazard ratio (coming off floor) | Read                                         |
| -------- | --------------------- | ------------------------------- | -------------------------------------------- |
| surv-009 | Phonetic spelling (P) | 0.84 (0.40 to 1.79)             | 67% below 1 — inconclusive                   |
| surv-011 | Nonword reading (N)   | 1.35 (0.66 to 2.79)             | 80% above 1 — suggestive (earlier off-floor) |

For an untreated child at average covariates, the fitted baseline chance of being off the floor in an interval ran ≈16–22% (P) and ≈22–29% (N).

**Story:** the timing signal for coming off the floor is weak and uncertain in both measures — leaning toward _earlier_ movement for nonword reading, but well short of a confident call. Consistent with the [ITT](202607161800-findings-itt.md) floor-rule outcomes, which were also inconclusive-to-suggestive.

## What is causal

**Nothing in either family.** `block_exposure` is a non-randomised parallel-trends association; the `survival` treated term is a whole-trial prognostic association (both arms end up treated). Both are exploratory triangulation around the floored outcomes.
