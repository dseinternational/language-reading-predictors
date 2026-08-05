<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Findings 16 — the block-exposure family

Reports every model in the `block_exposure` family from the 2026-08-04/05 `reporting` refit. **4 models, all passing the convergence gate.** Reading conventions in note 00. Preliminary research data — all estimates provisional.

## What these models do

The intervention taught vocabulary in **two blocks** of words. The ITT suite's taught/not-taught models cover block 1. This small family asks the block-2 version of the same question, but with a sharper exposure definition: does a child's score on the block-2 word sets move specifically during the period when **block 2 was actually being taught**?

That "block-active" framing is what makes the family worth having. If the intervention's vocabulary effect is genuinely teaching-specific, the taught block-2 words should move when block 2 is active and not otherwise, and the matched not-taught words should not move at either time.

**Design.** The block-2 outcome regressed on its own baseline with a block-active exposure indicator, reported as `delta` on the logit scale and translated to items.

**Status.** Exploratory. The exposure is not randomised at the block level — which block was active is determined by the programme schedule, so it is confounded with time, age and cumulative exposure. Every estimate is an **association**.

## Results

| Model    | Outcome                              | Block-active effect (89%)        | P(>0) | Favoured direction   |
| -------- | ------------------------------------ | -------------------------------- | ----: | -------------------- |
| `bx-001` | Taught expressive, block 2 (TE2)     | **+0.73** items (−0.44 to +1.94) | 0.837 | positive, suggestive |
| `bx-004` | Not-taught receptive, block 2 (UR2)  | +0.15 items (−0.51 to +0.82)     | 0.640 | inconclusive         |
| `bx-003` | Not-taught expressive, block 2 (UE2) | −0.28 items (−0.91 to +0.38)     | 0.247 | negative, suggestive |
| `bx-002` | Taught receptive, block 2 (TR2)      | **−0.74** items (−1.88 to +0.42) | 0.151 | negative, suggestive |

**Nothing here is resolved, and the two taught outcomes point in opposite directions.** Taught expressive block-2 words lean positive (+0.73, suggestive); taught receptive block-2 words lean _negative_ (−0.74, suggestive) by almost exactly the same amount. The two not-taught sets sit near zero.

**The honest reading is that this family is uninformative about block-2 teaching specificity.** A teaching-specific account predicts both taught sets positive and both not-taught sets flat. What the data show is one taught set up, one taught set down, neither resolved, on intervals wide enough to contain both. With four models, ~50 children and a non-randomised exposure, that is about what should be expected.

It is worth stating plainly because the opposite mistake is easy: picking `bx-001` out of the four and reporting "block-2 taught expressive vocabulary improved during block-2 teaching (suggestive)" would be selecting the one result that fits the story from a set that, taken together, does not support it.

## Relation to the block-1 findings

The block-1 taught/not-taught contrasts (notes 01 and 02) are better identified — they use the randomised window and estimate the taught-versus-not-taught difference within one model. Even there the generalisation contrast was only suggestive for expressive and inconclusive for receptive. This family adds no further support and should not be quoted as if it did.

## Caveats

- **Not randomised at the block level.** Block-active timing is confounded with period, age and cumulative exposure.
- **Exploratory status**; small n, wide intervals, no resolved result.
- **Do not cherry-pick `bx-001`.** The family's four results are mutually inconsistent and should be reported together or not at all.
- **Predictive calibration.** 50% bands cover about 81% of observations.
