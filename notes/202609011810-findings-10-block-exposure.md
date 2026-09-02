> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5.1).

# Findings: the `block_exposure` family — the second teaching block

**Read `findings-00-overview` first.** This note covers the 5 models in the `block_exposure` family. All 5 pass the convergence gate with zero divergences and all are publishable (2026-09-01 rebuild). This is one of the families whose fitted rows changed under the #631 ERB quarantine; the numbers below are from the post-quarantine fits.

## The data

**RLI trial only**, the per-wave levels panel: 54 children contributing 158–159 rows across timepoints. The intervention taught vocabulary in two blocks of items, and the two arms reached block-2 teaching at different times: immediate-arm children were block-2 active by wave 3, waiting-list children not until wave 4. The outcomes are block-2 specific: taught receptive and expressive vocabulary from block 2 and the matched not-taught items from the same block.

## What the model is for

A staggered-exposure design. Wave intercepts absorb whatever changes over time for everyone, and the model estimates the difference associated with block-2 teaching being **active** rather than block-1 teaching being active — both states are on-intervention, so the comparison is between teaching blocks, not between taught and untaught. Identification rests on an untestable parallel-trajectories assumption, and by wave 3 the arms differ in how much other intervention they have had.

## What was found

| Outcome                                   | Block-2 active | 89% range    | P(favoured direction) | Evidence     |
| ----------------------------------------- | -------------- | ------------ | --------------------- | ------------ |
| Taught expressive vocabulary, block 2     | +0.7 items     | −0.4 to +1.9 | 0.84 (positive)       | suggestive   |
| Not-taught receptive vocabulary, block 2  | +0.2 items     | −0.5 to +0.8 | 0.65 (positive)       | inconclusive |
| Not-taught expressive vocabulary, block 2 | −0.3 items     | −0.9 to +0.4 | 0.75 (negative)       | inconclusive |
| Taught receptive vocabulary, block 2      | −0.7 items     | −1.9 to +0.4 | 0.85 (negative)       | suggestive   |

**None is well determined and the directions are not coherent.** Every interval contains zero; the two taught outcomes point in opposite directions (expressive +0.7, receptive −0.7), and nothing reaches moderate evidence. The wide-prior sensitivity for not-taught expressive vocabulary (`bx-103`) gives −0.3 (89% −1.0 to +0.4, P(negative) = 0.77), the same answer.

The honest statement is that **this design does not show a coherent or well-determined block-2 effect**, not that block-2 teaching did nothing. The block-2 window is short, the item sets are narrow, and by the time block 2 is active the clean randomised contrast has gone. The block-1 taught-vocabulary results in the `itt` family (+1.4 and +1.5 items, well supported) come from the randomised window with the full contrast available; these data cannot distinguish a real difference between blocks from the different windows and identifying assumptions.

## What changed since the August notes

The quarantined ERB cell moved these estimates only in the second decimal: the taught estimates are unchanged to the first decimal, the not-taught receptive direction probability moved from 0.64 to 0.65, and the not-taught expressive probability sits at the ladder's 0.75 boundary, so its label has flipped from suggestive to inconclusive without the number having meaningfully changed.

## What these models cannot tell you

**Nothing here is causal.** **Wide intervals are not evidence of no effect** — they are wide enough to contain effects of the size found for block 1. **Block-2 outcomes are not comparable to block-1 outcomes** without care.

## Model inventory

All 5 pass the convergence gate with zero divergences and are publishable: `bx-001` (taught expressive), `002` (taught receptive), `003` (not-taught expressive), `004` (not-taught receptive), `103` (wide-prior sensitivity for `003`).
