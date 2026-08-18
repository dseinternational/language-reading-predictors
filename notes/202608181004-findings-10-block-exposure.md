> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).
>
> Substantially corrected by a LLM-based AI tool (Codex/GPT-5).

# Findings: the `block_exposure` family — the second teaching block

**Read `findings-00-overview` first.** This note covers the 5 models in the `block_exposure` family.

## The data

**RLI trial only**, using the per-wave levels panel: 54 children contributing about 158–159 rows across timepoints. Data are stacked by wave, not collapsed.

The intervention taught vocabulary in two blocks of items. These models concern **block 2**, and they exploit the fact that the two arms reached block-2 teaching at different times: immediate-arm children were block-2 active by wave 3, while waiting-list children were still on block 1 until both arms were block-2 active at wave 4.

The outcomes are block-2 specific: taught receptive and expressive vocabulary from block 2, and the matched not-taught items from the same block.

## What the model is for

This is a staggered-exposure design. Because the two arms switch to block-2 teaching at different waves, there is a window where one group is receiving block-2 teaching and the other is not — and every child eventually gets it. Wave intercepts absorb whatever is changing over time for everyone, and the model estimates the difference associated with block-2 teaching being **active** rather than not.

The estimand is a pooled "block-active" coefficient: the adjusted change in the outcome when block 2 is being taught, translated to items.

**Identification rests on an assumption worth stating plainly:** that in the absence of block-2 teaching, the two arms' trajectories on these measures would have run parallel. That is not testable here, and by wave 3 the arms differ in how much _other_ intervention they have already had.

## What was found

| Outcome                                   | Block-2 active | 89% range    | P(favoured direction) | Evidence     |
| ----------------------------------------- | -------------- | ------------ | --------------------- | ------------ |
| Taught expressive vocabulary, block 2     | +0.7 items     | −0.4 to +1.9 | 0.84 (positive)       | suggestive   |
| Not-taught receptive vocabulary, block 2  | +0.2 items     | −0.5 to +0.8 | 0.64 (positive)       | inconclusive |
| Not-taught expressive vocabulary, block 2 | −0.3 items     | −0.9 to +0.4 | 0.75 (negative)       | suggestive   |
| Taught receptive vocabulary, block 2      | −0.7 items     | −1.9 to +0.4 | 0.85 (negative)       | suggestive   |

**None is well determined, but they are not all labelled inconclusive.** All four 89% intervals contain zero; under the project's probability-based vocabulary, three nevertheless reach _suggestive_ evidence in their favoured direction and only not-taught receptive vocabulary is inconclusive. The directions are not coherent across outcomes — the two taught outcomes point oppositely, expressive +0.7 and receptive −0.7 — and no result reaches moderate evidence.

A prior-sensitivity variant for not-taught expressive vocabulary (`bx-103`) gives −0.3 [−1.0, +0.4], with P(negative) = 0.78. Its near-agreement with the main fit supports numerical stability across those priors, but it does not establish a negligible effect.

## How to read a set of null-looking results

The honest statement is that **this design does not show a coherent or well-determined block-2 effect across the block-2 vocabulary outcomes**, not that block-2 teaching did nothing. Several things limit it: the block-2 exposure window is short, the measures are narrow item sets, and by the time block 2 is active the clean randomised contrast is gone — both arms have had intervention, just different amounts.

It is worth contrasting this with the block-1 taught-vocabulary results in the `itt` family (+1.4 and +1.5 items, well supported). Those come from the randomised window with the full contrast available. These data cannot distinguish a real difference between teaching blocks from the different exposure windows and identifying assumptions of the two designs.

## What these models cannot tell you

**Nothing here is causal.** The parallel-trajectories assumption is untestable, and the comparison sits after the randomised window.

**Wide intervals are not evidence of no effect** — see the overview. The suggestive direction labels are not firm findings, and the intervals are wide enough to contain effects of the size found for block 1.

**These outcomes are not comparable to block-1 outcomes** without care: different items, different exposure windows.

## Model inventory

All 5 pass the convergence gate with zero divergences and are publishable: `bx-001` (taught expressive), `002` (taught receptive), `003` (not-taught expressive), `004` (not-taught receptive), `103` (wide-prior sensitivity).
