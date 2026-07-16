# Findings — growth family (multivariate growth curves; does baseline ability shape trajectories?)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8). Numbers from the `reporting`-config refit of 2026-07-16 (see [process note](202607161130-full-statistical-refit.md)). Preliminary.

## What these models ask

Fit each child's **trajectory** (level + growth rate) across waves for several outcomes jointly, and ask: **does a child's baseline non-verbal ability predict how fast their verbal/reading skills grow?** The trajectory coefficients are **between-child adjusted associations** — they describe which children grow faster, not what would happen if ability were changed.

## Convergence gate

All 3 **passed** (ESS ≈ 2300–3200; R̂ ≤ 1.003).

## Results — clearest ability→growth-rate association

| Model  | Structure                                | Clearest result (Receptive grammar, TROG)                         | Evidence    |
| ------ | ---------------------------------------- | ----------------------------------------------------------------- | ----------- |
| gc-085 | age × ability interaction on growth rate | +0.15 logit growth-rate change per +1 SD ability (+0.03 to +0.28) | very strong |
| gc-069 | independent-core growth curves           | +0.11 logit (+0.00 to +0.23)                                      | strong      |
| gc-070 | shared growth-tempo factor               | +0.11 logit (−0.01 to +0.23)                                      | moderate    |

## The one-paragraph story

Higher **baseline non-verbal ability** is associated with **somewhat faster growth**, most clearly for receptive grammar — children who start with stronger general ability tend to progress a little faster on the verbal/reading measures. The effect is consistent across three model structures but modest in size. This describes heterogeneity in trajectories; it is not a claim that raising ability would speed growth.

## What is causal

**Nothing.** These are between-child associations between baseline ability and trajectory shape. They characterise _who_ grows faster — useful context for the causal intervention findings elsewhere, but not levers.
