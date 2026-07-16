# Findings — joint family (all outcomes together, and taught-vs-not-taught contrasts)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8). Numbers from the `reporting`-config refit of 2026-07-16 (see [process note](202607161130-full-statistical-refit.md)). Preliminary.

## What these models ask

The [ITT models](202607161800-findings-itt.md) fit one outcome at a time. The **joint** models fit several outcomes **in one model**, letting their residuals correlate. This does two jobs: (1) it gives a single, internally-consistent picture across outcomes, and (2) it lets us form **contrasts between outcomes** — e.g. "is the effect on _taught_ vocabulary bigger than on _not-taught_ vocabulary?" — with the uncertainty handled correctly. As with ITT, each outcome's treatment effect is **causal** (random assignment); the cross-outcome contrasts are causal comparisons of those effects.

## Convergence gate

All 4 joint models **passed**.

## Results

| Model       | What it combines                                               | Headline                                                                                                                                                                                                       |
| ----------- | -------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ITT-012** | The full 10-outcome suite jointly                              | Best estimates span **−0.1 to +3.5 items**; the clearest is **letter sounds** at 99.9% positive (very strong). Of the 9 outcomes with a "matters in practice" bar, **4** are more likely than not to clear it. |
| **ITT-016** | Taught expressive (TE) **vs** taught receptive (TR) vocabulary | Both positive (+1.4 to +1.5 items); TE 99% positive (strong). Both outcomes more likely than not to clear their 1-item bar.                                                                                    |
| **ITT-015** | Generalisation, expressive: taught (TE) **vs** not-taught (UE) | +0.4 to +1.5 items; TE 98% positive (strong). Only the **taught** side clears its bar — the benefit is concentrated on what was directly taught.                                                               |
| **ITT-115** | Generalisation, receptive: taught (TR) **vs** not-taught (UR)  | +0.7 to +1.4 items; TR 97% positive (moderate). Again the **taught** side is the stronger.                                                                                                                     |

## The one-paragraph story

Fitting the outcomes together tells the same story as the single-outcome ITT models — the effect is largest and clearest on letter sounds and the directly-taught vocabulary. The generalisation contrasts (015/115) sharpen one point: the vocabulary benefit is **concentrated on the words the programme actually taught** and does not spill over strongly to untaught words within the trial window. This is the expected signature of a targeted teaching effect rather than a broad language boost.

## What is causal

Each outcome's treatment effect is causal (randomised). The taught-vs-not-taught contrasts are causal comparisons _of those effects_. Residual correlations between outcomes are descriptive.
