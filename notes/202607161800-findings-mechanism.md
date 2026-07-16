# Findings — mechanism family (dose–response between measured skills)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8). Numbers from the `reporting`-config refit of 2026-07-16 (see [process note](202607161130-full-statistical-refit.md)). Preliminary.

## What these models ask

"If a child has a higher level of skill X, how much higher is skill Y, among children who are otherwise alike?" Each model fits a **curve** relating one measured skill (the exposure) to another (the outcome) across the fitted range, adjusting for the variables the causal diagram says to adjust for, and reports the **average items-scale difference** across that range. These are **adjusted associations between skills** — emphatically **not** "changing X causes Y". They describe the shape of the reading system, and complement the [mediation](202607161800-findings-mediation.md) models, which try (under stronger assumptions) to make the causal split.

## Convergence gate

All 12 **passed**.

## Headline associations (outcome = word reading W unless noted)

| Model        | Exposure → outcome                                     | Avg difference across range     | P(>0) | Evidence    |
| ------------ | ------------------------------------------------------ | ------------------------------- | ----- | ----------- |
| **mech-088** | Taught receptive vocab (TR) → W                        | **+10.1 items** (+5.2 to +15.0) | 99.9% | very strong |
| **mech-089** | Taught expressive vocab (TE) → W                       | +9.3 items (+3.0 to +14.9)      | 99.8% | very strong |
| **mech-073** | Letter sounds (L) → W, age-moderated                   | +7.3 items (+2.2 to +12.7)      | 99.8% | very strong |
| **mech-158** | L → W, complete-case comparator                        | +6.9 items (+1.5 to +12.4)      | 99.6% | very strong |
| **mech-058** | Letter sounds (L) → W                                  | +6.7 items (+1.6 to +11.9)      | 99.6% | very strong |
| mech-173     | L → W (+age, no interaction baseline)                  | +6.6 items (+1.6 to +11.8)      | 99.6% | very strong |
| mech-071     | L → W, moderated by expressive vocab                   | +5.2 items (+0.2 to +10.9)      | 98%   | strong      |
| mech-057     | Expressive vocab (E) → W                               | +5.3 items (−1.9 to +12.1)      | 93%   | moderate    |
| **mech-072** | L (moderated by blending B) → **nonword decoding (N)** | +4.0 items (+2.5 to +5.0)       | 99.9% | very strong |
| mech-172     | L + B main effects → N (baseline)                      | +3.3 items (+2.0 to +4.4)       | 99.9% | very strong |
| mech-090     | Phonological memory → W                                | +3.1 items (−0.5 to +6.6)       | 95%   | moderate    |
| mech-056     | Receptive vocab, std (R) → W                           | +2.9 items (−3.8 to +9.4)       | 80%   | suggestive  |

## The one-paragraph story

The skills that sit closest to reading in the causal diagram — **letter-sound knowledge** and the **directly-taught vocabulary** — show the strongest, most clearly-resolved associations with word reading, and **letter sounds → nonword decoding** is very strongly positive (the expected code-route signature). Associations with **broad standardised receptive vocabulary** are weaker and less certain. These curves are consistent with the mediation story (reading gains run through the code) but, on their own, only describe co-variation among skills.

## What is causal

**Nothing here is causal.** Every curve is an adjusted association. Do not read "L → W, +6.7 items" as "teaching one more letter sound adds 6.7 words read" — it means children who happen to know more letter sounds also tend to read more words, among children otherwise alike. (For the cross-model LOO comparisons of these mechanism variants, see `output/statistical_models/comparison/`.)
