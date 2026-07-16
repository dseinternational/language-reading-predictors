# Findings — concurrent family (per-wave conditional associations between skills)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8). Numbers from the `reporting`-config refit of 2026-07-16 (see [process note](202607161130-full-statistical-refit.md)). This family was added in issue #312; this refit is its first gate-passing `reporting` fit. Preliminary.

## What these models ask

At a single wave, "among children alike on age and the other skills, does a child who is higher on skill X also tend to be higher on the focal skill Y?" This is the regression-style description of the **joint distribution of skills at each timepoint** — a modern version of the paper's own correlation tables. Each coefficient is reported **adjusted** (holding other same-wave skills fixed) and **bivariate**, on the items scale.

**This family makes no causal claim at all — by design.** All children are on the programme from t2, so these coefficients condition on post-treatment skills; that would wreck a causal reading, but it is fine for pure description. Every term is flagged `Status.ASSOCIATION`, and the report carries the Table-2-fallacy caveat (each coefficient answers a _different_ conditional question) and a regression-dilution note (observed-score predictors attenuate the associations relative to the latent truth).

## Convergence gate

All 6 **passed** cleanly (0 divergences, very high ESS ≈ 25 000).

## Results — the clearest same-wave adjusted link per focal skill

| Model  | Focal skill                  | Strongest adjusted same-wave partner | Association                                  | Evidence    |
| ------ | ---------------------------- | ------------------------------------ | -------------------------------------------- | ----------- |
| ca-001 | Word reading (W)             | Letter sounds, at t2                 | +8.1 outcome items per +1 SD (+4.9 to +11.3) | very strong |
| ca-002 | Letter sounds (L)            | Word reading, at t2                  | +4.3 items (+2.9 to +5.4)                    | very strong |
| ca-006 | Expressive vocab (E)         | Taught expressive vocab, at t3       | +6.4 items (+2.3 to +10.7)                   | very strong |
| ca-005 | Receptive vocab (R)          | Expressive vocab, at t4              | +6.2 items (+1.2 to +11.5)                   | very strong |
| ca-004 | Taught expressive vocab (TE) | Expressive vocab, at t3              | +2.5 items (+1.2 to +3.7)                    | very strong |
| ca-003 | Taught receptive vocab (TR)  | Receptive vocab, at t4               | +1.1 items (+0.2 to +1.9)                    | very strong |

## The one-paragraph story

Within any given wave, the skills cluster exactly as reading science would predict: **word reading and letter-sound knowledge move together most tightly**, and the vocabulary measures cluster with each other. These are strong, clean _descriptions_ of how skills co-occur in the cohort — useful for understanding structure, but they say nothing about what drives what.

## What is causal

**Nothing — intentionally.** The models condition on post-treatment skills, so no causal pathway is licensed; the report says so up front. For the causal questions see [ITT](202607161800-findings-itt.md); for the assumption-heavy "which skill carries the effect" question see [mediation](202607161800-findings-mediation.md).
