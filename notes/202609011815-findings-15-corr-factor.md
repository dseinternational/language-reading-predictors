> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5.1).

<!-- cspell:ignore basnum basspel woco -->

# Findings: the `corr_factor` and `long_corr_factor` families — skill domains behind the tests

**Read `findings-00-overview` first.** This note covers the 5 `corr_factor` models and the single `long_corr_factor` model. **Nothing here is causal.** All 6 pass the convergence gate with zero divergences; 5 are publishable and one, the historical-cohort measurement model, is withheld at the inputs stage (2026-09-01 rebuild).

## The data

**Mostly the RLI trial.** Four `corr_factor` models use 51 children at baseline, one row each. The `long_corr_factor` model uses all four waves — 54 children, 216 rows. `rlm-mm-001` uses 75 children from the historical cohort at wave 3.

## What the model is for

Every test score contains measurement error, and correlating two noisy tests directly attenuates the correlation. These are **measurement models**: the underlying domains — vocabulary {R, E, TR, TE}, code {L, B}, grammar {F, T} — are treated as unobserved quantities that several tests each measure imperfectly, and the correlations _between the domains_ are estimated with measurement error removed. The `long_corr_factor` model extends this across all four waves with wave-invariant loadings and separates the stable trait part of each domain from its wave-specific state part.

## What was found

| Model     | Relationship                                                | Latent correlation | 89% range      |
| --------- | ----------------------------------------------------------- | ------------------ | -------------- |
| `mm-001`  | Vocabulary ↔ grammar                                        | **+0.83**          | +0.67 to +0.93 |
| `mm-001`  | Vocabulary ↔ code                                           | **+0.78**          | +0.57 to +0.93 |
| `mm-001`  | Code ↔ grammar                                              | **+0.70**          | +0.43 to +0.88 |
| `mm-002`  | Vocabulary ↔ code                                           | **+0.80**          | +0.60 to +0.93 |
| `mm-101`  | Vocabulary ↔ code                                           | +0.76              | +0.52 to +0.92 |
| `mm-102`  | Vocabulary ↔ code                                           | +0.80              | +0.61 to +0.93 |
| `lcf-001` | Vocabulary ↔ grammar, wave 1 (waves 2–4 within a hundredth) | +0.88              | +0.82 to +0.93 |
| `lcf-001` | Vocabulary ↔ code, wave 1 (waves 2–4 within a hundredth)    | +0.70              | +0.58 to +0.79 |
| `lcf-001` | Code ↔ grammar, wave 1 (waves 2–4 within a hundredth)       | +0.61              | +0.46 to +0.72 |

These correlations are **very high** — high enough to raise a question the models cannot settle: whether these are distinct domains in this sample or facets of one broadly varying ability. A latent correlation of 0.83 means vocabulary and grammar share about two-thirds of their variance once measurement error is removed.

The loading-prior sensitivity `mm-101` gives +0.76 for vocabulary–code against +0.78 in `mm-001`, so that correlation is broadly stable to the tested loading geometry; `mm-102` changes only the structural code-slope prior and is not a correlation-prior sensitivity. Power scaling flags several cross-sectional parameters for potential prior–data conflict, so "not an artefact of the prior" would be stronger than the checks establish.

The longitudinal model adds two results. Its per-wave latent correlations are effectively constant across the four waves (vocabulary–code 0.70 at every wave, vocabulary–grammar 0.88, code–grammar 0.61), and it attributes **95–96% of each domain's latent variance to the stable child trait** rather than wave-specific state (vocabulary 0.96, 89% 0.93 to 0.99; code 0.95, 0.87 to 0.99; grammar 0.96). Its item-scale translation at wave 1 is that one additional letter-sound item corresponds to +0.61 receptive-vocabulary items (89% +0.50 to +0.69) — an association at the mean operating point, not a longitudinal coupling.

## Why the high correlations matter for the rest of the project

If vocabulary, code and grammar are as tightly bound as these estimates suggest, any model that adjusts for one while estimating the association of another is adjusting for something largely overlapping. That makes individual coefficients in the `mechanism`, `concurrent` and `adjusted` families harder to interpret, and it gives a concrete measurement to set beside the non-specific component the `mechanism` family's negative controls expose and the between-child dominance the `pooled_levels` and `mech-301` splits find. All three are compatible with a substantial shared component across skills whose source none of them identifies.

## The structural coefficients

All four cross-sectional trial models carry a structural leg regressing word-reading gain on the latent factors. `mm-001` and `mm-101` give code slopes of +0.20 (89% −0.15 to +0.56, P = 0.82) and +0.19 — suggestive; `mm-002`, with a code-focused adjustment set, gives +0.35 (+0.10 to +0.61, P = 0.99), and widening only that prior in `mm-102` moves it to +0.47 (+0.18 to +0.77), so the structural estimate is visibly prior- and specification-sensitive. Two other structural terms are resolved in all four: age is negative (about −0.32 to −0.37 per SD, P(negative) ≥ 0.998) and, in the `mm-002`/`102` specification, the hearing-risk flag is positive (+0.21, 89% +0.04 to +0.38, P = 0.97) — the same counter-intuitive lean the `adjusted` family's RLI model shows. Every structural coefficient is an adjusted association confounded by latent ability.

## The withheld model

`rlm-mm-001`, the historical-cohort version (reading, language, memory and ability domains at wave 3), is **withheld at the inputs stage**: three of its measures (BAS spelling, WORD comprehension, BAS number skills) have no confirmed maximum score, so the bounded-count likelihood rests on a guess about the instruments. It is computationally clean and withheld because a measurement fact is missing.

## What these models cannot tell you

**Correlation between domains says nothing about direction or cause.** **High correlations do not prove the domains are one thing.** **The measurement model is an assumption** — which tests load on which domain was specified in advance. **The cross-sectional fits describe baseline between-child correlations**; the longitudinal model adds stability over waves, not cross-lagged change (that is `lcsm`).

## Model inventory

All six pass the convergence gate with zero divergences. Publishable: `mm-001` (three-domain measurement model), `mm-002` (errors-in-variables code → word reading), `mm-101`/`102` (prior sensitivities) and `lcf-001` (four-wave longitudinal). Withheld at the inputs stage: `rlm-mm-001`. `mm-101` was one of the eight fits unblocked by PR #650.
