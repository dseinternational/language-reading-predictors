> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5.1).

<!-- cspell:ignore bassim basdig -->

# Findings: the `adjusted` family — which baseline measures track later progress?

**Read `findings-00-overview` first.** This note covers the 7 models in the `adjusted` family. **Nothing here is causal.** All 7 pass the convergence gate with zero divergences; 6 are publishable and one historical model is withheld at the inputs stage (2026-09-01 rebuild).

## The data

**Both studies.** One model uses the RLI trial (51 children); six use the historical Reading, Language and Memory cohort (22–84 children). Most use one row per child — a set of baseline measures and a later outcome score, with the child's own baseline outcome as a covariate, a single-span post-score-given-baseline framing — and `rlm-adj-006` pools annual transitions across waves 1 to 5 (84 children, 225 rows).

## What the model is for

Given several baseline measures at once, which ones track later progress? Each predictor's coefficient is estimated holding the others constant, so it describes what that measure adds beyond the rest; the simple unadjusted association is reported alongside, and the two often differ. This is the `horseshoe` question asked with a conventional prior — more willing to report a value, more exposed to overfitting.

## What was found

| Model         | Cohort                         | Outcome                   | Strongest adjusted predictor | Association per +1 SD       | P(favoured) |
| ------------- | ------------------------------ | ------------------------- | ---------------------------- | --------------------------- | ----------- |
| `adj-065`     | RLI trial                      | word-reading gain         | **Age at t1**                | **−2.9 words** [−4.4, −1.3] | 0.997       |
| `rlm-adj-006` | historical, pooled transitions | word-reading progress     | **Age**                      | **−2.7 items** [−3.9, −1.2] | 0.998       |
| `rlm-adj-005` | historical                     | digit-recall gain         | Age                          | −1.1 items [−2.2, +0.0]     | 0.94        |
| `rlm-adj-003` | historical                     | receptive-vocabulary gain | Verbal reasoning             | +0.9 items [−0.2, +2.0]     | 0.89        |
| `rlm-adj-004` | historical                     | receptive-grammar gain    | Verbal reasoning             | +0.4 items [−0.2, +1.0]     | 0.88        |
| `rlm-adj-002` | historical, Down syndrome only | word-reading gain         | Receptive vocabulary         | −2.0 items [−4.5, +0.8]     | 0.88        |

**Age is the strongest adjusted predictor of word-reading progress in both cohorts, and it is negative in both.** Older children showed less progress on the item scale, conditional on their starting score. The repeated association across two separate studies makes a one-dataset quirk less likely; it does not identify why. Older children may begin nearer the top of a bounded test, item-scale development may slow with age, and cohort composition and residual confounding remain; nothing here shows that being older impedes learning.

The RLI model has three further resolved adjusted associations. The baseline **hearing-risk flag** is associated with **+2.2 words** per SD of the 0/1 indicator (89% +0.3 to +4.2, P = 0.97) — the flagged-versus-clear contrast is about two standard deviations, roughly twice that. The same counter-intuitive positive lean appears in the `corr_factor` structural leg and, weakly, in several `gain_factors` and `lcsm` fits; the flag is a coarse three-valued composite with nine unknowns, it is not randomised, and it may proxy unmeasured differences, so it does not imply that hearing difficulty improves reading. The **behaviour** rating is associated with −1.9 words per SD (89% −3.5 to −0.1, P(negative) = 0.96). And **letter sounds** carries +1.6 words (−0.6 to +4.2, P = 0.88) after mutual adjustment, half its unadjusted association (+0.29 logits, 89% +0.09 to +0.50, P = 0.99): letter sounds predict later reading gain on their own, and much of that is shared with the other baseline measures.

In the historical cohort the verbal-reasoning results point the expected way — more able children gaining somewhat more on vocabulary and grammar, and in the pooled word-reading model +1.8 items per SD (89% +0.2 to +3.4, P = 0.96) — but are small. **`rlm-adj-002` deserves its own caution**: 22 children with three mutually adjusted predictors, a wide interval, suggestive evidence at best, and no horseshoe companion.

## Set this beside the horseshoe result

The `horseshoe` family asked essentially this question with an aggressively sceptical prior and found no gain predictor that survived shrinkage, in either cohort. That strengthens the case for caution about unstable rankings and large point estimates without proving the associations absent: with correlated predictors and small samples a horseshoe prior shrinks genuine but weakly identified associations towards zero. The fair conclusion is that most baseline-predictor estimates are imprecise, some (the RLI and 22-child fits) are also prior-sensitive, and the resolved age, hearing-flag and behaviour associations remain descriptive.

## What these models cannot tell you

**No predictor here is a lever**, age least of all. **Mutually adjusted coefficients are not independent effects**, and given the very high latent correlations in the `corr_factor` note they overlap heavily. **A single baseline-to-post span discards the trajectory.** **Small samples.**

## The withheld model

`rlm-adj-001` (historical word-reading gain) is withheld at the inputs stage because one of its measures has no confirmed maximum score; its confirmed-input siblings cleared the gate.

## Model inventory

Six of seven are publishable: `adj-065` (RLI trial), `rlm-adj-002` (Down syndrome only), `003` (receptive vocabulary), `004` (receptive grammar), `005` (verbal memory), `006` (pooled annual transitions). Withheld: `rlm-adj-001`. The six historical fits record no data checksum, a provenance gap that nothing consumes.
