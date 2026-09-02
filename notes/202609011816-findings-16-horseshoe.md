> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5.1).

<!-- cspell:ignore basnum bassim -->

# Findings: the `horseshoe` family — ranking many predictors at once

**Read `findings-00-overview` first.** This note covers the 7 models in the `horseshoe` family. **Nothing here is causal.** All 7 pass the convergence gate with zero divergences; 6 are publishable and one historical model is withheld at the inputs stage (2026-09-01 rebuild).

## The data

**Both studies.** Four models use the RLI trial (51–54 children; the level models stack waves for 210–214 rows, the gain models use one row per child). Three use the historical cohort (69–71 children).

## What the model is for

A **cross-check on the gradient-boosting layer** that forms step 1 of the project's methodology. With many candidate predictors and few children, ordinary regression overfits; the regularised horseshoe prior pulls every coefficient hard towards zero unless the data insist, letting a few strong predictors escape. The headline quantity is a ranking score, the posterior probability that a standardised predictor's absolute association exceeds 0.1 logit units; coefficients are quoted with 89% highest-density intervals.

## What was found

| Model        | Outcome                      | Rank 1            | P(\|β\| > 0.1) | Association (logit units) | Rank 2                                 |
| ------------ | ---------------------------- | ----------------- | -------------- | ------------------------- | -------------------------------------- |
| `hs-004`     | Letter-sound **level**       | Word reading      | 1.000          | **+0.70** [+0.52, +0.87]  | Expressive vocabulary 0.999, +0.47     |
| `hs-002`     | Word-reading **level**       | Letter sounds     | 0.993          | **+0.33** [+0.19, +0.48]  | Expressive vocabulary 0.992, **+0.41** |
| `hs-001`     | Word-reading **gain**        | Age               | 0.586          | −0.14 [−0.34, +0.02]      | Letter sounds 0.419, +0.06             |
| `hs-003`     | Letter-sound **gain**        | Receptive grammar | 0.456          | +0.07 [−0.04, +0.37]      | Basic concepts 0.292                   |
| `rlm-hs-002` | BPVS vocabulary gain (Byrne) | Verbal reasoning  | 0.193          | +0.02 [−0.03, +0.15]      | BAS word reading 0.118                 |
| `rlm-hs-003` | TROG grammar gain (Byrne)    | BPVS vocabulary   | 0.175          | +0.01 [−0.05, +0.16]      | Verbal reasoning 0.163                 |

**The level and gain models tell different stories, and that contrast is the main finding.** For **levels** — who is scoring highly now — letter sounds and word reading each top the other's ranking with intervals clear of zero, but expressive vocabulary comes second in both and in the word-reading level model its coefficient (+0.41) is the larger of the two. For **gains** — who improves more — nothing survives the shrinkage: the top-ranked predictor of word-reading gain is age at 0.59 and of letter-sound gain grammar at 0.46, both below the weakest positive rung and both with intervals spanning zero. The two publishable historical gain models are flatter still (top scores 0.19 and 0.18, coefficients of a hundredth or two). When even the best predictor cannot clear a sceptical prior, the honest conclusion is that **these data do not identify who will gain most**.

That is an informative result, not a failed analysis. Level models benefit from stable between-child differences that persist across waves; gains strip that out and leave a smaller, noisier increment plus regression to the mean. And it guards against over-reading the other families: where `gain_factors` or `adjusted` report a predictor of progress, this family says no such predictor is strong enough to survive honest shrinkage — which is not the same as saying the associations are absent.

**Against the gradient-boosting layer.** The cross-check pairs each RLI model with its boosting counterpart: the top-3 construct overlaps are 2/3 (`hs-001`/`gbg-012`: age and letter sounds lead both), 2/3 (`hs-002`/`gbl-012`: letter sounds and expressive vocabulary), 1/3 (`hs-003`/`gbg-009`: only basic concepts) and 2/3 (`hs-004`/`gbl-009`: word reading and blending). The Bayesian sparse-regression ranking broadly corroborates the boosting ranking without matching it term for term, and agrees most where there is most signal (levels) and least where there is least (letter-sound gain).

**The ranking sweep is incomplete.** Five of the twenty horseshoe prior-sensitivity cells did not converge (1–33 divergences, on `hs-002`, `hs-004` and `rlm-hs-001`), which limits the robustness cross-check rather than the primaries; there is no earlier batch record to compare against.

## What these models cannot tell you

**A ranking is not a causal ordering.** **Absence from the ranking is not evidence of irrelevance** — the horseshoe suppresses weak signals by design and with 51 children a real but modest predictor will be flattened. **Level results are close to tautological.** **These are mutually adjusted associations** whose sets differ between models.

## The withheld model

`rlm-hs-001` (word-reading gain in the historical cohort) is withheld at the inputs stage because BAS number skills, one of its candidate predictors, has no confirmed denominator. Its two siblings cleared the gate because neither includes that measure.

## Model inventory

All seven pass the convergence gate with zero divergences. Publishable: `hs-001` (word-reading gain), `002` (word-reading level), `003` (letter-sound gain), `004` (letter-sound level), `rlm-hs-002` (receptive-vocabulary gain), `rlm-hs-003` (receptive-grammar gain). Withheld: `rlm-hs-001`. All seven declare a raised acceptance target in-module (0.99 or 0.999); any refit must read it from `config.json` rather than assume the reporting preset's 0.95.
