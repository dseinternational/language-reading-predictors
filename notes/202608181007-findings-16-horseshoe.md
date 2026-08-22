> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).
>
> Ranking probabilities, second-ranked level predictors, the cross-cohort claim and the gate/publishable counts corrected against the stored artefacts by a LLM-based AI tool (Claude Code/Opus 5).

# Findings: the `horseshoe` family — ranking many predictors at once

**Read `findings-00-overview` first.** This note covers the 7 models in the `horseshoe` family. **Nothing here is causal.**

## The data

**Both studies.** Four models use the RLI trial (51–54 children; the level models stack waves for 210–214 rows, the gain models use one row per child). Three use the historical cohort (69–71 children). One historical model is **withheld**.

## What the model is for

This family exists as a **cross-check on the machine-learning analysis** that forms step 1 of the project's methodology. That analysis ranks predictors by importance using gradient boosting. The obvious question is whether a completely different statistical approach picks out the same variables.

The difficulty is that with many candidate predictors and few children, ordinary regression overfits badly — it will happily assign large coefficients to noise. The **regularised horseshoe** prior addresses this by being aggressively sceptical: it pulls all coefficients hard towards zero unless the data insist otherwise, letting a few genuinely strong predictors escape while flattening the rest.

The headline quantity is not a coefficient but a **ranking score**: `p_abs_gt_delta`, the posterior probability that a standardised predictor's absolute association exceeds δ = 0.1 logit units. Coefficients are quoted below with 89% **highest-density** intervals, which is the interval this family's tables store; most other families in the series report equal-tailed intervals, so the two are not exactly interchangeable.

## What was found

| Model    | Outcome                | Rank 1        | P(&#124;β&#124; > 0.1) | Association (logit units) |
| -------- | ---------------------- | ------------- | ---------------------- | ------------------------- |
| `hs-004` | Letter-sound **level** | Word reading  | 1.000                  | **+0.70** [+0.52, +0.87]  |
| `hs-002` | Word-reading **level** | Letter sounds | 0.993                  | **+0.33** [+0.19, +0.48]  |
| `hs-001` | Word-reading **gain**  | Age           | 0.595                  | −0.14 [−0.34, +0.02]      |
| `hs-003` | Letter-sound **gain**  | Grammar       | 0.448                  | +0.07 [−0.05, +0.36]      |

**The level and gain models tell different stories, and that contrast is the main finding.**

For **levels** — who is scoring highly right now — letter sounds and word reading each top the other's ranking, with intervals well clear of zero. But the ranking is not a two-variable story: **expressive vocabulary comes second in both level models**, and in neither case is it far behind. In `hs-004` it scores 0.999 with a coefficient of +0.47 [+0.30, +0.66]; in `hs-002` it scores 0.992 against letter sounds' 0.993 — effectively tied — and its coefficient, +0.41 [+0.22, +0.60], is the **larger** of the two. Reading the level results as a clean reciprocal pair between letter sounds and word reading would overstate what the ranking shows.

For **gains** — who improves more — **nothing survives the shrinkage.** The top-ranked predictor of word-reading gain is age at 0.595, and of letter-sound gain grammar at 0.448; both sit below the 0.75 threshold the project treats as the weakest positive rung, and both intervals span zero. When even the _best_ predictor cannot clear a sceptical prior, the honest conclusion is that **these data do not identify who will gain most.**

The two publishable historical gain models point the same way even more starkly: the top-ranked predictor of receptive-vocabulary gain and of receptive-grammar gain is verbal reasoning (BAS Similarities) in both, scoring 0.155 and 0.154 with coefficients of +0.010 and +0.007. That is a ranking in name only.

## Why "nothing predicts gains" is a real result

It is tempting to treat a flat ranking as a failed analysis. It is better read as an informative one, for two reasons.

**Levels are easy and gains are hard.** Level models benefit from stable between-child differences that persist across every wave. Gains strip that stability out and leave the much smaller, noisier increment — plus regression to the mean, which works against any baseline predictor.

**It guards against over-reading the other families.** Where `mechanism` or `concurrent` report a predictor associated with progress, this family says that no such predictor is strong enough to survive honest shrinkage. That is a useful check on the temptation to build a profile of "children who respond well".

A third reason — that the result is consistent across cohorts — needs stating more carefully than it first appears. The flat-gain finding rests on two RLI models and the **two** historical models that may be reported. The historical cohort's third gain model is withheld at the inputs stage, so it can neither corroborate nor qualify the pattern, and the cross-cohort claim should be read as covering the reportable models only rather than the cohort as a whole.

## What these models cannot tell you

**A ranking is not a causal ordering.** A top-ranked predictor is not a lever.

**Absence from the ranking is not evidence of irrelevance.** The horseshoe deliberately suppresses weak signals; with 51 children a real but modest predictor will be flattened.

**Level results are close to tautological.** That word reading predicts letter-sound level, and vice versa, largely restates that both track the same underlying development.

**These are mutually adjusted associations**, so each coefficient is conditional on all the others in the set — and the sets differ between models, so ranks are not comparable across rows without checking what each model held constant.

## The withheld model

`rlm-hs-001` (word-reading gain in the historical cohort) is **withheld at the inputs stage**: `basnum` (BAS number skills) is one of its candidate predictors and its bounded-count denominator is not confirmed against the instrument. Its two sibling models cleared the gate because neither includes `basnum` in its predictor set. Its ranking is therefore not reported here, in either direction.

## Model inventory

All seven pass the convergence gate with zero divergences. Six are publishable: `hs-001` (word-reading gain), `002` (word-reading level), `003` (letter-sound gain), `004` (letter-sound level), `rlm-hs-002` (receptive-vocabulary gain), `rlm-hs-003` (receptive-grammar gain). `rlm-hs-001` is computationally clean but withheld at the inputs stage.

All seven fits reach that clean result only because their modules declare a raised acceptance target in-module — `hs-001`, `rlm-hs-003` and — since the 2026-08-22 refit under the dispersion-scale concentration prior, which produced five divergences at 0.99 — `rlm-hs-002` set `target_accept` 0.999, and the other four (`hs-002`, `hs-003`, `hs-004`, `rlm-hs-001`) set 0.99. Any refit must read `config.json` → `sampling.target_accept` rather than assuming the reporting preset's 0.95, since passing a uniform lower value would silently loosen these.
