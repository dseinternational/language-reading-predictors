> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Findings: the `horseshoe` family — ranking many predictors at once

**Read `findings-00-overview` first.** This note covers the 7 models in the `horseshoe` family. **Nothing here is causal.**

## The data

**Both studies.** Four models use the RLI trial (51–54 children; the level models stack waves for 210–214 rows, the gain models use one row per child). Three use the historical cohort (69–71 children). One historical model is **withheld**.

## What the model is for

This family exists as a **cross-check on the machine-learning analysis** that forms step 1 of the project's methodology. That analysis ranks predictors by importance using gradient boosting. The obvious question is whether a completely different statistical approach picks out the same variables.

The difficulty is that with many candidate predictors and few children, ordinary regression overfits badly — it will happily assign large coefficients to noise. The **regularised horseshoe** prior addresses this by being aggressively sceptical: it pulls all coefficients hard towards zero unless the data insist otherwise, letting a few genuinely strong predictors escape while flattening the rest.

The reported quantity is not a coefficient but a **ranking**: the posterior probability that each standardised predictor's absolute association exceeds a small threshold.

## What was found

| Model    | Outcome                | Top-ranked predictor | Association (logit units) |
| -------- | ---------------------- | -------------------- | ------------------------- |
| `hs-004` | Letter-sound **level** | Word reading         | **+0.70** [+0.52, +0.87]  |
| `hs-002` | Word-reading **level** | Letter sounds        | **+0.33** [+0.19, +0.48]  |
| `hs-001` | Word-reading **gain**  | Age                  | −0.14 [−0.34, +0.02]      |
| `hs-003` | Letter-sound **gain**  | Grammar              | +0.07 [−0.05, +0.36]      |

**The level and gain models tell completely different stories, and that contrast is the main finding.**

For **levels** — who is scoring highly right now — the answer is emphatic and mutual: letter sounds and word reading each top the other's ranking, with intervals well clear of zero. Children who read words know letter sounds and vice versa.

For **gains** — who improves more — **nothing survives the shrinkage.** The top-ranked predictor of word-reading gain is age, at −0.14 with an interval spanning zero; the top-ranked predictor of letter-sound gain is grammar at +0.07, likewise spanning zero. When even the _best_ predictor cannot clear a sceptical prior, the honest conclusion is that **these data do not identify who will gain most.**

The historical cohort gives the same answer even more starkly: the top-ranked predictors of receptive-vocabulary gain and of receptive-grammar gain are both verbal reasoning, at **+0.01** logit units, with intervals spanning zero. That is a ranking in name only.

## Why "nothing predicts gains" is a real result

It is tempting to treat a flat ranking as a failed analysis. It is better read as an informative one, for three reasons.

**It is consistent across cohorts and methods.** Two different studies, different measures, different children, same answer.

**Levels are easy and gains are hard.** Level models benefit from stable between-child differences that persist across every wave. Gains strip that stability out and leave the much smaller, noisier increment — plus regression to the mean, which works against any baseline predictor.

**It guards against over-reading the other families.** Where `mechanism` or `concurrent` report a predictor associated with progress, this family says that no such predictor is strong enough to survive honest shrinkage. That is a useful check on the temptation to build a profile of "children who respond well".

## What these models cannot tell you

**A ranking is not a causal ordering.** A top-ranked predictor is not a lever.

**Absence from the ranking is not evidence of irrelevance.** The horseshoe deliberately suppresses weak signals; with 51 children a real but modest predictor will be flattened.

**Level results are close to tautological.** That word reading predicts letter-sound level, and vice versa, largely restates that both track the same underlying development.

**These are mutually adjusted associations**, so each coefficient is conditional on all the others in the set.

## The withheld model

`rlm-hs-001` (word-reading gain in the historical cohort) is **withheld at the inputs stage** because one of its measures has no confirmed maximum score. Its two sibling models cleared the gate because they do not use the affected measure.

## Model inventory

Six of seven pass the convergence gate with zero divergences and are publishable: `hs-001` (word-reading gain), `002` (word-reading level), `003` (letter-sound gain), `004` (letter-sound level), `rlm-hs-002` (receptive-vocabulary gain), `rlm-hs-003` (receptive-grammar gain). `rlm-hs-001` is withheld.
