> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Findings: the `level_factors` family — scores at each timepoint

**Read `findings-00-overview` first.** This note covers the 11 models in the `level_factors` family, the companion "levels" view to the change-based `gain_factors` models.

## The data

**RLI trial only.** Each child contributes one row per timepoint across the study, giving 54 children and roughly 210–215 rows. Data are **stacked across timepoints and not collapsed**, but unlike every other treatment family, these models do **not** use the child's own baseline score as a covariate.

## What the model is for

Where `gain_factors` asks "did this child improve more than their starting point predicts?", this family asks the simpler question: **"how high was the score at each timepoint, and did the two arms differ?"**

Dropping the baseline adjustment is deliberate, not an oversight. It makes the model answer a question about levels rather than change, which is sometimes the more natural thing to report — parents and teachers care about where a child is, not only about the increment. The cost is precision: without the baseline covariate, the model cannot remove the large stable differences between children, so its intervals are wider.

Group and ability enter as **per-timepoint** coefficients rather than single overall terms, so the model estimates a separate arm difference at every wave. A child random intercept handles the repeated measurements.

The same crossover logic governs interpretation: **only the timepoint-2 contrast is randomised.** Later timepoints compare two treated groups.

Unlike `gain_factors`, this family adjusts for the non-measure background factors (hearing, speech, phonological memory) but takes **no other skill measures** as covariates. That is a considered restriction: conditioning a levels model on another skill's score at the same timepoint would be conditioning on something the intervention itself may have changed, which would distort the very group difference being estimated.

## How to read the results

The reported quantity is the arm difference at timepoint 2, in items, positive favouring the immediate-intervention group.

| Measure                           | t2 contrast (items) | 89% range     |
| --------------------------------- | ------------------- | ------------- |
| Letter-sound knowledge (L)        | **+2.5**            | +0.2 to +4.9  |
| Word reading (W)                  | +1.7                | −1.0 to +4.2  |
| Basic concept knowledge (F)       | +0.5                | −0.6 to +1.6  |
| Phoneme blending (B)              | +0.4                | −0.4 to +1.3  |
| Taught expressive vocabulary (TE) | +0.4                | −1.0 to +1.8  |
| Taught receptive vocabulary (TR)  | +0.3                | −1.0 to +1.7  |
| Receptive grammar (T)             | +0.2                | −1.4 to +1.8  |
| Nonword reading (N), off-floor    | +2 pp               | −10 to +13 pp |
| Phonetic spelling (P), off-floor  | −1 pp               | −9 to +8 pp   |
| Expressive vocabulary (E)         | −2.2                | −6.3 to +1.9  |
| Receptive vocabulary (R)          | −3.7                | −8.1 to +0.7  |

## What was found

**The direction of the pattern survives, but almost everything is less certain.** Letter-sound knowledge remains the strongest result (+2.5 items, interval clearing zero), and word reading remains positive (+1.7) but with an interval that now spans zero. Compare across families for word reading: `itt` +2.4 [+0.7, +4.1], `gain_factors` +2.6 [+0.9, +4.3], `did` +2.2 [−0.3, +4.7], and here +1.7 [−1.0, +4.2]. The estimates are compatible; the precision differs markedly, exactly as the design predicts.

**This family is the weakest of the four, and it should be read as the least informative rather than as a contradiction.** Removing the baseline covariate discards the single most useful predictor of a child's score. The intervals here are roughly 50% wider than the corresponding `itt` intervals for the same outcomes and children.

**The two broad vocabulary measures come out negative — R at −3.7 items and E at −2.2 — and this is the largest apparent disagreement anywhere in the treatment families.** It deserves an explanation rather than a shrug, so I checked the obvious one: chance baseline imbalance. It does not hold. The difference-in-differences models estimate the pre-treatment arm gap directly, and for both measures it is essentially zero — receptive vocabulary −0.011 on the log-odds scale (89% −0.184 to +0.163) and expressive vocabulary +0.032 (89% −0.155 to +0.216). The arms started balanced on vocabulary.

**These two results should not be dismissed as noise, and calling them "inconclusive" would understate them.** Both intervals include zero, but direction is judged by the tail probability, not by whether the band clears zero. The posterior probability that the t2 contrast is negative is **0.91 for receptive vocabulary and 0.80 for expressive vocabulary** — _suggestive_ evidence of a negative contrast on the project's ladder, and the fits label them that way themselves.

So the four treatment families do not simply agree on "nothing happened" for broad vocabulary. Two of them lean negative at the suggestive level — this family and `gain_factors` (−1.8 items, P(negative) = 0.78) — while `itt` (+0.2) and `did` (−0.1) are genuinely inconclusive. That is a real inconsistency and it is worth stating plainly.

Where does that leave the reader? With the `itt` estimate as the one to prefer for the randomised question, because it is the best-identified: it uses the randomised window, adjusts for each child's own baseline, and has by far the tightest interval. This family's estimate is the least precise of the four — it discards the baseline covariate entirely — and `gain_factors` pools across post-crossover periods. The honest summary is that broad standardised vocabulary showed no reliable benefit, with a weak negative lean in two of four specifications that is not strong enough to claim harm but is too consistent to call noise. It is worth watching in any future analysis rather than filing as a null.

## What these models cannot tell you

**Only the timepoint-2 contrast is randomised.** The timepoint-3 and timepoint-4 group coefficients compare two treated groups and are associations.

**Ability and interaction terms are adjusted associations**, describing which children scored higher.

**These are levels, not changes.** A child can be lower in level and still have gained more; the two questions are different and this family answers only the first.

**Do not treat this family as a refutation of the others.** It is the same children analysed with less statistical leverage. Where it and the `itt` family disagree, the `itt` estimate is better identified for the treatment question.

## Model inventory

All 11 pass the convergence gate with zero divergences and are publishable. `lf-001` (W), `002` (R), `003` (E), `004` (L), `005` (P), `006` (B), `007` (F), `008` (T), `009` (TR), `010` (TE), `011` (N). `lf-005` and `lf-006` were initially withheld for prior-dominance and released after a prior-sensitivity sweep confirmed directional stability.
