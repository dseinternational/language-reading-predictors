> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).
>
> Substantially corrected by a LLM-based AI tool (Codex/GPT-5).

# Findings: the `level_factors` family — scores at each timepoint

**Read `findings-00-overview` first.** This note covers the 11 models in the `level_factors` family, the companion "levels" view to the post-score-given-pre-score `gain_factors` models.

## The data

**RLI trial only.** Each child contributes one row per timepoint across the study, giving 53–54 children and roughly 207–215 rows. Data are **stacked across timepoints and not collapsed**, but unlike the `itt` and `gain_factors` families, these models do **not** use the child's own baseline score as a covariate (the arm-by-wave `did` models likewise model levels without it).

## What the model is for

Where `gain_factors` asks whether the post-period score is higher after conditioning on the pre-period score, this family asks the simpler question: **"how high was the score at each timepoint, and did the two arms differ?"**

Dropping the baseline adjustment is deliberate, not an oversight. It makes the model answer a question about levels rather than change, which is sometimes the more natural thing to report — parents and teachers care about where a child is, not only about the increment. Without the baseline covariate the model cannot remove the same stable between-child variation, which can reduce precision, although the interval-width difference is not uniform across outcomes.

Group and ability enter as **per-timepoint** coefficients, so the model estimates a separate arm difference and ability association at every wave. The registered models also contain one time-invariant group-by-ability term shared across all waves. A child random intercept handles the repeated measurements.

The same crossover logic governs interpretation: **only the timepoint-2 contrast is randomised.** Later timepoints compare two treated groups.

Unlike `gain_factors`, this family adjusts for the non-measure background factors (hearing, speech, phonological memory) but takes **no other skill measures** as covariates. That is a considered restriction: conditioning a levels model on another skill's score at the same timepoint would be conditioning on something the intervention itself may have changed, which would distort the very group difference being estimated.

## How to read the results

The reported quantity is `b_grp_time[1]` translated to items, positive favouring the immediate-intervention group. The translation applies that same group increment across the observed timepoint-2 profiles, which evaluates the contrast at centred mean ability and deliberately excludes the time-invariant group-by-ability term. It is therefore not a population-average heterogeneous effect that lets the treatment contrast vary with ability; that interaction is partly informed by post-crossover waves and remains an adjusted association.

| Measure                           | t2 contrast at mean ability (items) | 89% range     |
| --------------------------------- | ----------------------------------- | ------------- |
| Letter-sound knowledge (L)        | **+2.5**                            | +0.2 to +4.9  |
| Word reading (W)                  | +1.7                                | −1.0 to +4.2  |
| Basic concept knowledge (F)       | +0.5                                | −0.6 to +1.6  |
| Phoneme blending (B)              | +0.4                                | −0.4 to +1.3  |
| Taught expressive vocabulary (TE) | +0.4                                | −1.0 to +1.8  |
| Taught receptive vocabulary (TR)  | +0.3                                | −1.0 to +1.7  |
| Receptive grammar (T)             | +0.2                                | −1.4 to +1.8  |
| Nonword reading (N), off-floor    | +2 pp                               | −10 to +13 pp |
| Phonetic spelling (P), off-floor  | −1 pp                               | −9 to +8 pp   |
| Expressive vocabulary (E)         | −2.2                                | −6.3 to +1.9  |
| Receptive vocabulary (R)          | −3.7                                | −8.1 to +0.7  |

## What was found

**The direction of the main pattern survives, but several outcomes are less certain.** Letter-sound knowledge remains the strongest result (+2.5 items, interval clearing zero), and word reading remains positive (+1.7) but with an interval that now spans zero. Compare across families for word reading: `itt` +2.4 [+0.7, +4.1], `gain_factors` +2.6 [+0.9, +4.3], `did` +2.2 [−0.3, +4.7], and here +1.7 [−1.0, +4.2]. The estimates are compatible; for this outcome the levels model is less precise.

**This family is generally less informative for the randomised treatment question than the baseline-adjusted ITT model, rather than a refutation of it.** The interval-width change is outcome-specific: word reading is about 53% wider than its `itt` counterpart, receptive vocabulary about 10% wider, and the two off-floor intervals are narrower rather than wider. The levels specification changes both the adjustment and the repeated-measures model, so a blanket 50% rule is not supported.

**The two broad vocabulary measures come out negative — R at −3.7 items and E at −2.2 — and this is the largest apparent difference among the treatment-family medians.** The arm-by-wave models put the pre-treatment arm-gap medians near zero: receptive vocabulary −0.011 on the log-odds scale (89% −0.184 to +0.163) and expressive vocabulary +0.032 (89% −0.155 to +0.216). Those wide intervals show no clear baseline imbalance, but they also do not rule out chance imbalance large enough to matter. This family's own timepoint-1 group coefficient points the same way as its timepoint-2 one: for receptive vocabulary the ability-adjusted arm offset is −0.154 log-odds at timepoint 1 (89% −0.305 to −0.001) against −0.125 at timepoint 2 (89% −0.274 to +0.026), and for expressive vocabulary −0.091 against −0.085. A negative arm offset that is already present before anyone was treated is a starting-point difference, not an effect, and it is exactly the weakness of a levels model that carries no baseline covariate — so the negative vocabulary contrasts here should be read as inheriting a baseline offset rather than as evidence of harm.

Both treatment intervals include zero, but direction is judged by the tail probability rather than whether the band clears zero. The posterior probability that the timepoint-2 contrast is negative is **0.909 for receptive vocabulary and 0.80 for expressive vocabulary** — _suggestive_ evidence of a negative contrast on the project's ladder. The receptive-vocabulary value sits on the boundary of the next category (0.91), and with an effective sample size around 10,000 the Monte Carlo error on that tail probability is about ±0.003, so it is best read as borderline suggestive-to-moderate rather than as one or the other.

The four treatment families therefore should not be summarised as proving that nothing happened on broad vocabulary. For receptive vocabulary this family and `gain_factors` (−1.8 items, P(negative) = 0.78) lean negative at the suggestive level, while `itt` (+0.2) and `did` (−0.1) are directionally inconclusive; for expressive vocabulary only the levels model leans negative — `gain_factors` is +0.9, `itt` +0.2 and `did` +0.8, all inconclusive. Their 89% intervals overlap, and no posterior contrast between specifications was fitted, so this is **specification sensitivity**, not a demonstrated inconsistency.

The `itt` estimate remains the planned headline for the randomised question because it uses the randomised window and adjusts for each child's own baseline. The levels model omits that precision term, while `gain_factors` fits one shared treatment coefficient across stacked pre- and post-crossover periods before standardising its marginal to period 1. The honest summary is that broad standardised vocabulary showed no well-resolved benefit, with a weak negative lean in two specifications that is not strong enough to establish harm. That sensitivity warrants follow-up rather than being filed either as a null or as an adverse effect.

## What these models cannot tell you

**Only the reported timepoint-2 group contrast at mean ability is randomised.** The timepoint-3 and timepoint-4 group coefficients compare two treated groups and are associations. The time-invariant group-by-ability interaction is estimated across all waves, so it is not folded into the causal headline.

**Ability and interaction terms are adjusted associations**, describing which children scored higher.

**These are levels, not changes.** A child can be lower in level and still have gained more; the two questions are different and this family answers only the first.

**Do not treat this family as a refutation of the others.** It is the same children analysed with less statistical leverage. Where it and the `itt` family disagree, the `itt` estimate is better identified for the treatment question.

## Model inventory

All 11 pass the convergence gate with zero divergences and are publishable. `lf-001` (W), `002` (R), `003` (E), `004` (L), `005` (P), `006` (B), `007` (F), `008` (T), `009` (TR), `010` (TE), `011` (N). `lf-005` and `lf-006` were initially withheld for prior-dominance and released after a prior-sensitivity sweep confirmed directional stability.
