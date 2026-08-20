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

**This family was re-parameterised in #552 and refitted on 2026-08-20.** The arm-by-time terms are now centred on the timepoint-1 gap: `arm_gap_t1` is the covariate-adjusted arm difference _before anyone was treated_, and `d_grp_time[t]` is the change in that gap at each later wave. The reported quantity is therefore **`d_grp_time[t2]` — a difference-in-differences of adjusted levels**, translated to items, positive favouring the immediate-intervention group. The per-wave gaps the family used to report (`b_grp_time`) are retained as a `levels_view` Deterministic, and the free per-timepoint parameterisation survives as an explicit comparator (`arm_gap_reference="free"`).

The change matters because the two quantities answer different questions. `b_grp_time[1]` is _where the arms stood at timepoint 2_, which includes any difference they started with; `d_grp_time[t2]` is _how much the gap moved over the randomised window_, which is the quantity randomisation identifies. The translation to items applies the increment across the observed timepoint-2 profiles at centred mean ability and excludes the time-invariant group-by-ability term, so it is not a population-average heterogeneous effect.

| Measure                           | t2 change in the adjusted gap (items) | 89% range    |
| --------------------------------- | ------------------------------------- | ------------ |
| Letter-sound knowledge (L)        | **+2.8**                              | +0.8 to +4.9 |
| Word reading (W)                  | **+2.3**                              | +0.3 to +4.3 |
| Taught expressive vocabulary (TE) | +1.3                                  | −0.1 to +2.7 |
| Taught receptive vocabulary (TR)  | +1.2                                  | −0.3 to +2.6 |
| Basic concept knowledge (F)       | +0.8                                  | −0.3 to +1.8 |
| Phoneme blending (B)              | +0.6                                  | −0.1 to +1.4 |
| Receptive grammar (T)             | +0.6                                  | −1.0 to +2.3 |
| Nonword reading (N), off-floor    | +2.9 pp                               | −8 to +14 pp |
| Receptive vocabulary (R)          | +0.2                                  | −4.2 to +4.6 |
| Expressive vocabulary (E)         | +0.1                                  | −3.9 to +4.2 |
| Phonetic spelling (P), off-floor  | +0.2 pp                               | −8 to +9 pp  |

## What was found

**Under the timepoint-1-centred parameterisation this family now agrees with the rest of the suite.** Letter-sound knowledge is the strongest result (+2.8 items, interval clearing zero, P = 0.988) and word reading is second (+2.3, interval clearing zero, P = 0.964). Compare across families for word reading: `itt` +2.4 [+0.7, +4.1], `gain_factors` +2.6 [+0.9, +4.3], `did` +2.2 [−0.3, +4.7], and here +2.3 [+0.3, +4.3]. The four are now close in both point and precision.

**The negative vocabulary contrasts this note previously reported were the baseline gap, and the new parameterisation shows it arithmetically.** Under the old free parameterisation the reported quantity was the timepoint-2 arm gap itself, which came out at −3.7 items for receptive vocabulary and −2.2 for expressive. Those numbers have not changed — they are still what the arms looked like at timepoint 2, and the `levels_view` rows still report them — but they decompose exactly:

| Measure                   | Gap before treatment (`arm_gap_t1`) | Change over the randomised window (`d_grp_time[t2]`) | Timepoint-2 gap (`b_grp_time[1]`) |
| ------------------------- | ----------------------------------: | ---------------------------------------------------: | --------------------------------: |
| Receptive vocabulary (R)  |             −0.161 [−0.314, −0.007] |                               **+0.008** (P = 0.535) |                            −0.153 |
| Expressive vocabulary (E) |             −0.123 [−0.290, +0.044] |                               **+0.005** (P = 0.522) |                            −0.118 |
| Word reading (W)          |             +0.033 [−0.356, +0.425] |                               **+0.347** (P = 0.964) |                            +0.380 |

For the two vocabulary measures the whole of the timepoint-2 gap is the gap the arms started with, and the randomised change is indistinguishable from zero. For word reading almost all of it is the randomised change. The August reading of this note — that the negative vocabulary contrasts were "inheriting a baseline offset rather than evidence of harm" — was correct, and is now a property the model reports rather than an inference a reader has to make. **The specification sensitivity that discussion was built on is much reduced, though not gone.** This family no longer leans negative on broad vocabulary — receptive +0.2 (P(positive) = 0.54) and expressive +0.1 (0.52), both inconclusive — and neither does `itt` (+0.2) or `did` (−0.1). `gain_factors` still does: receptive vocabulary −2.0 items with P(negative) = 0.80, _suggestive_ on the ladder. So the disagreement that used to run across two families now rests on one, and the family that has dropped out did so because the quantity it reports was corrected, not because its data changed.

**The precision penalty is smaller than previously reported.** With the contrast centred on the timepoint-1 gap, this family's 89% intervals are 11–23% wider than the matching `itt` intervals (word reading 1.20×, letter sounds 1.11×, receptive vocabulary 1.11×, expressive vocabulary 1.23×), not the ~53% recorded under the old parameterisation for word reading. The levels specification still changes both the adjustment and the repeated-measures model, so no single ratio describes the family.

**Power scaling no longer flags the treatment term.** In August two fits in this family were classified prior-dominant and needed a treatment-prior sweep before they could publish. All eleven now publish without one: `d_grp_time[t2]` is likelihood-dominated in nine of the eleven, and the "strong prior / weak likelihood" flag has moved to `arm_gap_t1` in eight of them, where it bears on the baseline-balance quantity rather than on the effect. The two exceptions on the focal term are the floor-rule outcomes P and N, and the gate classifies neither as prior-dominant.

The `itt` estimate remains the planned headline for the randomised question because it uses the randomised window and adjusts for each child's own baseline. This family is now a corroborating view of the same contrast rather than a discrepant one.

## What these models cannot tell you

**Only the reported timepoint-2 group contrast at mean ability is randomised.** The timepoint-3 and timepoint-4 group coefficients compare two treated groups and are associations. The time-invariant group-by-ability interaction is estimated across all waves, so it is not folded into the causal headline.

**Ability and interaction terms are adjusted associations**, describing which children scored higher.

**These are levels, not changes.** A child can be lower in level and still have gained more; the two questions are different and this family answers only the first.

**Do not treat this family as a refutation of the others.** It is the same children analysed with less statistical leverage. Where it and the `itt` family disagree, the `itt` estimate is better identified for the treatment question.

## Model inventory

All 11 pass the convergence gate with zero divergences and are publishable. `lf-001` (W), `002` (R), `003` (E), `004` (L), `005` (P), `006` (B), `007` (F), `008` (T), `009` (TR), `010` (TE), `011` (N). `lf-005` and `lf-006` were initially withheld for prior-dominance and released after a prior-sensitivity sweep confirmed directional stability.
