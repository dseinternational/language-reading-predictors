<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Findings 17 — the survival family (time to come off the floor)

Reports every model in the `survival` family from the 2026-08-04/05 `reporting` refit. **2 models, both passing the convergence gate.** Reading conventions in note 00. Preliminary research data — all estimates provisional.

## What these models do

Phonetic spelling (PS) and nonword reading (NW) sit hard on the floor — most children score zero, and a graded count model has little to work with. The ITT suite handles this with a single-transition off-floor indicator (note 01). This family generalises that to **all four waves**: for children starting at the floor, how long until they come off it, and does the intervention shorten that time?

**Design.** A discrete-time hazard model on person-period rows. Each row is one child in one interval, at risk of coming off the floor; the model estimates the per-interval hazard with a complementary log-log link, an intervention-aligned hazard shift (τ), and baseline covariates.

**The estimand is prognostic, not a clean treatment effect.** Because both arms are treated by t4, the treatment term here mixes assignment with timing; it is closer to "does being on the programme raise the off-floor hazard" than to a randomised contrast. Read τ as suggestive evidence at best, and the covariate slopes as adjusted associations.

## Results

| Term                                  | `surv-009` (Phonetic spelling)                | `surv-011` (Nonword reading)                  |
| ------------------------------------- | --------------------------------------------- | --------------------------------------------- |
| **τ — treatment hazard shift**        | −0.17 (−0.78 to +0.44), HR 0.84, P(>0) = 0.33 | +0.30 (−0.28 to +0.89), HR 1.35, P(>0) = 0.80 |
| Baseline letter sounds (per SD)       | **+0.34** (+0.03 to +0.66), HR 1.41, P = 0.96 | **+0.33** (+0.03 to +0.64), HR 1.40, P = 0.96 |
| Baseline word reading (per SD)        | +0.28 (−0.04 to +0.58), HR 1.32, P = 0.92     | **+0.50** (+0.17 to +0.84), HR 1.65, P = 0.99 |
| Baseline age (per SD)                 | +0.16 (−0.13 to +0.44), HR 1.17, P = 0.82     | −0.09 (−0.39 to +0.18), HR 0.91, P = 0.30     |
| Baseline off-floor probability, t1→t2 | 0.22 (0.12 to 0.36)                           | 0.27 (0.16 to 0.43)                           |
| t2→t3                                 | 0.19 (0.08 to 0.38)                           | 0.29 (0.14 to 0.53)                           |
| t3→t4                                 | 0.16 (0.06 to 0.35)                           | 0.22 (0.09 to 0.47)                           |

**The treatment terms are inconclusive and point in opposite directions.** For nonword reading, being on the programme is associated with a 35% higher off-floor hazard (P = 0.80, suggestive); for phonetic spelling the point estimate is _below_ one (HR 0.84, P(>0) = 0.33, inconclusive). Neither is resolved, and the opposite signs across two closely related floored skills are a fair indication of how little information there is in these data once the analysis is restricted to floor-starters.

This does not contradict the ITT floor-rule results (note 01: PS inconclusive, NW suggestive) — the directions match for nonword reading, and both families put phonetic spelling at nothing. It does mean the survival framing adds no additional evidence for a treatment effect. Note that the ITT floor-rule headlines are themselves **withheld** under the robustness release gate adopted on 2026-08-05 (note 01), so neither family currently offers a releasable treatment claim for these two outcomes.

**The prognostic result is much clearer, and it is the useful part.** Baseline **letter-sound knowledge** predicts coming off the floor on _both_ outcomes (hazard ratios 1.41 and 1.40, both P = 0.96), and baseline **word reading** predicts it strongly for nonword reading (HR 1.65, P = 0.99). A child one SD higher on letter sounds at baseline is around 40% more likely to come off the floor in any given interval.

That is coherent with the mediation family (note 08), which found the letter-sound route carries the intervention's effect on the nonword off-floor transition. Here the same skill predicts the transition irrespective of arm.

**The baseline hazard is lower by the last interval than the first** in both models — 0.22 → 0.19 → 0.16 for spelling, a clean monotone decline; 0.27 → 0.29 → 0.22 for nonword reading, which rises slightly before falling. The children who were going to come off the floor easily did so early, leaving a progressively harder-to-move group — a standard selection pattern in survival data, not a finding about the intervention. The intervals overlap heavily throughout, so the shape is a description of point estimates, not a resolved trend.

## Caveats

- **Prognostic estimand, not a randomised effect.** Both arms are treated by t4.
- **At-risk set only.** These models use only children at the floor at t1, so they describe a subgroup, and that subgroup is selected on the outcome measure.
- **The two treatment terms disagree in sign** and neither resolves — do not quote either.
- **Floor and survival estimands are zero-divergence-only** under the divergence policy; both fits meet that cleanly.
- **The prior is optimistic about off-floor movement.** For `surv-009` the prior-predictive event rate centres at 0.60 against an observed 0.17 — inside the prior's range but below its 25th percentile. The posterior is data-dominated, but this is a real prior-data tension worth recording, and it was only visible once this family began emitting prior-predictive checks in this run.
- **Small numbers.** 100 person-period rows for `surv-009`.
