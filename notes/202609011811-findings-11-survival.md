> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5.1).

# Findings: the `survival` family — how quickly children first move off the floor

**Read `findings-00-overview` first.** This note covers the 2 models in the `survival` family. Both pass the convergence gate with zero divergences and are publishable (2026-09-01 rebuild). Since the #631 review the treatment term is described as a model-based, available-case modified-ITT assignment contrast in the randomised first interval among children at the floor at baseline, and the predictive check leaves out children rather than person-period rows.

## The data

**RLI trial only**, restricted to children **at the floor at timepoint 1**. Phonetic spelling has 41 children contributing 100 person-period rows; nonword reading 36 children contributing 74 rows. Each child contributes one row per interval during which they were still at the floor and still followed; once they move off the floor they stop contributing.

## What the model is for

For a child sitting at zero, how likely are they to move above zero during the next interval, and did the randomised arms differ in that chance over the one interval where they can be compared? A discrete-time survival model with a complementary-log-log link, a separate baseline hazard per interval, and baseline letter sounds, word reading and age as prognostic covariates. After the crossover every child in the risk set is on the intervention, so the treatment contrast is fitted **in the randomised first interval only** and the later intervals fit their own both-arms-treated hazards. The result is a **hazard ratio**: how much the intervention arm multiplies the underlying first-interval hazard of coming off the floor.

## What was found

| Outcome           | Hazard ratio (interval 1) | 89% range    | P(faster) | Evidence     | Untreated first-interval off-floor probability | Later intervals, both arms treated |
| ----------------- | ------------------------- | ------------ | --------- | ------------ | ---------------------------------------------- | ---------------------------------- |
| Nonword reading   | 1.60                      | 0.85 to 3.00 | 0.88      | suggestive   | 25% [14%, 40%]                                 | 37%, 29%                           |
| Phonetic spelling | 1.08                      | 0.56 to 2.09 | 0.57      | inconclusive | 19% [10%, 33%]                                 | 17%, 14%                           |

**Neither supports a firm claim, but they are not equivalent.** Nonword reading reaches suggestive evidence for a faster exit under the intervention; phonetic spelling is close to even. The prognostic covariates are more resolved than the treatment term: baseline word reading (hazard ratio 1.65 per SD, 89% 1.19 to 2.31, P = 0.99) and baseline letter sounds (1.38 per SD, P = 0.95) both predict an earlier exit from the nonword floor, and letter sounds (1.38, P = 0.95) predicts an earlier exit from the spelling floor. Power scaling flags the treatment term in both models (a strong-prior flag for phonetic spelling, a potential prior–data conflict for nonword reading; the regularising prior is strong relative to a contrast estimated from about 40 children), so prior robustness has not been established even for the directional reading.

For nonword reading three approaches now lean the same way: +10 percentage points off-floor in `itt`, +2 points in `gain_factors`, +6 points in `did`, and a first-interval hazard ratio of 1.6 here. They share the randomised anchor and children, so this is descriptive triangulation rather than independent confirmation, and none is strong. The phonetic-spelling picture is flat rather than coherent: `itt` (+4 points), `did` (+2), `level_factors` (+0) and this model lean faintly positive while `gain_factors` (−1) and `aligned` (−1) lean faintly negative, every interval straddling no difference.

## What these models cannot tell you

**They cannot show the intervention did not help these skills**; inconclusive is not null. **They describe only the floored subgroup.** **Moving off the floor is a low bar** — one nonword of six is the event. **Later intervals carry no arm comparison.** **The hazard ratio is not released as a causal headline**: it is randomisation-anchored but adjusted, available-case and hazard-model-dependent.

## Model inventory

Both pass the convergence gate with zero divergences and are publishable: `surv-009` (phonetic spelling) and `surv-011` (nonword reading). Neither records a data checksum, one of the 14 provenance gaps the rebuild note lists; nothing consumes the field for this family.
