> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5.1).

# Findings: the `mechanism` family — which skills track which

**Read `findings-00-overview` first.** This note covers the 46 models in the `mechanism` family, the largest in the project. **Nothing in this family is causal.** Every number is an adjusted association. All 46 pass the convergence gate with zero divergences and all are publishable (2026-09-01 rebuild); the five Batch C variants (`mech-301`–`305`) are reported here for the first time at reporting tier.

## The data

**RLI trial only.** These models stack all period transitions (1→2, 2→3, 3→4): most fits have 52–54 children contributing 144–159 rows; the complete-case and positive-attendance comparators (`mech-158`, `191`) have 44 and 52 children on 128 rows. For the score-exposure models a row carries the outcome at the end of the period, the same outcome at the start (an autoregressive baseline), the exposure skill's post-period level and the adjustment terms; the raw-covariate variants (phonological memory, speech production, sessions) use a standardised raw score instead. Each child has a random intercept.

## What the model is for

For the score-exposure fits: **is a higher post-period exposure level associated with a higher post-period outcome after conditioning on the outcome's own starting score and the adjustment set?** Some models fit a straight line; others a flexible curve. Since #602 every fit publishes **one declared natural-scale headline**: the predicted outcome difference between the 75th and 25th percentile of the fitted exposure, standardised over the fitted rows with each child's own intercept retained. The full observed-range contrast is kept as a labelled secondary. Adjustment terms (age, hearing, speech production, phonological memory, attendance) come from the causal diagram.

## The specificity test

The suite includes **negative-control outcomes** the diagram gives letter-sound knowledge no decoding route to. On the common per-SD logit scale, and in items across the interquartile letter-sound range (17 to 28 sounds):

| Route                                     | Role               | Slope per SD (logit) | 89% range      | Items, IQR contrast      |
| ----------------------------------------- | ------------------ | -------------------- | -------------- | ------------------------ |
| Letter sounds → **nonword reading**       | positive control   | **1.03**             | +0.74 to +1.34 | +1.0 of 6 [+0.7, +1.3]   |
| Letter sounds → **basic concepts**        | _negative control_ | 0.29                 | +0.16 to +0.42 | +1.2 of 18 [+0.7, +1.8]  |
| Letter sounds → **word reading**          | positive control   | 0.25                 | +0.15 to +0.35 | +2.6 of 79 [+1.6, +3.5]  |
| Letter sounds → **grammar**               | _negative control_ | 0.12                 | +0.04 to +0.20 | +1.1 of 32 [+0.4, +1.8]  |
| Letter sounds → **receptive vocabulary**  | _negative control_ | 0.11                 | +0.06 to +0.16 | +4.0 of 170 [+2.2, +5.7] |
| Letter sounds → **expressive vocabulary** | _negative control_ | 0.10                 | +0.06 to +0.15 | +3.4 of 170 [+1.8, +4.9] |

**The negative controls are not zero.** All four are clearly positive: letter-sound knowledge carries a broad association that reaches outcomes the diagram gives it no causal path to, so something shared runs through every skill here. But the panel has a gradient. The three load-bearing oral-language controls sit at 0.10 to 0.12 per SD; word reading at 0.25 and nonword reading at 1.03, roughly two and nine times higher. Basic concepts (0.29) is the exception and the row that carries the least weight: an 18-item measure the design listed as optional, whose interval overlaps word reading's across almost its whole length, with no contrast between the two ever fitted.

The pre-specified primary signature is the **nonword-minus-word contrast**. Pairing the two separate fits gives **+0.78 log-odds per SD** (89% +0.47 to +1.10), which the project's own artefact flags as not an identified posterior contrast; the identified within-model version from the bivariate `jm-002` fit is **+0.81** (+0.49 to +1.13) with essentially all posterior mass above zero. The decoding-specificity test passed, clearly. What stops that settling the question: nonword reading is a 6-item floored measure whose logit slope is not commensurate with a 79-item scale, and general ability remains unblockable by construction.

## Testing the ability explanation directly

The whole panel was refitted with the measured block-design ability score partialled out (`mech-196`–`201`, identical rows and terms otherwise):

| Route                      | Without ability | With ability        | Ability's own coefficient (per SD) |
| -------------------------- | --------------- | ------------------- | ---------------------------------- |
| Letter sounds → nonwords   | 1.03            | **1.03**            | +0.03 [−0.21, +0.28]               |
| Letter sounds → word read. | 0.25            | **0.24**            | +0.04 [−0.04, +0.12]               |
| Letter sounds → concepts   | 0.29            | 0.25                | **+0.23** [+0.10, +0.35]           |
| Letter sounds → grammar    | 0.12            | 0.07 [−0.01, +0.15] | **+0.19** [+0.11, +0.27]           |
| Letter sounds → rec. vocab | 0.11            | 0.09                | **+0.10** [+0.05, +0.16]           |
| Letter sounds → exp. vocab | 0.10            | 0.10                | **+0.08** [+0.03, +0.14]           |

**The written-code slopes are untouched** and the decoding contrast is unchanged. **Measured ability really does predict the oral-language outcomes** and clearly not the written-code ones — the adjustment does real work exactly where a general-ability path was predicted. **But the negative controls mostly survive it**: only grammar attenuates substantially (to an interval that includes zero); the other three keep their direction with very strong evidence. The same holds for the fitted shape: the ability-adjusted curve `mech-258` reads +2.7 items across the interquartile range (89% +0.6 to +4.6) against +2.8 (+0.8 to +4.6) for `mech-058`. The shared cause is therefore not, or not only, the general ability this battery measures — though a single noisy subtest cannot rule the latent node out, because adjusting for a mismeasured confounder removes only part of its influence.

## The between/within split — the caveat that matters most

`mech-301` splits the letter-sound exposure of the linear `mech-101` model into each child's fitted-row mean and their deviation from it. **Between** children the slope is +0.44 per SD (89% +0.30 to +0.60); **within** a child it is +0.03 (−0.11 to +0.18), and the within-child interquartile contrast is +0.3 words (89% −1.2 to +1.8, P = 0.65) — inconclusive. The pooled `mech-101` slope of 0.25 is a precision-weighted blend of those two. In other words, almost all of the letter-sound-to-word-reading association in this family separates children who know more letter sounds from those who know fewer; the association between a child's own letter-sound movement and their own reading movement is not resolved here. The `pooled_levels` family reaches the same conclusion from a levels design, and finds a larger within-child part for nonword decoding (`pl-002`, +0.33 per SD). This is the single most important qualification on the mechanism story.

## Other results

**Word reading's other candidate routes**, per SD of the exposure: letter sounds 0.25 [+0.15, +0.35] (linear) or a secant of 0.27 [+0.07, +0.45] across the interquartile range of the curve; taught receptive vocabulary 0.20 [+0.11, +0.29] (+2.0 words, IQR); taught expressive vocabulary 0.19 [+0.07, +0.32] (+2.2 words); phonological memory 0.12 [+0.02, +0.22] (+1.7 words, P = 0.97); expressive vocabulary 0.13 [−0.01, +0.26] (+1.3 words, P = 0.93); receptive vocabulary 0.07 [−0.06, +0.19] (+0.8 words, P = 0.81). Letter sounds lead; the taught-vocabulary and phonological-memory routes are resolved but smaller; broad vocabulary is weak once the reading baseline is in the model. Toward **nonword decoding**, phonological memory (0.94 per SD; +1.4 nonwords across the IQR) and speech production (0.70; +0.8 nonwords) are both clearly positive.

**Curve tests.** No stored curve has a qualified knee: `mech-058`'s steepest interval sits at the top of the letter-sound range (29.5 sounds, boundary-pinned) on both the logit and the items scale, so no threshold is located within the data. The vocabulary-route curves are flat (`mech-156` +0.1 words, `mech-157` −0.1) and the taught-vocabulary curves positive (`mech-188` +1.9 [+0.2, +3.7]; `mech-189` +0.9 [−0.3, +2.9]). **The sessions-to-reading curve has collapsed**: once the 28 rows without observed attendance are excluded (`mech-191`, 128 rows) the interquartile contrast is +0.2 words (89% −0.8 to +1.4, P = 0.64); the apparent dose signal the August series reported was carried by the excluded zero-attendance rows.

**Phase stability (`mech-302`, `303`).** Letting the letter-sound slope vary by period gives 0.32, 0.29 and 0.19 per SD for word reading across the three transitions (between-period spread 0.14, 89% 0.02 to 0.50) and 0.06, 0.14 and 0.10 for receptive vocabulary; the nested predictive comparisons against the pooled fits are inconclusive. A difference between periods is evidence against pooling, not evidence that the relationship changed, because only the first transition is randomised-arm-clean and the periods differ in age and treatment history. **Dispersion prior (`mech-304`, `305`)**: +2.6 and +4.0 items against +2.6 and +4.0 for the parents — the concentration prior is not what these slopes rest on.

**Moderation.** Every moderated letter-sound-to-word-reading fit returns a negative product term of about the same size whatever the moderator: blending −0.11 (89% −0.20 to −0.01), taught expressive −0.09, age −0.06, nonword decoding −0.06, expressive vocabulary −0.06, taught receptive −0.06, receptive vocabulary −0.05, phonological memory −0.01. Re-expressed in words at the interquartile cells, the substitution holds in items for **blending** (the 17-to-28-sound increment is worth +2.8 words at 4 blending items and +1.6 at 8: a difference of −1.1 words, 89% −2.3 to −0.0, P = 0.95) and for **age** (+3.5 words at 77 months, +2.6 at 101: −1.0, 89% −1.8 to −0.1, P = 0.96); the other moderators are suggestive or inconclusive on the items scale, and every nested predictive comparison is inconclusive. Two cautions travel with all of these: a logit-scale product on a bounded outcome is not a statement about items, and the uniform negativity may be curvature at the letter-sound ceiling that the curve cannot express. The blending-on-decoding interaction (`mech-072`, −0.32, 89% −0.56 to −0.09) is strong on the logit scale but worth only −0.1 nonwords on the items scale.

## What changed since the August notes

Every headline in this family now uses the interquartile estimand, so the items numbers are smaller than the observed-range contrasts the August notes quoted (e.g. `mech-101` +2.6 rather than +9.8 words) while the posteriors are unchanged; the observed-range row still reproduces the old figures. Three substantive movements since August, all from the 2026-08-26 batch and reproduced here: the `mech-191` sessions signal disappeared under the population fix; `mech-301` put the within-child letter-sound slope at inconclusive; and the phase-varying, dispersion and complete-case (`mech-158`, +3.1 words, 89% +0.9 to +5.1) variants all agree with their parents.

## What these models cannot tell you

**No slope here is identified as a lever.** **Adjusting for measured covariates does not remove residual confounding**; general ability is structurally unblockable, and the ability panel partials out a proxy only. **Direction is not established**: these are contemporaneous-period associations with an autoregressive baseline. **The pooled slopes are mostly between-child** (`mech-301`), which is the pattern a shared cause predicts as readily as a direct influence. **The exposure was not manipulated.**

## Model inventory

All 46 pass the convergence gate with zero divergences and are publishable. Key models: `056`/`057`/`058` (R/E/L → W), `096`/`101` (Tier-1 decoding contrast), `097`–`100` (negative controls), `196`–`201` (ability-adjusted panel), `258` (ability-adjusted curve), `088`/`089` (taught vocabulary → W), `090`/`102` (phonological memory → W/N), `103` (speech → N), `061`/`063`/`071`/`073`/`093`–`095`/`104` (moderation) with `161`/`163`/`172`/`204` (no-interaction baselines), `072` (code route), `156`–`158`/`188`–`191` (curve tests and the complete-case comparator), `301` (between/within split), `302`/`303` (phase-varying slopes), `304`/`305` (dispersion prior). `mech-204` was refitted at a higher acceptance target after one divergence; `mech-302`/`303` were among the eight fits unblocked by PR #650.
