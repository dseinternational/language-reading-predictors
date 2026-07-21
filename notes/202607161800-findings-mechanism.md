# Findings — mechanism family (dose–response between measured skills)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8). Numbers from the `reporting`-config refit under the median + inner-50% + outer-89% equal-tailed credible-interval standard (2026-07-18; see [the credible-interval standard note](202607172359-credible-interval-standard.md) and [process note](202607161130-full-statistical-refit.md)); reviewed and extended on 2026-07-17 to cover all models in the family. Only the credible-interval brackets changed when we moved from 95% to 89% — medians, direction probabilities and evidence labels are unchanged. Preliminary.

## What these models ask

"If a child has a higher level of skill X, how much higher is skill Y, among children who are otherwise alike?" Each of the 12 `mechanism`-family models fits a **curve** relating one measured skill (the exposure) to another (the outcome) across the exposure's fitted range, adjusting for the variables the causal diagram says to adjust for, and reports the **average items-scale difference** in the outcome between the bottom and the top of that range. All phases are stacked with a child random intercept (a partial, shrunken stand-in for stable between-child differences — **not** a control for latent general ability). These are **adjusted associations between skills** — emphatically **not** "changing X causes Y". They describe the shape of the reading system, and complement the [mediation](202607161800-findings-mediation.md) models, which try (under stronger assumptions) to make the causal split. See also the [ITT](202607161800-findings-itt.md) findings for the randomised-effect anchors.

A note on the Bayesian vocabulary used below, for readers more used to frequentist reporting: an **89% credible range** is the interval the parameter lies in with 89% probability _given the data and priors_ — a direct probability statement about the parameter itself, unlike a confidence interval, which is a statement about the procedure. **P(>0)** is the posterior probability that the true effect is positive (so P(>0) = 0.98 means a 98% chance the association points the way stated). Evidence labels follow the project ladder graded on that direction probability: inconclusive (<0.75), suggestive (0.75–0.91), moderate (0.91–0.97), strong (0.97–0.99), very strong (≥0.99).

## Convergence gate

All 12 **passed**: zero divergences in every model, and the worst R-hat across the family is 1.0025 (threshold 1.01). No models flagged.

## Headline associations (outcome = word reading W unless noted)

Each row is the average items-scale difference in the outcome from the bottom to the top of the exposure's own fitted range. **The exposure ranges differ enormously, so the raw items numbers are not directly comparable across rows** — a bigger swing over a wider range is not "more important". Ranges are given so the numbers can be read honestly; mech-090's exposure is on the phonological-memory raw-score (erbto) scale, **not** the items scale of the other rows.

| Model        | Exposure → outcome                                      | Exposure range | Avg difference across range         | P(>0) | Evidence    |
| ------------ | ------------------------------------------------------- | -------------- | ----------------------------------- | ----- | ----------- |
| **mech-088** | Taught receptive vocab (TR) → W †                       | TR 6→23        | **+9.0 items** (+4.9 to +13.0)      | ~100% | very strong |
| **mech-089** | Taught expressive vocab (TE) → W †                      | TE 0→19        | +8.6 items (+3.2 to +13.4)          | 99.3% | very strong |
| **mech-073** | Letter sounds (LS) → W, age-moderated                   | L 2→32         | +7.3 items (+3.2 to +11.7)          | 99.8% | very strong |
| **mech-158** | L → W, complete-case comparator                         | L 2→32         | +6.9 items (+2.4 to +11.4)          | 99.6% | very strong |
| **mech-058** | Letter sounds (LS) → W                                  | L 2→32         | +6.7 items (+2.6 to +10.9)          | 99.6% | very strong |
| mech-173     | L → W (+age, no-interaction baseline)                   | L 2→32         | +6.6 items (+2.5 to +10.9)          | 99.6% | very strong |
| mech-057     | Expressive vocab (EV) → W                               | E 10→77        | +5.3 items (−0.5 to +10.9)          | 92.7% | moderate    |
| mech-071     | L → W, moderated by expressive vocab                    | L 2→32         | +5.2 items (+1.0 to +9.8)           | 98.3% | strong      |
| **mech-072** | L (moderated by blending B) → **nonword decoding (NW)** | L 2→32         | +4.0 items (+2.8 to +4.8)           | ~100% | very strong |
| mech-172     | L + B main effects → N (no-interaction baseline)        | L 2→32         | +3.3 items (+2.2 to +4.2)           | ~100% | very strong |
| mech-090     | Phonological memory (erbto) → W                         | 3→36 (erbto)   | +3.1 raw-score units (+0.1 to +6.0) | 95.2% | moderate    |
| mech-056     | Receptive vocab, std (RV) → W                           | R 12→82        | +2.9 items (−2.6 to +8.2)           | 80.4% | suggestive  |

The "~100%" cells for mech-072 and mech-172 are exactly 1.0 in the source CSVs; the earlier draft printed "99.9%" as a display cap, which slightly understated them and was inconsistent with the verbatim 99.6/99.8 elsewhere, so they are shown as ~100% here. (mech-088's P(>0) is 0.9996 after the IS correction below — still ~100%, no longer exactly 1.0.)

**† mech-088 (TR → W) and mech-089 (TE → W) now adjust for intervention sessions (IS).** They were re-fitted on 2026-07-17 to add `attend` to the adjustment set, reversing the earlier #309 decision to leave IS unadjusted. `IS → TR`/`IS → TE` and `IS → WR` make session dose a genuine confounder, and the collider path a naive adjustment could open (`IG → IS ← GA → W`) is closed at the always-conditioned randomised arm `G` — so adjusting IS is both safe and necessary (full reasoning: [collider-review note](202607171700-mech-intervention-sessions-adjustment-collider-review.md)). The correction moved the associations modestly **down** — TR from +10.1 to **+9.0** items, TE from +9.3 to **+8.6** — i.e. shared session dose had been mildly inflating both slopes, exactly as the previous draft anticipated. Both still converge cleanly (0 divergences) and remain very strong. This brings the taught-vocabulary models into line with mech-058 (L → W), which already adjusted IS.

## Results — all models (with 50% ranges and adjusted-association detail)

Every coefficient below is a **latent-ability-confounded adjusted association**, never a causal dose (see "What is causal"). The `mechanism` family does not compute a ROPE ("region of practical equivalence" — a band around zero deemed too small to matter) for its headline items-scale difference, so the "big enough to matter" judgement is left to the reader against the exposure range; the direction probability and evidence label are the formal read-outs.

| Model    | Exposure → outcome    | Best est. (median) | 89% credible range | 50% range     | P(>0)  | Evidence    | Causal?     |
| -------- | --------------------- | ------------------ | ------------------ | ------------- | ------ | ----------- | ----------- |
| mech-056 | R → W                 | +2.9 items         | −2.6 to +8.2       | +0.6 to +5.2  | 0.804  | suggestive  | association |
| mech-057 | E → W                 | +5.3 items         | −0.5 to +10.9      | +2.9 to +7.7  | 0.927  | moderate    | association |
| mech-058 | L → W                 | +6.7 items         | +2.6 to +10.9      | +5.0 to +8.4  | 0.996  | very strong | association |
| mech-071 | L → W (× E)           | +5.2 items         | +1.0 to +9.8       | +3.5 to +7.1  | 0.983  | strong      | association |
| mech-072 | L → N (× B)           | +4.0 items         | +2.8 to +4.8       | +3.5 to +4.3  | ~1.0   | very strong | association |
| mech-073 | L → W (× age)         | +7.3 items         | +3.2 to +11.7      | +5.6 to +9.1  | 0.998  | very strong | association |
| mech-088 | TR → W (adjusts IS) † | +9.0 items         | +4.9 to +13.0      | +7.3 to +10.7 | 0.9996 | very strong | association |
| mech-089 | TE → W (adjusts IS) † | +8.6 items         | +3.2 to +13.4      | +6.4 to +10.7 | 0.993  | very strong | association |
| mech-090 | erbto → W             | +3.1 raw units     | +0.1 to +6.0       | +1.8 to +4.3  | 0.952  | moderate    | association |
| mech-158 | L → W (complete-case) | +6.9 items         | +2.4 to +11.4      | +5.0 to +8.8  | 0.996  | very strong | association |
| mech-172 | L → N (+ B main)      | +3.3 items         | +2.2 to +4.2       | +2.9 to +3.7  | ~1.0   | very strong | association |
| mech-173 | L → W (+ age main)    | +6.6 items         | +2.5 to +10.9      | +4.9 to +8.4  | 0.996  | very strong | association |

Sample sizes: 156 stacked observations for every model except the complete-case comparator **mech-158** (125 observations — it drops rows with any missing adjuster, and reassuringly lands on +6.9 items, essentially on top of the +6.7 of the full-data mech-058).

### The moderation / interaction models (071, 072, 073, 172, 173)

Five models add a moderator on top of the letter-sound exposure, so the single headline marginal above hides the more interesting story — how the moderator shifts the slope. These secondary terms are on the **logit (log-odds) scale** of the Beta-Binomial linear predictor, not the items scale, and every one is an adjusted association.

- **mech-071 — L → W moderated by expressive vocabulary (EV).** The E-moderator **main term** is +0.191 logit (89% +0.078 to +0.303, P(>0) = 0.997, very strong): among otherwise-alike children, higher expressive vocabulary is associated with more word reading. The **L × E interaction** is −0.065 (89% −0.152 to +0.019, P(>0) = 0.110) — i.e. unresolved: the data neither confirm nor rule out that expressive vocabulary flattens or steepens the letter-sound slope.
- **mech-072 — L → N moderated by blending (PA), the code-route signature.** Blending **main term** +0.361 logit (89% +0.095 to +0.625, P(>0) = 0.985, strong). The **L × B interaction** is a clear **negative**: median −0.327 (89% −0.570 to −0.093, P(>0) = 0.013, i.e. P(<0) = 0.987, strong). Read plainly: higher blending skill **flattens** the letter-sound → nonword-decoding slope — the two code skills are partial substitutes rather than additive, so knowing more letter sounds buys less extra nonword decoding once a child already blends well.
- **mech-073 — L → W moderated by age.** The **age × L interaction** is −0.057 (89% −0.119 to +0.006, P(>0) = 0.072) and the **age main term** is −0.069 logit (89% −0.152 to +0.013, P(>0) = 0.091) — both point mildly negative but are unresolved.
- **mech-172 — L + B main effects → N (the no-interaction baseline for mech-072).** Blending **main term** +0.281 logit (89% +0.024 to +0.531, P(>0) = 0.958, moderate). With the interaction removed, blending still carries a positive additive association with nonword decoding.
- **mech-173 — L → W with an age main term (the no-interaction baseline for mech-073).** The **age main term** is −0.100 logit (89% −0.177 to −0.024, P(>0) = 0.019, i.e. P(<0) = 0.981, strong negative): holding letter sounds fixed, older children in this stacked cross-phase view are associated with slightly _less_ word reading — a between-child confounding pattern, not a within-child decline, and certainly not causal.

### Threshold / "readiness" shape of the letter-sound → word-reading curve

The five L → W models fit with a Gaussian-process curve (mech-058, -071, -073, -158, -173) each ship a `readiness_threshold.csv`, and they tell the same story that a single average slope hides: **the letter-sound → word-reading association is not a straight line — it accelerates as children approach mastery of the ~32 letter sounds.**

- **Knee** (where the curve steepens): median ≈ **29.5 of 32 letter-sound items**, credible range roughly 20 to 29.5 — near the top of the observed range.
- **Half-rise** (the exposure level at which half the total rise has happened): ≈ 23–25 letter-sound items.
- **Slope above the knee** ≈ 0.058–0.070 (logit per item) is roughly **double the slope below** (≈ 0.023–0.032).
- **increasing_frac** ≈ 0.993–0.999: the fitted curve is monotonically increasing across essentially every posterior draw.
- The fitted-items curve (`mechanism_curve_items.csv`) confirms the smooth rise — e.g. mech-058 goes from ≈5.3 words read at L = 2 to ≈11 at L = 32.

Plainly: children associate with more word reading right across the letter-sound range, and the pay-off _steepens_ rather than saturating as they near the full set of sounds — consistent with a "readiness" reading where the code becomes usable for reading once most sounds are in place.

**Caveats.** (1) This is an **adjusted association, not a causal dose** — it is not evidence that teaching a child from 29 to 32 sounds _adds_ the steep slope. (2) The knee credible range is wide (20–29.5), so its exact location is uncertain. (3) Bounded/logit-link Beta-Binomial models can manufacture _some_ curvature mechanically near the ceiling, so the acceleration should not be over-read as a pure cognitive threshold. (4) For nonword decoding, the analogous "threshold" signal is the strong **negative** L × B interaction in mech-072 above (blending flattens the letter-sound slope), not a knee.

## The one-paragraph story

The skills that sit closest to reading in the causal diagram — **letter-sound knowledge** and the **directly-taught vocabulary** — show the strongest, most clearly-resolved associations with word reading (taught receptive vocabulary +9.0 items and taught expressive +8.6, both now adjusting for intervention sessions and over narrow exposure ranges; letter sounds +6.6 to +7.3 items across variants over the full 2→32 range), and **letter sounds → nonword decoding** is very strongly positive (+3.3 to +4.0 items, the expected code-route signature). Broad **standardised receptive vocabulary** is the weakest and least certain link (+2.9 items, P(>0) = 0.80, suggestive only). The letter-sound → word-reading curve **accelerates** toward the top of the range rather than saturating, and among the code skills, blending and letter sounds appear to partly substitute for one another (the negative L × B interaction). All of these are cross-phase dose–response associations of one skill's _level_ on another — **not** within-child growth trajectories across the four timepoints, and not causal. They are consistent with a code-route mediation story but do not, on their own, identify it.

## What is causal

**Nothing here is causal.** Every curve, main term and interaction is an adjusted association, confounded by latent general ability the child random intercept only partially soaks up. Do not read "L → W, +6.7 items" as "teaching one more letter sound adds 6.7 words read" — it means children who happen to know more letter sounds also tend to read more words, among children otherwise alike. The only randomisation-licensed causal contrasts in this project live elsewhere (τ in the [ITT](202607161800-findings-itt.md) family, τ_t2 in the DiD family, the period-1 treatment marginal in the gain-factors family). The mechanism family's job is to describe co-variation in the reading system and hand the harder causal split to the [mediation](202607161800-findings-mediation.md) models. (For the cross-model LOO comparisons of these mechanism variants, see `output/statistical_models/comparison/`.)
