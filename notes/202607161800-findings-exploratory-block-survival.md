# Findings — exploratory families: block_exposure & survival

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8). Numbers from the `reporting`-config refit under the median + inner-50% + outer-89% equal-tailed credible-interval standard (2026-07-18; see [the credible-interval standard note](202607172359-credible-interval-standard.md) and [process note](202607161130-full-statistical-refit.md)); reviewed and extended on 2026-07-17 to cover all models in the family. Only the credible-interval brackets changed when we moved from 95% to 89% — medians, direction probabilities and evidence labels are unchanged. Preliminary. These are two small exploratory families grouped in one note.

## What these models ask

These are two small exploratory triangulation families that sit alongside the main [ITT](202607161800-findings-itt.md) analyses. Neither contains a randomisation-licensed contrast, so **nothing in either family is causal** — every quantity below is either an adjusted association or a descriptive prognostic quantity. They exist to cross-check the main story from a different angle, not to overturn it.

A quick reading guide for the Bayesian summaries, aimed at a reader more used to frequentist output. A **89% credible interval/range** is the range within which the parameter lies with 89% probability given the data and priors — unlike a confidence interval, it is a direct probability statement about the parameter itself. **P(effect > 0)** (reported here as a direction probability) is the posterior probability that the true effect points in a given direction. We grade that probability on the project evidence ladder (#179): below 0.75 is _inconclusive_, 0.75–0.91 _suggestive_, 0.91–0.97 _moderate_, 0.97–0.99 _strong_, and 0.99+ _very strong_. The ladder grades our confidence in the _direction_ of an effect, not that the effect is large enough to matter.

**block_exposure (`bx-001…004`) — what it asks.** The vocabulary teaching was delivered in two consecutive **blocks**: a first set of target words (block 1) was taught for a period, then a second, different set of target words (block 2) was taught in a later period. Because the block-2 words are only actively taught during the second window, we can ask a **parallel-trends** question: is a child's score on the block-2 words higher in the window when block-2 teaching is _active_ than in the earlier window when only block-1 teaching was running? "Parallel-trends" here means we compare the same measure across two exposure windows and read the difference (`delta`) as an exposure association. Crucially, **which words fell in which block was not randomised**, so `delta` is an association, not an effect. The four models cover taught expressive vocabulary (TE2), taught receptive vocabulary (TR2), and the two not-taught generalisation measures (UE2, UR2).

**survival (`surv-009`, `surv-011`) — what it asks.** Two outcomes sit almost entirely on their floor early in the trial: phonetic spelling (PS) and nonword reading (NW). Rather than model a score that is mostly zero, these discrete-time survival models ask a timing question: in each interval between waves, what is the **hazard** — the chance, among children still on the floor, of first "coming off the floor" (moving above zero) in that interval? A **hazard ratio (HR)** compares that per-interval chance between groups: HR > 1 means the group comes off the floor _earlier_ (higher chance each interval), HR < 1 means _later_. The headline `tau` term is a treated-vs-not hazard ratio, but because **both arms have been treated by the final wave**, it is a whole-trial prognostic association, not a randomised effect. Each model also carries baseline-skill covariates (baseline word reading W0, baseline letter-sound/L level L0, baseline age-and-ability A0), all entered per standard deviation.

## The one-paragraph story

The exploratory checks broadly agree with the main analyses and add no new causal claim. In `block_exposure`, the headline block-2-active differences are modest: the directly-taught expressive measure (bx-001) trends positive (suggestive), the directly-taught receptive measure (bx-002) actually trends _negative_ (suggestive), and both not-taught generalisation measures are inconclusive. What is consistent and strong in all four models is not the exposure term at all but the covariates: general cognitive ability and the block-2 vocabulary-exposure/teaching covariate are very strongly positively associated with the block-2 score everywhere. In `survival`, the treated-arm timing signal is weak — inconclusive for phonetic spelling (leaning later) and suggestive for nonword reading (leaning earlier) — while the stronger and more consistent signal is that children with higher **baseline word reading** and higher **baseline letter-sound level** come off the floor earlier. Every one of these stronger signals is a latent-ability-confounded prognostic association, not a treatment effect.

## Results — block_exposure (all four models)

**Gate:** all four models **passed** the convergence gate (R-hat ≤ 1.01, ESS ≥ 400, BFMI ≥ 0.3, 0 divergences); no flags. Detail: bx-001 max R-hat 1.0014 / min ESS 5427; bx-002 1.0012 / 4954; bx-003 1.0008 / 6726; bx-004 1.0018 / 4572; 0 divergences throughout.

### Headline block-2-active difference (`delta`) — association, not causal

`delta` is reported on two scales: the logit-scale coefficient (the model's linear-predictor units) and its translation into items on the block-2 word list. The direction probability and evidence label are identical on both scales.

| Model  | Measure                                | Difference (items) | 89% credible range | Direction prob | Evidence              |
| ------ | -------------------------------------- | ------------------ | ------------------ | -------------- | --------------------- |
| bx-001 | Taught expressive vocab, block 2 (TE2) | +0.72              | −0.46 to +1.91     | 0.834 positive | suggestive (positive) |
| bx-002 | Taught receptive vocab, block 2 (TR2)  | −0.74              | −1.87 to +0.43     | 0.845 negative | suggestive (negative) |
| bx-003 | Not-taught expressive, block 2 (UE2)   | −0.28              | −0.91 to +0.38     | 0.749 negative | inconclusive          |
| bx-004 | Not-taught receptive, block 2 (UR2)    | +0.15              | −0.51 to +0.82     | 0.640 positive | inconclusive          |

Logit-scale medians for reference: bx-001 +0.144 (−0.093 to +0.380), bx-002 −0.158 (−0.404 to +0.091), bx-003 −0.114 (−0.381 to +0.156), bx-004 +0.061 (−0.210 to +0.335). Note the directly-taught receptive measure (bx-002) is the one that trends the "wrong" way (suggestive negative, favoured-negative probability 0.845), and bx-003 sits just below the suggestive threshold at 0.749. There is no ROPE band defined for these exploratory models, so we report the credible range and direction only; every credible range comfortably straddles zero, so none of these is "big enough to matter" on the evidence here.

### Adjusted associations (`gamma_*`) — none causal

These are the covariates entered alongside `delta`. They are the more informative part of the block_exposure family: the exposure term is weak, but the ability and vocabulary-exposure covariates are strong and consistent. All are adjusted associations, latent-ability-confounded, never "X drives the block-2 score". Values are logit-scale coefficients per standard-deviation of the predictor; direction probability is for the favoured direction.

| Term                                            | bx-001                                     | bx-002                                    | bx-003                                     | bx-004                                     |
| ----------------------------------------------- | ------------------------------------------ | ----------------------------------------- | ------------------------------------------ | ------------------------------------------ |
| `gamma_ability` (general cognitive ability)     | +0.311 (0.170–0.454), 0.9998 — very strong | +0.400 (0.261–0.537), 1.000 — very strong | +0.348 (0.181–0.516), 0.9994 — very strong | +0.316 (0.185–0.450), 0.9999 — very strong |
| `gamma_erbto` (block-2 vocab exposure/teaching) | +0.375 (0.162–0.583), 0.997 — very strong  | +0.207 (0.066–0.345), 0.990 — very strong | +0.109 (−0.127–0.351), 0.773 — suggestive  | +0.276 (0.143–0.407), 0.9994 — very strong |
| `gamma_deapp_c` (DEAP phonology composite)      | −0.030 (−0.238–0.182), inconclusive        | not in model                              | +0.213 (−0.025–0.446), 0.925 — moderate    | not in model                               |
| `gamma_hs` (hearing status)                     | −0.043, inconclusive                       | −0.046, inconclusive                      | −0.057, inconclusive                       | +0.074 (−0.066–0.213), 0.804 — suggestive  |
| `gamma_A` (linear age)                          | −0.022, inconclusive                       | −0.026, inconclusive                      | −0.044, inconclusive                       | −0.031, inconclusive                       |

The pattern is clear and consistent: **general cognitive ability** is very strongly positively associated with the block-2 score in every model (medians +0.31 to +0.40, direction probability 0.9994–1.000), and the **block-2 vocabulary-exposure/teaching covariate** (`gamma_erbto`) is very strongly positive in three of four (bx-001/002/004) and suggestive in bx-003. The DEAP phonology composite is moderately positive for the not-taught expressive measure (bx-003, +0.213) but inconclusive elsewhere. Hearing status and linear age are inconclusive throughout (the one exception, bx-004 hearing status, is only suggestive). These covariates dominate the fit; the highlighted exposure `delta` is comparatively weak.

## Results — survival (both models)

**Gate:** both models **passed** cleanly (surv-009 max R-hat 1.0002 / min ESS 12497 / BFMI ≈ 1.00–1.03 / 0 divergences; surv-011 max R-hat 1.0007 / min ESS 14024 / 0 divergences); no flags.

### Headline treated-arm timing (`tau`) — association, not causal

`tau` is a log-hazard shift for the treated arm; exponentiated it is the hazard ratio for coming off the floor. Because both arms are treated by the final wave, it is a whole-trial prognostic association.

| Model    | Measure                | Hazard ratio | 89% credible range | Direction (P(tau>0)) | Evidence                       |
| -------- | ---------------------- | ------------ | ------------------ | -------------------- | ------------------------------ |
| surv-009 | Phonetic spelling (PS) | 0.84         | 0.46 to 1.56       | 0.329 (67% below 1)  | inconclusive (leans later)     |
| surv-011 | Nonword reading (NW)   | 1.35         | 0.75 to 2.44       | 0.798 (80% above 1)  | suggestive (earlier off-floor) |

Reading these: for phonetic spelling the treated arm has, if anything, a _lower_ per-interval chance of coming off the floor (HR 0.84, so slightly later), but the credible range spans HR 0.46–1.56 and only 67% of the posterior is below 1 — inconclusive, no confident call. For nonword reading the treated arm leans toward coming off the floor _earlier_ (HR 1.35, 80% of the posterior above 1), which is suggestive but well short of a confident effect — and in any case not randomised.

### Baseline-skill hazard slopes (`beta_*`) — none causal

These per-standard-deviation baseline covariates are the stronger and more consistent survival signals, and were absent from the earlier version of this note. All are prognostic (baseline-predictor) associations, latent-ability-confounded, never causal. HR > 1 means children higher on that baseline skill come off the floor earlier.

| Term                                    | surv-009 (PS)                                        | surv-011 (NW)                                                                      |
| --------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `beta_W0` (baseline word reading)       | HR 1.32 (log 0.275), P>0 0.918 — moderate            | HR 1.64 (1.18–2.31 span; log 0.498, 89% 0.168–0.839), P>0 0.9925 — **very strong** |
| `beta_L0` (baseline letter-sound level) | HR 1.41 (1.03–1.94; log 0.342), P>0 0.960 — moderate | HR 1.39 (log 0.333), P>0 0.958 — moderate                                          |
| `beta_A0` (baseline age/ability)        | HR 1.17 (log 0.160), P>0 0.815 — suggestive          | HR 0.91 (log −0.093), P>0 0.295 — inconclusive                                     |

The strongest single signal in either survival model is **baseline word reading** predicting earlier off-floor movement for nonword reading: HR 1.64, 89% hazard-ratio range roughly 1.18–2.31 (log-scale 0.498, 89% 0.168–0.839), direction probability 0.9925 — very strong. Baseline word reading is also moderately positive for phonetic spelling (HR 1.32, 0.918). **Baseline letter-sound level** (L0) is moderately positive in both (HR ≈ 1.40). Baseline age-and-ability is only suggestive for P and inconclusive for N. These are exactly the kind of prognostic baseline associations one would expect — children who already read words and letter-sounds better move off these floors sooner — and none of them is an effect of the intervention.

### Baseline off-floor probabilities (descriptive)

For an untreated child at average covariates, the fitted per-interval chance of coming off the floor, with 89% credible ranges:

| Interval | Phonetic spelling (PS) | Nonword reading (NW)   |
| -------- | ---------------------- | ---------------------- |
| t1 → t2  | 22.0% (11.9% to 36.4%) | 27.2% (15.6% to 42.5%) |
| t2 → t3  | 18.8% (8.0% to 38.4%)  | 29.4% (14.1% to 53.3%) |
| t3 → t4  | 15.7% (5.9% to 35.2%)  | 22.4% (8.6% to 47.3%)  |

These are descriptive baseline hazards, not effects; they set the scale against which the `tau` hazard ratios operate.

## What is causal

**Nothing in either family.** `block_exposure` is a non-randomised parallel-trends association — the block-2-active `delta` and every `gamma_*` covariate are adjusted associations, and the dominant signals (general ability, block-2 vocabulary exposure) are latent-ability-confounded. The `survival` treated `tau` term is a whole-trial prognostic association because both arms are treated by the final wave, and the `beta_*` baseline slopes (including the very strong baseline-word-reading signal in surv-011) are prognostic baseline-predictor associations. Both families are exploratory triangulation around the floored outcomes and are consistent with the [ITT](202607161800-findings-itt.md) floor-rule results, which were themselves inconclusive-to-suggestive. No term here is randomisation-licensed, so no term is a treatment effect.
