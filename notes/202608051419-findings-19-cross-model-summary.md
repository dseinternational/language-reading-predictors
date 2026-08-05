<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Findings 19 — cross-model summary

Synthesis across all 194 models of the 2026-08-04/05 `reporting` refit. Read note 00 first for conventions; each claim below points to the family note that supports it. Preliminary research data — all estimates provisional.

## The suite in one paragraph

A reading and phonics intervention was evaluated in a waitlist-crossover randomised trial with about 54 children with Down syndrome, measured at four timepoints. The intervention produced clear gains in **letter-sound knowledge** and **word reading**, smaller gains in the specific vocabulary it teaches and in phoneme blending, and **no detectable change in broad standardised vocabulary**. The word-reading gain appears to run almost entirely **through** letter-sound knowledge. Baseline characteristics predict where a child _is_ almost perfectly and how much they _gain_ barely at all — the intervention is one of the few things in this study that moves a gain measure.

## The headline effects, and how well they replicate

Five designs estimate the intervention's effect on the same outcomes from different rows and different identification arguments. Items scale, median.

| Outcome                 | ITT (01) | Gain factors (03) | DiD (05) | Levels (04) | Aligned (06) |
| ----------------------- | -------: | ----------------: | -------: | ----------: | -----------: |
| **Letter sounds**       | **+3.5** |              +3.3 |     +3.5 |        +2.5 |         +2.2 |
| **Word reading**        | **+2.4** |              +2.6 |   +2.2 ‡ |      +1.5 ‡ |         +2.1 |
| Taught expressive vocab |     +1.5 |              +1.2 |     +1.5 |        +0.3 |            — |
| Taught receptive vocab  |     +1.4 |              +1.1 |     +1.2 |        +0.4 |            — |
| Phoneme blending        |   +1.0 † |              +0.8 |   +0.9 ‡ |        +0.4 |         +0.3 |
| Receptive vocabulary    |     +0.2 |              −1.5 |     −0.1 |        −3.8 |         +2.7 |
| Expressive vocabulary   |     +0.2 |              +1.1 |     +0.8 |        −2.4 |         −3.0 |

† response-link sensitive — see below. ‡ **withheld from release** under the robustness gate (notes 04, 05) — shown because this is a technical record, not a published figure.

**The two reading outcomes replicate across every design.** Letter sounds land between +2.2 and +3.5 items and word reading between +1.5 and +2.6. Across the three randomisation-anchored designs (ITT, gain factors, DiD) the spread is 0.24 items for letter sounds and 0.37 for word reading — well inside any one design's interval. The formal triangulation check confirms it: for W, L, TR, TE, E and F all three designs converge, agree in direction and have overlapping intervals.

**The vocabulary rows are all over the place, and that is informative rather than alarming.** RV ranges from −3.8 to +2.7 across designs. The triangulation check flags **R as the one inconsistent outcome** (directions disagree, though intervals still overlap). This is what an inconclusive effect looks like when estimated five ways: the designs are sampling noise around zero, and the ones without an own-baseline term (levels, aligned) swing furthest because they cannot difference out baseline imbalance. The honest conclusion is that broad standardised vocabulary did not move detectably, not that any one design found something.

## What carries the effect

The mechanism picture (notes 07, 08) is unusually clean for observational work:

- The word-reading gain is **almost entirely indirect** through letter-sound knowledge: NIE +1.69 items (P = 0.997) against a direct effect of +0.16 (P = 0.55).
- The letter-sound route **beats every competitor** put head-to-head with it — expressive vocabulary, phoneme blending and nonword decoding each carry essentially nothing when fitted alongside it.
- A **negative-control mediator** (receptive grammar) correctly carries nothing, which is meaningful evidence the decomposition is not manufacturing indirect effects.
- Letter-sound knowledge is associated with **pure decoding about four times more strongly than with word reading** (identified contrast +0.81, P = 0.9999), the signature of a skill actually being used to decode.

**Three limits travel with that story.** The decoding-specificity contrast holds for _conditional change_, and reverses in a levels view of the same children. The negative-control _outcomes_ do not come out clean — letter sounds predict every oral-language measure at P ≈ 0.99, attenuated but not null, the expected fingerprint of general-ability confounding. And reverse mediation (word reading carrying the effect to letter sounds) is smaller but not excluded.

## The cross-cutting finding: levels are predictable, gains are not

Four families reach this independently, which makes it the most robust non-treatment result in the suite:

| Family           | Evidence                                                                                                                          |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Horseshoe (13)   | Level models select predictors at P ≥ 0.99; **gain models select nothing** (max 0.59)                                             |
| Adjusted (12)    | Only age is decisively associated with gain; non-verbal ability adds almost nothing                                               |
| Growth (15)      | Ability predicts baseline level for the language measures (P ≥ 0.999) but not growth rate (never above suggestive except grammar) |
| Measurement (14) | **93–95%** of latent skill variance is stable trait, not wave-specific state                                                      |

Together these say that this cohort's skill ordering is highly stable, and that who gains is close to unpredictable from baseline. Three consequences follow. The intervention effect is not competing against a strong prognostic signal. Modest item-scale effects should be read against a system that barely moves. And the many level-based associations elsewhere in the suite must not be read as telling us who will progress.

**The one baseline characteristic that does predict gain is age, negatively** — in the trial cohort (−2.9 words per SD), in the historical cohort (−8.4 words per SD), under shrinkage selection, and in the gain-factor and LCSM families. It was tested directly and is not a difficulty-ladder or likelihood artefact. It cannot separate developmental timing from trajectory selection: being older at the same score _means_ having progressed more slowly, and no cross-sectional adjustment distinguishes those.

## Four results that need stating carefully

**1. Phoneme blending is response-link sensitive.** The instrument is ten three-alternative forced-choice items, so chance is about 3.3. Under the ordinary logit the effect is +1.0 items (strong evidence); under a mechanically-motivated guessing-floor link it is +0.5 (suggestive). Predictive comparison cannot separate the links. The pair is the result; the logit figure alone overstates it. (Note 01.)

**2. Teaching-specific vocabulary transfer is weakly supported at best.** Reading the single-outcome table, taught receptive vocabulary gains +1.4 items against not-taught +0.6, inviting a teaching-specific conclusion. The model that estimates the difference properly gives **P(taught > not-taught) = 0.47 for receptive** — inconclusive — and 0.76 for expressive. Comparing two credible intervals by eye is not a test of their difference. (Note 02.)

**3. Direction and size are different claims, and they diverge.** Phoneme blending has strong evidence of a positive effect and only a 49% chance of clearing the 1-item threshold. Several taught-vocabulary results are the same shape. "Probably positive" is not "probably big enough to matter".

**4. Dose associations are not effects.** Attendance was not randomised, and the original trial's own caveat — that the children least able to learn attended least — is exactly the confounding path. The association survives ability adjustment, which is a genuine robustness result, but adjusting for measured skills cannot remove confounding by unmeasured ability. (Note 09.)

## Computational status

**All 194 models pass the convergence gate, with zero divergent transitions across the entire suite.** Under the study's divergence policy every divergent fit fails closed with no qualification pathway, so this is the first run in which the whole suite clears computation. Thirteen fits initially failed and were repaired by geometry changes rather than waivers — the account is in `notes/202608050649-reporting-refit-predictive-checks.md`.

**Clearing computation is not the same as clearing release.** A robustness gate adopted on 2026-08-05, after these fits were made, additionally classifies each randomised-effect fit on the power-scaling sensitivity of the coefficient its headline rests on. It covers the four families with a randomisation-anchored estimand — ITT (τ), the arm-by-wave DiD (`tau_t2`, or a dose slope), gain factors (`beta_trt`) and level factors (the t2 element of `b_grp_time`) — **74 fits, of which 10 are withheld**.

The distribution is the informative part, because it is not uniform:

| Family        | gated | withheld |                                                             |
| ------------- | ----: | -------: | ----------------------------------------------------------- |
| ITT           |    28 |        2 | the floor-rule outcomes, missing their treatment-prior grid |
| DiD           |    14 |    **4** | prior-dominant `tau_t2` / dose slope                        |
| Gain factors  |    13 |        2 | prior-dominant `beta_trt`, both floor-rule outcomes         |
| Level factors |    11 |        2 | prior-dominant t2 contrast                                  |

**No ITT fit is prior-dominant; all eight of the others' withholds are.** The two ITT withholds fail on a missing grid instead. So the family that carries the study's headline estimates is also the one whose estimates lean least on their priors, while the arm-by-wave parameterisation — estimating overlapping effects from the same children — leans most. That is a statement about how much each design asks of n ≈ 54, not about the intervention.

Three cells of the effects table above are affected: DiD word reading and phoneme blending, and the levels word-reading contrast. **Letter sounds is withheld in no design**, so the suite's strongest result is unaffected. A further eleven ITT fits, including letter sounds and phoneme blending, release with an attenuation caveat: the conservative zero-centred prior is pulling those estimates toward zero, so their direction is more reliable than their size. Details in notes 01, 03, 04 and 05.

Every model emits a prior-predictive check (previously 24 did not), and **all 189 with a comparable summary schema cover the observed range**. Posterior-predictive coverage reproduces the 2026-07-26 baseline to within 0.009 across every outcome and family, so the calibration picture is unchanged on independently re-sampled fits. The 50% bands overcover throughout (0.58–0.85 by family); that is substantially mechanical — discrete counts, small denominators, and in-sample checks conditioning on fitted child effects in the repeated-measures families — rather than a likelihood defect, as the Conway–Maxwell-binomial probe established.

## What this suite cannot tell you

- **Whether the effects last.** The post-crossover comparisons are not randomised; the observation that the arm gap had not fully closed by t3 is descriptive.
- **Whether any skill causes any other.** Latent general ability is unblockable, and the measurement models show the domains correlating at 0.82–0.95 in the historical cohort — adjusting for one removes much of the others.
- **The effect for every randomised child.** All causal estimates are available-case, typically 53–54 of 57.
- **Whether more sessions help.** Dose is not randomised.
- **Whether gains are equal-interval.** Items are not equal units of learning and are not comparable across measures.

## Reading order

For a first pass: note 00 (conventions) → note 01 (the effects) → note 08 (what carries them) → this note. For the "who progresses" question: notes 12, 13 and 15 together, then note 14 for why they agree.
