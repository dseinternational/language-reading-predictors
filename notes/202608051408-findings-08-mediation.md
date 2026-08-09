<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted and edited by LLM-based AI tools (Claude Code/Opus 5 and Codex/GPT-5).

# Findings 08 — the mediation family (what carries the effect)

> [!NOTE]
> **Corrected MED-086/186 and MED-087/187 reporting fits completed (2026-08-09).** The four current fits adjust for baseline word reading, use exact matching fitted-child identities within each natural/interventional pair, pass the automatic computational gate and replace every pre-2026-08-08 result for these model IDs. Their indirect contrasts remain model-based decompositions rather than identified causal routes; the detailed correction and refit record are in `notes/202608081805-med-086-187-wr-baseline-correction.md` and `notes/202608091335-med-wr-baseline-reporting-refit.md`.

Reports every model in the `mediation` family (15) and the `mediation_multi` family (4). Most results come from the 2026-08-04/05 `reporting` refit; MED-086/186 and MED-087/187 are replaced by the corrected 2026-08-09 fits. **All 19 current model reports pass the automatic convergence gate.** Reading conventions in note 00. Preliminary research data — all estimates provisional.

## What these models do

The ITT family shows the intervention raises word reading. This family asks how fitted g-formula models allocate that contrast between direct and mediator-indexed components. If the programme teaches letter sounds and letter sounds are used to read words, the fitted letter-sound component should be positive; that pattern alone would not identify a causal route.

**Design.** A g-formula decomposition by counterfactual simulation. The model is fitted, then the posterior is used to simulate what would have happened under combinations of treatment assignment and mediator value, splitting the total effect into:

- **NDE** (natural direct effect) — the modelled contrast that would remain if the mediator did not respond to treatment;
- **NIE** (natural indirect effect) — the modelled contrast assigned to the mediator response;
- **total** = NDE + NIE.

Several models also report the **interventional-effects** version (IDE/IIE), which targets a slightly different and more robustly-defined estimand under interference between mediators.

**What this family can and cannot claim.** For the single-window models using random assignment in the trial window, the modelled _total_ contrast is randomisation-anchored within its fitted analysis population. The **split** between direct and indirect is not: decomposing a total requires assuming there is no unmeasured confounding of the **mediator–outcome** relationship, and general ability plausibly confounds letter sounds and word reading together. Randomisation balances the arms; it does not make the mediator exogenous. MED-092 is a separate exception even at the total: its all-period model uses the time-varying on-programme exposure and therefore depends on conditional ignorability, while only its period-1 restricted readout is randomisation-anchored. The NIE/IIE quantities below are **model-based decompositions under untestable assumptions**, not randomised quantities.

All quantities are on the word-reading items scale unless marked otherwise.

## The headline: fitted decompositions consistently assign a positive component to letter sounds

| Model     | Mediator                                    | Total | NDE (direct)   | NIE (indirect)             |  P(NIE>0) |
| --------- | ------------------------------------------- | ----- | -------------- | -------------------------- | --------: |
| `med-059` | Letter sounds                               | +1.95 | +0.16 (P=0.55) | **+1.69** (+0.58 to +3.22) | **0.997** |
| `med-078` | Letter sounds (interventional)              | +1.95 | +0.15 (P=0.55) | **+1.70** (+0.58 to +3.24) | **0.996** |
| `med-076` | Letter sounds at t2 (longitudinal ordering) | +1.39 | −1.68 (P=0.18) | **+2.92** (+1.12 to +5.32) | **0.997** |
| `med-092` | Letter sounds (period-stacked)              | +3.03 | +2.24 (P=0.97) | +0.75 (+0.29 to +1.39)     |     0.999 |
| `med-062` | Code-based reading route (composite)        | +1.62 | +0.61 (P=0.68) | +0.92 (+0.06 to +2.17)     |     0.959 |

**In the cross-sectional models, most of the fitted total is assigned to the letter-sound indirect component.** `med-059` puts the direct contrast at +0.16 items with a direction probability of 0.55, while the letter-sound-indexed component is +1.69 with strong directional evidence. The interventional-effects functional agrees to two decimal places because it uses the same fitted model; this is an implementation check, not independent evidence. The longitudinal-ordering model, which requires the mediator to be measured before the outcome, also assigns a larger positive component to letter sounds. None of these ratios identifies how much of the causal effect travels through a pathway.

`med-092` is the exception worth noting: stacking every period, the direct effect is +2.24 and the indirect +0.75. That model uses far more rows and a different treatment definition ("on the programme" rather than assigned), so it is not directly comparable, but it is a reminder that the "almost all indirect" split is strongest in the single-window models.

## Which routes do _not_ carry the effect

Several plausible mediator-indexed components remain small or uncertain in the fitted decompositions.

| Model     | Mediator                                 | NIE                    | P(NIE>0) | Reading               |
| --------- | ---------------------------------------- | ---------------------- | -------: | --------------------- |
| `med-074` | Nonword decoding                         | +0.02 (−0.41 to +0.50) |    0.555 | nothing               |
| `med-079` | Receptive grammar (**negative control**) | +0.08 (−0.19 to +0.63) |    0.713 | nothing — as designed |
| `med-080` | Taught receptive vocabulary              | +0.21 (−0.30 to +1.23) |    0.754 | weak at best          |
| `med-068` | Taught expressive vocabulary             | +0.30 (−0.23 to +1.32) |    0.814 | weak at best          |

**The negative-control pattern is reassuring but not decisive.** `med-079` indexes the decomposition by receptive grammar — a skill the intervention has no proposed mechanism to use for word reading — and assigns little to that component (NIE +0.08, inconclusive), with the fitted direct component absorbing most of the total. This reduces concern that the machinery produces a large indirect estimate for every mediator, but it neither validates the letter-sound decomposition nor repairs its identification assumptions.

**The nonword-decoding component is unresolved, which is initially surprising.** The mechanism family (note 07) found the letter-sounds/nonword-reading coupling to be the strongest adjusted association in the suite. Yet the single-mediator decomposition assigns +0.02 to the nonword-indexed component and +2.83 to the fitted direct component. The sequential model `med-060` shows the same descriptive pattern: NIE_L +1.66 (P = 0.989) and **NIE_N −0.07 (P = 0.31)**. These fits are compatible with little additional nonword-indexed component after letter sounds, but do not show that the causal intervention effect bypasses nonword decoding.

## The two-mediator models

These fit two mediators at once, so the routes compete for the same effect rather than being estimated in separate models.

| Model     | Mediators                                     | NIE via letter sounds | NIE via the other |
| --------- | --------------------------------------------- | --------------------- | ----------------- |
| `med-064` | Letter sounds + expressive vocabulary         | **+1.88** (P=0.996)   | +0.03 (P=0.58)    |
| `med-066` | Letter sounds + phoneme blending              | **+1.61** (P=0.995)   | −0.03 (P=0.42)    |
| `med-075` | Letter sounds + phoneme blending (sequential) | **+1.62** (P=0.995)   | −0.04 (P=0.42)    |
| `med-060` | Letter sounds + nonword decoding (sequential) | **+1.66** (P=0.989)   | −0.07 (P=0.31)    |

**The fitted letter-sound route survives every tested competitor.** Put head to head with expressive vocabulary, phoneme blending or nonword decoding, the model assigns +1.6 to +1.9 items to the letter-sound indirect component with very strong directional evidence, while the alternative component is unresolved. This is the suite's most consistent model-based pathway pattern, but it remains conditional on the mediation assumptions rather than an identified mechanism.

## Other outcomes — corrected refits require a more qualified reading

| Models                | Outcome and scale               | Indirect contrast (89% interval)       | Direct contrast (89% interval)         | Total contrast (89% interval)          |
| --------------------- | ------------------------------- | -------------------------------------- | -------------------------------------- | -------------------------------------- |
| `med-086` / `med-186` | Nonword reading, off-floor risk | +0.079 (+0.028 to +0.148), P>0 = 0.996 | +0.005 (−0.078 to +0.090), P>0 = 0.538 | +0.087 (−0.011 to +0.188), P>0 = 0.924 |
| `med-087` / `med-187` | Phoneme blending, items         | +0.292 (−0.098 to +0.761), P>0 = 0.888 | +0.441 (−0.471 to +1.366), P>0 = 0.778 | +0.748 (−0.056 to +1.560), P>0 = 0.932 |

The natural and interventional companions are numerically identical in this implementation because they use the same fitted probabilistic model and g-formula functional; their fitted-child identities also match exactly within each pair. This is an implementation and comparability check, not evidence that the indirect contrasts are identified.

**Nonword reading.** The corrected primary MED-086 fit puts the letter-sound indirect contrast at an increase of 7.9 percentage points in the probability of reading at least one nonword. Its 89% interval excludes zero, but the total contrast's interval includes zero. The separately fitted t3 off-floor sensitivity likewise has a positive indirect contrast of +0.103 (+0.037 to +0.186; P>0 = 0.996), alongside a direct contrast of −0.085 (−0.195 to +0.024; P>0 = 0.106) and a total of +0.022 (−0.106 to +0.148; P>0 = 0.608). These are all off-floor risk differences; no comparison is made with any superseded graded-item t3 quantity.

The confounding sweep weakens the apparent certainty. MED-086's primary NIE 89% interval first includes zero at an outcome-leg shift of $\delta^*=0.636$, 45% of the fitted effective mediator coefficient; at that point the median remains +0.048 and the interval is −0.007 to +0.115. The median changes sign only between $\delta=1.379$ and 1.485. The session-dose point calibration ($\delta=0.247$) is below the interval tipping point and maps to a +0.068 NIE (+0.017 to +0.136), but its deliberately wide 89% endpoint scenario (0.000 to 2.656) extends beyond both thresholds. Session-strength confounding could therefore plausibly remove credible evidence for a positive indirect contrast and, at the upper end of the scenario, reverse its median; $\delta^*=0.636$ itself does not set the median to zero. Power-scaling sensitivity also flags 19 of 24 scanned MED-086 parameters, including `b_M`, `b_GM` and the newly fitted `b_conf_W`; this is a substantial prior/likelihood caution even though the sampler converged cleanly.

**Phoneme blending.** The corrected MED-087 primary indirect contrast is +0.292 items, but its 89% interval already includes zero. The t3 sensitivity agrees in that limited sense: NIE +0.263 items (−0.144 to +0.747; P>0 = 0.854), NDE +0.102 (−0.846 to +1.078; P>0 = 0.569) and total +0.385 (−0.447 to +1.222; P>0 = 0.772). The session-dose point calibration maps to +0.266 items (−0.127 to +0.732), but there is no credibly non-zero indirect contrast for dose confounding to explain away. Power-scaling sensitivity flags 9 of 22 scanned parameters, including `b_GM` and the own-baseline term `b_W`; it does not flag `b_M` or the new `b_conf_W`.

## Could the causality run the other way?

Two models reverse the ordering — does word reading carry the effect to letter sounds, rather than the reverse?

| Model     | Route                              | NIE                    | P(NIE>0) |
| --------- | ---------------------------------- | ---------------------- | -------: |
| `med-176` | Word reading at t2 → letter sounds | +0.45 (+0.04 to +1.11) |    0.963 |
| `med-276` | Same, t3 outcome (less ceilinged)  | +0.50 (+0.05 to +1.23) |    0.966 |

**The reverse route is not empty.** It is roughly a quarter the size of the forward route (+0.45 against +1.69) and carries moderate rather than very strong evidence, but it does not vanish. This is an honest limit on the mediation story: with these data the forward direction is clearly the larger, but reverse mediation cannot be ruled out, and both would be consistent with a shared underlying process driving letter sounds and word reading together.

## Caveats

- **The decomposition is not randomised.** Only the total effect inherits the ITT warrant. The direct/indirect split assumes no unmeasured mediator–outcome confounding, and latent general ability is a plausible violation.
- **`proportion_mediated` is not quoted.** It is a ratio of two uncertain quantities whose denominator's interval spans zero for most of these models, so it is unstable and easily over-read. The NDE and NIE with their own intervals are the honest summary.
- **PSIS-LOO is not computed** for this family — the g-formula fits are simulation-based decompositions rather than predictive models, so the usual model-comparison route does not apply.
- **The two-mediator models are the slow ones** (`med-064/066/075` take 36–41 minutes each at reporting tier) because the sensitivity sweep runs 42 full counterfactual decompositions over all 36,000 draws.
- **Reverse mediation is not excluded** — see above.
- **Predictive calibration.** 50% bands cover about 71% of observations (79% for the two-mediator models).
- **Clean computation is not identification.** The corrected MED-086/186 and MED-087/187 primary fits and the two natural-model t3 subfits all have zero divergences and pass the automatic R-hat, effective-sample-size and BFMI gates, but those checks cannot remove latent-general-ability confounding, treatment-induced session confounding or same-wave mediator/outcome ambiguity.

## Where this leads

Taken with note 07, the fitted decompositions are consistent with much of the word-reading contrast being assigned to letter-sound learning: no vocabulary or blending route competes with that model-based component, a grammar negative control carries little, and the reverse ordering, while smaller, is not excluded. The corrected code-route models add narrower, qualified evidence: the off-floor nonword indirect contrast is positive but sensitive to model, prior and unmeasured-confounding assumptions, while its total interval includes zero; the graded blending indirect interval already includes zero. None of these results establishes that a reading or code-route outcome is causally carried through letter sounds.
