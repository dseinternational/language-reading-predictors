<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Findings 09 — the dose-response family

Reports every model in the `dose_response` family from the 2026-08-04/05 `reporting` refit. **5 models, all passing the convergence gate.** Reading conventions in note 00. Preliminary research data — all estimates provisional.

## What these models do

Assignment was randomised; **attendance was not**. These models ask how progress relates to the number of intervention sessions a child actually received, with partial pooling across the three periods and a test of whether the dose slope varies by period.

**Design.** Beta-Binomial ANCOVA on the period's post-score given its own baseline logit, with per-period intervention dose (`attend`) as the exposure and a child random intercept. The adjustment set is `{arm, own baseline, age}`: arm is the sole confounder blocking the path from assignment through dose to outcome; own baseline handles regression to the mean; age is a precision and maturation term.

**One deliberate omission worth understanding.** An earlier version adjusted for _cumulative prior dose_. That was dropped, because cumulative dose is a running sum of earlier sessions and conditioning on it reopens a latent-ability back door. Nothing downstream of, or aggregating, the focal dose is conditioned on. It remains available only as a flagged sensitivity.

**Nothing here is causal, and the reason is specific and known.** The study's causal diagram deliberately omits an arrow from ability to dose, but the original 2012 trial's own caveat was that "the children least able to learn tended to show the poorest attendance" — which is exactly that arrow. Dose is not randomised, so randomisation does not rescue it. The child random intercept absorbs _stable_ child differences but cannot remove ability-driven attendance. Every slope below is an **adjusted association**, and "more sessions cause more progress" is not a supported reading.

## Results

Dose slopes are on the logit scale per SD of sessions (mean ≈ 54 sessions, SD ≈ 31). The marginal is the items-scale effect across the fitted dose range.

| Model      | Outcome                        | Period-1 slope                         | Overall slope   | Items-scale marginal         | P(>0) |
| ---------- | ------------------------------ | -------------------------------------- | --------------- | ---------------------------- | ----: |
| `dose-077` | Word reading                   | **+0.179** (+0.077 to +0.293), P=0.998 | +0.136, P=0.903 | +1.16 items (+0.38 to +1.97) | 0.992 |
| `dose-177` | Word reading, ability-adjusted | **+0.199** (+0.093 to +0.316), P=0.999 | +0.151, P=0.914 | +1.28 items (+0.48 to +2.12) | 0.994 |
| `dose-277` | Word reading, pooled slope     | — (pooled +0.143, P=0.998)             | +0.143          | +1.28 items (+0.56 to +2.04) | 0.998 |
| `dose-083` | Letter sounds                  | **+0.226** (+0.087 to +0.380), P=0.995 | +0.153, P=0.853 | +0.80 items (+0.16 to +1.40) | 0.974 |
| `dose-084` | Phoneme blending               | +0.104 (−0.075 to +0.275), P=0.829     | +0.137, P=0.807 | +0.29 items (−0.01 to +0.57) | 0.940 |

**A positive dose association is present for the reading skills, and it is best resolved in period 1.** Word reading and letter sounds both show clear period-1 slopes (P = 0.998 and 0.995), with the items-scale marginals at about +1.2 and +0.8 items across the fitted dose range.

**The period pattern is the interesting part: for the two reading outcomes the slope decays.** For word reading, period 1 +0.179 → period 2 +0.136 → period 3 +0.096. For letter sounds, +0.226 → +0.170 → +0.053. By period 3 neither is resolved (P = 0.87 and 0.65). Phoneme blending does **not** follow that shape — it runs +0.104 → +0.288 → +0.036, peaking in period 2 — so the decay is a two-outcome pattern rather than a family-wide one. Two readings are compatible with this and the data do not separate them: the intervention's marginal value may genuinely diminish once the early alphabetic content has been taught, or the later periods may simply be noisier and more confounded (all children have been treated by then, so between-child dose differences increasingly reflect who chose to keep attending).

**Ability adjustment does not remove the association.** `dose-177` adds the baseline-skill cluster (letter sounds, expressive vocabulary, blending) to the word-reading model. If the dose signal were substantially ability-confounded, it should collapse. It does not — the period-1 slope moves from +0.179 to +0.199 and the marginal from +1.16 to +1.28 items. That is a genuine robustness result and the strongest thing this family can say. It does **not** establish causality: baseline skills are an imperfect proxy for the latent ability that plausibly drives attendance, and adjusting for measured skills cannot remove confounding by an unmeasured one.

**Period variation buys nothing.** `dose-277` fits a single pooled slope (+0.143, P = 0.998) with an items-scale marginal (+1.28) indistinguishable from the period-resolved model's. The between-period variation parameter `sigma_dose_between_period` is small with a wide interval in every model. The decay pattern above is a description of the point estimates rather than a resolved finding about heterogeneity.

## Caveats

- **Not causal, with a named and plausible confounder.** Attendance reflects family circumstances and, per the original trial, the child's capacity to learn.
- **Dose is a partial collider** in the causal diagram, which is why cumulative dose is excluded from the headline and why this family is a sensitivity view rather than an effect estimate.
- **Period 3 is uninformative** in every model.
- Four of the five required `target_accept` above the reporting preset to reach zero divergences (`dose-077/083/177` at 0.99, `dose-084` at 0.97); the values are now declared in their specs so the fits are registry-reproducible (run record note). `dose-277` sampled cleanly at the preset 0.95 and declares nothing.
- **Predictive calibration.** 50% bands cover about 70% of observations.
