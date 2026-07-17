# Findings — dose_response family (outcome vs amount of intervention)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8). Numbers from the `reporting`-config refit of 2026-07-16 (see [process note](202607161130-full-statistical-refit.md)); reviewed and extended on 2026-07-17 to cover all models in the family. Preliminary.

## What these models ask

"Do children who did **more** intervention (more sessions) score higher?" Each model relates a bounded outcome count to the standardised number of teaching sessions a child received, and — in the period-resolved fits — lets that relationship differ across the study's transitions ("periods"). **Session dose is not randomised**: how much a child attended reflects ability, family circumstances, attendance, and availability. Worse, dose is a **partial collider** in the study's causal diagram — a variable that sits downstream of two things we care about, so conditioning on it (or reading a slope through it) can open a spurious back-door path between them. In plain terms: abler or better-attending children both do more sessions _and_ score higher, and a raw dose slope cannot separate those two stories. So every number here is strictly an **observational (adjusted) association** and a _sensitivity/triangulation view_, never a treatment effect. See the [ITT note](202607161800-findings-itt.md) and [DiD note](202607161800-findings-did.md) for the randomised, genuinely causal contrasts.

The family has five models — three view **word reading (W)**, one views **letter sounds (L)**, one views **phoneme blending (B)**:

- `dose-277` — W, a single **pooled** dose slope held constant across periods (the simple headline, and the no-period-variation comparator for `dose-077`).
- `dose-077` — W, **period-resolved** (a separate slope for each transition).
- `dose-177` — W, period-resolved and additionally **adjusted for a baseline-skill / general-ability cluster** (sensitivity check on `dose-077`).
- `dose-083` — L (letter sounds), period-resolved.
- `dose-084` — B (phoneme blending), period-resolved.

A note on reading the numbers. A **95% credible interval** is the range within which the parameter lies with 95% probability _given the data and priors_ — a direct probability statement about the quantity, unlike a frequentist confidence interval. **P(effect > 0)** ("prob_pos" / "p_pos" in the CSVs) is the posterior probability the true slope is positive. The **evidence label** (per the project ladder, #179) grades that direction probability — _not_ the size of the effect: inconclusive < 0.75 ≤ suggestive < 0.91 ≤ moderate < 0.97 ≤ strong < 0.99 ≤ very strong. A "very strong association" therefore means we are very sure of the _sign_, and says nothing about whether the effect is large or causal. No region-of-practical-equivalence (ROPE) verdict is computed for this family, so there is no separate "big enough to matter" flag — read the items-scale magnitude directly.

Throughout, "1-SD increase in sessions" means **about 31 more sessions** (the dose SD is ≈30.7 sessions; the mean child had ≈54 sessions). So the items-scale slopes below are "extra items per ≈31 extra sessions".

## Convergence gate

`dose-277` (pooled comparator) **passed all four checks** cleanly (0 divergences, max R̂ 1.002, min ESS 3574, BFMI 0.88–0.94). The four period-resolved fits each **fail only the divergence check** — R̂, ESS and BFMI all pass — and are **usable with a caveat**. Every divergence rate is far under the project's 1% guideline: `dose-077` 17 divergences (0.047%, min ESS 4088, R̂ 1.001, BFMI 0.86–0.96), `dose-177` 13 (0.036%, min ESS 3792, R̂ 1.001, BFMI 0.85–0.93), `dose-083` 4 (0.011%, min ESS 4251, R̂ 1.001, BFMI 0.51–0.57), `dose-084` 2 (0.006%, min ESS 6411, R̂ 1.001, BFMI 0.68–0.73). All were sampled at 6 chains × 6000 draws (36,000 total). One caveat on the "healthy diagnostics" summary: `dose-083`'s BFMI (≈0.51–0.57) is markedly lower than the rest of the family (≈0.7–0.96) — still comfortably above the 0.3 gate, but the least energetic sampler of the five.

Because a gate-flagged model is excluded from cross-model LOO comparison (the #340 discipline), the dose-response LOO comparison (`dose-077` vs its baseline) was **correctly skipped** this run — see `output/statistical_models/comparison/`.

## Results — all five models

**Items-scale marginal association** (extra items on the outcome per ≈31 extra sessions; averaged over the period-1, all-untreated-baseline transition). All are associations — none is causal.

| Model      | Outcome                  | Best estimate (items) | 95% credible range | P(>0) | Evidence    | Causal?     |
| ---------- | ------------------------ | --------------------- | ------------------ | ----- | ----------- | ----------- |
| `dose-277` | W (word reading), pooled | **+1.28**             | +0.41 to +2.22     | 0.998 | very strong | association |
| `dose-077` | W, period-resolved       | **+1.16**             | +0.21 to +2.16     | 0.992 | very strong | association |
| `dose-177` | W, ability-adjusted      | **+1.28**             | +0.28 to +2.33     | 0.994 | very strong | association |
| `dose-083` | L (letter sounds)        | **+0.80**             | −0.01 to +1.54     | 0.973 | strong      | association |
| `dose-084` | B (phoneme blending)     | **+0.29**             | −0.07 to +0.63     | 0.942 | moderate    | association |

The cleanest single number is the **pooled** slope (`dose-277`): a 1-SD (≈31-session) increase went with **+1.3 more words read**, 95% credibly +0.4 to +2.2, 99.8% positive — a very strong _association_. On the logit (model) scale this is `dose_pooled` = 0.143 (95% 0.046 to 0.243). Its period-resolved twin `dose-077` agrees (+1.16 items, very strong), and adjusting for general ability (`dose-177`) leaves it essentially unchanged (+1.28 items) — so the word-reading dose link is not merely a baseline-ability artefact, though it remains confounded. Letter sounds (`dose-083`) is intermediate (+0.80 items, strong), and phoneme blending (`dose-084`) is the weakest signal in the family (+0.29 items, moderate — its interval just crosses zero).

### Period-resolved logit slopes (the "front-loading" pattern)

The period-resolved models let the dose slope differ by transition; period 1 is the randomised, all-untreated-baseline transition. Slopes below are on the logit scale (`dose_period1/2/3` in `dose_slope_summary.csv`), with P(>0) in brackets.

| Model      | Outcome        | Overall (p)               | Period 1 (p)      | Period 2 (p)      | Period 3 (p)  | σ between periods (95%) |
| ---------- | -------------- | ------------------------- | ----------------- | ----------------- | ------------- | ----------------------- |
| `dose-077` | W              | 0.137 (0.904, suggestive) | **0.177 (0.998)** | 0.135 (0.937)     | 0.097 (0.867) | 0.107 (0.005–0.572)     |
| `dose-177` | W, ability-adj | 0.149 (0.909, suggestive) | **0.199 (0.999)** | 0.150 (0.948)     | 0.100 (0.872) | 0.117 (0.006–0.592)     |
| `dose-083` | L              | 0.148 (0.846, suggestive) | **0.227 (0.995)** | 0.169 (0.899)     | 0.054 (0.648) | 0.180 (0.010–0.718)     |
| `dose-084` | B              | 0.138 (0.811, suggestive) | 0.104 (0.828)     | **0.291 (0.962)** | 0.037 (0.597) | 0.229 (0.012–0.799)     |

For **word reading (W, both `dose-077` and `dose-177`)** and **letter sounds (L, `dose-083`)** the dose association is **front-loaded**: strongest in period 1 (logit ≈0.18–0.23, direction near-certain at P(>0) ≈0.995–0.999) and attenuating monotonically to a weak period-3 slope (logit ≈0.05–0.10, direction only suggestive-to-inconclusive). **Phoneme blending (B, `dose-084`)** breaks the pattern — its slope **peaks in period 2** (logit 0.291, P(>0) 0.962) with a near-zero period-1 and period-3 slope. The _overall_ per-period-model slopes are only "suggestive" (P(>0) 0.81–0.91) because letting the slope wander across periods widens the pooled uncertainty; the pooled `dose-277` model, which forbids that wandering, sharpens the same average slope back to "very strong".

**How real is the between-period variation?** Weak. The between-period slope SD (`sigma_dose_between_period`) is 0.107 (W), 0.117 (ability-adj W), 0.180 (L) and 0.229 (B), but in _every_ model its lower 95% bound sits essentially at zero (0.005–0.012). So the data are consistent with the slope being genuinely period-varying **or** flat — which is exactly why the pooled no-variation comparator (`dose-277`) fits well and lands on the period-average (≈0.143 logit). Read the front-loading as a suggestive pattern, not an established fact. Note also there is **no nonlinear-in-dose knee/threshold** estimated anywhere here: each period's slope is linear in standardised sessions, and no dose-by-covariate interaction is reported.

## The one-paragraph story

More sessions go with higher scores across all three outcomes, and for word reading the pooled association is clearly and strongly positive (+1.3 words per ≈31 extra sessions, 99.8% positive). The link is strongest for word reading, intermediate for letter sounds, weakest for phoneme blending, and — for word reading and letter sounds — appears concentrated in the early (period-1) transition, though the evidence for genuine period-to-period variation is thin. But this is the one family where the direction of causation is genuinely ambiguous: abler and better-attending children both do more sessions _and_ score higher. Adjusting word reading for general ability barely moves the slope, which is reassuring but not decisive. So everything here is reported as an association and used only to triangulate alongside the randomised ITT and DiD estimates — never to claim a dose effect.

## What is causal

**Nothing.** Session dose was not assigned; every slope is observational and dose is partly a collider. Even the ability-adjusted word-reading model (`dose-177`) remains confounded by unmodelled ability and by attendance. Read the whole family as "children who did more also scored higher, most markedly early on", **not** "more sessions cause faster growth".
