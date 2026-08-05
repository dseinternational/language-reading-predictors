<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Findings 04 — the level-factor family (per-wave levels)

Reports every model in the `level_factors` family from the 2026-08-04/05 `reporting` refit. **11 models, all passing the convergence gate.** Reading conventions in note 00. Preliminary research data — all estimates provisional.

## What these models do

The companion _levels_ view to the gain-factor family (note 03). Where the gain models ask "given where a child started this period, where did they end?", these ask "what was the **score at each timepoint**?" — no own-baseline term at all.

**Design.** Each wave's score is modelled with a Beta-Binomial likelihood on the randomised group entered **per timepoint**, non-verbal ability, an optional group × ability term, and a non-centred child random intercept for the repeated measures. Group and ability therefore appear as per-timepoint coefficient vectors rather than single numbers.

**What is causal, and it is only one number.** Only the **t2 group contrast** is randomised — that is the one timepoint after the intervention started and before the waitlist arm crossed over. Every later timepoint is **post-crossover**, so those contrasts compare groups that have both been treated and are associations, not effects. Ability and group × ability terms are latent-ability-confounded associations at every timepoint.

**Why no skill adjusters.** Unlike the gain family, these models take the exogenous confounders (hearing, speech, phonological memory) but deliberately carry **no other measure's contemporaneous level**. Conditioning a levels model on another skill measured at the same time would condition on a post-treatment mediator of the very group × time effect being estimated. That restriction was re-derived against the revised causal diagram.

**Read this family as the weaker design of the pair.** Dropping the own-baseline term costs a great deal of precision — a levels model has to explain the whole between-child spread rather than the change — and the intervals below show it.

## Results — the t2 randomised contrast

The only causal quantity in each model, on the items scale, median with 89% range.

| Model    | Outcome                    | t2 contrast (89%)             | P(>0) | Evidence     | ROPE |
| -------- | -------------------------- | ----------------------------- | ----: | ------------ | ---: |
| `lf-004` | Letter sounds (LS)         | **+2.5** items (+0.1 to +4.8) | 0.954 | moderate     | 0.38 |
| `lf-001` | Word reading (WR)          | +1.5 items (−1.2 to +4.0)     | 0.813 | suggestive   | 0.32 |
| `lf-007` | Basic concepts (LF)        | +0.5 items (−0.6 to +1.6)     | 0.772 | suggestive   | 0.75 |
| `lf-006` | Phoneme blending (PA)      | +0.4 items (−0.4 to +1.2)     | 0.797 | suggestive   | 0.87 |
| `lf-009` | Taught receptive (TR)      | +0.4 items (−1.0 to +1.7)     | 0.662 | inconclusive | 0.72 |
| `lf-010` | Taught expressive (TE)     | +0.3 items (−1.0 to +1.7)     | 0.650 | inconclusive | 0.72 |
| `lf-008` | Receptive grammar (RG)     | +0.2 items (−1.4 to +1.8)     | 0.584 | inconclusive | 0.67 |
| `lf-011` | Nonword reading (NW)       | +0.0 items (−0.1 to +0.1)     | 0.529 | inconclusive | 0.84 |
| `lf-005` | Phonetic spelling (PS)     | −0.0 items (−0.1 to +0.1)     | 0.434 | inconclusive | 0.94 |
| `lf-003` | Expressive vocabulary (EV) | −2.4 items (−6.5 to +1.7)     | 0.181 | inconclusive | 0.40 |
| `lf-002` | Receptive vocabulary (RV)  | −3.8 items (−8.2 to +0.6)     | 0.084 | inconclusive | 0.24 |

**The direction survives for the two reading skills; the precision does not.** Letter sounds is the only outcome retaining even moderate evidence (+2.5 items, P = 0.95), and word reading drops to suggestive (+1.5, P = 0.81) — against +3.5 and +2.4 with strong-to-very-strong evidence in both the ITT and gain-factor families. The point estimates are compatible; the intervals are roughly twice as wide.

**Two negative point estimates need care.** RV (−3.8, P(>0) = 0.08) and EV (−2.4, P(>0) = 0.18) lean negative. Applying the evidence ladder honestly, the favoured direction for RV is _negative_ with a probability of 0.92 — which on a naive reading would be "moderate evidence the intervention harmed receptive vocabulary". **That reading is not warranted**, for a specific structural reason: these are levels without a baseline term, so any pre-existing arm imbalance in vocabulary at t1 propagates straight into the t2 contrast rather than being differenced out. The gain-factor and ITT families, which _do_ condition on the child's own baseline, both put RV and EV flat and inconclusive. The levels result is best read as an artefact of the weaker design, and it is a good illustration of why this family is the companion view rather than the model of record.

## Post-crossover timepoints

Each model also reports t3 and t4 group contrasts. These are **not** intervention effects — by then both arms have received the intervention, differing only in timing — and they are reported as associations. They are useful for describing the trajectory of the two cohorts, not for estimating benefit.

## Caveats

- **One causal number per model.** `b_grp_time[1]` only. Everything else, including all later timepoints, is an adjusted association.
- **The t2 estimand is under methodological review** (population-standardised average versus conditional-at-a-profile, and the treatment of the currently time-invariant group × ability term). Read the t2 contrast as a well-defined marginal effect on the available-case t2 rows, and expect the precise standardisation to be refined.
- **No own-baseline term** is the design choice that costs the precision and admits baseline-imbalance leakage. This is the family's defining limitation.
- **Available-case**, about 53–54 children depending on outcome, under ignorable missingness given the modelled covariates.
- **Predictive calibration.** 50% bands cover about 77% of observations — higher overcoverage than the cross-sectional families, as expected for a repeated-measures model whose in-sample check conditions on fitted child effects (note 00).

## Where this leads

Taken with note 03, the pair says the same thing from two directions: conditioning on the child's own baseline (gains) gives a sharp, replicated effect for letter sounds and word reading; not conditioning on it (levels) gives the same directions with much less resolution. The gain-factor family is the one to quote.
