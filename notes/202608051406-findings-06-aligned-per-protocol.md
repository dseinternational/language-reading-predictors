<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Findings 06 — the aligned per-protocol family

Reports every model in the `aligned` family from the 2026-08-04/05 `reporting` refit. **9 models, all passing the convergence gate.** Reading conventions in note 00. Preliminary research data — all estimates provisional.

## What these models do

Every other family compares the arms at a fixed calendar timepoint. This one instead **aligns children by when their intervention started** — immediate-arm children over t1→t3, waitlist-arm children over t2→t4 — so each child contributes a comparable 40-week window beginning at their own onset.

**Design.** One cross-sectional Beta-Binomial ANCOVA per child on the aligned window: post-score given own pre-score. No child random intercept, because each child appears once.

**Nothing in this family is causal, and that is the defining fact about it.** Aligning by onset deliberately breaks the randomised comparison. The two cohorts are no longer being compared at the same point in calendar time or at the same age, so the contrast is confounded by **age at onset** and by **cohort and timing** effects. Every coefficient here, including the cohort contrast itself, is an association. The family exists as a per-protocol view — "what did 40 weeks of intervention look like for the children who received it?" — not as an effect estimate.

## Results — the cohort contrast

Immediate-minus-waitlist cohort difference on the items scale, median with 89% range. **Read every row as an association.**

| Model    | Outcome                    | Cohort contrast (89%)     | P(>0) | Favoured direction   |
| -------- | -------------------------- | ------------------------- | ----: | -------------------- |
| `al-002` | Receptive vocabulary (RV)  | +2.7 items (−1.8 to +7.2) | 0.832 | positive, suggestive |
| `al-004` | Letter sounds (LS)         | +2.2 items (+0.2 to +4.2) | 0.961 | positive, moderate   |
| `al-001` | Word reading (WR)          | +2.1 items (−0.5 to +4.8) | 0.906 | positive, suggestive |
| `al-006` | Phoneme blending (PA)      | +0.3 items (−0.6 to +1.2) | 0.706 | inconclusive         |
| `al-005` | Phonetic spelling (PS)     | +0.0 items (−0.1 to +0.1) | 0.698 | inconclusive         |
| `al-007` | Basic concepts (LF)        | −0.6 items (−1.7 to +0.5) | 0.180 | negative, suggestive |
| `al-008` | Receptive grammar (RG)     | −1.4 items (−3.1 to +0.3) | 0.091 | negative, moderate   |
| `al-003` | Expressive vocabulary (EV) | −3.0 items (−6.9 to +0.8) | 0.102 | negative, suggestive |

`al-101` is a dose sensitivity variant of `al-001` adding cumulative sessions; it returns +2.1 items (P = 0.91), indistinguishable from the base model.

**The reading skills agree with the randomised families; the language measures do not.** Letter sounds (+2.2) and word reading (+2.1) point the same way as the ITT, gain-factor and DiD estimates, though smaller and less certain. But three language measures — expressive vocabulary, receptive grammar and basic concepts — lean **negative**: receptive grammar at moderate strength (0.91), expressive vocabulary and basic concepts suggestively (0.90 and 0.82).

**Do not read those as harm.** This is exactly the confounding the design admits. Aligning by onset means the waitlist cohort's window runs later in calendar time and at older ages; any age-related or maturational difference in vocabulary and grammar trajectories loads directly onto the cohort contrast. The randomised families — which compare arms at the same timepoint — put EV and RG flat and inconclusive (note 01: EV +0.2, RG +0.7). Where a confounded design and a randomised design disagree, the randomised one settles it.

The honest summary is that this family reproduces the reading findings under a different windowing and produces uninterpretable language contrasts, which is about what a per-protocol alignment should be expected to do.

## The dose variant

`al-101` adds cumulative sessions to the word-reading model. Cumulative dose is a **collider** on the causal diagram — it is downstream of both intervention status and the child's capacity to attend — so conditioning on it can open a back-door path. It is included only as a flagged sensitivity, and any movement under it should be read as a back-door sensitivity rather than a better estimate. In practice the estimate does not move at all (+2.14 → +2.15), so the question does not arise here.

## The one clearly-resolved association

Across all nine models the same term dominates: the child's **own starting point** on the measure, at direction probability 1.000 in every one. Children higher at onset were higher 40 weeks later. That is autocorrelation in a skill measure, not a finding about what drives progress, and it should not be reported as one.

## Caveats

- **No causal term.** Not one coefficient in this family carries a causal reading; the cohort contrast is confounded by age at onset and cohort timing.
- **No child random intercept**, by design — one row per child.
- **Cumulative dose is a collider**; `al-101` is a sensitivity, not an estimate.
- **Predictive calibration.** 50% bands cover about 63% of observations — the second-best-calibrated family in the suite, which follows from its one-row-per-child cross-sectional structure.
- **Small samples**, as everywhere: 52–54 children per model.
