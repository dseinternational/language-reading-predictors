# Findings — Byrne/RLM historical growth (historical_growth & historical_joint)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8). Numbers from the `reporting`-config refit under the median + inner-50% + outer-89% equal-tailed credible-interval standard (2026-07-18; see [the credible-interval standard note](202607172359-credible-interval-standard.md) and [process note](202607161130-full-statistical-refit.md)); reviewed and extended on 2026-07-17 to cover all models in the family. Part of the Byrne, MacDonald & Buckley (2002) comparable-model suite (issue #338). Only the credible-interval brackets changed when we moved from 95% to 89% — medians, direction probabilities and evidence labels are unchanged. Preliminary.

## What these models ask — and the key caveat

This is a **separate, historical study** (`study_id="rlm"`) reproduced with the project's modern Bayesian machinery. Children were **not randomised**, so — by design — **every estimate here is descriptive**: natural-history growth in a historical cohort, not an intervention effect and not an explanation of group differences. The groups are the original study's three reading-ability strata: **Down syndrome**, **reading-matched** (typically developing children matched to the Down syndrome group on reading level, hence younger), and **average readers** (age-appropriate typically developing children). Because the design is observational, this note reads across to the intervention suites ([ITT](202607161800-findings-itt.md) and the gain/level-factor families) only as a natural-history backdrop, never as a causal comparison.

Two model families sit here:

- **`historical_growth`** (`rlm-hg-001…009`): per-measure **group-by-wave growth** — how much each group's score rises between waves — for word reading, spelling, reading comprehension, receptive vocabulary, receptive grammar, digit recall, similarities (verbal reasoning), number skills, and matrices (non-verbal reasoning). Each model reports both the **within-group trajectory** (wave-to-wave and cumulative increments for every group where the measure was collected) and the **between-group total-growth contrasts** — differences in total growth across the common window, which are effectively group×wave interaction terms. Each model also carries a **reproduction check**: the largest gap between the model's fitted cell mean and the observed mean, a fidelity test that the modern re-analysis recovers the historical data.
- **`historical_joint`** (`rlm-jc-001`): fits word reading, receptive vocabulary and digit recall **together** and reports **between-child correlations** of their stable (person-level) latent levels — how children who sit high on one measure tend to sit on the others.

A brief note on the numbers below. A **89% credible range/interval** is the range within which the parameter lies with 89% probability _given the data and the priors_ — a direct probability statement about the quantity itself, unlike a frequentist confidence interval. **P(growth>0)** (the `p_gt_0` column in the CSVs) is the posterior probability that the true increment is positive. The **evidence label** grades _only that direction probability_ on the project ladder (inconclusive < 0.75 ≤ suggestive < 0.91 ≤ moderate < 0.97 ≤ strong < 0.99 ≤ very strong); it says how confident we are that growth is _positive_, **not** that the amount of growth is large or clinically meaningful — read the item counts for magnitude. This descriptive family does **not** define a ROPE ("region of practical equivalence", a band around zero deemed too small to matter), so there is no formal "big enough to matter" verdict; judge size from the median in test items and its interval.

## Convergence gate

All 10 models (9 `historical_growth` + 1 `historical_joint`) **passed** the gate cleanly with **0 divergences** each. Maximum R-hat ranged from 1.0005 (`hg-002`) to 1.0045 (`hg-001`), all comfortably under the 1.01 threshold; no model was flagged. (The Byrne _measurement_ model `rlm-mm-001` is a `corr_factor` funnel case handled separately — see the [measurement-models note](202607161800-findings-measurement-models.md).)

## Results — within-group growth, all three groups

Cumulative growth over the widest window each group was measured (median = best estimate; 89% credible range in brackets). All quantities are **descriptive** natural-history growth, not causal. Growth is clearly positive on every measure for every group, but the _amount_ varies widely — from ~+2 to +3 items on the reasoning and vocabulary measures up to ~+21 (Down syndrome) and ~+37 (average readers) items on word reading.

| Model  | Measure                    | Group           | Window | Growth (items)         | P(>0) | Evidence    |
| ------ | -------------------------- | --------------- | ------ | ---------------------- | ----- | ----------- |
| hg-001 | BAS word reading           | Down syndrome   | w1→w5  | +21.3 (+18.4 to +24.1) | 1.00  | very strong |
| hg-001 | BAS word reading           | Average readers | w1→w4  | +37.1 (+34.4 to +39.8) | 1.00  | very strong |
| hg-001 | BAS word reading           | Reading-matched | w1→w4  | +33.3 (+28.8 to +37.6) | 1.00  | very strong |
| hg-002 | BAS spelling               | Down syndrome   | w1→w5  | +3.9 (+2.9 to +4.8)    | 1.00  | very strong |
| hg-002 | BAS spelling               | Average readers | w1→w4  | +6.7 (+5.7 to +7.6)    | 1.00  | very strong |
| hg-002 | BAS spelling               | Reading-matched | w1→w4  | +6.8 (+5.4 to +8.1)    | 1.00  | very strong |
| hg-003 | WORD reading comprehension | Down syndrome   | w1→w5  | +4.5 (+3.2 to +5.9)    | 1.00  | very strong |
| hg-003 | WORD reading comprehension | Average readers | w1→w4  | +13.0 (+11.5 to +14.4) | 1.00  | very strong |
| hg-003 | WORD reading comprehension | Reading-matched | w1→w4  | +10.1 (+8.5 to +11.8)  | 1.00  | very strong |
| hg-004 | BPVS receptive vocabulary  | Down syndrome   | w1→w5  | +3.5 (+1.9 to +5.1)    | 1.00  | very strong |
| hg-004 | BPVS receptive vocabulary  | Average readers | w1→w4  | +4.6 (+3.2 to +5.9)    | 1.00  | very strong |
| hg-004 | BPVS receptive vocabulary  | Reading-matched | w1→w4  | +5.2 (+3.3 to +7.0)    | 1.00  | very strong |
| hg-005 | TROG receptive grammar     | Down syndrome   | w1→w5  | +2.8 (+1.6 to +4.0)    | 1.00  | very strong |
| hg-005 | TROG receptive grammar     | Average readers | w1→w4  | +3.4 (+2.6 to +4.2)    | 1.00  | very strong |
| hg-005 | TROG receptive grammar     | Reading-matched | w1→w4  | +5.3 (+4.1 to +6.5)    | 1.00  | very strong |
| hg-006 | BAS recall of digits       | Down syndrome   | w1→w5  | +2.8 (+1.3 to +4.3)    | 1.00  | very strong |
| hg-006 | BAS recall of digits       | Average readers | w1→w4  | +5.1 (+3.6 to +6.5)    | 1.00  | very strong |
| hg-006 | BAS recall of digits       | Reading-matched | w1→w4  | +4.7 (+2.8 to +6.6)    | 1.00  | very strong |
| hg-007 | BAS similarities           | Down syndrome   | w1→w5  | +2.9 (+1.9 to +3.9)    | 1.00  | very strong |
| hg-007 | BAS similarities           | Average readers | w1→w4  | +4.9 (+3.9 to +6.0)    | 1.00  | very strong |
| hg-007 | BAS similarities           | Reading-matched | w1→w4  | +5.3 (+3.8 to +6.6)    | 1.00  | very strong |
| hg-008 | BAS number skills          | Down syndrome   | w1→w4  | +7.1 (+5.1 to +9.0)    | 1.00  | very strong |
| hg-008 | BAS number skills          | Average readers | w1→w4  | +15.6 (+13.8 to +17.4) | 1.00  | very strong |
| hg-008 | BAS number skills          | Reading-matched | w1→w4  | +18.6 (+16.1 to +21.0) | 1.00  | very strong |
| hg-009 | BAS matrices               | Down syndrome   | w3→w5  | +1.0 (+0.1 to +1.9)    | 0.96  | moderate    |
| hg-009 | BAS matrices               | Average readers | w3→w4  | +3.3 (+1.9 to +4.7)    | 1.00  | very strong |
| hg-009 | BAS matrices               | Reading-matched | w3→w4  | +4.0 (+2.3 to +5.6)    | 1.00  | very strong |

**Reproduction check (fidelity to the original observed means):** the largest fitted-vs-observed cell gap was tiny in every model — hg-001 **0.42** items (Reading-matched wave 1), hg-002 **0.085** (DS wave 2), hg-003 **0.24** (DS wave 4), hg-004 **0.052** (DS wave 1), hg-005 **0.13** (Reading-matched wave 4), hg-006 **0.072** (DS wave 4), hg-007 **0.083** (DS wave 1), **hg-008 0.12 (DS wave 1)**, hg-009 **0.18** (DS wave 3). Every model recovers the historical group means to within roughly half a test item.

**Late-wave plateaus (Down syndrome).** The cumulative headlines above conceal a consistent flattening at the top of the Down syndrome trajectory. Word reading accelerates at w3→w4 (+8.6 items, P(>0)=1.00) then decelerates sharply at w4→w5 to **+2.0 items (−1.1 to +5.1, P(>0)=0.85, suggestive)**; digit recall's final increment is essentially flat at **−0.1 items (−1.7 to +1.5, P(>0)=0.46, inconclusive)**; matrices' final increment is +0.28 (−0.69 to +1.26, P(>0)=0.68, inconclusive); and similarities shows _no_ w1→w2 movement (+0.002, P(>0)=0.50, inconclusive) before climbing later. Growth in these historical cohorts is therefore clearly **non-constant across waves** (trajectory curvature), though no formal knee/threshold or interaction term is fitted — the curvature is read off the successive wave increments, all descriptive.

## Results — between-group growth divergences (group×wave contrasts)

These `total_growth_*` rows contrast _total_ growth over the common window between two groups — the substantive story a reader would otherwise miss. A positive value means the first-named group grew more; all are **descriptive** group×wave interactions, never causal. The gaps are large and well-resolved on word reading and number skills; the standout _reversal_ is reading comprehension, where **reading-matched children grew less than average readers**.

| Model / measure        | Contrast                        | Difference in total growth (items) | P(>0) | Evidence              |
| ---------------------- | ------------------------------- | ---------------------------------- | ----- | --------------------- |
| hg-001 word reading    | Average − Down syndrome         | +18.6 (+14.9 to +22.3)             | 1.00  | very strong           |
| hg-001 word reading    | Reading-matched − Down syndrome | +14.8 (+9.7 to +19.9)              | 1.00  | very strong           |
| hg-001 word reading    | Reading-matched − Average       | −3.8 (−9.0 to +1.3)                | 0.11  | suggestive (negative) |
| hg-002 spelling        | Average − Down syndrome         | +4.4 (+3.2 to +5.6)                | 1.00  | very strong           |
| hg-002 spelling        | Reading-matched − Down syndrome | +4.5 (+3.0 to +6.0)                | 1.00  | very strong           |
| hg-002 spelling        | Reading-matched − Average       | +0.10 (−1.5 to +1.7)               | 0.54  | inconclusive          |
| hg-003 comprehension   | Average − Down syndrome         | +9.2 (+7.3 to +11.0)               | 1.00  | very strong           |
| hg-003 comprehension   | Reading-matched − Down syndrome | +6.3 (+4.3 to +8.3)                | 1.00  | very strong           |
| hg-003 comprehension   | Reading-matched − Average       | **−2.8 (−5.0 to −0.6)**            | 0.02  | strong (negative)     |
| hg-004 receptive vocab | Average − Down syndrome         | +2.6 (+0.62 to +4.61)              | 0.98  | strong                |
| hg-004 receptive vocab | Reading-matched − Down syndrome | +3.2 (+0.91 to +5.51)              | 0.99  | strong                |
| hg-004 receptive vocab | Reading-matched − Average       | +0.60 (−1.67 to +2.86)             | 0.66  | inconclusive          |
| hg-005 grammar         | Average − Down syndrome         | +1.2 (−0.09 to +2.58)              | 0.93  | moderate              |
| hg-005 grammar         | Reading-matched − Down syndrome | +3.2 (+1.52 to +4.77)              | 1.00  | very strong           |
| hg-005 grammar         | Reading-matched − Average       | +1.9 (+0.46 to +3.34)              | 0.98  | strong                |
| hg-006 digit recall    | Average − Down syndrome         | +2.2 (+0.28 to +4.21)              | 0.97  | moderate              |
| hg-006 digit recall    | Reading-matched − Down syndrome | +1.9 (−0.43 to +4.23)              | 0.90  | suggestive            |
| hg-006 digit recall    | Reading-matched − Average       | −0.34 (−2.71 to +2.03)             | 0.41  | inconclusive          |
| hg-007 similarities    | Average − Down syndrome         | +2.5 (+1.15 to +3.91)              | 1.00  | very strong           |
| hg-007 similarities    | Reading-matched − Down syndrome | +2.8 (+1.18 to +4.49)              | 1.00  | very strong           |
| hg-007 similarities    | Reading-matched − Average       | +0.31 (−1.44 to +2.07)             | 0.61  | inconclusive          |
| hg-008 number skills   | Average − Down syndrome         | +8.5 (+5.9 to +11.2)               | 1.00  | very strong           |
| hg-008 number skills   | Reading-matched − Down syndrome | +11.5 (+8.4 to +14.6)              | 1.00  | very strong           |
| hg-008 number skills   | Reading-matched − Average       | +3.0 (−0.07 to +6.04)              | 0.94  | moderate              |
| hg-009 matrices        | Average − Down syndrome         | +2.6 (+0.96 to +4.24)              | 0.99  | very strong           |
| hg-009 matrices        | Reading-matched − Down syndrome | +3.3 (+1.45 to +5.15)              | 1.00  | very strong           |
| hg-009 matrices        | Reading-matched − Average       | +0.70 (−1.47 to +2.85)             | 0.70  | inconclusive          |

The recurring pattern: over these windows the two typically developing groups grow **markedly faster than the Down syndrome group** on word reading (+18.6 / +14.8 items) and number skills (+8.5 / +11.5 items), with smaller but still well-resolved gaps on comprehension, similarities and matrices. Down syndrome children grow steadily and reliably (all their own trajectories are "very strong" positive) but at a slower rate over the same calendar window. The reading-matched-vs-average contrasts are mostly inconclusive **except** comprehension, where reading-matched children grew _less_ (−2.8 items, 98% probability the gap is negative) and grammar, where reading-matched grew _more_ (+1.9 items, strong).

## Results — joint model (rlm-jc-001) between-child correlations

Fitting word reading, receptive vocabulary and digit recall together yields three **between-child** stable-level correlations — how children rank together across measures. All three are positive; all are **descriptive** couplings between latent person-levels, not causal paths (a child high on one skill tends to be high on the others, but neither drives the other).

| Correlation                         | Estimate | 89% credible range | P(>0) | Evidence    |
| ----------------------------------- | -------- | ------------------ | ----- | ----------- |
| Word reading ↔ receptive vocabulary | +0.68    | +0.53 to +0.81     | 1.00  | very strong |
| Word reading ↔ digit recall         | +0.64    | +0.50 to +0.77     | 1.00  | very strong |
| Receptive vocabulary ↔ digit recall | +0.53    | +0.32 to +0.71     | 1.00  | very strong |

(The word-reading↔vocabulary probability is a full 1.00 in `measure_correlation_summary.csv`; an earlier draft of this note rounded it to "99.9%".)

## The one-paragraph story

The historical cohorts show **clear, well-resolved positive growth on every measure for every group** — Down syndrome word reading alone rises ~+21 items across waves 1–5 — and the models **reproduce the original observed group means to within roughly half a test item** (largest gap ≈ 0.42 items), a strong fidelity check that the modern re-analysis faithfully recovers the historical data. The richer story is in the _between-group_ growth: the two typically developing groups pull away from the Down syndrome group on word reading and number skills (gaps of +15 to +19 items, ~100% probability positive), while reading-matched children uniquely grow _less_ than average readers on comprehension (−2.8 items). Within the Down syndrome group, later-wave increments flatten (word reading +2.0, digit recall −0.1, matrices +0.28 at the top wave). Between children, reading, vocabulary and digit-recall levels are all positively correlated (+0.53 to +0.68). None of this is causal — it is a faithful, uncertainty-quantified description of how these historical groups progressed.

## What is causal

**Nothing — structurally.** With no randomisation in this study, growth curves, between-group divergences and between-child correlations are descriptive by construction; every table above reports an association or a natural-history trajectory, never "X drives Y". Contrast this with the randomised suites, where only a randomisation-licensed term is causal (τ in the [ITT family](202607161800-findings-itt.md), τ_t2 in the DiD family, the period-1 treatment marginal in gain-factors). Outstanding Byrne design decisions (instrument ceilings, group-scope framing, reading-matched selection handling) are tracked in issue #338 and its planning notes.
