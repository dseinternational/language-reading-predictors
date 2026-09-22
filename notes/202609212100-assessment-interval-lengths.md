<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Unequal assessment intervals: what they do to the gain models

Date: 2026-09-21 — **Status: DECIDED** (author instruction, 2026-09-21) for the three actions below. Whether `lcsm-167` replaces `lcsm-067` as the model of record remains open.

## The question

Word reading looked as if it had slowed over the last period: children gained 2.5 words between t3 and t4 against 4.4 between t2 and t3. That raised two questions. Did progress really slow? And should the gain models, gradient-boosting and Bayesian, allow for the periods being different lengths?

All numbers below come from `scripts/assessment_interval_check.py` (child bootstrap, 20,000 resamples, seed 20260921) run against the reporting fits stored on 21 September 2026. They are descriptive; the bootstrap intervals are 89% percentile intervals, not Bayesian credible intervals.

## What the data record

Both `rli_data_long.csv` and the deposited trial archive record age at each wave in whole months. Neither holds assessment dates. Every interval is therefore a difference of two whole-month ages, and a true gap of 5.4 months appears as 5 or 6.

| Transition | Arm       | Children | Range (months) | Mean | Median | Children by whole months |
| ---------- | --------- | -------: | -------------- | ---: | -----: | ------------------------ |
| t1 → t2    | Both      |       54 | 6–8            | 7.11 |      7 | 6: 6, 7: 36, 8: 12       |
|            | Immediate |       28 | 6–8            | 7.32 |      7 | 6: 1, 7: 17, 8: 10       |
|            | Wait-list |       26 | 6–8            | 6.88 |      7 | 6: 5, 7: 19, 8: 2        |
| t2 → t3    | Both      |       54 | 7–9            | 8.43 |      8 | 7: 3, 8: 25, 9: 26       |
|            | Immediate |       28 | 7–9            | 8.18 |      8 | 7: 3, 8: 17, 9: 8        |
|            | Wait-list |       26 | 8–9            | 8.69 |      9 | 8: 8, 9: 18              |
| t3 → t4    | Both      |       53 | 4–7            | 5.36 |      5 | 4: 1, 5: 33, 6: 18, 7: 1 |
|            | Immediate |       27 | 4–6            | 5.26 |      5 | 4: 1, 5: 18, 6: 8        |
|            | Wait-list |       26 | 5–7            | 5.46 |      5 | 5: 15, 6: 10, 7: 1       |

Within a transition the standard deviation is about 0.6 months, and t1 → t4 spans 20–22 months for every child. Assessments followed a common calendar; the spread within a transition is mostly rounding. One immediate-arm child has no t4 age.

`attend` on a child's row at wave t counts the sessions between t and the next wave; it matches `attend_cumul` exactly. Both arms attended sessions during t3 → t4 (means 52 immediate, 56.5 wait-list). This fits the report plan's inference that t4 closes a third teaching block for the immediate arm and a second for the wait-list arm, which no document yet confirms (`notes/202609071300-technical-report-plan-v2.md`, item 5a).

## Did word reading slow?

Among the 51 children with word reading at t2, t3 and t4:

| Arm       | Children | Gain t2 → t3 | Gain t3 → t4 | Difference (89% interval) | Words per month t2 → t3 | Words per month t3 → t4 | Rate ratio (89% interval) |
| --------- | -------: | -----------: | -----------: | ------------------------- | ----------------------: | ----------------------: | ------------------------- |
| Both      |       51 |         4.39 |         2.53 | −1.86 (−3.06 to −0.71)    |                   0.522 |                   0.473 | 0.90 (0.62 to 1.28)       |
| Immediate |       27 |         4.56 |         2.22 | −2.33 (−4.11 to −0.56)    |                   0.559 |                   0.423 | 0.76 (0.42 to 1.32)       |
| Wait-list |       24 |         4.21 |         2.88 | −1.33 (−2.88 to +0.17)    |                   0.483 |                   0.527 | 1.09 (0.66 to 1.67)       |

Words per month is total words gained divided by total months. Fewer words were gained over t3 → t4, but per month the rate barely changed: the interval runs from a 38% slowdown to a 28% speedup. The median child gained 0.33 words a month in both periods. Excluding children at zero at t2 gives a ratio of 0.93; restricting to children who never scored above 25 (and so never met the extended word list) gives 1.18. The smaller t3 → t4 gain mostly reflects a shorter window. The immediate arm's rate fell in what was probably its third teaching block (80% of resamples lower), a hint that is not established.

## How each gain model treats period length

| Family or model                                                                                            | How the transition enters                         | Affected by unequal lengths?                     |
| ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------ |
| Gradient-boosting gain models                                                                              | `time` is one of the 33 predictors                | Absorbed                                         |
| `gain_factors`, `mechanism`, `dose_response`, `mediation`, `adjusted`, `joint_mechanism`, `block_exposure` | Per-period, per-phase or per-timepoint intercepts | Absorbed                                         |
| `survival`                                                                                                 | Per-interval baseline hazard                      | Absorbed                                         |
| `growth`                                                                                                   | Standardised age is the time axis                 | Handled directly                                 |
| `lcsm-081`, `082`, `091`, `181`                                                                            | Arm × window change intercepts                    | Absorbed                                         |
| `lcsm-067`                                                                                                 | One change intercept pooled over all transitions  | **Affected** — decision 1                        |
| t2 contrasts: `itt`, `joint`, `did` `tau_t2`, `level_factors` `d_grp_time[t2]`, `gain_factors` period 1    | Single-window arm contrast                        | **Arm timing gap at t2** — decision 2            |
| `aligned`                                                                                                  | One onset-aligned window per child                | **Windows differ in length by arm** — decision 3 |

An intercept per transition absorbs the average length of that transition. Within a transition every child's interval is the same to within rounding, so apart from the small arm gaps in decision 2 nothing systematic is left for a per-child term to explain.

**No per-child interval covariate or per-month outcome.** An interval covariate would carry mostly rounding error, so its coefficient would be biased towards zero and would correct little; it would also not remove a between-arm gap it measures badly. A words-per-month outcome would divide every gain by a rough whole number between 4 and 9. Neither is adopted.

## Decision 1: register `lcsm-167` (`lcsm-067` with arm × window intercepts)

`lcsm-067` gives each measure one change intercept for all three transitions. The last transition is the shortest and starts from the highest levels of word reading, letter sounds and vocabulary, so its smaller change can load onto the couplings from prior levels, the reading self-feedback and the age term. Treatment status also differs by transition, which the pooled intercept ignores. The 1 September findings already flagged this for the age term.

`lrp-rli-lcsm-167` is `lcsm-067` with `arm_window_intercepts=True` and nothing else changed. The intercepts absorb each transition's mean change and each arm's treatment schedule, so a shift between the two fits cannot be assigned to interval length alone.

Both models were fitted at the `rep-lite` tier (4 chains × 4,000 draws, `target_accept` 0.95) on commit `fb10529e` with this change uncommitted, so both configurations record `dirty: true` and neither is a publication. Both pass the clean gate: zero divergences, R-hat at most 1.004 (`067`) and 1.005 (`167`), minimum ESS 1,080 and 1,231, BFMI at least 0.55. The `067` refit reproduces its stored 7 September reporting fit to within 0.02 on every median below, so the code changes since then do not affect the comparison. Medians with 89% equal-tailed credible intervals; P is the posterior probability that the coefficient is positive:

| Coefficient (latent logit scale)             | `lcsm-067` (pooled intercept)     | `lcsm-167` (arm × window intercepts) |
| -------------------------------------------- | --------------------------------- | ------------------------------------ |
| `g_L`: prior letter sounds → reading change  | +0.13 (+0.04 to +0.23), P = 0.990 | +0.22 (+0.13 to +0.33), P = 1.000    |
| `g_E`: prior vocabulary → reading change     | +0.29 (+0.11 to +0.48), P = 0.993 | +0.24 (+0.06 to +0.43), P = 0.983    |
| `b_self[W]`: reading self-feedback           | −0.22 (−0.29 to −0.15)            | −0.26 (−0.34 to −0.19)               |
| `d_age[W]`: age → reading change             | −0.15 (−0.21 to −0.09)            | −0.11 (−0.17 to −0.05)               |
| Letter sounds, SD of change per SD of level  | +0.63 (+0.22 to +0.99)            | +0.73 (+0.45 to +1.01)               |
| Vocabulary, SD of change per SD of level     | +0.43 (+0.16 to +0.72)            | +0.25 (+0.07 to +0.45)               |
| Letter sounds minus vocabulary, standardised | +0.20 (−0.39 to +0.73), P = 0.71  | +0.48 (+0.08 to +0.86), P = 0.97     |

(`167`'s P for `g_L` is 0.9995, shown rounded.)

With per-arm, per-transition intercepts the letter-sound coupling is larger, the vocabulary coupling smaller and the age term closer to zero, though still negative. The standardised comparison of the two predictors moves from inconclusive (P = 0.71) to 0.97 in favour of letter sounds. The letter-sound and age terms moved as the pooled-intercept concern predicted. The vocabulary coupling and the reading self-feedback moved the other way, so the pooled intercept did not simply pull every level-coupled term down; it redistributed between them. Because the intercepts absorb treatment schedules as well as interval lengths, the shift cannot be assigned to length alone. Both couplings remain adjusted associations with latent general ability unblocked.

The window-1 consistency contrast agrees with the ITT estimate: immediate minus wait-list latent change in word reading is +0.40 logits (+0.09 to +0.71, P = 0.98), beside `itt-010`'s conditional logit coefficient of +0.35 (+0.10 to +0.61, P(average marginal effect > 0) = 0.986).

PSIS-LOO cannot choose between them. `167`'s expected log predictive density is 1.5 higher (−1786.0, standard error 23.6, against −1787.5, 22.3), inside the project's |difference| < 4 inconclusive band, and 5–7% of observations have Pareto k above 0.7, so the estimates are unreliable anyway. The case for `167` rests on design: the transitions are known to differ in length and treatment status, and `lcsm-081` already treats arm × window intercepts as mandatory because of the crossover. Recommendation: after a reporting-tier fit, make `lcsm-167` the model of record for this question and keep `lcsm-067` as the pooled-intercept comparator.

## Decision 2: state a timing bound beside every t2 contrast

Relative to the wait-list arm, the immediate arm's t2 assessment came about two weeks later:

| Interval | Immediate minus wait-list (months) | 89% bootstrap interval | Gap ÷ standard error |
| -------- | ---------------------------------: | ---------------------- | -------------------: |
| t1 → t2  |                              +0.44 | +0.21 to +0.66         |                  3.0 |
| t2 → t3  |                              −0.51 | −0.74 to −0.29         |                 −3.5 |
| t3 → t4  |                              −0.20 | −0.43 to +0.03         |                 −1.3 |
| t1 → t3  |                              −0.08 | −0.34 to +0.18         |                 −0.5 |

The standard error uses the observed within-arm spread. The gaps are too large to be rounding, and t1 → t3 is level, so the arms' t1 and t3 assessments were in step and only t2 moved. Every t2 contrast therefore includes slightly more elapsed time in the immediate arm; contrasts at t3 are unaffected.

A first-order bound multiplies the gap by a per-month gain rate. The two ends use the wait-list's untreated t1 → t2 rate and the immediate arm's treated rate; the extra weeks probably fell within teaching, which favours the upper end.

| Outcome | ITT model | Gap (months) | Bound (items) | Stored effect (items) | Upper end ÷ effect |
| ------- | --------- | -----------: | ------------- | --------------------: | -----------------: |
| `W`     | `itt-010` |         0.48 | 0.14 to 0.31  |                  2.37 |               0.13 |
| `L`     | `itt-007` |         0.44 | 0.20 to 0.41  |                  3.52 |               0.12 |
| `B`     | `itt-008` |         0.44 | 0.00 to 0.07  |                  0.99 |               0.08 |
| `TE`    | `itt-002` |         0.44 | 0.11 to 0.20  |                  1.55 |               0.13 |
| `TR`    | `itt-001` |         0.44 | 0.13 to 0.21  |                  1.37 |               0.15 |
| `UE`    | `itt-004` |         0.44 | 0.03 to 0.05  |                  0.31 |               0.16 |
| `UR`    | `itt-003` |         0.44 | 0.02 to 0.08  |                  0.60 |               0.13 |
| `F`     | `itt-025` |         0.44 | 0.00 to 0.09  |                  0.87 |               0.10 |
| `T`     | `itt-026` |         0.44 | 0.06 to 0.10  |                  0.65 |               0.15 |
| `R`     | `itt-005` |         0.44 | 0.19 to 0.19  |                  0.23 |               0.83 |
| `E`     | `itt-006` |         0.44 | 0.26 to 0.27  |                  0.11 |               2.41 |

The gap is computed on each outcome's observed children, which is why word reading's is 0.48. Stored effects are probability-scale average marginal effects times the item count. For every graded outcome except receptive and expressive vocabulary, the upper end is 8–16% of the effect: a small bias in the intervention arm's favour, too small to change any conclusion. For receptive and expressive vocabulary the bound is as large as the effect. Both effects were already inconclusive (P(effect > 0) = 0.54 and 0.53), so no conclusion changes, but their point estimates should not be read as even small benefits.

The bound is reported, not modelled, for the reason given above: an interval covariate measured in whole months cannot remove a between-arm gap of half a month. Phonetic spelling and nonword reading use the floor rule's off-floor risk difference and are not bounded here.

## Decision 3: name window length as an `aligned` confounder

The onset-aligned windows differ by arm: the immediate arm's t1 → t3 window averages 15.5 months, the wait-list's t2 → t4 window 14.2 months.

| Outcome | Aligned model | Gap (months) | Bound (items) | Stored cohort contrast (items) |
| ------- | ------------- | -----------: | ------------- | -----------------------------: |
| `W`     | `al-001`      |         1.33 | 0.67 to 0.77  |                          +2.12 |
| `R`     | `al-002`      |         1.35 | 0.47 to 0.75  |                          +2.65 |
| `E`     | `al-003`      |         1.35 | 0.67 to 0.85  |                          −3.10 |
| `L`     | `al-004`      |         1.34 | 0.47 to 0.70  |                          +2.20 |
| `B`     | `al-006`      |         1.35 | 0.08 to 0.12  |                          +0.29 |
| `F`     | `al-007`      |         1.34 | 0.26 to 0.33  |                          −0.62 |
| `T`     | `al-008`      |         1.35 | 0.19 to 0.24  |                          −1.43 |

The bound uses each arm's own gain per month over its aligned window. About a third of the word-reading contrast and a fifth to a third of the letter-sound and receptive-vocabulary contrasts could be extra elapsed time. Because the gap favours the immediate arm, allowing for it would lower every contrast, making the negative expressive-vocabulary, basic-concepts and grammar contrasts more negative. The family's contrasts are already associations; this adds a named, sized confounder.

## What changed

- `lrp-rli-lcsm-167` registered (module, `MODEL_REGISTRY` companion entry with base `lrp67`, report template, catalogue row, registry counts).
- `METHODS.md`: a new **Assessment intervals** paragraph; window length in the aligned design row; interval length in the period-specific associations paragraph and the gradient-boosting reading of `time`.
- Model catalogue: an **Assessment timing** paragraph in the ITT suite, window length in the aligned purpose.
- `CLAUDE.md`, `AGENTS.md` and `.github/copilot-instructions.md`: the ITT and aligned interpretation rules.
- `scripts/assessment_interval_check.py`: reproduces every table in this note.

## Follow-ups

- Fit `lcsm-167` at reporting tier from a committed tree and decide whether it replaces `lcsm-067` as the model of record.
- The `a_change` prior rationale in `factories/lcsm.py` calls the intercept "the mean annual change"; the transitions are 5–8 months. Changing the text alters stored prior descriptors, so correct it in a batch that refits the family.
- Confirm the t4 teaching schedule (report plan item 5a).

```bash
uv run python scripts/assessment_interval_check.py --output-dir <output root holding the reporting fits>
```
