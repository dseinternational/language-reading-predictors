<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).
>
> Available-case modified ITT terminology added by a LLM-based AI tool (Codex/GPT-5).

# Findings 05 — the waitlist-crossover (DiD) family

Reports every model in the `did` family from the 2026-08-04/05 `reporting` refit. **14 models, all passing the convergence gate.** Reading conventions in note 00. Preliminary research data — all estimates provisional.

## What these models do

The trial was a **waitlist crossover**: one arm received the intervention immediately, the other after a wait. That design offers a second route to the treatment effect — compare the arms at the timepoint where one has been treated and the other has not — and it also lets us watch what happens after the waitlist arm catches up.

**Design.** Arm-by-wave models fitting the bounded t1, t2 and t3 levels jointly, with separate immediate-minus-waitlist gaps at each wave and a child random intercept.

**Four quantities, only one of them causal:**

| Quantity          | What it is                                      | Status                     |
| ----------------- | ----------------------------------------------- | -------------------------- |
| `tau_t2`          | The immediate-minus-waitlist gap at t2          | **Randomised — causal**    |
| `arm_gap_t1`      | The gap at baseline, before anyone was treated  | Balance check              |
| `arm_gap_t3`      | The gap after the waitlist arm crossed over     | Post-crossover association |
| `delta_crossover` | `tau_t2 − arm_gap_t3` — how much the gap closed | Post-crossover association |

**Two design choices worth stating**, because they are what keep the t2 contrast clean. These models do **not** condition on the t2 period-start score, which the treatment has already affected — conditioning on it would be conditioning on a post-treatment variable. And the child random intercept partially pools stable between-child variation rather than making each child their own fixed-effect control.

## Results — the t2 randomised contrast

Items scale (percentage points for the two floor-rule outcomes), median with 89% range.

| Model       | Outcome                        | t2 contrast (89%)             | P(>0) | Evidence     | ITT (note 01)         |
| ----------- | ------------------------------ | ----------------------------- | ----: | ------------ | --------------------- |
| `did-002`   | Letter sounds (LS)             | **+3.5** items (+1.2 to +5.8) | 0.991 | very strong  | +3.5                  |
| `did-001` ‡ | Word reading (WR)              | +2.2 items (−0.3 to +4.7)     | 0.920 | moderate     | +2.4                  |
| `did-004`   | Taught expressive (TE)         | +1.5 items (+0.0 to +3.0)     | 0.949 | moderate     | +1.5                  |
| `did-008`   | Taught receptive (TR)          | +1.2 items (−0.3 to +2.7)     | 0.902 | suggestive   | +1.4                  |
| `did-003` ‡ | Phoneme blending (PA)          | +0.9 items (+0.1 to +1.7)     | 0.956 | moderate     | +1.0 (link-sensitive) |
| `did-009`   | Expressive vocabulary (EV)     | +0.8 items (−4.0 to +5.5)     | 0.608 | inconclusive | +0.2                  |
| `did-010`   | Basic concepts (LF)            | +0.6 items (−0.5 to +1.8)     | 0.809 | suggestive   | +0.9                  |
| `did-005`   | Receptive vocabulary (RV)      | −0.1 items (−5.1 to +5.0)     | 0.492 | inconclusive | +0.2                  |
| `did-012`   | Nonword reading (NW) — floor   | +6 pp (−6 to +18)             | 0.788 | suggestive   | +10 pp                |
| `did-011`   | Phonetic spelling (PS) — floor | +2 pp (−7 to +12)             | 0.652 | inconclusive | +4 pp                 |

‡ **Withheld from release** under the robustness gate — see below. The numbers are shown because this note is a technical record, not a published finding.

**A third cross-design consistency check, and the closest agreement yet.** Letter sounds comes out at +3.5 items — the same number as the available-case modified ITT estimate to one decimal place — and word reading at +2.2 against +2.4. The taught word sets and blending match too. Broad vocabulary is flat and inconclusive, as everywhere.

The evidence labels sit a rung lower than the available-case modified ITT suite for several outcomes (WR moderate rather than strong) because this design spends precision on estimating three wave levels jointly rather than one contrast. The _estimates_ agree; the _intervals_ are slightly wider.

**Baseline balance is good, with one qualification worth stating.** Every `arm_gap_t1` is small — the largest is phonetic spelling at −0.26 logits — and on the P(>0) reading every one is inconclusive. Read the ladder against the _favoured_ direction, as note 00 prescribes, and three of the eleven lean **suggestively** toward the waitlist arm starting higher: basic concepts (0.82), phonetic spelling (0.82) and nonword reading (0.75). None resolves, all three are small, and the direction is the conservative one — a waitlist arm starting slightly ahead would bias `tau_t2` down, not up. Randomisation did its job well enough to license reading `tau_t2` as an effect rather than a pre-existing difference, but "perfect balance" would overstate it.

## What happens after the waitlist arm catches up

`arm_gap_t3` and `delta_crossover` describe the post-crossover picture. **Neither is causal** — by t3 both arms have been treated, differing only in timing.

The pattern is consistent: the t3 gaps are smaller than the t2 gaps but mostly still positive (LS +0.24 logits, P = 0.83; WR +0.23, P = 0.84), and `delta_crossover` — the amount the gap closed — is positive for most outcomes. For taught expressive vocabulary the closing is the clearest in the family (`delta_crossover` P = 0.98).

The tempting reading is "the waitlist children caught up, but not completely, so early intervention has a lasting advantage". **That reading is not supported by this design.** The t3 comparison is between two treated groups differing in timing _and_ in everything that timing correlates with, with no randomised contrast left to anchor it. What can be said is descriptive: the gap narrowed after crossover and, on these point estimates, had not fully closed by t3.

## The dose companions

Four models (`did-006`, `did-007`, `did-013`, `did-107`) look at **how much** intervention a child received rather than whether they were assigned to it.

- `did-006` (WR) and `did-007`/`did-107` (LS) relate outcomes to session attendance, separating current treatment status from treated-centred session intensity, adjusting for arm, the shared pre-randomisation t1 score and t1 age.
- `did-013` (WR) adds exploratory catch-up heterogeneity; its t2 contrast (+2.2 items) is identical to `did-001`, so the heterogeneity terms do not disturb the headline.
- `did-107` is the pooled-slope comparator to `did-007`; the two agree closely (`beta_period` +0.285 vs +0.282), so allowing the dose slope to vary by period buys nothing.

**Dose is not randomised — only assignment is.** Attendance is a choice, and the 2012 trial caveat that "the children least able to learn tended to show the poorest attendance" is exactly the confounding path at issue. These slopes are observational associations and must not be read as "more sessions cause more progress".

One reporting gap: `did-007` produces no plain-language key findings because the findings builder does not handle the dose companion's schema (it looks for a t2 items-scale contrast, which a dose model has no reason to produce). The numbers themselves are sound and are reported above; this is a builder limitation, not a data problem.

## Withheld from release under the robustness gate

The key-findings release gate (`notes/202608051500-decision-key-findings-robustness-release-gate.md`, `statistical_models/release.py`) was extended on 2026-08-05 from the ITT family to every family whose headline rests on a randomised term. It classifies the focal coefficient on power-scaling sensitivity, and where the posterior responds to the **prior** but not to the **likelihood** the direction is not established by the data alone, so the causal headline is withheld and the report's result tables are suppressed.

- **`did-001`** (word reading, `tau_t2` prior 0.053 vs likelihood 0.031) and **`did-003`** (phoneme blending, 0.066 vs 0.047) — the two headline arm-by-wave models flagged.
- **`did-013`** (word reading, 0.055 vs 0.026) — the catch-up-heterogeneity variant, whose t2 contrast is identical to `did-001`, so it fails on the same statistic.
- **`did-007`** (letter sounds, `mu_dose` 0.063 vs 0.031) — the dose companion, gated on its own focal slope rather than on `tau_t2`, which it does not have.

Note the shape of it: **the DiD family is the worst-affected in the suite**, with four of fourteen fits withheld against two of twenty-eight in ITT. The two families estimate overlapping effects from the same children, so this is a statement about how much the arm-by-wave parameterisation leans on its priors at n ≈ 54, not about the intervention.

Nothing here needs refitting. Attaching a `tau_prior_sensitivity.csv` sweep to the fit, showing the sign holds across the treatment-prior grid, lifts the withhold. **Until then these rows are not results.**

## Caveats

- **One causal number per model** (`tau_t2`), on the available-case t2 population.
- **The t3 story is descriptive.** Do not convert "the gap had not fully closed" into a claim about lasting advantage.
- **Dose slopes are associations**, with a known and plausible confounding path.
- **Predictive calibration.** 50% bands cover about 82% of observations — among the highest in the suite, expected for a repeated-measures design whose in-sample check conditions on fitted child effects.
- The nested PSIS-LOO comparison for the `did-007`/`did-107` dose pair is **unavailable**: both fits have unreliable Pareto-k (1.17 and 1.15) and the exact-refit repair path is mechanism-only, so the comparison script reports per-model `elpd_loo` instead of a difference. Treat the pooled-versus-period-varying question as unresolved rather than settled in favour of either.
