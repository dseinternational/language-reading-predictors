<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Findings 18 — the historical Byrne cohort

Reports every model in the `historical_growth` family (9) and the `historical_joint` family (1) from the 2026-08-04/05 `reporting` refit. **10 models, all passing the convergence gate.** Reading conventions in note 00. Preliminary research data — all estimates provisional.

## What these models do

A **separate, older, non-randomised cohort** — the "reading-language-memory" (`rlm`) study — providing natural-history context for the trial findings. Children with Down syndrome were followed alongside two comparison groups, with no intervention.

**This cohort answers a question the trial cannot**: what does progress look like _without_ an intervention, and how do children with Down syndrome compare with typically-developing children matched in different ways?

**Design.** Descriptive repeated-measures growth models on a group × wave grid, one per measure, with a child random intercept. Waves 1–4 for all groups, plus a fifth wave for the Down syndrome group only.

**Three groups:**

- **Down syndrome** — the focal group;
- **Reading-matched** — typically-developing children matched on reading level;
- **Average readers** — typically-developing children of similar age.

**Nothing here is causal.** There is no intervention and no randomisation; these are descriptions of observed trajectories. Group comparisons are confounded by everything that differs between the groups by construction.

## `historical_growth` — nine measures

Each model reports fitted cell means for every group × wave combination. Because the note's question is about the Down syndrome group, the table below reports one consistent quantity throughout: **that group's growth over the longest window the model covers**. (Comparison-group and between-group rows are in each fit's `posterior_growth_summary.csv`; they are described but not tabulated here.)

| Model    | Measure                    | Window    |   n | Down syndrome growth             |  P(>0) |
| -------- | -------------------------- | --------- | --: | -------------------------------- | -----: |
| `hg-001` | BAS word reading           | waves 1→5 |  16 | **+21.3 items** (+18.4 to +24.1) |  1.000 |
| `hg-008` | BAS number skills          | waves 1→4 |  17 | **+7.1 items** (+5.1 to +9.0)    |  1.000 |
| `hg-003` | WORD reading comprehension | waves 1→5 |  17 | **+4.5 items** (+3.2 to +5.9)    |  1.000 |
| `hg-002` | BAS spelling               | waves 1→5 |  16 | **+3.9 items** (+2.9 to +4.8)    |  1.000 |
| `hg-004` | BPVS receptive vocabulary  | waves 1→5 |  17 | +3.5 items (+1.9 to +5.1)        | 0.9997 |
| `hg-007` | BAS similarities           | waves 1→5 |  17 | +2.9 items (+1.9 to +3.9)        |  1.000 |
| `hg-006` | BAS recall of digits       | waves 1→5 |  16 | +2.8 items (+1.3 to +4.3)        |  0.999 |
| `hg-005` | TROG receptive grammar     | waves 1→5 |  16 | +2.8 items (+1.6 to +4.0)        | 0.9999 |
| `hg-009` | BAS matrices               | waves 3→5 |  17 | +1.0 items (+0.1 to +1.9)        |  0.958 |

**The central descriptive fact is that children with Down syndrome do make measurable progress across every domain measured, without intervention, over the study window.** Reading, spelling, comprehension, number skills, verbal reasoning and vocabulary all move, with intervals excluding zero throughout. Non-verbal matrices is the weakest and the only one below very strong evidence (+1.0 items, P = 0.96, moderate), on the shortest window.

**Do not read the rows against each other.** Items are not comparable across instruments — +21 BAS word-reading items and +2.8 TROG items are not statements about relative amounts of learning — and the windows differ (`hg-008` has no fifth wave, `hg-009` starts at wave 3). The comparison groups grow faster than the Down syndrome group on every measure where a between-group contrast is reported, which is expected and not the point of these fits.

That matters for reading the trial results. The waitlist arm was not static, and any intervention effect is a gain **over and above** a background trajectory that is itself positive. It is also a caution against reading the trial's modest item-scale effects as disappointing: they sit on top of natural progress, not against a flat baseline.

**Reproduction quality is high.** Every model reports its largest fitted-minus-observed cell-mean gap, and across the nine those gaps run **0.05 to 0.42 items** (median gap per model 0.02–0.17). The models reproduce the observed group × wave means almost exactly, which is what a descriptive model of this kind should do and is a useful check that the machinery is behaving.

## `jc-001` — the joint correlated-growth model

Fits word reading, receptive vocabulary and digit recall **together**, so the between-child correlations of their stable levels are estimated within one model.

| Measure pair                            | Stable-level correlation (89%) |  P(>0) |
| --------------------------------------- | ------------------------------ | -----: |
| Word reading ↔ receptive vocabulary     | **+0.69** (+0.53 to +0.81)     |  1.000 |
| Word reading ↔ recall of digits         | **+0.65** (+0.50 to +0.77)     |  1.000 |
| Receptive vocabulary ↔ recall of digits | +0.54 (+0.32 to +0.71)         | 0.9998 |

Children who sit higher on one of these measures sit higher on the others, with very strong evidence throughout. These are **observed-scale** correlations of stable levels, so they are lower than the disattenuated latent correlations in note 14 (0.82–0.95 for the same cohort's domain factors) — the gap between the two is measurement error, and comparing them directly would be a mistake.

## Reading this cohort alongside the trial

Two findings from this cohort replicate trial results by an independent route, which is the most valuable thing it contributes:

1. **The negative age–gain association** (notes 12 and 13) appears here too, and more strongly (−8.4 words per SD of age; the horseshoe selects age and nothing else). That the same signal appears in a different cohort, different instruments and no intervention argues it is a property of development in this population rather than of the trial sample.
2. **Near-unidimensionality** (note 14) — the four latent domains in this cohort correlate at 0.82–0.95 — is the clearest available statement of why observational adjustment cannot separate specific skills from general ability anywhere in the suite.

## Caveats

- **Not randomised, no intervention.** Descriptive only; group differences are confounded by construction.
- **Different instruments** from the RLI trial. Items scales are not comparable between the two studies.
- **The fifth wave is Down-syndrome-only**, so any waves 1→5 quantity has no comparison-group counterpart.
- **`hg-009` (matrices) starts at wave 3**, so its window is shorter than the others.
- **Predictive calibration.** 50% bands cover about 85% of observations — the highest in the suite alongside `lcsm`, expected for a panel design whose in-sample check conditions on fitted child effects.
- Until this run, these ten models emitted no prior-predictive check at all; they now do, and all cover the observed range.
