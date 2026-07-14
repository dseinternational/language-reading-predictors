<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8).

# Age × ability on the growth rate (#228 item 10)

Date: 2026-07-14

Related: #228 (suite-gap analysis, item 10), #187 (Q5 growth-curve trajectories, LRP69/70), the gain-factor family (the age × ability interaction finding).

## What this adds

The gain-factor models found the **age × ability interaction** positive in 6 of 8 outcomes — very strong for basic concepts and grammar — i.e. _older-and-more-able children progress more than age and ability predict separately_. No model made that a first-class question. This adds **`lrp-rli-gc-085`**, a growth-curve model that puts a baseline-age × ability interaction on the growth **rate**, extending the independent-core LRP69.

## Design

`build_growth_model` gains an opt-in `age_ability_interaction` flag (default off, so LRP69/70 are byte-unchanged). When on, the per-measure slope gains:

```
slope[i,k] += gamma_age_k · age0_i + gamma_int_k · (age0_i · z(ability_i))
```

where `age0_i` is the child's **baseline (t1) age, standardised across children** — a between-child moderator, distinct from the within-child `age_std` time axis the slope multiplies. `gamma_int_k` is the headline "older-and-more-able grow faster" estimand, on the same unit-scaled age × ability scale as the gain factors' `gamma_int_A_ability`. `gamma_age_k` (baseline age → rate) and the existing `gamma_k` (ability → rate) are the interpretable main effects. Slope-only (age0 is kept off the intercept to avoid collinearity with the mean-age anchor). Priors are the shared `Normal(0, 0.3)` cross-association prior. Status ASSOCIATION, GA-confounded, exploratory — never causal.

## Scope caveat (load-bearing for interpretation)

The growth family models the **five verbal/reading trajectories R/E/T/W/L**. **Concepts (F) and blending (B) — where the gain factors' age × ability signal was strongest — are not growth-curve outcomes**, so gc-085 tests the interaction only on those five and cannot directly reproduce the F/B signals. Read it as a direction check on the modelled five, echoing the gain factors' 6-of-8 positive pattern.

## Dev-tier read (reporting folds into the next refit)

All 5 outcomes, 0 divergences. The pattern is coherent:

- `gamma_int` (age × ability on rate) is **positive** for R, E, W and L (largest for letter sounds, ≈ +0.27) and ≈ 0 for grammar T — a mostly-positive direction consistent with the gain factors, on the modelled outcomes.
- `gamma_age` (baseline age → rate) is **negative** throughout — younger children grow faster, echoing the adj-065 "younger age predicts more word-reading gain" finding.

Wide intervals at n≈54; the deliverable is direction, not magnitude.

## Files

- `factories.build_growth_model` — opt-in `age_ability_interaction` flag (age0 + gamma_age + gamma_int on the slope).
- `pipeline.fit_growth` — reads the flag from `extra`, forwards it, adds `gamma_age`/`gamma_int` to the diagnostic vars.
- `lrp_rli_gc_085.py` — the new model (independent-core + the age × ability layer).
- `definitions.py` — `_STRUCT` entry (`lrp85` → `gc-085`).
- `docs/models/lrp-rli-gc-085/index.qmd` — report chapter.

This is preliminary, exploratory work in progress.
