# Findings — Byrne/RLM historical growth (historical_growth & historical_joint)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8). Numbers from the `reporting`-config refit of 2026-07-16 (see [process note](202607161130-full-statistical-refit.md)). Part of the Byrne, MacDonald & Buckley (2002) comparable-model suite (issue #338). Preliminary.

## What these models ask — and the key caveat

This is a **separate, historical study** (`study_id="rlm"`) reproduced with the project's modern Bayesian machinery. Children were **not randomised**, so — by design — **every estimate here is descriptive**: natural-history growth in a historical cohort, not an intervention effect and not an explanation of group differences. The groups are the original study's: **Down syndrome**, **reading-matched**, and **average readers**.

- **`historical_growth`** (`rlm-hg-001…009`): per-measure **group-by-wave growth** — how much each group's score rises between waves — for word reading, spelling, comprehension, receptive vocabulary, grammar, digit recall, similarities, number skills, and matrices. Each model also reports a **reproduction check**: the largest gap between the model's fitted cell mean and the observed mean.
- **`historical_joint`** (`rlm-jc-001`): fits word reading, receptive vocabulary and digit recall **together** and reports **between-child correlations** of their stable levels.

## Convergence gate

All 10 (9 growth + 1 joint) **passed** cleanly (0 divergences). (The Byrne _measurement_ model `rlm-mm-001` is a `corr_factor` funnel case — see the [measurement-models note](202607161800-findings-measurement-models.md).)

## Results — growth (headline interval per measure)

| Measure              | Group / interval reported | Growth                    | Evidence    | Reproduction gap |
| -------------------- | ------------------------- | ------------------------- | ----------- | ---------------- |
| BAS word reading     | Down syndrome, w1→w2      | +5.4 items (+2.6 to +8.1) | very strong | 0.4 items        |
| BAS spelling         | Down syndrome, w1→w5      | +3.9 items (+2.7 to +5.0) | very strong | 0.1              |
| WORD comprehension   | Down syndrome, w1→w5      | +4.5 items (+2.9 to +6.2) | very strong | 0.2              |
| BPVS receptive vocab | Average readers, w1→w4    | +4.6 items (+2.9 to +6.2) | very strong | 0.1              |
| TROG grammar         | Average readers, w1→w4    | +3.4 items (+2.4 to +4.3) | very strong | 0.1              |
| BAS digit recall     | Average readers, w1→w4    | +5.1 items (+3.3 to +6.8) | very strong | 0.1              |
| BAS similarities     | Down syndrome, w1→w5      | +2.9 items (+1.7 to +4.1) | very strong | 0.1              |
| BAS number skills    | Down syndrome, w1→w4      | +7.1 items (+4.7 to +9.5) | very strong | (see report)     |
| BAS matrices         | Reading-matched, w3→w4    | +4.0 items (+2.0 to +6.0) | very strong | 0.2              |

## Results — joint (rlm-jc-001)

The clearest between-child coupling was **word reading ↔ receptive vocabulary**: a stable-level correlation of **+0.68** (95% range +0.49 to +0.84; 99.9% positive) — children who sit higher on one sit higher on the other.

## The one-paragraph story

The historical cohorts show **clear, well-resolved growth on every measure** across waves, and the model **reproduces the original observed group means to within a fraction of a test item** (largest gap ≈ 0.4 items) — a strong reproduction check that the modern re-analysis faithfully recovers the historical data. Between children, reading and vocabulary levels are strongly correlated. None of this is causal — it is a faithful, uncertainty-quantified description of how these historical groups progressed.

## What is causal

**Nothing — structurally.** With no randomisation in this study, growth curves and correlations are descriptive by construction. Outstanding Byrne design decisions (instrument ceilings, group-scope framing, reading-matched selection handling) are tracked in issue #338 and its planning notes.
