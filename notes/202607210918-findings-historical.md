# Historical growth (Byrne reading-language-memory cohort) findings (2026-07-21)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

This is one of a set of per-family notes from the full 2026-07-21 refit of every Bayesian statistical model in the study (production `reporting` configuration, 6 chains × 6000 draws, 89% credible intervals). Start with the index and reading guide (`notes/202607210900-findings-00-index-and-reading-guide.md`) for the study background, the outcome measures, and — importantly — the house rules for reading a posterior (median, 89% interval, tail probability, evidence ladder). This family has **ten models**, and **all ten pass the convergence gate**. Every number below comes from the current refit's own key-findings output; nothing here is a causal effect.

## What this family probes

These ten models describe a **separate, historical comparison cohort** — the Byrne "reading-language-memory" (RLM) study (`study_id = "rlm"`) — and are **not** part of the main randomised RLI intervention trial reported by the ITT, difference-in-differences and gain/level-factor families. There is **no intervention contrast anywhere in this cohort**: the children were never randomised to a treatment, so by construction every estimate is a **descriptive natural-history** quantity — how much a skill changes over the study's waves, and how skills co-vary between children. The family exists to give the trial a developmental backdrop: an independent, published benchmark of how these skills grow in the absence of any specific intervention.

The cohort is organised into the original study's three reading-ability groups — children with **Down syndrome**, **average readers** (age-appropriate typically-developing children), and a **reading-matched** group (typically-developing children matched to the Down-syndrome group's reading level, so chronologically younger). The models come in two shapes:

- **Nine per-measure growth models (`historical_growth`, `rlm-hg-001…009`).** Each takes one standardised test and fits every group its own trajectory across the study's waves, reporting how far scores rise over the observed window. There is no treatment term to adjust for; the only structure in the model is the group (a cohort label) crossed with wave, plus a per-child random intercept that partially pools each child's stable individual differences. The nine measures span reading (word reading, spelling, reading comprehension), language (receptive vocabulary, receptive grammar), memory (digit recall) and cognitive/ability domains (verbal similarities, number skills, non-verbal matrices).
- **One joint model (`historical_joint`, `rlm-jc-001`).** Fits three of those measures — word reading, receptive vocabulary and digit recall — together, giving each child a stable per-measure level (net of their group's wave-by-wave trajectory) and correlating those levels across measures. Its question is whether children who sit persistently high on one skill tend to sit high on the others.

Because the groups are a cohort factor rather than a randomised assignment, any between-group difference here describes **how the groups differ**, not what caused the difference; and any correlation describes **who ranks together**, not a lever. The child random intercept partially pools stable heterogeneity but does not stand in for latent general ability, and residual confounding by that ability remains for every quantity in the family.

## How to read these numbers

The point estimate is the posterior **median**; uncertainty is the **89% equal-tailed credible interval** — read as "there is an 89% posterior probability the value lies in this range", a direct probability statement about the quantity itself (unlike a frequentist confidence interval). Direction is read from the **tail probability**, e.g. P(growth > 0) = 0.999, taken directly and never as a p-value. The **evidence ladder** attaches a fixed label to that tail probability — inconclusive (< 0.75), suggestive (≥ 0.75), moderate (≥ 0.91), strong (≥ 0.97), very strong (≥ 0.99) — describing the strength of evidence for the directional claim only, never the size of the effect. Direction and size are separate: a very-strong label means "almost certainly positive", not "large" — judge magnitude from the item count and its interval.

**Units.** The growth summaries are already in **items of the test's own raw score** (e.g. BAS word-reading points), so they are quoted directly. These are the RLM cohort's own standardised tests (BAS, WORD, BPVS, TROG), with their own scales — do **not** apply the RLI trial's item maxima to them. The joint model reports a **correlation** on the usual −1 to +1 scale.

**Reproduction check.** Each growth model also reports the largest gap between its fitted group-by-wave cell mean and the observed mean — a fidelity test that the modern re-analysis recovers the historical published data. Small gaps (well under an item) mean the reproduction is faithful.

**Causal status.** `causal_status = none` for all ten. Nothing here is causal; every figure is a descriptive trajectory or a between-child association.

## Per-model findings — the nine growth models

Each model surfaces one headline growth summary — a representative group and wave window from that measure's trajectory. All values are in the test's own raw-score **items**, all are **descriptive** natural-history growth (not intervention effects), and all nine **pass the gate**.

| Model  | Measure (test)                      | Headline group  | Window   | Growth (items), 89% range | P(>0) | Evidence    | Reproduction gap | Gate |
| ------ | ----------------------------------- | --------------- | -------- | ------------------------- | ----- | ----------- | ---------------- | ---- |
| hg-001 | BAS word reading                    | Down syndrome   | wave 1→2 | +5.4 (+3.1 to +7.6)       | 0.999 | very strong | 0.4 items        | pass |
| hg-002 | BAS spelling                        | Down syndrome   | wave 1→5 | +3.9 (+2.9 to +4.8)       | 0.999 | very strong | 0.1 items        | pass |
| hg-003 | WORD reading comprehension          | Down syndrome   | wave 1→5 | +4.5 (+3.2 to +5.9)       | 0.999 | very strong | 0.2 items        | pass |
| hg-004 | BPVS receptive vocabulary           | Average readers | wave 1→4 | +4.6 (+3.2 to +5.9)       | 0.999 | very strong | 0.1 items        | pass |
| hg-005 | TROG receptive grammar              | Average readers | wave 1→4 | +3.4 (+2.6 to +4.2)       | 0.999 | very strong | 0.1 items        | pass |
| hg-006 | BAS recall of digits                | Average readers | wave 1→4 | +5.1 (+3.6 to +6.5)       | 0.999 | very strong | 0.1 items        | pass |
| hg-007 | BAS similarities (verbal reasoning) | Down syndrome   | wave 1→5 | +2.9 (+1.9 to +3.9)       | 0.999 | very strong | 0.1 items        | pass |
| hg-008 | BAS number skills                   | Down syndrome   | wave 1→4 | +7.1 (+5.1 to +9.0)       | 0.999 | very strong | 0.1 items        | pass |
| hg-009 | BAS matrices (non-verbal reasoning) | Reading-matched | wave 3→4 | +4.0 (+2.3 to +5.6)       | 0.999 | very strong | 0.2 items        | pass |

**Reading measures — word reading, spelling, comprehension (hg-001, hg-002, hg-003).** For the Down-syndrome group the historical cohort shows clear, well-resolved growth: BAS word reading rises +5.4 items over the wave-1-to-2 window (89% credible range +3.1 to +7.6; P(>0) = 0.999, very strong), BAS spelling +3.9 items across waves 1→5 (+2.9 to +4.8; very strong), and WORD reading comprehension +4.5 items across waves 1→5 (+3.2 to +5.9; very strong). The reproduction gaps (0.4, 0.1 and 0.2 items) confirm the re-analysis lands on the published means to within roughly a third of an item at worst.

**Language measures — receptive vocabulary and grammar (hg-004, hg-005).** These two surface the average-readers benchmark: BPVS receptive vocabulary grows +4.6 items across waves 1→4 (+3.2 to +5.9; very strong) and TROG receptive grammar +3.4 items across waves 1→4 (+2.6 to +4.2; very strong). Reproduction gaps of 0.1 items each.

**Memory — digit recall (hg-006).** BAS recall of digits grows +5.1 items across waves 1→4 for the average-readers group (+3.6 to +6.5; very strong; reproduction gap 0.1 items).

**Cognitive/ability domains — verbal similarities, number skills, non-verbal matrices (hg-007, hg-008, hg-009).** For the Down-syndrome group, BAS similarities rises +2.9 items across waves 1→5 (+1.9 to +3.9; very strong) and BAS number skills +7.1 items across waves 1→4 (+5.1 to +9.0; very strong) — the largest single Down-syndrome increment in the surfaced headlines. BAS matrices surfaces the reading-matched group over the shorter wave-3-to-4 window it supports: +4.0 items (+2.3 to +5.6; very strong). Reproduction gaps are 0.1, 0.1 and 0.2 items.

Across all nine, the direction is uniformly the same — every surfaced growth summary is positive with 99.9% posterior probability (very strong) — so the developmental signal is unambiguous. What varies is the **amount**: from roughly +2.9 items (verbal similarities) to +7.1 items (number skills) on the windows the key-findings surface. Lead with the intervals, not the points: these historical group-by-window increments rest on small per-group samples (each model draws on ~58–77 children across two phases and two-to-three waves), so a point that just clears a threshold is on average inflated.

## Per-model findings — the joint correlated-trajectories model

**rlm-jc-001 — Byrne joint correlated growth: word reading, receptive vocabulary and digit recall (waves 1–4 + Down-syndrome wave 5).** Gate: pass. This model fits the three measures together, gives each child a stable per-measure level net of their group's trajectory, and correlates those stable levels across measures. The key-findings surfaces the **clearest coupling**: between BAS word reading and BPVS receptive vocabulary, a between-child stable-level correlation of **+0.69** (89% credible range +0.53 to +0.81; P(>0) = 0.999, very strong). In words: a child who sits persistently higher on word reading tends to sit higher on receptive vocabulary. This is a **between-child** association (who ranks high together), **not** a within-child coupling — it does not say that improving one skill improves the other, and residual confounding by general ability is exactly what would be expected to produce a correlation of this kind. The model also estimates the word-reading-to-digit-recall and vocabulary-to-digit-recall couplings, but the current refit surfaced only the word-reading-to-vocabulary correlation, so I do not quote specific values for the other two pairs.

## What to take away

The Byrne RLM cohort is a **historical, non-randomised comparison sample**, separate from the trial, and it plays one role: a natural-history benchmark. Read that way, its message is coherent. Every measure grows, in every group the key-findings surfaces, with very-strong evidence that the growth is positive — from around +3 items on the reasoning and grammar measures up to +7 items on number skills over the observed windows — and the models faithfully reproduce the original published group means (largest fitted-minus-observed cell gap 0.4 items, most within 0.1–0.2). The joint model adds that, between children, word-reading and receptive-vocabulary levels are strongly correlated (+0.69), consistent with a shared general-ability influence on where a child sits across skills. None of this is causal: it describes how these historical groups progressed and how their skill levels co-vary, providing developmental context for the randomised trial without ever standing in for an intervention effect. As with the rest of the suite, samples are small and the study is preliminary — lead with the intervals.
