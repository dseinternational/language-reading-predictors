# Growth trajectories & historical-cohort comparison findings (2026-07-20)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

This is family note 10 in the series introduced by the findings index and reading guide (`notes/202607200900-findings-00-index-and-reading-guide.md`). Read that first for the study background, the outcome measures, and the house rules for reading a posterior. Everything here is reported model by model, and — as the brief for the whole series insists — the adjusted **associations** are treated as seriously as the causal effects reported elsewhere. **Nothing in this family is a causal effect.** Every number below describes _who progresses together_ or _how two cohorts differ_, not _what an intervention causes_.

## What this family does, and the questions it answers

This family has three sub-groups that share a common shape — each fits a growth trajectory (a starting level plus a rate of change) — but they answer different questions and, importantly, use **two different samples**.

**A. Growth-coupling (`kind=growth`, three models: gc-069, gc-070, gc-085).** These use the RLI intervention-study children (54 children, four waves, 216 child-wave rows). They fit every child a starting point and a growth speed for five skills at once and ask two between-child questions: (i) does a child's _baseline non-verbal ability_ — the t1 WPPSI Block Design puzzle score, "blocks" — predict the _shape_ of their trajectories, and (ii) do the five skills grow _together_ (a shared developmental tempo)? The five skills are R = receptive vocabulary (ROWPVT), E = expressive vocabulary (EOWPVT), T = receptive grammar (TROG-2), W = word reading (EWRSWR) and L = letter-sound knowledge (YARC-LSK). Note: this model does **not** split by randomised arm — it treats intervention exposure as part of each child's latent developmental tempo, so no line here is a randomised contrast.

**B. Historical growth (`kind=historical_growth`, nine models: hg-001…hg-009).** These use a _different, older, published_ comparison cohort — the Byrne et al. "Reading and Language" study — reproduced and audited against its published Table 2. That cohort has three reading groups: children with **Down syndrome**, a group of typically-developing **average readers** (older children reading at an age-average level), and a **reading-matched** group (younger typically-developing children matched to the Down syndrome group's reading level — they read similarly at baseline but are chronologically younger). Each model takes one standardised test and asks: how fast does each group grow, and — the headline — how does the growth of children with Down syndrome compare to the two typically-developing benchmarks? This is a **natural-history** benchmark, entirely separate from the RLI intervention.

**C. Historical joint (`kind=historical_joint`, one model: jc-001).** The same Byrne cohort, but three measures (word reading, receptive vocabulary, digit recall) fitted jointly with the per-child stable levels correlated across measures. Its headline question: do children who sit persistently high on one skill also sit high on the others (a between-child association), net of their group's trajectory?

## How to read these numbers (compact recap)

See the shared reading guide (`notes/202607200900-findings-00-index-and-reading-guide.md`) for the full version. In brief: the point estimate is the posterior **median**; uncertainty is the **89% equal-tailed credible interval** ("89% posterior probability the value lies in this range"), with the inner **50%** interval as the most-of-the-mass range where quoted; direction is the **tail probability** (e.g. P(>0) = 0.98), read directly, never as a p-value. The evidence ladder attaches a fixed label to that tail probability — **inconclusive** (< 0.75) / **suggestive** (≥ 0.75) / **moderate** (≥ 0.91) / **strong** (≥ 0.97) / **very strong** (≥ 0.99) — describing the _strength of evidence for a directional claim_, oriented to the favoured direction, and never the size of the effect. With ~54 children (group-coupling) or 15–32 per group (historical), point estimates that just clear a threshold are on average inflated, so lead with the interval.

**Scale and units.** The growth-coupling coefficients are on the **logit** (log-odds) scale, because each skill is modelled as a proportion-correct with a logistic link; I anchor them below with the population trajectory in **items**. The historical-cohort growth summaries are already reported in **items of the test's own raw score** (e.g. BAS word-reading points), so I quote them directly. Do **not** apply the RLI measures' item maxima (W = 79, etc.) to the historical models — those are different standardised tests with their own scales.

**Causal status.** `causal_status` is `adjusted` for the growth-coupling models and `none` for the historical models — all associations. For the growth-coupling models, block design is an off-DAG child covariate and a proxy for latent general ability, so its coefficients are confounded and must never be read as "non-verbal ability drives growth" (the Table-2 fallacy). For the historical models, `readgrp` is a cohort label, so a between-group gap describes how the groups differ, not what caused the difference. The child random intercept (a per-child offset that partially pools stable individual differences) does not stand in for general ability, and residual confounding by latent general ability remains throughout.

---

## Sub-group A — Growth-coupling (RLI cohort; associations)

**Headline coefficient: `gamma` — does +1 SD of baseline non-verbal ability go with a faster logit growth _rate_ for each skill?** (This is the study's "Q5" estimand.) All three models pass the gate (0 divergences; max R-hat ≤ 1.003; min ESS ≥ 2,300).

| Model                     | Skill               | `gamma` median (logit) | 89% CI           | P(>0) | Favoured direction + label | Gate |
| ------------------------- | ------------------- | ---------------------- | ---------------- | ----- | -------------------------- | ---- |
| gc-069 (independent core) | R receptive vocab   | −0.035                 | −0.093 to +0.023 | 0.169 | negative, suggestive       | pass |
| gc-069                    | E expressive vocab  | −0.017                 | −0.079 to +0.044 | 0.328 | negative, inconclusive     | pass |
| gc-069                    | T receptive grammar | +0.114                 | +0.023 to +0.206 | 0.977 | positive, **strong**       | pass |
| gc-069                    | W word reading      | −0.047                 | −0.228 to +0.129 | 0.333 | negative, inconclusive     | pass |
| gc-069                    | L letter-sound      | −0.148                 | −0.332 to +0.042 | 0.107 | negative, suggestive       | pass |
| gc-070 (shared tempo)     | R receptive vocab   | −0.041                 | −0.104 to +0.020 | 0.138 | negative, suggestive       | pass |
| gc-070                    | E expressive vocab  | −0.024                 | −0.092 to +0.042 | 0.281 | negative, inconclusive     | pass |
| gc-070                    | T receptive grammar | +0.111                 | +0.015 to +0.206 | 0.966 | positive, moderate         | pass |
| gc-070                    | W word reading      | −0.074                 | −0.269 to +0.111 | 0.261 | negative, inconclusive     | pass |
| gc-070                    | L letter-sound      | −0.149                 | −0.337 to +0.040 | 0.104 | negative, suggestive       | pass |
| gc-085 (age×ability)      | R receptive vocab   | −0.012                 | −0.072 to +0.047 | 0.370 | negative, inconclusive     | pass |
| gc-085                    | E expressive vocab  | +0.011                 | −0.053 to +0.076 | 0.612 | positive, inconclusive     | pass |
| gc-085                    | T receptive grammar | +0.155                 | +0.057 to +0.252 | 0.993 | positive, **very strong**  | pass |
| gc-085                    | W word reading      | +0.095                 | −0.069 to +0.256 | 0.826 | positive, suggestive       | pass |
| gc-085                    | L letter-sound      | −0.066                 | −0.264 to +0.132 | 0.299 | negative, inconclusive     | pass |

**What a `gamma` coefficient means, in words.** It is a between-child association: children who scored one standard deviation higher on the baseline block-design puzzle tend to have a logit growth _rate_ higher (positive) or lower (negative) by this amount, holding the model's other terms fixed. It is _not_ a lever — you cannot raise a child's growth rate by raising their block-design score.

**The one consistent rate signal is receptive grammar (T).** In all three models, higher baseline non-verbal ability goes with faster growth in receptive grammar: gc-069 median +0.114 logit (P = 0.977, strong), gc-070 +0.111 (P = 0.966, moderate), gc-085 +0.155 (P = 0.993, very strong). For the two broad vocabulary measures (R, E) and word reading (W) the rate association is **inconclusive** in the base models. For letter-sound knowledge (L) there is a _negative_-leaning association in gc-069/gc-070 (median ≈ −0.148, P(<0) ≈ 0.89, suggestive) — higher-ability children grow slightly _slower_ on letter sounds, most plausibly a ceiling/scaling artefact, because those same children start much higher on L (see `delta` below).

**gc-085 adds age, and the picture for word reading shifts.** gc-085 (`age_ability_interaction = True`) separates a child's _age_ effect on the growth rate from the ability effect and adds their product. Once age carries its own (decelerating) effect — the age-on-rate term `gamma_age` is negative for every skill, most sharply for word reading (posterior mean −0.396, 89% −0.541 to −0.256) — the residual ability→rate association for word reading turns mildly positive (+0.095, suggestive) and the letter-sound negative fades toward zero. The age×ability interaction itself (`gamma_int`, reported as posterior mean with 89% ETI) is near zero for R (+0.045, −0.022 to +0.112), E (+0.073, +0.002 to +0.145), T (+0.008) and W (+0.100, −0.075 to +0.278), and clearest for letter-sound L (+0.268, +0.043 to +0.494) — a weak hint that older-and-more-able children add a little to the letter-sound growth rate beyond age and ability acting separately. Treat this as one wide-interval coefficient among several, not a finding.

**Ability predicts _level_, robustly (`delta`).** The far more robust signal is on the trajectory's _level_ at the pooled mid-study age. Higher baseline non-verbal ability goes with a higher level on every skill: R median +0.187 (89% +0.110 to +0.262, very strong), E +0.223 (very strong), T +0.228 (very strong), W +0.224 (89% −0.079 to +0.519, suggestive) and L +0.136 (suggestive) in gc-069, with near-identical values in gc-070 and gc-085. Read together with `gamma`: **baseline non-verbal ability tracks where a child _is_, much more than how fast they are _moving_.**

**Every skill rises with age on average (`beta`, population slope).** All five population slopes are very strongly positive in all three models (e.g. gc-069: W +1.148, L +1.053, E +0.294, R +0.252, T +0.221 logit per +1 SD of age; all P ≈ 1.000). To anchor these in items, the fitted population trajectory (gc-069 `group_trajectory.csv`, marginal over children, no arm split) rises across the four waves from roughly: word reading 7.0 → 16.7 of 79; letter sounds 15.9 → 23.0 of 32; receptive vocab 36.5 → 45.4 of 170; expressive vocab 30.3 → 39.6 of 170; receptive grammar 13.2 → 15.2 of 32.

**The five skills share a common growth tempo (`loading`, gc-070 only).** gc-070 adds a single latent "developmental tempo" that all five growth rates load on. Every loading is very strongly positive (R +0.107, E +0.130, T +0.170, W +0.378, L +0.349; all P ≈ 1.000): a child who grows fast on one skill tends to grow fast on all. **But baseline non-verbal ability does not explain that tempo** — the post-hoc correlation between each child's latent tempo and their block-design score is median −0.016 (89% −0.237 to +0.205; P(>0) = 0.453, inconclusive). So the shared tempo is real but is not the ability proxy; consistent with `delta`/`gamma`, ability sorts children by level, not by tempo.

**Model comparison.** By PSIS-LOO (leave-one-out predictive accuracy) the three are close: gc-085 elpd −3129.1, gc-069 −3139.2, gc-070 −3141.5. gc-085 (adding age and its interaction) is marginally preferred, but the differences are small and none of this changes the qualitative reading.

> The `gamma`/`delta` coefficients are **adjusted associations, not causes.** Block design is an off-DAG, pre-randomisation ability proxy and latent general ability is the unobserved common cause of block design and every skill, so these couplings are confounded and not point-identified. The child random intercept only partially adjusts.

---

## Sub-group B — Historical growth (Byrne comparison cohort; associations)

**Headline: the natural-history growth of children with Down syndrome, and how it compares to typically-developing benchmarks, one standardised test per model.** All values are in the **test's own raw-score items**. Growth over an interval is computed only on children observed at both endpoint waves, so attrition cannot masquerade as growth. All nine pass the gate (0 divergences; max R-hat ≤ 1.005; min ESS ≥ 1,986). The audit checks (`posterior_cell_summary.csv`) confirm the fitted group-by-wave cells land on the published Table 2 means (the fitted-minus-observed gaps are small, typically well under half an item on the core window).

| Model  | Test (measure)                    | Down-syndrome own growth, items (window) | Avg readers − DS gap, items | P(>0) | Label       |
| ------ | --------------------------------- | ---------------------------------------- | --------------------------- | ----- | ----------- |
| hg-001 | BAS word reading (basread)        | +21.26 (w1→5) [18.37, 24.06]             | +18.63 [14.88, 22.33]       | 1.000 | very strong |
| hg-002 | BAS spelling (basspel)            | +3.88 (w1→5) [2.94, 4.81]                | +4.42 [3.19, 5.64]          | 1.000 | very strong |
| hg-003 | WORD reading comprehension (woco) | +4.51 (w1→5) [3.16, 5.91]                | +9.17 [7.28, 10.99]         | 1.000 | very strong |
| hg-004 | BPVS receptive vocabulary (bpvs)  | +3.49 (w1→5) [1.92, 5.06]                | +2.61 [0.62, 4.61]          | 0.982 | strong      |
| hg-005 | TROG receptive grammar (trog)     | +2.78 (w1→5) [1.61, 3.96]                | +1.25 [−0.09, 2.58]         | 0.932 | moderate    |
| hg-006 | BAS recall of digits (basdig)     | +2.79 (w1→5) [1.30, 4.28]                | +2.23 [0.28, 4.21]          | 0.966 | moderate    |
| hg-007 | BAS similarities (bassim)         | +2.90 (w1→5) [1.91, 3.89]                | +2.53 [1.15, 3.91]          | 0.998 | very strong |
| hg-008 | BAS number skills (basnum)        | +7.06 (w1→4) [5.10, 9.01]                | +8.49 [5.85, 11.17]         | 1.000 | very strong |
| hg-009 | BAS matrices (basmat)             | +0.97 (w3→5) [0.07, 1.90]                | +2.60 [0.96, 4.24]          | 0.994 | very strong |

_("Avg readers − DS gap" is the between-group difference in total growth over the common window every group supports — waves 1→4 for hg-001–008, waves 3→4 for hg-009. A positive value means the typically-developing average readers gained more items than the Down-syndrome group over the same span.)_

**What the historical comparison asks, in words.** Within this older published cohort, each group has its own fitted trajectory. The contrast asks whether children with Down syndrome accumulate skill at the same rate as typically-developing children over the study window. Because the groups are a cohort factor (not randomised), a gap describes _how the groups differ_ — it is a benchmark, not a treatment effect, and it says nothing about the RLI intervention.

**The consistent story: children with Down syndrome grow more slowly than both benchmarks on almost every measure.** The average-reader gap is positive and at least moderate for all nine tests, and largest for the skills where typically-developing children accelerate fastest: BAS word reading (+18.6 items), BAS number skills (+8.5) and WORD reading comprehension (+9.2). The reading-matched contrast (typically-developing children who _read_ at the Down-syndrome group's level at baseline but are younger) tells the same story — they also out-grow the Down-syndrome group: word reading +14.79 [9.70, 19.87] (very strong), spelling +4.52 [2.97, 6.04] (very strong), reading comprehension +6.34 [4.31, 8.34] (very strong), receptive vocab +3.22 [0.91, 5.51] (strong), grammar +3.16 [1.52, 4.77] (very strong), digits +1.90 [−0.43, 4.23] (moderate), similarities +2.84 [1.18, 4.49] (very strong), number +11.52 [8.36, 14.64] (very strong), matrices +3.30 [1.45, 5.15] (very strong). The reading-matched comparison is the more telling one: even against children reading at the _same starting level_, the Down-syndrome group's subsequent growth is slower — a slower developmental slope rather than merely a lower starting point.

**Two smaller between-benchmark contrasts worth naming.** For most measures the two typically-developing groups grow at similar rates (reading-matched − average-readers gaps are inconclusive). The exceptions: on **WORD reading comprehension** (hg-003) the reading-matched group grows _less_ than average readers (−2.82 items [−5.00, −0.64]; P(average > reading-matched) = 0.979, strong), and on **TROG grammar** (hg-005) the reading-matched group grows _more_ than average readers (+1.91 [0.46, 3.34]; P = 0.982, strong). These reflect the ages and starting points of the two comparison groups, not anything about Down syndrome.

**The Down-syndrome cohort does keep progressing.** Every within-group total is positive with high probability — over its full window the Down-syndrome group gains ~21 BAS word-reading points, ~7 number-skills points, ~3–4 points on spelling, comprehension, vocabulary, grammar, digits and similarities, and ~1 matrices point. The wave-by-wave rows show the growth is uneven (e.g. hg-001 word reading gains most from wave 3→4, +8.6 items, and slows to +2.0 by wave 4→5), but the direction is consistently upward. The point of the comparison is the _rate relative to typically-developing peers_, which is slower — not an absence of progress.

---

## Sub-group C — Historical joint (Byrne cohort; associations)

**jc-001 — Byrne joint correlated growth: word reading, receptive vocabulary and digit recall (waves 1–4 + DS wave 5).** Gate: pass (0 divergences; max R-hat 1.003; min ESS 3,085). This model has no PSIS-LOO by design (one likelihood node per measure, so a single pointwise LOO is undefined) — the posterior-predictive checks are its fit diagnostics.

**Headline: how strongly do children's _stable levels_ on the three skills move together?** The model gives each child a stable per-measure offset (their persistent level, net of their group's wave-by-wave trajectory) and correlates those offsets across measures. The correlations are shared across the three reading groups.

| Measure pair                        | Correlation (median) | 89% CI         | P(>0)  | Label       |
| ----------------------------------- | -------------------- | -------------- | ------ | ----------- |
| Word reading ↔ receptive vocabulary | 0.688                | 0.528 to 0.811 | 1.000  | very strong |
| Word reading ↔ digit recall         | 0.650                | 0.497 to 0.766 | 1.000  | very strong |
| Receptive vocabulary ↔ digit recall | 0.538                | 0.322 to 0.711 | 0.9999 | very strong |

**In words.** A child who sits persistently higher on one of these three skills tends to sit higher on the others — strongly so between word reading and both vocabulary and digit recall, and moderately-to-strongly between vocabulary and digit recall. Crucially this is a **between-child** association (children who are high on X tend to be high on Y), _not_ a within-child coupling (it does not say that improving one skill improves another). Residual confounding by general ability is exactly what one would expect to produce correlations of this kind.

**Consistency check.** jc-001's per-measure growth reproduces the single-measure fits: its Down-syndrome word-reading growth (w1→5 +21.33 [18.44, 24.24]) matches hg-001 (+21.26), its receptive-vocab growth (+3.33 [1.71, 4.96]) matches hg-004 (+3.49), and its digit-recall growth (+2.77 [1.25, 4.29]) matches hg-006 (+2.79). The average-reader-minus-Down-syndrome gaps likewise match (word reading +18.61 vs +18.63; vocab +2.51 vs +2.61; digits +2.24 vs +2.23). Fitting the three jointly does not move the marginal growth story; it adds the cross-measure correlation structure.

---

## What the family concludes

Two coherent, non-causal pictures, on two different samples.

**In the RLI cohort (growth-coupling),** baseline non-verbal ability sorts children by _level_ far more than by _rate_: the level associations (`delta`) are very strong on the broad language measures, while the rate associations (`gamma`) are inconclusive for vocabulary and word reading and clear only for receptive grammar. The five skills do grow on a shared developmental tempo, but that tempo is not the ability proxy — children who start abler are not, net of everything, growing systematically faster. This is the associational backdrop to the causal ITT and gain-factor families: those estimate what _assignment_ to the intervention changes; this family describes the between-child developmental structure the intervention operates within.

**In the historical Byrne cohort,** children with Down syndrome grow more slowly than typically-developing children on essentially every standardised measure — and, tellingly, more slowly even than younger children _matched to their reading level_ at baseline. This is the natural-history benchmark: it quantifies the developmental gap the intervention is trying to narrow, on an independent published sample, and the audit confirms the reproduction lands on the source's published means. The joint model adds that, across children, word reading, vocabulary and digit recall levels are strongly correlated — the skills co-vary between children, consistent with a shared general-ability influence.

Together the family says nothing about _cause_ and everything about _structure_: who progresses together, and how far a Down-syndrome cohort's trajectory sits below typically-developing peers.

## Caveats & convergence

- **All 13 models pass the convergence gate** — zero divergences, max R-hat ≤ 1.005, min ESS ≥ 1,986 across the family. No held coefficients, no funnel flags.
- **Associations only.** No quantity here is causal. For the growth-coupling models the block-design coefficients are confounded ability proxies (the Table-2 fallacy warning applies); for the historical models the group is a cohort label. Adjustment sets are DAG-pre-specified, so skills absent from a model were excluded by the diagram, not found unimportant.
- **Small samples, wide intervals.** The growth-coupling models rest on ~54 children; several historical group-by-interval growth estimates rest on 11–24 children (and fewer at the attrition-selected extension waves — hg-009 matrices covers only waves 3–5, with the Down-syndrome wave-3→5 total on 17 children). Lead with the intervals.
- **Two samples, do not conflate.** Sub-group A is the RLI intervention cohort and its own measures (ROWPVT/EOWPVT/TROG-2/EWRSWR/YARC); sub-groups B and C are the older published Byrne comparison cohort and its standardised tests (BAS/WORD/BPVS/TROG). The item scales are not interchangeable, and the RLI item maxima do not apply to the historical tests.
- **Extension-wave tails.** Historical growth beyond the complete-case core window uses attrition-selected follow-up cells; every interval is computed on children present at both endpoints, but the extension rows should be read as within-cohort follow-up, not as the audited core.
- **Logit-scale coefficients (sub-group A).** The `gamma`/`delta`/`beta`/`loading` values are log-odds; a slope-modifier like `gamma` has no clean single "items per SD" translation, so it is reported on the logit scale and anchored with the items-scale population trajectory rather than converted.
