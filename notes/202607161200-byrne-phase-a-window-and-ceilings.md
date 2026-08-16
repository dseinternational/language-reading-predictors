<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- cspell:ignore Byrne MacDonald Buckley readgrp basread basspel bpvs trog woco basdig bassim basnum basmat rlm dagitty Pintilie Golombok Trickey Whetton BJEP prov -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5).
>
> Source identification and the later `basmat` caveat corrected by a LLM-based AI tool (Codex/GPT-5).

# Byrne suite: Phase A window extension, group-indexed random-effect scales, and the instrument-ceilings sign-off

**Status: decisions record (issue #338), 2026-07-16.** This note records the data-owner decisions that resolve four of the gating items on #338 — the Phase A wave window, random-effects heterogeneity, group scope, and the instrument ceilings — and the design rationale behind the window extension implemented alongside it (`lrp-rlm-hg-001…009`). The companion planning documents are `notes/202607131500-byrne-dag-proposal.md` and `notes/202607131600-byrne-comparable-models-plan.md`.

## Decisions (data owner, 2026-07-16)

1. **Group scope.** All models keep the **three-group fit** over the common window (the extra groups stabilise the shared scales), with **Down syndrome vs reading-matched** reported as the headline developmental contrast and average readers as context. Wave 5 extends the Down-syndrome trajectory only.
2. **Random-effects heterogeneity (follow-up-plan decision 7).** Both the subject-intercept SD (`sigma_subject`) **and** the Beta-Binomial overdispersion (`kappa`) are **indexed by group**, each under the shared HalfNormal priors (0.5 / 50). Heterogeneous between-child variance is exactly where the Down-syndrome vs reading-matched contrast can diverge from a pooled model; indexing was applied before the window extension so it propagates to every per-measure fit.
3. **Phase A window.** Extended from the shipped waves 1–3 to **waves 1–4 plus a Down-syndrome-only wave-5 tail** (design below); `basnum` ends at wave 4 (not assessed at wave 5) and `basmat` (wave-3+ only) gets its own model, `lrp-rlm-hg-009`, on a waves 3–4 core with the wave-5 tail.
4. **Reading-matched selection.** Confirmed standing treatment: `basread` is a selection variable wherever group 3 enters; the per-model reports carry the descriptive-only framing and the catalogue entry carries the selection caveat.
5. **Instrument ceilings** — researched against published sources and signed off; see the table below.

## The window design: complete-case core + extension waves

The obvious extension — moving the complete-case rule to waves 1–4 — was rejected for two reasons measured directly from the extract: it shrinks the reading-matched cells to n = 11–14 (from 18–21), and it silently **breaks the Table 2 audit anchor**, because the waves 1–3 cells would then be computed on the waves-1–4-complete subset rather than the subset the paper's Table 2 reproduction uses. Requiring Down-syndrome children to be complete to wave 5 was rejected for the same reason (n drops to ~17 and the between-group window weakens).

The implemented design keeps both properties:

- The **complete-case core stays waves 1–3** (waves 3–4 for `basmat`) — the audit-anchored subset, unchanged from the shipped fits, so the observed core cells still reproduce Table 2 exactly.
- **Waves 4 and 5 enter as extension waves**: children in the core contribute wherever the measure was observed. Extension cells are an attrition-selected follow-up tail and always carry their own per-cell `n` in the reports. A child not complete on the core contributes nothing (their later waves are discarded with them) — the extension follows the audited cohort, it does not re-open the sample.
- The population level is parameterised over **supported (group, wave) cells only** (`eta_cell`), so the Down-syndrome-only wave 5 adds no prior-only parameters for the other groups and the convergence gate never sees an unidentified cell.
- **Interval growth is summarised on the children observed at both endpoint waves** (`n_subjects` in `posterior_growth_summary.csv`), so the stable per-child offsets cancel exactly in every growth quantity and attrition at an extension wave cannot masquerade as growth. On the core window this restriction is a no-op.
- **Between-group total-growth contrasts run over the common window** (waves 1–4; waves 3–4 for `basmat`) so the groups are compared on like horizons; wave 5 quantities are within-Down-syndrome follow-up only.

Complete-case core sizes are unchanged from the shipped fits (e.g. `basread` 23/32/21); the wave-4 extension cells run 11–20 per group and the Down-syndrome wave-5 cells 14–17, all reported per cell.

## Instrument ceilings (researched 2026-07-16, data-owner sign-off)

The battery, editions and use of raw scores are confirmed by the cohort's primary paper (Byrne, MacDonald & Buckley, 2002, _British Journal of Educational Psychology_ 72, 513–529, <https://doi.org/10.1348/00070990260377497>): BAS first edition (Elliott, 1983), TROG (Bishop, 1983), BPVS (Dunn, 1982) and WORD (Rust, Golombok & Trickey, 1993). The primary paper does not state instrument item counts; those require the corresponding manuals or other instrument-specific evidence.

| Measure   | Old (provisional)  | New        | Basis                                                                                                                                                                                                                                                |
| --------- | ------------------ | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `basread` | 87 (mis-confirmed) | **90** ✓   | BAS Word Reading has 90 words (Beech, 2004, _Reading Psychology_, DOI 10.1080/02702710490271819). The previous 87 was the observed extract maximum.                                                                                                  |
| `trog`    | 20                 | **20** ✓   | 80 items in 20 blocks of 4, scored as blocks passed (Bishop's original TROG manual, OSF <https://osf.io/z8wbs/>). The cohort reaches this ceiling.                                                                                                   |
| `basdig`  | 34                 | **34** ✓   | 34 items (CLOSER cognitive-measures guide; Parsons, 2014, CLS working paper). The cohort reaches this ceiling.                                                                                                                                       |
| `bassim`  | 18                 | **21** ✓   | 21 items (CLOSER guide; Parsons, 2014). Counts documented for the 1978/79 scale; 1983 was a re-norming.                                                                                                                                              |
| `basmat`  | 22                 | **28** ✓   | BAS Matrices, 28 items (CLOSER guide; Parsons, 2014). The retained cohort source names the later-wave fields `BASMAT3`-`BASMAT5`. Laws et al. (1995) used Raven's CPM in a separate memory-training cohort, so it creates no identity conflict here. |
| `bpvs`    | 29                 | **32** ✓   | BPVS **Short Form**, 32 items (Ripley & Yuill, 2005, _BJEP_, DOI 10.1348/000709904X22656): observed maxima to 29 across ages 11+ are only consistent with the short form.                                                                            |
| `basspel` | 18                 | 18 (prov.) | The primary paper explicitly identifies Spelling among the administered 1983 BAS subtests and analyses raw scores. Its item count is not reported, so 18 remains an observed-maximum placeholder.                                                    |
| `basnum`  | 60                 | 60 (prov.) | The primary paper labels BAS number-skills raw scores, and its Table 3 means reproduce exactly from the prepared extract. The administered form and maximum are not reported, so 60 remains an observed-maximum placeholder.                         |
| `woco`    | 31                 | 31 (prov.) | WORD Reading Comprehension count unverified (the parent WIAT subtest is commonly described as 38 items). Needs the 1993 WORD manual.                                                                                                                 |

✓ = `n_trials_confirmed=True` in `datasets.RLM_MEASURES`. Two consequences worth noting: `trog` (20/20) and `basdig` (34/34) sit **at** their true ceilings in the extract, so their Beta-Binomial upper bounds are real, not observed-max artefacts; and the `basread` correction (87 → 90) slightly shifts `lrp-rlm-hg-001`'s fitted probabilities relative to the shipped fit, which the next reporting sweep will absorb.

The three provisional measures keep their unconfirmed-ceiling report callouts, and the remaining manuals check is scoped to the administered 1983 BAS Spelling and Number Skills tests and the 1993 WORD Reading Comprehension subtest. The primary paper confirms the first two source identities but not their denominators.

## Addendum (same day) — Phase B/D decisions and models

The ceilings sign-off above unblocked Phases B and D, which were designed and fitted the same day with four further data-owner decisions:

1. **Phase D scope: pooled + nuisance.** `lrp-rlm-adj-001` and `lrp-rlm-hs-001` fit all three groups (n = 69) with two non-interpretable group-nuisance dummies (`Normal(0, 1)`, reference = average readers) rather than Down-syndrome-only (n = 21, where seven mutually-adjusted slopes would sit at the prior). A DS-only companion is deferred, not rejected.
2. **Phase D horizon: w1 → w3.** The outcome is wave-3 word reading given its wave-1 baseline — the audited core window (w1→w4 loses 14 children to attrition and ends on the extension tail). Predictors: standardised wave-1 Haldane logits of `bpvs`/`trog`/`basdig`/`bassim`/`basnum` plus age; `basmat` excluded (wave-3+ only), `basspel`/`woco` excluded as reading-route measures too close to the outcome.
3. **mm-001: wave 3, measurement-only.** The only wave carrying the full ability triad (n = 75). No structural leg — the factor→gain question is Phase D's. The **single-indicator memory domain** (`basdig`; the paper's visual-recall measures are absent from the retained source) takes a fixed reliability of 0.8 (`lambda = sqrt(0.8)`), stated in the report; loadings are pooled across groups (invariance assumed, not tested).
4. **jc-001 design.** Measures `basread` + `bpvs` + `basdig` (the reading-language-memory trio), the suite's waves-1–3 complete-case core jointly across the three measures (n = 71) plus the #338 extension waves; per-measure supported-cell grids with group-indexed scales; the per-child stable offsets correlated through one LKJ(2) matrix **shared across groups** (a stated parsimony assumption). Registered as the new `historical_joint` kind (family code `jc`). No PSIS-LOO — the model has one likelihood node per measure, so a single pointwise LOO is not defined; the per-measure `hg` fits carry LOO.

One id-scheme consequence: the `rlm` study's legacy ids now always embed their family code (`rlmadj01`, `rlmjc01`, ...) — the RLI bare aliases (`lrp65`) are historical and preserved unchanged.

## What this unblocks

The ceilings gate was the blocker on Phases B and D of the #338 plan (`lrp-rlm-jc-001`, `lrp-rlm-mm-001`, `lrp-rlm-adj-001`, `lrp-rlm-hs-001` and the concurrent ports `lrp-rlm-ca-001/002`). Six of nine measures are now confirmed; models over the confirmed set can proceed, and any model touching `basspel`/`basnum`/`woco` inherits the provisional flag. The remaining #338 gates are the Phase C lagged-DAG green-light (untouched here) and the manuals check for the provisional trio.
