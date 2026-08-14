<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- cspell:ignore basnum basspel -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# BAS spelling and number-skills primary-source correction

**Status: corrected evidence audit for #338 and #409, 2026-08-14.** This note supersedes the earlier inference that the Byrne cohort used a 1992 stand-alone spelling scale and that the prepared `basnum` column had an unidentified score transformation. The correct primary source is Byrne, MacDonald and Buckley (2002), not an instrument-description thesis or the 1995 companion paper.

## Evidence

The Methods section of the primary article states that all children were assessed with subtests from the British Ability Scales (Elliott, 1983) and explicitly lists Word Reading, Spelling, Recall of Digits, Immediate and Delayed Visual Recall, and Similarities. It separately identifies WORD reading comprehension, BPVS receptive vocabulary and TROG receptive grammar. The procedure includes number in the assessment order, Table 3 labels the measure “BAS number skills”, and the Results state that all analyses used raw scores (Byrne, MacDonald & Buckley, 2002, pp. 517–519, DOI [10.1348/00070990260377497](https://doi.org/10.1348/00070990260377497)).

The prepared extract independently matches that publication record. Restricting `basnum` to the paper's first three waves gives the following baseline means:

- Average readers observed at all three waves: 35.290, printed as 35.3 in Table 3.
- Average-reader non-completers with a baseline value: 37.600, printed as 37.6.
- Reading-matched children observed at all three waves: 25.857, printed as 25.9.
- Reading-matched non-completers with a baseline value: 31.429, printed as 31.4.

The exact reproduction makes a cohort-specific transformation error unlikely and confirms that the prepared column is the BAS number-skills raw score analysed in the primary paper. Descriptions of other BAS forms do not identify which form this cohort received and cannot override the cohort's own source article.

## Decision

Confirm the source identities of both measures: `basspel` is the Spelling subtest in the administered 1983 BAS battery, and `basnum` is the BAS number-skills raw score reported by Byrne et al. (2002). Remove the separate score-definition blocker proposed for `basnum`; the existing instrument-ceiling gate is the correct representation of the remaining uncertainty.

Keep `n_trials=18` for `basspel` and `n_trials=60` for `basnum` only as operational observed-maximum placeholders, with `n_trials_confirmed=False`. The primary paper does not give the item counts or maximum raw scores, so it cannot validate either Beta-Binomial denominator. Every affected fit remains withheld by the central denominator gate and by the unresolved 96-versus-97 extract lineage.

The directly affected registered models remain `lrp-rlm-hg-002`, `lrp-rlm-hg-008`, `lrp-rlm-mm-001`, `lrp-rlm-adj-001` and `lrp-rlm-hs-001`. Future models using either provisional measure inherit the same denominator blocker.

## Resolution needed

Obtain the administered 1983 BAS manual or cohort test record and identify the maximum raw score for Spelling and Number Skills. If each is a bounded item count, record the defensible denominator and refit every dependent model. If either raw score is not a simple item count, replace its count-scale preprocessing or outcome likelihood with a distribution appropriate to the documented scoring rule before refitting. The primary paper resolves source identity; it does not resolve the ceiling.
