<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# The four-child production puzzle: a speech case review

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8). Figures recomputed from `data/rli_data_long.csv`; this is a case review at n = 4, not a model — read it as hypothesis-generating, not as evidence.

Date: 2026-07-13 — relates to issue #230 §5, companion to `notes/202607131700-persistent-floor-sitters-nonword-spelling.md`.

## The pattern

The descriptive floor-sitter cut turned up four children who, at the final wave, have near-complete letter-sound knowledge (≥ 26 of 32) and some blending and word reading, yet score **zero** on nonword reading. One of them spells 53 words phonetically while reading no nonwords aloud. The question raised was whether the nonword task's demand — reading an unfamiliar string _aloud_ — is a speech-production bottleneck rather than a gap in decoding knowledge. Bringing the speech battery to bear (DEAP articulation composite `deapp_c`, Early Repetition Battery nonword `erbnw` and word `erbword` repetition; cohort t4 medians `deapp_c` ≈ 240, `erbnw` = 11, `erbword` = 14) shows the four are **not one story**:

| subject_id          | letter sounds | blending | word reading | nonword | phonetic spelling | `deapp_c` | `erbnw` | `erbword` | age began speaking |
| ------------------- | ------------- | -------- | ------------ | ------- | ----------------- | --------- | ------- | --------- | ------------------ |
| ID_0D23DC86A37B807D | 31            | 10       | 21           | 0       | 53                | 254       | 16      | 18        | 30                 |
| ID_43A40173C81CBD21 | 28            | 9        | 12           | 0       | 0                 | 216       | 10      | 12        | 40                 |
| ID_BB12CED680895743 | 27            | 10       | 9            | 0       | 44                | 232       | 14      | 16        | 24                 |
| ID_C722B3B63C6E496F | 26            | 5        | 5–9          | 0       | 0                 | 199       | 6       | 6         | —                  |

## What the speech data says

- **The speech-production explanation does not fit the strongest case.** `ID_0D23DC86A37B807D` has above-cohort articulation (`deapp_c` 254) and above-cohort repetition (`erbnw` 16, `erbword` 18) and spells 53 words phonetically — so this child can both produce speech and map sounds to letters, yet reads no nonwords aloud. For this child a speech-production limit is the _least_ likely account; the zero looks more like a task- or decoding-application boundary (blending and phonetic spelling are intact, but the specific act of sounding out an unfamiliar written string and blending it aloud has not transferred).
- **It fits the weakest case better.** `ID_C722B3B63C6E496F` has below-cohort articulation (`deapp_c` 199) and low repetition (`erbnw`/`erbword` = 6), the lowest blending of the four (5), and no phonetic spelling — a profile where a speech-production and phonological-processing limit is a plausible contributor, tangled with an incomplete blending prerequisite.
- **The middle two are ambiguous.** `BB12` spells phonetically (44) with solid repetition; `43A4` is a late talker (began speaking at 40 months) with no phonetic spelling. Neither cleanly implicates or exonerates speech production.

## Reading

Across four children the "nonword reading is gated by speech production" hypothesis is **partly supported at most, and contradicted for the clearest case**. What the four share is not a speech deficit but the last, hardest step of decoding: applying letter-sound and blending knowledge to _read an unfamiliar string aloud_, a step that here lags phonetic spelling (encode) and sight-word reading. This is consistent with the descriptive finding that floor status tracks decoding-path position, not inability. The honest next step is not a model on n = 4 but to (a) keep the speech battery (`deapp*`, `erb*`, and the t1–t2 language-sample `lsam*`) in view as candidate covariates for the floor-sitter survival family (`lrp-rli-surv-011`), after the missingness audit issue #230 §3 calls for, and (b) flag the nonword task's aloud-reading demand as a measurement caveat when reporting the nonword floor. `lsam*` intelligibility is unavailable at t4 (language sampling stopped after t2), so any speech-production follow-up must lean on the DEAP and repetition measures.
