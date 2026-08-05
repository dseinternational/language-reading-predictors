<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Findings 12 — the adjusted-association family

Reports every model in the `adjusted` family from the 2026-08-04/05 `reporting` refit. **2 models, both passing the convergence gate.** Reading conventions in note 00. Preliminary research data — all estimates provisional.

## What these models do

A single question asked cleanly: **which baseline characteristics predict how much a child gains, once the others are held?** One model for the RLI trial cohort, one for the historical Byrne cohort.

**Design.** Beta-Binomial ANCOVA of the post-score on its own baseline, with every candidate baseline predictor entered together. Results are reported as the **change in gain, in words, associated with a +1 SD difference in the predictor** — which is far more readable than a logit coefficient and is directly comparable across predictors within a model.

**Every coefficient is an adjusted association.** Nothing here is randomised. These describe _who_ progresses, not _what to do_, and reading them as levers is the Table-2 fallacy.

## `adj-065` — the RLI trial cohort (word-reading gain)

Effect of a +1 SD difference in each baseline predictor on word-reading gain, in words.

| Predictor                 | Effect (89%)                  | P(>0) | Reading                                        |
| ------------------------- | ----------------------------- | ----: | ---------------------------------------------- |
| **Age**                   | **−2.9 words** (−4.3 to −1.3) | 0.003 | very strong evidence of a negative association |
| **Hearing status**        | **+2.4 words** (+0.5 to +4.5) |  0.98 | strong                                         |
| Language composite        | +1.7 words (−1.0 to +4.9)     |  0.84 | suggestive                                     |
| Letter sounds             | +1.7 words (−0.5 to +4.3)     |  0.89 | suggestive                                     |
| Behaviour                 | −1.7 words (−3.3 to +0.0)     |  0.06 | moderate, negative                             |
| Speech-missing indicator  | −2.0 words (−4.0 to +0.0)     |  0.06 | moderate, negative                             |
| Non-verbal mental age     | +0.5 words (−1.5 to +2.6)     |  0.65 | inconclusive                                   |
| Hearing-missing indicator | +0.4 words (−1.3 to +2.2)     |  0.63 | inconclusive                                   |
| Speech production         | +0.5 words (−1.9 to +3.5)     |  0.63 | inconclusive                                   |
| Blending                  | +0.2 words (−1.4 to +2.1)     |  0.59 | inconclusive                                   |
| Phonological memory       | −0.1 words (−2.4 to +2.7)     |  0.47 | inconclusive                                   |

**Age dominates**, and it is the only predictor with a decisively resolved association: a child one SD older gained about 3 fewer words. This is the same signal that appears in the gain-factor family (note 03) and the LCSM family (note 10), and it survived a dedicated test that it is not a difficulty-ladder or likelihood artefact. The interpretive limit is unchanged and important: this cannot separate developmental timing from trajectory selection.

**Hearing status is the surprise.** Better hearing at baseline is associated with about 2.4 more words gained (P = 0.98). It is the second-best-resolved predictor in the model and is not one the study's causal diagram foregrounds. Worth noting rather than over-reading — hearing is measured coarsely and the missingness indicator carries its own coefficient — but it is a real signal in these data.

**Non-verbal mental age is essentially uninformative** (+0.5 words, P = 0.65), which is worth stating because it is the predictor most people expect to dominate. Once age and the language/reading baselines are held, general non-verbal ability adds little to predicting _gain_.

Of the two "missing" indicators, one carries a real coefficient: speech-missing at −2.0 words (P = 0.06, moderate evidence of a negative association), where hearing-missing is flat. Missingness on the speech measure is informative, which is a caution against treating the imputation as neutral.

## `rlm-adj-001` — the historical Byrne cohort (word-reading gain over waves 1–3)

A different cohort, different instruments, no intervention — natural history rather than trial.

| Predictor                           | Effect (89%)                   | P(>0) |
| ----------------------------------- | ------------------------------ | ----: |
| **Age (months)**                    | **−8.4 words** (−12.5 to −4.0) | 0.000 |
| BAS recall of digits                | +4.6 words (−0.9 to +10.1)     |  0.91 |
| BAS similarities / verbal reasoning | +2.2 words (−3.1 to +7.5)      |  0.74 |
| BAS number skills                   | +0.4 words (−6.0 to +6.7)      |  0.54 |
| BPVS receptive vocabulary           | +0.1 words (−4.5 to +4.8)      |  0.52 |
| TROG receptive grammar              | −2.1 words (−7.3 to +3.2)      |  0.26 |

**The age finding replicates in an independent, non-randomised cohort**, and more strongly: −8.4 words per SD of age, decisively resolved. That the same negative age–gain association appears in a separate cohort with different tests is the single most useful thing this model contributes, because it argues against the RLI result being an artefact of that particular sample or instrument set.

**Verbal short-term memory (recall of digits) is the best-resolved skill predictor** (+4.6 words, P = 0.91, moderate). Receptive vocabulary and grammar are flat, and grammar leans mildly negative without resolving.

## Reading the two together

Both cohorts agree that **age at baseline is the dominant predictor of subsequent word-reading gain, negatively**. Beyond age, the two disagree about which skills matter — letter sounds and language in the RLI cohort, digit recall in the Byrne cohort — but every one of those is only suggestive-to-moderate, so the disagreement is within noise rather than a substantive conflict.

The honest summary is that **baseline characteristics predict gain poorly**. This is corroborated directly by the horseshoe family (note 13), whose shrinkage prior selects essentially nothing for gain outcomes while selecting decisively for level outcomes.

## Caveats

- **Associations, not levers.** No causal reading is available for any coefficient.
- **The age finding cannot separate timing from selection.**
- **Different instruments** in the two cohorts; the "words" scales are not the same test, so compare directions and relative ordering, not magnitudes.
- **Missingness is informative** in `adj-065`.
- **Predictive calibration.** 50% bands cover about 67% of observations — among the best-calibrated families, consistent with their one-row-per-child cross-sectional structure.
