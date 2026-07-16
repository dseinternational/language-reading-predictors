# Findings — horseshoe family (regularised predictor-ranking cross-check)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8). Numbers from the `reporting`-config refit of 2026-07-16 (see [process note](202607161130-full-statistical-refit.md)). Preliminary.

## What these models ask

A **regularised horseshoe** is a Bayesian variable-selection prior: it shrinks weak predictors hard toward zero while letting genuinely strong ones stand out, then **ranks** them. These models are a **cross-check on the gradient-boosting (Step 1) importance ranking** — a second opinion on "which measured features are most associated with the outcome", using a completely different method. The ranking is an **adjusted predictive sensitivity check, not a list of causal drivers**, and closely-ranked predictors should not be treated as meaningfully ordered.

## Convergence gate

4 of 5 **passed**. `hs-001` (word-reading gain) is flagged for **2 divergences (0.006%)** only — usable with a caveat.

## Results — top-ranked predictor per model

| Model      | Target                  | Top predictor         | Standardised association         | P(> threshold) |
| ---------- | ----------------------- | --------------------- | -------------------------------- | -------------- |
| hs-004     | Letter-sound **level**  | **Word reading (W)**  | +0.70 logit (HDI +0.49 to +0.91) | 99.9%          |
| hs-002     | Word-reading **level**  | **Letter sounds (L)** | +0.33 logit (+0.16 to +0.50)     | 99%            |
| hs-003     | Letter-sound **gain**   | Grammar (T)           | +0.13 logit (−0.06 to +0.43)     | 45% (weak)     |
| hs-001     | Word-reading **gain**   | (flagged; see report) | —                                | —              |
| rlm-hs-001 | Byrne word-reading gain | Age                   | −0.34 logit (−0.59 to +0.01)     | 91%            |

## The one-paragraph story

For **levels**, the horseshoe cleanly picks out the obvious reading-system partners — **word reading ↔ letter sounds** top each other's rankings with high confidence, agreeing with the mechanism and concurrent analyses. For **gains**, no single predictor stands out strongly (the top gain predictors sit below the "worth noticing" threshold), which is the honest result at this sample size: gain is hard to predict from baseline features. In the Byrne cohort, age again leads (negatively), echoing the [adjusted](202607161800-findings-adjusted.md) family.

## What is causal

**Nothing.** A ranking is a predictive/associational summary. Its value is as a _convergent_ check: when a very different method (horseshoe) highlights the same features as gradient boosting, we trust the description more — but neither identifies a cause.
