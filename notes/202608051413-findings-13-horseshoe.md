<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Findings 13 — the horseshoe ranking family

Reports every model in the `horseshoe` family from the 2026-08-04/05 `reporting` refit. **5 models, all passing the convergence gate.** Reading conventions in note 00. Preliminary research data — all estimates provisional.

## What these models do

A **Bayesian cross-check on the gradient-boosting predictor ranking** from step 1 of the study's methodology. The regularised horseshoe is a shrinkage prior that pulls weak coefficients hard toward zero while leaving genuinely large ones almost untouched — so it acts as a principled variable selector rather than reporting a coefficient for everything.

**The reported quantity is `p_abs_gt_delta`**: the posterior probability that a predictor's standardised coefficient exceeds a small threshold in absolute value. Read it as "how confident are we this predictor is doing real work?" — high means selected, low means shrunk away.

**Associations only.** No causal content. The horseshoe selects predictive relevance, not mechanism.

## The result that matters: levels are predictable, gains are not

This family's most useful output is not any individual ranking — it is the contrast between the _level_ models and the _gain_ models.

| Model    | Target                 | Top predictor         | `p_abs_gt_delta` |
| -------- | ---------------------- | --------------------- | ---------------: |
| `hs-004` | Letter-sound **level** | Word reading          |        **1.000** |
|          |                        | Expressive vocabulary |        **0.999** |
| `hs-002` | Word-reading **level** | Letter sounds         |        **0.995** |
|          |                        | Expressive vocabulary |        **0.993** |
|          |                        | Receptive grammar     |            0.884 |
| `hs-001` | Word-reading **gain**  | Age                   |        **0.588** |
|          |                        | Letter sounds         |            0.427 |
|          |                        | Behaviour             |            0.383 |
| `hs-003` | Letter-sound **gain**  | Receptive grammar     |        **0.451** |
|          |                        | Basic concepts        |            0.288 |
|          |                        | Word reading          |            0.279 |

**For levels, selection is decisive.** Where a child stands on word reading is predicted near-certainly by where they stand on letter sounds and expressive vocabulary (both ≥ 0.99), and vice versa. The model has no difficulty identifying which predictors matter.

**For gains, the horseshoe selects nothing.** The highest selection probability across both gain models is 0.59 (age, for word-reading gain) — below any reasonable selection threshold — and every other predictor in the two models sits between 0.10 and 0.45. On this evidence, baseline characteristics have **little predictive purchase on how much a child gains**, even though they predict where a child _is_ almost perfectly.

This is an important and easily-missed result. It corroborates the adjusted family (note 12) directly, and it is a caution against reading the level-based associations elsewhere in the suite as though they told us who will progress. They tell us who is ahead, which is a different and much easier question.

It also gives the ITT results their proper context: the intervention produced a detectable gain in a setting where **baseline characteristics predict gain barely at all**. The treatment signal is not competing with a strong prognostic signal; it is one of the few things in this study that moves the gain measures.

## `rlm-hs-001` — the historical Byrne cohort

| Predictor              | `p_abs_gt_delta` | Sign     |
| ---------------------- | ---------------: | -------- |
| Age                    |        **0.912** | negative |
| BAS recall of digits   |            0.350 | positive |
| BAS similarities       |            0.209 | positive |
| BAS number skills      |            0.200 | positive |
| TROG receptive grammar |            0.174 | negative |

The same pattern in an independent cohort: **age is the only predictor the horseshoe comes close to selecting** (0.91, negative), and everything else is shrunk away. This replicates the adjusted family's Byrne result (note 12) with a completely different estimation approach — a shrinkage prior rather than an all-in regression — which makes the "age is the one real prognostic signal, and it is negative" conclusion considerably more robust.

## Caveats

- **`p_abs_gt_delta` is a selection probability, not an effect size.** A predictor can be selected with a small coefficient and vice versa; read the ranking with the coefficient beside it.
- **Shrinkage priors are informative by design.** High prior sensitivity in this family under power-scaling is expected — the whole point of a horseshoe is to move the posterior — and is not a defect (run record note).
- **No causal content.**
- **Horseshoe rankings are zero-divergence-only** under the divergence policy; `hs-001` needed `target_accept` lifted from its family's 0.99 to 0.999 to reach that, with the ranking unmoved (identical rank order, max change in `p_abs_gt_delta` 0.008).
- **Predictive calibration.** 50% bands cover about 72% of observations.
