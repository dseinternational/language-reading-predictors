# Findings — horseshoe family (regularised predictor-ranking cross-check)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8). Numbers from the `reporting`-config refit of 2026-07-16 (see [process note](202607161130-full-statistical-refit.md)); reviewed and extended on 2026-07-17 to cover all models in the family. Preliminary.

## What these models ask

A **regularised horseshoe** is a Bayesian variable-selection prior: it shrinks weak predictors hard toward zero while letting genuinely strong ones stand out, then **ranks** them. These five models are a **cross-check on the gradient-boosting (Step 1) importance ranking** (the reference fits are the matching GB models, e.g. `lrp-rli-gbg-012` for word-reading gain) — a second opinion on "which measured features are most associated with the outcome", obtained by a completely different method. The ranking is an **adjusted predictive sensitivity check, not a list of causal drivers**, and closely-ranked predictors should not be treated as meaningfully ordered.

Every coefficient below is a **standardised logit slope** (the change in the log-odds of the bounded outcome per one-standard-deviation change in the predictor, all predictors and outcome on a common scale), and it is an **adjusted association**: latent general ability confounds these fits, so no number licenses a causal reading. In the RLI models the general-ability measure (block design, `blocks`) and a behaviour rating (`behav`) are entered as pre-covariates, so the other coefficients are already adjusted for those.

Two reading conventions matter for the numbers, and both were previously left implicit:

- **Intervals are 94% credible ranges** (the fits use `ci_prob = 0.94`, HDI from the 3rd to the 97th posterior percentile), **not** 95%. A 94% credible range is the interval the true standardised slope lies within with 94% probability given the data and priors — a direct probability statement about the parameter, unlike a frequentist confidence interval. Where you would normally expect a 95% range in this project, read these as very slightly narrower.
- **The ranking metric is `P(|beta| > 0.1)`**, the posterior probability that the coefficient's magnitude exceeds a **"worth-noticing" threshold of delta = 0.1 logit**. This is a **magnitude / region-of-practical-equivalence (ROPE) exceedance probability** — the chance the effect is big enough to matter — and **not** a direction probability. It does **not** say "99% sure it helps"; it says "99% sure the association, in whichever direction, is larger than a trivial 0.1-logit band around zero". Because the ranking is by this exceedance probability, the project's **evidence ladder (a direction-probability grading) does not strictly apply** to these numbers; the `sign` column and whether the credible range excludes zero carry the direction information instead.

## Convergence gate

**4 of 5 passed.** `lrp-rli-hs-001` (word-reading gain) is the sole gate failure, flagged for **2 divergences (2 of 36,000 draws, 0.006%)**; R-hat, ESS and BFMI all pass, so its ranking is reported below but treated as usable-with-a-caveat rather than clean. The other four pass with **0 divergences**, max R-hat <= 1.0018 and min ESS >= 4221.

## Results — all models

For each model the top predictors are shown with the standardised logit slope (posterior mean, the best estimate), the 94% credible range, the sign, and `P(|beta| > 0.1)` — the "big enough to matter" exceedance probability. A full ranking beyond the rows shown adds only predictors with even smaller exceedance probabilities. **None of these coefficients is causal**; all are adjusted, latent-ability-confounded associations.

### `lrp-rli-hs-002` — word-reading **level** (RLI cohort) · association

Concurrent reading-system partners top the list. Both leaders have credible ranges wholly above zero, so their direction is effectively certain; the ranking-by-exceedance ordering puts letter sounds first even though expressive vocabulary is the larger slope.

| Rank | Predictor            | Standardised slope | 94% credible range | Sign | P(\|beta\|>0.1) — matters?                |
| ---- | -------------------- | ------------------ | ------------------ | ---- | ----------------------------------------- |
| 1    | Letter sounds (L)    | +0.33 logit        | +0.16 to +0.50     | +    | 0.995 — clears                            |
| 2    | Expressive vocab (E) | +0.41 logit        | +0.18 to +0.64     | +    | 0.992 — clears (larger slope than rank 1) |
| 3    | Grammar / TROG (T)   | +0.19 logit        | +0.05 to +0.32     | +    | 0.886 — likely                            |
| 4    | Age                  | +0.19 logit        | −0.03 to +0.42     | +    | 0.745 — plausible                         |
| 5    | Receptive vocab (R)  | +0.15 logit        | −0.02 to +0.35     | +    | 0.670 — plausible                         |

Below rank 5: block-design ability B +0.12 (0.629), F +0.06 (0.261).

### `lrp-rli-hs-004` — letter-sound **level** (RLI cohort) · association

The mirror image of hs-002: word reading and expressive vocabulary dominate, both with credible ranges wholly positive (direction effectively certain).

| Rank | Predictor                | Standardised slope | 94% credible range | Sign | P(\|beta\|>0.1) — matters?       |
| ---- | ------------------------ | ------------------ | ------------------ | ---- | -------------------------------- |
| 1    | Word reading (W)         | +0.70 logit        | +0.49 to +0.91     | +    | 1.00 — clears (renders as 99.9%) |
| 2    | Expressive vocab (E)     | +0.47 logit        | +0.26 to +0.69     | +    | 0.998 — clears                   |
| 3    | Block-design ability (B) | +0.06 logit        | −0.05 to +0.21     | +    | 0.285 — below threshold          |
| 4    | F                        | +0.06 logit        | −0.06 to +0.21     | +    | 0.259 — below threshold          |
| 5    | Age                      | +0.03 logit        | −0.10 to +0.19     | +    | 0.176 — below threshold          |

Below rank 5: R +0.01 (0.113), T −0.01 (0.071).

### `lrp-rli-hs-001` — word-reading **gain** (RLI cohort) · association · GATE-FAILED (2 divergences)

Reported for completeness with the divergence caveat. **No predictor clears the 0.1-logit threshold**; every credible range spans zero, so no direction is established. Gain is essentially unpredictable from baseline features at this sample size.

| Rank | Predictor           | Standardised slope | 94% credible range | Sign | P(\|beta\|>0.1) — matters? |
| ---- | ------------------- | ------------------ | ------------------ | ---- | -------------------------- |
| 1    | Age                 | −0.15 logit        | −0.38 to +0.03     | −    | 0.582 — below threshold    |
| 2    | Letter sounds (L)   | +0.11 logit        | −0.06 to +0.39     | +    | 0.429 — below threshold    |
| 3    | Behaviour (behav)   | −0.09 logit        | −0.34 to +0.05     | −    | 0.380 — below threshold    |
| 4    | F                   | +0.08 logit        | −0.05 to +0.29     | +    | 0.331 — below threshold    |
| 5    | Receptive vocab (R) | +0.04 logit        | −0.12 to +0.29     | +    | 0.224 — below threshold    |

Below rank 5: T +0.04 (0.209), E +0.01 (0.160), block-design B +0.02 (0.132), B(item) +0.01 (0.103).

### `lrp-rli-hs-003` — letter-sound **gain** (RLI cohort) · association

Like the word-reading gain model, near-flat: the top predictor sits below the threshold and its credible range spans zero.

| Rank | Predictor                     | Standardised slope | 94% credible range | Sign | P(\|beta\|>0.1) — matters?                  |
| ---- | ----------------------------- | ------------------ | ------------------ | ---- | ------------------------------------------- |
| 1    | Grammar / TROG (T)            | +0.13 logit        | −0.06 to +0.43     | +    | 0.453 — below threshold (informally "weak") |
| 2    | F                             | +0.07 logit        | −0.08 to +0.34     | +    | 0.287 — below threshold                     |
| 3    | Word reading (W)              | +0.07 logit        | −0.09 to +0.35     | +    | 0.280 — below threshold                     |
| 4    | Block-design ability (blocks) | −0.06 logit        | −0.31 to +0.09     | −    | 0.238 — below threshold                     |
| 5    | Behaviour (behav)             | −0.05 logit        | −0.25 to +0.07     | −    | 0.225 — below threshold                     |

Below rank 5: age −0.03 (0.164), E −0.01 (0.148), R −0.004 (0.141), B +0.02 (0.139).

### `lrp-rlm-hs-001` — Byrne-cohort word-reading **gain** · association

A separate historical cohort. **Only age clears the threshold**, negatively (older children gaining less); every other predictor is shrunk to near-zero. All Byrne/RLM estimates are associational, never causal.

| Rank | Predictor                 | Standardised slope | 94% credible range | Sign | P(\|beta\|>0.1) — matters? |
| ---- | ------------------------- | ------------------ | ------------------ | ---- | -------------------------- |
| 1    | Age                       | −0.34 logit        | −0.59 to +0.01     | −    | 0.912 — likely             |
| 2    | BAS digit span (basdig)   | +0.09 logit        | −0.09 to +0.39     | +    | 0.354 — below threshold    |
| 3    | BAS similarities (bassim) | +0.04 logit        | −0.13 to +0.25     | +    | 0.211 — below threshold    |
| 4    | BAS number (basnum)       | +0.01 logit        | −0.18 to +0.23     | +    | 0.199 — below threshold    |
| 5    | TROG grammar (trog)       | −0.02 logit        | −0.20 to +0.14     | −    | 0.172 — below threshold    |

Below rank 5: BPVS vocabulary −0.001 (0.139).

## The one-paragraph story

For **levels**, the horseshoe cleanly picks out the obvious reading-system partners — **word reading and letter sounds top each other's rankings** (hs-002 and hs-004) with the leaders' credible ranges wholly above zero, and expressive vocabulary is a strong second concurrent associate in both (E is actually the larger slope in hs-002, ranked second only because ordering is by exceedance probability, not effect size). This agrees with the mechanism and concurrent analyses. For the **RLI gain models** (hs-001 word reading, hs-003 letter sounds), no single predictor clears the worth-noticing threshold — the honest result at this sample size: gain is hard to predict from baseline features. The Byrne gain model (rlm-hs-001) is the one exception where a predictor does clear the threshold: age leads negatively (older children gaining less, `P(|beta|>0.1) = 0.91`), echoing the [adjusted](202607161800-findings-adjusted.md) family. These are shrinkage-ranked adjusted associations throughout — a convergent description, not a causal story — and the note is a companion cross-check to the [ITT](202607161800-findings-itt.md) causal estimates.

## What is causal

**Nothing.** A ranking is a predictive/associational summary. Its value is as a _convergent_ check: when a very different method (the horseshoe) highlights the same features as gradient boosting, we trust the description more — but neither identifies a cause, and every slope here is confounded by latent general ability. For causal contrasts see the randomisation-licensed families ([ITT](202607161800-findings-itt.md), DiD, gain-factors).
