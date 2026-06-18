# LRP65 — independent baseline predictors of word-reading gain (between-child)

> [!WARNING]
> AI-assisted analysis. Preliminary `dev`-tier fit; numbers will firm up at
> reporting tier and after the DAG review (PR #87). Not for citation.

Date: 2026-06-18

## Context

The Bayesian "interpretable estimand" follow-up to the LRP01 discovery layer. Three
descriptive methods (responder/non-responder, decode-predictor test, LightGBM
LRP01) agreed on the carry signal for word-reading gain — baseline letter-sound
knowledge, language/vocabulary, blending; younger children gain more — and that
non-verbal MA (block design), SES and behaviour add little once language +
letter-sounds are in. LRP65 puts that on a Bayesian footing with the estimand fixed
**before** fitting via a DAG.

The hazard: the candidate predictors are a correlated general-ability cluster
(vocabulary–block design $\rho \approx 0.63$; vocabulary–letter-sounds
$\approx 0.4$–$0.5$), so a naive joint regression is multicollinear and "independent
predictor" is ambiguous.

## Method

- **Estimand (locked):** the mutually-adjusted **between-child** association of each
  wave-1 baseline with word-reading gain. DAG drawn first (latent general ability
  `g`, drawn not fitted, drives the correlated baselines); see PR #87 / `docs/models/lrp65/dag.svg`.
- **Design (locked):** genuinely between-child — **one row per child** ($n \approx 51$),
  $x$ = T1 baseline levels, $y$ = word reading at the last wave conditioned on
  $W_{\text{T1}}$ (Beta-Binomial, EWRSWR, $N=90$). **No** child random intercept (a
  pooled all-phase design with a random intercept would tilt the estimand toward the
  *within-child* question — that is the deferred cross-lagged model, not this one).
- **Predictors (standardised, per-SD):** letter sounds (`yarclet`), an equal-weight
  language composite (ROWPVT+EOWPVT+CELF), blending, age; plus non-verbal MA and
  behaviour as tested covariates. SES (`mumedupost16`) in a complete-case sensitivity
  fit ($n = 39$).
- **Priors:** fixed weakly-informative `Normal(0, 0.5)` on the standardised slopes;
  checked against `sigma` ∈ {0.3, 0.7}. nutpie; headline R-hat ≤ 1.005, min ESS ≈ 320,
  0 divergences.
- Each predictor's **adjusted** (mutual) coefficient is reported alongside its
  **bivariate** (baseline-only-adjusted) association to expose the shared-variance shift.

## Results (preliminary, dev-tier; logit scale, per-SD; 94% intervals)

| Predictor (T1) | Adjusted [94%] | P(>0) | Bivariate [94%] | P(>0) |
| --- | --- | --- | --- | --- |
| Letter sounds | 0.19 [−0.07, 0.45] | 0.92 | 0.37 [0.13, 0.63] | 1.00 |
| Language composite | 0.28 [−0.03, 0.61] | 0.95 | 0.22 [−0.02, 0.48] | 0.95 |
| Blending | 0.04 [−0.16, 0.24] | 0.67 | 0.07 [−0.14, 0.27] | 0.75 |
| Age | −0.27 [−0.49, −0.06] | 0.01 | −0.19 [−0.38, 0.00] | 0.02 |
| Non-verbal MA | 0.00 [−0.22, 0.23] | 0.49 | 0.13 [−0.07, 0.33] | 0.88 |
| Behaviour | −0.15 [−0.37, 0.08] | 0.12 | −0.28 [−0.53, −0.06] | 0.01 |

- **Letter sounds and the language composite retain credible positive signal** when
  mutually adjusted (P(>0) ≈ 0.92, 0.95). Letter sounds attenuates from 0.37 → 0.19 —
  the expected shared-variance shift.
- **Non-verbal MA collapses to ~0** under adjustment (bivariate 0.13, P 0.88 →
  adjusted 0.00, P 0.49): its bivariate link is explained by the shared general-ability
  it shares with language + letter-sounds. Exactly the DAG's prediction.
- **Younger children gain more** (age −0.27, P(>0) ≈ 0.01).
- **Behaviour**: a negative bivariate association (P(<0) ≈ 0.99) attenuates toward zero
  when adjusted — separates extremes but does not carry independent gain-magnitude signal.
- **Prior sensitivity:** the ordering and the clear-zero reading are stable across
  `sigma` ∈ {0.3, 0.5, 0.7} (letter sounds 0.19–0.20; language 0.24–0.30; non-verbal MA
  ≈ 0 throughout).
- **SES sensitivity** ($n = 39$): SES has no independent signal (−0.03, P(>0) ≈ 0.39);
  the language/letter-sound pattern holds (language strengthens to 0.40).

## Honest reading

- **Between-child associations, $n \approx 51$ — not causal effects.** The randomised
  ITT models (LRP52/55) are the causal evidence that the *programme* works; LRP65 is
  about *which starting skills go with more gain*.
- Wide intervals are the honest result: even the headline predictors' 94% intervals
  cross (or nearly cross) zero at this n. Read the *pattern* (which predictors survive
  mutual adjustment), not a single predictor "winning".
- A near-zero adjusted coefficient for non-verbal MA means "no signal beyond the shared
  ability already captured by language + letter-sounds", **not** "unrelated to gain".

## Follow-on — the dynamic / within-child question (deferred)

A **cross-lagged panel model**: does an *early gain* in one skill (e.g. letter sounds
T1→T2) predict a *later gain* in another (word reading T2→T3)? This is the within-child
question, distinct from LRP65's between-child one; it drops usable $n$ from ~216
observations to ~50 children, so it is a separate, lower-power follow-up. Template:
Yoder, Woynaroski, Fey, Warren & Gardner (2015), *Why Dose Frequency Affects Spoken
Vocabulary in Preschoolers With Down Syndrome* (AJIDD) — an early-change → later-outcome
cascade (flagged by Sue).
