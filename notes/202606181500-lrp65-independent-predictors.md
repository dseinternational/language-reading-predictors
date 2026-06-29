# LRP65 — independent baseline predictors of word-reading gain (between-child)

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8).

> [!WARNING]
> Reporting-tier fit (DAG review, PR #87). Between-child associations at n = 51, not
> causal. Not for citation.

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
  This DAG is a **subset of the locked DAG** (`notes/202606231600-dag-revision-consolidated.md`):
  age acts _through_ `g` (no direct age→skill edges) and the intervention/dose arm
  is omitted because LRP65 is a between-child predictor model, not ITT.
- **Design (locked):** genuinely between-child — **one row per child** ($n = 51$; 2 of 53 children dropped for a missing baseline covariate),
  $x$ = T1 baseline levels, $y$ = word reading at the last wave conditioned on
  $W_{\text{T1}}$ (Beta-Binomial, EWRSWR, $N=90$). **No** child random intercept (a
  pooled all-phase design with a random intercept would tilt the estimand toward the
  _within-child_ question — that is the deferred cross-lagged model, not this one).
- **Predictors (standardised, per-SD):** letter sounds (`yarclet`), an equal-weight
  language composite (ROWPVT+EOWPVT+CELF), blending, age; plus non-verbal MA and
  behaviour as tested covariates. SES (`mumedupost16`) in a complete-case sensitivity
  fit ($n = 39$).
- **Priors:** fixed weakly-informative `Normal(0, 0.5)` on the standardised slopes;
  checked against `sigma` ∈ {0.3, 0.7}. nutpie (reporting: 6 chains × 6000 tune ×
  6000 draws); R-hat = 1.00, min ESS_bulk ≈ 9,668, max Pareto-k 0.46, 0 divergences,
  elpd_loo −167.97.
- Each predictor's **adjusted** (mutual) coefficient is reported alongside its
  **bivariate** (baseline-only-adjusted) association to expose the shared-variance shift.

## Results (reporting-tier; logit scale, per-SD; 94% intervals)

| Predictor (T1)     | Adjusted [94%]       | P(>0) | Bivariate [94%]      | P(>0) |
| ------------------ | -------------------- | ----- | -------------------- | ----- |
| Letter sounds      | 0.20 [−0.06, 0.46]   | 0.92  | 0.35 [0.10, 0.61]    | 1.00  |
| Language composite | 0.28 [−0.03, 0.58]   | 0.95  | 0.23 [−0.01, 0.48]   | 0.96  |
| Blending           | 0.04 [−0.16, 0.24]   | 0.65  | 0.07 [−0.14, 0.28]   | 0.74  |
| Age                | −0.26 [−0.47, −0.06] | 0.01  | −0.19 [−0.38, 0.00]  | 0.03  |
| Non-verbal MA      | 0.00 [−0.23, 0.23]   | 0.51  | 0.12 [−0.07, 0.32]   | 0.88  |
| Behaviour          | −0.15 [−0.39, 0.07]  | 0.11  | −0.27 [−0.51, −0.05] | 0.01  |

- **Letter sounds and the language composite retain credible positive signal** when
  mutually adjusted (P(>0) ≈ 0.92, 0.95). Letter sounds attenuates from 0.35 → 0.20 —
  the expected shared-variance shift.
- **Non-verbal MA collapses to ~0** under adjustment (bivariate 0.12, P 0.88 →
  adjusted 0.00, P 0.51): its bivariate link is explained by the shared general-ability
  it shares with language + letter-sounds. Exactly the DAG's prediction.
- **Younger children gain more** (age −0.26, P(>0) ≈ 0.01).
- **Behaviour**: a negative bivariate association (P(<0) ≈ 0.99) attenuates toward zero
  when adjusted — separates extremes but does not carry independent gain-magnitude signal.
- **Prior sensitivity:** the ordering and the clear-zero reading are stable across
  `sigma` ∈ {0.3, 0.5, 0.7} (letter sounds 0.19–0.20; language 0.23–0.29; non-verbal MA
  ≈ 0 throughout).
- **SES sensitivity** ($n = 39$): SES has no independent signal (−0.03, P(>0) ≈ 0.39);
  the language/letter-sound pattern holds (language strengthens to 0.40).
- **Natural scale (predicted gain, words out of 90, per +1 SD; Tier 1):** language
  composite ≈ **+3.4** words [−0.3, 7.7]; letter sounds ≈ **+2.3** [−0.7, 5.8]; age
  (older) ≈ **−2.5** [−4.3, −0.7]; non-verbal MA ≈ **+0.1** (≈ 0); behaviour ≈ −1.5.
  These translate the per-SD logit coefficients into words for expectations/teaching.
- **Influence (Tier 1):** max PSIS-LOO Pareto-$\hat{k}$ = 0.46, **0 of 51** children above
  the ≈ 0.7 threshold — the headline pattern is not driven by a few influential children.

## Honest reading

- **Between-child associations, $n \approx 51$ — not causal effects.** The randomised
  the word-reading ITT (now `lrpitt10`, superseding LRP52) is the causal evidence that the _programme_ works; LRP65 is
  about _which starting skills go with more gain_.
- Wide intervals are the honest result: even the headline predictors' 94% intervals
  cross (or nearly cross) zero at this n. Read the _pattern_ (which predictors survive
  mutual adjustment), not a single predictor "winning".
- A near-zero adjusted coefficient for non-verbal MA means "no signal beyond the shared
  ability already captured by language + letter-sounds", **not** "unrelated to gain".

## Follow-on — the dynamic / within-child question (deferred)

A **cross-lagged panel model**: does an _early gain_ in one skill (e.g. letter sounds
T1→T2) predict a _later gain_ in another (word reading T2→T3)? This is the within-child
question, distinct from LRP65's between-child one; it drops usable $n$ from ~216
observations to ~50 children, so it is a separate, lower-power follow-up. Template:
Yoder, Woynaroski, Fey, Warren & Gardner (2015), _Why Dose Frequency Affects Spoken
Vocabulary in Preschoolers With Down Syndrome_ (AJIDD) — an early-change → later-outcome
cascade (flagged by Sue).
