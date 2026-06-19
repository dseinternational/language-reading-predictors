# LRP66 — latent general-ability model (general vs specific predictors of gain)

> [!WARNING]
> AI-assisted analysis. Preliminary `dev`-tier fit; a latent model at n ≈ 51 is
> fragile. Triangulation against LRP65, not a definitive decomposition. Not for citation.

Date: 2026-06-19

## Context

Tier 2 follow-up to LRP65. LRP65 used mutual adjustment and found letter sounds + language
retain signal for word-reading gain while non-verbal MA collapses. But mutual adjustment
cannot separate *general ability* from *specific skill* when the predictors are noisy
indicators of one factor. LRP66 fits the latent general ability `g` the LRP65 DAG only drew,
and asks two things: how much of gain is general ability, and does any specific skill predict
gain **beyond** `g` (the actionable "direct teaching target vs ability marker?" question).

## Method

- **Design:** between-child, one row per child (n ≈ 51), outcome `W_last | W_T1` — identical
  to LRP65 (reuses `phase_mode="span"`).
- **Measurement:** one-factor Gaussian CFA. `g ~ Normal(0,1)` (scale fixed) with **positive**
  loadings (orientation fixed; all positive-manifold) on the standardised T1 skills — letter
  sounds, ROWPVT, EOWPVT, CELF, blending, non-verbal MA.
- **Structural (Beta-Binomial on `W_post | W_pre`):** two estimands —
  `beta_g_total` (gain ~ g + age, no observed skills) and `beta_g_residual / beta_L /
  beta_lang` (gain ~ g + observed skills + age). The latter give effects *beyond g*.
- **Robustness arm:** an orthogonal language-specific latent factor on the three language
  measures (`beta_lang_specific`), fit at `target_accept = 0.99`. Fragile by design.
- nutpie. Headline: R-hat ≤ 1.02, **0 divergences**. Latent arm: 8 divergences at
  target_accept 0.99 (down from 38 at 0.95) — read as "n too small to identify cleanly".

## Results (preliminary, dev-tier; per-SD logit; 94% intervals)

**Loadings on `g` (communality = share of the skill explained by `g`):**

| Skill (T1) | Loading | Communality |
| --- | --- | --- |
| Expressive vocab (EOWPVT) | 0.98 | 0.88 |
| Receptive vocab (ROWPVT) | 0.90 | 0.75 |
| **Non-verbal MA (block design)** | **0.78** | **0.57** |
| CELF concepts | 0.71 | 0.48 |
| Letter sounds | 0.60 | 0.34 |
| Blending | 0.56 | 0.30 |

**Structural paths:**

| Path | Mean [94%] | P(>0) | Reading |
| --- | --- | --- | --- |
| `beta_g_total` | **0.43** [0.16, 0.74] | 0.997 | general ability predicts gain |
| `beta_g_residual` | 0.06 [−0.48, 0.67] | 0.58 | ~0 once skills enter directly |
| `beta_L` (letter sounds beyond g) | 0.22 [−0.04, 0.47] | 0.95 | retains signal |
| `beta_lang` (language beyond g) | 0.27 [−0.25, 0.75] | 0.85 | positive, uncertain |
| `beta_age` | −0.28 [−0.51, −0.07] | 0.01 | younger gain more |

Latent robustness arm: `beta_lang_specific` = 0.14 [−0.32, 0.56] (P 0.74) — weak/uncertain;
`beta_g` recovers to 0.29 (P 0.96) when observed language is replaced by the latent specific.

## Honest reading

- **General ability matters — but largely *through* letter sounds and language.**
  `beta_g_total` is credibly positive (0.43), yet the residual `g` path drops to ~0 once
  letter sounds and language enter directly, while those skills keep positive beyond-`g`
  effects. So the "general ability" signal is substantially the *specific* letter-sound /
  language skills — which supports teaching them directly rather than only broad enrichment.
- **Non-verbal MA loads heavily on `g` (0.78) but has no specific beyond-`g` path** — the
  clean explanation of its LRP65 collapse.
- **Triangulates with LRP65:** same skills carry the signal; this adds *why* (shared `g`
  redistributes) and *which is specific*.
- **Caveats:** between-child associations at n ≈ 51, not causal; the latent model is fragile
  (the language-specific arm is weakly identified — priors do real work); wide intervals.
  See [[202606181500-lrp65-independent-predictors]].
