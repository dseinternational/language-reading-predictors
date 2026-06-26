# LRP66 — latent general-ability model (general vs specific predictors of gain)

> [!WARNING]
> AI-assisted analysis. Reporting-tier fit; the latent model at n = 51 is
> fragile (47 divergences in the latent/robustness arm). Triangulation against
> LRP65, not a definitive decomposition. Not for citation.

Date: 2026-06-19

## Context

Tier 2 follow-up to LRP65. LRP65 used mutual adjustment and found letter sounds + language
retain signal for word-reading gain while non-verbal MA collapses. But mutual adjustment
cannot separate *general ability* from *specific skill* when the predictors are noisy
indicators of one factor. LRP66 fits the latent general ability `g` the LRP65 DAG only drew,
and asks two things: how much of gain is general ability, and does any specific skill predict
gain **beyond** `g` (the actionable "direct teaching target vs ability marker?" question).

## Method

- **Design:** between-child, one row per child (n = 51 `n_obs`; 2 rows dropped for missing baseline), outcome `W_last | W_T1` — identical
  to LRP65 (reuses `phase_mode="span"`).
- **Measurement:** one-factor Gaussian CFA. `g ~ Normal(0,1)` (scale fixed) with **positive**
  loadings (orientation fixed; all positive-manifold) on the standardised T1 skills — letter
  sounds, ROWPVT, EOWPVT, CELF, blending, non-verbal MA. This is the **reflective-`g`
  specialisation of the shared DAG v5** (`notes/202606221200-shared-dag-v5.md`): `g` is fitted
  as a reflective factor of the baseline skills (no edge changes from v5).
- **Structural (Beta-Binomial on `W_post | W_pre`):** two estimands —
  `beta_g_total` (gain ~ g + age, no observed skills) and `beta_g_residual / beta_L /
  beta_lang` (gain ~ g + observed skills + age). The latter give effects *beyond g*.
- **Robustness arm:** an orthogonal language-specific latent factor on the three language
  measures (`beta_lang_specific`), fit at `target_accept = 0.99`. Fragile by design.
- nutpie (reporting: 6 chains × 6000 tune × 6000 draws). Headline: R-hat ≤ 1.003,
  **0 divergences**. Latent/robustness arm (`latent_target_accept = 0.99`): **47
  divergences** at reporting tier — read as "n too small to identify cleanly".

## Results (reporting-tier; per-SD logit; 94% intervals)

**Loadings on `g` (communality = share of the skill explained by `g`):**

| Skill (T1) | Loading | Communality |
| --- | --- | --- |
| Expressive vocab (EOWPVT) | 0.97 | 0.88 |
| Receptive vocab (ROWPVT) | 0.89 | 0.74 |
| **Non-verbal MA (block design)** | **0.77** | **0.56** |
| CELF concepts | 0.70 | 0.47 |
| Letter sounds | 0.59 | 0.34 |
| Blending | 0.56 | 0.30 |

**Structural paths:**

| Path | Mean [94%] | P(>0) | Reading |
| --- | --- | --- | --- |
| `beta_g_total` | **0.41** [0.14, 0.70] | 0.997 | general ability predicts gain |
| `beta_g_residual` | 0.04 [−0.50, 0.61] | 0.54 | ~0 once skills enter directly |
| `beta_L` (letter sounds beyond g) | 0.22 [−0.04, 0.49] | 0.95 | retains signal |
| `beta_lang` (language beyond g) | 0.29 [−0.19, 0.77] | 0.88 | positive, uncertain |
| `beta_age` | −0.28 [−0.49, −0.07] | 0.01 | younger gain more |

Latent robustness arm: `beta_lang_specific` = 0.10 [−0.39, 0.52] (P 0.68) — weak/uncertain;
`beta_g` recovers to 0.31 (P 0.96) when observed language is replaced by the latent specific.

## Honest reading

- **General ability matters — but largely *through* letter sounds and language.**
  `beta_g_total` is credibly positive (0.41), yet the residual `g` path drops to ~0 once
  letter sounds and language enter directly, while those skills keep positive beyond-`g`
  effects. So the "general ability" signal is substantially the *specific* letter-sound /
  language skills — which supports teaching them directly rather than only broad enrichment.
- **Non-verbal MA loads heavily on `g` (0.78) but has no specific beyond-`g` path** — the
  clean explanation of its LRP65 collapse.
- **Triangulates with LRP65:** same skills carry the signal; this adds *why* (shared `g`
  redistributes) and *which is specific*.
- **Caveats:** between-child associations at n = 51, not causal; the latent model is fragile
  (the language-specific arm is weakly identified — priors do real work); wide intervals.
  See [[202606181500-lrp65-independent-predictors]].
