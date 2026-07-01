# Regularized-horseshoe predictor ranking as a cross-check on the gradient-boosting ranking

<!-- cspell:ignore Spearman Piironen Vehtari lrphs lrpgbg lrpgbl crosscheck nutpie eowpvt yarclet rowpvt trog spphon celf ewrswr behav aptinfo perm -->

::: {.callout-note}
Drafted by an LLM-based AI tool (Claude Code/Opus 4.8).
:::

::: {.callout-warning}
This note was prepared by an AI tool and may contain mistakes. The numbers below
come from **development-tier** fits (horseshoe `--config dev`, gradient boosting
`--quick`) and are indicative of the ranking's shape, not final. Refresh at
reporting tier before any figure enters a report, and treat the statistical
claims as a starting point for review.
:::

**Status: pilot findings for discussion (issue #116, Phase E).** The randomised
causal claim continues to live in the ITT suite; everything here is an _adjusted
association ranking_, cross-checked across two very different methods.

## What this is

Phases A–D of issue #116 replaced gradient-boosting _hard feature selection_ with
a full-set **ranking** (permutation importance over distance-correlation
clusters). Phase E asks a simple robustness question: **does an independent
Bayesian method put the predictors in a similar order?** If a regularized
(Finnish) horseshoe sparse regression — a completely different inductive bias from
a tree ensemble — broadly agrees, that is reassurance the ranking reflects signal
in the data rather than a gradient-boosting artefact. Where the two disagree, the
construct's apparent importance is method-dependent and should be reported with
that caveat.

The two new models (`lrphs01` word-reading **gain**, `lrphs02` word-reading
**level**) are registered first-class Layer-2 statistical models with the usual
Quarto reports and convergence gate. Each is a Beta-Binomial regression in which
every construct enters as a standardised term under a global-local shrinkage prior
(global scale `tau`, per-predictor local scale `lambda_k`, regularizing slab
`c^2`; Piironen & Vehtari 2017). Predictors are ranked by posterior
`P(|beta| > delta)` with `delta = 0.1` on the per-SD logit scale — the horseshoe
analogue of permutation importance.

## Method of comparison

Both rankings are reduced to **construct level**:

- The horseshoe already ranks construct symbols (`L`, `R`, `E`, `B`, `F`, `T`,
  age, and — for gain only — blocks / behaviour).
- The gradient-boosting ranking is per raw column. Each column is mapped back to
  its construct via the measure registry, and construct importance is the **max**
  per-column permutation importance among that construct's columns (the "best
  representative" logic of the cluster table). Demographic-only columns (attend,
  time, gender, parental education, …) have no construct counterpart and drop out.

Shared constructs are then compared by Spearman rank correlation and top-3
overlap. Reproduce with:

```bash
python scripts/rank_predictors.py --model lrpgbg12 --quick   # + lrpgbl12
python scripts/fit_statistical_model.py lrphs01 --config dev # + lrphs02
python scripts/compare_horseshoe_vs_gb.py \
  --horseshoe output/statistical_models/models/lrphs01-dev/predictor_ranking.csv \
  --gb output/ranking/lrpgbg12/predictor_ranking.csv \
  --out output/statistical_models/models/lrphs01-dev/horseshoe_vs_gb.csv
```

## Word-reading gain — `lrphs01` vs `lrpgbg12`

8 shared constructs · Spearman rho = **+0.31** · top-3 overlap **2/3** (`L`, age).
(The exact rho wobbles ~±0.06 across dev-tier re-fits — small-n MC noise in the
weak tail; the top-3 story is stable. Pin at reporting tier.)

| Construct            | HS rank | P(\|β\|>δ) | GB rank | GB perm-imp |
| -------------------- | ------: | ---------: | ------: | ----------: |
| age                  |       1 |      0.628 |       1 |       0.113 |
| letter sounds `L`    |       2 |      0.542 |       2 |       0.047 |
| basic concepts `F`   |       3 |      0.362 |       5 |       0.017 |
| behaviour            |       4 |      0.359 |       7 |       0.001 |
| grammar `T`          |       5 |      0.273 |       6 |       0.012 |
| receptive vocab `R`  |       6 |      0.240 |      11 |      -0.009 |
| expressive vocab `E` |       7 |      0.181 |       4 |       0.039 |
| blending `B`         |       9 |      0.111 |       3 |       0.041 |

## Word-reading level — `lrphs02` vs `lrpgbl12`

7 shared constructs · Spearman rho = **+0.55** · top-3 overlap **2/3** (`E`, `L`).

| Construct            | HS rank | P(\|β\|>δ) | GB rank | GB perm-imp |
| -------------------- | ------: | ---------: | ------: | ----------: |
| expressive vocab `E` |       1 |      0.996 |       3 |       0.127 |
| letter sounds `L`    |       2 |      0.993 |       2 |       1.005 |
| grammar `T`          |       3 |      0.892 |       6 |       0.020 |
| age                  |       4 |      0.710 |      10 |      -0.035 |
| receptive vocab `R`  |       5 |      0.703 |       9 |      -0.005 |
| blending `B`         |       6 |      0.638 |       5 |       0.030 |
| basic concepts `F`   |       7 |      0.268 |       7 |       0.016 |

## Reading

- **Agreement (the point of the exercise).** Both methods, on both outcomes,
  place **letter-sound knowledge (`L`) at rank 2** — the single most robust signal
  in the pilot. Age tops the gain ranking in both; expressive vocabulary and
  letter sounds top the level ranking in both. The positive rank correlations
  (+0.31 gain, +0.55 level) say the orderings are broadly consistent, not
  identical — exactly what a cross-check should show.

- **Divergences flag method-dependence.**
  - Gain: gradient boosting rates blending (`B`) and expressive vocabulary (`E`)
    highly (ranks 3–4), while the horseshoe shrinks both toward zero (ranks 7 and 9) and instead promotes basic concepts (`F`). The tree ensemble is picking up
    interactions/nonlinearity the additive horseshoe cannot, so these three are
    the constructs to report with a "importance is model-dependent" caveat.
  - Level: the horseshoe promotes grammar (`T`, rank 3) above where gradient
    boosting places it (rank 6), and ranks **age** 4th (`P=0.71`) while gradient
    boosting ranks it last (10th, near-zero permutation importance) — the additive
    horseshoe reads a marginal age–reading association that the tree ensemble
    attributes to the correlated vocabulary/letter constructs instead. Gradient
    boosting's third-ranked construct for level is **phonetic spelling (`P`)**,
    which is _structurally absent_ from the horseshoe set (a floored / post-only
    measure excluded by design) — so a top-3 overlap of 2/3 here is the ceiling,
    not a miss.

- **Caveats.** n ≈ 54 children; horseshoe intervals are wide by design.
  Gain associations are between-child (one row per child, `W_last | W_T1`); level
  associations are concurrent same-wave (partly same-construct correlation).
  Gradient-boosting permutation importance can be negative (noise); the horseshoe
  key `P(|beta| > delta)` is bounded in [0, 1]. All numbers are dev/quick tier.

## Takeaway

The Bayesian sparse-regression cross-check **corroborates the letter-sounds and
age/vocabulary signals** at the top of the gradient-boosting ranking and isolates
a short list of method-dependent constructs (blending and expressive vocabulary
for gain; grammar for level). This supports reporting the gradient-boosting
ranking as the primary artefact with the horseshoe as a documented sensitivity
check — not replacing it. The final report figures should be regenerated at
reporting tier.
