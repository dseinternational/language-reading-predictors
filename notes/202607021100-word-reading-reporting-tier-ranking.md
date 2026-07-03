# Word-reading predictor ranking at reporting tier — GB + regularized-horseshoe cross-check (#116)

<!-- cspell:ignore Spearman Piironen Vehtari lrphs lrpgbg lrpgbl crosscheck nutpie eowpvt yarclet yarcsi rowpvt trog spphon celf ewrswr behav aptinfo aptgram nonword erbnw erbword deappin deappvo deappfi agespeak agebooks numchil mumedupost dadedupost perm retune Optuna GroupKFold -->

::: {.callout-note}
Drafted by an LLM-based AI tool (Claude Code/Opus 4.8).
:::

::: {.callout-warning}
This note was prepared by an AI tool and may contain mistakes. The numbers below
come from **reporting-tier** fits and are the committed reporting-quality result
for the word-reading flagship; treat the statistical reading as a starting point
for review. `n ≈ 53` children, so intervals are wide throughout.
:::

**Status: reporting-tier findings (issue #116, flagship word-reading slice).**
Supersedes the dev/`--quick`-tier pilot in
[`202607011303-horseshoe-vs-gb-ranking-crosscheck.md`](202607011303-horseshoe-vs-gb-ranking-crosscheck.md).
The randomised causal claim continues to live in the ITT suite; everything here
is an _adjusted association ranking_, cross-checked across two very different
methods. This is one outcome (word reading) — the outcome for which we have both
a gradient-boosting model and a horseshoe cross-check. The remaining GB models
are retuned + swept in the #116 follow-up.

## What changed from the pilot

Two things moved the numbers relative to the dev-tier note:

1. **Reporting-tier fidelity.** GB rankings now use the pinned reporting CV
   (`cv_splits` 53 gain / 51 level, `perm_repeats` 50, cluster cutoff 0.4,
   seed 47); the horseshoe uses six chains × 6000 post-warmup draws
   (`target_accept` 0.99). Convergence is clean (below).
2. **The GB pair was re-tuned on the full predictor set** (150-trial Optuna,
   seed 47, MAE objective; #116 retire-selection follow-through). The earlier
   note's GB numbers came from hyperparameters tuned on the old _pruned_ subset.
   Retuning shifts the gradient-boosting ordering — most visibly it promotes
   phonetic spelling (`P`, `spphon`) on the level model — so the two methods'
   top-3 overlap is lower here than in the pilot, while the overall rank
   correlation is essentially unchanged.

## Fit quality (reporting tier)

| Model      | Outcome                               | n (children) | Pooled OOF R² | Pooled OOF MAE | Convergence                               |
| ---------- | ------------------------------------- | -----------: | ------------: | -------------: | ----------------------------------------- |
| `lrpgbg12` | word-reading **gain** (`ewrswr_gain`) |     157 (53) |     **0.100** |           3.00 | —                                         |
| `lrpgbl12` | word-reading **level** (`ewrswr`)     |     210 (53) |     **0.565** |           6.31 | —                                         |
| `lrphs01`  | gain (horseshoe)                      |     157 (53) |             — |              — | 0 divergences, R-hat 1.00, min ESS ≈ 9000 |
| `lrphs02`  | level (horseshoe)                     |     210 (53) |             — |              — | 0 divergences, R-hat 1.00, min ESS ≈ 5600 |

Gain is **near-noise** (R² ≈ 0.10, baseline-driven regression to the mean);
level is **well-predicted** (R² ≈ 0.57). This gap governs how much weight the
respective rankings can bear.

## GB cluster-first ranking (the #116 primary artefact)

Clusters are groups of distance-correlated predictors, ranked by joint
out-of-fold permutation importance. Read clusters, not single features
(per-feature `z` is `cv_splits`-sensitive; clusters are the stable unit).

**Gain (`lrpgbg12`) — top clusters:** `attend` (0.075 ± 0.333), `age`
(0.039 ± 0.260); every other cluster is ≈ 0. The standard deviations dwarf the
means, so beyond a weak attendance/age signal the gain ranking is not
distinguishable from noise — consistent with R² ≈ 0.10. **Treat the gain
ranking as exploratory.**

**Level (`lrpgbl12`) — top clusters:** (1) phonetic spelling + single-word
reading `{spphon, yarcsi}` (0.425 ± 1.35); (2) a six-member vocabulary/language
cluster `{eowpvt, rowpvt, b1exto, b1reto, aptinfo, celf}` (0.214); (3) letter
sounds `yarclet` (0.068); (4) `age` (0.051); (5) non-word reading `nonword`
(0.034). This is a coherent phonics + vocabulary + letter-knowledge story with
real signal behind it.

## Horseshoe vs GB, construct level

Both rankings are reduced to construct symbols (see
`scripts/compare_horseshoe_vs_gb.py`): the horseshoe ranks symbols directly by
posterior `P(|β| > δ)` (`δ = 0.1`, per-SD logit); the GB per-column permutation
importances are collapsed to the max within each construct. Demographic-only
columns (`attend`, `time`, gender, parental education, …) have no construct
symbol and drop out of this alignment.

### Word-reading gain — `lrphs01` vs `lrpgbg12`

8 shared constructs · Spearman ρ = **+0.295** · top-3 overlap **2/3** (`age`, `behav`).

| Construct            | HS rank | P(\|β\|>δ) | GB rank | GB perm-imp |
| -------------------- | ------: | ---------: | ------: | ----------: |
| age                  |       1 |      0.585 |       1 |       0.039 |
| letter sounds `L`    |       2 |      0.522 |       8 |      −0.010 |
| behaviour `behav`    |       3 |      0.380 |       2 |       0.000 |
| basic concepts `F`   |       4 |      0.363 |       9 |      −0.010 |
| grammar `T`          |       5 |      0.251 |      10 |      −0.020 |
| receptive vocab `R`  |       6 |      0.243 |       5 |      −0.003 |
| expressive vocab `E` |       7 |      0.175 |      11 |      −0.026 |
| blending `B`         |       9 |      0.115 |       4 |      −0.001 |

Both methods put `age` first. Almost every GB perm-importance is ≈ 0 or
negative — the gain model has essentially no signal to rank beyond age, so the
horseshoe's letter-sounds/basic-concepts ordering is _not_ corroborated by
gradient boosting here. The gain agreement is real but thin (`age`, `behav`).

### Word-reading level — `lrphs02` vs `lrpgbl12`

7 shared constructs · Spearman ρ = **+0.497** · top-3 overlap **1/3** (`L`).

| Construct            | HS rank | P(\|β\|>δ) | GB rank | GB perm-imp |
| -------------------- | ------: | ---------: | ------: | ----------: |
| letter sounds `L`    |       1 |      0.994 |       2 |       0.081 |
| expressive vocab `E` |       2 |      0.992 |       5 |       0.026 |
| grammar `T`          |       3 |      0.881 |      10 |      −0.010 |
| age                  |       4 |      0.755 |       3 |       0.055 |
| receptive vocab `R`  |       5 |      0.671 |       6 |       0.022 |
| blending `B`         |       6 |      0.626 |       7 |       0.010 |
| basic concepts `F`   |       7 |      0.256 |       8 |       0.005 |

## Reading

- **Agreement holds at reporting tier.** The rank correlations (+0.30 gain,
  +0.50 level) are within pilot noise of the dev-tier values (+0.31, +0.55) —
  the cross-check conclusion survives the move to reporting fidelity and the GB
  retune. Both orderings are broadly consistent, not identical.
- **Letter-sound knowledge (`L`) is the most robust level signal.** It is the
  horseshoe's #1 construct for level (`P = 0.99`) and gradient boosting's #2
  (cluster #3), i.e. corroborated across two very different inductive biases.
  For gain it is the horseshoe's #2 but GB assigns it ≈ 0 importance, because
  the gain model has almost no signal to distribute.
- **The level top-3 divergence is structural, not a contradiction.** GB's top
  level construct is phonetic spelling (`P`, `spphon`), which is _absent from the
  horseshoe set_ by design (a floored / post-only measure). With `P` excluded,
  1/3 is close to the achievable ceiling; the horseshoe instead promotes
  expressive vocabulary (`E`, #2) and grammar (`T`, #3), which gradient boosting
  attributes to the correlated letter/vocabulary constructs. Report `E`/`T`/`P`
  as the method-dependent constructs for level.
- **Gain remains a weak, exploratory ranking.** Beyond `age`, the gain cluster
  importances are indistinguishable from zero. This is a property of the
  outcome (baseline-driven regression to the mean at `n ≈ 53`), not of the
  method or the tier.

## Caveats

`n ≈ 53` children. Gain associations are between-child (one row per child,
`W_last | W_T1`); level associations are concurrent same-wave and partly reflect
same-construct correlation. Gradient-boosting permutation importance can go
negative (noise); the horseshoe `P(|β| > δ)` is bounded in [0, 1]. Phonetic
spelling is structurally absent from the horseshoe. GB hyperparameters are the
#116 full-set reporting retune, so these GB numbers supersede — and differ
from — the earlier pruned-subset pilot.

## Reproduce

```bash
# retune (already applied to lrpgbg12.py / lrpgbl12.py)
python scripts/tune_model.py lrpgbg12 --n-trials 150 --seed 47 --scoring mae --lgbm-objective mae
python scripts/tune_model.py lrpgbl12 --n-trials 150 --seed 47 --scoring mae --lgbm-objective mae
# reporting fits + rankings
python scripts/fit_model.py lrpgbg12 --config reporting --render   # + lrpgbl12
python scripts/rank_predictors.py --model lrpgbg12                 # + lrpgbl12
python scripts/fit_statistical_model.py lrphs01 --config reporting --render  # + lrphs02
# comparison
python scripts/compare_horseshoe_vs_gb.py \
  --horseshoe output/statistical_models/models/lrphs01-reporting/predictor_ranking.csv \
  --gb output/ranking/lrpgbg12/predictor_ranking.csv \
  --out output/statistical_models/models/lrphs01-reporting/horseshoe_vs_gb.csv   # + level
```

## Takeaway

At reporting tier the Bayesian sparse-regression cross-check still corroborates
the gradient-boosting ranking's headline signals — letter sounds and
vocabulary for level, `age` for gain — and isolates a short list of
method-dependent constructs (expressive vocabulary, grammar, and the
structurally-absent phonetic spelling for level). The gradient-boosting ranking
remains the primary artefact with the horseshoe as a documented sensitivity
check. Word-reading **level** supports a substantive ranking; word-reading
**gain** should be reported as exploratory only.
