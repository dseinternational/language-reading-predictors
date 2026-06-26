<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Gradient-boosting models — final selected features and near-miss predictors

> [!WARNING]
> This note was prepared by an AI tool and may contain mistakes. Treat it as a
> compiled reference, not a source of new claims; the underlying selections are
> defined in the model registry and `notes/202606211200-uniform-gb-fs.md`.

Date: 2026-06-23

## Purpose

A single reference compiling, for every GB primary model (LRP01–LRP22), the
**final feature set** after the uniform feature selection of 2026-06-21, split
into the two model families:

- **gain models** — predicting a `_GAIN` change score (post − pre);
- **level models** — predicting a concurrent (same-wave) level.

A third column lists **near-miss** predictors. Variants (`_noconstruct`) and the
taught-vocabulary primaries (LRP23/LRP24) are out of scope — see the bottom.

## How the columns were derived

- **Final features** come from the model registry (`models/registry.py`
  `MODELS[...].predictor_vars`), the source of truth; they match each fitted
  `output/models/{id}/config.json`. For gain models the same-skill baseline is
  force-kept as the regression-to-the-mean anchor (back-ticked in the tables).
- **Near-miss** = a predictor that cleared the importance noise floor in the
  *full* predictor set (full-set out-of-fold permutation-importance mean > 0.005)
  but was **dropped** during selection — almost always the redundancy-filter
  casualty (distance correlation ≥ 0.70) of a retained same-construct measure, or
  a standardised-instrument swap. The number in parentheses is that full-set
  importance. Source: the replication `facts.json` files
  (`full_top_importance` for LRP01–10; `importance_table` for LRP11–22, whose
  registered set equalled the full set at replication time). An empty cell means
  nothing dropped cleared the floor.

Note on precision: full-set permutation importances are noisy at this sample size
(no predictor reaches 1 SD above zero in any model — see the replication note).
The near-miss values rank-order reliably only near the top; read them as "carried
non-trivial full-set importance but was pruned," not as precise effect sizes.

## Gain models (`*_gain` targets)

| Model | Outcome (gain in…) | Final features selected | Near-miss (full-set imp.) |
|---|---|---|---|
| lrp01 | word reading `ewrswr` | `ewrswr`, age, attend | — |
| lrp03 | expressive vocabulary `eowpvt` | `eowpvt`, trog, deappvo | aptinfo (0.042) |
| lrp05 | letter-sound knowledge `yarclet` | `yarclet`, time, age, spphon, erbnw, deappfi, attend, dadedupost16 | — |
| lrp07 | receptive vocabulary `rowpvt` | `rowpvt`, trog | — |
| lrp09 | concept knowledge `celf` | `celf`, age, nonword, rowpvt, yarclet | — |
| lrp11 | receptive grammar `trog` | `trog`, celf, eowpvt, deappvo, deappfi | — |
| lrp13 | nonword reading `nonword` | `nonword`, erbnw | — |
| lrp15 | phoneme blending `blending` | `blending`, eowpvt | — |
| lrp17 | expressive grammar `aptgram` | `aptgram`, age, erbword, rowpvt, spphon, trog | b1reto (0.013) |
| lrp19 | expressive information `aptinfo` | `aptinfo`, erbword, spphon, trog, deappvo | b1reto (0.074), aptgram (0.036) |
| lrp21 | fine articulation `deappfi` | `deappfi`, age, eowpvt, nonword, deappvo, ewrswr | b1exto (0.080), deappin (0.008) |

Seven of eleven gain models have **no** above-floor near-miss: once the same-skill
baseline is in the set, nothing else clears the noise floor — the "gain models are
near-noise / baseline-driven" signature (`notes/202606201500-gb-replication-findings.md`).

## Level models (concurrent-level targets)

| Model | Outcome (level of…) | Final features selected | Near-miss (full-set imp.) |
|---|---|---|---|
| lrp02 | word reading `ewrswr` | spphon, yarclet, eowpvt, celf, erbword, blending | b1exto (0.047), yarcsi (0.032), aptinfo (0.007) |
| lrp04 | expressive vocabulary `eowpvt` | b1exto, celf, rowpvt, yarclet, deappfi, ewrswr, time | aptinfo (0.093), b1reto (0.013) |
| lrp06 | letter-sound knowledge `yarclet` | ewrswr, nonword, blending, eowpvt, erbword, time | b1exto (0.219), b1reto (0.022) |
| lrp08 | receptive vocabulary `rowpvt` | aptinfo, trog, nonword, erbword, celf, deappin, time | b1reto (0.146), eowpvt (0.049), b1exto (0.037) |
| lrp10 | concept knowledge `celf` | eowpvt, age, deappin | b1reto (0.010) |
| lrp12 | receptive grammar `trog` | 26 retained (corr-filter only, noise-floor skipped) ¹ | b1reto (0.065) |
| lrp14 | nonword reading `nonword` | aptinfo, yarclet, ewrswr | spphon (0.022) |
| lrp16 | phoneme blending `blending` | rowpvt, ewrswr | b1reto (0.018), spphon (0.007) |
| lrp18 | expressive grammar `aptgram` | aptinfo, erbnw, nonword, yarcsi | b1reto (0.011) |
| lrp20 | expressive information `aptinfo` | aptgram, b1exto, rowpvt, blending, erbnw, deappvo, ewrswr, age | eowpvt (0.198), b1reto (0.076) |
| lrp22 | fine articulation `deappfi` | deappin, trog, yarclet, ewrswr, agespeak, dadedupost16, time | — |

¹ lrp12 keeps the full set **minus** the six redundancy casualties (aptinfo,
b1reto, eowpvt, erbnw, spphon, deappfi). The uniform noise-floor cut is
deliberately skipped here because applying it prunes this flat-importance target
to 3 predictors and drops pooled R² 0.47 → 0.30 (`notes/202606211200-uniform-gb-fs.md`).

## The pattern in the near-miss column

The near-miss predictors are almost entirely the bespoke directly-taught Block-1
totals (`b1reto` = receptive total, `b1exto` = expressive total) and
same-instrument siblings (`aptinfo`↔`aptgram` — APT subtests; `eowpvt`, `spphon`,
`deappin`). They are the casualties of two selection rules: the distance-correlation
redundancy filter (dcor ≥ 0.70, keep the higher-importance representative) and the
standardised-instrument swap that prefers `eowpvt`←`b1exto` and `rowpvt`←`b1reto`.

In other words, nearly every near-miss carries *the same construct* as a retained
predictor — not an independent signal that narrowly lost out. This is the concrete
case for the combine-vs-pick decision: these are reflective indicators of one
latent construct, where a composite (or a latent measurement model) would stop the
pipeline from discarding a correlated sibling, and where "pick one" is what the
current selection does by default.

## Caveats (read the tables under these)

- **Gain-model rankings are near-noise** — baseline-driven regression to the mean;
  do not quote secondary features from gain models as predictors of progress.
- **Level-model selections are largely concurrent same-construct correlation**, not
  developmental prediction; the `_noconstruct` variants (lrp04/18/20/22) exist to
  expose this by dropping the same-skill sibling.
- Pinned hyperparameters are CV-equivalent optima, not unique.

## Out of scope

- `_noconstruct` variants (lrp04/18/20/22): same-skill sibling dropped to expose
  concurrent correlation; final sets in the registry / `fs_implementation.json`.
- LRP23 (`b1extau_gain`) / LRP24 (`b1extau`, taught expressive vocabulary): still
  exploratory baselines with the full 34 / 32-predictor set — no tune or feature
  selection applied yet, so excluded from the tables.

## Reproduction

Derivation scripts (gitignored working area):

```bash
# final sets from the registry + near-miss from replication facts.json
& "V:\miniconda3\Scripts\conda.exe" run -n dse-language-reading-predictors \
  --no-capture-output python output/replication/scratch/compile_features.py
& "V:\miniconda3\Scripts\conda.exe" run -n dse-language-reading-predictors \
  --no-capture-output python output/replication/scratch/nearmiss_final.py
```

## Related notes

- `notes/202606211200-uniform-gb-fs.md` — the uniform feature-selection method.
- `notes/202606201500-gb-replication-findings.md` — why gains ≈ noise and levels ≈
  concurrent correlation; the redundancy clusters.
