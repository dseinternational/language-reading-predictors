# Code-review refit: LRP01–LRP22 results summary

<!-- SPDX-License-Identifier: CC-BY-4.0 -->

Date: 2026-05-14
Branch: `dev/frank/code-review-fixes` (PR #60)
Run: `python scripts/fit_model.py all --config reporting`

## Why the refit

A line-by-line source review (see PR #60) surfaced several bugs whose direct effect was to mis-report numbers in `output/models/{model_id}/`. The ones that touched persisted artefacts:

- **Permutation importance** was being computed in-sample on training rows the LGBM had just memorised, with the default R² scorer and no group awareness. The values in `permutation_importance.csv` measured tree memorisation rather than held-out signal. New behaviour: per-fold OOF importance on validation rows from the same GroupKFold(51) splits used elsewhere, scored as `neg_root_mean_squared_error`, aggregated across folds.
- **Pooled R²** in `metrics.json` used the global mean of held-out values as the SS_tot baseline. New behaviour: per-fold training-mean baseline so the denominator is not contaminated by the rows being predicted.
- **Stability selection** (reporting config only) treated `resample(..., replace=True)` outputs as a *set* via `groups.isin(...)`, collapsing duplicates back to unique subjects — i.e. ~50% sub-sampling without replacement, not a real bootstrap. New behaviour: per-subject row repetition so a twice-drawn subject contributes its rows twice.
- **BEHAV dtype**: previously cast to `UInt8` in `configure_data_types` (the docstring describes BEHAV as an averaged 1-5 score). New behaviour: cast to `Float64`. The current `rli_data_long.csv` happens to store BEHAV as integer-valued floats already, so this change is latent for this dataset — no current predictor matrix entries shift.

All 22 final models were re-fitted with `--config reporting`. Total wall-clock ~85 minutes on the local machine. Zero failures.

## Snapshot methodology

Pre-fix `metrics.json`, `permutation_importance.csv`, `cv_scores.csv`, `stability_selection.csv`, `construct_importance.csv`, and `shap_direction_diagnostics.csv` were copied to `output/_pre_fix_snapshot/{model_id}/` before the re-fit cleared each model's output directory.

## What is unchanged across the refit

| Artefact | Reason |
| --- | --- |
| Per-fold `cv_*` columns in `cv_scores.csv` (mae / rmse / r2 / medae) | Same 51 GroupKFold splits, same LGBM params, same seed, same predictor matrix (BEHAV change was latent for this data). The per-fold scores match byte-for-byte. |
| `cv_pooled_rmse`, `cv_pooled_mae`, `cv_pooled_medae` | OOF predictions are now derived from `cross_validate`'s returned per-fold estimators rather than a second `cross_val_predict` call — but with deterministic folds the predictions are identical to the previous run. Only `cv_pooled_r2` shifts (different baseline). |
| `in_sample_*` metrics | Same fit on the same full data. |
| SHAP analysis (`shap_*.png`, `shap_direction_diagnostics.csv`) | Same fitted booster; the `_tree_estimator` refactor changed how the booster is looked up but not what it is. |

For completeness, every cv_rmse_mean delta across the 22 models is exactly 0.

## Pooled R² (changed baseline)

The new value uses per-fold training-mean as the constant-predictor baseline. SS_tot is therefore computed against a baseline that the held-out subject did not contribute to. For grouped CV with one subject per fold, held-out values are systematically more extreme relative to the train mean than relative to the global mean, so SS_tot grows and reported R² rises modestly.

| Model | Pre | Post | Δ |
| ----- | ---: | ---: | ---: |
| lrp01 | 0.1004 | 0.1173 | +0.017 |
| lrp02 | 0.6002 | 0.6130 | +0.013 |
| lrp03 | 0.1314 | 0.1368 | +0.005 |
| lrp04 | 0.6932 | 0.7018 | +0.009 |
| lrp05 | 0.1954 | 0.2016 | +0.006 |
| lrp06 | 0.6372 | 0.6479 | +0.011 |
| lrp07 | 0.1903 | 0.1934 | +0.003 |
| lrp08 | 0.6280 | 0.6382 | +0.010 |
| lrp09 | 0.2433 | 0.2471 | +0.004 |
| lrp10 | 0.4830 | 0.4959 | +0.013 |
| lrp11 | 0.1736 | 0.1790 | +0.005 |
| lrp12 | 0.4591 | 0.4716 | +0.012 |
| lrp13 | 0.2352 | 0.2389 | +0.004 |
| lrp14 | 0.3660 | 0.3811 | +0.015 |
| lrp15 | 0.1160 | 0.1213 | +0.005 |
| lrp16 | 0.3482 | 0.3634 | +0.015 |
| lrp17 | 0.0673 | 0.0731 | +0.006 |
| lrp18 | 0.6680 | 0.6772 | +0.009 |
| lrp19 | 0.0815 | 0.0911 | +0.010 |
| lrp20 | 0.7969 | 0.8022 | +0.005 |
| lrp21 | 0.0913 | 0.0948 | +0.004 |
| lrp22 | 0.5208 | 0.5361 | +0.015 |

All 22 models shift up by 0.003-0.017. Smallest deltas are on the lowest-signal gain models (LRP03, LRP07, LRP09, LRP13, LRP21); largest on level models with stronger within-subject means (LRP14, LRP16, LRP22). Direction and magnitude are consistent with the algebraic prediction; no qualitative interpretation changes.

## Permutation importance (algorithm change)

Old: in-sample R² drop on the training set, no group awareness. New: per-fold OOF RMSE drop on held-out validation rows with `groups=subject_id`, 50 repeats × 51 folds.

The ranking is partially preserved for the small-predictor models and substantially shuffled for the wide ones. Spearman rank correlation of pre- vs post- per-model importance ranks:

| Model | n features | Top-5 (pre) | Top-5 (post) | rank ρ |
| ----- | ---: | --- | --- | ---: |
| lrp01 | 6 | age, attend, yarclet, celf, b1exto | **attend, age**, blending, celf, yarclet | 0.543 |
| lrp02 | 13 | spphon, yarclet, eowpvt, nonword, aptinfo | spphon, **eowpvt, yarclet**, age, agebooks | 0.429 |
| lrp03 | 11 | eowpvt, deappvo, aptinfo, trog, age | eowpvt, deappvo, **trog, aptgram**, aptinfo | 0.545 |
| lrp04 | 7 | aptinfo, rowpvt, celf, ewrswr, yarclet | aptinfo, **yarclet, celf**, ewrswr, rowpvt | 0.679 |
| lrp05 | 15 | yarclet, deappvo, age, deappfi, blending | yarclet, **age, deappfi, attend, time** | 0.311 |
| lrp06 | 10 | ewrswr, b1exto, deappin, nonword, dadedupost16 | ewrswr, b1exto, **time**, nonword, **erbword** | 0.600 |
| lrp07 | 12 | rowpvt, b1reto, mumedupost16, attend, celf | rowpvt, **deappvo, attend**, trog, b1exto | 0.161 |
| lrp08 | 17 | b1reto, aptinfo, eowpvt, celf, trog | **aptinfo, eowpvt**, trog, b1reto, age | 0.642 |
| lrp09 | 17 | celf, deappfi, spphon, age, attend | celf, **nonword, yarclet, rowpvt, b1exto** | 0.297 |
| lrp10 | 10 | rowpvt, aptinfo, ewrswr, trog, agespeak | rowpvt, **ewrswr, aptinfo**, deappin, dadedupost16 | 0.612 |
| lrp11 | 34 | trog, eowpvt, celf, b1exto, aptinfo | trog, **celf, eowpvt, deappfi, deappvo** | 0.143 |
| lrp12 | 32 | b1exto, b1reto, rowpvt, aptinfo, celf | b1exto, b1reto, **deappin, deappvo, time** | 0.140 |
| lrp13 | 34 | nonword, erbnw, ewrswr, aptinfo, spphon | nonword, erbnw, ewrswr, aptinfo, **blending** | 0.119 |
| lrp14 | 32 | spphon, ewrswr, yarclet, aptinfo, blending | **ewrswr, spphon**, yarclet, aptinfo, **aptgram** | 0.076 |
| lrp15 | 34 | blending, eowpvt, rowpvt, attend, deappin | blending, eowpvt, **deappin, attend, erbword** | -0.012 |
| lrp16 | 32 | ewrswr, b1reto, spphon, aptinfo, yarclet | ewrswr, b1reto, spphon, **erbword, yarcsi** | -0.042 |
| lrp17 | 34 | aptgram, erbword, trog, attend, spphon | aptgram, erbword, **spphon, age, b1reto** | 0.128 |
| lrp18 | 32 | aptinfo, erbnw, erbword, deappin, deappfi | aptinfo, erbnw, **b1reto, nonword, yarcsi** | 0.105 |
| lrp19 | 34 | aptinfo, b1reto, erbword, ewrswr, aptgram | aptinfo, b1reto, aptgram, **deappvo, erbword** | 0.385 |
| lrp20 | 32 | aptgram, b1exto, eowpvt, b1reto, rowpvt | aptgram, b1exto, eowpvt, rowpvt, b1reto | 0.672 |
| lrp21 | 34 | deappfi, erbnw, deappvo, deappin, age | deappfi, **ewrswr, b1exto**, deappvo, age | 0.088 |
| lrp22 | 32 | deappin, erbword, ewrswr, numchil, dadedupost16 | deappin, ewrswr, **time, yarclet, trog** | 0.155 |

(Bold marks features that changed position within the top-5.)

Observations:

- **Top-1 is mostly stable.** Twenty of 22 models keep the same top-ranked predictor (LRP01 swaps `age` → `attend`; LRP14 swaps `spphon` → `ewrswr`). Headline "X dominates" claims survive almost everywhere.
- **Wide-predictor models reshuffle dramatically.** Models with 30+ predictors (LRP11–LRP22 mostly) show rank correlations 0.1 or below; LRP15 and LRP16 are essentially uncorrelated. The in-sample importance had been ranking by memorisation-of-noise variance, which the OOF view washes out.
- **Slim, tuned-down models reshuffle less.** LRP04 (7 predictors), LRP08 (17), LRP10 (10), LRP20 (32 but tuned post-Select) sit at ρ ≥ 0.6. Where feature selection has already pruned noisy predictors, OOF and in-sample agree.
- **Direction interpretation (per CLAUDE.md guidance) should be re-read against the new SHAP beeswarms** alongside these new rankings. SHAP itself is unchanged, but pairing it with the new importance ordering may shift which predictors warrant commentary in each model's `index.qmd`.

## Construct importance

`construct_importance.csv` rolls `importance_mean` up to construct families via `Variables.construct_of`. The new OOF view changes feature ranks (see above), so construct totals re-aggregate. Top-1 construct changed for **4 of 22** models:

| Model | Top-1 (pre) | Top-1 (post) | Rank ρ (constructs) |
| --- | --- | --- | ---: |
| **lrp01** | reading_decoding | **intervention** | 0.30 |
| **lrp10** | language_composite | **receptive_vocabulary** | 0.64 |
| **lrp12** | receptive_vocabulary | **expressive_vocabulary** | 0.25 |
| **lrp14** | reading_decoding | **reading_word** | 0.35 |

For these four the *dominant family* claim shifts and any narrative that says "X-related skills are the strongest predictors of Y" should be re-read against the new top-1. The construct distinction matters most for LRP10 (language → receptive vocabulary) and LRP14 (reading decoding → reading word) because those are different theoretical claims.

For the remaining 18 models the top-1 family is preserved, but lower-tier ordering reshuffles in line with the feature-level ranking changes:

- **High agreement (rank ρ ≥ 0.7)**: LRP02, LRP03, LRP04, LRP06, LRP07, LRP19, LRP20. Even the top-3 set is identical for LRP06, LRP08, LRP20.
- **Moderate agreement (0.4 ≤ ρ < 0.7)**: LRP05, LRP08, LRP09, LRP11, LRP16.
- **Low or negative agreement (ρ < 0.4)**: LRP13, LRP15, LRP17, LRP18, LRP21, LRP22. These are the wide models (30+ predictors) where the old in-sample importance had been rewarding tree memorisation; expected.

Quick-glance summary of cases worth re-reading per model:

| Model | Pre top-3 | Post top-3 |
| --- | --- | --- |
| lrp01 | reading_decoding, demographics_child, intervention | intervention, demographics_child, language_composite |
| lrp02 | reading_decoding, expressive_vocabulary, demographics_child | reading_decoding, expressive_vocabulary, demographics_child |
| lrp03 | expressive_vocabulary, language_composite, articulation | expressive_vocabulary, articulation, language_composite |
| lrp04 | language_composite, receptive_vocabulary, reading_word | language_composite, reading_decoding, reading_word |
| lrp05 | reading_decoding, articulation, demographics_child | reading_decoding, demographics_child, intervention |
| lrp06 | reading_word, expressive_vocabulary, study_structure | reading_word, expressive_vocabulary, study_structure |
| lrp07 | receptive_vocabulary, articulation, language_composite | receptive_vocabulary, articulation, intervention |
| lrp08 | language_composite, expressive_vocabulary, receptive_vocabulary | language_composite, expressive_vocabulary, receptive_vocabulary |
| lrp09 | language_composite, reading_decoding, articulation | language_composite, reading_decoding, expressive_vocabulary |
| lrp10 | language_composite, receptive_vocabulary, reading_word | receptive_vocabulary, reading_word, language_composite |
| lrp11 | language_composite, expressive_vocabulary, articulation | language_composite, articulation, expressive_vocabulary |
| lrp12 | receptive_vocabulary, expressive_vocabulary, language_composite | expressive_vocabulary, receptive_vocabulary, social |
| lrp13 | reading_decoding, phonological_memory, language_composite | reading_decoding, reading_word, phonological_memory |
| lrp14 | reading_decoding, reading_word, language_composite | reading_word, reading_decoding, language_composite |
| lrp15 | reading_decoding, expressive_vocabulary, receptive_vocabulary | reading_decoding, expressive_vocabulary, intervention |
| lrp16 | reading_word, receptive_vocabulary, reading_decoding | reading_word, receptive_vocabulary, reading_decoding |
| lrp17 | language_composite, phonological_memory, reading_decoding | language_composite, phonological_memory, demographics_child |
| lrp18 | language_composite, phonological_memory, articulation | language_composite, reading_decoding, receptive_vocabulary |
| lrp19 | language_composite, receptive_vocabulary, articulation | language_composite, receptive_vocabulary, articulation |
| lrp20 | language_composite, expressive_vocabulary, receptive_vocabulary | language_composite, expressive_vocabulary, receptive_vocabulary |
| lrp21 | articulation, phonological_memory, expressive_vocabulary | articulation, reading_word, expressive_vocabulary |
| lrp22 | articulation, phonological_memory, reading_word | articulation, reading_word, study_structure |

For models where the top-3 set is unchanged (LRP02, LRP06, LRP08, LRP16, LRP19, LRP20), no construct-level narrative needs updating. For the rest, the *families* in play are the same — the relative weight has shifted.

## Stability selection (reporting config only)

Bootstrap correctness fix (subject row repetition for with-replacement draws). The top-5 by `appearance_rate_top_k` shifts modestly for most models. Maximum per-feature appearance-rate change across the 22 models is 0.30 (LRP01, LRP02, LRP07, LRP19) and the mean change ranges 0.02 to 0.13.

Models where the top-1 *stability* feature changed:

- **LRP01**: `attend` (pre) → `age` (post).

For the other 21, the top-1 in the new ranking was already top-1 or top-2 pre-fix.

The shift in `rank_iqr` (rank variability across bootstraps) is larger than the appearance-rate shift in most models — under the old algorithm the "bootstrap" was effectively a fixed sub-sample, so IQR was systematically narrower than it should have been. New IQRs are wider on average, which is the expected (calibrated) behaviour.

## What still wants reviewing

1. **Per-model `index.qmd` narrative text** that quotes specific permutation-importance ranks. Where the previous narrative said "X is the second most important predictor", check the new ranking — for the wide models that ordering has likely moved.
2. **Selection-history rationale**. Any `SelectionStep.notes` justifying a feature drop by reference to its previous permutation rank is now resting on the in-sample, R²-scored value. The selection itself is preserved (the dropped features are still dropped), but if a future variant wants to undo a drop, the new OOF ranking is the better evidence base. The `SelectionStep` docstring in `models/common.py` carries a reader note pointing at this caveat — no per-step rewrites were applied, because the historical notes correctly document the evidence the decision *used*, even though that evidence is now superseded.
3. **Construct importance.** `construct_importance.csv` is a per-model aggregate of `importance_mean` rolled up by construct family. Top-construct claims should hold (the dominant feature is dominant in both views), but the relative ordering of lower-tier constructs (cognition vs language vs reading) may shift in wide models.
4. **LRP55** has not been re-fit here — it is a Bayesian model and was not part of `fit_model.py all`. The LKJ residual fix has no effect on the default (`use_residual_correlation=False`) run, but the next opt-in sensitivity fit will exercise the corrected non-centred construction.

## Reproducing this comparison

```bash
# Pre-snapshot was captured into output/_pre_fix_snapshot/ before the refit.
python scripts/fit_model.py all --config reporting
# Diffs:
python /tmp/compare_refit.py        # writes /tmp/refit_metric_deltas.csv, /tmp/refit_top_features.csv
python /tmp/compare_stability.py    # writes /tmp/refit_stability_diff.csv
```

The two ad-hoc comparison scripts are in `/tmp/` rather than the repo because the snapshot is one-off; treat them as throwaway. If you want a permanent comparison harness, the natural home is `scripts/compare_variants.py` (which already diffs two model dirs but currently focuses on `cv_scores.csv`).
