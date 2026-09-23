<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Gradient-boosting objective sensitivity: MAE against Huber, squared error and Poisson on four models

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5.1).

Date: 2026-09-22 — **Status: EVIDENCE; decision taken.** The author adopted Huber for both families on 2026-09-22 (PR #686 discussion); the retune and refit are recorded in [the Huber retune note](202609221800-gb-huber-retune-refit.md).

## The question

All 50 gradient-boosting models use LightGBM's MAE objective, adopted as a tuning policy on 2026-07-08 (`notes/202607081406-issue-169-gb-hyperparameter-retune.md`) without a recorded scientific rationale. MAE fits the conditional median, its gradients carry only a sign, and permutation importance is scored in held-out RMSE, which the objective never optimises. Two symptoms prompted the check: the word-reading level model was tuned to 11 trees and its partial-dependence curves are coarse staircases, and the word-reading gain model's SHAP dependence plots place age at a single threshold, with every child-wave on one of two bands. This note records a bounded check of whether the objective changes the predictor rankings, directions, dependence shapes and fit quality that the boosting layer reports, so that a decision can be taken on evidence rather than on the original policy.

## Protocol

Four models were chosen to span the cases that matter: `lrp-rli-gbl-012` (word-reading level; right-skewed, floored, 11 trees under MAE), `lrp-rli-gbl-013` (nonword level; 57 % of rows at zero), `lrp-rli-gbl-006` (expressive vocabulary level; roughly symmetric, well predicted) and `lrp-rli-gbg-012` (word-reading gain; near noise, can be negative).

Each model was fitted under five arms. `registry` fits the committed MAE-tuned parameters as a reproduction baseline. `mae`, `huber`, `l2` and `poisson` each re-tune from scratch with `scripts/tune_model.py` under the #169 protocol (150 Optuna trials, seed 47, child-grouped folds at the model's own `cv_splits`, inner early-stopping slice), with the scoring metric matched to the objective: MAE scoring for the MAE arm and RMSE scoring for the three mean-targeting arms. Poisson applies to level models only, because gains can be negative. The Huber threshold was set per model to 1.345 times a robust scale estimate (1.4826 × MAD of the target): 12.96 items for word-reading level, 3.99 for word-reading gain and 17.95 for expressive vocabulary; nonword's MAD is zero, so it used 1.345 times the mean absolute deviation from the median, 1.67. Every arm was then fitted at the reporting tier (51 folds, or 53 for the gain model; 50 subject-block permutation repeats; 30 subject-bootstrap draws; SHAP; clusters) into a scratch output root, so nothing under `output/` changed.

The code state was commit `927c14d5` on the committed `data/rli_data_long.csv` (SHA-256 prefix `1b85c53a4520`). The `registry` arms reproduce the stored 22 September fits in `output/models/` to every printed decimal for both word-reading models, so the committed plots discussed below are the stored ones. The driver, comparison script, tuned parameters, summary tables and six plots are in `notes/assets/` under the prefix `202609221700-objective-sensitivity-`. The full fit artefacts remain in the session scratch folder and are not retained.

The 34 stages (15 tunes, 19 fits) ran without error in 2 h 42 min. Tunes took 2 to 24 min (mean 8); reporting-tier fits took about 2 min each.

## Results

### Fit quality (pooled out-of-fold, post-selection cross-validation)

| Model                         | Arm      | Trees | OOF MAE | OOF RMSE | OOF R² | In-sample R² |
| ----------------------------- | -------- | ----: | ------: | -------: | -----: | -----------: |
| Word-reading level (0–64)     | registry |    11 |    6.15 |     9.45 |  0.578 |        0.828 |
|                               | mae      |    19 |    6.37 |     9.96 |  0.531 |        0.738 |
|                               | huber    |    54 |    6.06 |     8.77 |  0.637 |        0.934 |
|                               | l2       |    13 |    6.19 |     8.72 |  0.640 |        0.943 |
|                               | poisson  |    56 |    5.99 |     8.91 |  0.625 |        0.958 |
| Nonword level (0–6)           | registry |   118 |    0.86 |     1.37 |  0.436 |        0.689 |
|                               | mae      |   200 |    0.85 |     1.35 |  0.456 |        0.678 |
|                               | huber    |   139 |    0.91 |     1.31 |  0.490 |        0.799 |
|                               | l2       |   218 |    0.95 |     1.31 |  0.488 |        0.808 |
|                               | poisson  |   312 |    0.94 |     1.40 |  0.412 |        0.893 |
| Expressive vocabulary level   | registry |   103 |    6.07 |     7.66 |  0.709 |        0.959 |
|                               | mae      |   562 |    5.96 |     7.58 |  0.715 |        0.940 |
|                               | huber    |   103 |    5.93 |     7.49 |  0.722 |        0.945 |
|                               | l2       |   137 |    5.63 |     7.23 |  0.741 |        0.934 |
|                               | poisson  |   843 |    5.67 |     7.23 |  0.741 |        0.937 |
| Word-reading gain (−4 to +21) | registry |   193 |    2.98 |     4.16 |  0.083 |        0.300 |
|                               | mae      |   327 |    2.97 |     4.17 |  0.078 |        0.255 |
|                               | huber    |    94 |    3.01 |     4.02 |  0.140 |        0.495 |
|                               | l2       |    43 |    3.10 |     4.12 |  0.099 |        0.492 |

Huber and squared error raise out-of-fold R² over the re-tuned MAE arm in every model, by 0.01 to 0.11, and lower RMSE in every model. Absolute error is mixed: in word-reading and expressive-vocabulary level the mean-targeting arms also beat the re-tuned MAE arm on MAE, while in nonword level and word-reading gain the MAE arms keep the lowest absolute error, by 0.04 to 0.13 items. Poisson fits the most trees, has the widest train-to-test gap of any arm in the two skewed level models, and on nonword, the case it is nominally built for, has the lowest out-of-fold R² of the five arms.

### Ranking agreement

Spearman correlation of the permutation-importance vector over all predictors:

| Model                       | registry vs mae (tuning noise) | mae vs huber | mae vs l2 | huber vs l2 | huber vs poisson |
| --------------------------- | -----------------------------: | -----------: | --------: | ----------: | ---------------: |
| Word-reading level          |                           0.53 |         0.65 |      0.40 |        0.75 |             0.65 |
| Nonword level               |                           0.69 |         0.68 |      0.57 |        0.87 |             0.65 |
| Expressive vocabulary level |                           0.81 |         0.79 |      0.86 |        0.83 |             0.87 |
| Word-reading gain           |                           0.95 |         0.66 |      0.59 |        0.58 |                – |

The first column is the noise floor: the same objective, re-tuned under the same protocol on the same data, does not reproduce the committed parameters (for word-reading level, 19 trees against 11) and moves the ranking as much as a change of objective does in the three level models. Only the gain model separates the two: its committed and re-tuned MAE rankings agree at 0.95, while switching to Huber or squared error drops agreement to about 0.6.

Top predictors and replicated labels (permutation z ≥ 2 and top-five appearance ≥ 0.5 over the bootstrap draws):

| Model                       | Stable under every arm                                                                                                                                                     | Moves with the arm                                                                                                                                                                                                                                                                                            |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Word-reading level          | Phonetic spelling first (z 4.3–6.4) and letter sounds second or third (z 3.4–4.3), both replicated in every arm                                                            | Word repetition (second to sixth) and nonword reading (fourth to seventh) in the top seven everywhere but replicated only in some arms; basic concept knowledge fifth under the committed fit and 31st under re-tuned MAE; expressive information 19th under the committed fit and second under squared error |
| Nonword level               | Word reading and phonetic spelling first and second in four arms, first and third under Poisson; both replicated in every arm                                              | Letter sounds fifth under the MAE arms, second or third and replicated under Huber and squared error; blending 31st or 32nd under the MAE arms and fourth under Poisson                                                                                                                                       |
| Expressive vocabulary level | Taught expressive vocabulary first, with expressive information, receptive vocabulary and basic concept knowledge filling the next three places in every arm (all z ≥ 3.0) | Age fifth in four arms and sixth under re-tuned MAE; the replicated set has six members under the committed fit and four or five elsewhere                                                                                                                                                                    |
| Word-reading gain           | Age first and negative (z 2.7–3.0), replicated in every arm                                                                                                                | Letter sounds second and replicated under MAE, third and not replicated under Huber and squared error; expressive vocabulary sixth to ninth under MAE, second and replicated under Huber and squared error                                                                                                    |

### Direction

Among predictors with z ≥ 2 in any arm, one direction flip occurred: receptive taught vocabulary in the nonword model is positive and monotonic under both MAE arms but negative and noisy under Huber, squared error and Poisson, with z between 1.5 and 2.4 throughout. Expressive grammar in the word-reading level model turns negative and noisy under squared error at z 1.4. Every other top predictor keeps its sign across all five arms.

### Shape of the dependence plots

The objective changes the shape of the dependence plots more than it changes who is on top, but not in the same way for every plot.

- **Gain model, age.** Under the committed MAE fit (193 trees), every child-wave sits on one of two bands: about +0.7 to +0.8 items below 87 months and −0.85 to −1.0 items above 92 months, with a handful of points in between. Under Huber (94 trees) the same predictor is a continuous decline from about +1.2 items at 60 months to −1.1 above 95 months, steepest between 85 and 95 months; squared error (43 trees) gives the same curve from +1.5. All three agree on direction and span about 2 items across the age range, so the single threshold is a property of the MAE fit, not of the age effect (`-gbg012-age-mae.png`, `-gbg012-age-huber.png`).
- **Gain model, letter sounds.** MAE places the contribution on a handful of flat bands between −0.9 and +0.6 items. Huber is continuous below 24 letter sounds but still splits into two bands, at about +0.5 and +0.75, above it (`-gbg012-yarclet-mae.png`, `-gbg012-yarclet-huber.png`). Banding is reduced, not removed.
- **Level model, letter sounds.** The partial-dependence curve is a staircase under MAE (flat at 8.7 items to 15 letter sounds, then four steps to 13.1) and a smooth accelerating curve under Huber (flat at 10 to 15 letter sounds, then rising to 16) (`-gbl012-pdp-yarclet-mae.png`, `-gbl012-pdp-yarclet-huber.png`). The SHAP scatter for the same predictor is continuous under every arm: from about −3 to +5 items under MAE and −2 to +7 under Huber and squared error (`-gbl012-yarclet-mae.png`, `-gbl012-yarclet-huber.png`). The 11-tree MAE fit is not stepped in its SHAP scatter; it is stepped in its partial dependence, which averages the model over the sample and so exposes the few split thresholds directly.

## Interpretation

- **The top of each ranking is robust to the objective.** The first predictor is unchanged in all four models. Second place changes in every model under at least one arm, but in the three level models the same three or four predictors fill the leading places throughout. This is the part of the boosting layer the findings lean on, and it survives.
- **Below the leading places, rankings are within tuning noise whichever objective is used.** The committed-versus-re-tuned MAE comparison shows this without any change of objective. Reports should say so, and the "replicated" label should be read as fragile beyond the leading predictors, which the 22 September refit note already found for four labels near the 0.5 cut-off.
- **MAE produces coarser dependence structure than the mean-targeting objectives, but not uniformly.** It places the gain model's age effect on a single threshold that Huber and squared error spread into a continuous decline, and it makes the level model's partial dependence a staircase where Huber's is smooth. It does not make the level model's SHAP scatter stepped. Plot captions should state which artefact they describe and should not read a threshold into a band.
- **Huber and squared error behave alike.** Their rankings agree at 0.75 to 0.87 in the level models, they reach similar held-out error, and Huber keeps robustness to the extreme scorers that motivated MAE. Squared error trims absolute error less well on the two skewed targets.
- **Poisson does not earn its place.** It is the natural likelihood for a floored count, but it overfits most, is worst out of fold on the floored target, cannot be used for gains, and puts SHAP values in log-mean units that the reports would have to relabel.
- **The gain model is near noise under every objective** (out-of-fold R² 0.08 to 0.14), so the objective cannot rescue it, but the objective does decide which of letter sounds or expressive vocabulary sits second, which the gain findings should not lean on.

## Recommendation for the author's decision

1. Adopt Huber as the single objective for both families, with the threshold recorded per model as 1.345 × 1.4826 × MAD (falling back to 1.345 × mean absolute deviation from the median when the MAD is zero), and re-tune all 50 models under the #169 protocol with RMSE scoring. The July batch took about 4 h; this run's tunes averaged 8 min.
2. Whichever objective is kept, align the permutation-importance metric with it or report both metrics side by side and state which the ranking uses. Today the ranking metric (RMSE) and the objective (MAE) pull in different directions.
3. Label predicted values in the boosting reports, including the partial-dependence and SHAP plots, as conditional medians while MAE remains, and as conditional means once a mean-targeting objective is adopted.
4. Record the tuning-noise finding in the boosting findings review: rankings beyond the leading predictors move as much under re-tuning as under a change of objective.
5. Keep Poisson, if at all, as a recorded sensitivity for the heavily floored level outcomes, not as a model of record.

## What this check does not establish

Four of 50 models were checked, under one seed. With one seed the tuning-noise floor is a single draw, so a multi-seed re-tune would be needed to separate objective effects from tuning variation with any precision. All metrics are post-selection cross-validation on the same grouped folds the tuner used, so they compare arms but do not estimate generalisation. The Huber threshold is a per-model constant that was not itself tuned. Dependence shapes were compared by eye on two predictors in two models, not measured. The SHAP magnitudes of the Poisson arm are on a different scale from the others and were compared by direction only. None of this changes the standing reading of the boosting layer: level models describe concurrent associations between tests and gain models are near noise.
