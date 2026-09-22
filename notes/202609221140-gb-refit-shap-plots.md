<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Gradient-boosting refit after the SHAP plot fixes, 2026-09-22

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

This batch refitted all 50 registered LightGBM models at the `reporting` configuration and rendered every report. It was run to redraw the SHAP scatter and interaction plots after #681 made overlapping observations visible and #682 restored four dependence-pair scatter plots to `lrp-rli-gbg-012`. It covers the gradient-boosting layer only. Nothing was published to the public research site.

**Outcome in one paragraph.** All 50 fits completed, rendered and recorded a clean source commit. Compared with the 7 September fits at `173dc0aa`, every held-out metric, permutation importance, SHAP summary and SHAP direction is identical, and every model's five most important predictors are unchanged in order. Only the bootstrap stability of the rankings moved. #678 changed that check to shuffle whole children, and the largest effect is that the assessment occasion (`time`) no longer appears as a stable top-five predictor. This is the first boosting batch whose fits record their own source commit, data digest and environment lock, closing the provenance gap recorded in the [8 September rebuild](202609080119-full-rebuild-both-layers.md). The cross-layer comparisons, rerun against the 21 September statistical fits, reproduce the 8 September results.

## Run record

| Item            | Value                                                                                                 |
| --------------- | ----------------------------------------------------------------------------------------------------- |
| Output          | `output/models/<model_id>/`, the default output root                                                  |
| Sweep commit    | `494b93e0` (#683), from a detached clean worktree at `.claude/worktrees/gb-refit-20260922`            |
| Environment     | `uv sync --locked` unchanged; lock digest `1e1f5fb9…` (7 September: `a2e5c1a6…`)                      |
| Data            | `data/rli_data_long.csv`, SHA-256 `1b85c53a…`, identical to the tracked file at the sweep commit      |
| Driver          | `scripts/run_refit_sweep.py gb --config reporting --render`, one stream                               |
| Fit phase       | 10:13–11:32 BST, 1 h 19 m (7 September: 1 h 26 m); 72–193 s per model                                 |
| Fitted / failed | 50 / 0; no variants are registered                                                                    |
| Completion      | 50/50 record `fit_complete: true`, commit `494b93e0`, `dirty: false`, `run_config: reporting`         |
| Backup          | the 7 September fits were copied to `output/models.pre-shap-refit-20260922/` before the sweep started |

The sweep ran from its own worktree, as recommended after the [21 September refit](20260921-full-statistical-refit.md), so edits in the shared checkout could not mark the fits dirty. The stored fits did not satisfy the driver's reuse rule because they predate the `fit_complete` flag, so all 50 were refitted.

## What changed since the 7 September fits

| Change                                                  | Effect on stored artefacts                                                                                                                                  |
| ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| #681 SHAP scatter visibility                            | Scatter and interaction plots use transparency and a small seeded horizontal jitter                                                                         |
| #682 `gbg-012` dependence pairs                         | Four more scatter plots in `gbg-012` (70, previously 66), shown under "Selected dependence pairs"                                                           |
| #678 bootstrap importance and provenance                | `stability_selection.csv` and the `topk_freq` column of `predictor_ranking.csv` change; `config.json` gains provenance; `metrics.json` gains `fit_complete` |
| `dse-research-utils` 0.14.0 → 0.15.2 (#668, #670, #677) | No numerical effect found, although the environment lock digest changed                                                                                     |

Every other CSV in all 50 fit directories is byte-identical to the 7 September fit, including `permutation_importance.csv`, `shap_direction_diagnostics.csv`, `oof_predictions.csv` and the SHAP interaction tables. Within `predictor_ranking.csv`, only `topk_freq` differs, and `metrics.json` differs only by the added flag. The pooled held-out R² values in the [8 September rebuild](202609080119-full-rebuild-both-layers.md) therefore stand unchanged.

## Bootstrap stability of the rankings

The stability check refits each model on 30 bootstrap samples of children and records how often each predictor ranks in the top five by permutation importance. Before #678 it shuffled rows, which moved values between children and also scrambled each child's own sequence of assessments. The main permutation importance already moved whole children, remapping each child's rows to a donor child's rows in the same within-child order. #678 applied that scheme to the bootstrap check too.

Across the 1,632 model–predictor pairs, 46% of top-five frequencies changed, but the median change is zero and the 90th percentile is 0.07. Thirteen pairs moved by 0.2 or more, and ten of those are `time`. Averaged over the 50 models, `time`'s top-five frequency fell from 0.12 to 0.02. It fell to zero in seven models where it had been between 0.23 and 0.63, among them `gbg-009` (letter-sound gain, 0.63 → 0) and `gbg-001` (taught receptive vocabulary gain, 0.57 → 0). Under whole-child permutation a child keeps nearly the same assessment occasions, so shuffling `time` barely changes the data. Its earlier stability came from breaking each child's order, which the main ranking never did. Its main permutation rank is unchanged (between 5th and 30th across the models).

The three other pairs that moved by 0.2 or more are `age` in `gbg-013` (nonword reading gain, 0.07 → 0.30), `yarclet` in `gbl-002` (taught expressive vocabulary level, 0.63 → 0.37) and `trog` in `gbl-028` (language-sample total words, 0.73 → 0.50). Among each model's three most important predictors, the largest change is `trog` in `gbl-028`. Top-five frequencies are descriptive: 30 bootstrap refits give each a resolution of 1/30, and they are not confidence levels.

The [1 September review](202609012030-gb-findings-review.md) calls a predictor **replicated** when its permutation z is at least 2 and it is in the top five in at least half the bootstraps. The z values are unchanged, so under that rule only five labels change relative to the 7 September fits, all in level models, and the total rises from 134 to 137 model–predictor pairs. Receptive grammar (`trog`) in `gbl-002` (0.37 → 0.53), `age` in `gbl-006` (expressive vocabulary level, 0.43 → 0.50), word repetition (`erbword`) in `gbl-012` (word reading level, 0.47 → 0.50) and initial-consonant articulation (`deappin`) in `gbl-019` (total repetition level, 0.47 → 0.57) become replicated. Letter sounds (`yarclet`) in `gbl-002` (0.63 → 0.37) becomes borderline, so the review's statement that letter sounds replicate for that model no longer holds. All four new labels sit within 0.07 of the 0.5 cut-off, so they are as fragile as the threshold they cross. No gain-model label changes: `time` never reached z ≥ 2 in any model.

## Cross-layer comparisons

The comparisons were run against the [21 September](20260921-full-statistical-refit.md) statistical run root, `output/runs/20260921T084416Z-b22ea254c1f0/`, so that the newest fits of both layers meet. `compare_gb_vs_statistical.py` reads both layers from one output root, so the 50 boosting fits were copied into that root's `models/` directory. `rank_predictors.py` rebuilt the same 12 ranking directories as on 8 September (gain and level models 009, 012, 017, 018, 020 and 021) under its `ranking/` directory, and the comparison tables went to `statistical_models/comparison/`. The driver, `run_metadata/run_gb_cross_layer.sh`, ran from the clean `494b93e0` worktree. All 17 steps exited cleanly, and their logs are in `run_metadata/gb_cross_layer/`.

Every result reproduces the 8 September comparison. In the 12 rankings, only the bootstrap `topk_freq` column changes, as in the fits; cluster rankings and every other column are identical. The boosting side of each comparison is therefore unchanged, and the statistical side moved only within sampling noise.

| Horseshoe | Boosting  | Outcome            | Shared constructs | Spearman ρ | Top-three overlap                         |
| --------- | --------- | ------------------ | ----------------- | ---------- | ----------------------------------------- |
| `hs-001`  | `gbg-012` | Word reading gain  | 8                 | +0.50      | 2/3: letter sounds, age                   |
| `hs-002`  | `gbl-012` | Word reading level | 7                 | +0.32      | 2/3: letter sounds, expressive vocabulary |
| `hs-003`  | `gbg-009` | Letter-sound gain  | 8                 | −0.33      | 1/3: basic concepts                       |
| `hs-004`  | `gbl-009` | Letter-sound level | 7                 | +0.75      | 2/3: word reading, phoneme blending       |

Each horseshoe order is unchanged, and the posterior probability that a coefficient exceeds the practical threshold, P(|β| > δ), moved by at most 0.010 between the two statistical batches. The comparison with the structural families is identical to full precision. For word-reading gain against the gain-factor model (`gf-001`), 7 shared constructs give ρ = −0.11. For word-reading level against the pooled-levels between-child coefficients, 5 constructs give ρ = +0.70.

These correlations rest on five to eight constructs, so they describe agreement between two rankings rather than test it. They compare different quantities: permutation importance in a tree ensemble against the size of adjusted linear coefficients. The negative gain correlation is driven by the own baseline. The gain-factor model predicts the next score, so its word-reading coefficient (median 0.75) mainly reflects how scores persist between assessments, whereas the boosting model predicts the change, where the baseline has almost no permutation importance. Without it, the other six constructs give ρ = +0.43. The weakest agreement is for letter-sound gain. Among the constructs the comparison maps, the horseshoe leads with receptive grammar, basic concepts and word reading, while the boosting model leads with the letter-sound baseline, basic concepts and age. Which predictors matter most for that outcome depends on the method.

## Interpretation limits

The refit changes no ranking, direction or held-out performance. It revises only the review's statements that rest on bootstrap stability: the five replicated labels above and the observation that `time` is in the top five of half the bootstraps for four gain models. The review was written from the 31 August fits; this comparison is against the 7 September fits. Pooled held-out R² is internal performance after tuning on the same child-grouped folds, not independent validation. Permutation importance and SHAP values describe the fitted predictions: they are associations, not effects of changing a predictor. Same-skill predictors can restate a level outcome, and gain models include the baseline score, so a negative baseline association can reflect score limits or regression to the mean.

## Report index

`scripts/build_gb_index.py` (added with this note) writes `output/models/index.html` from the stored artefacts. The page links all 50 reports, grouped as in the [model catalogue](../docs/models/README.md), with each model's held-out R², MAE and three leading predictors with SHAP direction and bootstrap stability. It flags fits without a completion record, fits that have not been rendered and any source that was not recorded as clean. For this batch it reports 50 fitted models from `494b93e0` with a clean source tree and no flags.

## Residuals and follow-up

- The default `output/ranking/` and `output/statistical_models/comparison/` still hold the 8 September comparison output. The current versions are in the 21 September run root, which now also holds a copy of the 50 boosting fits.
- Each `config.json` records `data_path` as an absolute path inside the sweep worktree. The driver's reuse check re-digests that path, so removing `.claude/worktrees/gb-refit-20260922` makes these fits look stale to a later resumed sweep, which would refit them. The digest itself matches the tracked data file.
- In the restored `yarclet` by `blending` plot the colour bar is labelled in elevenths (1.818, 3.636, …) rather than whole scores. This is cosmetic and was not investigated.
- The backup at `output/models.pre-shap-refit-20260922/` (568 MB) can be deleted once this batch has been reviewed.
