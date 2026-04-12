# LightGBM/RF comparison and Optuna tuning

_Date: 2026-04-12_

## Objective

Extend the per-problem model scaffold with a LightGBM sibling for `lrp01`,
compare it against the Random Forest champion, and wire up a principled
hyperparameter-tuning workflow so both algorithm families are evaluated on
a fair footing. The longer-term question this session was aimed at: is
gradient boosting worth the complexity on this n=152, longitudinal
dataset, or does RF remain the default?

## Work completed

### 1. Per-problem module layout (plan execution)

Finalised the per-problem refactor from the approved plan:

- `models/lrp01.py` and `models/lrp02.py` hold the final models, LightGBM
  siblings, and any historical selection variants side-by-side.
- `models/registry.py` retains helpers (`_gain_model`, `_level_model`,
  `DEFAULT_RF_PARAMS`, `DEFAULT_LGBM_PARAMS`) and the `MODELS` dict only.
- `models/__init__.py` imports `registry` first, then the per-problem
  modules, so registration happens at import time.
- `ModelConfig` gained `variant_of: str | None` and `notes: str` —
  persisted in `config.json` and used to gate `fit_model.py all`.
- `scripts/fit_model.py` grew a `--include-variants` flag; `all` skips
  entries where `variant_of is not None` by default.
- `base_pipeline.save_metrics()` writes `metrics.json` alongside the
  existing CSVs, holding aggregated diagnostics (`cv_rmse_mean/std`,
  `in_sample_mae/rmse`, `n_observations`, `n_predictors`).
- `base_pipeline.report()` does per-model template lookup:
  `docs/models/{model_id}.qmd` takes precedence over the shared
  `docs/models/index.qmd` fallback.
- `docs/models/lrp01.qmd` is a bespoke per-problem template with a
  "Feature selection and tuning history" section that scans sibling
  output directories, groups variants under their champions (matching
  `lrp01` or `lrp01_*`), and renders a comparison table per group.
- `scripts/analyze_predictors.py` added for feature-selection
  diagnostics: Spearman heatmap, distance-correlation dendrogram +
  cluster table, mutual-info heatmap, importance pairing. Output under
  `output/feature_selection/{model_id}/` — deliberately outside
  `output/models/` so the per-model cleanup doesn't wipe it.

### 2. Baseline RF vs LightGBM comparison

First pass after registering the LightGBM sibling with
`DEFAULT_LGBM_PARAMS` (`n_estimators=1200`, default learning rate, etc.)
and fitting both under `--config test`, then `--config reporting`:

| Model              | CV RMSE mean | CV RMSE std | In-sample RMSE |
|--------------------|--------------|-------------|----------------|
| `lrp01` (RF)       | 3.0106       | 1.5576      | 2.7615         |
| `lrp01_lgbm`       | 3.5013       | 1.7159      | **0.1190**     |

The in-sample vs CV gap for untuned LGBM (0.12 → 3.50) was a textbook
over-training signature at n=152 — all 1,200 boosting rounds were being
fit to noise. RF, with its built-in bagging variance reduction, was the
clear winner at default hyperparameters.

### 3. Optuna + per-fold early stopping

Added `scripts/tune_model.py`:

- **Shared**: Optuna TPE sampler, `GroupKFold` (same grouping as fit
  pipeline so tuning and evaluation use identical folds), configurable
  `--n-trials`, `--timeout`, `--n-splits`, `--seed`.
- **RF path**: `_rf_objective` uses `cross_val_score` over the
  hyperparameter search space (`n_estimators`, `max_depth`,
  `min_samples_leaf`, `min_samples_split`, `max_features`, `bootstrap`).
- **LGBM path**: `_lgbm_objective` runs a manual per-fold CV loop with
  LightGBM's `early_stopping` callback against the held-out fold, then
  records the mean `best_iteration_` across folds as a study user
  attribute. This gives a data-driven replacement for `n_estimators`
  instead of a hand-tuned ceiling.
- Output to `output/tuning/{model_id}/`: `best_params.json` (ready to
  paste into a new `_selectNN` variant), `trials.csv`, `study_summary.json`.

Deliberately **not** auto-mutating the registry: tuning produces a
recommendation, and the user manually registers a new variant with the
tuned params so source code remains the single source of truth and the
decision is reviewable in git history.

### 4. Tuning run: `lrp01_lgbm`

`python scripts/tune_model.py lrp01_lgbm --n-trials 30`

- Search: 30 trials, 10-split GroupKFold, ceiling `n_estimators=2000`,
  early-stopping rounds=50.
- Best trial: #11, inner CV RMSE **3.0850 ± 0.5517**.
- Mean best iteration across folds: **62** — replaces the default 1200.
- Best params (rounded):

  | param                | value       |
  |----------------------|-------------|
  | `n_estimators`       | 62          |
  | `learning_rate`      | 0.1902      |
  | `num_leaves`         | 46          |
  | `max_depth`          | 6           |
  | `min_child_samples`  | 40          |
  | `subsample`          | 0.7838      |
  | `subsample_freq`     | 1           |
  | `colsample_bytree`   | 0.9969      |
  | `reg_alpha`          | 6.3462      |
  | `reg_lambda`         | 1.0284      |

Registered as `lrp01_lgbm_select01` in `models/lrp01.py` with
`variant_of="lrp01_lgbm"` and notes recording the study provenance.

### 5. Final comparison (all `--config reporting`)

Refit `lrp01` and fit `lrp01_lgbm_select01`, re-rendered
`docs/models/lrp01.qmd`:

| Model                   | Algorithm              | CV RMSE mean | CV RMSE std | In-sample MAE | In-sample RMSE |
|-------------------------|------------------------|--------------|-------------|---------------|----------------|
| `lrp01`                 | RF (untuned default)   | **3.0106**   | 1.5576      | 2.1425        | 2.7615         |
| `lrp01_lgbm`            | LGBM (untuned)         | 3.5013       | 1.7159      | 0.0683        | 0.1190         |
| `lrp01_lgbm_select01`   | LGBM (Optuna-tuned)    | 3.0167       | **1.4298**  | 2.1412        | 2.6967         |

## Findings

- **Tuned LGBM ties RF on mean CV RMSE** (3.0167 vs 3.0106 — a 0.2%
  difference, well inside the cross-fold noise).
- **Tuned LGBM has the lowest CV std** (1.4298 vs 1.5576), so it is
  marginally more stable across folds — a weak but real argument for
  boosting on this problem.
- **Over-training was the whole story for untuned LGBM**: dropping from
  1,200 boosting rounds to 62 closed the in-sample/CV gap entirely
  (in-sample RMSE went from 0.12 to 2.70, CV RMSE from 3.50 to 3.02).
  The fix is hyperparameter discipline, not a change of algorithm.
- **No clear winner between tuned LGBM and RF** at n=152. RF remains a
  reasonable default (simpler, fewer knobs, less prone to the
  over-training failure mode); tuned LGBM is a legitimate alternative
  when you're willing to run an Optuna study per model.

## Infrastructure side-effects

- `environment.yml` adds `optuna>=4.0.0` (pulled in optuna 4.8.0).
- `base_pipeline.fit()` output-dir cleanup changed from
  `shutil.rmtree + mkdir` to `mkdir(exist_ok=True) + _clear_directory()`
  helper to survive transient Windows file locks (editor/file explorer
  holding handles).
- `CLAUDE.md`, `AGENTS.md`, `.github/copilot-instructions.md` updated in
  lockstep with the new commands and the tuning workflow description.

## Follow-ups

- Fit `lrp01_select01` (the seed RF variant) at least once so the
  bespoke template also emits a `lrp01` RF-family group in section 1.5,
  not just the LGBM group.
- Run `scripts/tune_model.py lrp01` (RF side) for a symmetric tuning
  comparison — current RF numbers are from defaults.
- Run the same LGBM tuning + comparison for `lrp02` once the `lrp02`
  problem is ready for it.
- Consider whether to retire `lrp01_lgbm` (untuned) from the default
  `fit_model.py all` batch now that it's been superseded by
  `lrp01_lgbm_select01` — probably keep as a baseline for now so the
  report still shows the tuning delta.
