# Copilot Instructions

> **Keep in sync:** This file, `AGENTS.md`, and `CLAUDE.md` share the same content. When updating one, update all three.

## Project Overview

Exploratory research study on predictors of progress in language and reading skills for children with Down syndrome, by Down Syndrome Education International. Work in progress — all data and models are preliminary.

## Environment Setup

Uses conda with Python 3.14. There is a local dependency on `../research/src/python` (dse_research_utils).

```bash
conda env create -f environment.yml
conda activate dse-language-reading-predictors
```

## Commands

```bash
# Run all tests
pytest

# Run a single test file or test
pytest tests/test_stats_utils.py
pytest tests/test_stats_utils.py::test_standardize -v

# Lint
ruff check src/

# Spell check (markdown and Quarto files)
npm run spellcheck

# Fit a model (artifacts saved to output/models/{model_id}/)
python scripts/fit_model.py LRP01                               # dev config (fast, default)
python scripts/fit_model.py LRP01 --config test                 # test config (moderate)
python scripts/fit_model.py LRP01 --config reporting            # full config (production)
python scripts/fit_model.py all --config dev --render           # all final models, render reports
python scripts/fit_model.py all --include-variants --config dev # include selection variants
python scripts/fit_model.py lrp01_select01 --config dev         # run a specific variant

# Hyperparameter tuning with Optuna (output/tuning/{model_id}/)
python scripts/tune_model.py lrp01                 # LGBM, 50 trials, GroupKFold
python scripts/tune_model.py lrp01 --n-trials 200 --timeout 1800

# Preview research report
quarto preview docs/report/

# Render research report (HTML, PDF, DOCX)
quarto render docs/report/
```

## Architecture

The Python package is in `src/language_reading_predictors/` and is installed in editable mode via conda/pip.

### Central data schema (`data_variables.py`)

This is the **source of truth** for all variable names used across notebooks and utils. It defines:

- `Variables` class — column name constants (e.g., `Variables.AGE`, `Variables.GENDER`) and grouped lists (`NUMERIC`, `CATEGORICAL`, `GAINS`, `NEXTS`, `DEMOGRAPHICS`, `COGNITIVE`, `LANGUAGE`, `SPEECH`, `READING`).
- `Categories` class — integer-to-label mappings (e.g., `Categories.GENDER = {1: "Boy", 2: "Girl"}`).

When adding or renaming variables, update `data_variables.py` first — everything else references it.

### Data flow

1. `data_utils.load_data()` reads `data/rli_data_long.csv` and applies dtypes from `data_variables.py`.
2. Longitudinal data has 4 timepoints, grouped by `subject_id`. Derived columns use `_GAIN` (change scores) and `_NEXT` (next timepoint values) suffixes.
3. ML analysis uses GroupKFold cross-validation (grouped by `subject_id`) to prevent data leakage across timepoints for the same subject.

### Module responsibilities

- **ml_utils.py** — RandomizedSearchCV wrapper, cross-validation reporting, GP kernel functions.
- **stats_utils.py** — Standardization, descriptive stats with normality tests, distance correlation matrices, mutual information dissimilarity, hierarchical clustering.
- **plot_utils.py** — Visualization functions. Saves figures to `output/`.

### Models (`models/`)

Models are organised as **one Python module per problem** (`models/lrp01.py`, `models/lrp02.py`). Each module holds the final model and any historical selection variants side-by-side. All models currently use LightGBM (the Random Forest code path was retired on 2026-04-12 after tuned LGBM reached equivalent CV RMSE ~30× faster — see `notes/202604121451-lightgbm-model-selection.md`). Shared hyperparameters (`DEFAULT_LGBM_PARAMS`) and helper functions (`_gain_model`, `_level_model`) live in `models/registry.py`; predictor sets come from `Predictors.DEFAULT_GAIN` / `Predictors.DEFAULT_LEVEL` in `data_variables.py`, so adding a variable to a group auto-propagates. `models/__init__.py` imports `registry` first and then the per-problem modules so registration happens at import time; downstream code still imports `MODELS` from `models.registry`.

`ModelConfig` carries two selection-history fields:

- `variant_of: str | None` — if set, this model is a selection variant of another (e.g. `"lrp01"`). Variants are **skipped** by `fit_model.py all` unless `--include-variants` is passed. Explicit model ids always run.
- `notes: str` — free-text rationale persisted in `config.json` and surfaced in the rendered report.

Pipelines are class-based. The generic steps live on `EstimatorPipeline` in `models/base_pipeline.py` (`prepare_data`, `configure_model`, `cross_validate`, `fit_model`, `evaluate`, `save_metrics`, `permutation_importance_analysis`, `shap_analysis`, `partial_dependence_plots`, `save_config`, `report`, and the `fit()` orchestrator). Each method operates on `self.context` (a `ModelFitContext` from `models/common.py`) so individual steps can still be called from a notebook for debugging. Subclasses override only `configure_model()`: `LGBMPipeline` in `models/lgbm_pipeline.py` (the only estimator pipeline currently registered) builds a sklearn `Pipeline([("est", estimator)])`. `ModelConfig.model_params` holds the estimator kwargs and `ModelConfig.pipeline_cls` selects which pipeline runs the model. `scripts/fit_model.py` discovers models from the registry and dispatches via `cfg.pipeline_cls(cfg, run_config).fit()`. Artifacts are saved to `output/models/{model_id}/`.

Each fit writes two JSON artifacts alongside the CSVs:

- `config.json` — inputs (model_id, pipeline_cls, variant_of, notes, target, predictors, model_params, cv_splits, ...).
- `metrics.json` — aggregated outputs (n_observations, n_predictors, cv_rmse_mean/std, in_sample_mae/rmse). This is the file the cross-variant comparison in bespoke templates reads.

Report templates use per-model lookup: `base_pipeline.report()` checks `docs/models/{model_id}/index.qmd` first, then `docs/models/{variant_of}/index.qmd` for selection variants. Each model that needs a report must have its own template (e.g. `docs/models/lrp01/index.qmd`).

Feature-selection diagnostics (Spearman correlation, distance-correlation dendrogram + cluster table, mutual-information heatmap, and importance pairing) are produced as part of every model fit by `EstimatorPipeline.feature_selection_diagnostics()`. Output lives alongside other model artifacts in `output/models/{model_id}/`. The diagnostics are skipped in `dev` run config for speed but run in `test` and `reporting` configs.

Hyperparameter tuning is driven by `scripts/tune_model.py`, which runs an Optuna TPE study using the same `GroupKFold` grouping as the fit pipeline. The tuning loop carves an inner `GroupShuffleSplit` slice out of each training fold for early stopping — the outer val fold is never shown to `early_stopping`, so the reported CV RMSE and `best_iteration_` are independent. The mean best iteration across folds is saved as the tuned `n_estimators`. Output lives in `output/tuning/{model_id}/` and contains `best_params.json` (ready to paste into a new `_selectNN` variant or back into the final model), `trials.csv`, and `study_summary.json`. The tuning script does **not** automatically mutate the registry — updating the model config is a manual, reviewable step so the source remains the single source of truth.

## Notebooks

Notebooks in `notebooks/` use **Jupytext** (synced `.ipynb` and `.py:percent` formats). Edit either format; Jupytext keeps them in sync. Some legacy notebooks predate the pipeline refactor and still reference Random Forest — they will be updated separately.

Notebooks reference a shared external package (`dse_research_utils`) for environment setup and metadata.

## Conventions

- All source files include SPDX license headers: `# SPDX-License-Identifier: AGPL-3.0-or-later`
- Spell checking uses British English (`en-GB`) configured in `.cspell.config.yaml` with a custom allow list at `config/spellcheck/allow-en.txt`.
- The Quarto report (`docs/report/`) uses `execute: freeze: true` — computational output is cached, not re-run on render.
- Build system is Hatch (`pyproject.toml`). Version is read from `src/language_reading_predictors/__init__.py`.

## Interpreting model results

When describing what a fitted model indicates — whether writing notes, report sections, PR bodies, or commit messages — **inspect the SHAP beeswarm plot (`output/models/{model_id}/shap_summary.png`) alongside the permutation-importance ranking**. Permutation importance tells you *how much* each feature contributes; the beeswarm shows *which direction* the effect runs and *how consistently*. The two frequently disagree in interpretively important ways:

- A predictor can rank #1 on importance while acting in the opposite direction to every other predictor (e.g. LRP03's `eowpvt` — lower baseline vocabulary predicts more gain, a regression-to-the-mean signal).
- A predictor can rank #1 on importance while having a non-monotonic or mixed-sign effect (e.g. LRP01's `age`).
- Two predictors with similar importance can have opposite directions (e.g. LRP02's `agebooks` / `agespeak` are negatively directional while `mumedupost16` is positive).

Check each top predictor's beeswarm row for: (i) whether blue (low feature value) and red (high) cluster cleanly on opposite sides of zero (monotonic), (ii) whether the effect is tight or wide, (iii) whether any tails show observations acting opposite to the dominant direction. Report the direction explicitly when describing results — don't leave readers to infer it from importance alone.

## Pre-commit checks

Before creating a commit or opening a pull request, both of the following must pass:

```bash
ruff check src/         # Python lint
npm run spellcheck      # Markdown + Quarto spelling (British English, en-GB)
```

If `ruff` reports issues, fix them — do not silence rules or add blanket `noqa` pragmas without justification.

If `cspell` flags a legitimate term (Python identifier, package name, domain term, project acronym, British spelling not in the base dictionary), add it to `config/spellcheck/allow-en.txt` rather than rewording the prose. Only add terms that are genuinely correct — do not use the allow list to paper over actual typos.

Do not bypass these checks with `--no-verify`, skipped CI, or by committing from a different working tree. If either command cannot run (e.g. the conda env is inactive, `npm` is missing), resolve the setup issue rather than proceeding.

## Licensing

- **Code**: AGPL-3.0
- **Documentation and data**: CC BY 4.0
