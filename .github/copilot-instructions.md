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
python scripts/fit_model.py LRP01

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

Each model lives in its own module under `src/language_reading_predictors/models/` (e.g., `model_lrp01.py`). Models expose discrete pipeline functions (`prepare_data`, `configure_model`, `cross_validate`, `fit_model`, `evaluate`, etc.) that accept a shared `ModelFitContext` dataclass from `models/common.py`. This allows individual steps to be called from a notebook for debugging. Each model also exposes a `fit()` entry-point that orchestrates the full pipeline and is called by `scripts/fit_model.py`. Artifacts are saved to `output/models/{model_id}/`.

## Notebooks

Notebooks in `notebooks/` use **Jupytext** (synced `.ipynb` and `.py:percent` formats). Edit either format; Jupytext keeps them in sync. Analysis uses Random Forest, permutation importance, and SHAP values.

Notebooks reference a shared external package (`dse_research_utils`) for environment setup and metadata.

## Conventions

- All source files include SPDX license headers: `# SPDX-License-Identifier: AGPL-3.0-or-later`
- Spell checking uses British English (`en-GB`) configured in `.cspell.config.yaml` with a custom allow list at `config/spellcheck/allow-en.txt`.
- The Quarto report (`docs/report/`) uses `execute: freeze: true` — computational output is cached, not re-run on render.
- Build system is Hatch (`pyproject.toml`). Version is read from `src/language_reading_predictors/__init__.py`.

## Licensing

- **Code**: AGPL-3.0
- **Documentation and data**: CC BY 4.0
