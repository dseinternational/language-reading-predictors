# Agents Instructions

> **Keep in sync:** This file, `CLAUDE.md`, and `.github/copilot-instructions.md` share the same content. When updating one, update all three.

## Project Overview

Exploratory research study on predictors of progress in language and reading skills for children with Down syndrome, by Down Syndrome Education International. Work in progress — all data and models are preliminary.

The project takes a deliberate **two-step methodology**:

1. **Exploratory analysis with gradient-boosting models** (LightGBM, permutation importance, SHAP) to learn which predictors matter for each outcome.
2. **Statistical models** (Bayesian, PyMC) for interactions and — where the DAG supports it — causal estimation, with intuitive interpretable estimands and quantified uncertainty.

See `METHODS.md` for the full methodology: the gradient-boosting and Bayesian workflows (fit, tune, select, evaluate, compare, interpret), reporting guidance, conventions, glossary, and references.

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

### Gradient-boosting models (`models/`)

One Python module per problem (`models/lrp01.py`, `models/lrp02.py`), each holding the final model plus its historical selection variants. All use LightGBM (the Random Forest path was retired 2026-04-12 — see `notes/202604121451-lightgbm-model-selection.md`). Shared hyperparameters (`DEFAULT_LGBM_PARAMS`) and helpers (`_gain_model`, `_level_model`) live in `models/registry.py`; predictor sets come from `Predictors.DEFAULT_GAIN` / `Predictors.DEFAULT_LEVEL` in `data_variables.py`, so adding a variable to a group auto-propagates. Models register at import time; downstream code imports `MODELS` from `models.registry`.

`ModelConfig` carries two selection-history fields: `variant_of` (marks a selection variant of another model — variants are **skipped** by `fit_model.py all` unless `--include-variants` is passed) and `notes` (free-text rationale persisted to `config.json`).

Pipelines are class-based: `EstimatorPipeline` (`models/base_pipeline.py`) holds the generic steps, subclasses override only `configure_model()`, and `LGBMPipeline` is the only one registered. `scripts/fit_model.py` dispatches via `cfg.pipeline_cls(cfg, run_config).fit()`, writing `config.json`, `metrics.json`, and CSVs to `output/models/{model_id}/`. Feature-selection diagnostics run on every fit but are skipped in `dev` config. Reports are looked up per model at `docs/models/{model_id}/index.qmd` (variants fall back to their parent's template).

Hyperparameter tuning (`scripts/tune_model.py`) runs an Optuna TPE study under the same `GroupKFold` grouping, writing `best_params.json` to `output/tuning/{model_id}/`. It does **not** mutate the registry — applying tuned params is a manual, reviewable step.

### Statistical models (`statistical_models/`)

Step 2 of the methodology: Bayesian models fit with PyMC. One module per model — `lrpittNN.py` (the DAG-faithful ITT suite + companions), `lrpdidNN.py` (the waitlist-crossover / DiD family), `lrpgfNN.py` / `lrplfNN.py` (the DAG-focused gain- and level-factor families), `lrpalNN.py` (the aligned-40-week per-protocol family) and `lrpNN.py` (mechanism/mediation models) — each defining a `SPEC = ModelSpec(...)` and a `fit(config)` that calls the matching pipeline entry point. Eight families, keyed by `ModelSpec.kind`, each with a factory in `factories.py` and a pipeline in `pipeline.py`:

- **`itt`** — single-outcome intention-to-treat: the uniform DAG-faithful **LRPITT01–11** suite (own baseline + linear age as *precision* terms, no cross-baselines — the ITT effect is identified by the empty adjustment set), with **LRPITT13/13b/14/14b** adding SES adjustment + matched complete-case comparators, and **LRPITT17–24** adding a general-ability (block-design) robustness adjustment across the vocabulary family (TR/TE/UR/UE/R/E) and the reading anchors (W, L). Heavily-floored outcomes (P, N) take a pre-specified **floor rule**: a binary off-floor primary estimand plus a flagged graded secondary. → `build_itt_model` / `fit_itt`.
- **`joint`** — the suite outcomes jointly, optional LKJ residual correlation (**LRPITT12**; the taught-vs-not-taught generalisation contrasts **LRPITT15/15b**) → `build_joint_model` / `fit_joint`.
- **`mechanism`** — adjustment-set dose-response of one measure on another across all phases, with subject random intercepts and optional linear moderation (LRP56–58, 71, 72/72base, 73/73base) → `build_mechanism_model` / `fit_mechanism`.
- **`mediation`** — g-formula NDE/NIE decomposition by counterfactual simulation (LRP59 count mediator, LRP62 Gaussian reading-route composite) → `build_mediation_model` / `fit_mediation`.
- **`did`** — waitlist-crossover / difference-in-differences (**LRPDID01–06**): a within-person replication of the randomised ITT effect, stacking the waitlist arm's untreated P1 vs its crossover P2 with each child as its own control and the immediate arm anchoring the time/maturation trend (Beta-Binomial logit, so the ceiling is respected; optional session-dose response) → `build_did_model` / `fit_did`.
- **`gain_factors`** — DAG-focused ANCOVA on a period's post-score given its own pre-score (**LRPGF01–08**, one per outcome W/R/E/L/P/B/F/T, each with a `b` treated-only companion): stacks every on-intervention and untreated period with a child random intercept (latent general-ability repair). The randomised on-intervention term is the *only* causal coefficient; every covariate (own baseline, linear age, cognitive ability, upstream DAG skills, focal interactions) is an explicit *adjusted association*. SES is excluded (not a DAG node, statistically redundant). Heavily-floored P takes the suite floor rule (`likelihood="bernoulli_offfloor"`: a Bernoulli on the off-floor indicator, treatment marginal an off-floor risk difference). → `build_gain_factors_model` / `fit_gain_factors`.
- **`level_factors`** — the companion *levels* view (**LRPLF01–08**): the score at each timepoint (no own baseline), with group×time and ability×time as per-timepoint coefficient vectors. Only the t2 group contrast (`b_grp_time[1]`) is a clean randomised effect; later timepoints are post-crossover and flagged as associations. → `build_level_factors_model` / `fit_level_factors`.
- **`aligned`** — onset-aligned per-protocol single gain (**LRPAL01–08**, plus a `…d` cumulative-session dose variant): aligns both arms by intervention onset (immediate t1→t3, wait-list t2→t4) into one cross-sectional Beta-Binomial ANCOVA per child (no random intercept). The cohort contrast is **not** randomised — confounded by age-at-onset and cohort/timing — so *no* term is flagged causal; every coefficient is an association, and dose (a collider) enters only the sensitivity variant. → `build_aligned_model` / `fit_aligned`.

All use a Beta-Binomial likelihood on bounded post-score counts via a logit linear predictor. Shared priors live in `priors.py` (shared constructors so the factories can't drift), HSGP helpers in `hsgp.py`, the g-formula in `mediation.py`. Each pipeline runs prior-predictive → NUTS (`nutpie`) → posterior-predictive, plus PSIS-LOO (ArviZ, pointwise) for the `itt`/`joint`/`mechanism`/`gain_factors`/`level_factors`/`aligned` families, then writes `trace.nc` (with the `prior`/`prior_predictive`/`log_prior` groups attached), `config.json`, `diagnostics_summary.json` (the pass/fail convergence gate), `priors_table.csv` (per-parameter distribution + role), diagnostic plots (convergence banner data, Pareto-k, rank, ESS-evolution, LOO-PIT, prior-vs-posterior overlay, τ forest), and family-specific CSVs (`tau_summary.csv`, `rope_summary.csv`, `prior_pushforward.csv`, `mechanism_curve.csv`, `mediation_summary.csv`, `factor_summary.csv`, `cohort_marginal.csv`, ...) to `output/statistical_models/models/{model_id}-{config}/`, copying `docs/models/{model_id}/index.qmd` and the shared `docs/models/_partials/` alongside.

The report templates are **thin** (issue #125): each `docs/models/{model_id}/index.qmd` is a title + model-specific Overview/Model prose + a sequence of `{{< include _partials/… >}}` directives. The shared workflow-order partials (`_header`, `_setup`, `_convergence` [renders first], `_priors`, `_prior_predictive`, `_diagnostics`, `_footer`) plus per-archetype result partials (`_results_itt`, `_results_floored`, `_results_joint`, `_results_factors`) live in `docs/models/_partials/` and are driven by `config.json` + `measures` so prose is not hard-coded. They are copied next to each report at fit time so Quarto includes resolve in the output dir.

Fit with `scripts/fit_statistical_model.py {model_id|all} --config dev|test|reporting [--render] [--target-accept X]` (the script's `MODELS` dict registers modules; sampling presets come from `dse_research_utils`). `scripts/compare_statistical_models.py` writes cross-model comparisons (ITT-vs-joint τ consistency, τ and mechanism-slope forests, nested PSIS-LOO for interaction models) to `output/statistical_models/comparison/`.

## Notebooks

Notebooks in `notebooks/` use **Jupytext** (synced `.ipynb` and `.py:percent` formats). Edit either format; Jupytext keeps them in sync. Some legacy notebooks predate the pipeline refactor and still reference Random Forest — they will be updated separately.

Notebooks reference a shared external package (`dse_research_utils`) for environment setup and metadata.

## Conventions

- All source files include SPDX license headers: `# SPDX-License-Identifier: AGPL-3.0-or-later`
- Spell checking uses British English (`en-GB`) configured in `.cspell.config.yaml` with a custom allow list at `config/spellcheck/allow-en.txt`.
- The Quarto report (`docs/report/`) uses `execute: freeze: true` — computational output is cached, not re-run on render.
- Build system is Hatch (`pyproject.toml`). Version is read from `src/language_reading_predictors/__init__.py`.

## Interpreting & reporting results

Report direction and uncertainty — never a bare ranking or point estimate.

- **Gradient boosting:** read the SHAP beeswarm (`output/models/{model_id}/shap_summary.png`) with the permutation-importance ranking; the two disagree, so state the direction.
- **Bayesian:** check convergence (R-hat ≈ 1.00, ESS, ≤ 1 % divergences) *before* interpreting; report the posterior (mean + 95 % credible interval + tail probability, no p-values); positive τ = intervention helps; only τ is causal — observational couplings (`gamma_cross`, `f_mech`, mediator → outcome) are adjusted associations, never "X drives Y".
- **Notes, issues, PRs:** write for a frequentist-leaning science reader; expand shorthand and read credible intervals in plain words; record decisions a future reader might question as a dated `notes/` note; verify citations and always include DOIs.

Full rationale, workflow, conventions, glossary, and references: **`METHODS.md`**.

## AI-authored content labelling

Content drafted or substantially edited by an AI tool **must** carry a visible label identifying it as AI-authored. This applies to **document drafts, pull requests, issues, and comments on pull requests and issues** — and to similar prose such as `notes/` entries, release notes, and discussion posts.

Put the label at the very top, before the substantive text, naming the specific tool and model you actually are (e.g. `Claude Code/Opus 4.8`, `GitHub Copilot`). Use the form that renders in the target — the GitHub alert and Quarto callout syntaxes are **not** interchangeable:

**GitHub** (pull requests, issues, comments, Markdown viewed on GitHub) — a GitHub alert:

```
> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).
```

**Quarto** (`.qmd` documents — e.g. `docs/report/`, `docs/models/`) — a Quarto callout, because Quarto renders its own `::: {.callout-note}` blocks and does **not** understand GitHub `> [!NOTE]` alerts (they would show as a plain blockquote):

```
::: {.callout-note}
Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).
:::
```

**Plain text** (or anything that renders neither) — an equivalent leading line:

```
Note: Drafted by a LLM-based AI tool (<tool>/<model>).
```

Do not remove or hide a label that another tool has added.

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
