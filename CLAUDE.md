# Claude Instructions

> [!NOTE]
> Substantially edited by a LLM-based AI tool (Codex/GPT-5).
>
> Divergent-transition qualification policy updated by a LLM-based AI tool (Codex/GPT-5).
>
> Phoneme-blending link-sensitivity policy updated by a LLM-based AI tool (Codex/GPT-5).

> **Keep in sync:** This file, `AGENTS.md`, and `.github/copilot-instructions.md` share the same content. When updating one, update all three.

## Project Overview

Exploratory research study on predictors of progress in language and reading skills for children with Down syndrome, by Down Syndrome Education International. Work in progress — all data and models are preliminary.

The project takes a deliberate **two-step methodology**:

1. **Exploratory analysis with gradient-boosting models** (LightGBM, permutation importance, SHAP) to learn which predictors matter for each outcome.
2. **Statistical models** (Bayesian, PyMC) for interactions and — where the DAG supports it — causal estimation, with intuitive interpretable estimands and quantified uncertainty.

See `METHODS.md` for the full methodology: the gradient-boosting and Bayesian workflows (fit, tune, select, evaluate, compare, interpret), reporting guidance, conventions, glossary, and references.

## Environment Setup

Hybrid two-layer environment (shared across DSE research repos): the compiled scientific core (`numpy`/`scipy`/`pandas`/`pymc`/`nutpie`/`jax`/`arviz`, …) comes from **conda-forge** and must match the canonical spec shipped in `dse-research-utils` (verify with `dse-check-env environment.yml`); the pure-Python tail and the shared library install in the pip layer. `dse-research-utils` installs from the public git tag `v0.9.0` (`dse-research-utils[viz,notebook,dependence,tuning,io] @ git+https://github.com/dseinternational/research.git@v0.9.0#subdirectory=src/python`); a commented local-dev override in `environment.yml` points at a sibling `../research/src/python` checkout instead. On Windows there is no conda-forge `jax`/`jaxlib` win-64 build, so use **WSL** (Ubuntu, linux-64). GPU acceleration is an opt-in `jax[cuda]` overlay.

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

# Format Markdown
npm run format
npm run format:check

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

**Output location.** Runs write under a configurable **output root** (default: repo-local `output/`, with the relative layout unchanged). Redirect it to a scratch disk for VM runs via the `DSE_LRP_OUTPUT_DIR` environment variable, or per command with `--output-dir` (which takes precedence): `DSE_LRP_OUTPUT_DIR=/mnt/scratch/lrp python scripts/fit_statistical_model.py lrp-rli-itt-010 --config reporting`, or `python scripts/fit_model.py lrp-rli-gbg-012 --output-dir /mnt/scratch/lrp`. Resolution lives in `src/language_reading_predictors/paths.py`; the resolved root is printed at the start of each long-running command and recorded in `config.json`. Scratch disks are ephemeral — `--upload` (or copy) durable artefacts before teardown.

## Architecture

The Python package is in `src/language_reading_predictors/` and is installed in editable mode via conda/pip.

### Central data schema (`data_variables.py`)

This is the **source of truth** for all variable names used across notebooks and utils. It defines:

- `Variables` class — column name constants (e.g., `Variables.AGE`, `Variables.GENDER`) and grouped lists (`NUMERIC`, `CATEGORICAL`, `GAINS`, `NEXTS`, `DEMOGRAPHICS`, `COGNITIVE`, `LANGUAGE`, `SPEECH`, `READING`).
- `Categories` class — integer-to-label mappings (e.g., `Categories.GENDER = {1: "Male", 2: "Female"}`).

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

One Python module per outcome, named `models/lrp_rli_gbg_NNN.py` (gain models) and `models/lrp_rli_gbl_NNN.py` (level models), each holding the model definition. All use LightGBM (the Random Forest path was retired 2026-04-12). Models are **declarative classes**: `ModelDefinition` and its `GainModel` / `LevelModel` subclasses (`models/base_model.py`) configure a model through class-level attributes (`target_var`, `include`, `exclude`, `params`, `cv_splits`, …), and any concrete class (one that sets `model_id`) auto-registers into the global `MODELS` dict at import time via `__init_subclass__`. Predictor sets come from `Predictors.DEFAULT_GAIN` / `Predictors.DEFAULT_LEVEL` in `data_variables.py`, so adding a variable to a group auto-propagates. `models/registry.py` is now a thin re-export of `MODELS`; downstream code still imports `MODELS` from `models.registry`. The per-model dataclasses (`ModelConfig`, `RunConfig`, `ModelFitContext`, `ShapScatterSpec`) live in `models/common.py`.

A model may mark itself a variant of another via `variant_of` (variants are **skipped** by `fit_model.py all` unless `--include-variants` is passed); `notes` holds free-text rationale persisted to `config.json`. Hard feature selection was retired in #116 Phase D — models fit the full `DEFAULT_*` set and the ranking is the deliverable.

Pipelines are class-based: `EstimatorPipeline` (`models/base_pipeline.py`) holds the generic steps; subclasses override only `configure_model()` / `_wrap_estimator()`. `LGBMPipeline` is the default, with `LGBMLogPipeline` and `LGBMSignedLogPipeline` wrapping the target in a log / signed-log `TransformedTargetRegressor`. `scripts/fit_model.py` dispatches via `cfg.pipeline_cls(cfg, run_config).fit()`, writing `config.json`, `metrics.json`, and CSVs to `output/models/{model_id}/`. Feature-selection diagnostics, SHAP-interaction analysis, and the cluster-first predictor ranking run on every full fit but are skipped in `dev` config. Reports are looked up per model at `docs/models/{model_id}/index.qmd` (variants fall back to their parent's template).

Hyperparameter tuning (`scripts/tune_model.py`) runs an Optuna TPE study under the same `GroupKFold` grouping, writing `best_params.json` to `output/tuning/{model_id}/`. It does **not** mutate the registry — applying tuned params is a manual, reviewable step.

### Statistical models (`statistical_models/`)

Converted families declare immutable typed settings rather than free-form dictionary keys, and resolve them into a validated run plan. Seven families have one so far (#394 pillar 4): `itt`, `gain_factors`, `level_factors`, `did`, `concurrent`, `aligned` and `growth`, each with a `<family>.py` module exporting `<Family>ModelSettings` and `resolve_<family>_run_plan()`. Resolution rejects unknown keys, mixed typed/`extra` declarations and incoherent cross-field combinations **before an output directory is reset or data are loaded** — settings-only constraints belong there rather than in the factory, which only runs after both (#455). The plan then maps to loader arguments, the effective adjustment set, PyMC factory arguments and diagnostic variables; for ITT, metadata and influence refits consume the same plan. Each such fit records it under `resolved_run_plan` in `config.json` and writes `model_recipe.md`, a plain-language account of the question, analysis rows, likelihood, causal treatment term, associational precision terms and required checks. Where the declared settings and the fitted model differ — a `gain_factors` treated-only fit drops its `trt` interactions, since that indicator is constant — the plan records both the declared and the active set, so `config.json` never names a coefficient the posterior lacks.

Step 2 of the methodology: Bayesian models fit with PyMC. One module per model — `lrp_rli_itt_NNN.py` (the DAG-faithful ITT suite + companions), `lrp_rli_did_NNN.py` (the waitlist-crossover / DiD family), `lrp_rli_gf_NNN.py` / `lrp_rli_lf_NNN.py` (the DAG-focused gain- and level-factor families), `lrp_rli_al_NNN.py` (the aligned-40-week per-protocol family) and the bare-family modules `lrp_rli_mech_NNN.py` / `lrp_rli_med_NNN.py` (mechanism/mediation models) — each defining a `SPEC = ModelSpec(...)` and a `fit(config)` that calls the matching pipeline entry point. Families are keyed by `ModelSpec.kind`; `definitions.KINDS` is the authoritative family list. `stages.py` owns the shared attach, sampling, diagnostics, posterior-predictive, metadata and report-finalisation order. ITT is the reference family-owned implementation; the families still awaiting a typed run plan (see above) retain the `factories.py` and `pipeline.py` compatibility boundaries during the incremental Phase 3 migration — `mediation` is the outlier, with three fit functions. The eight detailed below are joined by `adjusted`, `corr_factor`, `dose_response`, `lcsm`, `mediation_multi`, `horseshoe`, `growth`, `historical_growth`, `historical_joint`, `joint_mechanism`, `survival`, `block_exposure`, `concurrent` and `long_corr_factor`. The exact model-id ranges in the per-family bullets below can go stale as models are added — `docs/models/README.md` and `definitions.MODEL_REGISTRY` are the authoritative catalogue:

- **`itt`** — single-outcome intention-to-treat: the uniform DAG-faithful **LRP-RLI-ITT-001–011** suite (own baseline + linear age as _precision_ terms, no cross-baselines — the ITT effect is identified by the empty adjustment set), with **LRP-RLI-ITT-013/113/014/114** adding SES adjustment + matched complete-case comparators, and **LRP-RLI-ITT-017–024** adding a general-ability (block-design) robustness adjustment across the vocabulary family (TR/TE/UR/UE/R/E) and the reading anchors (W, L). Heavily-floored outcomes (P, N) take a post-hoc, arm-blind, data-adaptive **floor rule**: a binary off-floor exploratory headline estimand plus a flagged graded secondary. Phoneme blending (`B`) keeps **LRP-RLI-ITT-008** with the ordinary logit mean as its primary fit and requires the registered **LRP-RLI-ITT-108** one-third guessing-floor companion; neither result may be released without the validated trace-backed paired bundle. → `build_itt_model` / `fit_itt`.
- **`joint`** — the suite outcomes jointly, optional LKJ residual correlation (**LRP-RLI-ITT-012**; the taught-vs-not-taught generalisation contrasts **LRP-RLI-ITT-015/115**) → `build_joint_model` / `fit_joint`.
- **`mechanism`** — adjustment-set dose-response of one measure on another across all phases, with subject random intercepts and optional linear moderation (LRP-RLI-MECH-056–058, 071, 072/172, 073 — its no-interaction companion 173 was retired in #438) → `build_mechanism_model` / `fit_mechanism`.
- **`mediation`** — g-formula NDE/NIE decomposition by counterfactual simulation (LRP-RLI-MED-059 count mediator, LRP-RLI-MED-062 Gaussian reading-route composite) → `build_mediation_model` / `fit_mediation`.
- **`did`** — waitlist-crossover arm-by-wave models (**LRP-RLI-DID-001–013**, plus the companions **LRP-RLI-DID-101** — the independent-prior intercept sensitivity for 001 — **LRP-RLI-DID-102** and **LRP-RLI-DID-107**): binary models jointly fit bounded t1/t2/t3 levels with separate immediate-minus-waitlist gaps. `tau_t2` is the clean randomised t2 contrast; `arm_gap_t1` is a baseline-balance quantity; `arm_gap_t3` and `delta_crossover = tau_t2 - arm_gap_t3` are post-crossover associations. The models do not condition on the treatment-affected t2 period-start score, and the child random intercept partially pools stable heterogeneity rather than making each child their own fixed-effect control. Dose companions retain P1/P2 transition rows, separate current treatment from treated-centred session intensity, adjust for arm, shared pre-randomisation t1 outcome and t1 age, and report dose slopes as observational associations; LRP-RLI-DID-007 has the pooled LOO comparator LRP-RLI-DID-107. → `build_did_model` / `fit_did`.
- **`gain_factors`** — DAG-focused ANCOVA on a period's post-score given its own pre-score (**LRP-RLI-GF-001–013**, one per outcome W/R/E/L/P/B/F/T plus taught-vocabulary TR/TE and nonword N; 001–008 each have a `b` treated-only companion, LRP-RLI-GF-101–108, and each per-outcome primary 001–011 an explicitly associational `m` moderation variant, LRP-RLI-GF-201–211): stacks every on-intervention and untreated period with a child random intercept — a partial, shrunken stand-in for between-child heterogeneity, **not** a control for latent general ability. The randomised on-intervention term is the _only_ causal coefficient, and its probability/items-scale marginal effect is averaged over the **period-1** (randomised, all-untreated-baseline) transition only; every covariate (own baseline, linear age, cognitive ability, upstream DAG skills via `skill_symbols`, the revised-DAG non-measure confounders hearing/speech/phonological memory via `adjust_for`, the age×ability precision interaction) is an explicit _adjusted association_ (adjustment sets re-derived against the revised DAG in #247). SES is excluded (not a DAG node, statistically redundant). The causal headline is interaction-free in trt (#391 finding 3 decision): the pre-specified trt×ability / trt×own moderation questions live only in the `m` variants, whose interaction-aware netted marginal is model-dependent (partly post-crossover-informed) and never released as causal — `release.gate_applies` skips them. Heavily-floored P and N take the suite floor rule (`likelihood="bernoulli_offfloor"`: a Bernoulli on the off-the-floor-at-post indicator, treatment marginal an off-floor risk difference) with the **binary off-floor-at-pre indicator** as the always-on baseline main effect (`gamma_own_offfloor` ~ Normal(0, 1), #391 finding 2 decision — the graded pre logit of a floored measure is a near-degenerate spike). → `build_gain_factors_model` / `fit_gain_factors`.
- **`level_factors`** — the companion _levels_ view (**LRP-RLI-LF-001–011**): the score at each timepoint (no own baseline), with group×time and ability×time as per-timepoint coefficient vectors. Only the t2 group contrast (`b_grp_time[1]`) is a clean randomised effect; later timepoints are post-crossover and flagged as associations. Takes the revised-DAG exogenous confounders (hearing/speech/phonological memory) via `adjust_for` but **no** measure-skill adjusters — a levels model conditioning on another skill's contemporaneous level would condition on a post-treatment mediator of the group×time effect (#247). → `build_level_factors_model` / `fit_level_factors`.
- **`aligned`** — onset-aligned per-protocol single gain (**LRP-RLI-AL-001–008**, plus the cumulative-session dose variant **LRP-RLI-AL-101**): aligns both arms by intervention onset (immediate t1→t3, wait-list t2→t4) into one cross-sectional Beta-Binomial ANCOVA per child (no random intercept). The cohort contrast is **not** randomised — confounded by age-at-onset and cohort/timing — so _no_ term is flagged causal; every coefficient is an association, and dose (a collider) enters only the sensitivity variant. → `build_aligned_model` / `fit_aligned`.

All use a Beta-Binomial likelihood on bounded post-score counts via a logit linear predictor. Shared priors live in `priors.py` (shared constructors so the factories can't drift), HSGP helpers in `hsgp.py`, the g-formula in `mediation.py`. Each pipeline runs prior-predictive → NUTS (`nutpie`) → posterior-predictive, plus PSIS-LOO (ArviZ, pointwise) for every family except the g-formula mediation fits (`mediation`, `mediation_multi`) and the `corr_factor` measurement model — i.e. `itt`/`joint`/`mechanism`/`did`/`dose_response`/`gain_factors`/`level_factors`/`aligned`/`adjusted`/`horseshoe`/`lcsm`/`growth`/`historical_growth` — then writes `trace.nc` (with the `prior`/`prior_predictive`/`log_prior` groups attached), `config.json`, `diagnostics_summary.json` (the pass/fail convergence gate), `key_findings.json` (the plain-language key-findings box, #320 — generated from the fit's own CSVs, gate-interlocked, regenerable without a refit via `scripts/regenerate_key_findings.py`), `ppc_summary.csv` (the posterior-predictive coverage statistic, #318 — the share of observations, or group cells for floor-rule outcomes, inside the model's 50%/90% prediction ranges; rendered as one sentence by `_diagnostics.qmd`), `priors_table.csv` (per-parameter distribution + role), `psense_summary.csv` (power-scaling prior/likelihood sensitivity, #381 — backfillable without a refit via `scripts/regenerate_psense.py`; its `diagnosis` column writes a tick `✓` for an _unflagged_ parameter, so code reading it must treat the tick as clear rather than as a verdict), diagnostic plots (convergence banner data, Pareto-k, rank, ESS-evolution, LOO-PIT, prior-vs-posterior overlay, τ forest), and family-specific CSVs (`tau_summary.csv`, `rope_summary.csv`, `prior_pushforward.csv`, `mechanism_curve.csv`, `mediation_summary.csv`, `factor_summary.csv`, `cohort_marginal.csv`, ...) to `output/statistical_models/models/{model_id}-{config}/`, copying `docs/models/{model_id}/index.qmd` and the shared `docs/models/_partials/` alongside. The shared report setup suppresses scientific result tables and figures after any failed automatic gate while retaining prior, sampler and predictive diagnostic material for repair.

The report templates are **thin** (issue #125): each `docs/models/{model_id}/index.qmd` is a title + model-specific Overview/Model prose + a sequence of `{{< include _partials/… >}}` directives. The shared findings-first order is `_header` → `_setup` → `_gate_badge` (the compact pass/fail verdict) → `_key_findings` (a dumb renderer of fit-time `key_findings.json`, #320) → `_reading_guide` (collapsed) → model prose → the family result partial → `_priors` → `_prior_predictive` → `_technical` (collapsed full `_convergence` + `_diagnostics`) → `_footer`. Per-archetype result partials (`_results_itt`, `_results_floored`, `_results_joint`, `_results_factors`, `_results_mechanism`, `_results_mediation`, `_results_did`, `_results_aligned`, `_results_adjusted`, `_results_block_exposure`, `_results_concurrent`, `_results_dose_response`, `_results_lcsm`, `_results_corr_factor`, `_results_long_corr_factor`, `_results_growth`, `_results_historical_growth`, `_results_historical_joint`, `_results_horseshoe`, `_results_joint_mechanism`, `_results_survival`) live in `docs/models/_partials/` and are driven by `config.json` + `measures` so prose is not hard-coded. They are copied next to each report at fit time so Quarto includes resolve in the output dir.

Fit with `scripts/fit_statistical_model.py {model_id|all} --config dev|test|rep-lite|reporting [--render] [--target-accept X]`. The script's `MODELS` object is a lightweight filename-derived map of `LazyModel` entries, so a model module is imported only when selected. Sampling presets come from `dse_research_utils`; `--target-accept` is an immutable option scoped to one invocation, with precedence command override > model-specific default > preset and no mutation of the shared sampling function. `rep-lite` keeps `reporting`'s `target_accept=0.95` but samples lighter (4 chains × 4000 draws vs 6 × 6000) — ESS, not raw draws, is the binding metric, so it still clears the ESS gate and is portable on ≤5-core machines. `scripts/compare_statistical_models.py` writes cross-model comparisons (ITT-vs-joint τ consistency, τ and mechanism-slope forests, nested PSIS-LOO for interaction models) to `output/statistical_models/comparison/`.

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
- **Bayesian:** check computation before interpretation. A clean pass requires R-hat ≤ 1.01, bulk/tail ESS ≥ 400, BFMI ≥ 0.3 and 0 divergences. A genuinely small divergence-only exploratory fit may be labelled **qualified, not passed** only through the trace- and estimand-bound policy in `METHODS.md`; a low percentage is never enough, and causal/model-of-record, mediation, floor/survival, nonlinear-shape, dose-heterogeneity, horseshoe-ranking, covariance and latent-structure results remain zero-divergence-only. Report the posterior — the **median** with an inner **50 %** and outer **89 %** equal-tailed credible interval, plus the tail probability; no p-values. **89 %** is the house standard, not 95 %: 95 % is an arbitrary NHST convention and its 2.5 / 97.5 % limits are the least MCMC-stable quantiles, so we report a deliberately non-round 89 % (Kruschke 2021 BARG; `notes/202607172359-credible-interval-standard.md`). Positive τ = intervention helps; only τ is causal — observational couplings (`gamma_cross`, `f_mech`, mediator → outcome) are adjusted associations, never "X drives Y".
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

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/): a `<type>(optional scope): <summary>` subject line in the imperative mood, with any detail and rationale in the body. Common types: `feat`, `fix`, `docs`, `refactor`, `test`, `perf`, `build`, `ci`, `chore`. Examples: `feat(itt): add HPDI sensitivity intervals`, `fix(preprocessing): drop empty four-cell rows`, `docs: record the ROPE threshold sign-off`. Reference the issue a commit or PR closes (`Closes #123`) in the body or PR description.

## Writing Markdown

When generating Markdown — `notes/` entries and documents, and especially pull request and issue descriptions and comments — do not insert superfluous line breaks. Write each paragraph as one continuous line and let it reflow; do not hard-wrap prose at a fixed column, and avoid stray blank lines. Prettier is configured with `proseWrap: "preserve"`, so it will **not** rewrap prose for you, and pull-request / issue text is not run through Prettier at all — hard-wrapped paragraphs therefore render as awkward mid-sentence breaks on GitHub and stay that way.

## Pre-commit checks

Before creating a commit or opening a pull request, all of the following must pass:

```bash
ruff check src/         # Python lint
npm run format:check    # Markdown formatting
npm run spellcheck      # Markdown + Quarto spelling (British English, en-GB)
```

If `ruff` reports issues, fix them — do not silence rules or add blanket `noqa` pragmas without justification.

If `cspell` flags a legitimate term (Python identifier, package name, domain term, project acronym, British spelling not in the base dictionary), add it to `config/spellcheck/allow-en.txt` rather than rewording the prose. Only add terms that are genuinely correct — do not use the allow list to paper over actual typos.

Do not bypass these checks with `--no-verify`, skipped CI, or by committing from a different working tree. If either command cannot run (e.g. the conda env is inactive, `npm` is missing), resolve the setup issue rather than proceeding.

## Licensing

- **Code**: AGPL-3.0
- **Documentation and data**: CC BY 4.0
