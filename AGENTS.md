> [!NOTE]
> Conciseness edits by a LLM-based AI tool (Codex/GPT-6).
>
> Substantially edited by a LLM-based AI tool (Codex/GPT-5).
>
> Divergent-transition qualification policy updated by a LLM-based AI tool (Codex/GPT-5).
>
> Phoneme-blending link-sensitivity policy updated by a LLM-based AI tool (Codex/GPT-5).
>
> Available-case modified ITT terminology updated by a LLM-based AI tool (Codex/GPT-5).
>
> conda-to-uv environment migration updated by a LLM-based AI tool (Claude Code/Opus 5).
>
> Mediation integration and gain-factor holdout wording updated by a LLM-based AI tool (Claude Code/Opus 5).
>
> Assessment-interval wording updated by a LLM-based AI tool (Claude Code/Opus 5).

> **Keep in sync:** This file, `CLAUDE.md`, and `.github/copilot-instructions.md` share the same content. When updating one, update all three.

# Agents instructions

## Project overview

Down Syndrome Education International studies predictors of language and reading progress in children with Down syndrome. All data and models are preliminary.

We use two stages:

1. LightGBM, permutation importance and SHAP to identify useful predictors.
2. Bayesian PyMC models to estimate interactions and, where the DAG supports it, causal effects with quantified uncertainty.

See `METHODS.md` for the methods, reporting rules, glossary and references. Paths in these instructions are relative to the repository root.

## Environment setup

Use [uv](https://docs.astral.sh/uv/) to install the locked Python environment:

```bash
uv sync
```

Run commands with `uv run <command>`; activation is optional. PyMC uses the Numba-backed `nutpie` sampler. Supported platforms are declared in `pyproject.toml`; Windows runs natively and Intel macOS is excluded.

`dse-research-utils` supplies the scientific dependencies. This repository declares its required extras and pinned git tag in `pyproject.toml`; do not duplicate those version lists here. For local library development, replace its `[tool.uv.sources]` git entry with a path to `../research/src/python`. The project uses neither the `jax` nor the `storage` extra.

Model graphs also need the system Graphviz `dot` binary (`brew install graphviz`, `apt install graphviz` or `winget install Graphviz.Graphviz`).

## Commands

```bash
# Run all tests
uv run pytest

# Run a single test file or test
uv run pytest tests/test_stats_utils.py
uv run pytest tests/test_stats_utils.py::test_standardize -v

# Lint
uv run ruff check src/

# Spell check (markdown and Quarto files)
npm run spellcheck

# Format Markdown
npm run format
npm run format:check

# Fit a model (artifacts saved to output/models/{model_id}/)
uv run python scripts/fit_model.py lrp-rli-gbg-001                    # dev config (fast, default)
uv run python scripts/fit_model.py lrp-rli-gbg-001 --config test      # test config (moderate)
uv run python scripts/fit_model.py lrp-rli-gbg-001 --config reporting # full config (production)
uv run python scripts/fit_model.py all --config dev --render           # all final models, render reports
uv run python scripts/fit_model.py all --include-variants --config dev # include variants

# Hyperparameter tuning with Optuna (output/tuning/{model_id}/)
uv run python scripts/tune_model.py lrp-rli-gbg-001 # LGBM, 50 trials, GroupKFold
uv run python scripts/tune_model.py lrp-rli-gbg-001 --n-trials 200 --timeout 1800

# Preview research report
quarto preview docs/report/

# Render research report (HTML, PDF, DOCX)
quarto render docs/report/
```

**Output location.** Runs default to `output/`. Set `DSE_LRP_OUTPUT_DIR` or pass `--output-dir` to redirect them; the command option takes precedence. `paths.py` resolves the root, which commands print and record in `config.json`. Copy or upload artefacts from ephemeral scratch disks before teardown. For a full publication run, follow `docs/runbooks/full-statistical-model-refit.md`.

## Architecture

The Python package is in `src/language_reading_predictors/` and is installed in editable mode by `uv sync`.

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

Each outcome has a `lrp_rli_gbg_NNN.py` gain model or `lrp_rli_gbl_NNN.py` level model. Concrete `ModelDefinition` subclasses declare class attributes and register in `MODELS` when they set `model_id`. Import the registry through `models.registry`. Shared dataclasses live in `models/common.py`.

Predictors come from `Predictors.DEFAULT_GAIN` / `DEFAULT_LEVEL` in `data_variables.py`. Models fit the full default set and deliver rankings; hard feature selection is retired. `variant_of` identifies companions, which `fit_model.py all` skips unless passed `--include-variants`. Model `notes` are saved in `config.json`.

`EstimatorPipeline` owns shared steps. Subclasses override `configure_model()` / `_wrap_estimator()`; `LGBMPipeline` is the default, with log and signed-log target variants. The fit script calls `cfg.pipeline_cls(cfg, run_config).fit()` and writes configuration, metrics and tables under `output/models/{model_id}/`. Full fits add selection diagnostics, SHAP interactions and clustered rankings; `dev` skips these. Reports use `docs/models/{model_id}/index.qmd`, with a parent's template as the variant fallback.

`scripts/tune_model.py` uses Optuna TPE with child-grouped cross-validation and writes `output/tuning/{model_id}/best_params.json`. Applying tuned parameters is a manual, reviewable registry edit.

### Statistical models (`statistical_models/`)

Each statistical model declares a `SPEC = ModelSpec(...)` with immutable typed `model_settings`; registered models leave `extra` empty. `from_legacy_extra` adapters support archived configurations. `target_accept` is a separate `ModelSpec` field. `settings_validation.py` rejects non-Boolean values such as `include_group="false"`; `mechanism_design.py` shares cross-field rules between mechanism settings and construction.

Resolve settings before resetting output or loading data. Each family's `resolve_<family>_run_plan()` rejects unknown keys, mixed typed/legacy declarations and inconsistent choices. Settings-only constraints belong here. The plan supplies loader and factory arguments, adjustment terms and diagnostic variables. Primary fits, ITT/joint influence refits and mechanism exact-LOO refits use the same family plan.

Persist the declaration as `resolved_run_plan` in `config.json` and explain the question, rows, likelihood, terms and checks in `model_recipe.md`. Preserve both declared and active settings when data remove terms, such as treatment interactions in a treated-only gain model. Mediation settings live in `mediation_settings.py`; `mediation.py` owns the g-formula algorithms.

`definitions.KINDS` lists the families. `family_registry.py` maps each kind to its settings, resolver, pipeline entry points and findings builder; derive other lookups from it. Each model's `fit(config)` imports its owning `pipelines/` module directly. Mediation shares one pipeline module for its three entry points. Use `docs/models/README.md` and `definitions.MODEL_REGISTRY` for current model IDs.

Keep construction in `factories/`, orchestration in `pipelines/` and shared operations below them:

- Family factories depend on `factories/base.py`, not on other families. Package exports remain compatibility paths.
- `stages.py` owns `PrimaryFitPlan` and `run_primary_fit`: prior prediction, sampling/LOO, summary diagnostics, posterior prediction, convergence and extended diagnostics, then trace persistence. Declare power-scaling timing and family hooks in the plan, including `psense_timing="after_trace"` and `after_trace_audit` where needed. Preserve family-specific ordering and record executed stages in `ctx.lifecycle_stages`.
- `subfits.py` owns secondary and sensitivity sampling through `run_subfit`. `SubfitResult` records the trace, convergence, fitted-data identity, sampling settings, saved trace path and structured failure. Do not call `pm.sample` inside `pipelines/`.
- `runtime.py` supplies `StageHooks` and `require_spec`; `artifacts.py`, `publication.py` and the prior/predictive/figure modules write outputs. `adjustment.py` records fitted adjustment terms; `run_metadata.py` records fit identities and reuse requirements.
- `posteriors.py`, `summaries/` and `findings/` own posterior helpers, family calculations and findings prose. `key_findings.py` assembles and writes the findings. `estimands.py` and `reporting.py` retain compatibility imports. `predictive_checks.py` handles prior pushforwards and predictive coverage; `lcf_inference.py` / `lcf_summaries.py` hold correlated-factor algorithms.
- `convergence.py` owns the sampling gate. `release/` combines input, computation, artefact and robustness checks into the publication decision.

`tests/statistical_models/test_pipeline_boundaries.py` enforces these boundaries. ITT is the reference implementation. Read `METHODS.md` and the model catalogue (`docs/models/README.md`) before changing a family's scientific specification. The main interpretation rules are:

- **`itt`** estimates assigned-arm differences among archived children with the required observations. Use "available-case modified ITT" and state exclusions by arm. Own baseline and linear age are precision terms; they do not repair selection. The P/N floor rule is post-hoc and exploratory. Its headline is an off-floor transition risk difference among children observed at the baseline floor. The immediate arm's t2 assessment came about two weeks later than the wait-list's; state the timing bound from `scripts/assessment_interval_check.py` beside any t2 headline.
- **`joint`** fits several outcomes together. Parent models factorise by outcome; registered dependence companions add within-child residual covariance. Read the corresponding paired comparison before interpreting a difference between outcome effects.
- **`mechanism`** estimates adjusted skill or exposure associations, with child random intercepts and optional moderation or nonlinear curves. Neither the adjustment set nor the random intercept identifies a skill effect under the study's latent-ability confounding.
- **`mediation`** computes model-based direct/indirect decompositions. Every leg must use the same pre-exposure adjustment vector, including outcome and mediator baselines and bounded-measure confounders. Floored baselines use binary off-floor indicators. Integrate finite count supports exactly and normal mediators with checked quadrature; posterior sampling error does not measure integration error. The period-stacked headline uses period 1. Interventional companions change the target's interpretation but remain unidentified under the study's unmeasured confounding.
- **`did`** models arm gaps at t1, t2 and t3. `arm_gap_t1` is baseline balance. `tau_t2` is the adjusted t2 arm-gap level, with soft, prior-weighted baseline adjustment. `arm_gap_t3` is a randomised early-start versus delayed-start schedule contrast. `delta_crossover` is the change between those contrasts, not an identified catch-up mechanism. Dose variants report observational treated-row dose marginals; their saturated arm-by-period design does not isolate a treatment-presence coefficient. Bind every prior sweep to the focal estimand and primary run-plan digest.
- **`gain_factors`** models post-score given pre-score across periods, with child random intercepts. Standardise the treatment headline over period 1 and retain the mandatory period-1-only sensitivity. Primary headlines have no treatment interactions; the moderation variants are associational. P/N use Bernoulli off-floor outcomes and the binary pre-score indicator (`gamma_own_offfloor`). All natural-scale summaries must use `GainFactorsPayload.score_mean_link`. Hold out every transition of a child together for LOO (`loo_unit="child"`).
- **`level_factors`** models per-wave levels without an own-baseline covariate. The default arm-gap reference is t1; `d_grp_time[t2]` is the randomised t2 change in the gap. Later changes compare randomised treatment schedules and have role `regime`. The free-reference specification is a comparator. Do not adjust the group contrast for contemporaneous measured skills, which may be post-treatment mediators.
- **`aligned`** compares one intervention-aligned window per child, immediate t1→t3 and waitlist t2→t4. Age at onset, timing and window length (immediate windows average 1.3 months longer) confound the cohort contrast. Every term is an association; dose enters only the sensitivity variant.

**Phoneme blending.** Every registered `outcome_symbol="B"` fit requires its guessing-floor companion or a dated exemption. An unregistered family pairing fails closed. Both fits must pass their required checks, have matching resolved plans apart from the link and pairing fields, and match their current specifications. ITT uses a trace-backed content-addressed archive; the other seven paired families use stored-artefact checks. Source commit, dirty flag and environment lock are recorded but need not match across the pair. The concurrent comparison uses its marginals table; mediation uses its `total` row and applies the link to the outcome, not the mediator. Treated-only and moderation gain variants have recorded exemptions. See the scope decision (`notes/202608242000-blending-guessing-floor-scope-608.md`), binding amendment (`notes/202608252100-blending-pair-binding-608-decision-2.md`) and gain exemptions (`notes/202608251100-gain-blending-guessing-floor-596.md`).

Bounded-score models commonly use a Beta-Binomial likelihood with a logit predictor; floor models and other families have different likelihoods. Shared priors live in `priors.py`, HSGP helpers in `hsgp.py` and the g-formula in `mediation.py`.

Record a `PriorDescriptor` when each free random variable is built, including its constructor, role, rationale, panel and provenance. Named constructors use `to_pymc(role=..., rationale=...)` for role overrides; inline priors use `priors.declare`. `priors.EXTERNAL_PRIORS` describes the HSGP variables created by the shared library. Missing descriptors stop the fit; never infer a prior's identity or role from its variable name.

Fits write under `output/statistical_models/models/{model_id}-{config}/`:

- `trace.nc` retains the prior, prior-predictive and log-prior groups. Sampling uses NUTS through `nutpie`; PSIS-LOO availability and holdout units depend on the family.
- `config.json` records fit metadata and the versioned `reuse_contract`. Compatibility checks must compare persisted fields. Keep the duplicated top-level fields for existing consumers.
- `diagnostics_summary.json` records convergence. `ppc_summary.csv` gives coverage of 50%/90% prediction ranges for observations or floor-rule group cells. Prior tables/panels, power-scaling summaries, diagnostic plots and family-specific tables supply the supporting evidence. A tick in `psense_summary.csv` means an unflagged parameter.
- Write pipeline tables through `artifacts.save_table`. `artifact_manifest.json` reconciles recorded writes and skips with a directory scan; unrecorded files appear as `untracked`.
- `release_decision.json` records the ordered decision (`inputs` → `computation` → `artifacts` → `robustness`) before `key_findings.json` is generated. `release.evaluate_publication` can re-evaluate stored fits. Failed gates suppress scientific tables and figures while retaining diagnostic material.
- `subfit_provenance.csv` records each secondary fit's convergence, scanned variables, row counts and data digest, sampling settings and saved trace. Standalone influence and prior-sensitivity sweep tools retain their own provenance conventions.

`scripts/regenerate_key_findings.py` and `scripts/regenerate_psense.py` rebuild their respective artefacts from saved fits without resampling.

The `joint`, `joint_mechanism` and `historical_joint` families declare prediction for a new child in a replicate cohort. `new_child_predictive.py` integrates child-level latent variables over their population distribution. A child-indexed free variable omitted from that declaration fails the fit. Withhold the estimate if Pareto-k exceeds `good_k` or half-split integration error rivals the ELPD standard error. The alternative is grouped child-level K-fold refitting through `new_child_kfold.py`, recorded as `cross_validation` sub-fits. Publish the matching `new_child_loo` / `new_child_pareto_k` / `new_child_pit` or `new_child_kfold` tables and figures. Retain and label conditional leave-one-cell-out LOO-PIT plots separately. See `notes/202609011600-joint-new-child-prediction-target-626.md` for the decision.

Each report template contains a title, model-specific prose and shared includes. Keep this order: `_header` → `_setup` → `_gate_badge` → `_key_findings` → collapsed `_reading_guide` → model prose → `_priors` → `_prior_predictive` → family results → collapsed `_technical` (convergence and diagnostics) → `_footer`. `scripts/restructure_statistical_reports.py` validates the order. Shared partials in `docs/models/_partials/` read `config.json` and `measures`; copy them beside the report at fit time so includes resolve.

Fit with `uv run python scripts/fit_statistical_model.py {model_id|all} --config dev|test|rep-lite|reporting [--render] [--target-accept X]`. The filename-derived `LazyModel` map imports only selected models. Sampling presets come from `dse_research_utils`; target-acceptance precedence is command override, model default, then preset, without mutating shared sampling code. `rep-lite` uses 4 chains × 4000 draws and `reporting` uses 6 × 6000, both with preset `target_accept=0.95`. Check attained ESS; a draw count does not guarantee a pass. `scripts/compare_statistical_models.py` writes comparisons under `output/statistical_models/comparison/`.

## Notebooks

Notebooks in `notebooks/` use **Jupytext** (synced `.ipynb` and `.py:percent` formats). Edit either format; Jupytext keeps them in sync. Some legacy notebooks predate the pipeline refactor and still reference Random Forest — they will be updated separately.

Notebooks reference a shared external package (`dse_research_utils`) for environment setup and metadata.

## Conventions

- All source files include SPDX license headers: `# SPDX-License-Identifier: AGPL-3.0-or-later`
- Spell checking uses British English (`en-GB`) configured in `.cspell.config.yaml` with a custom allow list at `config/spellcheck/allow-en.txt`.
- The Quarto report (`docs/report/`) uses `execute: freeze: true` — computational output is cached, not re-run on render.
- Hatch builds the package; `src/language_reading_predictors/__init__.py` supplies the version. `uv run mypy` checks the whole package under strict flags. Keep exemptions in the single `pyproject.toml` list; `tests/test_type_coverage.py` rejects newly failing modules and exemptions that have become unnecessary.

## Interpreting and reporting results

Report direction and uncertainty — never a bare ranking or point estimate.

- **Gradient boosting:** read the SHAP beeswarm (`output/models/{model_id}/shap_summary.png`) with the permutation-importance ranking; the two disagree, so state the direction.
- **Bayesian:** check computation before interpretation. A clean pass requires R-hat ≤ 1.01, bulk/tail ESS ≥ 400, BFMI ≥ 0.3 and zero divergences. Only the trace- and estimand-bound policy in `METHODS.md` can permit a qualified exploratory result; a low divergence percentage is insufficient. Causal/model-of-record, mediation, floor/survival, nonlinear-shape, dose-heterogeneity, horseshoe-ranking, covariance and latent-structure results require zero divergences. Report the median, inner 50% and outer 89% equal-tailed credible intervals, and the named estimand's tail probability; no p-values. `notes/202607172359-credible-interval-standard.md` records the house convention. Positive τ means the intervention arm scores higher; causal interpretation requires the stated assumptions. Skill couplings and mediator-to-outcome terms are adjusted associations.
- **Notes, issues, PRs:** write for a frequentist-leaning science reader; expand shorthand and read credible intervals in plain words; record decisions a future reader might question as a dated `notes/` note; verify citations and always include DOIs.

Full rationale, workflow, conventions, glossary, and references: **`METHODS.md`**.

## AI-authored content labelling

Label AI-drafted or substantially edited prose at the top, before the substantive text. Name the actual tool and model. This covers documents, notes, PRs, issues, review comments, release notes and discussion posts. Keep labels added by other tools.

Use a GitHub alert for Markdown, PRs, issues and comments:

```markdown
> [!NOTE]
> Drafted by a LLM-based AI tool (<tool>/<model>).
```

Use a Quarto callout in `.qmd` files; GitHub alerts render there as plain blockquotes:

```markdown
::: {.callout-note}
Drafted by a LLM-based AI tool (<tool>/<model>).
:::
```

For plain text, use `Note: Drafted by a LLM-based AI tool (<tool>/<model>).`

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/): `<type>(optional scope): <imperative summary>`. Put detail, rationale and closing issue references (`Closes #123`) in the body or PR description. Common types are `feat`, `fix`, `docs`, `refactor`, `test`, `perf`, `build`, `ci` and `chore`.

## Writing Markdown

Write each prose paragraph on one continuous line and let the renderer wrap it. Avoid fixed-column wrapping and stray blank lines, including in PR and issue text. Prettier uses `proseWrap: "preserve"` and will not repair hard-wrapped paragraphs.

## Pre-commit checks

Before creating a commit or opening a pull request, all of the following must pass:

```bash
uv run ruff check src/  # Python lint
npm run format:check    # Markdown formatting
npm run spellcheck      # Markdown + Quarto spelling (British English, en-GB)
```

Fix Ruff findings; do not silence rules or add blanket `noqa` pragmas without justification.

Add valid terms missing from CSpell's dictionary to `config/spellcheck/allow-en.txt`. Correct actual typos; do not hide them in the allow list.

Do not bypass checks with `--no-verify`, skipped CI or another working tree. Resolve missing tools or environment problems before proceeding.

## Licensing

- **Code**: AGPL-3.0
- **Documentation and data**: CC BY 4.0
