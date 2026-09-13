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

`tests/statistical_models/test_pipeline_boundaries.py` enforces the import and sampling boundaries. ITT is the reference family implementation. The family summaries below retain their scientific interpretation rules; the catalogue covers additional families, including the associational wave-pooled `pooled_levels` models.

- **`itt`** — single-outcome available-case modified intention-to-treat: the uniform DAG-faithful **LRP-RLI-ITT-001–011** suite (own baseline + linear age as _precision_ terms, no cross-baselines). Randomisation identifies the assigned-arm contrast in the full cohort in principle, but the fitted estimate is an **available-case modified ITT estimate** because the analysis starts from the 54 archived children and applies model-specific observed-data requirements; the empty adjustment set does not repair that selection. **LRP-RLI-ITT-013/113/014/114** add SES adjustment + matched complete-case comparators, and **LRP-RLI-ITT-017–024** add a general-ability (block-design) robustness adjustment across the vocabulary family (TR/TE/UR/UE/R/E) and the reading anchors (W, L). Heavily-floored outcomes (P, N) take a post-hoc, arm-blind, data-adaptive **floor rule**: a binary off-floor exploratory headline estimand plus a flagged graded secondary. Phoneme blending (`B`) keeps **LRP-RLI-ITT-008** with the ordinary logit mean as its primary fit and requires the registered **LRP-RLI-ITT-108** one-third guessing-floor companion; neither result may be released without the validated trace-backed paired bundle. That pairing requirement is **not ITT-specific**: the #608 decision (2026-08-24 — `notes/202608242000-blending-guessing-floor-scope-608.md`) binds every registered model with `outcome_symbol="B"` in every family, whether its published quantity is a randomised contrast or an adjusted association, unless it carries a recorded, dated exemption. **All eight `B` families are now paired** (#619): `itt`, `level_factors`, `did`, `gain_factors`, `aligned`, `concurrent`, `dose_response` and `mediation`. `itt` binds through the trace-backed content-addressed archive; the other seven use the lighter stored-artefact check that reads both fit directories (equally binding, one rung down in evidence strength). Two carry family-shaped cards: `concurrent` compares its marginals table's shape rather than a headline row, because it names no single card, and `mediation` selects the `total` row from its decomposition table. The gate's **default is keyed on the outcome symbol**: a `B` fit whose `kind` has no registered pair gate fails closed rather than slipping through, which is what let four families publish unpaired for months. Every pair also **binds its resolved run plan** — the halves must agree apart from the link, and each stored plan must still match what its module resolves today, so a fit that predates a change to its own specification cannot publish (#608 decision 2 as amended 2026-08-25 — `notes/202608252100-blending-pair-binding-608-decision-2.md`; source commit, dirty flag and environment lock are recorded and surfaced but not required to match, because a companion is registered in a later commit than its primary by construction). The content-addressed archive remains ITT-only. The observational pairings are `LRP-RLI-AL-006` + `LRP-RLI-AL-306`, `LRP-RLI-CA-007` + `LRP-RLI-CA-307`, `LRP-RLI-DOSE-084` + `LRP-RLI-DOSE-384` and `LRP-RLI-MED-087` + `LRP-RLI-MED-387`; all four bind an **association**, not a randomised contrast, because the link determines the natural scale a quantity is published on regardless of what identifies it — `dose_response` is the case #608 used to settle that, since `METHODS.md` defines its focal estimand as the natural-scale treated-row dose marginal. In `mediation` the link goes further than a summary correction: every NDE/NIE/total is a difference of _simulated outcome means_, so `score_mean_link` enters `decompose`'s `outcome_p` and each counterfactual cell is accumulated on the response scale — and it governs the **outcome** only, a mediator being a separate leg with its own measure. The concurrent link governs blending as the _outcome_ only — CA-001–006 carry `B` as a standardised logit predictor and model no `B` score mean. In the gain family the pairing is `LRP-RLI-GF-006` + `LRP-RLI-GF-306` and its scope is the **model of record**: the treated-only `LRP-RLI-GF-106` and moderation `LRP-RLI-GF-206` variants carry a recorded exemption (#596, 2026-08-25 — `notes/202608251100-gain-blending-guessing-floor-596.md`), the same boundary `release.gate_applies` and the level family's window comparator already draw. → `build_itt_model` / `fit_itt`.
- **`joint`** — the suite outcomes jointly, optional LKJ residual correlation (**LRP-RLI-ITT-012**; the taught-vs-not-taught generalisation contrasts **LRP-RLI-ITT-015/115**) → `build_joint_model` / `fit_joint`.
- **`mechanism`** — adjustment-set dose-response of one measure on another across all phases, with subject random intercepts and optional linear moderation (LRP-RLI-MECH-056–058, 071, 072/172, 073 — its no-interaction companion 173 was retired in #438) → `build_mechanism_model` / `fit_mechanism`.
- **`mediation`** — g-formula NDE/NIE decomposition by integrating the mediator distribution within each posterior draw — finite count supports summed exactly, normal mediators by checked Gauss-Hermite quadrature, both recorded under `integration` in `config.json`; the derived-effect MCSE is posterior sampling precision and does not measure integration error (LRP-RLI-MED-059 count mediator, LRP-RLI-MED-062 Gaussian reading-route composite) → `build_mediation_model` / `fit_mediation`. Since #585 every leg conditions on one **common pre-exposure vector** — the outcome baseline, the mediator baseline(s) and every bounded-measure confounder — because the g-formula composes `E[Y | g, m, C]` with `P(m | g', C)` for a single `C`; the terms restoring it are named `a_base_*` / `b_base_*` (prefixed so they cannot collide with the hard-coded outcome own-baseline `b_W`), resolved in the run plan and recorded per leg under `leg_contract` in `config.json`. A floored measure's baseline enters **every** leg as the binary off-floor-at-baseline contrast, never as its degenerate logit, and `pre_required` is derived from those terms so a loaded-but-unmodelled measure cannot exclude a child. The resolver fails before any I/O when a declared bounded-measure confounder is not in the load set. The period-stacked LRP-RLI-MED-092 headlines the **period-1** window (the only one with untreated rows); its all-period average is written separately as an explicit extrapolation alongside `period_treatment_support.csv`. The `interventional` companions (078/186/187) are numerically identical relabellings: intervention dose is in neither leg, so the label changes the target, not the identification.
- **`did`** — waitlist-crossover arm-by-wave models (**LRP-RLI-DID-001–015**, plus the companions **LRP-RLI-DID-101** — the independent-prior intercept sensitivity for 001 — **102**, **103** (the mandatory phoneme-blending guessing-floor twin of 003), **104** (the baseline-allocation prior sensitivity for 001), **105**/**106** (the low-/high-denominator dispersion-prior sensitivities) and **107**): binary models jointly fit bounded t1/t2/t3 levels with separate immediate-minus-waitlist gaps. `arm_gap_t1` is a baseline-balance quantity. `tau_t2` is the randomised contrast of _assignment to immediate treatment versus no treatment yet_ — the covariate-adjusted t2 arm-gap **level**, whose baseline adjustment is soft and prior-weighted (#576 finding 4 sign-off; LRP-RLI-DID-104 is the sensitivity over that allocation). `arm_gap_t3` is **also randomised**, but of a different exposure — early-start versus delayed-start treatment history — so it is not a treated-versus-untreated effect and is _not_ latent-ability-confounded; what it cannot supply is the mechanism, and `delta_crossover = tau_t2 - arm_gap_t3` is the change between two randomised regime contrasts, never an identified catch-up (#576 finding 3). The models do not condition on the treatment-affected t2 period-start score, and the child random intercept partially pools stable heterogeneity rather than making each child their own fixed-effect control — `did_within_child_ppc.csv` tests the covariance that implies. Every fit records one named `focal_estimand` used identically by its posterior headline, prior pushforward, prior-sensitivity sweep and release gate, and the sweep binds to the primary's `run_plan_digest` so a newer plan cannot certify an older fit. Dose companions retain P1/P2 transition rows, carry a saturated arm-by-period cell design (`theta_treated` at the mean treated dose is the crossover _cell_ contrast, not an isolated treatment-presence effect), adjust for arm, shared pre-randomisation t1 outcome and t1 age, and report their treated-row dose marginal and per-period slopes as observational associations; LRP-RLI-DID-007 has the pooled LOO comparator LRP-RLI-DID-107. → `build_did_model` / `fit_did`.
- **`gain_factors`** — DAG-focused ANCOVA on a period's post-score given its own pre-score (**LRP-RLI-GF-001–013**, one per outcome W/R/E/L/P/B/F/T plus taught-vocabulary TR/TE and nonword N; 001–008 each have a `b` treated-only companion, LRP-RLI-GF-101–108, and each per-outcome primary 001–011 an explicitly associational `m` moderation variant, LRP-RLI-GF-201–211): stacks every on-intervention and untreated period with a child random intercept — a partial, shrunken stand-in for between-child heterogeneity, **not** a control for latent general ability. The randomised on-intervention term is the _only_ causal coefficient, and its probability/items-scale marginal effect is averaged over the **period-1** (randomised, all-untreated-baseline) transition only; every covariate (own baseline, linear age, cognitive ability, upstream DAG skills via `skill_symbols`, the revised-DAG non-measure confounders hearing/speech/phonological memory via `adjust_for`, the age×ability precision interaction) is an explicit _adjusted association_ (adjustment sets re-derived against the revised DAG in #247). SES is excluded (not a DAG node, statistically redundant). The causal headline is interaction-free in trt (#391 finding 3 decision): the pre-specified trt×ability / trt×own moderation questions live only in the `m` variants, whose interaction-aware netted marginal is model-dependent (partly post-crossover-informed) and never released as causal — `release.gate_applies` skips them. Heavily-floored P and N take the suite floor rule (`likelihood="bernoulli_offfloor"`: a Bernoulli on the off-the-floor-at-post indicator, treatment marginal an off-floor risk difference) with the **binary off-floor-at-pre indicator** as the always-on baseline main effect (`gamma_own_offfloor` ~ Normal(0, 1), #391 finding 2 decision — the graded pre logit of a floored measure is a near-degenerate spike). Phoneme blending (`B`) carries the family's response-link pair **LRP-RLI-GF-006 + LRP-RLI-GF-306** (#596): the graded primary and the same model under the one-in-three guessing floor `mu = 1/3 + (2/3)·expit(eta)`, released together or not at all. Every natural-scale summary the pipeline derives — treatment marginal, association marginals, ROPE, prior pushforward, predicted scores — maps through the link the factory _built_ (read from `GainFactorsPayload.score_mean_link`), so a floor-link posterior cannot publish ordinary-link items. The family declares `loo_unit="child"` and registers a `loo_child_idx` map, so PSIS-LOO drops every one of a child's transitions together — holding out a single row would leave that post-score in training as the next transition's baseline. → `build_gain_factors_model` / `fit_gain_factors`.
- **`level_factors`** — the companion _levels_ view (**LRP-RLI-LF-001–011**): the score at each timepoint (no own baseline), with group×time and ability×time as per-timepoint coefficient vectors. The arm-by-time vector is centred on the timepoint-1 arm gap (#552): `arm_gap_t1` is the covariate-adjusted pre-randomisation balance quantity (never an effect) and `d_grp_time[t]` the change in the gap at each later wave, with the per-wave levels view `b_grp_time` kept as a Deterministic; only the t2 change (`d_grp_time[t2]`, a difference-in-differences of adjusted levels) is the randomised treated-versus-untreated effect, the t3/t4 changes are randomised early-start-versus-delayed-start schedule contrasts — identified by the original randomisation, not treated-versus-untreated effects and with no mechanistic reading, reported under the `regime` role rather than as adjusted associations (#631) — and `arm_gap_reference="free"` retains the pre-#552 free per-timepoint vector (focal `b_grp_time[1]`) as an explicit comparator. Takes the revised-DAG exogenous confounders (hearing/speech/phonological memory) via `adjust_for` but **no** measure-skill adjusters — a levels model conditioning on another skill's contemporaneous level would condition on a post-treatment mediator of the group×time effect (#247). → `build_level_factors_model` / `fit_level_factors`.
- **`aligned`** — onset-aligned per-protocol single gain (**LRP-RLI-AL-001–008**, plus the cumulative-session dose variant **LRP-RLI-AL-101**): aligns both arms by intervention onset (immediate t1→t3, wait-list t2→t4) into one cross-sectional Beta-Binomial ANCOVA per child (no random intercept). The cohort contrast is **not** randomised — confounded by age-at-onset and cohort/timing — so _no_ term is flagged causal; every coefficient is an association, and dose (a collider) enters only the sensitivity variant. → `build_aligned_model` / `fit_aligned`.

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
