# Plans — orientation for contributors

> A quick reference to the current methodological shape of the project and
> the modelling work in progress. Intended as a starting point for anyone —
> human or AI assistant — joining the repository.

## The two-step methodology

The project takes a deliberate two-step approach to understanding what
influences progress in language and reading skills for children with Down
syndrome.

### Step 1 — Exploratory analysis with gradient-boosting models

**Purpose**: discover *which predictors matter* for each outcome, across a
space that is too large to specify in advance.

- LightGBM fit with `GroupKFold` grouped by `subject_id`, MAE-tuned via
  Optuna (see `notes/202604121451-lightgbm-model-selection.md` for the
  RF → LGBM switch rationale).
- **Permutation importance** on the fitted model ranks predictors by their
  contribution to generalisation error — more stable than
  split-based importances and resistant to correlated-feature pathologies.
- **SHAP values** surface the *direction* and *consistency* of each
  predictor's contribution; permutation importance alone can hide sign
  reversals and non-monotonic effects (see `CLAUDE.md` §"Interpreting
  model results").
- **Iterative feature selection** prunes from ~32–34 predictors down to
  6–17 per model, removing zero-importance noise while tracking CV MAE,
  R² and MedAE. Each Select step is a named `SelectionStep` on the model
  class plus a dated note in `notes/`.

**Current state**: ten GB models (LRP01–LRP10) — level and gain targets
for word reading, receptive vocabulary, expressive vocabulary, letter-
sound knowledge and CELF basic concepts. See the individual feature-
selection notes (`notes/2026041X-lrpNN-feature-selection.md`) and the
cross-model status summary in `notes/202604181235-status.md`.

### Step 2 — Statistical models for interactions and causal estimation

**Purpose**: take the predictors that Step 1 identifies and estimate
interpretable quantities with quantified uncertainty, answering specific
research questions rather than learning a black-box function of the data.

- Bayesian Beta-Binomial models fit with PyMC via `nutpie`, evaluated
  with PSIS-LOO-CV via ArviZ. Posteriors persisted as NetCDF
  InferenceData under `output/statistical_models/models/`.
- **Intention-to-treat (ITT) models** estimate randomised-phase treatment
  effects on bounded-count outcomes with appropriate priors. Results are
  reported on both the logit scale (the natural parameter) and the
  probability scale at sample-mean baseline.
- **Joint outcome models** give a single posterior over all eight
  outcomes' treatment effects simultaneously, supporting cross-outcome
  contrasts (e.g. "is τ_L more negative than τ_W?").
- **Mechanism models** estimate a dose-response curve `f_mech` of one
  measure on another (e.g. letter-sound knowledge → word reading),
  conditioned on a DAG-derived adjustment set. The adjustment set is the
  single most scrutinised modelling decision in each mechanism model and
  is documented prominently in the corresponding Quarto report.
- All effect estimates come with 95 % credible intervals and posterior
  tail probabilities (e.g. `P(τ > 0)`). No p-values.

**Current state**: seven Bayesian models (LRP52–LRP58) — three ITT, one
joint, three mechanism. All fit in reporting config with 0 divergences
(ITT, joint) or ≤ 1 % (mechanism). See the consolidated findings note
`notes/202604181600-lrp52-58-findings.md`.

### Why both steps?

Neither step alone is sufficient:

- **Step 1 alone** gives rankings and importances but does not quantify
  uncertainty about effect sizes, cannot produce interpretable causal
  estimands, and does not cleanly support confounder adjustment.
- **Step 2 alone** requires a predictor set to enter the linear
  predictor. Picking that set without Step 1 to inform it risks
  specification mining (if the set is tuned iteratively) or arbitrary
  omission (if the set is fixed up front).

The two steps compose cleanly: Step 1 narrows the candidate predictor
space using the full data support; Step 2 takes a stable selection and
turns it into probabilistic statements that a domain reader can act on.

## What is currently in the repository

- **`src/language_reading_predictors/models/`** — LGBM models
  (LRP01–LRP10) plus variants (log/signed-log, quantile,
  construct-driven Select02, RMSE-tuned prediction).
- **`src/language_reading_predictors/statistical_models/`** —
  Bayesian models (LRP52–LRP58): preprocessing, priors, HSGP helpers,
  likelihood, factories (`build_itt_model`, `build_joint_model`,
  `build_mechanism_model`), pipeline orchestration, and per-model
  thin wrappers.
- **`scripts/fit_model.py`** — fits LGBM models.
- **`scripts/tune_model.py`** — Optuna TPE hyperparameter tuning.
- **`scripts/fit_statistical_model.py`** — fits Bayesian models
  (`{model_id|all} --config dev|test|reporting [--render] [--target-accept X]`).
- **`scripts/compare_statistical_models.py`** — cross-model comparison
  (τ forest, mechanism-slope forest, ITT-vs-joint consistency).
- **`docs/models/lrpNN/index.qmd`** — one Quarto report per model.
- **`notes/YYYYMMDDHHMM-*.md`** — decision trail in dated markdown.

See `CLAUDE.md` for full build / test / render commands.

## Next phase (planned, not yet implemented)

Summarised in `notes/202604181453-review-next-steps.md` alongside
analytical sketches and open data-dictionary questions. Priority
ordering:

1. **LRP60** — SES-adjusted ITT (mum / dad education, agebooks) as a
   confounder check on the LRP52–LRP54 treatment effects.
2. **LRP61** — taught-vocabulary ITT (`b1exto` / `b1reto`) as a
   teaching-gradient outcome and methodological positive control.
3. **LRP62** — phonological-memory mechanism (`erbword` / `erbnw` → W).
4. **LRP63** — articulation mechanism (DEAP composites → W, via L).
5. **LRP64** — grammar ITT (`aptgram` / `aptinfo`), contingent on
   confirming item maxima from the data dictionary.
6. **LRP65+** — intervention-dose response with `attend_cumul`,
   explicitly deferred in the original Bayesian brief until a formal
   decision on phase-wise dose semantics is made (see
   `project_rli_phase_structure.md` in the Claude project memory).

A formal mediation decomposition (NDE / NIE) through L is also a
candidate for this phase; it was called out as out-of-scope for
LRP52–LRP58 in the original brief.

## Key methodological conventions

- **No outlier exclusion** unless explicitly justified on the model page.
- **GroupKFold** grouped by `subject_id` is used everywhere
  longitudinal leakage would otherwise inflate performance.
- **Bayesian sign convention**: `G = 1` is the wait-for-intervention
  (control) arm after the `group − 1` recode, so **negative τ means the
  intervention raises the outcome**. See the headline tables in the ITT
  reports.
- **Mechanism-model β_G is not a direct-effect estimate.** Both arms of
  the study are on intervention during phases 1 and 2, so the pooled β_G
  averages over phases in which group is no longer a treatment contrast.
  This is documented inside each mechanism Quarto report.
- **RLI intervention scope**: the intervention targets vocabulary and
  grammar in addition to reading. Null effects on receptive / expressive
  vocabulary (R, E) in the data are substantive findings, not by-design
  predictions.
- **Priors are tightened only when the posterior shows clear evidence
  that the looser prior is prior-dominant**. GP amplitude went from
  `HalfNormal(1.0)` to `HalfNormal(0.3)` after LRP52 sensitivity;
  the age GP was dropped from LRP55/56/57/58 after posteriors showed the
  amplitudes hugging zero.

## Where to start as a contributor

- **New to the project**: read this file, then `CLAUDE.md`, then
  `notes/202604181600-lrp52-58-findings.md` for the Bayesian headline and
  `notes/202604181235-status.md` for the GB headline.
- **Considering a new LGBM model**: follow the Select01 → Select02 →
  tune pattern established for LRP01–LRP10; each step is a note in
  `notes/`.
- **Considering a new Bayesian model**: see the next-phase list above;
  each candidate has an analytical spec sketch in
  `notes/202604181453-review-next-steps.md`.
- **Making a decision that future readers will question**: write a dated
  note in `notes/` before the decision leaves your head. The six LRP52–
  LRP58 decision notes (2026-04-18 1239 through 1800) show the pattern.
