# Model code review — bug fixes, tests, and flagged findings (2026-06-17)

A focused review of the model code built so far (LRP01–LRP22 gradient-boosting
models and LRP52–LRP60 Bayesian models) plus the shared pipeline and utility
modules. This note records what was changed and — more importantly — the
findings that need a **modelling decision** before any code change, so they are
not silently "fixed" in a way that alters a scientific estimand.

Environment caveat: this review was done in a workspace without the conda env,
so `pytest` / `lightgbm` / `pymc` could not be run here.
`ruff check src/ tests/` passes, `npm run spellcheck` passes, and the new
pure-Python tests were executed against the real modules
(numpy/pandas/scipy/scikit-learn) outside pytest. The full suite still needs a
run in `dse-language-reading-predictors`.

## 1. Fixes applied in this PR (safe — no model results change)

All of these are either in code paths that the fitted models do not exercise,
or are byte-for-byte behaviour-preserving for the live pipeline.

1. **`stats_utils.spearman_distance_matrix` — input-dependent results under
   missing data.** The DataFrame branch used pandas pairwise-complete Spearman
   (`min_periods=2`); the ndarray branch used
   `scipy.stats.spearmanr(nan_policy="propagate")`, which (a) zeroed out whole
   rows/columns when a single value was NaN and (b) returns a _scalar_ for
   exactly two columns (which would then crash the symmetry/diagonal code).
   Both inputs now route through the same pairwise-complete path. The live
   caller (`feature_selection_diagnostics`) passes an already-NaN-filled
   DataFrame, so the diagnostics output is unchanged; this removes a latent
   correctness trap for any other caller.

2. **`ml_utils.report_cross_validation_scores` — `abs()` hid bad models.**
   `float(np.abs(np.mean(score)))` was applied to _every_ metric, so a negative
   mean R² (a model worse than predicting the mean) was displayed as a positive
   number. Now only `neg_*` error scorers are flipped to positive; `r2` keeps
   its true sign. The sign logic was extracted into a pure
   `cross_validation_score_rows()` so it is unit-testable without the console
   dependency. (This function is not on the fit pipeline's path — the pipeline
   has its own reporting — but it is a real correctness landmine.)

3. **`models/base_pipeline.py` — matplotlib `boxplot(labels=)` deprecation.**
   Every model fit's permutation-importance boxplot used `labels=`, which is
   deprecated in matplotlib ≥3.9 (emits `MatplotlibDeprecationWarning`, removed
   in a future release). Switched to the supported `tick_labels=`.

4. **`plot_utils.plot_gaussian_process` — crash on its own default.** The
   signature defaults `samples=None` but the body iterates `samples`
   unconditionally. Guarded with `samples or []`.

5. **`ml_utils.hyperparam_search_randomized` — fragile best-params selection.**
   Replaced a redundant `sort_values(...).idxmin()` over `cv_results_` with
   sklearn's own `search.best_params_`.

6. **`statistical_models/priors.py` — dead `sigma_outcome_prior`.** Removed; it
   is not in `SHARED_PRIORS` and has no callers (the joint model's residual SDs
   come from the `LKJCholeskyCov` `sd_dist`, as its own comment documents).

7. **LRP09–LRP14 stale "untuned" documentation.** The module/class docstrings,
   `notes` strings, and section-divider comments claimed "no tuning has been
   run" and referenced a non-existent `_LGBM_BASELINE_PARAMS`, while the params
   are demonstrably Optuna-tuned (the param-block comments quote the tuner-inner
   CV MAE). Corrected to match reality. This matters because `notes` is
   persisted to `config.json` and surfaced in the rendered report.

### Tests added

- `tests/test_stats_utils.py` — fills the gap `CLAUDE.md` documents but which
  did not exist; covers `standardize`, `logit`/`invlogit`,
  `differential_entropy_standardized` edge cases, the `spearman_distance_matrix`
  regression above, `mutual_info_dissimilarity`, and (optional, `importorskip`)
  `distance_corr_matrix`. `dcor` is now imported lazily inside
  `distance_corr_matrix` so the module is importable/testable without the
  optional dependency.
- `tests/test_ml_utils.py` — pins the R²-sign fix and the GP kernel helpers.
- `tests/test_data_variables.py` — invariants on the source-of-truth schema
  (group-list uniqueness, NUMERIC/CATEGORICAL disjointness, default-predictor
  exclusions, and the `construct_of` round-trip). A typo in `data_variables`
  propagates into every model, so these are cheap insurance.

## 2. Findings that need a modelling decision (NOT changed here)

These are flagged rather than fixed because changing them alters a scientific
estimand or a reported number, which is the researcher's call.

1. **Mechanism models LRP56/57/58 do not adjust for age, despite their DAG
   adjustment set listing it.** Each spec declares `adjustment=["G", "A",
"W_pre"]` and the report docstring states age blocks a developmental
   backdoor path. But `extra["use_age_gp"]` is `False`, and in
   `factories.build_mechanism_model` the confounder loop does
   `if s in {"G", "A"}: continue` with the comment "A handled via age GP". With
   the GP off and no linear age term, **age never enters `eta`** — the
   `pm.Data("A_std", …)` node is registered but never read. `beta_G` and the
   mechanism curve `f_mech` are therefore not adjusted for the confounder the
   DAG says is required. The age **GP** was deliberately dropped (divergences,
   LRP55 note), but that decision appears to have dropped age entirely rather
   than replacing it with a cheaper **linear** age term. Suggested resolution:
   add a `gamma_A * A_std` linear term when `"A"` is requested and the GP is
   off, and make the factory raise if `"A"` is in the adjustment set but no age
   representation is enabled (so a declared confounder can never be silently
   ignored). This re-fits LRP56/57/58 and changes their headline estimates, so
   it should be a reviewed, noted decision.

2. **Probability-scale marginal effect baseline (`reporting.tau_summary_itt`).**
   `tau_prob_*` is evaluated at `expit(alpha + gamma_own · pre_logit_mean)`,
   omitting the cross-baseline contribution (`gamma_cross · pre_logit`) and any
   SES adjustment terms in the fitted `eta`. Standardised adjusters are ~0 in
   expectation, but cross-baseline pre-logit means are not (e.g. R ≈ −1.5), so
   the non-linear marginal effect is evaluated at a slightly wrong operating
   point. The logit-scale `tau` is unaffected. Either compute the effect by
   perturbing `G` on the full posterior `eta`, or document the reference point
   explicitly.

3. **`pipeline.fit_itt` passes the full-data `pre_logit_mean` to
   `tau_summary_itt`** (`prepared`, not `built.prepared`). Harmless on today's
   data (the factory drops no rows for the current outcomes), but wrong by
   construction if the factory ever subsets — use `built.prepared`.

4. **LRP55 joint model hardcodes its priors inline** (`pm.Normal("alpha", …,
sigma=1.5)`, etc.) instead of calling the `priors.py` constructors that the
   module exists to centralise. The values match today, so the model is
   correct; but a future edit to `priors.py` would silently not reach LRP55 —
   the drift the shared module is meant to prevent. (Not changed here because
   routing through `.to_pymc(dims=…)` needs a run in the real env to confirm an
   identical graph.)

5. **Unconfirmed Beta-Binomial `n_trials` ceilings** (`measures.py`: W=90,
   P=100, both `n_trials_confirmed=False`). `n_trials` is the likelihood
   denominator and scales every probability-scale effect; worth a runtime guard
   asserting `post ≤ n_trials` and confirming the maxima against the data
   dictionary.

## 3. Refactoring / maintainability opportunities (recommended, not done)

- **`.qmd` report templates repeat the stale "`_LGBM_BASELINE_PARAMS` / not
  tuned" passage** for ~14 models (`docs/models/lrp03–16/index.qmd`), now
  inconsistent with the corrected model modules. A templated sweep would fix
  them, but the reports could not be rendered here to verify.
- **Per-model class boilerplate.** Every concrete model repeats
  `cv_splits = 51`, `outlier_threshold = None`, and the single-entry
  `shap_scatter_specs`. The first two equal the `ModelDefinition` defaults and
  could be dropped; the third could become a shared constant. Deferred because
  it touches every model file and could not be regression-tested here.
- **Statistical-model pipeline duplication.** `fit_itt` / `fit_joint` /
  `fit_mechanism` are ~80% identical; a small per-kind config driving one
  templated function would remove the copy-paste (and would have caught
  finding 2.3, where only one path recomputes `pre_logit_mean`).
- **Legacy `DEFAULT_RF_*` names in `ml_utils`** survive the Random-Forest →
  LightGBM switch and now mislabel GP/LGBM helpers.
