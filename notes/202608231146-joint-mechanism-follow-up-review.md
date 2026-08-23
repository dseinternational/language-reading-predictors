> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# Joint-mechanism family follow-up review: residual code and statistical findings, 2026-08-23

## Status and scope

This is a documentation-only follow-up to the joint-mechanism review and fix batch recorded in `notes/202608211130-joint-mechanism-code-review.md` and `notes/202608211200-jm-refit-reporting.md`. It does not change a model or a released result. It records residual findings in the current implementation and proposes a fix sequence with acceptance criteria.

Tracking issue: #591.

The review traced both registered models (`lrp-rli-jm-001` and `lrp-rli-jm-002`) through their typed run plans, data preparation, PyMC factory, pipeline lifecycle, sub-fit provenance, predictive and PSIS diagnostics, release evaluation, key-findings generation, comparison script and Quarto reports. It also reconstructed the fitted row populations and letter-sound exposure scales for the matched comparator models from `data/rli_data_long.csv`.

Within the inspected paths and checks, no sign, tensor-indexing, likelihood-denominator or LKJ covariance error was identified. Outcome-specific missing cells are masked correctly, the transition model's main PSIS-LOO is aggregated by child, and the latent conditional-slope identity is implemented with the correct covariance orientation. The material findings are instead about the publication lifecycle, comparator equivalence, diagnostic semantics and the strength of the scientific interpretation.

## Findings

Priority labels follow the review convention: P1 is material to publication or the principal scientific claim, P2 affects a diagnostic or interpretation but does not invalidate the fitted posterior, and P3 is lower-risk robustness, metadata or documentation work.

### P1: `lrp-rli-jm-001` publishes non-anchor waves without the promised diagnostic lifecycle

`_fit_joint_mechanism_levels` selects one operational anchor by fitted row count, breaking ties in favour of the latest wave. Only this anchor passes through `run_primary_fit`, so only it receives the full prior-predictive, posterior-predictive, power-scaling, extended-diagnostic and persisted-trace lifecycle. Every other wave uses a labelled but unpersisted `run_subfit` with no posterior predictive request or trace filename. Those sub-fits receive a free-random-variable `subfit_convergence` check and a provenance row; they do not receive the complete fit-level gate or an explicit scan of every reported deterministic.

All waves nevertheless enter `joint_mechanism_slopes.csv`, `joint_mechanism_fit_diagnostics.csv` and the per-wave forest plot. The pipeline records `all_published_fits_converged` in `config.json`, but the release evaluator does not consume it. The key-findings builder explicitly notes that the fit-level release gate covers only the anchor, filters failed waves from prose, and then selects the converged wave whose `P(Delta > 0)` is furthest from 0.5. The plot neither removes nor visibly flags a failed wave.

This is not merely hypothetical lifecycle drift. In the checked-in reporting record, timepoint 3 is the diagnostic anchor while timepoint 1 supplies the headline. All four wave fits reportedly passed their sampler checks, so there is no evidence that the current timepoint-1 posterior failed to converge. The problem is that the published headline did not receive the predictive and power-scaling checks the generated model recipe says every published fit must pass.

The exact unpersisted wave posteriors cannot be independently audited, reused or reproduced from persisted artefacts, and they make `--reuse-trace` fail closed: `run_subfit` requires a trace filename for reuse and persistence. A fresh refit from code and data is possible, but it is not the exact published fit.

**Proposed fix.** Make every published wave trace-backed and release-gating. Persist a named trace per wave; run the reported deterministics through the convergence scan; generate the informative new-child posterior predictive and power-scaling diagnostics per wave; and require internally consistent slope, diagnostic, provenance and trace records at release. If this cost is not acceptable, publish one prespecified fully diagnosed wave and label all others as exploratory rather than inferential. Do not use the operational row-count anchor as a scientific primary wave.

Remove the data-selected "clearest wave" headline in the `lrp-rli-jm-001` levels design. Report the four wave estimates without selecting a winner, or fit an explicitly longitudinal/hierarchical time model if a single across-wave conclusion is needed.

### P1: the joint models are not like-for-like replacements for their separate comparators

The `lrp-rli-jm-002` report says it holds the historical mechanism-model parameterisation fixed and changes only joint fitting. The comparison script then describes the gap between the joint contrast and the product-of-marginals sensitivity as evidence of what the working-independence assumption cost. The current fits change more than dependence:

- `lrp-rli-jm-002` requires both outcome baselines for every retained transition, retains the union of rows with either post-outcome, and standardises the letter-sound logit once over that joint union;
- `lrp-rli-mech-096` and `lrp-rli-mech-101` each require their own analysis variables, filter to their own outcome rows, and re-standardise letter sounds on their own fitted samples.

The current data reconstruction is:

| Fit                |                                  Fitted population | SD of letter-sound logit |
| ------------------ | -------------------------------------------------: | -----------------------: |
| `lrp-rli-jm-002`   | 153 union rows; 152 W and 152 N cells; 53 children |                 1.411770 |
| `lrp-rli-mech-096` |                                         152 N rows |                 1.385682 |
| `lrp-rli-mech-101` |                                         156 W rows |                 1.433543 |

Thus one standard deviation denotes a different raw letter-sound increment across the three fits, and the word-reading marginal contains four rows that the joint analysis excludes. The within-`lrp-rli-jm-002` difference remains a valid joint-model contrast for its own common sample and common scale. Its difference from the historical marginal contrast cannot be attributed to cross-outcome covariance alone.

`lrp-rli-jm-001` has an analogous but more explicitly acknowledged mismatch with `lrp-rli-ca-010`/`lrp-rli-ca-011`: it uses a bivariate logistic-normal Binomial model and conditions on latent nonword skill, whereas the concurrent comparators use separate Beta-Binomial models and `lrp-rli-ca-011` conditions on an observed transformed nonword count with missing predictor values mean-imputed. These are related sensitivity estimands, not exact replacements.

**Proposed fix.** Treat the two comparisons differently. For `lrp-rli-jm-002`, define and persist a comparison contract containing the same fitted rows, observed cells, priors, likelihoods and marginal random-effect priors, together with a frozen letter-sound `Standardiser`; comparison-only marginal refits under that contract can isolate the dependence change. Add row-digest and scaler-equality tests so a future data change cannot silently reintroduce comparison drift or overstatement. For `lrp-rli-jm-001`, common rows and scaling are insufficient because latent-outcome conditioning in a logistic-normal Binomial model is intrinsically different from the observed-predictor Beta-Binomial comparators. Keep it explicitly as a distinct sensitivity estimand unless a genuinely nested comparator is designed.

### P1: the decoding-specificity contrast does not identify a decoding-use signature

The code correctly calculates the joint posterior contrast

`Delta = beta(LS -> N) - beta(LS -> W)`.

The stronger report claim is not established: a positive contrast does not by itself distinguish a decoding route from an unobserved common-factor explanation. As a linear latent-scale approximation, let letter sounds and both outcomes depend only on general ability `G`, with mutually independent errors:

`LS = a_L G + error_L`, `N = a_N G + error_N`, `W = a_W G + error_W`.

With no causal letter-sound route, the two latent-scale slopes are still proportional to `a_N` and `a_W`, so `Delta` is proportional to `a_N - a_W`. A positive contrast can therefore arise from different construct loadings, outcome-link discrimination or scaling, non-classical measurement error, floor compression or other nonlinearity. Classical outcome measurement error often changes precision rather than an unstandardised slope, so reliability differences alone are not a sufficient explanation. The joint model imposes no cross-instrument measurement invariance. This matters particularly because N contains six heavily floored items while W contains 79 items.

**Proposed fix.** Describe `Delta` as a measurement-scale-dependent, adjusted association contrast. A positive value is consistent with the proposed decoding account, but does not reject general-ability confounding or identify a mechanism. An item-response or calibrated latent measurement model can address cross-instrument scaling and floor differences, but cannot by itself rule out common-factor confounding; that requires an explicit structural common-factor alternative. Design-based manipulation can add causal leverage, but magnitude comparisons across N and W still require measurement calibration.

### P2: `lrp-rli-jm-002`'s outcome-specific LOO-PIT uses a cell rather than a child as its unit

The factory correctly stores `loo_child_idx`, and the main PSIS-LOO correctly sums all outcomes and transitions for each child. `_joint_outcome_predictive_tree`, used by the two outcome-specific LOO-PIT plots, instead subsets the raw `y_post` likelihood cells and omits the child mapping. Each plot therefore approximates leaving out one outcome at one transition while the same child's other transitions, other outcome and fitted child intercept remain available.

This is a conditional leave-one-cell-out calibration diagnostic, not the leave-one-child-out target declared for the main LOO, and it can look optimistic for prediction to a new child. Existing tests confirm that a plot is written but do not include repeated transitions with a child map.

The current reporting note also records two child-level Pareto-k values above 0.7, with a maximum near 0.94. The main child-level `elpd_loo` is therefore unreliable and must not be used for model comparison until exact re-LOO, moment matching or child-level K-fold validation resolves those children.

**Proposed fix.** Implement a grouped calibration diagnostic whose importance weights use the sum of all likelihood contributions for the omitted child. A genuine new-child check must also integrate or redraw the omitted child's random effect from the population distribution rather than retain its full-data posterior value; exact child re-LOO or child-level K-fold should generate that predictive explicitly. If the existing cell diagnostic is retained, name and label it as conditional leave-one-cell-out and do not present it as the calibration companion to child-level LOO. Add a synthetic repeated-child regression test that asserts the intended number of leave-out units.

### P2: `share_retained` is algebraically correct but is not a bounded pathway share

For the levels model the implemented identity is

`beta(W | latent N) = beta_W - rho * (sigma_W / sigma_N) * beta_N`,

followed by

`share_retained = beta(W | latent N) / beta_W`.

This is a conditional-to-marginal latent-logit slope ratio. It is only stable while the denominator remains away from zero, can be negative under suppression or sign reversal, and can exceed one under amplification. The pipeline has no denominator-stability guard, computes a mean for every term even though the model documentation says never to report the ratio's mean, and renders the generic table containing that mean. The report classifies every median below 0.5 as meaning that most of the association "runs through" decoding, which is wrong for a negative ratio and implies mediation that this observational model does not identify.

The additional statement that latent conditioning will generally retain less than observed-score conditioning, and that the `lrp-rli-jm-001` and `lrp-rli-ca-011` results bracket the answer, is not guaranteed. Classical additive measurement-error intuition does not establish an ordering across these nonlinear models with floor compression, different likelihoods, different missing-data handling and measurement error that may not be classical.

**Proposed fix.** Rename the quantity in scientific prose and define a denominator-stability rule in advance. For example, suppress ratio interpretation when the 89% interval for `beta_W` crosses zero or when excessive posterior mass lies inside a scientifically chosen neighbourhood of zero. Omit the posterior mean and report `P(ratio < 0)`, `P(0 <= ratio <= 1)` and `P(ratio > 1)` rather than classifying the median alone. Remove causal-pathway and bracketing language.

### P2: prior sensitivity is recorded but not carried into the headline

The current reporting note records potential prior-data conflict for `lrp-rli-jm-002`'s nonword slope, `Delta` and `rho_outcome`, and potential strong-prior/weak-likelihood behaviour for `lrp-rli-jm-001`'s slope quantities. The key-findings builder reads the slope table and convergence flag but does not carry this diagnosis into the headline. Power scaling is useful as a local derivative diagnostic, but it is not a substitute for refitting scientifically plausible alternative prior widths.

**Proposed fix.** Surface a material power-scaling diagnosis beside the corresponding result. For `lrp-rli-jm-002`, compare the current mechanism-slope prior standard deviation of 1.0 with 0.5, and the current own-baseline prior standard deviation of 0.25 with 0.5. Add scientifically justified alternatives for the current LKJ concentration of 2 and child/residual standard-deviation priors because varying regression priors alone cannot resolve a `rho_outcome` sensitivity. For `lrp-rli-jm-001`, compare its current slope-prior standard deviation of 0.3 with the documented 0.7 alternative separately at every published wave, together with dependence-block prior sensitivities where `rho_outcome` is interpreted.

### P2: `lrp-rli-jm-002` is an ANCOVA association, not a within-child change effect

The model regresses each post-outcome level on the same-period post-letter-sound level, its own pre-outcome logit, phase, covariates and a child random intercept. It does not model a letter-sound gain, an outcome change score, or a child-centred letter-sound deviation. The common slope therefore combines between-child and within-child information. A random intercept does not by itself remove stable general-ability confounding when that intercept is correlated with letter-sound level.

**Proposed fix.** Call the estimand a post-level association conditional on own baseline, or an ANCOVA-parameterised association. Do not describe it as how much more a child's score moves. If a within-child estimand is required, decompose letter sounds into child mean and within-child deviation, or specify an appropriate longitudinal change model. Such a decomposition clarifies the estimand but does not solve same-wave reverse causation, time-varying confounding or measurement error.

### Other robustness, metadata and documentation gaps

1. **P2:** The `lrp-rli-jm-001` minimum-wave rule requires the exposure and at least one outcome in ten rows. It does not require enough observations for each outcome or enough jointly observed pairs to estimate the residual correlation. Current waves are safely above this edge, but the validation should require per-outcome and overlap minima.
2. **P2:** The only informative `lrp-rli-jm-001` calibration check, the new-child marginal posterior predictive, is optional and can silently disappear. Because the ordinary conditional check is nearly saturated and PSIS-LOO is deliberately disabled, the marginal check should be a required publication artefact and should report coverage per outcome. Poor predictive adequacy should trigger a predeclared qualification or withhold rule rather than an unspecified binary pass/fail criterion.
3. **P3:** Wave subsetting inherits the full-panel `dropped_rows` ledger. The saved anchor metadata can therefore be neither the full-panel drop count nor the true wave-specific count. Record wave-specific eligibility and factory drops separately.
4. **P3:** Source comments incorrectly say only the levels design yields `rho_outcome`; the transition model also reports it. Only the conditional slope and ratio are levels-only.
5. **P3:** `tier1_decoding_specificity` returns early when no separate mechanism rows exist, before attempting to load an otherwise valid `lrp-rli-jm-002`-only result. Load and validate `lrp-rli-jm-002` before this early return so a joint-only contrast can be emitted, or explicitly declare and enforce that the comparison requires marginal rows.
6. **P3:** Direct factory validation accepts a duplicate contrast such as `("N", "N")`; registered typed run plans reject it, so the current models are protected, but the public factory boundary should enforce the same invariant.
7. **P2:** The models make a conditional missing-at-random assumption for missing outcomes and use filled covariates plus missingness indicators. No missing-not-at-random or complete-case joint-model sensitivity is registered.
8. **P2:** `lrp-rli-jm-001` standardises letter sounds independently at every wave. Its fitted logit standard deviations are approximately 1.59, 1.38, 1.39 and 1.44, so cross-wave coefficient ranges do not represent a fixed raw letter-sound increment.

## Proposed implementation sequence

1. **Close the publication lifecycle first.** Persist and fully diagnose every published wave, make the release evaluator validate the complete wave bundle, require the marginal predictive check, and remove data-selected lead-wave reporting.
2. **Make comparison claims auditable.** For `lrp-rli-jm-002`, introduce a fully common comparison contract and refit the marginal comparators. Keep `lrp-rli-jm-001` explicitly distinct unless a genuinely nested comparator is designed.
3. **Correct diagnostic semantics.** Repair or relabel LOO-PIT and resolve high-k child-level LOO with exact or K-fold validation.
4. **Tighten estimand language and ratio governance.** Revise `Delta`, `share_retained`, latent-versus-observed and ANCOVA-change claims in the modules, recipes, key findings, reports and synthesis notes.
5. **Resolve prior dependence.** Run direct prior-width sensitivities and surface material diagnoses beside results.
6. **Harden tests and metadata.** Add grouped-LOO-PIT, asymmetric-missingness, common-row/common-scaler, release-bundle, ratio-edge, overlap-minimum, report-render and simulation-recovery tests.

## Acceptance criteria

- Every scientifically published `lrp-rli-jm-001` wave has a persisted trace, a reproducible convergence verdict covering reported deterministics, an informative posterior predictive check and a recorded power-scaling result.
- `release_decision.json` fails closed when any required wave trace, computational diagnostic or provenance row is absent, inconsistent or failed, and when a required predictive artefact is absent or invalid. A predeclared qualification or withhold rule governs substantive predictive misfit.
- No report or key-findings path selects the most extreme wave after seeing the posterior.
- The `lrp-rli-jm-002` joint-versus-marginal comparison proves common rows, observed cells, scaling, priors and marginal model structure; `lrp-rli-jm-001` remains explicitly a distinct sensitivity estimand unless a nested comparator is introduced.
- Outcome-specific LOO-PIT names and implements the same leave-out target it claims.
- The child-level LOO estimate is not used for model ranking while unresolved Pareto-k values exceed the project threshold; exact, moment-matched or child-level K-fold validation uses the same new-child predictive target.
- `Delta` is reported as an adjusted, measurement-dependent association contrast rather than a formal decoding-mechanism test.
- The conditional/marginal slope ratio is guarded against an unstable denominator, is not described as a mediated share, does not publish a mean, and reports posterior probabilities for negative, zero-to-one and above-one cases.
- Material prior-sensitivity diagnoses appear beside the affected scientific result. Direct fits compare the `lrp-rli-jm-002` reference mechanism-slope and own-baseline widths of 1.0 and 0.25 with alternatives of 0.5 and 0.5; compare the `lrp-rli-jm-001` reference slope width of 0.3 with 0.7 at every published wave; and vary the relevant dependence-block priors when `rho_outcome` is interpreted.
- `lrp-rli-jm-002` is described as post-level ANCOVA conditional on baseline, not as a within-child gain effect.
- The levels-design validator enforces prespecified minima for each outcome and for jointly observed outcome pairs.
- Fit metadata separates full-panel eligibility, wave-specific eligibility and factory-stage drops.
- `tier1_decoding_specificity` can emit a valid joint-only result, and the direct factory rejects duplicate or incomplete contrasts.
- Missing-data sensitivity is implemented or explicitly deferred with a documented reason and consequence; the primary conditional missing-at-random assumption remains visible.
- Cross-wave reporting either uses a frozen letter-sound scale or discloses the wave-specific raw-logit increment represented by one standard deviation.
- New tests cover repeated-child diagnostics, asymmetric outcome missingness, row/scaler identity, release evaluation, ratio edge cases, overlap minima, metadata ledgers, joint-only comparison loading, factory contrast validation and basic simulation-based parameter recovery.

## Verification performed for this review

The PR branch was based on commit `3c572b58ea1955d3673c2002089347a58fff0546`. The reviewed data file had SHA-256 `dc8dda5780b705e902155372c135a993778506c547ef8ebb2b5b03668c11f043`.

The focused current-branch selection was:

```bash
uv run pytest \
  tests/statistical_models/test_joint_mechanism_run_plan.py \
  tests/statistical_models/test_joint_mechanism_pipeline.py \
  tests/statistical_models/test_factories.py \
  tests/statistical_models/test_diagnostics.py \
  tests/statistical_models/test_key_findings.py \
  tests/statistical_models/test_prior_inventory.py \
  tests/statistical_models/test_pipeline_boundaries.py \
  -k 'joint_mechanism or joint_loo or joint_outcome_predictive' -q
```

Result: 49 tests passed, with three existing SHAP deprecation warnings and one PyTensor/Numba Beta-Binomial object-mode warning.

The broader registry, identifier, boundary, prior and report-order selection was:

```bash
uv run --offline pytest -q \
  tests/test_model_definitions.py \
  tests/test_model_ids.py \
  tests/statistical_models/test_pipeline_boundaries.py \
  tests/statistical_models/test_prior_inventory.py::test_every_family_prior_is_documented \
  tests/statistical_models/test_prior_inventory.py::test_used_panels_exist_for_every_family \
  tests/statistical_models/test_prior_inventory.py::test_joint_mechanism_group_terms_and_dependence_blocks \
  tests/statistical_models/test_key_findings.py::test_every_remaining_family_has_bespoke_findings \
  tests/statistical_models/test_key_findings.py::test_builder_registry_covers_every_declared_family \
  tests/statistical_models/test_key_findings.py::test_all_statistical_reports_use_the_findings_first_order
```

Result: 153 tests passed, with three existing SHAP deprecation warnings.

Focused linting passed:

```bash
uv run ruff check \
  src/language_reading_predictors/statistical_models/joint_mechanism.py \
  src/language_reading_predictors/statistical_models/pipelines/joint_mechanism.py \
  src/language_reading_predictors/statistical_models/lrp_rli_jm_001.py \
  src/language_reading_predictors/statistical_models/lrp_rli_jm_002.py \
  tests/statistical_models/test_joint_mechanism_run_plan.py \
  tests/statistical_models/test_joint_mechanism_pipeline.py
```

The note passed Prettier and CSpell checks. Read-only construction probes performed during the original audit built all four `lrp-rli-jm-001` waves and `lrp-rli-jm-002` against the current CSV with finite initial log probabilities. This was a build-time check, not posterior validation. The matched mechanism comparators were prepared far enough to reconcile fitted rows and letter-sound scalers.

No publication-grade NUTS refit was run for this follow-up. Numerical MCMC qualifications above come from the checked-in 2026-08-21 reporting note rather than a live trace in this worktree.
