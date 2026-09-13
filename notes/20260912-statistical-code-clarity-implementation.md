> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

# Statistical code clarity implementation

This implements the approved [12 September review](20260912-statistical-code-clarity-review.md). It changes how model code is organised, how priors are described and how incomplete derived draws are diagnosed. It does not rerun the study's reporting fits or replace their saved results.

## Correctness changes

- Prior tables and density panels use the specifications recorded when each random variable is built. Two parameter settings from one constructor receive distinct panel names. Panel titles name the fitted parameters, so a reused treatment-prior constructor does not label an association as causal. Conditional priors and priors with several locations remain in the table and prior predictive checks. The family-specific override table has been removed, and its scientific roles and explanations now live beside construction. Repeated numerical defaults have been removed from call-site rationales.
- Derived effective sample sizes and Monte Carlo errors require the original chain and draw layout. An incomplete or non-finite set produces unavailable diagnostics rather than a fabricated single chain. Complete draws retain the existing calculation.
- The obsolete pooled-moderation command exits with an explanation. Its fitting module has been removed. The [retirement note](20260912-pooled-moderation-retirement.md) explains why selecting newer model numbers would not define a valid replacement.
- The architecture test names modules by their actual relative paths and resolves package imports. Deliberately cyclic and acyclic fixtures test the graph itself.
- Extracting the findings builders also exposed a survival fallback that treated a dictionary as a pandas row. The fallback now reports a covariate association without inventing an assignment contrast. Missing optional subject counts in historical-growth findings retain their previous behaviour.

The prior-table repair command validates the stored run plan, variable names and distributions before changing a fit. It can restore missing panels, removes only unused named-prior density files and updates only the affected manifest entries. It does not refit a trace. No stored study output was repaired during this implementation.

## Code organisation and compatibility

`posteriors.py` holds shared interval and simulation-precision helpers. `summaries/` holds family calculations, and `findings/` holds family prose builders. Active callers import the owning module. `estimands.py`, `reporting.py` and the factory package exports remain compatibility paths for older scripts and notebooks.

Metadata uses the fit's attached validated plan. `reconstruct_run_plan` is the explicit route for a caller with an older declaration. It uses the family registry and rejects incomplete declarations. The shared context now requires the small `ResolvedRunPlan` interface; consumers retain concrete family plans and prepared arrays locally. Concurrent wave fits use a typed record in place of a dictionary. Gain-factor period-1 treatment comparisons and concurrent association comparisons have named functions.

Posterior helpers with a default interval now use the shared 89% reporting constant. Callers can still request another supported coverage explicitly. Prediction-check choices remain separate. `growth_association_summary` no longer accepts `ci_prob`, which it previously ignored; it continues to return fixed 50% and 89% bands. External callers should remove that argument. For model-specific density panels, callers should use `save_model_prior_panels(model, output_dir)`; `save_shared_prior_panel` is only the default constructor catalogue.

Comments now describe the current operations and statistical meaning. The changes remove empty type-checking blocks, replace tuple-producing side-effect lambdas with named hooks, correct the claim about transformation-invariant medians and describe later level-factor arm contrasts as randomised schedule comparisons. A separate formatter commit keeps line wrapping distinct from the functional review.

The [student walkthrough](../docs/learning/itt-model-walkthrough.md) follows the registered ITT-001 vocabulary model using synthetic children. It connects prepared arrays, priors, the likelihood, sampling checks, predictive checks and differences calculated within each draw. It explains the observed-data selection limit and includes a numerical counterexample to substituting median coefficients.

## Verification

The full repository run completed 3,468 tests in 27 minutes 33 seconds. It initially reported 3,437 passes, four skips and 27 failures. After correcting the extracted findings conversion, old test import targets and incomplete metadata fixtures, all 27 failures passed in an exact rerun. Two of those tests needed a writable Numba cache in this Windows sandbox; they needed no source change. Additional checks covered the new prior panels and repair command, derived diagnostics, deliberate import cycles, model construction, summaries and fitting stages. All nine Quarto report-rendering checks passed with Windows cache paths isolated inside their test directories.

Mypy passed across 533 source files. The new summary and findings modules have no type-checking exemptions; the obsolete exemptions for `estimands` and the retired pooling module were removed. Python lint, Markdown formatting and spelling checks passed.

An isolated comparison against the reviewed commit built 39 representative models. All 39 had unchanged design and data identities. Thirty-seven structure hashes were identical. The growth and latent change-score model hashes changed when inline priors moved to named constructors. Their variable names and shapes matched, and their log densities differed by at most `1.87e-7` across four checked parameter settings. This is a numerical consistency check, not proof of equality at every possible parameter value. The changed graph identities can invalidate trace reuse for those builds; this implementation does not weaken the reuse checks.

The synthetic teaching example completed with four chains of 1,000 retained draws, no divergences and bulk and tail effective sample sizes above 1,800 for its five parameters. Its figures were inspected. Those checks concern the example and do not certify the study's production fits.
