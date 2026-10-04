> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

# Corrections to the October statistical review

The user requested all six corrections from the [3 October review](20261003-statistical-methodology-review.md). The changes start from revision `ecbce3d22c5acd70f056e2edffe553dcf2ac62b0`. This note records the scientific choices and how saved fits must be treated. Correcting source code does not make a posterior from a different predictor transform current.

## Mediation sensitivity

The single-mediator pipeline now passes the built outcome's `score_mean_link` to its coefficient-bias sweep. Four pipeline tests cover ordinary and guessing-floor links under natural and interventional decompositions. Each checks that the zero-bias indirect effect equals the primary effect, including its interval endpoints and direction probability. The registered guessing-floor companion currently uses the natural decomposition.

The publication check also compares the stored zero-bias row with the primary fitted-link effect for guessing-floor mediation. This rejects the old table whose effects were 1.5 times too large. It does not change the direction probabilities or the bias values at which an interval crosses zero.

`scripts/regenerate_mediation_sensitivity.py` recomputes the sweep from a saved posterior. Before writing, it verifies the data digest, row and child counts, resolved plan, subject-sequence identity, observed-data identity and model-graph identity against the original reuse contract. It then checks agreement with the saved primary effect. It preserves the posterior and original fit provenance, records the trace and regenerator digests, refreshes the manifest, and re-evaluates publication. Environment-lock equality is required by the general fit-reuse path; this narrower calculation instead verifies the unchanged likelihood and data and records its reporting-only regeneration separately.

## Pooled-level data and units

The factory now restricts `PreparedData` to its requested waves and complete likelihood rows before constructing the model arrays. It returns that same frame. Wave restrictions and missing-score exclusions have separate ledger entries. The pipeline uses the returned frame for adjustment metadata as well as fitted-data identities.

Bounded exposures and same-wave skill adjusters now use `log((score + 0.5) / (maximum - score + 0.5))`. The factory standardises that transform on the final fitted rows and records its mean and standard deviation. It retains the raw-score route for exposures without a documented maximum. Boundary-score tests recover the original counts from the recorded scale; row tests check observed values, child indexing and covariate alignment after exclusions.

The transform name is part of the resolved plan and reuse contract. Stored pooled-level publication checks also require the fitted transform, scale and consistent likelihood-row counts. Old clipped-transform fits therefore remain withheld. The bounded-exposure models `pl-001`, `pl-002`, `pl-003`, `pl-004` and `pl-101` need new posterior fits. The raw-exposure models `pl-005` and `pl-006` have no bounded skill adjusters, so this correction does not change their regressors; they still need a verified metadata refresh or a refit before their records satisfy the new contract. Do not relabel an old clipped fit or replace its fingerprint without a trace-backed check.

The reports no longer claim a common per-standard-deviation unit across families. A common transform does not ensure that fitted standard deviations, row sets or estimands agree. Comparisons need declared score-scale contrasts over shared support.

## Interpretation

The dose-presence coefficient, its prior descriptor, run-plan explanation, findings and current reports now describe a conditional association. It conditions on treatment-induced attendance, including later attendance in each child's mean. It is not the assigned-arm causal effect, even in period 1. The separate available-case modified ITT analysis answers that assignment question under its stated observation assumptions. This follows the treatment-affected-adjustment problem described by Rosenbaum (1984), DOI [10.2307/2981697](https://doi.org/10.2307/2981697). The dose model's likelihood and prior distribution are unchanged.

The pooled-level report now states that a larger between-child association beside a smaller within-child association is compatible with stable shared causes but does not distinguish them from direct influence. Measurement error, little true change, ceiling compression and delayed influence can produce the same pattern. The original review's synthetic counterexample illustrates this logical limit; it does not estimate measurement error in these children. See Curran and Bauer (2011), DOI [10.1146/annurev.psych.093008.100356](https://doi.org/10.1146/annurev.psych.093008.100356).

## K-fold integration

The held-out scorer keeps two independent batches of population latent draws at the same fold posterior. It begins with 64 draws per posterior draw in total, split across the two batches. It doubles that budget up to 512 if the batches disagree. It keeps only one batch of outcome predictions for the calibration plots, as before.

The declared tolerances are an absolute difference of at most 0.1 log-score units for every child and 1.0 for the study total. The fold check receives one equal share of the total tolerance, so accepted fold discrepancies cannot accumulate into a failed study-total check. A difference of 0.1 corresponds to a ratio of about 1.105 between the two estimated predictive densities. These tolerances are operational stability choices, not validated bounds on the true integral or evidence that a close model comparison is resolved. The child-level score standard error still describes variation between children. Both batches can miss the same region, so agreement does not bound integration error. These limits are explicit in the methods and reports. Predictive-score definitions and sampling standard errors are discussed by Vehtari, Gelman and Gabry (2017), DOI [10.1007/s11222-016-9696-4](https://doi.org/10.1007/s11222-016-9696-4).

The scorer records the two pointwise score vectors, budget used, finite-value check and fold batch discrepancies. Completion now requires finite scores, every child, every converged fold and accepted integration checks. Failed scores are withheld. The report rechecks the numerical fields and requires K-fold validation schema version 1. It therefore rejects legacy `complete=True` tables that have no integration evidence.

Tests check agreement with an analytic Gaussian predictive distribution, averaging densities before logarithms, automatic budget increases, persistent disagreement, non-finite densities and rejection of incomplete or stale stored evidence. Integration settings remain outside the saved fold-fit identity because they change scoring rather than fold sampling. Existing fold traces can therefore be re-scored through the normal `--reuse-trace` path when their full reuse contracts match. A refusal by that path needs a refit or a separately justified migration, not removal of the guard.

## Saved-fit recovery

Run the reporting-only mediation correction with the installed environment.

```bash
uv run python scripts/regenerate_mediation_sensitivity.py output/statistical_models/models/lrp-rli-med-387-reporting
```

Re-score compatible K-fold runs through the existing fit command. The primary and fold traces remain subject to their full scientific and environment reuse checks.

```bash
uv run python scripts/fit_statistical_model.py lrp-rlm-jc-001 --config reporting --reuse-trace
```

Refit the bounded-exposure pooled models under the corrected transform. Use the publication runbook for full sampling and transfer of artefacts. Changing their configuration text or reusing the clipped-transform posterior cannot perform this repair.

## Validation and local results

The full suite passed 4,284 tests and skipped one, with two failures in older tests that still expected the former presence interpretation and scorer return shape. After updating those tests to require the corrected contracts, all 217 tests in the four affected modules passed. In total, 4,286 distinct tests passed and one was skipped. The full run included the Quarto report renders, strict type-coverage checks and real PyMC/nutpie sampling smoke test. Strict mypy passed for all 536 source files. Ruff, Markdown formatting and British-English spelling checks passed.

Checks used the current locked environment with `dse-research-utils` 0.17.0. The test commands used temporary writable numerical and plotting caches and disabled PyTensor's unavailable local C-linking path. Quarto tests needed access to the normal log directory and local kernel sockets. These results concern this macOS environment rather than a new Windows production refit.

The saved `lrp-rli-med-387-reporting` sweep was regenerated after every reconstruction check passed. Its zero-bias indirect-effect median is now 0.0264836, with outer 89% limits -0.0107235 and 0.0651323. Those equal the saved primary effect exactly. Its positive-direction probability remains 0.87936. The publication re-evaluation reports `status="ok"`. The original posterior was retained.

The two locally stored pooled fits, `lrp-rli-pl-001-reporting` and `lrp-rli-pl-006-reporting`, now have withheld publication decisions under the corrected contracts. The three local K-fold tables, for `lrp-rli-jm-001-reporting`, `lrp-rlm-jc-001-reporting` and `lrp-rlm-jc-102-reporting`, fail the current numerical-evidence rule. Their copied report partials were updated to withhold the scores. No new posterior was substituted for these fits. The bounded pooled models still need refits, and compatible K-fold posteriors still need re-scoring. Previously rendered static HTML must be rendered again before use.

The [corrected probe results](assets/20261003-statistical-review/corrected_probe_results.json) retain the aggregate evidence and regeneration digests. The original review results remain beside them as a dated snapshot. The corrected pooled construction records 210 likelihood and identity rows and has zero discrepancy from the shared corrected transform on those rows.
