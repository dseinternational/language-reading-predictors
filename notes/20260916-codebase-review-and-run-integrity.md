> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

# Codebase review and run integrity

## Scope

This review started from `c01584fa`. It followed the main data and execution paths through data loading, predictor construction, grouped cross-validation, tuning, permutation importance, statistical run settings, primary and secondary fits, child-level prediction, posterior summaries, output publication and sweep resumption. It also inspected the test and continuous-integration configuration. The statistical checks included representative ITT construction, mediation calculations, score-link summaries, fit identities, convergence checks and release decisions.

This was a workflow review supported by regression tests, not a proof of every model or a line-by-line review of every registered specification. It did not re-estimate the study's reporting fits or reassess causal identification. The changes below concern execution, provenance and the bootstrap importance diagnostic.

## Findings and fixes

| Priority | Finding                                                                                                                                                                                          | Correction                                                                                                                                                                                                                                                           |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P1       | A sweep could skip a stored fit when the user requested a different `target_accept`. Removing a previous override could also reuse the overridden fit.                                           | The sweep compares the stored numerical sampling settings against the same resolver used by a fresh fit. The resolver applies command override, model default and preset in that order.                                                                              |
| P1       | Matching dirty flags did not establish matching source code. Missing data hashes could pass the resume check, and boosting fits carried no source or data identity.                              | Both layers require a known clean commit, matching environment digest and a verified data digest. Boosting configuration now records those facts through the existing provenance helpers.                                                                            |
| P1       | Boosting wrote `metrics.json`, its resume marker, before permutation importance, SHAP, ranking and report preparation. A failure in one of those stages could be mistaken for completion.        | The pipeline writes metrics after these stages and records `fit_complete=true`. The sweep requires that explicit flag. Standalone metric writes do not certify a completed fit.                                                                                      |
| P2       | The main importance calculation shuffled whole children, while bootstrap stability shuffled individual rows. The two diagnostics therefore applied different changes to longitudinal predictors. | Bootstrap stability now calls the same subject-block permutation evaluator on children omitted from each bootstrap training sample. It skips samples with fewer than two omitted children.                                                                           |
| P2       | Output cleanup suppressed deletion failures and left directory symlinks in place. Old outputs could survive a fresh fit or tuning run.                                                           | Cleanup propagates failures and unlinks directory symlinks without removing their targets. A boosting refit invalidates the previous completion record before removing any other output. If that record is locked, the refit stops and preserves the previous files. |
| P2       | `--stop-on-failure` counted models that had never run as successful.                                                                                                                             | The sweep tracks attempted models and reports successes, failures and models not run separately. Its final companion check receives only attempted models.                                                                                                           |

Resume checks also reject malformed configuration objects and mismatched model identifiers. An explicit randomised-archive request forces a rerun because the sweep does not yet bind that optional audit input's identity. Invalid sampler overrides fail before creating output directories. Statistical-only options are rejected for boosting sweeps.

## Refactoring choices

`resolve_sampling_configuration` separates settings resolution from fit-context construction. The sweep can now compare requested settings without creating staging directories or initialising plots. Existing fits use that same function, so the comparison does not duplicate the precedence rules.

Bootstrap importance reuses `pooled_permutation_deltas`. The bootstrap still samples whole children with replacement, refits on those rows and evaluates only children omitted from training. The shared evaluator supplies the permutation and scoring rules. No second permutation implementation is needed.

The boosting loader hashes the exact bytes it parses and carries that identity through filtering. Final configuration reads that recorded identity rather than hashing a potentially changed source file after fitting. Both model layers use the existing source and environment provenance helpers.

The review retained the existing separation of factories, orchestration, summaries and release checks. Combining scientific calculations across families would require evidence that their adjustment terms, holdout units and reported quantities agree. Similar-looking formulas alone do not establish that.

## Compatibility

Older boosting outputs lack the completion flag and provenance needed for safe resumption. The sweep will refit them. Statistical outputs from dirty or unknown checkouts also require a fresh fit. Commit intended source changes before starting a resumable sweep.

The bootstrap stability values can change because their permutation rule now matches the main importance calculation. Existing output files are not rewritten by this change. The primary prediction fits, model specifications and statistical estimands are unchanged.

## Verification

Regression tests reproduced the sampler-override, missing-identity, dirty-source, premature-completion, row-permutation and early-stop defects before their source changes. Added checks cover matching reuse cases, changed and missing data, malformed metadata, source identity after filtering, locked outputs and directory symlinks. A synthetic repeated-measures example confirms that whole-child permutation preserves an identical wave profile while still detecting a predictor that differs between children.

The sampler resolver was checked for all 276 registered statistical models at all four sampling tiers without creating fit outputs. A development-tier boosting fit completed against the study data in a temporary directory. Its saved completion flag, source-data digest, environment digest and report template were checked. A separate three-bootstrap check on the study data produced finite importance values for all 33 predictors, with every bootstrap completed.

The first full test run exposed a local PyTensor compiler problem. Its installed compiler helper appends `-ld64` on this version of macOS, but the installed Apple compiler rejects that obsolete linker selection. Verification uses a temporary compiler adapter that removes only that argument and then invokes `/usr/bin/clang++`. The real PyMC/nutpie sampler test passes with native compilation retained. No test is disabled and no dependency or global compiler configuration is edited.

The final full suite passed with 3,593 tests passed and one skipped in 386.64 seconds. The Python source and test hashes were unchanged throughout the final verification. Ruff, the configured mypy check for 533 source files, registry documentation checks, Markdown formatting and spelling all passed.
