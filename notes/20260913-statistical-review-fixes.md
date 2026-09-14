> [!NOTE]
> Conciseness edits by a LLM-based AI tool (Codex/GPT-6).
>
> Drafted by a LLM-based AI tool (Codex/GPT-6).

# Statistical review fixes

This implements the remaining proposals accepted after the review of `17733d30` (#671): model construction, input validation, child-fold assignment, fitted-model descriptions and removal of repeated gain-factor calculations and outdated comments.

## Correctness changes

The DiD dose factory uses the declared `sigma_child_prior_sigma`, including in the period-specific slope model. The default remains 0.5. Regression checks inspect the built PyMC distributions for default, narrower and wider priors.

Child K-fold validation now recovers each child's group from the fitted row mapping. Repeated rows must agree. Historical panels use group labels from their longitudinal rows, in fitted-child order; their list of group categories is not a child mapping. A stratified split now requires valid labels for every child. A caller that wants an unstratified split must request it explicitly. Tests cover unequal numbers of rows, shuffled rows, conflicting or missing labels, and the case where the number of group categories happens to equal the number of children. The final review also caught and corrected infinite numeric labels stored in an `object` array. Valid numeric and text labels remain supported without coercion.

The main, aligned and later-outcome loaders now use the shared bounded-count validator. It rejects both signs of infinity before logits or integer casts. Missing values represented by `NaN` remain allowed. The existing exception for known finite counts above a measure's ceiling still converts those cells to missing values and records a warning. It does not admit infinity, negative counts or fractions.

The fit context now retains an `effective_plan` separately from `resolved_plan`. Supplying a filtered plan to `write_model_recipe` preserves it for later recipe and metadata writes. `resolved_run_plan` still records the declaration used by the specification checks. `effective_model_settings` records the fitted settings. Regression checks build an adjusted model whose constant missingness indicator is removed, verify that final metadata does not restore its coefficient, and exercise the recipe and trace-compatibility checks. The mediation, joint-mechanism and correlated-factor cases also retain their filtered plans.

## Refactoring and comments

Gain-factor factories now carry the fitted term vectors, main-effect scales, active interaction pairs and binary-baseline flag in `GainFactorsPayload`. Association summaries read these records. They no longer calculate the same standardisation or reconstruct treatment-interaction partners. The raw-logit scales for graded baseline main effects, the shared age scale, and the binary baseline contrast in floor models are preserved. Missingness indicators remain nuisance terms rather than reported associations.

Comments now describe current Boolean validation, prior-evidence requirements, historical-joint prediction and persistence of filtered recipes. They distinguish posterior sampling error from mediator-integration error. Prior-report docstrings, scientific explanations and licence headers remain.

## Compatibility and verification

Corrected recipes and effective settings can make older fits fail trace-reuse checks. Those checks remain intact. This work neither rewrites saved study outputs nor reruns reporting fits.

Before and after the gain-factor refactor, all 33 registered gain models were built from the current study inputs. Their computational-graph and data-design fingerprints, fitted row and child counts, and complete association-summary inputs were identical. This comparison concerns model construction and summary inputs; it does not establish posterior convergence or validate the scientific identification assumptions.

The original study-data cases also pass. `LRP-RLI-JM-002` retains 153 rows from 53 children and now assigns wait-list/immediate fold counts of `(5, 6)`, `(5, 6)`, `(5, 6)`, `(5, 5)` and `(5, 5)`. `LRP-RLI-ADJ-065` retains the effective recipe without `erbto_missing`. Its compatibility check accepts the matching effective description and rejects an incorrect declared-only fitted description. This check used labelled trace bytes for the checksum step; no posterior was sampled or reused, and the five child-fold models were not refitted.

The initial regressions reproduced the defects before the source changes. After the final review, the complete repository suite passed with 3,519 passes and four skips in 756.37 seconds. This includes the module-boundary and typing-coverage checks, the missing compatibility export corrected during the earlier run, and the added correlated-factor and object-array group-label cases. File hashes confirm that the Python source and tests stayed unchanged throughout this run.

Two skipped checks require archived fit files that are absent from this checkout. The other two test Unix permission bits and are skipped on Windows.

Ruff lint, Python formatting, Markdown formatting and spelling checks passed. The configured mypy check passed for 533 source files with the project's existing exemptions.
