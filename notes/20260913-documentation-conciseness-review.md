> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

# Documentation conciseness review

**Completed review, 13 September 2026.** The two passes below were merged in PR #673 (`95955b92`). This note records that change and its checks; it is not a pending editing plan. The [17 September review](20260917-documentation-review.md) covers the later cleanup.

## Scope and approach

The inventory covered 732 tracked Markdown and Quarto files, including 339 notes and 379 files under `docs/`. A scan identified long passages and exact repetition. Close review covered selected sections of the files below, checked against instructions, code and dated decisions. This was not a sentence-by-sentence review of the archive.

Cuts target repetition, filler, lists maintained in code and completed migration history. Scientific qualifications, teaching examples, dated evidence and pending decisions take priority over word count.

## Proposed changes

Counts are whitespace-delimited words, including tables and commands, compared with the starting commit. Totals include the new shared prose but exclude this review note and the new tests.

| File                                                                               | Before |  After | Main change                                                                                            |
| ---------------------------------------------------------------------------------- | -----: | -----: | ------------------------------------------------------------------------------------------------------ |
| [Shared instructions](../AGENTS.md), each of three identical copies                |  6,027 |  4,086 | Shorter architecture and workflow guidance; remove migration history and repeated registries           |
| [Methods](../METHODS.md)                                                           |  9,844 |  9,107 | Shorter workflow guidance; correct holdout and posterior-summary explanations                          |
| [Project README](../README.md)                                                     |    680 |    470 | Combine introductory material and simplify setup                                                       |
| [Model catalogue](../docs/models/README.md)                                        | 12,505 | 12,299 | Shorter introduction and pointers; retain model tables                                                 |
| [Refit runbook](../docs/runbooks/full-statistical-model-refit.md)                  |  5,956 |  5,760 | Shorter instructions around unchanged command blocks                                                   |
| [Report plan](202609071300-technical-report-plan-v2.md)                            | 12,336 |  9,728 | Shorter chapter briefs; use the claims register and source map instead of repeating planning estimates |
| [Findings by question](202608182200-findings-by-question.md)                       | 13,314 | 12,425 | Shorter summaries; identify the historical evidence snapshot and decision history                      |
| [Shared reading guide](../docs/models/_partials/_reading_guide.qmd)                |    492 |    323 | Shorter definitions with more precise statistical qualifications                                       |
| [Clarity implementation note](20260912-statistical-code-clarity-implementation.md) |    866 |    817 | Shorter descriptions of completed work                                                                 |
| [Statistical review fixes](20260913-statistical-review-fixes.md)                   |    762 |    715 | Shorter change and compatibility descriptions                                                          |

Across both passes, documentation source loses **18,015 words**, including savings from sharing report prose. Counting the instruction files once gives 14,133 words. The second pass contributes 10,816 of the total. The report plan is 21% shorter; the reading guide is 34% shorter. Sharing source text reduces maintenance; readers still receive the included explanation in each report.

### Shared report explanations

Four new files in [docs/models/_partials/model/](../docs/models/_partials/model/) replace repetition in 29 templates:

- Twelve four-wave level reports share the arm-gap explanation. Eleven t1/t2 comparators use a separate version with no post-crossover contrasts. Their introductions now use the comparator IDs and correct time window.
- All 23 level reports share the intercept explanation. It accommodates both ordinary and guessing-floor links.
- Six concurrent reports share the design and missingness explanation.

The existing report-copy function copies these files with the partials directory. The managed results-section order is unchanged. The shared report header records Codex/GPT-6 authorship alongside the existing notices.

### Corrections needing scientific review

- **Gain-factor holdout.** The gain-factor plan declares child-level LOO. The methods guide now distinguishes it from conditional row-level validation and states that supplied baseline predictors remain conditioned on.
- **Posterior summaries.** The methods guide distinguishes monotone transformations of a scalar distribution from a marginal effect combining uncertain coefficients. The latter must be calculated within each posterior draw before summarising.
- **Interval precision.** The methods and reading guide treat 89% as a reporting convention. Tail ESS at 5%/95% is a nearby diagnostic for the 5.5%/94.5% limits, not an exact precision check. The methods link to the official definitions and quantile-specific Monte Carlo error guidance.
- **Blending.** The methods follow the [signed-off amendment to #608](202608252100-blending-pair-binding-608-decision-2.md): the content-addressed archive remains ITT-only, while every pair requires matching, current plans. LF-106 now distinguishes its expected-score floor from individual predicted scores and fixes an escaped equation symbol.
- **Historical findings.** The August note now describes later arm contrasts as randomised treatment-schedule contrasts, following the [later interpretation correction](202608262110-did-lf-estimand-label-sync-631.md). Its numerical tables remain historical evidence.

The first pass also removed an unnecessary sibling-library clone from setup, replaced historical fit examples with a registered model ID, and corrected the catalogue's pointer to the priors-before-results order.

## Checks and limits

The audit confirms identical instruction files and preservation of all eight family-specific instruction bullets. Methods/runbook command blocks, catalogue and findings tables, recorded validation in the two implementation notes, and the findings note's detailed decisions remain unchanged. The report plan retains its claims register, all 13 author decisions, source map, sample lay summary and decision-register appendix. Added local file links resolve.

**Validation passed:**

- All 18 tests in [test_shared_model_prose.py](../tests/statistical_models/test_shared_model_prose.py) and [test_report_restructure.py](../tests/statistical_models/test_report_restructure.py). These check declared wave scopes and copied includes, render four representative prose fixtures in Quarto, and exercise the existing report-order fixtures.
- Repository Markdown formatting, spelling, Python lint and formatting of the new test file.
- Registry documentation checks, all 276 report-template order checks and Git diff whitespace checks.
- A numerical check of the linked median-coefficient counterexample and the 89% interval probabilities.

The installed npm launcher could not find its CLI, so the package's formatting and spelling commands ran directly through the installed tools. Ruff ran without its unwritable cache. Tests passed using the task's writable temporary area after the repository temporary directory rejected creation. An initial render assertion expected wording absent from the unchanged LF-106 source; the assertion now checks its actual guessing-floor label and pairing warning.

No model implementation, data or fitted outputs changed. No models were fitted or historical numerical findings revalidated. The renders check synthetic report fixtures, not complete production reports; the full Python suite was not run.
