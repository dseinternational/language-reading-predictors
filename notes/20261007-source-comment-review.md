> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

# Source comment review

The review covered comments and documentation strings across the repository's tracked source files and proposes edits to 350 of them. It corrected misleading claims, shortened explanations and removed inactive code or comments that repeated the next statement. The proposed changes preserve the fitted equations, settings, algorithms, test assertions and stored result text.

The review started at `38c4aeec` and moved to current main at `c44bb40f` before final validation. The intervening commit changed the shared-library dependency to the v0.18.0 release tag. It did not change this repository's Python source. Validation uses that commit's locked environment.

## Coverage

[The file-level record](assets/20261007-source-comment-review.csv) lists every source file reviewed and whether it changed.

| Source group                                                  | Files |
| ------------------------------------------------------------- | ----: |
| Python package                                                |   536 |
| Python tests                                                  |   172 |
| Python scripts                                                |    85 |
| Python probes retained with notes                             |    19 |
| Paired Python notebook sources                                |     5 |
| Quarto sources                                                |   380 |
| Graphs, styles, bibliography, build scripts and configuration |    28 |
| Upstream citation-style sources                               |     2 |
| Total                                                         |  1227 |

For Python, the review extracted comments, module and function documentation, class documentation and strings that document attributes. The reviewer read the relevant surrounding code before changing a claim. For Quarto, the review covered comments and documentation in executable chunks and hidden HTML comments. Repeated template contexts were compared together. The upstream citation styles retain their formatting rules and author credits, with one comment typo corrected in each. The review did not re-evaluate every scientific result or prove that every algorithm is correct. Pure prose documents, data files, generated output and dependency locks were outside the source comment review.

## Changes

The review used [METHODS.md](../METHODS.md) and the [model catalogue](../docs/models/README.md) to check scientific wording. A concentration prior can put negligible probability near the Binomial limit without imposing a strict variance floor. A guessing-floor response link constrains the expected score, while an observed count can fall below that expectation. A child random intercept does not establish control for latent ability or identify a causal skill effect. Later assigned-arm gaps can compare randomised treatment schedules, although they no longer compare treated with untreated children.

Comments now distinguish predictive importance from a causal effect, baseline adjustment from complete correction for regression to the mean, and numerical stability from an error bound. Descriptive score bands do not establish prerequisites. A synthetic test fixture does not guarantee the same result in every real or simulated dataset. Several comments also named the wrong units, timing, prior distribution or active model structure.

The edits remove obsolete migration accounts, historical error counts and unused notebook predictor lists. They retain licence notices, notebook cell boundaries, lint and typing directives, report options, and explanations of assumptions or failure checks.

## Refactoring and separate code questions

Most long comments recorded history or scientific assumptions. Removing that history or stating the assumption directly made the source clearer without changing calculations. The following questions need separate code changes and their own tests.

- The settings-class mapping in `family_settings.py` repeats information in `family_registry.py`. Deriving the lookup from the registry could prevent disagreement, subject to import-order checks.
- Regeneration scripts repeat directory filters and model-ID aliases. Their filters differ in how they handle backup directories, so a shared helper needs explicit policies rather than one assumed default. Descriptive plot scripts also repeat a recipe, but their documented standalone use should survive any shared helper.
- Directory publication in `output_transaction.py` relies on one writer and passes a context that acquires no lock. A future change should enforce that assumption or document the external guarantee that prevents concurrent publication to the same directory.
- Dose-response `loo_note` still describes prediction of a new child, while its child-summed score conditions on supplied transition baselines. Joint-mechanism recipe text also needs to distinguish conditional PSIS diagnostics from the separate new-child validation. Changing either prediction target would require scientific review.
- Horseshoe recipe text claims that mean imputation pulls a coefficient towards zero. Mechanism `causal_status` text says that a between-child and within-child decomposition removes stable between-child confounding. Neither claim is guaranteed by the calculation alone. The comment corrections do not change this persisted report text.
- The negative-Binomial historical sensitivity builder has prior rationale strings copied from the bounded-score model and constructs `alpha` without a prior descriptor. Its rationale should describe the actual log-link prior and declared scale. This needs a provenance fix rather than a comment substitution.
- Missingness provenance still states that the source archive is supplied at run time and is not held in the repository. The source resolver now defaults to a committed deposit. The saved provenance should describe the selected source.
- Archived adjustment-set probes use a greedy search. Their calculations cannot prove that a found set is minimal or that no smaller set exists. Their comments now state that limit. An exhaustive or otherwise justified search would be a separate change.

Legacy names such as `rtm_partials` and `readiness_threshold.csv` remain for compatibility. Their documentation now states what they calculate and what they cannot establish.

Scripts that use their module documentation for command-line help show the revised wording. Named prior constructors retain the first documentation line because the code uses it as the recorded rationale.

## Validation

Every tracked Python source compiles. A syntax-tree comparison, with standalone documentation strings removed, verifies that the changed Python files retain their executable statements. The changed Quarto Python chunks retain the same statements and report options. The graph and JavaScript edits change only comments. Parsed `pyproject.toml` settings match the current main branch.

Lint, Markdown formatting, British English spelling, strict type checks and the registry documentation check pass. The full test run began before the final comment edits and reported 4,283 passed, nine skipped and one failed. The sole failure came from source inspection after a comment edit moved a function that Python had already loaded. The old start line, 1118, now held a closing parenthesis. A fresh process found the function at line 1107, and all 24 tests in the affected file passed. Validation did not run a publication fit or regenerate saved research artefacts.
