> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Response to the review of PR 691

Date 2026-09-23. This note records corrections to the first implementation at commit `6e3978b0`, following the [review comments](https://github.com/dseinternational/language-reading-predictors/pull/691). The reviewers found gaps in the fixes and in their reporting. This follow-up does not complete the broader file-by-file review or validate production results.

## New-child predictive validation

The raw maximum difference between two integration batches is no longer compared with the standard error of expected log predictive density (ELPD). That maximum depends on the number of posterior draws and children examined. Multiplying it by the child count did not provide a calibrated measure of error in the final Pareto-smoothed importance sampling (PSIS) score. It could withhold a stable score because one likelihood cell differed. The maximum remains a finite, non-negative diagnostic in the saved table.

The gate retains the comparison on the score's own scale. It recomputes PSIS for the full integration batch and each half, takes absolute differences for each child, and sums those differences. The largest of the three pairwise sums must not exceed the ELPD standard error. All batches must have acceptable Pareto diagnostics. This is a project stability rule, not an integration-error bound or a rule prescribed by the PSIS paper. The reference for PSIS itself is Vehtari et al. (2024), [Journal of Machine Learning Research](https://www.jmlr.org/papers/v25/19-556.html), preprint DOI [10.48550/arXiv.1507.02646](https://doi.org/10.48550/arXiv.1507.02646).

The review's forecast that almost every latent-variable fit would fail the old rule has not been tested on production fits. The mismatch between the diagnostic and the threshold is enough to warrant the correction. Neither agreement of two batches nor passing Pareto diagnostics proves integration accuracy; both batches may miss the same region.

Validation schema version 3 and `new_child_evidence.new_child_validation_verdict` define one rule for live results and saved tables. The writer records a verdict with reasons. The report rechecks the saved evidence through that same function, including draw counts, finite integration errors, batch reliability and any inconsistent saved flags. It does not trust a stored `reliable=True` by itself. Older tables are withheld. Invalid values yield a withholding reason without numeric formatting that could stop rendering.

## Permutation and rankings

The same-wave, same-schedule donor rule cannot change a predictor determined by assessment wave, including `time`. It also cannot change any predictor whose values are identical among eligible donors at each wave. Such columns now have missing importance values and an explicit "not assessable under this design" status. They receive no individual rank, bootstrap frequency or representative status. Their SHAP values remain separate evidence about the fitted predictions.

For each bootstrap, ranks use only predictors that can change in that sample. The frequency denominator is the number of samples in which that predictor was assessable, recorded as `n_assessable_bootstraps`. An aggregate cluster score uses its assessable members and reports their count; a cluster with none receives no rank. A grouped permutation of a mixed cluster still measures the joint change in the columns that can move. It does not establish the individual importance of its fixed members.

Configurations and standalone ranking metadata now record `subject_blocks_same_wave_schedule_v2`. Both the headline findings and detailed permutation sections check metadata through the same reporting helper. A support CSV alone cannot certify a current ranking. The headline is withheld if the design version is absent, old or inconsistent. All 50 templates use the shared helper. Schedule blocks are built once per scoring call or bootstrap and reused for support, assessability and all repeats.

## Dependence reports and remaining wording

The dependence table now shows `prior_source`, including the closed-form fallback. A missing `verdict` column no longer stops rendering. The variance helper has been renamed to `_joint_variance_channels`; its unused interval-width arguments and the obsolete correlation threshold have been removed.

The three dependence companions and the shared joint report no longer equate similar standard deviations with equal prior and posterior distributions. They no longer infer a covariance decomposition or a cause of interval-width changes from interval widths. The tracked gradient-boosting skill now states the corrected Huber target and the restrictions on permutation donors.

## Verification and scope

Regression checks cover inconsistent saved flags, non-finite integration errors, invalid counts, missing verdict columns, visible prior sources, unassessable predictors, reused schedule blocks, and missing ranks through the saved cluster tables. The headline and detailed support blocks are executed in both old-output and current-output cases across all 50 boosting templates.

The full run reported 3,861 passed, four skipped and one failure. The failed test reads the function's source code. I edited its source file during that run, which moved the lines after Python had loaded the function. It read a table-column line instead of the function body. A subsequent run with unchanged files passed all 412 affected tests, including that check and the final construct-summary and dropout-refit regressions. The full run is not recorded as a clean pass.

Ruff, Mypy across 536 source files, Markdown formatting and spelling passed. All 277 statistical templates passed the include-order check, and all 922 Python blocks in the changed Quarto files parsed. The three shared instruction files match.

Production rankings, predictive validation and posterior summaries still need the regeneration described in the [correction record](20260923-statistical-review-corrections.md). No production refit or revalidation was performed for this response.
