> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Statistical and methodological source review

Date 2026-09-23. Reviewed commit `9ecb4a36e3cc20c5c3d8bbeb37ab4c2fd766fa3f`.

This review records ten problems found in the reviewed code or reporting and one problem in an unused moderation path. [Corrections are recorded separately](20260923-statistical-review-corrections.md); the findings below describe the pre-correction snapshot. The findings below have source evidence and reproducible numerical checks. They do not establish how much any stored study estimate would change. No production model was fitted or revalidated during this review. Its original coverage and numerical evidence are retained below.

The requested exhaustive manual review is **not complete**. The [file ledger](assets/20260923-statistical-review-inventory.json) accounts for every tracked file and distinguishes full text reads, selected sections, automated screening and binary inventory. Files marked `automated_screen_only` still need detailed review. Importing a model and resolving its settings does not establish that its equations or interpretation are sound.

Current coverage is 60 full text reads and 39 selected-section reads. Another 1,398 files have only been screened automatically, and 26 binary files have only been inventoried. These categories account for all 1,523 tracked files. A full text read is a record of review coverage, not approval of every claim in the file.

## Reproduction and scope

The [probe script](assets/20260923-statistical-review-probes.py) records its results in [JSON](assets/20260923-statistical-review-probes.json). The scripts and JSON files are historical evidence for the reviewed commit. Their assertions deliberately reproduce the pre-correction defects. Run them only against that source snapshot, using the locked environment; they are not regression tests for the corrected tree. Current regression tests and verification are listed in the correction record. The inventory also describes the original review coverage, not subsequent edits.

```powershell
uv run --no-sync --cache-dir .uv-cache python notes/assets/20260923-statistical-review-probes.py
uv run --no-sync --cache-dir .uv-cache python notes/assets/20260923-statistical-review-continuation-probes.py
uv run --no-sync --cache-dir .uv-cache python notes/assets/20260923-statistical-review-inventory.py
```

The review checked model declarations across all 277 registered statistical models in 23 families. All imported and resolved their run plans. Detailed checks concentrated on data preparation, loss functions, repeated-measures permutation, posterior contrasts, mediation integration, prediction for a new child, convergence and the recent assessment-interval and Huber decisions. The ledger records the limits of those reads by file.

The initial full test suite passed with 3,704 tests passed, four skipped and 497 warnings. The completed run used a temporary directory inside this worktree because the default Windows temporary directory was not writable. An earlier run stopped on those environment errors. Whole-repository Markdown formatting and Markdown/Quarto spelling checks passed in that initial pass. After the continuation, Ruff passed for `src/` and all three review scripts, and this note passed direct formatting and spelling checks. A passing suite does not resolve the counterexamples below.

The continuation read the full missingness implementation, all seven release modules, the artifact writer, the secondary-fit runner, dependence summaries and the joint-results partial. It also checked the related decision notes and missingness tests. The [continuation probes](assets/20260923-statistical-review-continuation-probes.py), with [saved results](assets/20260923-statistical-review-continuation-probes.json), reproduce findings 9 to 11 and independently verify the missingness calculation. All 227 targeted tests passed across `test_itt_missingness.py`, `test_release_decision.py`, `test_joint_run_plan.py` and `test_subfits.py`. Some of those tests assert the reporting rules challenged below; passing them verifies the implementation of the rules, not their statistical validity.

## Findings

### 1. P1. Permutation importance can replace a child's value with a different assessment wave

**Location.** `src/language_reading_predictors/models/permutation.py`, lines 78–82. Callers include the pooled permutation and bootstrap ranking paths in `models/base_pipeline.py`. `tests/test_permutation_importance.py`, lines 90–111, tests the current positional rule rather than assessment-wave alignment.

`subject_block_permutation_indices` selects a donor child and maps the recipient's first row to the donor's first row, and so on. If the donor has fewer rows, it cycles through them. The function receives no wave identifier. Consequently, missing assessments can turn a value at wave 3 into a replacement for wave 2 or wave 1. Reordering rows within a child can also change the result. Choosing one donor child preserves donor identity, but this mapping does not preserve a longitudinal trajectory at its assessment waves.

Two numerical checks isolate the problem. First, two children have exactly the same trajectory, `x(t) = t`, but one lacks wave 2. An exact predictor of `y(t) = t` then acquires a mean permutation increase of 0.537 in root mean squared error across 100 repeats. A replacement aligned by wave would leave the available values unchanged. Second, on complete two-wave data, reversing one child's row order changes the mean squared-error increase from 0 to 0.52 despite leaving the scientific observations unchanged.

A separate check used the repository's filtered study rows. Over 50 permutations, the mean fraction receiving a donor value from a different wave was 0.90% for word-reading levels, 1.20% for word-reading gains, 10.94% for nonword levels and 10.61% for nonword gains. These percentages describe the donor mapping across the eligible rows. They do not quantify the change in any fitted ranking, and they are not a replay of each bootstrap replicate.

**Correction.** Define the intended permutation as a transformation of child trajectories with explicit assessment-wave alignment. Specify how donors with missing waves are handled, such as restricting donors to compatible observed schedules. Test invariance to input row order and the identical-trajectory example. Recalculate permutation importance, bootstrap stability and any downstream ranking labels after that decision. If valid fold estimators are retained, this need not require retuning the prediction models.

### 2. P2. The adopted Huber objective does not generally estimate the conditional mean

**Location.** `METHODS.md`, line 33; `models/objective.py`, lines 8–12; `models/base_pipeline.py`, lines 294–299; all 50 boosting report templates; and the 22 September objective and retune notes. For example, `docs/models/lrp-rli-gbg-012/index.qmd`, line 216, says that the objective and ranking metric both target the conditional mean.

For a fixed Huber threshold, the fitted location solves the equation that the expected clipped residual is zero. The conditional mean instead solves the equation that the expected unmodified residual is zero. These locations can agree, for example under appropriate symmetry, but need not agree for skewed or floored outcomes. Choosing hyperparameters by root mean squared error does not change the functional targeted by the training loss.

The probe uses nine zero values and one value of ten. The arithmetic mean is 1. The repository's threshold rule gives 1.345, and the location that minimises the Huber loss is 0.14944. This is a direct counterexample to the stated equivalence. The [LightGBM 4.7.0 implementation](https://raw.githubusercontent.com/microsoft/LightGBM/v4.7.0/src/objective/regression_objective.hpp) uses the clipped residual gradient in `RegressionHuberLoss`. Feng and Wu discuss the distinction between minimising Huber risk and learning the conditional mean, including the role of an increasing threshold. DOI [10.48550/arXiv.2009.12755](https://doi.org/10.48550/arXiv.2009.12755).

**Correction.** Describe the outputs as predictions trained under the specified Huber loss. State that root mean squared error evaluates those predictions. Do not describe the predictions, partial-dependence curves or SHAP base values as estimates of the conditional mean without further assumptions. Because permutation importance still scores squared residuals through root mean squared error, robust training alone also does not guarantee that extreme observations cannot dominate an importance value. Whether the scientific target requires a different loss is a separate modelling decision. This finding does not invalidate the reported empirical comparison of prediction errors by itself.

### 3. P2. The latent-integration check can cancel errors that matter for predictive scoring

**Location.** `statistical_models/new_child_predictive.py`, lines 414–439, with publication logic at lines 219–247.

`_half_split_error` takes the mean of each half's log likelihood over posterior draws before taking the absolute difference. Positive and negative disagreements across draws can cancel. Predictive density and importance weights depend on nonlinear functions of those log likelihoods, so equality of their arithmetic means does not establish precision of the reported predictive score.

The probe compares halves with log likelihoods `[-1, -9]` and `[-5, -5]`. Both means are −5, and the function returns an integration error of exactly zero. Yet the corresponding raw importance-sampling leave-one-out log scores, `-log(mean(exp(-log_likelihood)))`, are −8.307 and −5. Multiplying the returned error by the number of children, as the current gate does, cannot repair a cancelled zero. This two-draw example isolates the algebra before Pareto smoothing. It is not a production PSIS estimate or evidence that a particular study fit has failed.

**Correction.** Assess precision of the actual pointwise and total predictive scores and their importance weights across independent integration batches and increasing integration effort. Retain checks for disagreement within posterior draws so opposite discrepancies cannot disappear. A change in the mean log likelihood alone should not certify the final nonlinear score. Re-evaluate stored new-child validation before relying on its precision statement.

### 4. P2. Missing Pareto diagnostics can pass the new-child publication check

**Location.** `statistical_models/new_child_predictive.py`, lines 215–247, and `docs/models/_partials/_new_child_validation.qmd`, lines 55–62.

`n_unreliable` counts only `pareto_k > good_k`. A missing numeric value, represented by `NaN`, does not satisfy that comparison. The probe constructs a result with one missing Pareto value and zero integration error; `reliable` returns `True`. The report can then claim that every Pareto value meets the threshold, although one is unavailable. The per-child comparison using `<= good_k` disagrees with the overall result for the same missing value.

**Correction.** Require complete, valid diagnostics for the declared set of children before the result can be reliable. Check dimensions, missing values, thresholds and the predictive summary itself. Treat unavailable diagnostics as unavailable validation. Add a regression test for the missing-value example and for an incomplete diagnostic vector. No stored production fit with this defect was demonstrated in this review.

### 5. P2. The assessment-timing calculations are scenarios rather than bounds

**Location.** `scripts/assessment_interval_check.py`, lines 198–213 and 229–244; `METHODS.md`, line 174; and `notes/202609212100-assessment-interval-lengths.md`, lines 104–122.

The script multiplies the difference in mean assessment intervals by each arm's average gain per month over the whole interval. It sorts these two products and calls them lower and upper bounds. The actual timing contribution depends on growth during the extra interval. Neither whole-period average constrains that local rate without additional assumptions.

The probe gives both arms exactly the same monotone growth trajectory and no treatment effect. Scores remain at zero until month 5 and rise to ten by month 5.44. The earlier arm is assessed at month 5 and the later arm at month 5.44. The script's rule gives limits of 0 and 0.809, but the difference caused entirely by assessment timing is ten. This example is deliberately simple. It proves that the proposed bound does not follow from the available endpoint measurements, even if growth is monotone.

The note also treats an upper product equal to 8–16% of a stored median effect as too small to alter any conclusion. A ratio of point estimates does not establish that a posterior tail probability or practical-benefit conclusion is unchanged. Uncertainty in the timing gap, rates and effect is absent from that calculation. Likewise, a small estimated t1-to-t3 timing difference does not establish exact absence of timing effects at t3.

**Correction.** Label these products as constant-rate sensitivity scenarios and state their assumptions. If the conclusion is that the substantive result survives them, propagate the chosen timing scenarios through the relevant effect draws and report the resulting intervals and tail probabilities. If genuine bounds are required, specify and defend assumptions that bound growth over the extra interval. Preserve the useful descriptive interval measurements.

### 6. P2. Period intercepts do not remove every consequence of unequal observation intervals

**Location.** `METHODS.md`, line 172, and the model-family table and explanation in `notes/202609212100-assessment-interval-lengths.md`, lines 48–63.

A separate intercept for each period can absorb a difference in average change. It cannot, by itself, account for a change in the relationship between starting skill and subsequent change caused by longer observation. Suppose monthly growth is `1 + 0.2 * baseline`. Over five months, the baseline slope for total gain is 1; over eight months, it is 1.6. The probe fits separate period intercepts and one common baseline slope. It obtains 1.3 and leaves systematic residuals of up to 0.3 despite noiseless data.

The same issue applies to self-feedback and cross-skill couplings in discrete-time change models. It does not prove that a pooled association is unusable, or that every family needs a new duration covariate. It shows that the blanket claim that timing is absorbed is too broad. Explicit continuous-time models are one way to make transition coefficients depend on duration; see Voelkle and colleagues, DOI [10.1037/a0027543](https://doi.org/10.1037/a0027543).

**Correction.** State which component the intercept handles and which common-slope assumptions remain. Treat the reported coefficients as associations over the observed mixture of intervals. Before interpreting them as general growth relations or monthly rates, examine period-specific couplings or another model that represents elapsed time. The `lcsm-167` comparison can remain informative while this limitation is stated.

### 7. P2. The PIT reading guide confuses prediction bias with prediction spread

**Location.** `docs/models/_partials/_new_child_validation.qmd`, lines 143–150. The plotted quantity is the empirical cumulative distribution of probability-integral-transform values minus the uniform cumulative distribution.

The guide says that a curve above the band in the middle means predictions are too dispersed and a curve below it means they are too confident. That interpretation is not valid for this cumulative-distribution plot. A shift in the predicted mean can give either sign without any error in spread.

For standard-normal observations and normal predictions with mean 1 and standard deviation 1, the exact cumulative PIT difference at 0.5 is +0.3413. Prediction spread is correct; the location is wrong. Conversely, correctly centred predictions with standard deviation 2 or 0.5 both give zero difference at the middle. Their departures change sign on opposite sides of the middle. The probe records these three distributions at probabilities 0.25, 0.5 and 0.75.

**Correction.** Explain location and dispersion patterns over the full curve, and avoid diagnosing spread from one signed deviation. Describe the Kolmogorov–Smirnov band as an independent-uniform reference unless its calibration has been established for the fitted cross-validation procedure; shared training data can make held-out diagnostic values dependent.

### 8. P2, inactive path. A moderated ITT card can report the opposite benefit probability to its marginal effect

**Location.** `statistical_models/summaries/rope.py`, lines 119–199, and `statistical_models/pipelines/itt.py`, lines 283–292.

The ITT caller supplies moderators to the marginal-effect calculation but leaves `direction_from_ame=False`. The card therefore takes its direction probability from the main treatment coefficient. With moderation and a nonlinear link, that coefficient's sign need not match the average marginal effect across children.

The synthetic trace gives the treatment coefficient a positive value near 0.2, a treatment interaction near 2 and different baseline operating points. The headline's probability of a positive average marginal effect is 0, while the card's benefit probability is 1. Its own median item-scale effect is −1.79. Both statements cannot describe the same benefit claim.

**Scope qualification.** The live registry check found no registered ITT specification with a treatment moderator or varying treatment effect. This is a reproducible latent defect, not evidence that current registered ITT headlines have the wrong sign. The gain-factor path already has an explicit option to derive direction from the average marginal effect.

**Correction.** Derive a benefit claim from the same draw-wise marginal effect used for its item-scale interval. Keep the main coefficient's direction as a separately named quantity. Add the sign-disagreement example before registering an ITT moderation model.

### 9. P2. A prior-data conflict does not make the estimate a lower bound or establish its direction

**Location.** `statistical_models/release/robustness.py`, lines 156–160 and 746–755, with the same note used by the floor and gain paths. The rationale comes from `notes/202608051500-decision-key-findings-robustness-release-gate.md` and is endorsed by `notes/202608201205-itt-code-review-findings.md`.

When both power-scaling scores pass the flag threshold, the graded release path automatically says that the cautious prior attenuates the estimate, that its size is a lower bound and that its direction is more reliable. This branch does not examine the direction of posterior movement or require a treatment-prior sweep. The classification measures sensitivity to changes in prior and likelihood weight. It does not establish any of those three claims. Kallioinen and colleagues describe a diagnostic of potential conflict and inspect posterior changes to understand it. They do not turn that diagnosis into a bound on an effect. DOI [10.1007/s11222-023-10366-5](https://doi.org/10.1007/s11222-023-10366-5).

The continuation probe uses an exactly solvable two-parameter normal model. The likelihood is centred at `(1, 4)` with unit marginal variances and correlation 0.9. Both parameters have independent, zero-centred Normal(0, 1) priors. The posterior mean of the first parameter is −0.50157, opposite to its likelihood centre of +1. Increasing the joint prior's power from 0.99 to 1.01 moves that mean from −0.49712 to −0.50594, further from zero. The installed ArviZ calculation gives prior sensitivity 0.12667 and likelihood sensitivity 0.11631. The repository therefore classifies this example as prior-data conflict and emits the attenuation and lower-bound sentence, although the exact calculation contradicts its explanation. Dependence between parameters lets regularisation of one change the apparent effect of another.

Even actual shrinkage does not create a lower bound on the true effect. In a scalar model with prior Normal(0, 1), unit observation variance and observed value 2, the posterior is Normal(1, 1/2). Its mean is smaller than the observation, but half its own posterior mass lies below that mean. Such an observation can also occur when the true effect is zero. A shrunken point estimate is not a one-sided uncertainty bound.

**Correction.** Describe the result as sensitive to the prior and likelihood until the fitted posterior changes have been examined. Report the named effect's intervals and direction probabilities across justified sensitivity fits. Remove the automatic lower-bound claim and the assertion that conflict proves a real effect. This finding challenges the statistical rationale of the earlier decision, not whether the code faithfully implements it. The synthetic normal model is not a refit or a claim that a registered study effect has reversed.

### 10. P2. Equal-tailed interval widths do not identify the covariance behind a contrast

**Location.** `statistical_models/release/dependence.py`, lines 162–199, and `docs/models/_partials/_results_joint.qmd`, lines 204–223. The method was introduced in `notes/202608241500-joint-588-review.md`.

The code substitutes credible-interval widths for standard deviations in `Var(A - B) = Var(A) + Var(B) - 2 Cov(A, B)`. It then labels the resulting split as measured and assigns the width change to marginal uncertainty or cross-outcome covariance. This substitution requires an appropriate common shape, such as jointly normal draws. An implied correlation inside [−1, 1] does not verify that requirement. A check on the factorised parent's approximation also does not establish that the companion has the same shape error.

The exact counterexample uses independent uniform effects on [0, 0.1]. Each has an 89% interval width of 0.089. Their difference has a triangular distribution with interval width `0.2 * (1 - sqrt(0.11))`, or 0.13367. Although their correlation is exactly zero, the repository infers −0.12782. Against independent normal effects with the same marginal interval widths, it attributes the entire 0.00780 difference in contrast width to covariance. Both constructed pairs have zero covariance; their distribution shapes differ. These are distributional counterexamples, not posterior draws from the study.

The August note separately reports correlations calculated from stored draws. This review has not reproduced or disproved those measurements. The finding concerns the general saved-summary calculation and the explanation it generates for later fits. The partial also retains an unconditional statement at lines 253–254 that positive residual correlation narrows the interval relative to the parent, although the same August note explains why changes to the marginal distributions prevent that inference.

**Correction.** Calculate covariance and variance from paired effect draws within each fit, then use the variance identity on those quantities. Keep the equal-tailed interval comparison as a separate description. If an interval-width approximation is retained, label it as an approximation and verify its error for both fits before attributing a change to covariance. Do not use interval widths alone to declare the cause of a change.

### 11. P2. Equal prior and posterior standard deviations do not show that the data left a parameter unchanged

**Location.** `statistical_models/summaries/dependence.py`, lines 23–30 and 101–118; `statistical_models/release/dependence.py`, lines 59–68; and `docs/models/_partials/_results_joint.qmd`, lines 276–294.

The dependence summary calls a parameter prior-dominated whenever its posterior-to-prior standard-deviation ratio is at least 0.95. The release note then says that the residual correlation did not move off its prior, and the report explains that a ratio near one means the posterior is the prior. A standard deviation measures spread. It cannot establish equality of location or distribution shape. A ratio above one can also reflect a change caused by conflict, rather than absence of learning.

The probe uses bounded correlation distributions. A prior of `2 * Beta(4, 4) - 1`, the two-outcome LKJ(4) marginal, has mean 0 and variance 1/9. A comparison posterior of `2 * Beta(4.592, 1.968) - 1` has mean 0.4 and the same variance. Deterministic quantile samples give a standard-deviation ratio of 0.9999993. The repository labels it prior-dominated and emits the unchanged-prior claim, although the posterior median is 0.44253 and its probability of positive correlation is 0.86710, compared with 0.5 under the prior. This checks the summary rule using valid bounded distributions; it does not claim that a study posterior has this shape.

**Correction.** Label the ratio as relative spread or contraction. Examine location and shape as well before claiming that the posterior is close to the prior, and account for sampling error in that comparison. Keep uncertainty about the correlation separate from evidence about how much the data changed its distribution. Recheck the generated statements on stored companions before retaining their current wording.

## Checks that supported the current implementation

The word-reading missingness loader reconciled the deposited archive with the derived data and retained 53 observed outcomes and 57 prediction profiles. It identified one missing intervention outcome and three missing control outcomes. On constructed interior prediction surfaces, every one of the 25 delta-grid cells matched the independently calculated shift `delta_intervention / 29 - 3 * delta_control / 28`, to a maximum absolute error of 8.9e-15 items. Averaging both treatment predictions over the same profiles remained distinct from comparing predictions over the two actual assigned-arm profile sets. This distinction is intentional and is recorded in the tables.

The review also traced the missingness release check through the saved trace, required prior groups, target and observation dimensions, output hashes and recomputed numerical diagnostics. The secondary-fit reuse path checks model structure and design, fitted rows, trace hashes, sampling settings and the declared diagnostic variables. These are useful safeguards. They do not validate the missingness assumptions, establish representativeness of the observed children or substitute for checking a particular production run.

## What these findings do not establish

The probes establish counterexamples to specific transformations, checks and claims. They do not estimate the bias of any production result. In particular, the donor-wave percentages do not measure the error in the published predictor ordering, the Huber example does not predict the size of a study-model discrepancy, and the new-child examples do not show that every saved predictive score is unreliable.

Several safeguards examined here address real methodological risks. The current code distinguishes adjusted associations from identified effects, preserves the score-mean link in the inspected natural-scale contrasts, integrates finite mediator-count supports explicitly, and records selection and diagnostic restrictions. Those safeguards should be retained. A model can pass its computational checks and still require corrections to its target, numerical validation or interpretation.

## Remaining review work

Use the ledger as the file-by-file continuation record. Detailed review remains for files marked as screened only, and parts of files marked as selected-section reads. Priority areas include the remaining family equations and loaders, missing-baseline floor-eligibility bounds, prior and influence refit producers, response-link pairing validators and other saved-artifact contracts, all remaining report partials, historical notes and notebooks. Historical statements need to be read against their dates and later decisions before being classed as current defects.

A separate review of saved production fits must check their data and source identities, diagnostics, posterior contrasts, sensitivity pairs and rendered findings. That work is needed before asserting that a scientific conclusion is unchanged. This source review is not publication approval.
