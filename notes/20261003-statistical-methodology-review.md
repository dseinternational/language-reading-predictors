> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

# Statistical methodology and implementation review

Review date 3 October 2026. Source revision `2203024a800c4ecbccd06c66a79ec915885c5772`.

Follow-up on 4 October 2026. The [correction note](20261004-statistical-review-corrections.md) records the implementation changes and the remaining work on saved fits. The results below describe the reviewed revision. `probe_results.json` retains that snapshot; the probe script now checks the corrected pooled-level construction and labels the omitted-link calculation as a legacy reconstruction.

The review found three implementation defects, two unsupported interpretation claims and one numerical-validation gap. The most immediate defect affects a saved phoneme-blending mediation sensitivity table. Its effect sizes and interval endpoints are 50% too large because the calculation uses the wrong outcome link. The pooled-level models also record the wrong fitted rows and claim common predictor units despite using a different transform. These findings do not establish that the primary word-reading intervention estimate is wrong. They do show that a publication verdict and a passing test suite cannot by themselves establish that every reported quantity matches its model.

This note records findings and proposed repairs. It does not change scientific specifications, fitted models or existing reports. Priority P1 means that the affected numerical result should be corrected before use. Priority P2 means that an implementation, interpretation or validation issue needs a planned correction before the affected claim is treated as reviewed evidence.

## Scope and evidence

The review covered `METHODS.md`, the model catalogue, the main report's methods chapter and data helpers, data validation and preparation, model construction across the 23 statistical families, selected family plans and pipelines, posterior summaries, prediction, priors, convergence, sensitivity and publication code. It also covered the current gradient-boosting pipelines, tuning, subject-level permutation, bootstrap rankings, target transformations and selected legacy notebooks. The earlier September review and correction notes supplied questions to recheck against the current code.

The current catalogue contains 277 statistical models. I imported each registered specification and resolved its family plan. All 277 resolved without error. The gradient-boosting registry contains 50 models, all using `LGBMPipeline`. This verifies declaration consistency; it does not fit every model or establish its scientific validity.

The local output tree contains 131 saved statistical configurations. Their saved release decisions record 130 publishable runs and one withheld run. I inventoried those files and checked selected artefacts relevant to the findings. I did not independently re-evaluate every release decision, load every full trace, rerun production fits or verify all 277 reporting results. In particular, a saved verdict is reported here as a stored fact rather than a fresh endorsement.

The [review probes](assets/20261003-statistical-review/review_probes.py) reproduce the transform, fitted-row and mediation-link findings without posterior sampling. They also inspect the current dose design and provide a synthetic counterexample to the pooled-level interpretation. The [probe results](assets/20261003-statistical-review/probe_results.json) retain aggregate results. The counterexample is a logical test of a general claim, not an estimate of measurement error in these children.

Run the probes from the repository root with the installed environment. The PyTensor setting avoids the local C-linking failure described under verification.

```bash
PYTENSOR_FLAGS='cxx=' uv run --no-sync python notes/assets/20261003-statistical-review/review_probes.py
```

## Findings

### 1. P1 The blending mediation sensitivity uses the wrong outcome link

**Implementation defect.** The main decomposition passes the fitted `score_mean_link` to `decompose`, but the sensitivity call at [`pipelines/mediation.py` line 333](../src/language_reading_predictors/statistical_models/pipelines/mediation.py#L333) omits it. `sensitivity_sweep` forwards its keyword arguments to `decompose`, whose default is the ordinary inverse-logit link. This affects the registered `lrp-rli-med-387` guessing-floor companion.

The fitted mean is `1/3 + 2/3 * expit(eta)`. A contrast between two such means is two-thirds of the contrast computed with `expit(eta)`. The additive one-third cancels. The omitted link therefore multiplies indirect-effect differences by 1.5 for every bias value, with the same factor applied to all interval endpoints. This is an exact consequence of the two links, rather than a Monte Carlo discrepancy.

The saved reporting artefacts confirm the error. The main table's indirect-effect median is 0.0264836 on the score-proportion scale, with an outer 89% interval from -0.0107235 to 0.0651323. At zero assumed bias, the sensitivity table instead gives 0.0397254, with limits -0.0160853 and 0.0976985. Each is exactly 1.5 times its main-table counterpart. On the ten-item scale, the main median is about 0.265 items and the sweep implies about 0.397 items. Both tables report the same direction probability, 0.87936.

**Repair.** Pass the built payload's outcome link into every downstream decomposition, including the sensitivity sweep. Regenerate this table and any derived text or plot from the saved posterior. A new fit is not needed for this calculation. Add a pipeline-level check that the zero-bias sweep agrees with the primary indirect effect on the fitted outcome scale. Testing the two links only inside `decompose` does not catch an omitted argument at its caller. Audit the other decomposition entry points when extending the guessing-floor policy; the current period-stacked and two-mediator registrations use ordinary outcome links.

### 2. P2 Pooled-level fitted-row identities describe rows outside the likelihood

**Implementation defect.** [`build_pooled_levels_model`](../src/language_reading_predictors/statistical_models/pooled_levels.py#L585) forms a mask for requested waves and complete outcome, exposure and skill values. It applies that mask to its model arrays, but returns the original `prepared` frame at [line 767](../src/language_reading_predictors/statistical_models/pooled_levels.py#L767). The metadata writer explicitly assumes that a factory returns the final fitted frame.

For the current `lrp-rli-pl-001` data, preparation produces 214 rows. The likelihood uses 210. The payload correctly records 210 fitted rows and four incomplete rows, but `built.prepared.n_obs`, the subject-sequence fingerprint and the reuse contract describe 214 rows. The saved reporting configuration contains the same disagreement between top-level `n_obs` and `extra.n_child_wave_rows`.

The fitted likelihood itself uses the intended complete rows. The confirmed error is in its record of those rows. This prevents a reader or reuse check from relying on the stated fingerprint as the identity of the fitted data. It also gives consumers inconsistent arrays if they obtain labels or observed scores from `prepared` rather than the trace. I have not shown that the stored predictive-coverage table is wrong; its generic implementation reads aligned observations from the trace.

**Repair.** Subset `PreparedData` once before building the final model arrays and return that subset. Retain pre-filter counts separately for exclusion accounting. Verify row order and child indexing as well as counts. The metadata, model coordinates, observed values and subject fingerprint should all identify the same 210 rows. Regenerate identities only after checking them against the saved trace; do not substitute a new fingerprint without that check.

### 3. P2 Pooled-level predictor units differ from their stated comparators

**Implementation and reporting defect.** The pooled-level factory uses `logit(clip(score / maximum, .001, .999))` at [`pooled_levels.py` line 630](../src/language_reading_predictors/statistical_models/pooled_levels.py#L630). The mechanism and concurrent factories use the Haldane-corrected transform `log((score + .5) / (maximum - score + .5))`. The pooled-level code and [report partial](../docs/models/_partials/_results_pooled_levels.qmd#L81) nevertheless say that their per-standard-deviation exposure units are the same.

These transforms differ throughout the scale and differ most at its boundaries. For letter sounds, zero out of 32 becomes -6.90675 under clipping and -4.17439 under the correction. On the current 210 fitted rows, four letter-sound scores are zero and nine are at the ceiling. After each vector is standardised, their largest absolute difference is 0.67245 standard deviations. Their correlation is 0.97499, so the difference is not a constant shift or rescaling that standardisation removes. The same clipped transform is used for pooled-level skill adjusters.

A coefficient per standard deviation of one transformed vector is therefore not a coefficient per standard deviation of the other. Even after using a common transform, a pooled four-wave standard deviation can differ from a transition or single-wave standard deviation. Different outcomes, adjustments and time windows also remain different estimands.

**Repair.** Either use the common corrected transform or document clipping as an intentional model sensitivity and remove the common-unit claim. Record the fitted transform and its scale. A change to the transform requires refitting the bounded-exposure pooled models, since it changes the regressors and the effective prior on score differences. Compare models through declared score-scale contrasts over shared support when the substantive question requires comparison.

### 4. P2 The attendance-adjusted presence coefficient is given an unwarranted randomisation label

**Interpretation defect.** [`DoseResponseRunPlan.coefficient_meanings`](../src/language_reading_predictors/statistical_models/dose_response.py#L282) says `theta_treated` is identified by randomisation in period 1. The [results partial](../docs/models/_partials/_results_dose_response.qmd#L17) and findings builder repeat this claim. The model does more than compare assigned groups after baseline adjustment. It also conditions on treatment-induced attendance and, by default, each child's mean attendance across treated periods.

The child's mean includes later-period attendance, so the period-1 equation conditions on information observed after its outcome. Centring attendance over all treated rows does not make that information pre-treatment. Nor does centring make its period-1 group average zero. In the current word-reading design, the treated-row mean is 66.2344 sessions, while the period-1 immediate-arm mean is 72.7143. The period-1 means of the between-child and within-child dose regressors are -0.11314 and 0.45780.

At a fixed period-1 baseline profile, the modelled treated mean linear predictor therefore contains the presence coefficient plus both dose contributions. `theta_treated` alone is a conditional contrast at zero on the fitted dose regressors. Randomisation identifies the assignment contrast before conditioning on treatment-induced variables, subject here to the documented observation assumptions. It does not by itself identify a dose-controlled contrast in the presence of the DAG's unmeasured ability-to-attendance and ability-to-outcome paths. The general problem of adjustment for a treatment-affected variable also applies to randomised experiments. See Rosenbaum (1984), DOI [10.2307/2981697](https://doi.org/10.2307/2981697).

**Repair.** Describe `theta_treated` as a model-based conditional presence association in this attendance-adjusted model. Keep the assigned-arm effect in the ITT analysis or a separate period-1 model using only pre-assignment precision terms. If a causal presence or dose effect is intended, define the intervention on attendance and state the additional identification assumptions and supported contrasts. This finding does not dispute that assignment was randomised or that the dose slopes are already labelled observational.

### 5. P2 A small within-child slope does not exclude direct influence

**Interpretation defect.** The [pooled-level report at lines 107 to 110](../docs/models/_partials/_results_pooled_levels.qmd#L107) says a large between-child coefficient beside a small within-child coefficient is a pattern predicted by a shared cause but not by direct influence. The model does not establish that distinction.

The probes give a counterexample with four waves. The true outcome equals the true contemporaneous exposure, so the direct effect is exactly one. Stable differences between children have standard deviation two, genuine within-child changes have standard deviation 0.1, and independent exposure measurement error has standard deviation two. Regressing on the observed child mean and within-child deviation gives a between-child slope of 0.80319 and a within-child slope of 0.00241. Thus the stated pattern occurs despite a direct effect. Averaging reduces the error in the child mean, while little true within-child variation remains relative to its measurement error.

This example does not show that these particular error levels occur in the study. It shows why the report's exclusion claim is false without a measurement and time-response model. Ceiling compression, delayed influence, little exposure change and other time-varying causes also deserve assessment. The separation of within-person and between-person associations needs its own assumptions when predictors change over time. See Curran and Bauer (2011), DOI [10.1146/annurev.psych.093008.100356](https://doi.org/10.1146/annurev.psych.093008.100356).

**Repair.** Report the fitted separation descriptively. A larger between-child association is compatible with stable shared causes, but does not distinguish them from a direct pathway. Any stronger claim needs explicit competing models and checks on predictor reliability, change support and temporal ordering.

### 6. P2 K-fold scores lack a check on latent integration precision

**Validation gap.** [`_score_held_out`](../src/language_reading_predictors/statistical_models/new_child_kfold.py#L508) correctly averages predictive densities before taking their logarithms. It draws child-level variables from the population and scores observations held out from fold fitting. However, it uses a fixed latent-draw count and records no independent-batch agreement, numerical Monte Carlo error or adaptive stopping result. [`KFoldValidation.complete`](../src/language_reading_predictors/statistical_models/new_child_kfold.py#L224) checks scored-child coverage and fold convergence only.

A well-sampled fold posterior does not establish the precision of the later integral over a child's latent variables. The standard error computed from variation in child log scores measures between-child variation in the predictive score. It is not the numerical error of that integral. This matters when a child's likelihood is concentrated in a region rarely reached by population draws. A finite answer can still be sensitive to the integration seed or budget.

The locally stored historical-joint runs `lrp-rlm-jc-001` and `lrp-rlm-jc-102` both declare complete five-fold validation for 71 children with 64 latent draws per posterior draw. Their validation table does not quantify this numerical uncertainty. The integrated PSIS implementation already has separate batch checks; the K-fold route does not. I have not shown that either stored K-fold score is inaccurate. The missing check means their precision remains unverified by the saved evidence.

**Repair.** Re-score existing fold traces with independent latent batches and larger budgets. Record pointwise and total-score agreement, then make acceptable precision a separate criterion from fold convergence and coverage. Increase the budget or use a more efficient integration scheme when agreement is inadequate. This can begin from saved fold posteriors rather than repeating fold sampling. Batch agreement itself should still be described as a stability check, not a bound on integration error. Predictive-score definitions and their sampling standard errors are discussed by Vehtari, Gelman and Gabry (2017), DOI [10.1007/s11222-016-9696-4](https://doi.org/10.1007/s11222-016-9696-4).

## Methodological assessment across the repository

### Intervention and causal interpretation

The main ITT implementation keeps assignment as the comparison and uses baseline score and age as precision terms. The methods distinguish the archived available-case population from the full randomised cohort. They describe arm-specific exclusions, archive sensitivity and missing-outcome scenarios. The original study used a waiting-list randomised design with 57 children. See Burgoyne et al. (2012), DOI [10.1111/j.1469-7610.2012.02557.x](https://doi.org/10.1111/j.1469-7610.2012.02557.x). The archived analysis and its further exclusions still need the repository's stated selection assumptions; the design alone does not verify them.

The later-wave models generally retain the distinction between an early-start versus delayed-start schedule contrast and an intervention versus no-intervention effect. The `did` treatment-period coefficient is an adjusted gap in levels; the level-factor coefficient is a change from its reference gap. Agreement between them can be useful, but neither their parameter names nor agreement turns them into the same estimand. The period-1 gain-factor headline and its required period-1-only sensitivity are appropriate safeguards against allowing later treated periods to define a randomised headline.

Assessment intervals remain a substantive uncertainty. The documented constant-rate timing scenarios state assumptions and no longer claim to bound the assessment-time effect. I did not fit a continuous-time alternative. An aligned window or a later-wave comparison does not automatically remove timing, onset-age or window-length differences.

Mediation now uses common pre-exposure adjustment information across its legs, exact summation for bounded count mediators and checked quadrature for normal mediators. The software labels the decomposition as model-based and acknowledges unmeasured ability and treatment-induced dose confounding. Those qualifications are essential. Interventional terminology does not supply the missing confounding information. A mediator-coefficient tipping sweep changes one fitted slope; it is not a measurement of all possible unmeasured confounding.

### Measurement priors and missing observations

The bounded-score likelihoods are stated as working models for totals. This avoids equating different test items with identical Bernoulli trials. Each denominator, floor rule and link must still match the instrument and the target population. Zero and ceiling mass, group-specific shape and prior pushforwards matter alongside broad predictive-interval coverage. The blending floor is conditional on the assumed random-guessing mechanism; it is not evidence that every child follows that mechanism.

Several families anchor priors or standardisation to observed data. The code often labels these choices as empirical Bayes or data-informed. Such fits condition uncertainty on those choices. A prior-predictive plot made after setting its centre from the same outcome is not an independent check of the outcome's location. I found no basis to treat a successful sampler or a prior-sensitivity tick as evidence that latent ability is measured or that the causal graph is correct.

Mean imputation with missingness indicators is an explicit adjustment policy, rather than a general solution to informative missingness. Complete-case exposure variants help assess sensitivity, but they also change the analysis population. Differences between their results can reflect that population change as well as imputation. The quarantine and bounded-count validation protect against known and future invalid cells; they do not verify all source-archive transcription or ascertainment decisions.

The correlated-factor models impose measurement structure, positive within-domain loadings and constraints on explained variance. The longitudinal version also imposes loading and residual invariance over waves. These constraints help define and estimate the model; a convergence pass cannot test that every constraint represents the instruments. Reliability, factor separation and correlations remain conditional on that structure. Two indicators per domain are a construction requirement, not an unconditional guarantee of identification under every correlation pattern or prior.

### Prediction and model comparison

The repository now labels conditional row prediction separately from prediction for a new child. The shared child maps, fresh latent draws and checks for undeclared child variables address an important source of apparent predictive success. Gain-factor validation holds out all transitions of a child together, so an outcome is not retained as a later baseline in the same validation exercise. Row-level diagnostics in other repeated-measure families answer a narrower, explicitly conditional question.

Model comparisons need the same observed outcomes, child set, held-out unit and prediction target. A gain-factor child score, a mechanism row score and a joint multi-outcome score cannot be ranked as interchangeable predictive evidence. Repeated analyses of the same children are also not independent confirmations. The remaining K-fold integration gap is finding 6.

The historical-joint factory centres child effects within groups and, in its within-child companion, also within group-by-wave cells. This induces dependence between children's latent departures. Its current pipeline uses actual grouped K-fold refits. I therefore did not report the generic integrated-PSIS formula as a defect in its current published route. If that alternative is enabled for this family, its importance-weight identity must be checked for the centred dependence structure. Declaring and redrawing every latent variable is necessary, but does not establish that the marginal likelihood factorises by child.

### Gradient boosting and exploratory analysis

The active machine-learning code groups cross-validation by child. The permutation design keeps donors at the same wave and observed-wave schedule, and marks unsupported schedules as unassessable. Tuning has a child-separated inner early-stopping split. The methods correctly call the reused tuning folds post-selection internal cross-validation, rather than an independent generalisation estimate.

These protections do not make predictor rankings causal. Native missing-value handling can use observation patterns; importance conditional on schedule does not estimate importance in every future observation schedule. Correlated skills can exchange importance, and a SHAP direction depends on the fitted model and the other predictors. Bootstrap rank stability under fixed fitted choices does not account for every tuning and scientific-selection decision. A study-level prediction claim would benefit from a separate grouped validation procedure that includes the intended selection steps.

The legacy notebooks still contain exploratory random-forest analyses, outcome-dependent exclusion of large gains and scatter plots headed as influences. They use grouped cross-validation in the inspected predictive examples, but they answer earlier exploratory questions. Their outputs should not be used as validation of the current pipelines or as causal estimates. The main study-level results and discussion chapters are explicitly unfinished scaffolds, so this review does not endorse a finished study narrative.

## Coverage by statistical family

The counts below come from resolving the current registry. The assessment column states what I checked or what limits its interpretation. It is not a certificate that every saved fit passes every check.

| Family              | Models | Assessment                                                                                                                                                                                                                    |
| ------------------- | -----: | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `itt`               |     31 | Assignment, baseline adjustment, off-floor outcomes, blending links, available cases and uncertainty rules checked.                                                                                                           |
| `joint`             |      7 | Outcome masks, dependence companions, child mapping and new-child prediction checked. Cross-outcome covariance needs its declared companion evidence.                                                                         |
| `mechanism`         |     53 | Fitted-row masks, score transforms, linear and nonlinear exposure terms, moderation, within-between split and frozen refit design examined. Skill terms remain associations.                                                  |
| `mediation`         |     16 | Common adjustment, temporal sensitivity, exact count integration, normal quadrature and derived-effect checks examined. Finding 1 affects the blending sensitivity sweep.                                                     |
| `mediation_multi`   |      4 | Ordered and parallel mediator calculations and separate path definitions examined. Path decomposition remains model-dependent.                                                                                                |
| `did`               |     22 | Gap parameterisation, reference waves, timing and observational dose terms examined. Later gaps concern assigned schedules.                                                                                                   |
| `gain_factors`      |     33 | Own baseline, off-floor pre-score, treatment support, period-1 headline, link pushforwards and child holdout examined.                                                                                                        |
| `level_factors`     |     23 | Reference-gap coding, zero-sum wave intercepts, baseline group nuisance and later regime roles examined. Its synthetic arm-gap-change pushforward has a different definition from a response-scale difference-in-differences. |
| `aligned`           |     10 | One window per child, cohort indicator and exposure sensitivity examined. Onset age, timing and window length remain confounding limits.                                                                                      |
| `adjusted`          |      7 | Own baseline, complete cases, skill adjustment and repeated-row child mapping examined. Observational skill and cohort terms remain non-causal.                                                                               |
| `corr_factor`       |      5 | Marginal Gaussian measurement covariance, conditional factor draws and structural count equation examined. Reliability depends on measurement constraints.                                                                    |
| `long_corr_factor`  |      1 | Trait/state covariance, invariant measurement structure, missing-pattern likelihood and exact child likelihood recovery examined.                                                                                             |
| `dose_response`     |      6 | Treated-row dose centring, child means, period support and child holdout examined. Finding 4 qualifies the presence term.                                                                                                     |
| `lcsm`              |      6 | Latent recursion, lagged couplings, arm-window intercepts, missing-cell masks and process/measurement variation examined. Unequal intervals and measurement assumptions limit coupling interpretation.                        |
| `horseshoe`         |      7 | Regularised shrinkage construction, target choice, predictor imputation and level random intercepts examined. Rankings are conditional and prior-sensitive.                                                                   |
| `growth`            |      4 | Age scale, intercept/slope terms, masked likelihood and common-factor structure examined. Pooled age combines between-child and within-child information.                                                                     |
| `historical_growth` |      9 | Group/wave cell means, group-centred child effects and dispersion construction examined. Zero-offset predictions need their median-child interpretation.                                                                      |
| `historical_joint`  |      3 | Correlated child effects, double-centred within-child departures, matched-child growth and K-fold route examined. Finding 6 concerns integration validation.                                                                  |
| `survival`          |      2 | Baseline-floor entry, first observed off-floor event, censoring at gaps, later-period hazards and grouped child likelihood examined. Event detection is interval-based and observation-dependent.                             |
| `block_exposure`    |      5 | Staggered block coding, observed-wave intercepts and child effects examined. A random intercept does not guarantee control of all arm differences or establish parallel untreated trajectories.                               |
| `concurrent`        |     14 | Single-wave row restrictions, focal-outcome masks, same-wave transforms, predictor imputation and nuisance terms examined. Conditional skill coefficients have no temporal identification.                                    |
| `joint_mechanism`   |      2 | Bivariate covariance, latent conditional slope, denominator stability, cell masks, fixed fold exposure scale and child holdout examined. The slope ratio is unbounded and is not a pathway share.                             |
| `pooled_levels`     |      7 | Fitted masks, exposure scale, within-between terms and report interpretation examined. Findings 2, 3 and 5 apply.                                                                                                             |

## Verification and limits

Across the initial run and the separate completion runs, 4,215 distinct tests passed and one test was skipped. The initial run passed 4,200 tests but had nine Quarto rendering failures. Re-running the 17 tests in those two report modules with access to Quarto's log directory and local kernel sockets passed all 17. The five type-coverage tests and the real PyMC/nutpie posterior sampling smoke test also passed. Ruff reported no source-code findings.

The test commands used the installed locked environment with `uv run --no-sync`. A fresh cache made `uv run --locked` attempt a network fetch unavailable in the restricted shell, so that attempt supplied no test evidence. Numerical tests used temporary writable caches and `cxx=''` for PyTensor because the local C-linking path is unavailable. The isolated sampler test also received that setting through `PYTENSORRC`, since it replaces `PYTENSOR_FLAGS` in its subprocess. This verifies the installed interpreter and sampling path; it does not reproduce the Windows production environment or fit-quality diagnostics for every model.

The added probes pass without resampling. Existing tests did not catch the omitted mediation caller argument, the pooled-level identity mismatch or the false unit-equivalence claim. Future tests should compare quantities that must agree for statistical reasons, such as a zero-bias sensitivity and its primary result or a recorded fitted identity and its likelihood rows. Tests that merely repeat the same implementation would not provide that protection.

The current review leaves several tasks open. Source-archive verification of the quarantined cell, measurement reliability and invariance checks, assessment-date recovery, independent validation after exploratory model choices and full trace-based verification of every production result need evidence beyond this review. The book's shared `show_gated` helper checks convergence but not the full release decision; it has no current result callers in the inspected scaffold. It should consume the publication verdict before substantive results are added.

## Proposed correction order

1. Thread the fitted outcome link into the blending mediation sensitivity, regenerate its saved outputs and compare the zero-bias row with the primary result.
2. Return the final fitted pooled-level frame and verify every metadata identity against the likelihood rows.
3. Decide whether pooled-level clipping is intentional. Use the common score transform and refit, or retain clipping with an explicit sensitivity label and distinct units.
4. Correct the dose-presence and pooled within-child interpretation claims. These prose repairs need no new posterior fit.
5. Re-score the saved K-fold traces with independent latent batches and record numerical stability separately from convergence and scored-child coverage.
6. Re-evaluate affected publication decisions after corrections. Passing those checks would establish the corrected computational contracts; the stated causal, measurement and observation assumptions would still need substantive evidence.
