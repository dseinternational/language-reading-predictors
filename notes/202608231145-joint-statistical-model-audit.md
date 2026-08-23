> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# Joint statistical-model audit and proposed remediation, 2026-08-23

## Status and scope

This is a review record and remediation proposal, not an implementation change or a release approval. It covers all eleven registered models in the three families that jointly model outcomes or mechanisms:

**Tracking issue:** [#588](https://github.com/dseinternational/language-reading-predictors/issues/588)

- `joint`: `lrp-rli-itt-012`, `lrp-rli-itt-015`, `lrp-rli-itt-016`, `lrp-rli-itt-115`, `lrp-rli-itt-215`, `lrp-rli-itt-216` and `lrp-rli-itt-315`;
- `joint_mechanism`: `lrp-rli-jm-001` and `lrp-rli-jm-002`;
- `historical_joint`: `lrp-rlm-jc-001` and `lrp-rlm-jc-002`.

The review followed each family from its typed settings and run-plan resolver through preprocessing, PyMC factory, primary and secondary fit orchestration, posterior summaries, predictive and PSIS-LOO diagnostics, release decision, key findings and report prose. It also checked the relevant registered specifications, focused tests, current source-data row requirements and the existing reporting artefacts where available.

The central conclusion is mixed. The core array indexing, count likelihoods, sign conventions and most posterior-summary calculations are internally coherent. However, five issues affect whether the current outputs support their stated release or scientific claims, and several further issues weaken diagnostic validity or reproducibility. In particular, the correlated joint-ITT companions are not merely uncertainty corrections to the factorised fits, the joint-mechanism reports over-interpret operational score associations, not every potentially published sub-fit receives the full diagnostic lifecycle, and the promised `lrp-rlm-jc-002` prior sensitivity has not been fitted or enforced.

## Release-critical findings

### 1. The correlated joint-ITT companions change the fitted mean model and estimand

**Affected models:** `lrp-rli-itt-215`, `lrp-rli-itt-216` and `lrp-rli-itt-315`, together with their factorised parents `015`, `016` and `115`.

The correlated companions add a child-specific bivariate-normal offset on the logit scale to each outcome's linear predictor in `build_joint_model`. `_joint_ame_draws` then computes the probability-scale treatment contrast by subtracting the treatment contribution from the stored linear predictor and toggling treatment while holding each posterior draw of that child's latent offset fixed. The parent models contain no such offset. The companion therefore uses a different hierarchical response model, and its reported average marginal effect is a finite-study-sample standardisation conditional on posterior draws of the fitted children's latent offsets, not the parent's point estimand with only its covariance corrected.

This directly contradicts repeated documentation claims that the point estimate is “unaffected”, “stands”, or should agree to Monte Carlo error while only the interval changes. Through the nonlinear inverse-logit function, a mean-zero random effect changes probability-scale marginal effects. Adding the random-effect mixture can also change the posterior of `tau` and `kappa`. Agreement may happen empirically, but it is not a mathematical invariant.

**Consequence:** the current parent-companion comparison must not be described as an uncertainty-only dependence correction. Integrating a new latent offset over its bivariate-normal distribution would define a useful new-child population-marginal estimand, but it would remain an estimand under the companion's different hierarchical model and would not restore point-estimate invariance with the parent.

**Proposed fix:** first choose and record the intended estimand. If the hierarchical model is retained, describe the companions as alternative dependence models, remove every point-invariance claim, and report separately the current finite-sample latent-conditional standardisation and a new-child population-marginal standardisation that integrates over the random-effect distribution. If the scientific target is instead the parent's marginal effect with paired uncertainty while preserving its marginal model, use a child-level paired bootstrap, randomisation inference, a robust marginal method or another dependence construction designed for that target. Validate the choice with simulation under zero, positive and negative within-child dependence.

### 2. The release gate does not test the declared joint contrast or fully bind the parent-companion pair

**Affected models:** the six two-outcome joint-ITT contrast fits.

The automatic computation gate assesses the sampled model, including the LKJ block's free variables, for convergence and related sampling failures. The subsequent robustness decision, however, classifies only the vector of conditional-logit treatment coefficients, `tau`. Although power-scaling rows are emitted for `sigma_outcome` and `u_corr_pair`, those diagnoses do not enter the release decision, and the derived probability-scale average-marginal-effect difference is not itself tested. Clean marginal `tau` diagnoses do not guarantee that a nonlinear difference of standardised average marginal effects is stable.

The parent-companion lookup uses the declared sibling identifier, the matching configuration-directory suffix, the companion's self-reported model identifier, broad publication eligibility and the source-data checksum. It does not verify ordered outcomes, contrast direction, retained-row identity, standardisation population, denominators, effective recipe, fitted-data digest or source-code provenance. A stored parent whose plan predates `dependence_companion` also bypasses the qualifier. A stale or structurally mismatched companion can therefore qualify a parent.

**Consequence:** a parent can be labelled dependence-checked without machine-verifiable evidence that the companion evaluated the same population and estimand, while the released headline contrast may remain sensitive to the dependence model.

**Proposed fix:** evaluate robustness for the actual per-draw average-marginal-effect difference and bind the pair with a fingerprint covering ordered outcomes, contrast sign, row identity, standardisation population, denominators, effective settings, sampling configuration, source commit and input checksums. Assess the dependence block through its consequence for the declared contrast, for example by stability across plausible dependence priors or a paired alternative analysis, rather than requiring every nuisance correlation to be sharply identified when its residual scale is near zero. Fail closed on binding mismatches and on legacy plans lacking the required pairing record. Tests should deliberately alter each binding field and demonstrate rejection.

### 3. `lrp-rli-jm-001` can headline a wave that did not receive the primary fit's full diagnostic lifecycle

`lrp-rli-jm-001` fits one bivariate levels model per wave. Only the selected anchor wave receives the full primary-fit lifecycle. Each non-anchor wave is sampled through `run_subfit` and receives divergence, BFMI and default free-random-variable R-hat/ESS checks plus persisted provenance, but it receives no prior-predictive check, posterior-predictive check, power-scaling sensitivity analysis, deterministic-estimand convergence check or persisted trace. The wave table nevertheless uses that narrower free-variable verdict as its `converged` flag, and the key-findings generator may select any wave from the filtered table.

Filtering to `converged == True` prevents an obviously failed sub-fit from being quoted, but the sub-fit verdict is not equivalent to the project's publication gate. All current non-anchor wave fits passed that narrower check, so this is a release-control gap rather than evidence that the stored estimates are numerically wrong. Selecting the most resolved wave after examining all four also introduces an exploratory winner-selection step that is not reflected in the headline wording.

**Consequence:** an incompletely diagnosed, post-selected wave can become the report's lead result even though the release decision is anchored to another fit.

**Proposed fix:** either pre-specify one primary wave and label every other wave exploratory, or give every headline-eligible wave a persisted trace and a complete fit-specific diagnostic bundle. Retain the existing sub-fit provenance, but do not treat its free-variable convergence flag as equivalent to the primary release gate. The key-findings builder must select only release-eligible wave results and label any across-wave “clearest” selection as exploratory.

### 4. The joint-mechanism scientific interpretation exceeds what the models identify

The fitted `delta_ls_decoding` quantity is a legitimate operational contrast between two adjusted test-scale associations: the difference in log-odds slopes per standard deviation of the exposure. It does not, by itself, identify “decoding specificity”. Word reading and nonword reading have different item counts, score distributions, discrimination, reliability and ceiling behaviour (`W` has a 79-item ceiling; `N` has a 6-item ceiling and pronounced flooring), and the model does not calibrate them to a common latent outcome scale. A common latent ability can therefore generate a non-zero slope contrast solely because the two tests load differently on that ability.

Likewise, `share_retained` is the ratio of a latent-Gaussian conditional slope to an unconditional fitted slope. It is not a mediation proportion or a causal path fraction, so phrases such as “runs through the decoding channel” are not supported. Nor is it justified to describe `jm-001` and the `ca-010`/`ca-011` observed-score sensitivities as bracketing one underlying answer: they differ in likelihood, treatment of missing predictors, conditioning target, fitted rows and estimand.

**Consequence:** the numerical summaries can be correct while the prose gives them a stronger construct-valid or causal interpretation than the design supports.

**Proposed fix:** rename the headline as an operational word-versus-nonword score-slope contrast; describe `share_retained` as a ratio of adjusted associations; remove pathway, mediation and bracketing language; and state the floor, ceiling, discrimination, reliability and scale limitations beside the result. If construct-level decoding specificity is the target, add a measurement model or a pre-specified reliability-standardised sensitivity rather than inferring it from raw test-score slopes.

### 5. The promised wider-prior sensitivity for `lrp-rlm-jc-002` is absent and unenforced

The current source declares only the `HalfNormal(0.5)` primary fit. No registered companion or historical-joint pipeline or sub-fit path changes `sigma_within_prior_sigma` to 1.0, and no repository-visible trace or provenance artefact records the promised alternative-prior fit. The report nevertheless says that a `HalfNormal(1.0)` sensitivity “checks” the conclusions. Power scaling is a useful local prior-and-likelihood perturbation diagnostic, but it is not the named alternative-prior refit.

**Consequence:** the reproducible pipeline does not perform a robustness check that the report describes as completed. This matters because the fitted `sigma_within` posterior determines whether each measure, and therefore each correlation pair, passes the current resolvability rule.

**Proposed fix:** add a typed, provenance-recorded wider-prior sub-fit or registered companion. Compare posterior medians, 50% and 89% intervals, sign probabilities and resolvability classifications across the two independently sampled fits; do not pair MCMC draws by draw number. Add a historical-joint-specific required artefact and robustness decision rather than placing this descriptive model in the causal-family gate. Until that check is current and passes its own convergence gate, withhold the prior-robustness claim or label it explicitly preliminary. If the sensitivity is no longer required, remove the promise only after a dated statistical decision explains why power scaling is an adequate replacement.

## Important diagnostic and comparison findings

### 6. Outcome-specific joint LOO-PIT uses cell-level rather than declared child-level weights

The main PSIS-LOO calculation correctly aggregates likelihood contributions to the declared child unit. The outcome-specific LOO-PIT helper instead constructs an ArviZ object containing the focal outcome's raw cells and raw pointwise likelihood, so each focal child-outcome cell receives its own leave-one-cell-out weight. The implementation and report disclose that outcome-specific weights are recomputed, but they do not state plainly that this is a different prediction target. In a correlated joint-ITT companion, the same child's sibling outcome remains in the posterior and informs the correlated latent offset. In `lrp-rli-jm-002`, the child's other outcomes and transition rows remain informative. The mismatch is less consequential for the factorised joint-ITT parents because their outcomes share no local dependence block, but the nominal holdout unit is still not the stored child-level target.

**Consequence:** these plots assess conditional prediction of a held-out outcome cell for an otherwise observed child, not calibration for a wholly left-out child.

**Proposed fix:** either label the current plots explicitly as conditional leave-one-cell-out diagnostics or construct an estimand-specific child-level diagnostic. For joint ITT, one focal outcome value per child can be paired with child-aggregated weights. For repeated-transition models, simply attaching one child weight to several cells is not automatically a valid scalar LOO-PIT; define a wave-specific target, use exact leave-child-out predictive draws, construct an explicitly multivariate or grouped calibration diagnostic, or suppress the plot until a defensible mapping is available. Add tests where one child supplies multiple outcomes and transitions.

### 7. `lrp-rli-jm-002` and its claimed single-outcome comparators do not use the same fitted rows

On the current data, the joint transition frame contains 153 rows, with 152 observed cells for each of `W` and `N` and 151 rows on which both post-scores are observed. The single-outcome comparison models use 152 rows for `mech-096` (`N`) and 156 for `mech-101` (`W`). Four otherwise valid word-reading transitions, about 2.6% of the `mech-101` sample, are absent from `jm-002` because the joint fit requires both outcomes' baselines. The exposure standardisation is consequently calculated over different populations too.

**Consequence:** the current “changes only the dependence treatment” and “like-for-like” descriptions are not literally true. Differences can arise from population and scaling changes as well as joint estimation. The row difference may have little numerical effect, but the comparison does not isolate dependence treatment alone.

**Proposed fix:** for a strict sensitivity comparison, either support outcome-specific baseline masking within the joint likelihood, with explicit handling of missing unused baseline entries, or refit both single-outcome comparators on the exact joint-model rows with the same exposure standardisation. If neither is done, describe the comparison as approximate and report the row-set difference. Persist and compare row digests in the cross-model table.

### 8. Historical-joint predictive validation is not mathematically precluded, but its target is undefined and exploratory PSIS was unreliable

The historical-joint run plans disable PSIS-LOO because the model has multiple likelihood nodes. Multiple nodes are not themselves an obstacle: the three nodes share an observation coordinate, so their conditional log-likelihood contributions can be summed for each child-wave row. That construction does not, by itself, define the predictive target. Row-level validation predicts another occasion for an already observed child and, for `jc-002`, must correctly integrate or otherwise handle the held-out occasion-specific latent deviation. Child-level validation predicts a new child and must integrate the stable and within-child latent effects under an explicit generative treatment of the sample-dependent centring constraints.

An exploratory post-hoc probe of older stored reporting traces produced maximum Pareto-k values of approximately 0.92 and 1.21 at the child-wave unit and 1.64 and 1.63 at the child unit for `jc-001` and `jc-002`. These traces predate current prior revisions, so the values are not diagnostics of the current specifications. They nevertheless show that straightforward conditional PSIS was unstable in those fits; they do not show that a predictive target is mathematically undefined.

**Consequence:** the production pipeline currently has no target-specific out-of-sample assessment, and its stated reason for omission is incorrect.

**Proposed fix:** first define whether the target is a new occasion for a known child or outcomes for a new child. For the latter, prefer grouped child-level K-fold or exact refits with held-out child effects integrated from their population distribution. Use moment matching only after validating that the likelihood and unit correspond to the intended target. Until a current target-specific implementation exists, state that production LOO is not computed and that exploratory PSIS on older stored traces was unreliable; do not report its ELPD or LOO-PIT as a current diagnostic.

### 9. Joint influence refits do not recompute the declared contrast

The exact influence path recomputes outcome-specific marginal effects for retained and refitted data but does not reconstruct the full declared average-marginal-effect difference. Per-outcome movement is not sufficient to determine contrast movement because both magnitude and posterior covariance matter.

**Consequence:** an influence audit can appear reassuring while the actual headline contrast is more sensitive.

**Proposed fix:** extend the contrast-summary path to accept the same retained-row mask used by `tau_summary_joint`, then compute the declared per-draw contrast for the full primary population, the primary posterior standardised over retained children, and the exact refit on those retained children. Persist its median, interval, direction probability, composition shift, refit shift and total shift. Any claim that the influence analysis preserved the scientific finding must refer to this declared contrast, not only to its marginal components. Generic joint influence is not currently a publication-gate criterion; if it becomes mandatory, add an explicit, pre-specified contrast-stability rule to `evaluate_publication`.

### 10. `share_retained` lacks a denominator-stability rule

`share_retained` is a ratio estimand with two important instability routes: division by an unconditional word-reading slope near zero, and the residual-scale ratio `sigma_W / sigma_N` inside the conditional slope. A finite Monte Carlo mean can therefore look reassuring even when the posterior ratio is highly skewed, heavy-tailed or lacks a scientifically useful mean. The pipeline currently writes and the report renders that mean without an automated denominator- or scale-stability check.

**Consequence:** a finite-looking summary can be a numerical property of the sampled draws rather than a stable scientific quantity.

**Proposed fix:** stop reporting the posterior mean. Report the median and quantile intervals only after a pre-specified, scale-appropriate stability rule shows that `beta_mech[W]` is meaningfully separated from zero and that `sigma_N` is not concentrated near zero. If the rule fails, mark the ratio undefined or unstable and report the unconditional word-reading slope, the word-reading slope conditional on nonword reading, and their absolute difference, with the sign convention stated explicitly. The failure state should be explicit in the CSV, key findings and report.

## Validation and policy hardening

### 11. Historical-joint score and index validation can silently coerce invalid input

The panel loader checks only whether a non-missing score exceeds its ceiling. The factory later converts fitted score arrays to integers, so a fractional score such as 3.5 is silently truncated to 3. A negative or non-finite value can reach model construction and produce an invalid likelihood. The loader also casts wave and group codes to integers before validating them, so fractional codes can be truncated, and it checks allowed group codes without asserting that each child remains in one cohort across waves.

The current source data pass explicit finite, integer, lower-bound, upper-bound and group-stability probes, so this is a latent validation defect rather than evidence of current data corruption.

**Proposed fix:** before any integer cast, require every non-missing requested score to be finite, integer-valued and within `0 <= y <= n`; retain `NaN` for the documented missing-data selection. Likewise validate wave and group codes as finite integers before casting, and require `nunique(group) == 1` within each child. Add separate failing tests for fractional, negative, infinite, over-ceiling, fractional-index and changing-group inputs, plus fixtures showing that permitted missing core and extension observations follow the documented selection rules.

### 12. The mandatory phoneme-blending response-link policy does not cover the joint estimate

`lrp-rli-itt-012` produces, and is capable of publishing, a graded ordinary-logit estimate for phoneme blending (`B`). Its key-findings builder already appends prose saying that this result is conditional on the mandatory `lrp-rli-itt-008` / `lrp-rli-itt-108` response-link sensitivity. However, the trace-backed pairing gate is restricted to `kind == "itt"` and therefore does not verify that condition for the joint model.

**Consequence:** if the response-link policy is estimand-wide, the joint table and pairwise rankings provide an unguarded alternate route to a phoneme-blending treatment claim. The prose caveat warns the reader but does not establish that the required evidence exists.

**Proposed fix:** record the policy scope explicitly. If it is estimand-wide, extend bundle validation to the joint `B` row and every cross-outcome comparison involving it, or suppress those released summaries when the bundle is not ready. If it is intentionally limited to the single-outcome model of record, state that the joint `B` estimate is a secondary structural cross-check, is not independently release-qualified and cannot supersede or weaken the paired `008`/`108` conclusion.

## Lower-priority reporting and API corrections

- Historical joint writes all three prior-predictive plots as `prior_predictive_check_{measure}.png`, while the shared partial checks only for `prior_predictive_check.png`. Consequently, none of the three plots is rendered and the partial incorrectly says that no prior-predictive plot is available.
- The historical-joint posterior-predictive writer keeps the first measure's overlay unsuffixed and writes the other two with measure suffixes, while the diagnostics partial renders only the unsuffixed file. BPVS and digit-recall distribution overlays are therefore omitted. The pooled quantitative coverage and all per-measure calibration panels are rendered correctly.
- In `jc-002`, the Key Findings box correctly treats within-child correlation, or failure to resolve it, as the headline, but the detailed Results section presents the secondary between-child matrix first and calls the within-child result a “companion headline”. Its between-pair instructions also direct readers to the posterior mean despite the project's median convention. Use the median for pair summaries; if the posterior-mean matrix is retained because averaging correlation matrices keeps the result positive semidefinite, label that distinction explicitly. The growth-contrast prose likewise points to `mean` even though `q50` is available, whereas cell-average calibration means should remain means.
- Key findings selects the pair with the largest `abs(P(rho > 0) - 0.5)` among the three pairs; `jc-002` first restricts this to pairs passing the residual-scale resolvability rule. The chosen pair is therefore an exploratory, uncertainty-based selection rather than a pre-specified or largest effect, and should be labelled accordingly without implying multiplicity adjustment.
- Historical prose says per-child offsets cancel exactly. The stable child intercept cancels from a child's latent-logit difference, but not after the inverse-logit transformation; in `jc-002`, the wave-specific departure also does not cancel across waves even on the latent scale. Matching the same children correctly holds composition fixed but does not make item-scale growth independent of their latent offsets.
- Where the `jc-002` report says the logistic-normal residual contains “measurement noise”, specify extra-Binomial occasion-specific measurement or test noise. Ordinary Binomial item-sampling variation remains represented by the likelihood; it is the extra-Binomial component that is inseparable from genuine within-child fluctuation and can attenuate the latent correlation.
- The current source confirms the `UR` denominator is 12, but `lrp-rli-itt-115` and `lrp-rli-itt-315` still describe it as undocumented.
- The typed joint API permits a one-outcome residual-correlation declaration even though the one-dimensional LKJ construction fails. No registered model triggers this, but the resolver should reject fewer than two outcomes when residual correlation is enabled.
- The joint-mechanism coverage summary pools `W` and `N` cell-coverage indicators. This aggregate is mathematically defined, but it can conceal outcome-specific miscalibration and weights outcomes according to their observed cell counts. Add per-outcome coverage rows and retain the pooled value only as a secondary summary.
- Adjustment columns are screened for constancy on the loader's prepared frame, but later wave, exposure and outcome masks can change the final fitted design and no second variance check is applied. All registered `jm-001` and `jm-002` adjustment columns vary in the current fitted samples, so this is a latent robustness defect rather than a current failure.

## What was verified as correct

- Registered parent-companion inheritance, group recoding and contrast direction are consistent with the declarations.
- The flattened joint likelihood maps observed cells to the correct outcome denominator, and the reviewed current data respect their declared count bounds.
- The joint average marginal effects and contrasts are computed per posterior draw rather than by subtracting marginal summary statistics.
- The LKJ Cholesky orientation and scale application in the correlated joint-ITT block are internally correct; the problem is the interpretation of the resulting model, not a matrix-algebra error.
- The overall repeated-row PSIS-LOO aggregation uses the persisted child map where the family supplies it; finding 6 concerns the separate outcome-specific LOO-PIT reconstruction.
- The joint-mechanism `delta_ls_decoding` sign and conditional-slope algebra are implemented as declared.
- The historical-joint complete-case window, extension rules and within-child double-centring are coherent. Exact double-centring relies on, and is protected by, the enforced balanced-panel condition. The current data pass the additional validation conditions proposed in finding 11.

Focused family tests passed during the review, and Ruff passed on the reviewed statistical-model source. A fresh `lrp-rlm-jc-002` development fit was not completed. An uncapped attempt exited on a native abort signal during Numba/PyTensor compilation before sampling and without a Python traceback; a thread-capped retry was stopped on request while compiling. Direct model construction and actual-data prior-predictive checks with 100 and 1,000 draws succeeded. The aborted run is therefore treated as environment- or resource-limited, not as evidence of a model defect, and it supplies no current end-to-end convergence or posterior-predictive verification.

## Proposed implementation sequence and acceptance criteria

1. **Protect interpretation and release first:** correct the correlated-companion estimand and prose; gate the derived joint contrast and paired bundle; restrict `jm-001` headlines to fully diagnosed fits; relabel the joint-mechanism claims; and either run or withdraw the promised `jc-002` sensitivity.
2. **Repair predictive and comparison evidence:** align LOO-PIT with its holdout unit, make the `jm-002` comparison population explicit or identical, define a defensible historical-joint predictive assessment, recompute influence for the declared contrast, and add the ratio-stability rule.
3. **Harden inputs and documentation:** add pre-cast count validation, settle the phoneme-blending policy scope, repair historical plot/report ordering, refresh ceiling prose, reject one-outcome LKJ declarations and split joint-mechanism predictive coverage by outcome.

Completion requires more than passing unit tests. Each changed model must persist an unambiguous estimand, analysis-row identity, diagnostic unit and release rationale; every result capable of entering `key_findings.json` must be backed by its own gate-eligible fit; simulation or exact-refit checks must demonstrate the intended dependence behaviour; and affected reporting artefacts must be regenerated from current traces or refitted where the model changes. Until the first implementation group is complete, the affected dependence-qualified joint-ITT contrasts, joint-mechanism construct/pathway claims and `jc-002` prior-robustness claim should be treated as preliminary.
