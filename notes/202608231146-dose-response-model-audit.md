<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# Dose-response model audit and repair plan (2026-08-23)

Preliminary research data and models — all conclusions remain provisional.

- **Tracking issue:** [#587](https://github.com/dseinternational/language-reading-predictors/issues/587)
- **Scope:** `lrp-rli-dose-077`, `lrp-rli-dose-083`, `lrp-rli-dose-084`, `lrp-rli-dose-177` and `lrp-rli-dose-277`, plus their shared preprocessing, factory, pipeline, diagnostics, comparison, release and report code
- **Review type:** static code and statistical audit, executable construction of all five models, analysis-frame support checks and focused tests

## Decision

The dose-response family is computationally coherent but is not ready for unqualified scientific headline reporting. Four defects materially affect the stated estimands or their validation:

1. `dose-177` is not the exclusively pre-randomisation baseline-ability sensitivity it claims to be;
2. the dose coefficient conflates treatment presence, attendance intensity, between-child differences and within-child changes;
3. the item-scale `+1 SD` headline is mostly extrapolated beyond observed dose support; and
4. observation-row leave-one-out validation retains held-out outcomes as later transition predictors.

The fitted logit-scale slopes may still be described as model-dependent **observational associations** under the working model. They must not be read as causal effects. Until the estimand and validation defects are repaired, the current items-scale headline, the `dose-177` ability-robustness conclusion and the `dose-077` versus `dose-277` predictive conclusion should be withheld or explicitly qualified.

## Verification performed

- All five registered specifications resolved, constructed and compiled with a finite initial log probability.
- Analysis sizes were 156 transition rows / 53 children for word reading, 159 / 54 for letter sounds and 160 / 54 for phoneme blending.
- Twenty-nine focused dose run-plan, factory, key-findings and comparison tests passed.
- The exact word-reading analysis frame was reconstructed to check session support, scaling, phase structure and design-matrix rank.
- The posterior and prior item-scale transforms were compared directly.
- No completed dose-response traces were present in this worktree, so this audit did **not** independently re-check reporting-tier posterior estimates, convergence, posterior-predictive coverage, Pareto-k values or stored release decisions.

The passing tests are mainly structural. They do not exercise the temporal meaning of `dose-177`'s adjusters, support of the `+1 SD` contrast, equivalence of the phase-specific prior and posterior transforms, outcome leakage in LOO or report integration.

## Findings

### 1. `dose-177` uses treatment-affected skills — high

`phase_mode="all"` constructs the transitions t1→t2, t2→t3 and t3→t4, and defines each `pre_logit` from that transition's starting wave ([preprocessing.py](../src/language_reading_predictors/statistical_models/preprocessing.py#L599-L632)). The factory then enters row-specific L/E/B `pre_logit` values ([factories.py](../src/language_reading_predictors/statistical_models/factories.py#L2003-L2007), [factories.py](../src/language_reading_predictors/statistical_models/factories.py#L2050-L2053)).

Consequently, `dose-177` uses t1 L/E/B in period 1, t2 L/E/B in period 2 and t3 L/E/B in period 3. The last two sets can already have been affected by earlier intervention and attendance. This contradicts the specification and report, which say that only baseline, pre-dose skills are conditioned on and never anything downstream of dose ([lrp_rli_dose_177.py](../src/language_reading_predictors/statistical_models/lrp_rli_dose_177.py#L22-L24), [report](../docs/models/lrp-rli-dose-177/index.qmd#L23-L37)).

**Consequence.** Later L/E/B terms can overadjust mediated paths or introduce bias from treatment-affected time-varying covariates. `dose-177` therefore does not establish robustness to baseline general-ability confounding.

**Required repair.** Broadcast genuine pre-randomisation t1 L/E/B across every transition, or fit an explicit baseline latent-ability model. Rename the current fit if it is retained as a transition-start adjustment sensitivity.

### 2. The dose slope is neither a pure intensity effect nor a within-child effect — high

In period 1, all 25 waitlist rows have zero sessions while all 28 immediate-arm rows have 45–91 sessions; the arm–dose correlation is 0.970. Period 1 therefore mixes treatment presence with attendance intensity. In periods 2 and 3 both arms have received treatment, so assigned arm instead represents intervention order and treatment history.

The factory nevertheless fits one common arm coefficient, no current-treatment indicator and no arm×phase term ([factories.py](../src/language_reading_predictors/statistical_models/factories.py#L2017-L2042)). Unmodelled arm/history differences can therefore load onto the period dose slopes. The DiD dose implementation already contains a cleaner separation of current treatment from treated-centred session intensity ([factories.py](../src/language_reading_predictors/statistical_models/factories.py#L2186-L2200)).

The catalogue also calls this an “adjusted within-child association” ([model catalogue](../docs/models/README.md#L390-L394)), but the model contains one dose covariate plus a standard child random intercept. It does not separate each child's mean dose from deviations around that mean. The coefficient therefore combines between-child and within-child associations; the random intercept does not, by itself, control stable attendance–prognosis confounding.

**Required repair.** Choose and name the target explicitly. For an intensive-margin target, add current-treatment presence, centre sessions among treated rows, allow arm×phase/history effects and decompose child-mean dose from within-child deviations. Retain the observational label.

### 3. The `+1 SD` item-scale headline is outside observed support — high

The reporting pipeline adds one global standard deviation of sessions to every fitted row and averages the inverse-logit change ([dose_response.py](../src/language_reading_predictors/statistical_models/pipelines/dose_response.py#L221-L279)). For word reading, one global SD is 30.66 sessions, observed sessions range from 0 to 94, 84/156 shifted rows exceed their phase-specific observed maximum and 81/156 exceed the global maximum. Period-1 waitlist profiles are shifted from 0 to 30.66 sessions although no period-1 treated child received fewer than 45 sessions.

The inverse-logit arithmetic is correct under the linear working model, but the reported item magnitude is dominated by extrapolation. Posterior-predictive checks at observed doses do not validate these counterfactuals.

**Required repair.** Use a support-respecting contrast. The recommended primary choice is a treated-row observed-IQR contrast within phase; a treated-row SD or smaller local step is acceptable if its support is checked. Persist the raw-session contrast, averaging population, support bounds and number of excluded profiles.

### 4. Row-level LOO leaks held-out outcomes into later predictors — high

The run plan declares `loo_unit="observation_row"` ([dose_response.py](../src/language_reading_predictors/statistical_models/dose_response.py#L350-L355)). However, t2 is both period 1's outcome and period 2's fixed own-baseline predictor; t3 has the analogous dual role for periods 2 and 3. Removing one `y_post` log-likelihood factor does not remove that observed value from the next row's design matrix.

The first two held-out targets therefore remain in model fitting through later rows. This is beyond the documented known-child conditioning and is not prospective out-of-sample prediction. Because `dose-077` and `dose-277` share the leak, their relative comparison is less compromised than their absolute ELPD, but it is still a non-prospective pseudo-score.

**Required repair.** Use sequential leave-future-out validation, or leave one whole child out and integrate the random effect. If the current factor-level score is retained, label it narrowly and do not call it future or out-of-sample prediction.

### 5. The prior pushforward does not match the posterior estimand — medium-high

The posterior marginal correctly indexes `beta_dose_phase` by each row's phase ([pipeline](../src/language_reading_predictors/statistical_models/pipelines/dose_response.py#L230-L256)). The claimed matching prior pushforward instead applies scalar `mu_dose` to every row ([pipeline](../src/language_reading_predictors/statistical_models/pipelines/dose_response.py#L292-L313), [reporting.py](../src/language_reading_predictors/statistical_models/reporting.py#L964-L981)). It omits `sigma_dose` and the phase deviations.

A 1,000-draw word-reading check found a mean absolute discrepancy of about 1.31 items per prior draw between the current and matching transforms. The pooled `dose-277` calculation is unaffected; the four period-varying models are affected.

**Required repair.** Implement one shared row-level transform used by both prior and posterior paths, including identical slopes, phase indexing, dose contrast, row mask, direction and outcome denominator. This shared-writer problem overlaps the DiD audit in issue #576 and should be fixed once rather than independently in the two families.

### 6. `dose-084` lacks the required blending-link sensitivity — medium-high

`dose-084` uses the ordinary Beta-Binomial logit link, which allows an expected score below the one-third guessing level of its ten three-choice items. The project policy requires any headline phoneme-blending interpretation to be accompanied by the mechanically motivated guessing-floor link sensitivity ([METHODS.md](../METHODS.md#L46)). The dose-response settings expose no score-mean-link option and no companion is registered.

**Required repair.** Add score-mean-link support, register a matching guessing-floor companion, compare their estimand-scale prior pushforwards and make the pair a release prerequisite. If the policy was intended to cover ITT only, narrow `METHODS.md`; as written, it covers any headline B interpretation.

### 7. The identification prose contradicts the authoritative DAG — medium

The DAG contains `A -> IS`, `GA -> IS` and `IG -> IS`, with the same variables also affecting outcomes ([DAG](../dag/dag-language-reading.dagitty#L52-L56)). The dose documentation instead says that group is the sole confounder, age has no path to dose and the ability-to-dose edge is absent ([factory rationale](../src/language_reading_predictors/statistical_models/factories.py#L1933-L1939)).

Age happens to be fitted, but latent general ability remains unresolved. Baseline skill proxies are sensitivity variables, not proof that the latent path has been blocked. The machine-readable observational label is correct and should remain.

### 8. The pooled-versus-period comparison is not report-visible — medium

The comparison script writes shared `dose_response_loo_compare.csv` but does not copy it beside either model ([comparison script](../scripts/compare_statistical_models.py#L1674-L1676)). The report looks locally for a differently named `dose_loo_compare.csv` ([report partial](../docs/models/_partials/_results_dose_response.qmd#L55-L60)). The report therefore omits the formal answer to whether the slope varies by period even after the comparison script succeeds.

The release decision also does not require this comparison, `dose-177`, the B link companion or successful power scaling, although the model recipe/report describes such checks as prerequisites ([release.py](../src/language_reading_predictors/statistical_models/release.py#L144-L149)).

### 9. Metadata and settings validation are incomplete — medium-low

- `effective_adjustment` is populated from prepared covariates, causing dose fits to record `["attend"]` — the exposure — while omitting group, age and baseline terms ([reporting.py](../src/language_reading_predictors/statistical_models/reporting.py#L2766-L2789)).
- Top-level family, design, estimand and causal-status fields are null although the nested resolved run plan is correct.
- Run-plan resolution accepts unknown outcome/covariate symbols and permits an ability symbol to duplicate the own baseline ([dose_response.py](../src/language_reading_predictors/statistical_models/dose_response.py#L295-L333)). The last case creates two coefficients on the identical predictor.

These issues do not corrupt the five current likelihoods, but undermine provenance and the stated pre-I/O validation contract.

### 10. Intercept parameterisation is rank-deficient but the posterior is proper — low-medium

The model fits a grand `alpha` plus all three unconstrained `alpha_phase` indicators ([factories.py](../src/language_reading_predictors/statistical_models/factories.py#L2009-L2015)). The four-column intercept design therefore has rank three. Proper Normal priors make the posterior and predictions proper, so this is not an invalid Bayesian model. The global/phase split is nevertheless prior-identified and can add posterior correlation and sampling cost.

**Required repair.** Use reference coding or zero-sum phase deviations and recalibrate the priors so the implied phase-intercept prior remains intentional.

### 11. Reports do not consistently describe the fitted models — low-medium

- `dose-077`'s displayed equation includes cumulative dose although the registered specification omits it ([report](../docs/models/lrp-rli-dose-077/index.qmd#L41-L49)).
- `dose-277` also claims to retain cumulative-dose control although it does not ([report](../docs/models/lrp-rli-dose-277/index.qmd#L23-L28)).
- The shared partial calls a per-1-SD coefficient “per additional session”, headlines posterior means rather than the house-standard medians and describes the wrong ArviZ comparison field/sign convention ([partial](../docs/models/_partials/_results_dose_response.qmd#L3-L9)).
- ~~All five templates put priors before results, contrary to the findings-first template contract~~ — **withdrawn (2026-08-24, #607).** The templates are correct and the validator was stale: it encoded the #352 order, which #373 deliberately reversed four days later. The same check failed 264 of 264 statistical templates, not five. Fixed in #607; see [the 2026-08-24 note](202608241900-report-template-contract-607.md).
- `dose-083` and `dose-084` contain malformed “Adjusted association” prose; the catalogue omits both and elsewhere incorrectly calls `dose_response` a randomised family.

### 12. Dose scaling can precede the final outcome mask — low

For `dose-177`, 157 rows define the loader scaler before the factory retains the 156 rows with observed W. `dose-077` scales over its final 156 rows. The maximum current discrepancy is 0.0064 SD, so the numerical effect is negligible, but it contradicts the claim that dose is standardised over final fitted rows ([factory subsetting](../src/language_reading_predictors/statistical_models/factories.py#L4384-L4393)).

## Aspects that checked out

- Session attendance is aligned to the following transition interval.
- Count integrality and instrument ceilings are validated; W/L/B denominators are 79/32/10.
- Group recoding, phase indexing, child reindexing and the ordinary-logit natural-scale formula are correct.
- The Beta-Binomial likelihood is internally coherent under its assumptions.
- The convergence gate covers all free random variables, including hierarchical components.
- Missingness and observational-association qualifications are generally explicit.
- Partial pooling across only three phase slopes is a substantive modelling choice rather than a coding error; its prior sensitivity still needs to be interpreted.

## Proposed implementation plan

### Phase 1 — fail closed and correct reader-facing artefacts

1. Withhold the current `dose-177` ability-robustness conclusion, unsupported item-scale marginal, prospective-LOO wording and unpaired `dose-084` headline.
2. Standardise the comparison filename and copy it beside both paired runs.
3. Correct report equations, units, median summaries, comparison schema, report order and catalogue entries.
4. Separate exposure metadata from the effective adjustment set and populate the top-level design/estimand/causal fields.
5. Align model-recipe statements with enforceable release prerequisites.

### Phase 2 — repair the estimands and validation target

1. Use genuine t1 ability covariates in `dose-177`.
2. Separate current treatment from treated attendance intensity and allow arm/history to vary by phase.
3. Add a child-mean/within-child dose decomposition or an explicitly labelled correlated-random-effects sensitivity.
4. Replace the all-row `+1 SD` marginal with a support-respecting, session-calibrated contrast.
5. Replace row LOO with leave-future-out or whole-child validation.
6. Add the `dose-084` guessing-floor response-link companion.

### Phase 3 — repair transformations, parameterisation and validation

1. Route prior and posterior dose marginals through one shared phase-indexed transform.
2. Reject unknown and overlapping symbols before data I/O.
3. Apply the target mask before defining the fitted dose scale.
4. Replace the redundant intercept structure with reference or zero-sum coding.
5. Add arm×phase×dose-band and dose-trend posterior-predictive checks; consider a nonlinear-dose comparator.

### Phase 4 — refit and verify release artefacts

Refit all five registered models and require:

- zero divergences, R-hat ≤ 1.01, bulk/tail ESS ≥ 400 and BFMI ≥ 0.3;
- identical prior/posterior estimand definitions;
- a recorded raw-session contrast and supported averaging population;
- a valid temporal or whole-child predictive comparison;
- the pooled comparison beside both reports;
- the paired B-link sensitivity where applicable;
- complete manifests and release decisions; and
- documented movement from the previous slopes and item-scale headlines.

## Acceptance criteria

- `dose-177` uses only verified t1 ability variables, or is renamed and no longer presented as a baseline sensitivity.
- Treatment presence, intensity, assigned arm/history, within-child and between-child dose have unambiguous coefficient meanings.
- Every reported item-scale contrast remains within its declared arm×phase support, unless an explicitly labelled extrapolation analysis is requested.
- Prior and posterior code paths use the same row population, phase slopes, contrast and item denominator; a synthetic unequal-phase-slope test proves it.
- Predictive validation does not retain a held-out outcome as a future predictor, and its target population is stated accurately.
- `dose-084` cannot release a headline without its response-link companion under the current `METHODS.md` policy.
- Report equations and metadata are derived from the resolved fitted plan rather than manually duplicated prose.
- Repository-template and comparison-copy tests exercise the real dose report files.

## Related work

- Issue #576 identifies the same scalar-versus-period-specific prior-pushforward mismatch in the DiD dose path; the shared writer should be repaired once for both families.
- Issue #269 / PR #284 removed cumulative prior dose from the headline family but did not address `dose-177`'s later-wave L/E/B adjustment.
- Issue #104 / PR #107 introduced `dose-077`, `dose-177` and `dose-277` and described the ability terms as baselines.
- PR #306 added `dose-083` and `dose-084`, including B without a guessing-floor-link companion.
- Issue #381 / PR #484 introduced the generic estimand-scale prior-pushforward path.
- PR #352 established the findings-first report order from which the five dose templates have drifted.

## References

- Mundlak, Y. (1978). On the pooling of time series and cross section data. _Econometrica_, 46(1), 69–85. <https://doi.org/10.2307/1913646>
- Robins, J. M., Hernán, M. A., & Brumback, B. (2000). Marginal structural models and causal inference in epidemiology. _Epidemiology_, 11(5), 550–560. <https://doi.org/10.1097/00001648-200009000-00011>
- Bürkner, P.-C., Gabry, J., & Vehtari, A. (2020). Approximate leave-future-out cross-validation for Bayesian time series models. _Journal of Statistical Computation and Simulation_, 90(14), 2499–2523. <https://doi.org/10.1080/00949655.2020.1783262>
