> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# Mechanism statistical-model audit and proposed fixes, 2026-08-23

## Purpose and relationship to the earlier review

This note records an independent functional and statistical audit of the
`mechanism` family. It follows
`notes/202608201900-mechanism-code-review.md`, confirms several findings from that
review, corrects or strengthens others, and adds findings from a wider examination
of the registered models, stored reporting fits, reporting templates, comparison
artefacts and dormant configuration paths.

This is a findings and remediation note only. It does not change model code or
reported results.

## Scope and verification

The review covered:

- all 41 registered `kind="mechanism"` specifications: 21 linear exposure models
  and 20 HSGP exposure-curve models;
- typed settings and run-plan resolution in `statistical_models/mechanism.py`;
- longitudinal preparation and row selection in `preprocessing.py`;
- `build_mechanism_model` and its transformations, likelihood, priors, random
  effects and moderation terms in `factories.py`;
- the family pipeline, natural-scale summaries, items-scale curves, readiness
  threshold, exact leave-one-out refits and release logic;
- prior inventories, power-scaling sensitivity, key-findings generation, the shared
  Quarto partial and cross-model comparison script;
- the available `reporting` artefacts for all 41 registered models.

Verification performed during the audit:

- all 41 specifications resolved, loaded the current data and built PyMC graphs;
- representative model initial log-probabilities were finite;
- 304 focused tests passed in the broad audit run, with additional specialised
  factory, report, prior, readiness and leave-one-out runs also passing;
- focused Ruff checks passed;
- every available stored reporting fit passed its computational gate: zero
  divergences, maximum R-hat 1.0042, minimum effective sample size 1,886 and
  minimum chain BFMI 0.663;
- model-frame claims for mech-058, 063, 071, 101, 158, 191, 200 and companions were
  reconstructed from the current data.

The full reporting samplers were not rerun. Stored reporting traces were inspected
alongside fresh run-plan, loader and graph builds.

## Overall judgement

For the active specifications, the likelihood and graph construction are internally
coherent. The Beta-Binomial parameterisation, Haldane count-logit transforms,
standardisation, child random intercept, linear and HSGP exposure terms, moderation
algebra, active exact-LOO design replay and convergence checks are implemented
consistently.

The family is nevertheless not reporting-ready without qualification. The most
important problems are downstream of graph construction: the readiness criterion
does not establish a knee, mech-191 fits a different dose population from the one it
documents, avoidable complete-casing changes some fitted samples, prior figures and
shared report prose misdescribe most models, and one comparison forest bypasses the
convergence gate. The fitted skill coefficients and curves are contemporaneous
conditional associations, not identified causal mechanisms.

## Findings and proposed fixes

### 1. Major: the readiness criterion does not establish that a knee exists

`reporting._readiness_knee` bins the exposure, finds the between-bin interval with
the largest derivative of `f_mech`, and calls the resulting knee well defined when
more than 90% of draws rise from the first to the last bin. That condition tests a
positive net association, not a local threshold. A perfectly linear increasing curve
has no knee, but all of its draws pass the increasing test and `argmax` still selects
one arbitrary interval.

The scale is also mislabelled. `f_mech` is a contribution on the outcome-logit scale,
whereas the report says the algorithm finds where outcome items rise fastest. On the
items scale the derivative additionally contains the inverse-link factor
`p * (1 - p)` and can peak at a different exposure value depending on the reference
baseline and covariates.

**Proposed fix.** Separate three claims:

1. Always permit a descriptively labelled "steepest latent-logit interval".
2. Call it a knee only when posterior draws support a local slope change or curvature,
   the winning interval is identifiable rather than tied, and a nonlinear model has
   adequate predictive support relative to its linear anchor.
3. If the scientific target is where expected outcome items rise fastest, compute
   that derivative after the inverse link under the declared population
   standardisation.

Report the posterior probability of an increasing curve, interval-selection
stability, local slope contrast and GP-versus-linear comparison separately. Add a
unit test showing that a perfectly linear increasing curve is not classified as a
well-defined knee, and an inverse-link test where the items-scale maximum differs
from the latent-logit maximum.

### 2. Major: mech-191 does not fit its documented on-intervention population

`lrp_rli_mech_191.py` and its report say 55 missing attendance values are dropped,
leaving exactly the on-intervention rows carrying a session dose. Attendance is read
from a transition's pre row. Of the 55 missing values, 54 are t4 cells that cannot
enter the three-transition frame and one is a relevant t2 cell.

The actual fitted frame contains 156 rows from 53 children and 28 zero-session rows.
In phase 1, all 25 fitted waitlist rows are at zero sessions and no immediate-arm row
is at zero; only seven fitted observations lie between 1 and 30 sessions. The low end
of the curve is therefore strongly arm- and period-structured, with sparse positive
low-dose support. An arm covariate does not create overlap where the exposure support
is structurally different.

**Proposed fix.** Make an explicit population decision before changing code:

- Recommended minimal correction: retain the established all-transition,
  zero-dose-anchored association, correct the module and report, publish support by
  phase and arm, use the existing interquartile exposure contrast as the headline,
  and retain 0-to-94 sessions only as a full-range secondary summary.
- Alternative scientific target: add an explicit on-intervention restriction and
  refit, accepting that this estimates an association among treated periods and no
  longer uses the randomized zero-dose anchor.
- Stronger redesign: fit a two-part exposure model separating any intervention from
  the positive-dose curve, or fit phase/arm-specific curves where support permits.

Whichever target is chosen, do not interpret the current knee as a causal or
intensive-margin session threshold. Add a data-backed population-contract test that
pins retained zero-dose rows and a report test forbidding the present
"exactly on-intervention" claim.

### 3. Major: mechanism prior reports show the wrong lengthscale distribution

The prior-figure key maps every `*__ell` random variable to the generic `ell`
constructor, which is `InverseGamma(3, 1)`. Mechanism HSGPs actually use
`InverseGamma(5, 5)` or, for 16 registered tight fits, `InverseGamma(8, 8)`.

Consequences:

- every one of the 20 HSGP mechanism reports displays the wrong lengthscale density;
- `prior_artifacts.py` hard-codes an `InverseGamma(5, 5)` rationale, so the 16 tight
  fits can have a correct distribution column but false rationale and figure;
- `_results_mechanism.qmd` calls the fit an HSGP and hard-codes the default prior for
  all 41 reports, even though 21 fits are linear and only four HSGP fits use the
  stated default.

The same shared paragraph asserts the association's sign and existence are robust
without consulting the posterior or sensitivity artefacts.

**Proposed fix.** Give mechanism lengthscales prior keys that preserve the actual
constructor, and generate the rationale and density from the resolved run plan or
prior inventory rather than parameter-name heuristics. Make the shared report prose
conditional on linear versus HSGP form and default versus tight lengthscale. Remove
the unconditional robustness statement.

Add prior-inventory and report tests for a linear model, default HSGP, tight HSGP and
continuous-covariate HSGP. Each test should reconcile the PyMC random variable,
`priors_table.csv`, rationale, density-panel constructor and rendered prose.

### 4. Major: preprocessing complete-cases unused pre-scores

The run plan sets `pre_required` to every measure in `settings.outcomes`, and the
loader drops rows missing any of those pre-scores. Active mechanism fits use only the
focal outcome's pre-score in the linear predictor; bounded exposures, measure
confounders and moderators are contemporaneous post measurements.

For mech-063/163, four otherwise eligible rows are removed solely because `N_pre` is
missing, although `N_post` and every fitted term are observed. This is avoidable loss
of information and selection on a measurement absent from the model.

**Proposed fix.** Set `pre_required=(adjust_baseline_symbol,)` for the default
post-exposure design. Add the exposure only when `mechanism_at_pre=True`, and add any
other pre-row term only when the factory actually consumes it. Add row-identity tests
for mech-063/163 and a synthetic missing-unused-pre-score case. Refit every registered
model whose fitted rows change and regenerate downstream comparisons.

### 5. Major: mech-158 is not an isolated missing-data comparator to mech-058

mech-158 says it is identical to mech-058 except for requiring observed confounders.
It instead omits mech-058's `outcomes=("W", "L")`, six-basis HSGP and tight
`InverseGamma(8, 8)` lengthscale, so it defaults to all-eight pre-score requirements,
ten bases and `InverseGamma(5, 5)`. Its prose also says phonological memory is an
adjuster and complete-case requirement although neither comparator fits `erbto`.

Fresh builds produced 156 rows/53 children for mech-058 and 128/44 for mech-158. The
population reduction is intended; the functional-form and loading-contract changes
are not. Even a correctly matched complete-case comparison cannot prove that
imputation "drives" a difference, because complete-casing selects a different
population in which the association can differ.

**Proposed fix.** Add `outcomes=("W", "L")`, `adjust_baseline_symbol="W"`,
`mech_hsgp_m=6` and `mech_lengthscale_tight=True` to mech-158; remove the false
`erbto` text; then refit. Add a paired-contract test comparing resolved mech-058 and
mech-158 plans after allowing only the declared missing-data restriction and its
resulting constant-indicator differences.

### 6. Major: the mechanism forest can publish failed fits without marking them

`compare_statistical_models.mechanism_forest` loads every available trace without
calling the script's gate helper and writes no convergence field. This contradicts
the comparison script's explicit rule that failed or review-status fits must not
enter an ordinary forest unmarked. The currently stored mech-056/057/058 traces all
pass, so the present forest is not known to be wrong, but the implementation fails
open for future fits.

The same forest uses posterior means rather than the median-first house convention.
For HSGP curves it drops duplicate exposure values and equally averages derivatives
over the remaining irregular unique grid. That is neither a fitted-row average nor
an exposure-interval-weighted average.

**Proposed fix.** Require a passed gate for every included trace, fail the complete
forest closed or visibly mark omissions, and write gate status to the CSV. Use the
posterior median. Define the nonlinear slope target before changing its weighting:
for example, a fitted-row average derivative, an integral over a declared reference
distribution, or a secant over a prespecified quantile interval. Pin the choice with
an irregular-grid test.

### 7. Moderate: interaction key findings can confirm an inconclusive interaction

The moderation key-findings builder checks whether items-scale evidence is settled
before checking whether `gamma_int` is directionally inconclusive. For example,
`P(gamma_int > 0) = 0.55` together with an items-scale probability of 0.97 produces
"strong evidence that the synergy holds ... not an artefact", despite the fitted
logit interaction being inconclusive.

**Proposed fix.** Require the fitted interaction to reach the declared directional
evidence threshold before items-scale results can confirm its interpretation. If it
does not, describe both scales without synergy or substitution language. Add the
missing inconclusive-logit/settled-items test.

### 8. Moderate: headline and reader-facing curves use different estimands

`mechanism_summary.csv` compares the observed minimum and maximum exposure while
averaging probabilities over fitted rows, retaining their phase, covariates,
baselines and fitted child intercepts. The items-scale curve instead uses an
interquartile worked contrast, removes the child intercept, averages the remaining
linear predictor and then applies the inverse link for a typical child.

Because the inverse-logit is nonlinear, averaging probabilities is not the same as
applying the inverse link to an average linear predictor. The two artefacts are valid
descriptive quantities but are not "the same curve" as the report states. Observed
extremes are also unstable anchors and are especially problematic for mech-191.

**Proposed fix.** Declare one headline population-standardised estimand and one
prespecified exposure interval, preferably an interquartile or scientifically chosen
range. Use it consistently in the table, plot, key findings and cross-model summary;
label any typical-child curve or full-range contrast as a secondary estimand.

### 9. Moderate: fitted-adjustment metadata omits moderators

The factory fits a moderator main effect and, where requested, an exposure-by-
moderator interaction. `effective_adjustment` records neither. For mech-073, age is
therefore listed as requested but absent from the fitted adjustment terms although
`gamma_mod` is its fitted age adjustment.

**Proposed fix.** Extend the fitted-adjustment schema with moderator main-effect and
interaction roles, including source, timing and scale. Add tests for age moderation,
measure moderation with an interaction, and the main-effect-only companion.

### 10. Moderate: the model is contemporaneous and blends within- and between-child associations

All active bounded-skill and state-covariate exposures are measured at the same wave
as their outcomes; the attendance exposure is an interval quantity read from the
transition's pre row. Conditioning the outcome on its previous score does not make a
same-wave exposure temporally prior and does not remove reverse or synchronised
development.

The child random intercept models repeated residual dependence under an independence
assumption. It does not block `exposure <- latent general ability -> outcome`, allow
the child intercept to correlate with exposure, or distinguish
`child mean exposure` from `within-child deviation`. The fitted coefficient or curve
therefore mixes between-child and within-child information. Reports that call the
random intercept a proxy for latent ability, or call descendant-moderated quantities
"controlled direct effects", are statistically incorrect.

Negative-control results support this caution. Letter-sound knowledge is positively
associated with several non-reading outcomes; most remain positive after adding the
available ability proxy. The ability-adjusted linear L-to-F coefficient, 0.249 logit
units per exposure SD (89% interval 0.121 to 0.379), is almost identical to the
linear L-to-W anchor, 0.248 (0.151 to 0.345). This does not disprove a reading pathway,
but it is consistent with broad developmental ability or shared measurement timing
rather than a reading-specific mechanism.

**Proposed fix.** Keep all current results explicitly associational. Replace
"earlier skill", "controlled direct effect", causal "effect modification" and
random-intercept-as-ability language. For a within-child question, add a Mundlak
between/within decomposition or a child-fixed-effect linear sensitivity, with a
clear change in estimand. Treat ability-proxy variants as residual-confounding
sensitivities, not backdoor closure.

### 11. Moderate: one common association is pooled across incompatible phases

Active models give each phase an intercept but use one common exposure slope or
curve and one original-arm coefficient across the randomized phase and two
post-crossover phases. This assumes the relationship and residual group contrast are
stable across substantively different treatment histories. Exposure support also
changes with phase, allowing omitted group-by-phase or history structure to be
absorbed by the exposure curve.

No registered model uses the phase-specific option, and some reports advertise a
linear phase-specific sensitivity that settings validation expressly rejects.

**Proposed fix.** First publish exposure support by phase and arm. Add a modest
phase-by-linear-exposure sensitivity where support is adequate, or stratify summaries
without claiming a common curve. Do not enable the current phase-specific HSGP flag
until finding 14 below is fixed. Correct the reports immediately.

### 12. Moderate: nonlinear shape and dispersion are more prior-dependent than release conveys

Every stored active HSGP fit has its focal amplitude or lengthscale flagged by power
scaling for possible prior-data conflict or weak likelihood information. Mechanism
fits are excluded from the treatment-effect robustness gate because they are
observational, so this prior sensitivity does not prevent release. Identification
status and prior robustness are separate questions: an observational knee can still
be too regularisation-dependent to interpret.

The shared `kappa ~ HalfNormal(50)` prior also strongly commits high-denominator
outcomes to residual overdispersion. For a Beta-Binomial outcome with denominator
`N`, the variance-inflation factor over a Binomial is
`1 + (N - 1) / (1 + kappa)`. Being within 10% of Binomial variance requires
`kappa >= 10 * (N - 1) - 1`, which is effectively outside this prior for the larger
tests. This is a substantive prior assumption, not a neutral weak prior.

**Proposed fix.** Make focal HSGP power sensitivity a release qualification for
shape and knee claims even though the association is non-causal. Compare the tight
and default lengthscales and linear anchor on the same estimand. Calibrate the
dispersion prior through prior predictive checks and add a near-Binomial-capable
dispersion-scale sensitivity, following the approach adopted in newer families.

### 13. Minor: two valid LOO comparisons do not reach their model reports

The shared report reads comparison CSVs beside each model run. The mech-058/071 and
mech-072/172 wrappers write only to the central comparison directory, unlike newer
wrappers that copy the result beside both runs. The 058-versus-071 comparison also
adds both the moderator main effect and its interaction, so it tests their joint
addition rather than isolating interaction.

**Proposed fix.** Copy each comparison beside both participating runs and test the
contract. Describe 058-versus-071 as a joint moderator-plus-interaction comparison;
use the coefficient posterior or introduce a main-effect-only companion if an
interaction-only predictive comparison is required.

## Dormant configuration defects

The following paths do not affect a currently registered model but should be fixed
before they are used:

1. **Overlapping focal roles.** Resolution permits outcome, exposure and moderator to
   coincide, and permits a bounded exposure to enter the adjustment set too. This can
   feed the outcome back as a predictor or create duplicate/algebraically identical
   columns. Reject these combinations before data loading.
2. **Phase-specific HSGP artefacts.** The factory builds per-phase curves, but the
   pipeline mixes them into one endpoint curve, items curve and knee; power scaling
   asks for non-existent global hyperparameter names. Either reject the option until
   supported or emit fully phase-specific artefacts and sensitivity variables.
3. **Age GP with age moderation.** This fits an HSGP age effect and a separate linear
   moderator main effect, and age-GP exact LOO does not freeze its boundary. Reject
   the combination or orthogonalise it, freeze the boundary, and add the age-GP
   hyperparameters to diagnostics and power scaling.
4. **Default outcomes.** Required-measure coverage validation is skipped when
   `outcomes=None`; a non-default bounded exposure can fail only after context and
   data work. Resolve a concrete effective outcome set and validate it unconditionally.
5. **Ability covariate.** Invalid types and unknown names are accepted until pandas
   access. Validate the non-empty string and supported baseline-covariate schema in
   the pure run-plan stage.

The dormant `mechanism_at_pre` path was specifically tested and appears internally
consistent.

## Recommended implementation sequence

### Batch A: fail-closed reporting corrections; no scientific refit required

1. Correct the mechanism prior constructor/rationale/panel mapping.
2. Make the shared report conditional on functional form, actual prior, exposure
   type, available artefacts and posterior evidence.
3. Fix interaction key-findings ordering.
4. Gate the mechanism forest, use medians and define its nonlinear slope estimand.
5. Add moderator terms to `effective_adjustment`.
6. Copy comparison CSVs beside participating runs.
7. Correct causal, temporal, random-intercept and phase-sensitivity prose.

Regenerate prior artefacts, key findings, comparisons and reports from stored traces
where the fitted estimand is unchanged.

### Batch B: sample-contract changes; refits required

1. Restrict `pre_required` to terms actually used at pre and enumerate every model
   whose row set changes.
2. Make mech-158 structurally identical to mech-058 except for complete-casing.
3. Decide the mech-191 population and implement the chosen restriction or two-part
   design if the current zero-anchored fit is not retained.
4. Refit affected models and rebuild nested comparisons and reports.

### Batch C: estimand and sensitivity improvements

1. Replace or relabel the readiness statistic and implement the required curvature
   and scale checks.
2. Declare a single population-standardised natural-scale contrast.
3. Add within/between and phase-stability sensitivities.
4. Add HSGP prior and dispersion-scale sensitivities, and connect focal robustness to
   readiness-result release.
5. Fail dormant unsupported configurations during pure resolution until their full
   artefact and exact-LOO contracts exist.

## Acceptance criteria

The mechanism family should be considered repaired when:

- every registered specification's report states its actual functional form, prior,
  sample, exposure timing and standardisation;
- a linear increasing curve cannot be reported as a well-defined knee;
- mech-191's population and dose support match its scientific description;
- mech-058/158 differ only by the declared missing-data policy;
- no row is dropped for an unused pre-score;
- all comparison artefacts fail closed on convergence and record their estimands;
- fitted-adjustment metadata names every coefficient-bearing conditioning term;
- nonlinear shape claims are qualified or withheld when focal prior sensitivity is
  unresolved;
- unsupported role overlaps and dormant configurations fail before context creation
  or data loading;
- affected models are refitted and the regenerated artefacts pass the standard
  convergence, completeness and report checks.

## Status

Documented only. No model code, data or fitted artefact was changed in this audit.
