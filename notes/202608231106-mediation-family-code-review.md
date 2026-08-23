> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# Mediation and mediation-multi code review, 2026-08-23

## Decision summary

This review found no simple arithmetic error in the implemented posterior simulation. The Beta-Binomial, Bernoulli and Gaussian draws are internally consistent with the fitted models; treatment-by-mediator interactions are carried through the response-scale simulation; the sign convention is consistent; and the reported total effect equals the implemented direct plus indirect effect by construction.

The larger problem is that several fitted models and counterfactual simulations do not implement the adjustment set, estimand or analysis population declared by their specifications and reports. Five findings are release-blocking:

1. The common baseline covariate set is split between the mediator and outcome regressions rather than conditioned on in both.
2. The three models labelled `interventional` run the same algorithm as the corresponding natural-effect models and do not address the exposure-induced mediator-outcome confounder that motivated them.
3. `LRP-RLI-MED-060` silently drops two declared confounders.
4. Off-floor models exclude children for baseline measurements that their likelihoods never use.
5. The all-period headline for `LRP-RLI-MED-092` toggles treatment in periods with no untreated observations.

**Recommendation:** withhold and invalidate the current artefacts for all 19 registered models, correct findings 1–5, and refit every retained specification whose likelihood or fitted sample changes. Findings 6–8 should be resolved in the same remediation because they affect the meaning and evidential checks of the reported decompositions. The existing reports are appropriately cautious that natural effects are not identified under the remaining unmeasured-confounding and timing assumptions; that caution does not repair a mismatch between the declared and fitted models.

This pull request records findings and proposed fixes only. It does not change a model, regenerate an artefact or endorse a numerical mediation result.

## Scope and verification

The review traced every registered `mediation` and `mediation_multi` specification through settings resolution, data preparation, PyMC construction, posterior counterfactual simulation, sensitivity analysis, diagnostics, metadata, release logic and report prose. The registered models are:

- Single mediator: `LRP-RLI-MED-059`, `062`, `068`, `074`, `076`, `078`, `079`, `080`, `086`, `087`, `092`, `176`, `186`, `187` and `276`.
- Two mediators: `LRP-RLI-MED-060`, `064`, `066` and `075`.

All 19 specifications resolve and all 19 PyMC models build on the current data. The broad relevant test selection passed (534 tests) and Ruff checks were clean. No full mediation NUTS fit was rerun, so this is a code, model-contract and statistical estimand review rather than a re-analysis of stored posterior results.

## Finding 1: the declared common baseline set is not fitted as a common set

### Evidence

The run-plan resolvers treat `W_pre` and the mediator baseline marker as structural terms and remove them from the confounder set (`mediation_settings.py:534-540` and `614-620`). The single-mediator factory then places the mediator's own baseline only in the mediator regression and the outcome's own baseline only in the outcome regression (`factories.py:2737-2784` and `2589-2620`). The two-mediator factory follows the same pattern (`factories.py:3041-3186`). The g-formula repeats that split rather than integrating both legs over a shared baseline vector (`mediation.py:235-339` and `636-800`).

This is not merely a naming issue. The documented mediation g-formula conditions the mediator law and the outcome law on an explicitly defined common pre-exposure covariate set, conventionally written as `C`. Reduced leg-specific sets can be valid when the omitted covariates are conditionally irrelevant, but that case must be justified model by model. Here, the specification and lagged DAG use cross-lagged and autoregressive relationships (`dag/dag-language-reading-lagged.dagitty:88-131`), while the implementation supplies each regression only selected own-baseline terms. The DAG tests validate the global declared adjustment set; they do not validate the two leg-specific design matrices (`tests/test_lagged_dag_adjustment_sets.py:362-415`).

The concrete affected groups are:

- `MED-059`, `062`, `068`, `074`, `076`, `078`, `080` and `092`: baseline word reading is absent from the mediator law despite the DAG's lagged word-reading-to-skill paths.
- `MED-086`, `186`, `087` and `187`: the letter-sound baseline is absent from the nonword or phonological-awareness outcome law despite its measured lagged paths. These specifications include a separate bare `W` term in both legs, so they are not examples of the first omission.
- `MED-176` and `276`: the reverse-direction models likewise fit different baseline vectors in the two legs and require a reverse-DAG-specific derivation rather than inheriting the forward marker-stripping rule.
- `MED-079`: the direct word-reading-to-mediator edge is deliberately severed as a negative control, but its advertised same-`C` law is still not fitted; the DAG retains baseline association through receptive grammar. This is a weaker and more model-dependent case than the directed omissions above.
- `MED-060`, `064`, `066` and `075`: neither mediator law receives baseline word reading, the outcome law receives neither mediator baseline, and the second-mediator law does not receive the first mediator's baseline. The chain term using post-`L` does not generally replace baseline `L`.

Consequently, every registered model has an unresolved fitted-versus-declared baseline contract, with especially direct DAG omissions in 18 of 19. If different sufficient sets were intended for the two regressions, those sets and their conditional-independence justification are not currently declared or tested.

### Proposed fix

1. Represent the mediator-leg and outcome-leg adjustment sets explicitly in the typed run plan. Include a separately named common baseline vector and distinguish own-lag terms from confounders.
2. Make the loader, factory and simulator consume those resolved leg-specific sets; do not reconstruct them independently.
3. Validate that every declared bounded measure is loaded and appears in the intended design matrix. Record declared, loaded and fitted sets for each leg in `config.json` and `model_recipe.md`.
4. Add design-matrix tests derived from the DAG, including a test that removing a required cross-baseline changes both the resolved plan and built graph.
5. Invalidate all 19 existing artefacts. Refit every retained model whose resolved leg sets change; existing posterior artefacts cannot be repaired by regenerating summaries because the likelihood changes.

## Finding 2: the `interventional` variants are algorithmic aliases

### Evidence

`LRP-RLI-MED-078`, `186` and `187` set `estimand="interventional"`, but the flag changes labels only (`mediation.py:198-212` and `345-359`). The counterfactual draws and contrasts are the same as for the natural-effect path (`mediation.py:271-339`). The tests require exact equality between natural and interventional output (`tests/statistical_models/test_mediation.py:308-317`).

Exact equality is possible for a stochastic analogue under a model with no exposure-induced mediator-outcome confounder. It is not evidence that such a confounder has been handled. These variants were introduced because intervention sessions (`IS`) are treatment-affected and can affect both mediator and outcome. The current model neither includes `IS` in an extended g-formula nor defines an intervention that integrates over its treatment-specific distribution. It therefore retains the same substantive vulnerability as the natural-effect fit.

### Proposed fix

Choose and document one of two paths:

- Implement a precisely defined interventional direct and indirect effect that models and integrates the exposure-induced confounder, with its counterfactual ordering, support conditions and identification assumptions made explicit. State which treatment-specific distribution of `IS` enters each counterfactual cell, whether the components are intended to sum to the ordinary total effect, and which interventional decomposition is being used.
- Relabel the current output as a stochastic plug-in analogue under the same fitted model and state that it is numerically identical and does not address confounding by `IS`.

Do not retain a separate model ID whose only difference is a different unsupported causal label. Replace the exact-equality test with tests of the chosen estimand's algebra and with a synthetic example where exposure-induced confounding makes the interventional and natural targets differ. Relevant methods are VanderWeele, Vansteelandt and Robins (2014), DOI [`10.1097/EDE.0000000000000034`](https://doi.org/10.1097/EDE.0000000000000034), and Vansteelandt and Daniel (2017), DOI [`10.1097/EDE.0000000000000596`](https://doi.org/10.1097/EDE.0000000000000596).

## Finding 3: `LRP-RLI-MED-060` silently omits declared confounders

### Evidence

The model specification declares expressive and receptive vocabulary (`E` and `R`) in its adjustment set but requests only `W`, `L` and `N` as outcomes (`lrp_rli_med_060.py:54-63`). The multi-mediator resolver checks only that the outcome and two mediators are present (`mediation_settings.py:609-613`); it does not require bounded-measure confounders to be loaded. The pipeline then filters the declared set to variables that happened to be prepared (`pipelines/mediation.py:570-584`). Thus `E` and `R` disappear before model construction rather than causing an error.

The fitted effective set is `hs`, `hs_missing`, `deapp_c`, `deapp_c_missing` and `erbto`; `E` and `R` are absent. The constant `erbto_missing` indicator is also absent, as intended. The metadata's `dropped_confounders` calculation compares only raw covariates (`pipelines/mediation.py:700-709`), so it does not expose the two omitted measure confounders.

### Proposed fix

Have the resolver add every declared bounded-measure confounder to the load set, or fail before data are loaded if a required measure is absent. After preparation, require equality between the resolved and fitted leg-specific sets except for a separately recorded, explicitly permitted constant-column removal. Compute `dropped_confounders` from the complete declared set. Add a regression test for `MED-060` that inspects the built coefficient names for both `E` and `R`.

## Finding 4: off-floor sample membership is not represented in the likelihood

### Evidence

The run plans do not set `pre_required` (`mediation_settings.py:277-296` and `391-398`). The shared loader therefore requires the baseline of every requested outcome by default (`preprocessing.py:665-693`). The off-floor outcome factory deliberately drops the graded own-baseline term (`factories.py:2589-2606`), and the off-floor second-mediator path likewise does not use its graded baseline. The selection rule is therefore stricter than the fitted likelihood.

On the current data, `LRP-RLI-MED-060`, `086` and `186` retain 50 children under the present rule but the present likelihood would retain 53 if it required only the baselines it actually uses. This comparison diagnoses a model/sample-contract mismatch; it does not by itself establish that the three children belong in the corrected sample. A binary off-floor-at-baseline predictor is scientifically usable and is already the project's convention for other floor models (`priors.py:132-152`). The corrected mediation model must first decide whether baseline nonword off-floor status belongs in the mediator or outcome law and how its missingness is handled. For multi-mediator models without an explicit outcome list, the default eight-outcome load set also makes membership depend on unrelated baselines, although it causes no further exclusion in the present data. The lagged sensitivity route must forward the resolved requirement rather than reintroducing the default.

### Proposed fix

First define the corrected baseline terms for each leg, including whether a binary off-floor baseline belongs in the model. Then resolve `pre_required` from that corrected contract and pass it through every preparation path. Persist a row-flow table naming each exclusion rule and a digest of the fitted child IDs. Add tests that perturb a genuinely unused baseline and confirm that fitted-row membership does not change, and tests that a modelled binary baseline is both required and present in the built graph. Refit every model whose likelihood or sample changes; summary regeneration is insufficient.

## Finding 5: the all-period `MED-092` contrast lacks positivity

### Evidence

The period-stacked treatment indicator is `T = (G == 1) | (phase >= 1)` (`factories.py:3348`). In the first study period (`phase == 0`), randomisation provides treated and untreated children. In later periods every child is treated. Nevertheless, `decompose_period_stacked` sets `T=0` and `T=1` for every row and averages those contrasts over all periods (`mediation.py:508-585`).

The later-period untreated counterfactual is therefore estimated entirely by model extrapolation: there are no later-period observations with `T=0` from which to estimate it. Adjustment does not create positivity. The all-period headline cannot be read as a supported causal mediation decomposition.

### Proposed fix

Make the first-period randomised contrast the primary estimand. Retain an all-period standardised contrast only as an explicitly model-dependent extrapolation, with a support table showing treated and untreated counts by period; alternatively remove it. Add a resolver or release check that refuses a causal headline when any standardisation cell has zero empirical support.

## Finding 6: the sensitivity analysis is a coefficient tipping grid

### Evidence

The single- and two-mediator sensitivity functions choose a global direction from the posterior-mean treated-arm mediator slope and shift the fitted mediator-to-outcome coefficient over a fixed grid while leaving the mediator law, treatment interaction and all other likelihood terms unchanged (`mediation.py:368-467` and `885-1005`). This answers the narrow question “how large a one-direction coefficient shift makes the selected credible interval first include zero?” It does not introduce an unmeasured variable, residual correlation, partial R-squared or risk-ratio association with both mediator and outcome.

The global direction moves the posterior-mean slope toward zero, but any individual posterior draw with the opposite sign is moved away from zero. The tipping point is grid-dependent and based on interval inclusion, not on a causal indirect effect being exactly null. Calling the result an E-value or a general robustness value would therefore overstate what it measures. The report prose currently gives the grid a broader unmeasured-confounding interpretation (`pipelines/mediation.py:261-294` and `docs/models/_partials/_results_mediation.qmd:159-216`).

The named-`IS` calibration does not convert the grid into a formal confounding analysis. The single-mediator calibration uses `delta_IS = |beta_IS_to_M * beta_IS_to_Y|` (`mediation_calibration.py:12-22` and `346-374`), which is not the general omitted-variable-bias formula for a mediator coefficient and is not scale-invariant. Its units do not generally match the shift applied to `b_M`. The multi-mediator treated-arm coefficient-change calibration is a different, explicitly descriptive construction; the two should not share a stronger interpretation.

### Proposed fix

Immediately rename the artefact and prose to “mediator-coefficient tipping analysis”, define its one-sided perturbation, distinguish the two `IS` calibrations, and avoid conclusions that `IS` “could account for” an effect without a derivation or simulation validation. For a substantive sensitivity analysis, specify a coherent bias model: for example, correlated mediator/outcome residuals where appropriate, calibrated partial-R-squared parameters, or an estimand-scale bias function. An E-value is a particular risk-ratio sensitivity measure, not a generic name for a coefficient grid; see VanderWeele and Ding (2017), DOI [`10.7326/M16-2607`](https://doi.org/10.7326/M16-2607), and the mediation-specific Smith and VanderWeele (2019), DOI [`10.1097/EDE.0000000000001064`](https://doi.org/10.1097/EDE.0000000000001064).

## Finding 7: the multi-mediator joint law relies on strong assumptions

### Evidence

For the parallel models, the two mediator likelihoods are conditionally independent given their predictors (`factories.py:3111-3163`), and the simulator draws from that product distribution (`mediation.py:636-784`). With a nonlinear outcome model, the joint distribution of the mediators—not only their marginal means—affects the response-scale expectation. The resulting quantity is a product-of-marginals stochastic intervention unless conditional independence is true; it is not automatically an effect through the observed joint mediator distribution. The latent general-ability common cause makes the independence assumption especially strong, and an empirical residual-correlation check cannot establish it when that cause is unmeasured.

For chain models, the same-world joint draws are internally coherent for the implemented ordering. However, the two reported per-leg quantities are a sequential, ordering-dependent allocation (`mediation.py:753-800`), not two order-invariant natural indirect effects. The second allocation includes pathways through the first mediator when the structural order permits them. The current names invite a simpler mediator-specific interpretation than the algebra supports.

### Proposed fix

State the factorisation of the joint mediator law in the run plan and report. For parallel mediators, model residual dependence, add a dependence sensitivity or explicitly rename the quantity as a product-of-marginals stochastic intervention; consider the mediator-dependence remainder in the Vansteelandt-Daniel decomposition. For chains, call the component quantities sequential path allocations, reject a non-topological order, and treat alternative telescoping allocations as allocation-order sensitivity—not as evidence for an opposite causal chain. The joint indirect effect can be primary only after its joint-law target is named and defended. A genuinely path-specific effect requires separately defended identification and coupling assumptions. Vansteelandt and Daniel (2017), DOI [`10.1097/EDE.0000000000000596`](https://doi.org/10.1097/EDE.0000000000000596), discusses interventional effects with multiple mediators.

## Finding 8: diagnostics do not cover the full mediation computation

### Evidence

Posterior predictions are sampled for every observation node, but the shared stage sends only the last node to the PPC writer (`stages.py:165-176`), and `ppc_artifacts.save_ppc` writes diagnostics for that one primary node (`ppc_artifacts.py:70-90`). Mediator fit is therefore not represented in the released coverage summary or calibration figure.

The code calculates effective sample sizes for derived g-formula quantities (`mediation.py:125-136` and `reporting.py:152-193`), but release gating is based on sampled model variables, not the PNDE, TNIE, total effect and mediated proportion. Each posterior draw also uses only 50 inner counterfactual replicates; there is no stored simulation seed or repeatability/stability check separating posterior uncertainty from inner Monte Carlo error.

Finally, both mediation plans hard-code `compute_loo=False` (`mediation_settings.py:576-577` and `644-645`). A mediation effect is not validated by predictive fit, but the phase-0 fits have ordinary mediator and outcome likelihood contributions that can be summed by child for a useful joint log score. `MED-092` is different: it has repeated rows and fitted child random effects, so a genuine leave-one-child-out score must integrate or refit the held-out child's effect; merely summing its current conditional likelihood nodes would answer a within-child prediction question. Omitting predictive validation should be an explicit decision, not a claim that mediation models have no meaningful predictive unit. The temporal-ordering robustness requirement also applies only to natural single-mediator fits, leaving other contemporaneous mediation headlines without an equivalent gate.

### Proposed fix

1. Write per-node PPC coverage and calibration for every mediator and outcome likelihood, using the correct denominator and binary/count treatment. Pre-specify the diagnostic, tolerance and sampling uncertainty before making a poor check an automatic release veto; in-sample posterior-predictive fit is not a causal-validity test.
2. Gate release on R-hat, effective sample size and Monte Carlo standard error for the derived decomposition draws as well as the free parameters.
3. Make the inner simulation seed and replicate count part of the resolved plan; compare repeated seeds or increase or integrate the inner simulation until Monte Carlo error is negligible relative to posterior uncertainty.
4. For phase-0 fits, compute optional child-level joint PSIS-LOO by summing pointwise log likelihoods across the mediator and outcome nodes. Give `MED-092` a separately defined marginal leave-child-out calculation that handles its random effect. Use either score for model criticism or comparison, never as evidence that a causal effect is identified.
5. Apply an equivalent temporal-ordering robustness rule to every contemporaneous mediation headline, or explain and record why a particular estimand is exempt.

## Smaller correctness and reporting defects

### Lagged fits record the loader row count rather than the fitted row count

The single-mediator pipeline writes `prepared.n_obs` (`pipelines/mediation.py:349-363`) even when the factory applies an additional outcome keep-mask. On the current data, `MED-076` loads 53 rows and fits 51; `MED-176` loads 53 and fits 52; `MED-276` happens to fit all 53. Metadata should use `built.prepared.n_obs`, plus a fitted-row identity digest.

### One settings combination is accepted but not implemented

The settings permit `mediator_kind="gaussian_composite"` together with `outcome_kind="bernoulli_offfloor"` (`mediation_settings.py:111-148`), but the route-composite factory always builds the graded outcome path. No registered model uses the combination. Reject it in resolution until the factory and simulator support it, so a future declaration cannot be silently misfit.

### Multi-mediator proportion summaries lack the single-mediator zero guard

The single-mediator summary guards the all-nonfinite or exact-zero-total case (`mediation.py:140-175`); the multi-mediator ratios do not (`mediation.py:838-859`). Reuse that guard so an empty ratio array cannot crash the decomposition. Separately define and justify any practical near-zero threshold before suppressing an unstable mediated proportion.

### `NDE` and `NIE` understate which natural effects are computed

With treatment-by-mediator interaction, the code's direct contrast fixes the mediator to its untreated distribution (the pure natural direct effect), while the indirect contrast holds treatment at one (the total natural indirect effect). The alternative total-direct plus pure-indirect decomposition differs. Name the quantities `PNDE` and `TNIE`, or define the counterfactual contrasts directly in the table and report.

### The proportion row's probability column has a different meaning

For a mediated-proportion row, the current generic `prob_pos` value refers to the sign of the total effect rather than the sign of the ratio. Either calculate `P(proportion > 0)` with the zero-total guard or rename the column for that row.

## What is correct and should be preserved

- Beta-Binomial mediator and outcome predictions draw a latent probability and then a bounded count; the Gaussian composite uses the fitted Normal law.
- Counterfactual outcome means use the inverse-logit transformation and the correct item denominator.
- Treatment-by-mediator interactions are evaluated at the relevant treatment value rather than dropped from the g-formula.
- The implemented single-mediator response-scale total effect equals its direct plus indirect components by construction.
- Off-floor outcomes are reported as risk differences, not as item gains.
- Positive values consistently mean that the intervention helps.
- Chain simulations reuse coherent same-world mediator draws for the structural ordering they implement.
- Model and report prose already acknowledge the principal unmeasured-confounding and contemporaneous-timing limitations. Those caveats should remain after the implementation is corrected.

## Proposed remediation sequence and acceptance criteria

### Phase A: make the fitted model contract explicit

- Add typed mediator-leg, outcome-leg, common-baseline and `pre_required` fields.
- Resolve all loaded variables before data access and fail on an omitted declared measure.
- Persist declared, loaded and fitted variables and fitted child IDs by leg.
- Correct `MED-060` and the off-floor analysis populations.

**Acceptance:** synthetic and real-data build tests show that every resolved term has a coefficient or documented structural role, no undeclared term is fitted, and no unused measurement changes row membership.

### Phase B: define estimands before implementing them

- Decide whether to implement an exposure-induced-confounder-aware interventional effect or retire or relabel the alias variants.
- Restrict `MED-092`'s primary contrast to supported first-period treatment cells.
- Define parallel joint-law and chain-order assumptions; use precise PNDE, TNIE and sequential-allocation names.

**Acceptance:** each output row has a mathematical counterfactual definition, an empirical support statement and a synthetic-data test whose expected result is known.

### Phase C: diagnose the full joint model and simulation

- Add per-node mediator and outcome PPCs with pre-specified interpretation, derived-estimand convergence checks, inner-simulation stability checks and design-appropriate child-level predictive validation.
- Rename the current tipping grid and, if required, add a coherent unmeasured-confounding sensitivity model.

**Acceptance:** release fails under pre-specified predictive criteria with uncertainty accounted for, when any headline decomposition has inadequate effective sample size or Monte Carlo precision, or when the inner simulation materially changes across seeds or replicate counts. Predictive checks remain model checks rather than causal-identification tests.

### Phase D: invalidate, refit and re-review

- Mark all 19 existing mediation artefact sets as pre-remediation and non-publishable.
- Retire unsupported aliases or refit every retained specification whose likelihood or fitted sample changes under the reporting preset.
- Compare fitted samples, total-effect direction and scale against the relevant ITT results as a coherence check, not an equality requirement.
- Re-run the code and statistical review against traces, PPCs, diagnostics, resolved plans, release decisions and rendered reports.

**Acceptance:** every model has zero divergences, passes the project's R-hat, effective-sample-size and BFMI thresholds, passes the new derived-estimand and per-node predictive checks, and carries no unsupported causal language.
