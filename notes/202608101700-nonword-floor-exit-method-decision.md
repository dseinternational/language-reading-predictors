<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# Method decision — word reading and subsequent nonword floor exit

**Date:** 2026-08-10  
**Issue:** #433  
**Status at decision time:** Pre-fit specification for a promotion probe; no production model is authorised by this note.

## Why a new probe is necessary

The earlier `18 ± 4.4` expected-log-predictive-density advantage for including word reading came from a graded Beta-Binomial model restricted to transitions with period-start nonword reading `N_pre <= 4`. It does not establish that a Bernoulli true-floor-exit model distinguishes `log1p(W_pre)` from a no-exposure null. Raw exit counts also cannot make that comparison. Registration therefore remains blocked until the Bernoulli comparison below is fitted and passes the pre-specified promotion rule.

## Locked analysis decisions

### Primary population and outcome

The primary population is every t1-to-t2, t2-to-t3 or t3-to-t4 transition with observed period-start word reading, period-start nonword reading, period-end nonword reading, age, randomised arm and current-treatment status, restricted to `N_pre == 0`. The Bernoulli outcome is `I(N_post > 0)`: whether a child who began the period at zero of six nonwords read at least one at period end. Because `N_pre` is constant by construction, no own-baseline coefficient enters. This is true floor exit, not off-floor prevalence over all rows.

The planning frame has 95 transitions from 48 children and 36 exits. The `W_pre <= 25` sensitivity has 92 transitions from 47 children and 33 exits. The probe must recompute these counts after applying the final row policy and record ordered-row and observed-array SHA-256 digests. Full and null fits within a comparison must have identical digests.

### Covariates, timing and missing rows

The model includes period-start age, randomised arm, current-treatment status, period-start hearing status and period-start speech score, plus a child random intercept. Hearing and speech use the established mean-fill-plus-missingness-indicator policy: means are estimated once in the full true-floor frame, then frozen for the tail sensitivity; separate binary indicators identify filled hearing and speech values. This retains the planning population rather than dropping 16 transitions with missing hearing and three with missing speech. Age and the filled hearing and speech terms are centred and scaled once in the full true-floor frame. Arm, current treatment and the missingness indicators remain binary. This policy relies on the modelled indicators to distinguish missing from observed values; it does not make the missingness mechanism ignorable.

The full and null models share the same intercept, covariate priors, child-effect prior, rows and sampler settings. The null removes only the word-reading coefficient.

### Exposure transform and priors

The candidate exposure is `log1p(W_pre)`, centred at its mean and divided by its population standard deviation in the full true-floor frame. Those centring and scaling constants are frozen for every fit, including `W_pre <= 25`, so the prior has the same meaning throughout. The primary slope prior is `Normal(0, 0.3)` per one standard deviation of transformed word reading. A `Normal(0, 1)` sensitivity tests the wider production-mechanism default. Shared regression coefficients use `Normal(0, 0.3)`, the intercept `Normal(0, 1.5)`, the child-effect standard deviation `HalfNormal(0, 0.5)`, and non-centred child effects `Normal(0, 1)`.

### Reported contrasts

The headline translation is the transition-standardised difference in posterior floor-exit risk between zero and five period-start words, averaging over the analysis transitions' observed covariates and fitted child effects. Zero to five words covers the steep early range that motivated the teaching question. Zero to 25 words is reported as a secondary scale check. Each contrast reports the posterior median, inner 50% and outer 89% equal-tailed credible intervals, and the posterior probability that the risk difference is positive.

### Graded secondary

The existing Beta-Binomial `N_pre <= 4` analysis remains a separately labelled historical graded secondary. It answers a different question about period-end nonword counts among children with headroom. It is not refitted in the Bernoulli grid, does not contribute evidence to the Bernoulli promotion rule and cannot be registered alone if the floor-exit probe fails.

### Predictive comparison and its unit

The primary and sensitivity comparisons use pointwise PSIS-LOO over child-period transitions, conditional on the fitted child intercept. This asks how well each model predicts another transition for a child represented in the cohort; it does not estimate generalisation to a new child. The unit is chosen to match the historical transition-level probe and must be stated beside every comparison. If either model has a Pareto value above ArviZ's sample-size-dependent `good_k` threshold, the comparison is invalid until exact row-level refits repair every flagged point. Technical reliability does not make a small difference conclusive: `|elpd difference| < 4` remains inconclusive.

Every fit must pass the project computational gate over all free variables: R-hat no greater than 1.01, bulk and tail effective sample sizes at least 400, BFMI at least 0.3 and zero divergences. Sampling is four chains, 3,000 tuning iterations and 3,000 retained draws per chain at `target_accept=0.95` with `nutpie`.

## Pre-specified promotion rule

Production implementation is authorised only if every condition is true:

1. All eight fits in the two-population by two-prior by full/null grid pass the computational gate.
2. All four full-versus-null comparisons are technically valid, with no unrepaired unreliable Pareto values.
3. In all four comparisons, the full model exceeds the null by at least 4 expected log predictive density units.
4. In the primary all-words, `Normal(0, 0.3)` fit, the median zero-to-five-word floor-exit risk difference is at least +0.10.
5. In all four full fits, the posterior probability that the zero-to-five-word risk difference is positive is at least 0.95.
6. Every sensitivity's median zero-to-five-word risk difference is within 0.10 of the primary median.

The 0.10 risk-difference thresholds are pre-specified practical tolerances: a ten-percentage-point change would materially alter the teaching interpretation of the sparse six-item floor-exit outcome. The rule is deliberately conjunctive. A favourable graded count result cannot compensate for an inconclusive Bernoulli comparison, a failed computational gate or instability across the instrument-tail and prior sensitivities.

If the rule fails, the promotion decision is **do not register**. Record the negative or inconclusive result in this note and in `notes/202607241900-findings-word-reading-threshold.md`, and close or re-scope #433. If it passes, a later pull request may add the Bernoulli likelihood, frozen `log1p` exposure transform, genuine null companion, registered modules and reports; this probe pull request must not do so.

## Interpretation boundary

Any resulting word-reading term is descriptive and associational. It is confounded by latent general ability, shared intervention exposure and measured or unmeasured code-skill routes. The model therefore has `causal_status="none"` and `estimand_type="descriptive"` if it is ever promoted. Its six-item binary outcome also discards graded information and may reflect instrument floor behaviour rather than a distinct learning state.

## Reproducibility

Run the locked grid from the repository root in the project conda environment:

```bash
python notes/assets/202608101700-nonword-floor-exit-probe.py
```

The script writes traces and audit tables under `output/notes/202608101700-nonword-floor-exit/`, including `analysis_identity.json`, `fit_diagnostics.csv`, `loo_comparisons.csv`, `risk_differences.csv` and `promotion_decision.json`.
