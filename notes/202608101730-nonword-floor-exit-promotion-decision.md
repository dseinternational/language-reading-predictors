<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# Promotion decision — do not register the nonword floor-exit model

**Date:** 2026-08-10

**Issue:** #433

**Decision:** Do not promote or register a production model.

## Result in plain language

Period-start word reading is predictively useful for whether a child at zero of six nonwords reads at least one nonword by period end. This conclusion is stronger than the historical graded-count evidence because the fresh full-versus-null comparisons use the proposed Bernoulli true-floor-exit outcome on exactly matched rows. However, the size of the translated association is materially sensitive to the slope prior: the median zero-to-five-word risk difference is about +25 percentage points under `Normal(0, 0.3)` and +39 points under `Normal(0, 1)`. That 14.5-point change exceeds the ten-point stability tolerance locked before fitting. The pre-specified conjunctive promotion rule therefore returns **do not promote**, even though the direction and predictive advantage are stable.

This is not evidence of no association. It is evidence that these data do not pin down a practically reported magnitude robustly enough for a registered model of record. Registering the narrower-prior answer would select the more convenient magnitude after seeing the sensitivity and would contradict the pre-fit rule.

## Frozen analysis populations

The row policy reproduced the issue's planning counts exactly:

| Population                                 | Transitions | Children | Floor exits | Hearing filled | Speech filled | Ordered-row SHA-256                                                |
| ------------------------------------------ | ----------: | -------: | ----------: | -------------: | ------------: | ------------------------------------------------------------------ |
| All observed period-start word scores      |          95 |       48 |          36 |             16 |             3 | `d0ad851ab502bb318f266b9324e61bcdcbd6f6d1a9896e78d34ee16d9d293f1b` |
| Period-start word score no greater than 25 |          92 |       47 |          33 |             15 |             2 | `57a302e66f60764b9eef3e1a5ca2c4d6d9affe5bf5e100a43d3b238261b5486d` |

The corresponding observed-array digests are `a40be2cdbd3e27fbfcd5b4fbee313bdc46a36bcce01da8f3aa1f5535b154ddee` and `62068986478a2743aa0542dc05a7ee4f3f6757c883138af1c7e9de5da6d6ca3c`. Within each population, full and null fits use the same row and observed-array digests. Hearing and speech were mean-filled with their missingness indicators as specified; the tail sensitivity reused the primary frame's fill, centring and scaling constants.

## Computation and predictive comparison

Every model used four chains, 3,000 tuning iterations and 3,000 retained draws per chain with `nutpie` and `target_accept=0.95`. The full and null models in each comparison also used the same random seed. Across all eight fits there were zero divergences, maximum R-hat was 1.0023, minimum bulk or tail effective sample size was 3,111, and minimum BFMI was 0.877. All therefore passed the project computational gate over every free variable.

Pointwise PSIS-LOO uses the child-period transition conditional on the fitted child intercept. This estimates prediction of another transition for a represented child, not generalisation to a new child. No Pareto value exceeded the sample-size-dependent `good_k=0.7` threshold; the largest was 0.535, so no exact refit was required.

| Population                     | Slope prior SD | Full minus null elpd | Difference SE | Largest Pareto k | Verdict                   |
| ------------------------------ | -------------: | -------------------: | ------------: | ---------------: | ------------------------- |
| All word scores                |            0.3 |               +10.76 |          2.54 |            0.535 | Full model discriminating |
| All word scores                |            1.0 |               +13.25 |          4.33 |            0.509 | Full model discriminating |
| Word scores no greater than 25 |            0.3 |                +8.29 |          2.20 |            0.498 | Full model discriminating |
| Word scores no greater than 25 |            1.0 |               +10.54 |          4.03 |            0.480 | Full model discriminating |

All four differences exceed the project's four-elpd inconclusive band. This supports including word reading for conditional transition prediction; it does not resolve the magnitude sensitivity or make the term causal.

## Posterior translation

The headline contrast standardises over analysis transitions' observed covariates and fitted child effects. Values are posterior median [inner 50% interval; outer 89% interval], followed by the posterior probability that the difference is positive.

| Population                     | Slope prior      | 0 to 5 words: floor-exit risk difference                | 0 to 25 words: floor-exit risk difference               |
| ------------------------------ | ---------------- | ------------------------------------------------------- | ------------------------------------------------------- |
| All word scores                | `Normal(0, 0.3)` | +0.245 [+0.205, +0.286; +0.149, +0.341], P(> 0) = 1.000 | +0.469 [+0.392, +0.541; +0.285, +0.629], P(> 0) = 1.000 |
| All word scores                | `Normal(0, 1)`   | +0.390 [+0.342, +0.439; +0.273, +0.505], P(> 0) = 1.000 | +0.721 [+0.647, +0.783; +0.526, +0.853], P(> 0) = 1.000 |
| Word scores no greater than 25 | `Normal(0, 0.3)` | +0.222 [+0.181, +0.265; +0.121, +0.321], P(> 0) = 1.000 | +0.426 [+0.347, +0.505; +0.231, +0.600], P(> 0) = 1.000 |
| Word scores no greater than 25 | `Normal(0, 1)`   | +0.373 [+0.321, +0.424; +0.251, +0.495], P(> 0) = 1.000 | +0.696 [+0.616, +0.766; +0.486, +0.844], P(> 0) = 1.000 |

Dropping the word-reading instrument tail changes the zero-to-five-word median by only 0.023 under the narrow prior and 0.017 under the wide prior. The problematic sensitivity is the slope prior: widening it changes the median by 0.145 in the full population and 0.151 in the tail-restricted population. Both exceed the locked 0.10 tolerance.

## Promotion-rule audit

| Pre-specified condition                                            | Result   |
| ------------------------------------------------------------------ | -------- |
| All eight fits pass the computational gate                         | Pass     |
| All four LOO comparisons are technically valid                     | Pass     |
| All four full models exceed the null by at least 4 elpd            | Pass     |
| Primary median zero-to-five-word risk difference is at least +0.10 | Pass     |
| Every zero-to-five-word contrast has P(positive) at least 0.95     | Pass     |
| Every sensitivity median is within 0.10 of the primary median      | **Fail** |

Because the rule was conjunctive, the final verdict is **do not promote**.

## Consequences

- Do not add a Bernoulli mechanism likelihood, production module, registry entry or report for this question.
- Retain the historical graded Beta-Binomial result as an exploratory secondary, clearly separated from the Bernoulli evidence.
- Close #433 when this decision merges. A future model would require either independently justified prior elicitation for the transformed slope or new data that reduce the prior dependence; it should not reopen merely to choose one of the two fitted priors post hoc.
- Keep the interpretation descriptive: latent general ability, shared intervention exposure and code-skill routes remain uncontrolled, and a six-item floor-exit outcome can reflect instrument behaviour as well as learning.

## Reproduction

The pre-fit decision is [`202608101700-nonword-floor-exit-method-decision.md`](202608101700-nonword-floor-exit-method-decision.md). The probe is [`202608101700-nonword-floor-exit-probe.py`](assets/202608101700-nonword-floor-exit-probe.py). Run it with the project conda environment; it writes `analysis_identity.json`, `fit_diagnostics.csv`, `loo_comparisons.csv`, `risk_differences.csv`, `promotion_decision.json` and the eight traces under `output/notes/202608101700-nonword-floor-exit/`.
