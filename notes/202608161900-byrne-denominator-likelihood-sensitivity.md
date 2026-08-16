> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

<!-- cspell:ignore basnum basspel Byrne MacDonald readgrp woco -->

# Byrne provisional-denominator likelihood sensitivity

**Decision for #338, 2026-08-16: the reported historical-growth directions are empirically robust to a wide denominator stress test and a denominator-free count likelihood, but this does not identify any instrument ceiling and does not clear the publication gate. Do not promote the Negative-Binomial prototype automatically.** The primary source confirms that raw scores were analysed but does not state the maxima for BAS spelling, WORD reading comprehension or BAS number skills. A broader likelihood comparison can test dependence on the operational observed maxima; it cannot turn those maxima into instrument facts.

## Question and boundary

The registered `lrp-rlm-hg-002`, `lrp-rlm-hg-003` and `lrp-rlm-hg-008` models use Beta-Binomial likelihoods with provisional observed-maximum denominators of 18, 31 and 60. The sensitivity retains each model's complete-case core, available-case extension, group-by-wave mean structure and group-specific child random intercept. It varies only the score likelihood:

| Variant             | Denominator                     | Purpose                                                                   |
| ------------------- | ------------------------------- | ------------------------------------------------------------------------- |
| `beta_binomial_1x`  | observed full-extract maximum   | reproduce the current operational choice                                  |
| `beta_binomial_2x`  | twice the observed maximum      | stress a materially wider bounded scale                                   |
| `beta_binomial_4x`  | four times the observed maximum | stress a very wide bounded scale                                          |
| `negative_binomial` | none                            | test a non-negative, overdispersed count likelihood with no score ceiling |

The 1×/2×/4× values are mathematical perturbations, not candidate test forms. This is an actual-data empirical sensitivity, not a parameter-recovery simulation and not an independent confirmatory analysis.

Before the full run, the implementation fixed a deliberately permissive but explicit pass rule: all four fits must pass the project's convergence gate; every reported growth median must keep its direction; all four 89% intervals for each quantity must share a common overlap; and no quantity's median range may exceed 10% of the measure's observed full-extract maximum. Passing means only that the growth read-out is not being created by the current denominator choice.

## Full-study result

All 12 four-chain fits passed: zero divergences, maximum R-hat 1.0095, minimum bulk/tail effective sample size 532 and minimum chain BFMI 0.584. The three measure-level bundles passed every empirical robustness rule.

| Measure                      | Analysis children | Analysis rows | Largest median range | Fraction of observed range | Quantity with largest range                                               |
| ---------------------------- | ----------------: | ------------: | -------------------: | -------------------------: | ------------------------------------------------------------------------- |
| BAS spelling (`basspel`)     |                69 |           271 |          0.39 points |                       2.2% | Down-syndrome wave 1→5 growth                                             |
| WORD comprehension (`woco`)  |                77 |           304 |          2.41 points |                       7.8% | Reading-matched wave 1→4 growth                                           |
| BAS number skills (`basnum`) |                73 |           272 |          0.86 points |                       1.4% | Reading-matched minus average-reader total-growth contrast over waves 1→4 |

Every median direction was stable and every four-model set of 89% intervals overlapped. These results support the narrow statement that the current descriptive growth directions are not an artefact of choosing the observed maximum rather than a denominator two or four times larger.

## Why the denominator-free fit is not yet a replacement

The Negative-Binomial model avoids a fabricated ceiling, but it is unbounded and therefore assigns some probability to scores beyond any real test maximum. Its posterior-predictive 99th/99.9th percentiles were 25/33 for spelling, 43/64 for WORD comprehension and 72/86 for number skills; the proportions above twice the observed maximum were 0.034%, 0.118% and less than 0.001%, respectively. Those tails are small but cannot be judged physically plausible without the same missing scale information. Posterior-predictive 90% coverage was also broad (96.7%–99.3% for the Negative-Binomial fits), so a stable mean-growth estimate is not evidence that the whole predictive distribution is well calibrated.

The defensible interpretation is therefore asymmetric: the sensitivity weakens concern that the published growth direction is denominator-driven, but it does not establish the Negative-Binomial likelihood as a model of record. The registered Beta-Binomial fits remain withheld. Clearing the existing models still requires the administered manuals/test records, or an explicitly approved raw-score analysis whose estimand and predictive limitations are accepted in advance. A participant-level Bayesian bootstrap of paired raw-score growth is the cleaner next denominator-free candidate for the Phase-A descriptive question because it would not invent an upper tail; it would not by itself repair the factor, adjusted or horseshoe models that use score transforms.

## Reproduction and artefacts

The implementation is in `src/language_reading_predictors/statistical_models/rlm_denominator_sensitivity.py`; the command-line harness is `scripts/rlm_denominator_sensitivity.py`; structural and decision-rule guards are in `tests/statistical_models/test_rlm_denominator_sensitivity.py`. The full study command was:

```bash
python scripts/rlm_denominator_sensitivity.py --mode study --draws 1500 --tune 1500 --chains 4 --cores 4 --output-dir output/statistical_models/sensitivity/rlm-denominator-likelihood
```

The ignored output directory contains a trace plus SHA-256 digest for every variant, per-measure `diagnostics.csv`, `growth.csv` and `comparison.csv`, and the top-level `decision.json`. Tables can be regenerated without resampling by adding `--reuse-traces` to the same command.

## Reference

Byrne, A., MacDonald, J., & Buckley, S. (2002). Reading, language and memory skills: A comparative longitudinal study of children with Down syndrome and their mainstream peers. _British Journal of Educational Psychology, 72_(4), 513–529. <https://doi.org/10.1348/00070990260377497>
