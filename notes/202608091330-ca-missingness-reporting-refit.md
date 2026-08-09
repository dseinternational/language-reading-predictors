<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# Concurrent-family missingness reporting refit — pinned closeout record

## Decision

The 2026-08-09 production refit of `lrp-rli-ca-001`–`lrp-rli-ca-011` satisfies the repository's automatic computational and artefact-release gates. All 11 model-level `release_decision.json` files say `status: ok`, `stage: robustness` and `publishable: true`; all 280 adjusted or single-skill posteriors pass the unrounded convergence thresholds. These outputs may replace the earlier concurrent-family reporting bundle once the Azure publication URLs below are populated and verified. Convergence is necessary for release but does not identify any causal effect, validate the missing-data assumptions or make the PSIS-LOO values uniformly reliable.

## Pinned run identity

| Field         | Pinned value                                                                                                       |
| ------------- | ------------------------------------------------------------------------------------------------------------------ |
| Run name      | `20260809T110432Z-df00982f53cc`                                                                                    |
| Output root   | `output/runs/20260809T110432Z-df00982f53cc`                                                                        |
| Run start     | `2026-08-09T11:04:32Z`                                                                                             |
| Git commit    | `df00982f53cc44f3635abaef01c4c09ecf185afe`                                                                         |
| Configuration | `reporting`                                                                                                        |
| Models        | `lrp-rli-ca-001`–`lrp-rli-ca-011`                                                                                  |
| Sampling      | nutpie; 6 chains; 6 cores; 6,000 tuning and 6,000 retained draws per chain; `target_accept = 0.95`; random seed 47 |
| Primary input | `data/rli_data_long.csv`, SHA-256 `dc8dda5780b705e902155372c135a993778506c547ef8ebb2b5b03668c11f043`               |

The run manifest pins all seven files in the repository's data identity, not just the primary long table:

| Data-relative file                                              | SHA-256                                                            |
| --------------------------------------------------------------- | ------------------------------------------------------------------ |
| `LICENSE`                                                       | `9e5f1b3c610b9c2da5c313bf81d577a7d1acec686bdb0384edefa6df0f90cd94` |
| `reading-language-memory/README.md`                             | `2c552ff0a34bc81ad4dadc1ec61adfd75e2e3de6b3ef5911f94bddd1060dabf2` |
| `reading-language-memory/reading_language_memory_data_long.csv` | `68ea2e9c847c908b7217431af76abd45a940099ced2bfd9acf4dd69ba7e2e5f6` |
| `reading-language-memory/reading_language_memory_data_wide.csv` | `b2262d6b3b7102594b3424c4a72f4237dc84087a7b18f6fc815ccdcd0d10a55c` |
| `readme.md`                                                     | `6205d3a5e49db8a8043b15de75a35b15086f7af0c67e873c1f56c19c517f90f8` |
| `rli_data_long.csv`                                             | `dc8dda5780b705e902155372c135a993778506c547ef8ebb2b5b03668c11f043` |
| `rli_data_wide.csv`                                             | `2c47eb49a96013a0283a225dcd8460ceb62720fdca60bcaeb3811345e5b7c99c` |

## Exact fit and artefact counts

Each full-family model has one adjusted fit and one single-skill comparator per eligible predictor at each of four waves. The 11 primary t3 posteriors are stored in the model-level `trace.nc` files; the other 269 fits are recorded row-for-row in `subfit_provenance.csv`. Together they account for all 280 diagnostic rows.

| Model            | Outcome                      | Fitted children at t1/t2/t3/t4 |    Fits | Association rows | Marginal rows | Subfit-provenance rows |
| ---------------- | ---------------------------- | ------------------------------ | ------: | ---------------: | ------------: | ---------------------: |
| `lrp-rli-ca-001` | Word reading                 | 53 / 53 / 53 / 51              |      28 |               24 |            96 |                     27 |
| `lrp-rli-ca-002` | Letter sounds                | 54 / 54 / 54 / 52              |      28 |               24 |            96 |                     27 |
| `lrp-rli-ca-003` | Taught receptive vocabulary  | 54 / 54 / 54 / 53              |      28 |               24 |            96 |                     27 |
| `lrp-rli-ca-004` | Taught expressive vocabulary | 54 / 54 / 54 / 53              |      28 |               24 |            96 |                     27 |
| `lrp-rli-ca-005` | Receptive vocabulary         | 54 / 54 / 54 / 53              |      28 |               24 |            96 |                     27 |
| `lrp-rli-ca-006` | Expressive vocabulary        | 54 / 54 / 54 / 53              |      28 |               24 |            96 |                     27 |
| `lrp-rli-ca-007` | Phoneme blending             | 54 / 54 / 54 / 53              |      28 |               24 |            96 |                     27 |
| `lrp-rli-ca-008` | Basic concepts               | 54 / 54 / 54 / 52              |      32 |               28 |           112 |                     31 |
| `lrp-rli-ca-009` | Receptive grammar            | 54 / 54 / 54 / 53              |      32 |               28 |           112 |                     31 |
| `lrp-rli-ca-010` | Word reading                 | 53 / 53 / 53 / 51              |       8 |                4 |            16 |                      7 |
| `lrp-rli-ca-011` | Word reading                 | 53 / 53 / 53 / 51              |      12 |                8 |            32 |                     11 |
| **Total**        |                              |                                | **280** |          **236** |       **944** |                **269** |

Every model directory also contains the model-level trace, configuration, resolved run plan, model recipe, diagnostics, key findings, artefact manifest, release decision, rendered report and supporting figures/tables required by the audited reporting contract.

## Exact computational extrema

All 280 `concurrent_fit_diagnostics.csv` rows report `converged = True` and zero divergences. The family-wide extrema, evaluated before display rounding, are:

- Maximum R-hat: **1.0008486151245863**, in the `lrp-rli-ca-007` t2 expressive-vocabulary single-skill comparator.
- Minimum effective sample size: **15848.055785505983**, in the `lrp-rli-ca-005` t3 letter-sound single-skill comparator. This is the gate's minimum effective-sample-size diagnostic across the fitted variables, not a raw draw count.
- Minimum per-chain BFMI: **0.8315319825399711**, in the `lrp-rli-ca-009` t2 adjusted fit.
- Divergences: **0 across all 280 fits**.

The thresholds are R-hat ≤ 1.01, effective sample size ≥ 400, BFMI ≥ 0.3 and zero divergences. The large margins establish stable computation for these fitted models; they do not address confounding, model misspecification, measurement error or causal identification.

## Missingness and comparator contract now fitted

`lrp-rli-ca-001`–`lrp-rli-ca-009` request non-verbal ability (`blocks`), hearing (`hs`), speech production (`deapp_c`) and phonological memory (`erbto`) as trait covariates, together with the corresponding missingness indicators where applicable. `lrp-rli-ca-010/011` intentionally use the narrower Q1/Q2 trait set (`blocks`, `hs`, `hs_missing`). Filled trait values carry a paired missing-indicator nuisance offset whenever that flag varies on the wave's outcome-complete rows; a fitted-row-constant flag is dropped instead of being aliased with the intercept. In `ca-001`, `erbto_missing` is constant after the word-reading outcome mask and is therefore dropped at all four waves. In `ca-002` it varies and is fitted at t1–t3 but is dropped as constant at t4; all requested flags vary and are fitted at all waves in `ca-003`–`ca-009`.

The missingness coefficients are nuisance subgroup offsets. They are not skill effects, evidence that missingness causes the outcome, or candidates for substantive ranking. This policy distinguishes an imputed value from an observed value at the fill point but does not recover unobserved data or establish a missing-at-random mechanism. Rows missing the focal outcome remain excluded, and the family retains its documented descriptive fill policy for missing skill predictors.

The comparison column is now a **single-skill comparator**. It retains the same effective trait covariates as the adjusted fit while omitting age, randomised group and the other skills. The artefacts retain legacy `biv_*` column names for compatibility, but the fit is neither raw nor bivariate. The adjusted–comparator difference therefore mixes adjustment for the other skills with adjustment for age and group; it is not an identified general-factor decomposition.

## Substantive readout

The detailed values and intervals are in `notes/202608051411-findings-11-concurrent.md`. The broad result is descriptive clustering: word reading, letter sounds and blending are each other's strongest reading-side partners, while vocabulary and grammar outcomes usually pair most strongly with another language measure. The strongest adjusted letter-sound/word-reading summaries remain positive in both directions: median-across-wave logit slopes are +0.461 for letter sounds in `ca-001` and +0.474 for word reading in `ca-002`.

The registered minimal models give the clearer Q1/Q2 readout. In `ca-010`, adjusted letter-sound slopes are +0.643, +1.070, +0.717 and +0.813 at t1–t4. In `ca-011`, the corresponding slopes after holding measured nonword decoding are +0.615, +0.792, +0.578 and +0.527; the nonword slopes are +0.378, +0.358, +0.418 and +0.515. All directions have posterior probability above 0.991. These are same-wave adjusted associations, not effects. The severe floor and six-item resolution of the nonword measure mean that `ca-011` only partially holds decoding ability fixed, while conditioning on this post-treatment mediator may open collider paths.

Across `ca-001`–`ca-009`, the strongest adjusted slope is 52%–70% of the single-skill comparator (median 63%). That attenuation is compatible with shared covariance among skills but does not identify a common factor or quantify confounding.

## Predictive qualifications

The primary t3 anchors contain 591 observations across the 11 models. Eleven observations have PSIS Pareto `k > 0.7`, one has `k > 1`, and the maximum is **1.105605700969821** in `lrp-rli-ca-002`. The above-threshold values occur in `ca-002`–`ca-009` and mostly identify the same pseudonymous child. The `ca-001`, `ca-010` and `ca-011` primary anchors have no value above the threshold. The affected stored LOO values should not support close model comparisons without a more robust refit or validation calculation. Pareto `k` concerns importance-sampling reliability; it neither negates the clean MCMC gate nor repairs identification.

Across the same primary anchors, nominal 50% posterior predictive intervals contain 432/591 observations (73.1%) and nominal 90% intervals contain 577/591 (97.6%). The intervals are conservative overall. This is a predictive calibration description, not a guarantee of correct conditional structure.

## Historical comparison and its limits

The retrievable historical public bundle from run `019f6be6-5b26-7399-b95c-81690837e08b` contains complete numerical artefacts for `ca-001`–`ca-006` only. Its adjusted fits omitted the present trait-covariate block, and its comparison fits were predictor-only. The new run adds the declared trait adjustment, varying missingness nuisances and the trait-retaining single-skill comparator. It also comes from the current typed-plan/provenance/reporting pipeline. The two bundles are therefore **not a controlled missingness ablation**.

Across the 144 matched wave/predictor rows in `ca-001`–`ca-006`, the new-minus-old change in adjusted posterior mean has median −0.0048, median absolute magnitude 0.0249 and maximum absolute magnitude 0.1507. Comparator posterior means fall by a median of 0.1217, with maximum absolute change 0.2538. The broad clustering survives, but some magnitudes and within-outcome rankings change. The larger comparator change follows its changed conditioning set and cannot be attributed specifically to adding missingness indicators. The historical bundle contains no trace-backed `ca-007`–`ca-011` artefacts, so no exact old/new claim is made for those models.

One dependent artefact is deliberately not promoted as current: the 32-row `lcf_concurrent_comparison.csv` in the 4/5-August longitudinal correlated-factor report reads sibling CA-002–006 tables from that earlier bundle. It remains a historical comparison until regenerated from these corrected CA tables. The LCF trace and latent-factor estimates themselves are unaffected, so this reporting follow-up does not require LCF resampling.

## Durable Azure publication — pending URLs

The authenticated upload and HTTP verification happen after this note is drafted. These placeholders are deliberate; they must be replaced with the actual Azure URLs recorded by the upload command, not inferred from a naming convention.

| Required published object                   | Verified Azure URL                                   |
| ------------------------------------------- | ---------------------------------------------------- |
| Run manifest                                | **PENDING AFTER AZURE UPLOAD AND HTTP VERIFICATION** |
| Published run root or URL inventory         | **PENDING AFTER AZURE UPLOAD AND HTTP VERIFICATION** |
| `lrp-rli-ca-001-reporting` report and trace | **PENDING AFTER AZURE UPLOAD AND HTTP VERIFICATION** |
| `lrp-rli-ca-002-reporting` report and trace | **PENDING AFTER AZURE UPLOAD AND HTTP VERIFICATION** |
| `lrp-rli-ca-003-reporting` report and trace | **PENDING AFTER AZURE UPLOAD AND HTTP VERIFICATION** |
| `lrp-rli-ca-004-reporting` report and trace | **PENDING AFTER AZURE UPLOAD AND HTTP VERIFICATION** |
| `lrp-rli-ca-005-reporting` report and trace | **PENDING AFTER AZURE UPLOAD AND HTTP VERIFICATION** |
| `lrp-rli-ca-006-reporting` report and trace | **PENDING AFTER AZURE UPLOAD AND HTTP VERIFICATION** |
| `lrp-rli-ca-007-reporting` report and trace | **PENDING AFTER AZURE UPLOAD AND HTTP VERIFICATION** |
| `lrp-rli-ca-008-reporting` report and trace | **PENDING AFTER AZURE UPLOAD AND HTTP VERIFICATION** |
| `lrp-rli-ca-009-reporting` report and trace | **PENDING AFTER AZURE UPLOAD AND HTTP VERIFICATION** |
| `lrp-rli-ca-010-reporting` report and trace | **PENDING AFTER AZURE UPLOAD AND HTTP VERIFICATION** |
| `lrp-rli-ca-011-reporting` report and trace | **PENDING AFTER AZURE UPLOAD AND HTTP VERIFICATION** |

Publication is complete only when these URLs are populated from the uploader's recorded output and each required report, trace and manifest returns successfully from Azure. Until then the local run is fully audited but the durable-publication step remains open.
