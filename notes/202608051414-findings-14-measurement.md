<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Findings 14 — the measurement models (correlated domain factors)

Reports every model in the `corr_factor` family (4) and the `long_corr_factor` family (1) from the 2026-08-04/05 `reporting` refit. **5 models, all passing the convergence gate.** Reading conventions in note 00. Preliminary research data — all estimates provisional.

## What these models do

Every other family treats each test score as the skill. These ask what the tests are _measuring_: they fit **latent domain factors** — vocabulary, code, grammar — behind the observed indicators, and report how strongly those latent domains correlate.

**Why this matters for the rest of the suite.** Test scores contain measurement error, which attenuates every observed correlation. Latent-variable correlations are **disattenuated** — corrected for that error — so they are systematically _higher_ than the correlations a reader would compute from raw scores. They are the right quantity for asking "are these really distinct abilities?" and the wrong quantity for predicting an individual child's scores.

**Design.** Confirmatory factor analysis with standardised indicators, an LKJ prior on the domain correlation matrix, and the factor scores marginalised out of the Gaussian measurement likelihood. Everything reported here is an **association**; there is no causal content and no treatment term.

## `mm-001` — the RLI trial cohort, three domains

| Domain pair          | Correlation (89%)          | P(>0) |
| -------------------- | -------------------------- | ----: |
| Vocabulary ↔ grammar | **+0.80** (+0.62 to +0.92) | 1.000 |
| Vocabulary ↔ code    | **+0.76** (+0.52 to +0.92) | 1.000 |
| Code ↔ grammar       | **+0.67** (+0.38 to +0.88) | 0.999 |

**The three factors are not equally well measured, and that matters for reading the table.** Vocabulary is defined sharply — receptive and expressive vocabulary load 0.88 and 0.87, with communalities 0.81 and 0.79. Grammar is middling (basic concepts 0.75, receptive grammar 0.73; communalities 0.58 and 0.55). **Code is the weakest**: letter sounds and blending load only 0.63 and 0.62, so barely 40% of each indicator's variance is factor rather than noise.

**The domains are highly but not perfectly correlated.** At 0.67–0.80 these are distinguishable constructs that share a great deal. The code domain — letter sounds and blending — is the most separable on the point estimates, sitting furthest from vocabulary and grammar. That is a mild piece of support for treating decoding as its own thing, consistent with the mechanism family's decoding-specificity result (note 07), but it should be leaned on lightly: code is the least well-measured factor, and its correlations are correspondingly the least precise in the table (code ↔ grammar spans 0.38 to 0.88, three times the width of vocabulary ↔ grammar).

`mm-101` is a prior-sensitivity companion with recalibrated priors; its correlations (+0.75, +0.80, +0.66) are within a hundredth or two of `mm-001`. The domain structure is not prior-driven.

`mm-002` is an **errors-in-variables** version of the code → word-reading mechanism, using the latent code factor instead of the observed letter-sound score. Its domain correlations match `mm-001` closely (+0.78, +0.79, +0.68). Its purpose is to check whether the mechanism-family coupling survives correcting the exposure for measurement error — it does, and more strongly, which is the expected direction since attenuation biases observed couplings downward.

## `rlm-mm-001` — the historical Byrne cohort, four domains

| Domain pair        | Correlation (89%)          | P(>0) |
| ------------------ | -------------------------- | ----: |
| Language ↔ ability | **+0.95** (+0.91 to +0.98) | 1.000 |
| Memory ↔ ability   | **+0.92** (+0.85 to +0.97) | 1.000 |
| Reading ↔ ability  | **+0.90** (+0.84 to +0.93) | 1.000 |
| Language ↔ memory  | **+0.89** (+0.80 to +0.95) | 1.000 |
| Reading ↔ language | +0.83 (+0.75 to +0.89)     | 1.000 |
| Reading ↔ memory   | +0.82 (+0.73 to +0.89)     | 1.000 |

**This cohort is close to one-dimensional.** Every pair correlates at 0.82–0.95, and language ↔ ability at 0.95 is barely distinguishable from unity. The dominant eigenvalue carries roughly 90% of the variance among these four domains.

**This is the single most important context for reading the whole suite.** When four notionally distinct abilities correlate at 0.9, "adjusting for" one of them removes a large share of the others too — which is exactly why the concurrent family (note 11) sees every association halve on adjustment, and why no observational model in this study can separate a specific skill effect from general ability. The latent structure makes concrete what the phrase "latent general ability is unblockable" means throughout these notes.

Two reporting obligations attach to these numbers. They are **latent** correlations, disattenuated for measurement error, and therefore sit systematically above any raw inter-test correlation a reader might compute. And every memory-domain correlation scales with an assumed single-indicator reliability of 0.8, because recall of digits is the only memory indicator in the prepared extract — so the memory correlations are the least secure in the table.

## `lcf-001` — the longitudinal version

Extends the factor model across all four waves, decomposing each domain into a stable **trait** part and a wave-specific **state** part.

| Domain     | Trait share (89%)       |
| ---------- | ----------------------- |
| Vocabulary | **0.95** (0.92 to 0.98) |
| Grammar    | **0.95** (0.89 to 0.99) |
| Code       | **0.93** (0.85 to 0.99) |

**The latent skills are almost entirely stable.** Between 93% and 95% of the variance in each latent domain is trait rather than state — meaning that where a child sits relative to their peers on vocabulary, grammar or code barely moves across the study's four waves.

The per-wave correlations are correspondingly flat: vocabulary ↔ grammar sits at 0.86, 0.85, 0.86 across waves 1–3, and vocabulary ↔ code at 0.66–0.67. The measurement structure is stable over time, so the cross-sectional factor models are not capturing a moment.

**This has a direct bearing on the intervention findings.** If 93–95% of latent skill variance is stable trait, then the movement the ITT family detects is happening in the remaining few per cent of between-child variation. That is not a criticism of the effect — a real gain in a highly stable system is a meaningful thing — but it does explain why baseline characteristics predict _level_ almost perfectly and _gain_ barely at all (note 13), and why the effects are modest in absolute items.

## Caveats

- **Latent correlations are disattenuated** and are not comparable with raw correlations between test scores.
- **Memory in the Byrne cohort rests on a single indicator** with an assumed reliability of 0.8; those correlations move with that assumption.
- **Near-unidimensionality in the Byrne cohort** (dominant eigenvalue ≈ 90%) should be reported explicitly rather than left to be inferred from six pairwise numbers.
- **No causal content** anywhere in this family.
- **These four `corr_factor` fits previously failed the convergence gate** and are reported here for the first time from clean fits. The failures were traced to unidentified scale components in the covariance prior rather than to the near-singular correlation geometry that had been blamed; switching to a bare correlation prior cleared all four. The domain correlations rose by 0.001–0.028 in the process (up to 0.23 posterior SD), so the numbers above supersede any earlier quotation — see the run record for the full account.
- **No coverage statistic is emitted for the latent measurement nodes**, by design: there is no single count outcome to score, so the measurement side gets a per-indicator distribution overlay and no interval-coverage number. The two purely measurement fits (`rlm-mm-001`, `lcf-001`) therefore report no coverage at all. The three RLI models do carry one — but for their **structural outcome leg** (0.57–0.61 on the 50% bands), not for the factor model, and it should not be read as a check on the measurement structure.
