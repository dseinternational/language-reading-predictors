<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5) and substantially revised for the pinned production refit by a LLM-based AI tool (Codex/GPT-5).

# Findings 11 — the concurrent-association family

Reports every model in the `concurrent` family from the pinned 2026-08-09 `reporting` refit at commit `df00982f53cc44f3635abaef01c4c09ecf185afe`. **All 11 model-level release decisions are publishable and all 280 adjusted or single-skill fits pass the automatic convergence gate.** This is evidence that the sampler was stable under the fitted models, not evidence that the associations are identified causally. Reading conventions are in note 00. Preliminary research data — all estimates remain provisional.

## What these models do

The family asks a deliberately limited question: **at a single point in time, which measured skills go together?** Each model fits four separate between-child Beta-Binomial regressions, one at each wave. `ca-001`–`ca-009` put the other eligible same-wave skills into the adjusted model alongside age, randomised group as a nuisance, non-verbal ability, hearing, speech production and phonological memory. `ca-010` and `ca-011` are the narrower, pre-specified letter-sound/word-reading comparisons with age, group, non-verbal ability and hearing; `ca-011` additionally conditions on same-wave nonword decoding.

The second column in the artefacts is a **single-skill comparator**, despite the retained legacy `biv_*` CSV column names. It keeps the adjusted model's same effective trait covariates but omits age, group and the other skills. It is therefore neither a raw association nor a bivariate regression. The adjusted–comparator gap combines mutual skill adjustment with removal of age and group, so it cannot by itself isolate a general factor.

Missing trait covariates use reference/mean filling paired with a missing-indicator **nuisance offset** whenever that indicator varies on the wave's outcome-complete rows. A constant indicator is dropped rather than aliased with the intercept. The indicator is not a skill, a substantive predictor or evidence about why data are missing; the policy only prevents an imputed value from being treated as an observed mean and does not guarantee unbiased associations.

**There is no causal content and no temporal ordering.** These are contemporaneous, mostly post-treatment conditional associations. They cannot distinguish whether one skill affects another, whether the reverse direction holds, or whether common causes such as latent general ability account for both.

## Results — strongest adjusted associations by outcome

The table is a descriptive summary of the four wave-specific posteriors: each number is the median across the four reported posterior medians on the outcome-logit scale per +1 SD of the predictor's same-wave logit. The probability is likewise the median of the four wave-specific `P(slope > 0)` values. These are not pooled posterior estimates; use the model reports for each wave's 50% and 89% credible intervals.

| Model    | Outcome                      | Strongest adjusted / single-skill comparator associations                                                      |
| -------- | ---------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `ca-001` | Word reading                 | Letter sounds **+0.461 / +0.734** (P = 0.996); taught expressive vocabulary +0.282 / +0.556 (P = 0.952)        |
| `ca-002` | Letter sounds                | Word reading **+0.474 / +0.728** (P = 0.998); expressive vocabulary +0.293 / +0.517 (P = 0.941)                |
| `ca-003` | Taught receptive vocabulary  | Taught expressive vocabulary **+0.194 / +0.362** (P = 0.939); receptive vocabulary +0.172 / +0.298 (P = 0.916) |
| `ca-004` | Taught expressive vocabulary | Expressive vocabulary **+0.324 / +0.504** (P = 0.986); taught receptive vocabulary +0.207 / +0.404 (P = 0.941) |
| `ca-005` | Receptive vocabulary         | Expressive vocabulary **+0.139 / +0.229** (P = 0.913); taught receptive vocabulary +0.092 / +0.158 (P = 0.872) |
| `ca-006` | Expressive vocabulary        | Taught expressive vocabulary **+0.146 / +0.267** (P = 0.966); letter sounds +0.092 / +0.201 (P = 0.892)        |
| `ca-007` | Phoneme blending             | Word reading **+0.313 / +0.447** (P = 0.976); letter sounds +0.214 / +0.444 (P = 0.881)                        |
| `ca-008` | Basic concepts               | Receptive vocabulary **+0.232 / +0.447** (P = 0.928); taught receptive vocabulary +0.211 / +0.478 (P = 0.913)  |
| `ca-009` | Receptive grammar            | Receptive vocabulary **+0.153 / +0.235** (P = 0.922); taught receptive vocabulary +0.147 / +0.286 (P = 0.923)  |

Two descriptive patterns survive the corrected specification. First, the strongest reading-side links remain word reading, letter sounds and blending, while the language measures most often pair with another vocabulary or grammar measure. The letter-sound/word-reading relationship is the strongest adjusted association in both directions (`ca-001` and `ca-002`). This is consistent with the wider mechanism work, but contemporaneous agreement supplies no additional evidence about direction or causation.

Second, mutual adjustment usually attenuates the strongest association. Across `ca-001`–`ca-009`, the strongest adjusted slope is 52%–70% of its single-skill comparator, with a median of 63%. Across all 56 predictor/outcome summaries the median ratio is 39%, but seven weak terms change sign and ratios become unstable when a comparator is close to zero. The attenuation is compatible with substantial shared covariance among skills; it does not identify a general factor, quantify confounding, or show that the comparator is biased.

## The two registered letter-sound/word-reading models

`ca-010` and `ca-011` give the production readout for the first two questions in `202607241000-findings-letter-sounds-word-reading-association.md`. Each cell below is `median [50% interval; 89% interval]`, followed by `P(>0)`. These are adjusted logit-scale slopes per +1 SD of the named same-wave predictor.

| Wave | `ca-010`: letter sounds                         | `ca-011`: letter sounds with decoding held      | `ca-011`: nonword decoding with letter sounds held |
| ---- | ----------------------------------------------- | ----------------------------------------------- | -------------------------------------------------- |
| t1   | +0.643 [+0.521, +0.768; +0.350, +0.941], 0.9997 | +0.615 [+0.494, +0.735; +0.328, +0.902], 0.9996 | +0.378 [+0.280, +0.475; +0.136, +0.602], 0.9916    |
| t2   | +1.070 [+0.966, +1.167; +0.819, +1.295], 1.0000 | +0.792 [+0.677, +0.904; +0.520, +1.056], 1.0000 | +0.358 [+0.280, +0.436; +0.173, +0.547], 0.9991    |
| t3   | +0.717 [+0.628, +0.806; +0.501, +0.929], 1.0000 | +0.578 [+0.488, +0.667; +0.362, +0.790], 1.0000 | +0.418 [+0.339, +0.497; +0.225, +0.605], 0.9993    |
| t4   | +0.813 [+0.736, +0.889; +0.626, +0.991], 1.0000 | +0.527 [+0.444, +0.611; +0.327, +0.726], 1.0000 | +0.515 [+0.430, +0.599; +0.312, +0.718], 0.9999    |

The production estimates support the same qualified descriptive conclusion as the exploratory probe. Letter sounds and word reading are strongly associated at every wave. Conditioning on the coarse six-item nonword score leaves the letter-sound slope clearly positive but reduces its median from t2 onward; the ratio of the separate posterior medians is 0.96, 0.74, 0.81 and 0.65 at t1–t4. That ratio is a separate-fit sensitivity, not a posterior estimand. The registered joint model `lrp-rli-jm-001` supplies a within-model share-retained quantity under a different latent-outcome model; it should be read alongside this observed-count conditioning analysis, not as an interchangeable correction.

On the 79-item word-reading scale, the `ca-010` +1 SD letter-sound marginal is +4.6, +11.0, +9.1 and +11.1 items at t1–t4. In `ca-011`, the corresponding letter-sound marginal is +3.9, +7.2, +6.8 and +6.5 items, while the nonword marginal is +2.2, +2.9, +4.7 and +6.3 items. These are average descriptive marginals over each wave's fitted children, not effects of an intervention.

The floor on nonword reading limits the estimand: conditioning on a heavily coarsened six-item measure does not fully hold decoding ability fixed. The residual letter-sound slope is therefore not a clean non-decoding pathway, and conditioning on a post-treatment mediator can also open collider paths.

## Computation, provenance and predictive qualifications

The refit comprises 280 fitted posteriors: 28 each for `ca-001`–`ca-007`, 32 each for `ca-008`–`ca-009`, 8 for `ca-010` and 12 for `ca-011`. The outputs contain 236 association rows, 944 marginal rows and 280 fit-diagnostic rows. The 269 rows in `subfit_provenance.csv` plus the 11 model-level primary traces account for all 280 fits. Every fit used six chains with 6,000 tuning and 6,000 retained draws, and every unrounded gate passed: maximum R-hat 1.0008486151245863, minimum effective sample size 15848.055785505983, minimum per-chain BFMI 0.8315319825399711 and zero divergences.

That clean sampler result is necessary but not sufficient. It neither resolves latent-ability confounding nor validates the missing-data assumptions. It also does not make all predictive diagnostics reliable. Across the 591 observations in the 11 primary-wave PSIS-LOO anchors, 11 have Pareto `k > 0.7` and one has `k > 1`; the maximum is 1.105605700969821 in `ca-002`. The above-threshold values occur in `ca-002`–`ca-009`, mostly for the same influential child. Do not use those models' stored LOO values for close model comparison without a more robust refit or validation calculation. `ca-001`, `ca-010` and `ca-011` have no primary-anchor value above the threshold. Pareto `k` diagnoses importance-sampling reliability; it is not an MCMC convergence or causal-identification test.

Primary-wave posterior predictive intervals are conservative overall: the nominal 50% ranges contain 432/591 observations (73.1%) and the nominal 90% ranges contain 577/591 (97.6%). This is a calibration description, not proof that the conditional associations are correctly specified.

## What changed from the earlier bundle

The retrievable historical public bundle contains `ca-001`–`ca-006`, not `ca-007`–`ca-011`. Its adjusted fits omitted the present trait-covariate block, and its comparison fits were predictor-only. The new fits add the declared trait adjustment and required varying missingness nuisances, and the single-skill comparator now retains the same effective trait covariates. Consequently, the two runs are **not** a controlled missing-indicator ablation.

For the 144 matched wave/predictor rows in `ca-001`–`ca-006`, the change in adjusted posterior mean has a median of −0.0048, a median absolute change of 0.0249 and a maximum absolute change of 0.1507. The comparator means fall by a median of 0.1217, with a maximum absolute change of 0.2538. The broad reading/language clustering is therefore robust, while some individual magnitudes and rankings move. The larger comparator change is expected from its changed adjustment geometry and must not be attributed specifically to missingness indicators. No trace-backed old/new numerical claim is made for `ca-007`–`ca-011` because the retrieved historical bundle does not contain them.

The 32-row `lcf_concurrent_comparison.csv` stored with the 4/5-August longitudinal correlated-factor report was built from the earlier CA-002–006 tables. It is therefore a historical cross-model comparison and must not be read as aligned to this corrected concurrent bundle. The latent-factor posterior itself is unaffected; rebuilding that derived table from the existing LCF trace and these new CA tables is a separate reporting refresh and does not require LCF resampling.

## Bottom line

The production refit supports a descriptive reading/language clustering and a strong same-wave letter-sound/word-reading relationship, including a positive residual letter-sound association when measured nonword decoding is held. It does not establish direction, mediation, a general factor, or an intervention effect. Use adjusted and single-skill columns as two different conditional descriptions, treat missingness indicators as nuisance offsets, and carry the Pareto-`k` limitation into any predictive comparison.
