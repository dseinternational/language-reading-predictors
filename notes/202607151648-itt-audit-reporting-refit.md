<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# ITT statistical audit closure and reporting refit

Date: 2026-07-15

Related: issue #341, PR #344, `METHODS.md`, `docs/models/PRIORS.md`, `docs/runbooks/full-statistical-model-refit.md`, and `notes/202607131900-attrition-audit.md`.

## Scope and decision

This record covers the production-config refit and post-fit audit of all 31 registered ITT-family models: `lrp-rli-itt-001` through `lrp-rli-itt-028`, plus `lrp-rli-itt-113`, `114`, and `115`. It also covers the 44-fit standard treatment-prior and likelihood-prior sensitivity sweep, the 12-fit floor-specific treatment-prior sweep, posterior-predictive calibration, Pareto-k influence diagnostics, three direct leave-one-child-out treatment-effect checks, cross-model ITT comparisons, and all 31 rendered model reports.

**Decision:** the implementation findings in issue #341 are addressed sufficiently to close the code audit and proceed with review of PR #344. This is not a declaration that every scientific estimand is confirmatory or publication-ready: the available-case population, post-hoc floor rule, prior sensitivity of the floor effects, joint-model posterior-predictive flags, and unreliable PSIS-LOO approximations remain substantive limitations and are now made explicit in the artefacts and reports.

All artefacts referenced below remain local under `output/statistical_models/`. Nothing from this run was uploaded or published.

## Audited code, data, and artefact identity

The implementation reviewed here is commit `0cd8e54221b54b135e11c5c82229f19401ba334f` on branch `dev/codex/itt-audit-fixes`. The input file `data/rli_data_long.csv` had SHA-256 `dc8dda5780b705e902155372c135a993778506c547ef8ebb2b5b03668c11f043`.

The primary-fit output root was reused during the audit and predates the new fresh-run manifest requirement. Its primary traces, configurations, counts, data digest, convergence summaries, and downstream sensitivity bindings were revalidated against the implementation above, and the checksums below identify the exact local snapshot inspected. They do not substitute for the fresh versioned output root and completed run manifest required by `docs/runbooks/full-statistical-model-refit.md` before any future external publication or upload.

| Audited artefact                                                                                    | SHA-256                                                            |
| --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Standard 44-cell sensitivity manifest, `tau_prior_sensitivity/tau_prior_sensitivity.csv`            | `256ef2029194073f20438016d1dd0dac397b40accabe66892fee1bf05b9d8cc8` |
| Central 12-cell floor sensitivity manifest, `floor_tau_prior_sensitivity/tau_prior_sensitivity.csv` | `a017ee69fdf82b8076844b5b3b5cea70a1ad91892dcbf1bf3096a06de90afcdd` |
| P report-local floor sensitivity manifest                                                           | `7081150987d76d0ab2a53ee5fa755c3a734fb70ad653d471c88c91e7e8be6db8` |
| N report-local floor sensitivity manifest                                                           | `cc1185e33445c5143a29e765af283cd00d3ad02cb90d72c36327faba2d578427` |
| ITT012 influence manifest                                                                           | `4d5e13e3c77c956feb6366f2460faa1982e114bdf097c9c00d1136f55e534e3`  |
| ITT013 influence manifest                                                                           | `3245ecf78312bc1faf1d15c99d3918a699f585b9b99da7cf997b74b35f2f56bf` |
| ITT023 influence manifest                                                                           | `dee8397172748226db1623924a6d84f5d63e401cb72cff2f0861155576c0e133` |
| ITT012 posterior-predictive shape calibration                                                       | `07ab157278012a6bf808064a0a491888762c3da5da4ef7967c516ce5ea89037a` |
| ITT012 prior-versus-posterior diagnostic PNG                                                        | `a6fd4e891a6514d45687fce2ea4f4dbfe6f633bb585ebc37301dfad33cfbb447` |

## Sampling and convergence

The reporting preset used six chains, 6000 warm-up iterations and 6000 retained draws per chain, with `target_accept = 0.95`. Every one of the 31 primary fits passed the strict gate of R-hat ≤ 1.01, effective sample size ≥ 400, BFMI ≥ 0.30, and zero divergences. Across the suite, the worst max R-hat was 1.00082, the lowest effective sample size was 6525, the lowest chain BFMI was 0.945, and the total divergence count was zero. The floor-rule graded and post-selected secondary fits also passed their separate all-free-parameter convergence gates, but convergence does not remove their estimand limitations.

## Headline estimates after the corrected refit

The table reports posterior medians, equal-tailed 95% credible intervals, and the posterior probability that the named effect is beneficial. For W and L, probability-scale average marginal effects are translated to their confirmed item denominators. P and N are risk differences among children observed at zero before randomisation, not graded all-child effects.

| Model             | Outcome or contrast                            |        Posterior median |             95% credible interval | P(benefit or positive contrast) | Reading                                                                                                                                                      |
| ----------------- | ---------------------------------------------- | ----------------------: | --------------------------------: | ------------------------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `lrp-rli-itt-010` | Word reading W                                 |       +2.37 of 79 items |              +0.28 to +4.48 items |                           0.986 | Strong evidence of benefit in the fitted available-case population.                                                                                          |
| `lrp-rli-itt-007` | Letter-sound knowledge L                       |       +3.52 of 32 items |              +1.27 to +5.74 items |                           0.999 | Very strong evidence of benefit in the fitted available-case population.                                                                                     |
| `lrp-rli-itt-009` | Phonetic spelling P, off-floor risk difference |  +4.1 percentage points |   -9.8 to +18.3 percentage points |                           0.724 | Inconclusive post-hoc exploratory subgroup estimate.                                                                                                         |
| `lrp-rli-itt-011` | Nonword reading N, off-floor risk difference   | +10.0 percentage points |   -7.1 to +26.8 percentage points |                           0.877 | Suggestive but interval-crossing post-hoc exploratory subgroup estimate.                                                                                     |
| `lrp-rli-itt-016` | Taught expressive vocabulary TE                | +6.41 percentage points | +0.68 to +12.13 percentage points |                           0.986 | Strong evidence that the intervention raised taught expressive performance.                                                                                  |
| `lrp-rli-itt-016` | Taught receptive vocabulary TR                 | +5.64 percentage points | -0.43 to +11.61 percentage points |                           0.966 | Direction favours benefit, but the 95% interval includes zero.                                                                                               |
| `lrp-rli-itt-016` | TE minus TR modality contrast                  | +0.77 percentage points |  -7.51 to +9.14 percentage points |                           0.572 | No evidence that the expressive and receptive taught-word effects differ. This is a modality contrast, not a taught-versus-untaught generalisation contrast. |

The ITT016 model factorises the two outcomes and does not estimate their within-child residual covariance. Its between-modality interval therefore omits that covariance and needs a paired child-level randomisation, bootstrap, sandwich, or dependence-model sensitivity before being treated as definitive.

## Prior-sensitivity results

### Standard graded-outcome sweep: 44 of 44 converged

The standard sweep in `output/statistical_models/tau_prior_sensitivity/tau_prior_sensitivity.csv` comprises 30 treatment-prior fits across R, E, UR, UE, T, F, L, and W; four own-baseline-prior fits for L/W; two unadjusted randomised-arm benchmarks; and eight Beta-Binomial concentration-prior fits for L/W. All 44 passed the all-free-parameter convergence gate.

- R and E remained near zero across treatment-prior SDs 0.20 to 0.50, with P(benefit) approximately 0.54 throughout.
- The broader-transfer outcomes were directionally more favourable as the treatment prior widened, but every 95% interval still crossed zero: UR P(benefit) 0.905 to 0.957, UE 0.740 to 0.794, T 0.746 to 0.772, and F 0.851 to 0.925. The evidence labels are therefore somewhat prior-sensitive even though no interval-supported positive effect appears.
- L remained positive across treatment-prior SDs 0.25, 0.50, and 0.75, with item-scale estimates +2.47 to +3.81 and P(benefit) 0.993 to 0.999.
- W remained positive in direction across the same grid, with item-scale estimates +1.83 to +2.51 and P(benefit) 0.974 to 0.988; at SD 0.25 the 95% lower endpoint was -0.01 items, so the interval-based reading is slightly sensitive to the narrowest prior.
- W/L estimates changed negligibly when the own-baseline prior SD moved from 0.25 to 0.50 or the concentration-prior scale moved from 25 to 200. Removing baseline and age precision terms widened uncertainty materially for W: +2.75 items, 95% interval -1.63 to +7.28, P(benefit) 0.892. The analogous unadjusted L estimate remained positive: +4.05 items, 95% interval +0.34 to +7.63, P(benefit) 0.983.

### Floor-specific sweep: 12 of 12 converged

Because power scaling diagnosed possible prior-data conflict for `tau` in both P and N, the release gate uses an estimand-matched 3 × 2 grid: treatment-prior SD 0.5, 1.0, and 1.5 crossed with linear age adjustment off/on. Every fit uses the observed-baseline-floor subgroup and Bernoulli off-floor likelihood, gates all free parameters, and persists its trace. All 12 fits passed, and their per-model CSVs and traces are beside the reporting outputs.

| Outcome |    Median risk-difference range | Envelope of the six 95% intervals | P(benefit) range | P(risk difference ≥ 10 percentage points) range | Interpretation                                                                                                                                                              |
| ------- | ------------------------------: | --------------------------------: | ---------------: | ----------------------------------------------: | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P       | +4.0 to +10.5 percentage points |  -12.2 to +32.0 percentage points |   0.718 to 0.831 |                                  0.197 to 0.519 | Age adjustment has little effect within a prior SD, but widening the treatment prior materially raises the estimate; every 95% interval includes zero.                      |
| N       | +9.9 to +24.4 percentage points |   -7.3 to +48.5 percentage points |   0.876 to 0.968 |                                  0.496 to 0.868 | Age adjustment again has little effect within a prior SD, while the wider prior materially raises the effect and directional probability; every 95% interval includes zero. |

The completed grid clears the computational release gate raised by the `psense` conflict, but it also demonstrates that the P/N magnitudes and evidence labels are prior-sensitive. These outcomes remain post-hoc exploratory subgroup analyses, not confirmatory trial primaries.

## Posterior-predictive audit

The corrected posterior-predictive code uses inclusive lower and upper tail areas and a two-sided tail, avoiding the earlier exact-tie error that could manufacture a zero lower tail. The P and N graded secondaries reproduced their aggregate zero proportions without a flag: two-sided tail areas were 0.833 and 0.971 respectively. Those checks assess only the aggregate zero rate and do not validate item exchangeability or make the binary floor rule prospective.

The full joint model `lrp-rli-itt-012` nevertheless failed two dataset-level shape checks. For E, the observed interquartile range was 11 items versus a posterior-predictive 95% range of 12 to 23, two-sided tail area 0.0328. For graded P, the observed upper quartile and interquartile range were both 27 items versus posterior-predictive 95% ranges of 0 to 22, two-sided tail area 0.0208. These flags are localised to the joint graded likelihood and prevent a blanket claim that all outcome shapes are well reproduced; they do not directly invalidate the separate Bernoulli off-floor P headline.

## Pareto-k and direct leave-one-child-out treatment-effect checks

Three reporting fits had a maximum Pareto-k above 0.70, so their PSIS-LOO approximations are unreliable: `lrp-rli-itt-012` k = 0.853, `lrp-rli-itt-013` k = 0.723, and `lrp-rli-itt-023` k = 0.708. These LOO values must not be used for model ranking without exact or moment-matched leave-one-out, or K-fold, validation of the same predictive target.

The audited reporting-config direct refits excluded the child associated with each flag, and all three passed the convergence gate. The comparison decomposes the total full-sample-to-leave-out change into (a) a **refit shift**, comparing the full and leave-out posteriors over the same retained children, and (b) a **composition shift**, caused only by removing the flagged child's covariate profile from the averaging population. For `lrp-rli-itt-012`, the largest change was for L: refit +2.19 percentage points, composition -0.06 points, total +2.13 points. For SES-adjusted W in `lrp-rli-itt-013`, the corresponding changes were refit +0.13 points, composition -0.17 points, total -0.04 points. For ability-adjusted L in `lrp-rli-itt-023`, they were refit +2.08 points, composition -0.06 points, total +2.02 points. The reports lead with the refit shift because it isolates sensitivity to refitting from the change in averaging population. No broad direction or evidence conclusion reversed.

The audit artefacts are `output/statistical_models/influence_sensitivity/lrp-rli-itt-012-reporting/influence_sensitivity.csv`, `.../lrp-rli-itt-013-reporting/influence_sensitivity.csv`, and `.../lrp-rli-itt-023-reporting/influence_sensitivity.csv`, with copies beside the corresponding reporting outputs. These influence refits test treatment-estimate stability; they do not repair the PSIS-LOO approximation itself.

The checked-in `scripts/influence_sensitivity.py` runner reproduced these results. It verifies the registered specification, saved data checksum and Pareto point-to-child mapping; excludes every child above the saved threshold; reuses the completed fit's sampling settings; gates all free variables; and persists both the trace and a provenance-rich comparison. These refits test treatment-effect stability; they do not recompute an exact LOO expected log predictive density and therefore do not repair the unreliable PSIS-LOO scores.

## Methodological limits retained after closure

- The published trial randomised 57 children, 29 immediate and 28 wait-list, while the repository contains 54. The models are available-case or modified-ITT randomised comparisons, not full-randomised-cohort ITT analyses; extending estimates to the missing children requires an untestable missingness assumption or recovered data. Trial counts are documented by Burgoyne et al. (2012), DOI [10.1111/j.1469-7610.2012.02557.x](https://doi.org/10.1111/j.1469-7610.2012.02557.x).
- P and N condition on an observed pre-randomisation baseline floor score, leaving 41 and 36 fitted children respectively. This preserves the randomised arm contrast within those observed subgroups, but does not identify effects for children with unknown eligibility, missing t2 outcomes, or absence from the archive.
- The 40% floor rule was chosen after examining this trial's post-treatment distribution and earlier results. Arm-blind application reduces treatment-label-driven selection but does not make the rule pre-specified; P/N must remain visibly labelled post-hoc and exploratory.
- The P/N graded all-child fits are detection-limited secondaries. Their post-selected graded contrasts among children observed off-floor at t2 condition on a post-randomisation event and are adjusted associations, not randomised causal effects.
- Baseline and age terms in the graded ITT models are precision terms; the randomised treatment effect is identified by the empty adjustment set. Cross-baselines remain excluded to avoid unnecessary postulated adjustment and unstable interpretation.
- Thirty-one related models and many outcome contrasts create ample scope for selective emphasis. Posterior probabilities and evidence labels are descriptive summaries, not protection against multiplicity or undisclosed outcome selection; interpretation must report the full pattern and uncertainty.
- The joint E/P posterior-predictive flags, the three Pareto-k failures, and ITT016's omitted within-child covariance remain active cautions even though the sampler convergence gates pass.

## Reproduction commands

The following are durable repository commands. The local run needed an explicit worktree `PYTHONPATH` because the active environment's editable install pointed to another checkout; a correctly installed editable environment does not need that prefix.

```bash
for n in {001..028} 113 114 115; do
  python scripts/fit_statistical_model.py "lrp-rli-itt-${n}" --config reporting --output-dir output
done

python scripts/tau_prior_sensitivity.py --config reporting --output-dir output --outcomes R E UR UE T F L W
python scripts/tau_prior_sensitivity.py --config reporting --output-dir output --out-dir output/statistical_models/floor_tau_prior_sensitivity --outcomes P N
python scripts/influence_sensitivity.py lrp-rli-itt-012 lrp-rli-itt-013 lrp-rli-itt-023 --config reporting --output-dir output

for n in {001..028} 113 114 115; do
  (
    cd "output/statistical_models/models/lrp-rli-itt-${n}-reporting"
    quarto render index.qmd
  )
done

pytest
ruff check src/ scripts/ tests/
npm run format:check
npm run spellcheck
```

The reporting run produced 31 of 31 rendered `index.html` files. A final scan found all HTML outputs newer than their 837 copied QMD inputs, all 806 copied partials byte-identical to source, and all 1058 local HTML resources present, including 624 images. The full Python suite passed all 744 tests; Ruff passed for `src/`, `scripts/`, and `tests/`; Markdown formatting, spellcheck over 322 files, and `git diff --check` also passed. The local output directories are gitignored and must be regenerated into a fresh versioned root before release; this note records the audited snapshot but does not publish or upload it.
