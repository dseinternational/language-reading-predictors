<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Full rebuild of both model layers, 2026-10-01

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5.5).

This batch refitted all 277 registered Bayesian models and all 50 gradient-boosting models at the `reporting` configuration into a fresh run root. It then re-ran the evidence loop and the comparisons and re-rendered every report. It is the first full rebuild since the dependency update (#700) and the first full statistical sweep run on Windows; every earlier batch ran on macOS arm64. Nothing was published to the public research site.

**Outcome in one paragraph.** All 277 Bayesian fits pass the convergence gate with zero divergences, after one remediation (`mech-190`, below) that #701 declares in its module. All 277 record a clean working tree at the sweep commit. 271 fits are publishable and the same 6 as in earlier batches are withheld at the `inputs` stage. The Windows host needed a Numba CPU-target workaround before the larger models would compile. Despite the different platform and code generation, the available-case modified intention-to-treat (ITT) suite and the difference-in-differences (DiD) t2 contrasts reproduce the 2026-09-28 batch to three decimals. The boosting layer ran without failures, and its horseshoe comparison matches the 2026-09-28 overlaps exactly.

## Run record

| Item             | Value                                                                                               |
| ---------------- | --------------------------------------------------------------------------------------------------- |
| Run root         | `output/runs/20261001T175406Z-399586f4988d/`                                                        |
| Sweep commit     | `399586f4` (#700), from a dedicated clean worktree (`.claude/worktrees/refit-20261001`)             |
| Host             | Windows 11, Intel Raptor Lake (24 cores, 32 threads, 128 GB); Python 3.14, `uv sync --locked`       |
| Environment      | Full `pytest` suite green in the worktree before compute                                            |
| Data             | 9 files under `data/`, SHA-256 recorded in `run_metadata/run_manifest.json`                         |
| Statistical fits | 18:11–01:27 UTC, 7 h 16 m wall-clock; 275 of 277 finished by 23:05, the last two were `med-064/075` |
| Boosting fits    | 50/50 in 2 h 46 m, alongside the statistical fits                                                   |
| Fitted / failed  | 277 / 0 after the Numba workaround (5 first attempts failed to compile; see below)                  |
| Convergence      | 277/277 clean passes after one remediation; 0 divergences in the batch                              |
| Release          | 271 publishable, 6 withheld (`inputs`)                                                              |

The statistical models ran through the same claim-a-model work queue as on 2026-09-28. Each worker calls the checked-in resumable driver (`scripts/run_refit_sweep.py --models <id> --render`), so every report was rendered as soon as its fit finished. Five workers started. Once the compile failure appeared they were told to stop after their current fit, replacement workers with the workaround below took over, and the pool reached nine once the boosting sweep finished. The journal holds 283 records: 277 fits, the 5 failed first attempts and the `mech-190` remediation.

## Windows: a Numba compile failure for the host CPU

Within a minute of starting, five mediation fits (`med-059`, `068`, `074`, `076` and `078`) failed while nutpie compiled them:

```
error: <unknown>:0:0: ran out of registers during register allocation in function '...tensor_basic4join...'
```

Small models compile; the failures were models with about 20 or more free variables. #700 changed none of numba, llvmlite, pytensor or nutpie, so this is not a regression in that update. It is a code-generation problem on this host (numba 0.67, llvmlite 0.49). Numba CPU targets were tested on `med-059` at the `dev` configuration:

| `NUMBA_CPU_NAME` / setting | Result |
| -------------------------- | ------ |
| host (`raptorlake`)        | fails  |
| `haswell`, `x86-64-v3`     | fail   |
| `NUMBA_OPT=1`              | fails  |
| `generic`                  | works  |
| `x86-64-v2`, `sandybridge` | work   |
| `NUMBA_OPT=0`              | works  |

The batch used `NUMBA_CPU_NAME=sandybridge` (AVX, no AVX2 or FMA). `x86-64-v2` and `sandybridge` gave the same maximum R-hat to full precision on the test fit; `generic` differed only at floating-point noise level. The five failed models had written no output, so they were simply requeued. Four fits that had already compiled at the host target (`med-062`, `064`, `075` and `079`) were left to finish there; all passed. Fit provenance does not record the variable, so the setting is kept in `run_metadata/workaround-numba-cpu.md` and here. #701 documents the workaround in the agent instructions and the refit runbook.

A smaller host difference: a smoke-test render into a directory outside the git checkout failed at Quarto's `code-links: repo`, which needs a GitHub project. Run roots under the checkout render normally.

## Convergence-gate failure and its remediation

| Model      | Failure                                                                             | Remediation            | Result                                                          |
| ---------- | ----------------------------------------------------------------------------------- | ---------------------- | --------------------------------------------------------------- |
| `mech-190` | 1 divergence at the declared 0.999; R-hat 1.0012, ESS 3,403, per-chain BFMI ≥ 0.918 | `target_accept` 0.9995 | Clean pass: 0 divergences, R-hat 1.0011, ESS 2,632; publishable |

`mech-190` is the HSGP knee-test of phoneme blending → word reading. Its nonlinear-shape result needs zero divergences. It passed at 0.999 in earlier batches on macOS; here the different code generation sent the sampler on a different path through the same hard region near the curve's boundary. The remediation did not move the science: across `mechanism_curve.csv`, `mechanism_curve_items.csv` and `mechanism_summary.csv`, the largest change in any numeric column is 0.013 on the model scale, or 0.03 items. The refit used the driver's `--target-accept` override from the clean sweep commit. #701 declares the value in `lrp_rli_mech_190.py`, so a registry rebuild reproduces it. The failing directory is kept in `statistical_models/_backups/lrp-rli-mech-190-reporting.pre-remediation-20261002`.

The four remediations declared after earlier batches (`hs-002` and `hs-004` at 0.999, `mech-104` at 0.98, `jm-002` at 0.99) all held at the first attempt.

## Evidence loop

| Sweep                                     | Result                                                                                          |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Standard ITT treatment-prior sweep        | 44/44 cells converged; independent evaluator passes                                             |
| P/N floor `tau` grid                      | 12/12 cells converged; evaluator passes for `itt-009` and `itt-011`                             |
| Phoneme-blending link pair                | validated; evaluator passes; key findings bound to the bundle                                   |
| Influential-child refits                  | `itt-012`, `itt-013` and `itt-023` (one child each above the Pareto-k threshold of 0.7)         |
| DiD prior sweep                           | 21/21 cells converged, with no command-line remediation (#699 declared `did-007`'s cell target) |
| Gain-factor and level-factor prior sweeps | 6 and 15 cells, all converged and attached                                                      |
| Dispersion prior sweep                    | 12 cells, all converged                                                                         |
| Horseshoe prior sweep                     | 18/20 cells converged; both failures are on the withheld `rlm-hs-001`                           |
| Boosting rankings (`rank_predictors.py`)  | the 4 models the horseshoe comparisons read (`gbg-012`, `gbl-012`, `gbg-009`, `gbl-009`)        |

**`itt-012` needs an influential-child refit again.** On 2026-09-28 its largest Pareto-k was 0.63; here one child reaches 0.78, as on 2026-09-21. For all three models, leaving out the flagged child changes neither the direction nor the strength of evidence. On the probability scale the average marginal effect goes from 0.055 to 0.060 for taught receptive vocabulary in `itt-012` (P(effect > 0) 0.961 → 0.973), from 0.065 to 0.063 for taught expressive vocabulary (0.985 → 0.980), from 0.032 to 0.032 for `itt-013` (0.967 → 0.969) and from 0.111 to 0.131 for `itt-023` (0.999 → 1.000).

**`did-007` no longer needs a command-line remediation.** Since #699 the sweep adopts the declared cell target, and all three of its cells converged at the first attempt.

**The nonword floor result is prior-sensitive.** Across the six-cell grid, the median off-floor risk difference for `itt-011` (N) runs from +0.10 at treatment-prior SD 0.5 to +0.24 at SD 1.5. For `itt-009` (P) it runs from +0.04 to +0.11. The grid passes its computational gate; the movement is prior sensitivity that a report of either result must state.

**The horseshoe sweep lost one more cell on `rlm-hs-001`.** Its τ₀ 0.05 cell (1 divergence) and slab-scale 4.0 cell (2 divergences) failed, where on 2026-09-28 only the slab-scale cell did. The model is withheld at `inputs`, so this limits only its own cross-check. In every converged cell the top three predictors match the reference fit, Kendall's τ against the reference is at least 0.94, and no predictor's P(|β| > 0.1) moves by more than 0.07.

## Release state

Release decisions were re-evaluated from the stored artefacts with `release.evaluate_publication` after every evidence bundle was attached. 271 fits are publishable and 6 are withheld at the `inputs` stage: the same six as in every recent batch. 51 publishable fits carry a robustness note: DiD 12, ITT 13, joint 4, level factors 15 and gain factors 7. On 2026-09-28 the count was 50, with the joint fits counted under ITT (17), so the difference is one more level-factor fit. Key findings were regenerated for all 277 fits after the evidence loop. All 277 reports were then re-rendered with no failures and pass the runbook's freshness check. No page contains a traceback or import error, and exactly the six withheld fits display "Findings withheld".

## Comparisons and reproduction

`compare_statistical_models.py` wrote 25 artefacts. It repaired unreliable Pareto-k values by exact refits for six nested mechanism comparisons. It again declined the `did-007`/`did-107` dose comparison's `az.compare` deltas because their PSIS-LOO is unreliable (Pareto-k 1.10 and 1.15), and reports per-model expected log predictive density instead.

Three results reproduce the 2026-09-28 batch across the change of platform and Numba target:

1. **ITT suite and DiD t2 contrasts**: every row in the tables below matches to three decimals, apart from broad expressive vocabulary's P(effect > 0) (0.535 against 0.529).
2. **`med-059` total effect**: 2.3330 words (mean) and 2.3266 (median), against 2.3275 and 2.3217. The difference in means, 0.006 words, is within one Monte Carlo standard error (about 0.009 words).
3. **Horseshoe versus boosting**: the top-3 construct overlaps for `hs-001` to `hs-004` are 1/3, 1/3, 0/3 and 1/3, identical to 2026-09-28. `compare_gb_vs_statistical.py` gives Spearman correlations of −0.11 (gain, 7 shared constructs) and +0.60 (level, 5).

The sampling diagnostics could not be compared fit by fit with the 2026-09-28 batch because that run root is on another machine.

## Results

**Available-case modified ITT suite** (`itt-001`–`011`): average marginal effect on the probability scale, median with inner 50% and outer 89% equal-tailed intervals, and the probability of a positive effect.

| Model     | Outcome                             | Median | 50% interval     | 89% interval     | P(>0) | Evidence     |
| --------- | ----------------------------------- | ------ | ---------------- | ---------------- | ----- | ------------ |
| `itt-007` | Letter sounds (L)                   | +0.110 | [+0.086, +0.134] | [+0.053, +0.166] | 0.999 | very strong  |
| `itt-010` | Word reading (W)                    | +0.030 | [+0.021, +0.039] | [+0.009, +0.051] | 0.986 | strong       |
| `itt-002` | Taught expressive vocabulary (TE)   | +0.064 | [+0.045, +0.084] | [+0.018, +0.111] | 0.985 | strong       |
| `itt-008` | Phoneme blending (B)                | +0.099 | [+0.067, +0.131] | [+0.022, +0.174] | 0.980 | strong       |
| `itt-001` | Taught receptive vocabulary (TR)    | +0.057 | [+0.037, +0.077] | [+0.008, +0.105] | 0.968 | moderate     |
| `itt-003` | Untaught receptive vocabulary (UR)  | +0.050 | [+0.028, +0.072] | [−0.002, +0.103] | 0.937 | moderate     |
| `itt-011` | Nonword reading (N)                 | +0.100 | [+0.042, +0.158] | [−0.038, +0.237] | 0.877 | suggestive   |
| `itt-004` | Untaught expressive vocabulary (UE) | +0.026 | [+0.003, +0.049] | [−0.029, +0.080] | 0.773 | suggestive   |
| `itt-009` | Phonics, floored (P)                | +0.041 | [−0.005, +0.088] | [−0.071, +0.155] | 0.724 | inconclusive |
| `itt-005` | Broad receptive vocabulary (R)      | +0.001 | [−0.008, +0.011] | [−0.022, +0.025] | 0.539 | inconclusive |
| `itt-006` | Broad expressive vocabulary (E)     | +0.001 | [−0.006, +0.007] | [−0.014, +0.016] | 0.535 | inconclusive |

A causal reading of these rows rests on the suite's stated randomisation and available-case assumptions. The letter-sound row, for example, says that under this model and prior there is an 89% probability that the intervention raises the proportion of letter sounds correct by between 5 and 17 percentage points. The broad standardised vocabulary rows (R and E) are inconclusive: their intervals are centred near zero but are still wide enough to include effects that would matter, so they are not evidence of no effect.

**Waitlist-crossover DiD**, `tau_t2` (the randomisation-anchored t2 arm gap), in items.

| Model     | Outcome              | Median | 50% interval   | 89% interval   | P(>0) | Evidence    |
| --------- | -------------------- | ------ | -------------- | -------------- | ----- | ----------- |
| `did-002` | Letter sounds (L)    | +3.53  | [+2.54, +4.50] | [+1.18, +5.81] | 0.991 | very strong |
| `did-001` | Word reading (W)     | +2.22  | [+1.18, +3.27] | [−0.31, +4.69] | 0.920 | moderate    |
| `did-003` | Phoneme blending (B) | +0.88  | [+0.54, +1.22] | [+0.06, +1.69] | 0.956 | moderate    |

These reuse the randomised t2 information under a different specification and are not independent replications of the ITT estimates. The period-1-standardised gain-factor contrasts point the same way (word reading +2.6 items, 89% interval +0.9 to +4.3; letter sounds +3.3, +1.6 to +5.0; phoneme blending +0.8, +0.1 to +1.6), as do the level-factor t2 changes (word reading +2.3, +0.3 to +4.4; letter sounds +2.8, +0.8 to +4.9).

**Mediation, `med-059`** (word reading via letter-sound knowledge): the model-based total effect is +2.33 words (89% interval +0.13 to +4.55; P(> 0) 0.954), the natural indirect effect +2.09 words (+0.84 to +3.71; P 0.998) and the natural direct effect +0.15 words (−1.81 to +2.18). This decomposition is model-based and not identified under the study's unmeasured confounding.

**Gradient boosting.** Held-out pooled R² for the 22 gain models runs from 0.06 to 0.31. In most of them the own baseline is the leading predictor, with a negative SHAP direction, as expected from score limits and regression to the mean. The 28 level models reach 0.22 to 0.998, mostly through measures of the same skill or the components of a composite; the highest, `gbl-019` (0.997), predicts the ERB total from its word-repetition component. Word-reading gain (`gbg-012`, R² 0.13) is led by age (negative SHAP direction), then hearing, phoneme blending and expressive vocabulary (positive). These are predictive associations from internal validation on the folds used for tuning.

## Open residuals

1. **The Numba CPU-target workaround** is not captured in fit provenance. A Windows fit cannot show which target compiled it without this note or the run's metadata.
2. **14 fits record no `data_sha256`**: `surv-009/011`, the six `rlm-adj-*`, `rlm-ca-001/002`, `rlm-hs-001/002/003` and `rlm-mm-001`. The same set as the last four batches.
3. **The 6 withheld fits** are the Byrne/RLM ports blocked at `inputs` on unconfirmed `basspel`/`woco`/`basnum` score denominators (#338): `rlm-adj-001`, `rlm-hg-002`, `rlm-hg-003`, `rlm-hg-008`, `rlm-hs-001`, `rlm-mm-001`.
4. **The ERB source-archive question** (#631) is still open, so fits use the quarantined values.
5. **Two horseshoe prior-sensitivity cells** on the withheld `rlm-hs-001` have divergences at the model's 0.99.
6. **The boosting layer records no fit-time provenance**, so a boosting fit cannot show which commit produced it without the sweep journal (`_sweep/journal-gb-reporting.jsonl`).
