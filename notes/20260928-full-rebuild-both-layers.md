<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Full rebuild of both model layers, 2026-09-28

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5.5).

This batch refitted all 277 registered Bayesian models and all 50 gradient-boosting models at the `reporting` configuration into a fresh run root, re-ran the evidence loop and the comparisons, and re-rendered every report. It is the first full rebuild since the Huber retune of the boosting layer (#690), the statistical validation and reporting corrections (#691), the Noto Sans font changes (#694, #696) and the registration of `lcsm-167` (#683), which takes the statistical registry from 276 to 277 models. Nothing was published to the public research site.

**Outcome in one paragraph.** All 277 Bayesian fits pass the convergence gate with zero divergences, after one remediation (`jm-002`, below) that is now declared in its module. All 277 record a clean working tree: 276 at the sweep commit and `jm-002` at its remediation commit. 271 fits are publishable and the same 6 as in earlier batches are withheld at the `inputs` stage. The available-case modified intention-to-treat (ITT) suite and the difference-in-differences (DiD) t2 contrasts reproduce the 2026-09-21 batch to three decimals. The boosting layer needed a code fix before it could run at `reporting` (#697). Because the boosting models changed in #690 and #691, the horseshoe-versus-boosting ranking comparison changed too; that is expected, not a reproduction failure.

## Run record

| Item               | Value                                                                                       |
| ------------------ | ------------------------------------------------------------------------------------------- |
| Run root           | `output/runs/20260928T193136Z-3119b55c5c24/`                                                |
| Sweep commit       | `3119b55c` (#696), from a dedicated clean worktree                                          |
| Boosting commit    | `b811fd9c`, the branch later merged as #697                                                 |
| Remediation commit | `0f1ca446` on `fix/remediate-jm-002-target-accept`                                          |
| Environment        | `uv sync` unchanged; full `pytest` suite green before compute (see the compiler note below) |
| Data               | 9 files under `data/`, SHA-256 recorded in `run_metadata/run_manifest.json`                 |
| Statistical fits   | 20:35–23:34 BST, 2 h 59 m wall-clock (4 h 20 m on 2026-09-21), then `jm-002` in 38 min      |
| Boosting fits      | 50/50 in 1 h 28 m, alongside the statistical fits                                           |
| Fitted / failed    | 277 / 0 process failures; 1 convergence-gate failure, remediated                            |
| Convergence        | 277/277 clean passes; 0 divergences in the batch                                            |
| Release            | 271 publishable, 6 withheld (`inputs`)                                                      |

The sweep ran from its own worktree (`.claude/worktrees/refit-20260928`), as the 2026-09-21 note recommended, so no other session could dirty it. The macOS 27 compiler workaround described in the 2026-09-21 note ([pymc-devs/pytensor#2268](https://github.com/pymc-devs/pytensor/issues/2268)) was still needed; it lives in `run_metadata/pytensorrc` and `run_metadata/bin/`. The first test run omitted it: seven tests that compile PyTensor C code failed with `library 'd64' not found` and every other test passed. Re-running the affected test files with the workaround in place passed.

## A dynamic queue instead of static streams

The last two batches split the statistical models into fixed streams balanced by each model's previous runtime, then stopped and relaunched drivers by hand when streams finished unevenly. This batch replaced that with a work queue. `run_metadata/queue-statistical.txt` lists every model, most expensive first. Each worker (`run_metadata/queue_worker.sh`) claims the next unclaimed model by creating a directory for it, which is atomic, and runs the checked-in resumable driver (`scripts/run_refit_sweep.py --models <id>`) for that one model. Workers can join at any time without coordination. The batch started with three and grew to seven once the long mediation fits reached their single-threaded g-formula and tipping-analysis phase and left cores idle.

Because each worker calls the driver, every fit has a journal entry, unlike the orphaned fits that the stop-and-relaunch method left behind. The fit phase took 2 h 59 m. The two longest fits set the floor: `med-075` took 2 h 44 m and `med-064` 2 h 23 m (about 95–130 min on 2026-09-08). Both started first and finished before the other workers ran out of models.

## Boosting layer: a failure the `dev` configuration cannot catch

The first boosting sweep, from the sweep commit, failed on its first model. #691 made permutation importance missing, rather than zero, for predictors that the schedule-matched permutation design cannot change, such as `time`. The feature-selection diagnostics step still cast the importance rank to integers, which fails on a missing value. `dev` fits skip that step, so neither the test suite nor the development fits used to check #691 reached it. The fix keeps unassessable predictors in the pairing (#697). All 50 boosting models were then fitted from that branch, with no failures. The boosting layer writes no fit-time provenance of its own, so the branch commit is recorded only in the sweep log (`run_metadata/gb-sweep.log`) and here.

## Convergence-gate failure and its remediation

One fit failed the gate:

| Model    | Failure                                                                                                                                     | Remediation                  | Result                                 |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- | -------------------------------------- |
| `jm-002` | Bulk/tail ESS 399.65 against 400 on one per-child latent (`u_child_z[33, N]`) at the 0.95 preset; 0 divergences, R-hat 1.0096, BFMI ≥ 0.885 | declare `target_accept` 0.99 | Clean pass, 0 divergences; publishable |

This is the standing precedent for a marginal ESS or R-hat on per-child latents. `jm-002` passed at 0.95 on 2026-09-21, so the failure reflects how close that one latent sits to the threshold rather than a change in the model. The failing directory is kept in `statistical_models/_backups/lrp-rli-jm-002-reporting.pre-remediation-20260928`. Following the 2026-09-21 lesson that command-line remediations do not survive a registry rebuild, the new acceptance target is declared in `lrp_rli_jm_002.py`, and the refit ran from that commit with a clean tree.

The three remediations declared after the 2026-09-21 batch (`hs-002` and `hs-004` at 0.999, `mech-104` at 0.98) held: all three passed at the first attempt.

## Evidence loop

| Sweep                                     | Result                                                                                          |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Standard ITT treatment-prior sweep        | complete (`tau_prior_sensitivity.py`, eight outcomes)                                           |
| P/N floor `tau` grid                      | complete                                                                                        |
| Phoneme-blending link pair                | complete                                                                                        |
| Influential-child refits                  | `itt-013` and `itt-023` (one child each above the Pareto-k threshold of 0.7)                    |
| DiD prior sweep                           | 21/21 cells converged after the standing `did-007` remediation (below)                          |
| Gain-factor and level-factor prior sweeps | complete and attached                                                                           |
| Dispersion prior sweep                    | complete                                                                                        |
| Horseshoe prior sweep                     | 19/20 cells converged; the one failure is withheld `rlm-hs-001`, the same cell as on 2026-09-21 |
| Boosting rankings (`rank_predictors.py`)  | the 11 models whose comparisons need them                                                       |

**`itt-012` no longer needs an influential-child refit.** Its largest Pareto-k is 0.63, below the 0.7 threshold, where on 2026-09-21 one child exceeded it. For the other two, leaving out the flagged child changes neither the direction nor the strength of evidence. On the probability scale the average marginal effect goes from 0.032 to 0.032 for `itt-013` (P(effect > 0) 0.967 → 0.969) and from 0.111 to 0.131 for `itt-023` (0.999 → 1.000), the same figures as on 2026-09-21.

**`did-007` needed its cell remediation for the fourth batch running.** The `mu_dose` cell at treatment-prior scale 1.5 again had 6 divergences at the default acceptance target. Refitting the model's cells with `--cell-target-accept 0.99` cleared all three, and the swept `tau_logit_mean` at that scale is +0.133, as on 2026-09-21. This fix still lives only on the command line.

## Release state

Release decisions were re-evaluated from the stored artefacts with `release.evaluate_publication` after every evidence bundle was attached, rather than read from the `release_decision.json` written at fit time. 271 fits are publishable, one more than on 2026-09-21 because of the new `lcsm-167`. The 6 withheld fits are the same as before, all at the `inputs` stage. 50 publishable fits carry a robustness note (DiD 12, ITT 17, level factors 14, gain factors 7), the same count and split as the last two batches, and none is left over from the order in which fits finished. Key findings were regenerated for all 277 fits after the evidence loop, and all 277 reports were then re-rendered with no failures.

## Comparisons and reproduction

`compare_statistical_models.py` wrote its comparison tables. It again declined the `did-007`/`did-107` dose comparison's `az.compare` deltas because their PSIS-LOO is unreliable (Pareto-k 1.17 and 1.15), and reports per-model expected log predictive density instead.

Three results from earlier batches reproduce:

1. **ITT suite and DiD t2 contrasts**: every row in the tables below matches the 2026-09-21 batch to three decimals (DiD to two, in items).
2. **`med-059` total effect**: 2.3275 words (mean) and 2.3217 (median), against 2.3199 and 2.3153. The difference in means, 0.008 words, is within one Monte Carlo standard error (about 0.009 words).
3. **The L × N nested LOO** is again `comparison_valid=True` via `psis+reloo` with one exact refit per model, verdict inconclusive (|elpd_diff| < 4).

The sampling diagnostics could not be compared fit by fit with the 2026-09-21 batch because that run root is no longer on this machine.

**The horseshoe-versus-boosting comparison changed.** The top-3 overlaps between the horseshoe and boosting rankings are now 1/3, 1/3, 0/3 and 1/3 for `hs-001` to `hs-004`, against 2/3, 2/3, 1/3 and 2/3 on 2026-09-08 (the 2026-09-21 batch did not refit the boosting layer). The boosting side of the comparison has changed since then: #690 changed the boosting objective from MAE to Huber and re-tuned every model, and #691 changed how permutation importance is assessed. This batch does not separate how much of the change comes from each layer. Any narrative comparison of the two layers written before #690 needs checking against these fits.

## Results

The same suite re-estimated; recorded so the batch has a readable headline and the next rebuild has something to check against.

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
| `itt-006` | Broad expressive vocabulary (E)     | +0.001 | [−0.006, +0.007] | [−0.014, +0.016] | 0.529 | inconclusive |

A causal reading of these rows rests on the suite's stated randomisation and available-case assumptions. The letter-sound row, for example, says that under this model and prior there is an 89% probability that the intervention raises the proportion of letter sounds correct by between 5 and 17 percentage points. The broad standardised vocabulary rows (R and E) are inconclusive: their intervals are centred near zero but are still wide enough to include effects that would matter, so they are not evidence of no effect.

**Waitlist-crossover DiD**, `tau_t2` (the randomisation-anchored t2 arm gap), in items.

| Model     | Outcome              | Median | 50% interval   | 89% interval   | P(>0) | Evidence    |
| --------- | -------------------- | ------ | -------------- | -------------- | ----- | ----------- |
| `did-002` | Letter sounds (L)    | +3.53  | [+2.54, +4.50] | [+1.18, +5.81] | 0.991 | very strong |
| `did-001` | Word reading (W)     | +2.22  | [+1.18, +3.27] | [−0.31, +4.69] | 0.920 | moderate    |
| `did-003` | Phoneme blending (B) | +0.88  | [+0.54, +1.22] | [+0.06, +1.69] | 0.956 | moderate    |

These reuse the randomised t2 information under a different specification and are not independent replications of the ITT estimates.

## Open residuals

1. **14 fits record no `data_sha256`**: `surv-009/011`, the six `rlm-adj-*`, `rlm-ca-001/002`, `rlm-hs-001/002/003` and `rlm-mm-001`. The same set as the last three batches.
2. **The 6 withheld fits** are the Byrne/RLM ports blocked at `inputs` on unconfirmed `basspel`/`woco`/`basnum` score denominators (#338): `rlm-adj-001`, `rlm-hg-002`, `rlm-hg-003`, `rlm-hg-008`, `rlm-hs-001`, `rlm-mm-001`.
3. **The ERB source-archive question** (#631) is still open, so fits use the quarantined values.
4. **`did-007`'s cell remediation** recurs every batch and lives only on the command line.
5. **One horseshoe prior-sensitivity cell** (`rlm-hs-001`, τ₀ 0.1, slab 4.0) has 3 divergences at the model's 0.99; the model is withheld, so this limits only its own cross-check.
6. **PyTensor on macOS 27** still needs the run-local compiler wrapper, which no fit's provenance records.
7. **The boosting layer records no fit-time provenance**, so a boosting fit cannot show which commit produced it without the sweep log.
