<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Full statistical-model refit, 2026-09-21

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

This batch refitted all 276 registered Bayesian models at the `reporting` configuration (6 chains × 6,000 draws after 6,000 tuning steps) into a fresh run root, rendered every report and re-ran the evidence loop. It covers the statistical layer only; the gradient-boosting layer was not refitted, so the comparisons that need it (`rank_predictors.py`, `compare_horseshoe_vs_gb.py`, `compare_gb_vs_statistical.py`) were not run. Nothing was published to the public research site.

**Outcome in one paragraph.** All 276 fits pass the convergence gate with zero divergences anywhere in the batch. That took three single-model remediations, each strictly above the model's previous acceptance target and now declared in its module. 270 fits are publishable and 6 are withheld at the `inputs` stage, the same six as the last two batches. The available-case modified intention-to-treat (ITT) suite and the waitlist-crossover difference-in-differences contrasts reproduce the 2026-09-08 batch to the third decimal, and 245 of 276 fits reproduce that batch's sampling diagnostics to full floating-point precision. The one blemish is provenance: another session edited this checkout during the sweep, so 268 fits record `dirty: true`. A model-design identity check shows that none of those fitted models differs from the clean commit.

## Run record

| Item               | Value                                                                                              |
| ------------------ | -------------------------------------------------------------------------------------------------- |
| Run root           | `output/runs/20260921T084416Z-b22ea254c1f0/`                                                       |
| Sweep commit       | `b22ea254` (#679), clean at launch                                                                 |
| Remediation commit | `083339c5` on `fix/remediate-target-accept-20260921`                                               |
| Environment        | `uv sync --locked` unchanged; full `pytest` suite green before compute                             |
| Data               | 9 files under `data/`, SHA-256 recorded in `run_metadata/run_manifest.json`                        |
| Fit phase          | 10:12–14:32 BST, 4 h 20 m wall-clock (8.6 h on 2026-09-08, which ran the boosting layer alongside) |
| Fitted / failed    | 276 / 0 process failures; 3 convergence-gate failures, all remediated                              |
| Convergence        | 276/276 clean passes; 0 divergences in the batch                                                   |
| Release            | 270 publishable, 6 withheld (`inputs`)                                                             |

The sweep used the resumable driver (`scripts/run_refit_sweep.py`) in three streams, split by each model's measured runtime from the 2026-09-08 journal rather than by family count, as that batch's note recommended. Each stream started with its most expensive fits (`med-064`, `med-075`, `jc-002`). About 30 minutes in, all three had reached their single-threaded post-sampling phase (g-formula integration and tipping analysis for the two mediation fits, new-child predictive sampling for `jc-002`) and load had fallen to about 4 on 16 cores. The drivers were stopped without stopping their running fits, which completed, passed their gates and rendered on their own. Three fresh drivers took the other 273 models. As recorded last time, this leaves those three fits without journal entries; the fit directories, not the journal, are the completeness authority.

## macOS 27 broke PyTensor's C compiler

This Mac moved from macOS 26.6.2 to 27.0 after the last batch. PyTensor 3.3.2, the current release, adds `-ld64` to its compiler flags on any macOS from 15 upwards. With Apple clang 21 the flag is parsed as "link library `d64`" and every C compilation fails, so no fit can start. Nutpie does not avoid it: nutpie samples through Numba, but it builds its initial-point function through PyMC's C-backed virtual machine, and PyMC's log-likelihood and predictive sampling use the C backend too. This is upstream issue [pymc-devs/pytensor#2268](https://github.com/pymc-devs/pytensor/issues/2268), open and unfixed on `main`.

The batch used a run-local workaround: a wrapper named `clang++` that drops exactly the `-ld64` argument and passes everything else to `/usr/bin/clang++`, selected through a `PYTENSORRC` file in `run_metadata/`. The name matters: PyTensor disables its architecture-specific optimisation flags when the configured compiler's path does not contain `clang++`. No repository file or locked dependency changed, and on this toolchain the flag has no target anyway (`-ld_classic` is now ignored with a warning). **This environment setting is not recorded in any fit's provenance**, so it is recorded here. The alternative, `cxx=""`, works end to end but runs every non-sampling PyTensor function in pure Python.

## Convergence-gate failures and their remediation

Three fits failed the gate, each with a single divergent transition in 36,000 draws and every other check (R-hat ≤ 1.01, bulk and tail ESS ≥ 400, BFMI ≥ 0.3) clean. None qualifies for the exploratory divergence route: horseshoe rankings and mechanism results are zero-divergence-only. Each was refitted from the clean remediation commit, after backing up the failing directory to `statistical_models/_backups/`.

| Model      | Failure                                                                                                     | Remediation                   | Result                                                                                                  |
| ---------- | ----------------------------------------------------------------------------------------------------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------- |
| `hs-002`   | 1 divergence at declared 0.99; divergent draw in the slab tail (`hs_tau` 98.5th, `hs_c2` 98.8th percentile) | declare `target_accept` 0.999 | 0 divergences, R-hat 1.0020, ESS 4,608; ranking order unchanged, largest change in P(\|β\| > 0.1) 0.006 |
| `hs-004`   | 1 divergence at declared 0.99; `hs_c2` 99.4th percentile                                                    | declare 0.999                 | 0 divergences, R-hat 1.0020, ESS 3,597; ranking order unchanged, largest change 0.004                   |
| `mech-104` | 1 divergence at the 0.95 preset                                                                             | declare 0.98                  | 0 divergences, R-hat 1.00088, ESS 4,841                                                                 |

`hs-002` and `hs-004` are the two horseshoe level models. Both passed at 0.99 in the two previous batches, but both also carried the unconverged horseshoe prior-sensitivity cells in those batches, which points to a hard region near where the regularised slab binds rather than to chance. The fix follows `hs-001` (lifted to 0.999 on 2026-08-05) and `rlm-hs-002` (2026-08-22).

**`mech-104` is a process finding.** Its failure reproduced the 2026-09-08 failure exactly (the same R-hat and ESS to full precision). That batch fixed it with a command-line `--target-accept 0.98`, which was never written into `lrp_rli_mech_104.py`, so rebuilding from the registry re-ran the failing contract. At 0.98 this batch reproduces the 2026-09-08 remediation exactly, including an identical `mechanism_curve`. A remediation that exists only as a command-line flag does not survive a registry rebuild; it has to be declared in the module.

## Uncommitted edits during the sweep

Between about 11:37 and 11:44 BST another session working in this checkout added one word (a JavaScript package name) to `config/spellcheck/allow-en.txt` and created untracked files under `docs/presentations/2026-09-preliminary-findings/`. Each fit records the working tree's state when it finishes, so every fit that finished after `lrp-rlm-jc-102` (11:37 BST) recorded `dirty: true`: 273 of 276. After the clean refits described below, 268 still carry the flag. None of these files is model code, data or environment, but a Boolean dirty flag cannot show that nothing else changed transiently.

The evidence that nothing did is a model-design identity check. Every fit stores `model_design_identity`, a SHA-256 of the built model's graph structure and of its design data. Each model was rebuilt from a clean worktree at `b22ea254` and hashed: 254 by capturing the model at the last hook before sampling, and 22 by a complete `dev` fit (14 that sample sub-fits before the primary model is built, and 8 that add variables to the model after sampling). **All 276 stored identities match the clean commit on both hashes.** The comparison is kept in `run_metadata/design_identity_verification.json`.

The flag has one consumer. The phoneme-blending (B) response-link bundle for the ITT pair requires a clean source commit, so `itt-008` and `itt-108` were refitted from the clean worktree; `itt-008` reproduced bit-for-bit and `itt-108` to within 1e-13. The other 18 B-outcome fits, across seven families, only gain a provenance note on their pair card ("fitted from a working tree with uncommitted changes"). They were left as they are, with the identity check as the stronger evidence.

## Evidence loop

| Sweep                                         | Result                                                                                                  |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Standard ITT treatment-prior sweep (44 cells) | complete; independent evaluator passes                                                                  |
| P/N floor `tau` grid (12 cells)               | complete; evaluator passes for `itt-009` and `itt-011`                                                  |
| Phoneme-blending link pair                    | validated after the clean `itt-008`/`itt-108` refit; evaluator passes; key findings bound to the bundle |
| Influential-child refits                      | `itt-012`, `itt-013`, `itt-023` (one child each above the Pareto-k threshold); bundles valid            |
| DiD prior sweep                               | 21/21 cells converged after the standing `did-007` remediation (below)                                  |
| Gain-factor prior sweep                       | 6 rows, all converged, attached to `gf-005` and `gf-011`                                                |
| Level-factor prior sweep                      | 15 rows (W, L, P, B, N), all converged, attached                                                        |
| Dispersion prior sweep                        | 12 cells, all converged                                                                                 |
| Horseshoe prior sweep                         | 19/20 cells converged (15/20 in both previous batches); the one failure is withheld `rlm-hs-001`        |

Leaving out the flagged child changes neither the direction nor the strength of evidence in any of the three influential-child ITT results, though `itt-023`'s estimate grows: on the probability scale the average marginal effect goes from 0.055 to 0.060 for `itt-012` (P(effect > 0) 0.961 → 0.973), 0.032 to 0.032 for `itt-013` (0.967 → 0.969) and 0.111 to 0.131 for `itt-023` (0.999 → 1.000).

**The horseshoe prior sweep improved because of the `hs-002`/`hs-004` fix.** The sweep inherits each model's declared acceptance target, so at 0.999 all eight of their cells converge, where on 2026-09-08 at 0.99 `hs-002` failed two cells (1 and 33 divergences) and `hs-004` one. The remaining failure is `rlm-hs-001` at slab scale 4.0 (3 divergences at its 0.99), a model withheld at `inputs` in any case. The rankings are insensitive to the prior over this grid: in every converged cell the top three predictors are the same as the reference fit's, the rank correlation (Kendall's τ) with the reference is at least 0.94, and no predictor's P(|β| > 0.1) moves by more than 0.07.

**`did-007` needed its cell remediation for the third batch running.** The `mu_dose` cell at treatment-prior scale 1.5 had 6 divergences at the default, and refitting that model's cells with `--cell-target-accept 0.99` cleared all three (its swept `tau_logit_mean` moved from +0.134 to +0.133). Like `mech-104`, this fix lives only on the command line; declaring it in the sweep definition would stop it recurring.

## Release state

Release decisions were re-evaluated from the stored artefacts with `release.evaluate_publication` after every evidence bundle was attached, rather than read from the `release_decision.json` written at fit time. 270 fits are publishable and 6 are withheld, all at the `inputs` stage. 50 publishable fits carry a robustness note, mostly that the treatment-effect prior carries weight in the estimate (DiD 12, ITT 17, level factors 14, gain factors 7); the 2026-09-08 batch had the same count. No note is left over from fit order, because key findings were regenerated for all 276 fits after the evidence loop. All 276 reports were re-rendered afterwards and pass the runbook's freshness check, no rendered page contains a traceback or import error, and exactly the six withheld fits display "Findings withheld".

## Comparisons and reproduction

`compare_statistical_models.py` wrote the same 24 artefacts as last time. It declined two comparisons for the reasons it gives: the Tier-1 joint-versus-marginal contrast is not like-for-like (the sources' row counts differ), and `did-007`/`did-107` have unreliable PSIS-LOO (Pareto-k 1.25 and 1.15), so the dose comparison falls back to per-model expected log predictive density.

Three results from earlier batches reproduce:

1. **`med-059` total effect**: 2.3199 words (mean) and 2.3153 (median), identical to the 2026-09-08 batch.
2. **The L × N nested LOO** is again `comparison_valid=True` via `psis+reloo`, verdict inconclusive (|elpd_diff| < 4).
3. **Sampling diagnostics**: 245 of 276 fits reproduce the 2026-09-08 divergences, maximum R-hat and minimum ESS to full floating-point precision, across three dependency upgrades (#668, #670, #677). The 31 that differ are mostly latent-heavy families (horseshoe, growth, latent change score, historical joint), and every one still has zero divergences. One curiosity: `med-086` and `med-186` share a single fitted model, and in both batches that model sampled to the same two exact outcomes, assigned to the two IDs the other way round. The two outcomes differ only by Monte Carlo error.

## Results

The same suite re-estimated; recorded so the batch has a readable headline and the next rebuild has something to check against.

**Available-case modified ITT suite** (`itt-001`–`011`): average marginal effect on the probability scale, median with inner 50% and outer 89% equal-tailed intervals, and the probability of a positive effect. Every row is identical to the 2026-09-08 batch to three decimals.

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

A causal reading of these rows rests on the suite's stated randomisation and available-case assumptions. Read in plain words, the letter-sound row says that, under this model and prior, there is an 89% probability that the intervention raises the proportion of letter sounds correct by between 5 and 17 percentage points, and the data leave little doubt about its direction. The broad standardised vocabulary rows (R and E) are inconclusive: their intervals are centred near zero but still wide enough to include effects that would matter, so they are not evidence of no effect.

**Waitlist-crossover DiD**, `tau_t2` (the randomisation-anchored t2 arm gap), in items. Identical to the 2026-09-08 batch.

| Model     | Outcome              | Median | 50% interval   | 89% interval   | P(>0) | Evidence    |
| --------- | -------------------- | ------ | -------------- | -------------- | ----- | ----------- |
| `did-002` | Letter sounds (L)    | +3.53  | [+2.54, +4.50] | [+1.18, +5.81] | 0.991 | very strong |
| `did-001` | Word reading (W)     | +2.22  | [+1.18, +3.27] | [−0.31, +4.69] | 0.920 | moderate    |
| `did-003` | Phoneme blending (B) | +0.88  | [+0.54, +1.22] | [+0.06, +1.69] | 0.956 | moderate    |

These reuse the randomised t2 information under a different specification and are not independent replications of the ITT estimates.

## Open residuals

1. **14 fits record no `data_sha256`**: `surv-009/011`, the six `rlm-adj-*`, `rlm-ca-001/002`, `rlm-hs-001/002/003` and `rlm-mm-001`. The same set as the last two batches.
2. **The 6 withheld fits** are the Byrne/RLM ports blocked at `inputs` on unconfirmed `basspel`/`woco`/`basnum` score denominators (#338): `rlm-adj-001`, `rlm-hg-002`, `rlm-hg-003`, `rlm-hg-008`, `rlm-hs-001`, `rlm-mm-001`.
3. **The ERB source-archive question** (#631) is still open, so fits use the quarantined values.
4. **`did-007`'s cell remediation** recurs every batch and lives only on the command line.
5. **One horseshoe prior-sensitivity cell** (`rlm-hs-001`, τ₀ 0.1, slab 4.0) has 3 divergences at the model's 0.99; the model is withheld, so this limits only its own cross-check.
6. **PyTensor on macOS 27** needs the run-local compiler wrapper until pymc-devs/pytensor#2268 is fixed and reaches this project through `dse-research-utils`.
7. **Nothing stops a sweep from running in a checkout another session is editing.** The driver prints the identity at launch but does not re-check it; running sweeps from a dedicated clean worktree would avoid a repeat.
