> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

<!-- cspell:ignore nutpie psense Mundlak deapp erbto basnum basspel woco -->

# Full refit of every model, August 2026 (issue #554)

**Run record for a complete refit of both model layers — 50 gradient-boosting models and 242 Bayesian models — at the `reporting` configuration, with every report rendered, the cross-model comparison rebuilt and the deferred post-review batch closed.** Run 2026-08-19/20 from an empty output root, so this run replaces the August 17–18 baseline rather than updating it.

## Why this run

Issue #554 collected the computational work deferred from the author's review of the August 2026 refit: the refits for #551 (dependence-aware taught-versus-not-taught contrasts), #552 (baseline-balance-adjusted `level_factors`) and #553 (between/within splits for four more `pooled_levels` exposures), the 108-fit hearing-composite recoding refit, and regeneration of the step-1 gradient-boosting layer with a reconciliation against the step-2 rankings. Rather than refit the 108 `hs`-carrying fits as a subset, **every registered model was refitted**, so the whole suite shares one code state, one data state and one environment, and no fit in the published set predates a merged change.

## What was run

| Layer             | Models | Config                                        | Result                                      |
| ----------------- | -----: | --------------------------------------------- | ------------------------------------------- |
| Gradient boosting |     50 | `reporting`                                   | 50 fitted and rendered, 0 failed, 1 h 36 m  |
| Bayesian          |    242 | `reporting` (6 chains x 6000 draws, `nutpie`) | 242 fitted and rendered, 0 failed, 9 h 26 m |

Both sweeps ran through a **checked-in resumable driver**, `scripts/run_refit_sweep.py`, added in this branch. The August run was driven by an untracked scratchpad script whose logic could not be reconstructed from the artefacts and whose skip rule checked completion markers rather than fit identity; that was recorded as a provenance limitation at the time. The new driver runs one subprocess per model so the fit and its render stay together (`fit_statistical_model.py all --render` batches every render until after the last fit, so an interruption leaves fitted-but-unrendered directories), writes a per-model log and a JSON-lines journal, and on resume reuses a stored fit only when its sampling preset, source commit, dirty flag, data digest and environment-lock digest all still match.

Two provenance improvements over the August run. First, the working tree was clean for every fit: 189 fits record commit `4e924948` and 53 record `83868a97`, the two commits differing only in report tooling that no model imports. One fit (`lrp-rli-med-062`) recorded `dirty=true` because a report script was created mid-sweep; it was refitted from a clean tree and the stored set now has no dirty fit, against 50 in August. Second, the word-reading missing-data archive was supplied to every fit up front (`--rli-randomised-archive`), so `lrp-rli-itt-010` ran its mandatory missing-outcome sensitivity inside the primary fit instead of needing the remediation refit it needed in August.

## Headline outcome

| Measure                                    | Result                                       |
| ------------------------------------------ | -------------------------------------------- |
| Bayesian models fitted                     | 242 / 242 (0 failed)                         |
| Reports rendered                           | 242 / 242, plus 50 gradient-boosting reports |
| Convergence gate passed on first fit       | 241 / 242                                    |
| Fits with any divergence after remediation | 0                                            |
| Publishable (`release_decision`)           | **236 / 242**                                |
| Withheld                                   | 6, all `inputs_unresolved`                   |

## Remediation

Ten fits needed work after the sweep, and none of it weakened a gate.

**One gate failure.** `lrp-rli-mech-104` returned exactly one divergence in 36,000 draws with otherwise clean diagnostics — the same fit, and the same single divergence, as in August. Under the divergent-transition policy a divergent fit fails closed however small the count, so it was refitted at `--target-accept 0.99`. The module declares no target-accept of its own, so this is a genuine raise rather than a silent lowering of one. It reached zero divergences and `release_decision` `ok`.

**Nine power-scaling flags.** Five `did` fits (`001`, `003`, `007`, `013`, `101`), two `gain_factors` fits (`gf-005`, `gf-011`) and the two floor-rule `itt` fits (`itt-009` P, `itt-011` N) shipped `robustness_unresolved` because the release gate requires a treatment-prior sweep computed from the fit's own trace before it will publish a prior-dominant effect. The repository's own runners produced that evidence without refitting the primaries: `did_prior_sensitivity.py --attach --cell-target-accept 0.99`, `gf_prior_sensitivity.py --attach`, and `tau_prior_sensitivity.py --outcomes P N`, which routes to the separate `floor_tau_prior_sensitivity` archive. Every grid cell converged with zero divergences. As in August, `did-007` needs the cell escalation to converge its widest cell.

**Two fits that pass but may not be read alone.** `lrp-rli-itt-008` and `lrp-rli-itt-108` are the phoneme-blending pair, and neither may be released without the trace-backed link comparison; `blending_link_sensitivity.py` was run after both completed and before key findings were regenerated.

Release decisions are written at fit time, so `regenerate_key_findings.py all` re-decided them over the stored fits without resampling. The 9 `robustness_unresolved` and the 1 `gate_failed` all cleared, leaving 236 publishable.

## What is still withheld, and why that is correct

The six withheld fits are unchanged from August: `lrp-rlm-adj-001`, `lrp-rlm-hg-002`, `lrp-rlm-hg-003`, `lrp-rlm-hg-008`, `lrp-rlm-hs-001` and `lrp-rlm-mm-001`, each because a bounded-count denominator (`basnum`, `basspel` or `woco`) is not confirmed against the instrument. This is the documented decision from issue #338, not an outstanding action from this run. Clearing them needs the administered manuals or test records, or an explicitly approved raw-score analysis.

## What the batch's own changes did

**#552, `level_factors` centred on the timepoint-1 gap.** The re-parameterisation did what it was for. In August, power scaling flagged the treatment term of two `level_factors` fits as prior-dominant; under the new parameterisation neither is flagged, because the prior sensitivity has moved off the randomised contrast and onto the pre-randomisation balance quantity where it says nothing about the effect. For word reading (`lf-010`): `d_grp_time[t2]`, the focal randomised contrast, has prior sensitivity 0.056 against likelihood sensitivity 0.140, while `arm_gap_t1` runs 0.094 against 0.031. Letter sounds (`lf-009`) show the same pattern (0.058 / 0.106 and 0.070 / 0.039). No `level_factors` fit needed a prior sweep in this run.

**Hearing-composite recoding.** Every fit that carries the hearing flag now uses the three-valued OR (25 flagged, 20 clear, 9 unknown) rather than the strict both-known OR that left one hearing-impaired child coded unknown and then filled to the clear reference. The six coefficients question 7 quotes moved as predicted — slightly toward zero, with no change of direction or of interpretation:

| Fit        | Quantity                                                | August (strict OR)                 | This run (three-valued OR)            |
| ---------- | ------------------------------------------------------- | ---------------------------------- | ------------------------------------- |
| `adj-065`  | whole-study word-reading gain, words per SD of the flag | +2.4 [+0.5, +4.5], P = 0.98        | +2.17 [+0.33, +4.23], P = 0.972       |
| `mm-002`   | latent code-factor gain                                 | +0.23 [+0.05, +0.40], P = 0.98     | +0.207 [+0.035, +0.379], P = 0.972    |
| `mech-058` | same-period gain given period-start reading             | +0.07 [-0.00, +0.15], P ~ 0.94     | +0.069 [-0.008, +0.146], P = 0.926    |
| `lcsm-082` | latent reading/blending change, flagged versus clear    | +0.09 [-0.02, +0.19], P = 0.90     | +0.071 [-0.038, +0.177], P = 0.854    |
| `pl-001`   | level pooled across the four waves                      | -0.00 [-0.19, +0.18], inconclusive | -0.021 [-0.204, +0.160], inconclusive |
| `ca-001`   | concurrent level at timepoint 3                         | +0.16 [-0.03, +0.35], P = 0.91     | +0.145 [-0.043, +0.334], P = 0.893    |

The counter-intuitive positive association between the hearing-difficulty flag and word-reading _gain_ therefore survives the correction, and so does its reading: the flag is coarse, it merges an impairment with an ear-infection history, nine children are unknown, and every one of these coefficients was included to clean some other estimate rather than to estimate hearing's own association. The August caveats stand; only the coding has changed.

## Step 1 regenerated and reconciled with step 2

All 50 gradient-boosting models were refitted at `reporting` and their rankings compared with the `horseshoe` Bayesian rankings of the same targets (`scripts/compare_horseshoe_vs_gb.py`, one comparison per pair, written to `output/statistical_models/comparison/`).

| Pair                 | Target              | Spearman rho |           Top-3 overlap |
| -------------------- | ------------------- | -----------: | ----------------------: |
| `hs-002` / `gbl-012` | word-reading level  |        +0.54 |           2 of 3 (E, L) |
| `hs-004` / `gbl-009` | letter-sound level  |        +0.43 |              1 of 3 (W) |
| `hs-001` / `gbg-012` | word-reading change |        +0.36 | 2 of 3 (age, behaviour) |
| `hs-003` / `gbg-009` | letter-sound change |        +0.00 |                  0 of 3 |

The two methods agree moderately about **levels** and much less about **change**, and for letter-sound change they do not agree at all. That is consistent with what the gradient-boosting models themselves report about predictability: pooled out-of-sample R-squared has a median of about 0.5 across the level models and about 0.1 across the change models (word-reading change 0.08, letter-sound change 0.24). When a target carries little predictable signal, two honest methods rank the noise differently — so a "top predictors of progress" list from either method alone would not replicate. The level models are concurrent descriptions (a child's score from their other scores at the same assessment) while the change models are genuinely prospective, which makes the contrast sharper still.

This closes the two-step loop the methodology promises for this refit: the exploratory layer surfaced no predictor the statistical models had missed, and what it contributed was the level-versus-change contrast, which is a substantive result rather than a methodological aside.

## Artefacts

- Sweep journals and per-model logs: `output/_sweep/` (`journal-statistical-reporting.jsonl`, `journal-gb-reporting.jsonl`, `logs/`).
- Cross-model comparison: `output/statistical_models/comparison/`.
- Prior-sensitivity archives: `output/statistical_models/did_tau_prior_sensitivity/`, `gf_tau_prior_sensitivity/`, `floor_tau_prior_sensitivity/`, and each fit's attached `tau_prior_sensitivity.csv`.
- Blending link bundle: `output/statistical_models/blending_link_sensitivity/`.
- Plain-language report inputs: `output/summary_report/results.json` and its figures.

## Known limitation of this run

The driver's resume rule compares a stored fit's source commit against the current `HEAD`, so any later commit invalidates the whole stored set for reuse purposes even when the change cannot affect a model. That is deliberately conservative, but it means a resumed sweep after unrelated work would refit everything; work around it by naming models explicitly (`--models`) until the check is relaxed to a set of accepted commits.
