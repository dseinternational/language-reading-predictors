<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Full `reporting` refit and prior/posterior predictive-check review (2026-08-04/05)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

Run record for a complete `reporting`-tier refit of every registered statistical model, commissioned to review the prior- and posterior-predictive checks across the suite. Preliminary research data — all estimates provisional.

## Run metadata

- **Command:** a resumable per-model driver calling `python scripts/fit_statistical_model.py <model_id> --config reporting --render` for each of the 194 discovered models in turn, rather than `all --render`. Two reasons: a monolithic sweep has previously been reaped mid-run, and `all --render` batches every Quarto render until after all fits complete, so an interruption leaves traces with no reports. Per-model invocation lands each report immediately after its own fit.
- **Sampling preset (`reporting`):** 6 chains × 6,000 draws × 6,000 tune, `target_accept = 0.95`, `nutpie`.
- **Scope:** 194 models. Note the count discrepancy worth knowing: `definitions.MODEL_REGISTRY` holds **181**, while the filename-derived `MODELS` map that `fit_statistical_model.py` actually iterates holds **194**. The sweep used the latter. The `lrp-fit-statistical` skill's "89 models across 16 families" is badly stale; `definitions.KINDS` now lists 22 kinds.
- **Wall time:** 8.9 h of accumulated model time on a 16-core workstation. Long pole as expected: the two-mediator g-formula fits `med-064` (41 min), `med-075` (39 min), `med-066` (36 min).
- **Result:** 194/194 fitted and rendered; **181 passed the convergence gate at the sweep and 13 failed**. All thirteen were subsequently resolved — the suite now stands at **194/194 passing** (see the gate section).

## A blocking defect found and fixed

Nine models died in `_finalize_report` **after sampling completed**, discarding the whole staging directory — no trace, no predictive-check artefacts.

`reporting.py` stamped a `blending_link_sensitivity.csv` SHA-256 into `key_findings.json` for any model with `outcome_symbol == "B"`. But that bundle is built only by the registered paired-link pair — `blending_sensitivity.BLENDING_LINK_MODELS` hard-codes `lrp-rli-itt-008` and `lrp-rli-itt-108`. Eleven models carry outcome `B`; the other nine (`al-006`, `ca-007`, `did-003`, `dose-084`, `gf-006`, `gf-106`, `lf-006`, `med-087`, `med-187`) never write the CSV and hit `FileNotFoundError`.

The stamp sat _outside_ the `try/except` directly above it — the one whose comment reads "A malformed CSV must degrade to an explicit note, never break a fit or a render (#320 acceptance criteria)". Only `_kf_build_itt` calls `_kf_blending_link_evidence`, which raises the catchable `_KeyFindingsUnavailable`; every other family's builder succeeds and falls through to the unguarded hash.

Introduced by `4e51cee` (#466). Fixed by keying the stamp on membership in `BLENDING_LINK_MODELS` instead of on the outcome symbol, with the outer `B` check retained so the import stays function-local (`blending_sensitivity` imports `reporting`, so a module-level import would be circular). The #466 release interlock is unchanged in the scope it was written for. Regression test: `test_non_itt_blending_outcome_does_not_require_the_paired_bundle`, verified to fail against a reverted copy of the source with the identical production error.

## Convergence gate — 13 failures at the sweep, all resolved

Under `notes/202608021625-divergence-qualification-policy.md` and `METHODS.md:65`, **every divergent fit fails closed**; the earlier percentage-based leniency is superseded and there is no live qualification pathway. All 13 were withheld by the report gate as fitted.

| model        | failing checks          | divergences | max R-hat | min ESS |
| ------------ | ----------------------- | ----------: | --------: | ------: |
| `mech-093`   | divergences             |           1 |    1.0020 |   2,644 |
| `mech-094`   | divergences             |           1 |    1.0034 |   2,680 |
| `mech-095`   | divergences             |           2 |    1.0028 |   2,824 |
| `mech-156`   | divergences             |          21 |    1.0024 |   3,502 |
| `mech-157`   | divergences             |          27 |    1.0019 |   2,810 |
| `mech-188`   | divergences             |           1 |    1.0017 |   2,976 |
| `mech-189`   | divergences             |           1 |    1.0014 |   2,614 |
| `mech-191`   | divergences             |          28 |    1.0024 |   3,726 |
| `hs-001`     | divergences             |           1 |    1.0005 |   9,452 |
| `mm-001`     | R-hat, divergences      |           1 |    1.0136 |     431 |
| `mm-002`     | R-hat, ESS              |           0 |    1.0158 |     328 |
| `mm-101`     | R-hat, ESS, divergences |           2 |    1.0241 |     213 |
| `rlm-mm-001` | R-hat, divergences      |          58 |    1.0159 |     458 |

The nine `mech`/`hs` failures are divergence-only with healthy R-hat and ESS. The four `corr_factor` failures are qualitatively different — they fail on **R-hat and/or ESS**.

**All thirteen were resolved on 2026-08-05; the suite now stands at 194/194 passing.** How, and one diagnosis that turned out to be wrong, are recorded below.

### The eight HSGP knee tests

`mech-093/094/095/156/157/188/189/191` all carry an HSGP mechanism curve, and a nonlinear knee or shape is zero-divergence-only, so no qualification route exists. All eight adopt the #438 thin-support reparameterisation (basis count 6 from 10, tighter `InverseGamma(8, 8)` lengthscale prior) as a per-model opt-in — the shared defaults stay, since the same lever regressed `mech-173` from 0 to 10 divergences. Every one went to **0 divergences**, and against #438's own acceptance criteria the curve is intact: amplitude _increases_ slightly in all eight (the direction #438 measured), max pointwise |Δf| is 0.007–0.031, and no parameter moves more than 0.064 posterior SD.

### `hs-001`

A single divergence in 36,000 draws at its declared 0.99, where its `hs-002/003/004` siblings pass at the same setting. A horseshoe ranking is likewise zero-divergence-only. Lifted to 0.999 — above its own default, not below, which is the distinction the remediation trap above turns on — giving 0 divergences, R-hat 1.0007, min ESS 9,491, with the ranking unmoved (identical rank order, max change in `p_abs_gt_delta` 0.008). The value is declared in the spec, and a refit with no CLI override reproduces it.

### The four `corr_factor` fits — the diagnosis was wrong

The reading above, and the standing account in `notes/202607241200-mm001-gate-exception.md`, was that this is intrinsic near-singular correlation geometry that sampler tuning cannot repair — `rlm-mm-001`'s wave-3 domains are near-collinear (correlations 0.82–0.95, first eigenvalue ≈ 89.5 % of variance) and pushing `target_accept` 0.99 → 0.999 was measured to _starve_ the chains (min ESS 749 → 99, max R-hat 1.006 → 1.045). That made it the repo's standing example of geometry that genuinely cannot be fixed.

It was not the geometry. **Every** R-hat/ESS failure across all four models sits on `factor_cov` alone, while every quantity the models actually report converges cleanly — `factor_corr` R-hat ≤ 1.003 with ESS 2.2 k–24 k, `lambda_load` and `communality` better still. `factor_cov` is an `LKJCholeskyCov` whose Cholesky factor and standard deviations are both discarded at construction (`_, corr, _`); in a CFA the factor scale is fixed by the loadings, so those sd components are **unidentified**. They wander, mix poorly, and — because the gate scans every free RV — fail the gate on behalf of a parameter nothing downstream reads. The existing override comment had already noticed the sds were discarded without drawing the conclusion.

Switching both builders to bare `pm.LKJCorr` (the choice `build_longitudinal_corr_factor_model` and the measure-correlation block already make for exactly this reason, and which `lcf-001` — passing where these four failed — is the natural experiment for) clears all four outright:

| model        | divergences | max R-hat       | min ESS     |
| ------------ | ----------: | --------------- | ----------- |
| `mm-001`     |   1 → **0** | 1.0136 → 1.0016 | 431 → 4,705 |
| `mm-002`     |   0 → **0** | 1.0158 → 1.0017 | 328 → 5,441 |
| `mm-101`     |   2 → **0** | 1.0241 → 1.0010 | 213 → 6,962 |
| `rlm-mm-001` |  58 → **0** | 1.0159 → 1.0004 | 458 → 5,183 |

Including `rlm-mm-001`, whose 58 divergences were the case held up as unfixable.

**A caveat that matters, stated precisely.** The _prior_ is unchanged — the correlation has an LKJ(η) marginal under either construction, and the discarded components provably fed nothing. The _posterior estimates_ did move: the off-diagonal correlations rise by 0.001–0.028 in absolute terms, up to **0.23 posterior SD**, which is roughly ten times the Monte-Carlo error implied by their old ESS. So this is not "the same posterior, better sampled". The reading taken here is that the old fits' conditioning contaminated exploration of the correlation block even though its own R-hat and ESS looked acceptable — the missing-mass failure `METHODS.md:65` describes, where chains agree with each other while jointly failing to enter a region. The new fits are the better-conditioned ones and their estimates should be preferred. No published number changes, because the old fits failed the gate and were withheld in full. The substantive reading — high domain correlations, 0.65–0.95 — is unchanged either way.

## Geometry remediation — and a methodological trap

Five models were genuinely remediated by raising `target_accept` **above** their registered default (all five had no spec default, so the `reporting` preset 0.95 applied):

| model      | target_accept | divergences before → after |
| ---------- | ------------: | -------------------------- |
| `dose-084` |          0.97 | 1 → 0                      |
| `did-007`  |          0.97 | 2 → 0                      |
| `dose-083` |          0.99 | 3 → 0                      |
| `dose-077` |          0.99 | 8 → 0                      |
| `dose-177` |          0.99 | 23 → 0                     |

**Recorded because it nearly produced a false pass:** a blanket escalation is wrong here, because models carry heterogeneous, deliberately tuned spec defaults. `mm-001`, `mm-002`, `mm-101` set `target_accept = 0.999` and `rlm-mm-001` sets `0.99`, with in-module comments explaining that the value was lifted above the preset specifically to clear boundary divergences; `mech-093/094/095` also set `0.999`. Forcing a uniform 0.99 therefore _lowered_ acceptance for seven models. The `corr_factor` group visibly degraded (`mm-001` 1 → 19 divergences, `mm-002` 0 → 12), but the more dangerous case was `mech-093/094/095`, which came back **zero-divergence at a weaker sampling contract** — a larger step size taking the sampler past the problematic region rather than through it, which `METHODS.md:65` names explicitly ("not evidence about posterior mass the sampler never reached"). `hs-001` cleared only by reseeding at its own registered 0.99, which the same policy rejects as positive evidence. All eight were restored from a pre-remediation backup and remain gate-failed.

**Provenance item — resolved 2026-08-05.** The five remediated fits were stored at a `target_accept` the registry did not reproduce. Each value is now declared in its model's `spec.extra["target_accept"]` (`dose-077/083/177` 0.99, `dose-084` and `did-007` 0.97), with the divergence counts before and after recorded in the module comment. `dose-084` was refit with **no** CLI override as a check and reproduced the remediated result exactly (0 divergences, R-hat 1.001, min ESS 6,720); the default seed is fixed, so the promoted values are exactly reproducible.

Promoting them surfaced a second defect. `spec.extra["target_accept"]` was applied by `pipeline._apply_spec_target_accept`, which **only six of the family entry points called** — `fit_mechanism`, `fit_horseshoe`, `fit_rlm_horseshoe`, `fit_rlm_corr_factor`, `fit_correlated_factor` and `fit_longitudinal_corr_factor`. Neither `fit_dose_response` nor `fit_did` was among them, so the declaration would have been accepted by the spec and then silently ignored at sampling time — the module would claim 0.99 while the fit sampled at the preset's 0.95. No model was affected in practice (an audit found no existing declaration in a non-honouring family), but the trap was live for exactly the change being made here. Resolution now lives in `context.make_context`, which every family goes through, with the same **CLI override > model default > preset** precedence; the six per-family calls and the helper were removed. `target_accept` is also now listed in the `_LEGACY_KEYS` of all seven typed-run-plan families (`itt`, `gain_factors`, `level_factors`, `did`, `concurrent`, `aligned`, `growth`) as a recognised sampler key — the DiD run plan's strict unknown-key check correctly rejected the new declaration until it was declared, which is the validation working as designed.

## Posterior-predictive checks — no drift

191/194 fits emit `ppc_summary.csv` (`lcf-001`, `rlm-jc-001`, `rlm-mm-001` do not). Per Decision 2 of `notes/202607261405-binomial-exchangeability-item-difficulty-review.md`, `scripts/ppc_coverage_sweep.py` is a standing diagnostic read **relatively** — a measure drifting from its peers — not against an absolute nominal level. Against that note's §4 baseline (2026-07-26, 190 fits):

| outcome | 2026-07-26 | this sweep |  drift |     | family              | 2026-07-26 | this sweep |  drift |
| ------- | ---------: | ---------: | -----: | --- | ------------------- | ---------: | ---------: | -----: |
| `T`     |       0.65 |      0.647 | −0.003 |     | `corr_factor`       |       0.58 |      0.582 | +0.002 |
| `R`     |       0.68 |      0.671 | −0.009 |     | `aligned`           |       0.63 |      0.631 | +0.001 |
| `L`     |       0.71 |      0.708 | −0.002 |     | `itt`               |       0.66 |      0.656 | −0.004 |
| `B`     |       0.74 |      0.737 | −0.003 |     | `did`               |       0.82 |      0.816 | −0.004 |
| `E`     |       0.75 |      0.747 | −0.003 |     | `lcsm`              |       0.85 |      0.848 | −0.002 |
| `W`     |       0.76 |      0.757 | −0.003 |     | `historical_growth` |       0.85 |      0.849 | −0.001 |
| `TE`    |       0.80 |      0.801 | +0.001 |     |                     |            |            |        |
| `N`     |       0.82 |      0.814 | −0.006 |     |                     |            |            |        |

Largest absolute drift anywhere: **0.009**. The 50 % bands overcover everywhere (0.58–0.92) and the 90 % bands sit at 0.93–1.00, exactly as before. Nothing has moved, on independently re-sampled fits — the established reading stands, and §6 of that note already settled that the overcoverage is substantially mechanical rather than a likelihood defect (the Conway–Maxwell-binomial probe, free to go below binomial variance, chose ν = 0.50 and coverage barely moved, 0.62 → 0.60).

## Prior-predictive checks — healthy where present, absent for 24 models

The sweep initially found **24 models (12 %) emitting no prior-predictive check at all** — neither CSV nor PNG — across `historical_growth` (9/9), `lcsm` (5/5), `growth` (3/3), `survival` (2/2), `long_corr_factor`, `historical_joint`, and the three `lrp-rlm-*` variants of otherwise-covered families (`rlm-adj-001`, `rlm-hs-001`, `rlm-mm-001`). **Closed 2026-08-05: all 194 models now emit one.** `METHODS.md:63` treats a material predictive shape flag as a qualification in its own right, so those models previously had one leg of that check missing.

Three distinct causes, not one:

1. **An unsupported container, not a missing call.** `save_prior_predictive_plot` read the observed values from `PreparedData.post_counts`, which the longitudinal containers do not have — `WavePanel` and `LongitudinalPanel` expose `counts` instead. On a panel family the lookup raised into the guarded handler and silently wrote nothing.
2. **A re-derived observed set.** The first repair sourced observed values from `prepared.counts[symbol]` and produced _misaligned_ overlays that looked plausible: `lrp-rli-lcsm-067` compared 639 replicate cells spanning W/L/E against 210 observed cells for W alone, and `lrp-rlm-hg-001` compared 300 modelled rows (core **plus** observed extension waves) against 228 complete-case core cells. Both were caught by checking that replicate cells equal observed count, and both are now sourced from the trace's `observed_data` group — the exact array passed to `observed=` — put through the same outcome selection as the replicates, so alignment holds by construction. This is the same failure mode as the mechanism forest's reconstructed-index mismatch: derive an index once, in the factory, and read it back; never rebuild it in a consumer.
3. **Stacked likelihoods need a persisted cell map.** `lcsm` and `growth` flatten every measure into one `y_obs` vector, so a single overlay pools scales with different maxima. Both factories now persist `y_obs_cell_outcome` (the joint family's existing `y_post_cell_outcome` idiom, via `pm.Data` — `pm.ConstantData` is gone in this PyMC), and the selector resolves a per-node map. They emit one correctly-selected check per measure.

Two families needed a different instrument rather than a missing call, and in both cases the suite already contained the right one:

- **`survival`** — `y_event` is a 0/1 person-period event, so a count histogram carries no shape information. The posterior side already made this distinction (`y_event` is in `_PPC_BINARY_NODES`, taking `ppc_offfloor_rate_coverage` "because per-observation interval coverage of a 0/1 indicator is degenerate"). The prior check is now the same statistic: the replicated **event-rate** distribution against the observed rate. It found something immediately — for `lrp-rli-surv-009` the prior-predictive rate centres at **0.60** (median; min 0.05, 25 % 0.46) against an observed **0.17**. Admissible, since 0.17 is inside the prior range, but below the 25th percentile: the hazard prior is optimistic about off-floor movement relative to these data.
- **`corr_factor` / `long_corr_factor`** — the _absence_ of a coverage statistic here is deliberate, not a gap: `pipeline._save_ppc`'s else-branch documents that measurement / latent nodes have "no single count outcome, so keep the legacy overlay and emit no coverage statistic". Manufacturing a coverage number against that decision would be wrong, and pooling nine standardised instruments into one histogram is meaningless. The prior side is now the symmetric counterpart of the legacy overlay — per-indicator marginals via `plot_ppc_dist(group="prior_predictive")`. The richer option (checking whether replicates reproduce the observed indicator **correlation matrix**, the standard CFA check and the one aimed at these models' actual headline) was considered and deliberately not taken: it is a new diagnostic that can flag a model, so it is a methods decision rather than a code one.

**Result.** All 194 models emit a prior-predictive check. Of the 189 with the comparable summary-CSV schema, **every one covers the observed range**, with the prior predictive wider than the data (per-family median sd ratio 1.15–3.46) and modest location shifts (−0.51 to +0.88 observed SD). `growth` is the most diffuse by a distance (sd ratio **3.46**, shift +0.86) — still admissible, but its priors are roughly three and a half times the spread of the data. The remaining five carry a different schema by design: the two `survival` fits report a rate rather than counts, the two CFAs a figure-only overlay, and `rlm-jc-001` writes one file per measure.

## Power-scaling sensitivity

Across 2,457 parameters: **1,025 clear**, 976 "potential prior-data conflict", 456 "potential strong prior / weak likelihood". Note that `psense_summary.csv` writes a tick `✓` for an _unflagged_ parameter, so the tick is the clear verdict, not a warning.

The prior-dominated parameters concentrate sharply: **356 of the 456 sit in just three `corr_factor` models** (`mm-001/002/101`, roughly 120 of ~176 parameters each) — the same three failing R-hat and ESS. That is one coherent finding rather than three: a weakly identified latent-factor model in which the data do not pin the loadings, so the prior carries them. The `horseshoe` family also shows high prior sensitivity (max 0.99), but a shrinkage prior is _designed_ to move the posterior under power-scaling, so that is expected by construction rather than a defect.

## Cross-model comparison

`python scripts/compare_statistical_models.py --config reporting` ran to completion (exit 0), writing `itt_vs_joint_tau.csv`, `triangulation_consistency.csv`, `tau_forest.png`, the Tier-1 decoding-specificity contrast and negative-control forest, `mediation_family.{csv,png}` and five nested PSIS-LOO tables. Two things did not come out clean:

- **The mechanism forest was not written at all.** `mechanism_forest.png` and `mechanism_forest.csv` are absent. `lrp-rli-mech-058` failed reconstruction — "reconstructed `mech_logit` size (157) != trace `obs_id` size (156) — likely confounder-only missingness the keep-mask does not model" — and the code drops that model from both the plot and the CSV, then skips the whole forest because a mechanism run is missing. The forest is a documented expected artefact of this script, so its absence is a gap, not a silent success. The size mismatch is the thing to fix.
- **`did-007` / `did-107` PSIS-LOO is unreliable** (Pareto-k 1.17 and 1.15, above `good_k`) and the exact-refit repair path is mechanism-only, so the script wrote per-model `elpd_loo` instead of `az.compare` deltas. The dose comparison for that pair is therefore not a nested test. This is consistent with the standing observation that nested LOO only validates for linear-mechanism pairs.

Exact-refit Pareto-k repair did succeed for the mechanism pairs `mech-058/071`, `mech-061/161`, `mech-063/163` and `mech-104/204` (one refit each).

## Follow-ups

1. ~~Promote or discount the five remediated `target_accept` values~~ **done 2026-08-05** — declared in each spec; resolution hoisted into `context.make_context` so a declaration binds for every family.
2. ~~Emit prior-predictive checks for the 24 uncovered models~~ **done 2026-08-05** — all 194 now emit one; backfilled onto existing traces with `--reuse-trace`, no re-sampling, and every regenerated fit kept its gate verdict.
3. **Apply the per-measure selection to the posterior side too.** `_PPC_MULTI_OUTCOME_KINDS = {joint, lcsm, growth}` currently _skips_ the posterior distribution overlay for those families because the node "pools measures with different denominators". The `y_obs_cell_outcome` map added here removes that obstacle, so the posterior overlay could now be emitted per measure exactly as the prior one is. Not done: it changes an existing artefact rather than adding a missing one. [open]
4. ~~`corr_factor` weak identification~~ **done 2026-08-05** — not weak identification of the deliverable at all; the gate was failing on the discarded, unidentified `LKJCholeskyCov` sd components. Bare `LKJCorr` clears all four.
5. ~~`mech-156/157/191` HSGP divergences~~ **done 2026-08-05** — resolved by the #438 thin-support reparameterisation, extended to all eight models in the same failure class.
6. ~~Repair `mech-058`'s reconstruction size mismatch~~ **done 2026-08-05** — the comparison script now reads the persisted `mech_post_logit` instead of rebuilding it; the mechanism forest is produced again.
7. **`MECH_IDS` is stale.** The mechanism forest is structurally a single point: the list still names `mech-056` and `mech-057`, which were linearised in the #258 review and therefore have no `f_mech` curve to contribute. Worth either dropping them or plotting their linear slopes on the same scale. [open]
8. ~~Revisit `notes/202607241200-mm001-gate-exception.md`~~ **done 2026-08-05** — a second supersession notice records that its central claim (divergences intrinsic to a near-singular correlation matrix) is wrong, that the causal attribution was to the unidentified `LKJCholeskyCov` sds rather than the collinearity, and that the note's own "Out of scope" bullet had already named `LKJCorr` + `L @ Lᵀ` as the route while under-scoping it to `mm-002` mixing.
