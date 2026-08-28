# The 2026-08-26 full-registry refit batch

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5).

## What ran and why

The standing decision of 2026-08-25 was that every statistical refit implied by the August remediation series (#575–#623) happens later, in **one batch**, rather than piecemeal — several PRs deliberately shipped code whose stored artefacts were stale, and #623 made a stale run plan a release blocker. Resolving the open issues #575, #584, #585, #586 and #588 triggered that batch: the last outstanding code work (the #575 gain-factor remediation, PR #625) landed first, and then the **entire registry — all 269 registered models — was refit at the reporting preset**, rather than only the ~160 plan-stale or never-fitted ones. Two reasons: artefact-writer changes (new CSVs, corrected arithmetic, new sub-fits) are invisible to the plan-currency check, and single-commit provenance is the cleanest footing for the #623 pair bindings. The pre-batch artefacts are preserved locally at `output/statistical_models/models.pre-batch-20260826/` (63 GB; not committed).

## Execution record

- **First launch** (commit `6bfcc95b`, clean tree, two parallel driver streams via `scripts/run_refit_sweep.py`): stopped after ~25 minutes when `lrp-rli-med-062` failed — the `gaussian_composite` mediation branch still returned `EmptyPayload` where the #619 link-threading requires a `MediationPayload`. Rather than discover sibling defects serially over five hours, the batch was stopped, the branch fixed (#627), and a **50-model dev-config pre-flight** run over every at-risk code path (all 20 mediation models, mech-301–305, lf-201–211, jc-102, did-103–106, itt-215/012, jm-001/002, jc-001, hg-001, mech-056, lf-001, gf-005): 50/50 passed, confirming the composite payload was the only structural bug.
- **Full run** (commit `dc1feea6`, clean tree): **269/269 fits completed, 0 failures** — 249 models in 4 h 20 m on one stream, the 20 mediation fits in 4 h 44 m on the other. Seeded samplers mean unchanged code paths reproduced byte-identical posteriors, so every observed movement is code-driven.
- **Convergence-gate triage.** Nine fits (mm-001/002/101/102, rlm-mm-001, lcf-001, rlm-jc-001/002/102) failed only the new upstream `diagnostics_assessable` check (dse-research-utils v0.12.0, adopted in #617 after those families' last fits) on the structurally constant entries of their `LKJCorr` Cholesky factors — every real diagnostic clean. #628 reclassifies a coordinate only when its index proves the constraint (upper-triangle exactly 0, or the `[0, 0]` corner exactly 1, across every chain and draw); the nine summaries were re-derived from their traces without refits and all pass. One fit (`lrp-rli-mech-104`) drew a single stochastic divergence in 36,000 draws; because sampling is seeded, a plain rerun reproduces it, so it was refit once at `--target-accept 0.98` (recorded in its config): 0 divergences, max R-hat 1.0012.
- **Evidence loop.** The ITT blending archive was rebuilt from the fresh itt-008/108 traces; the tau (44-cell), DiD, gain, level and dispersion treatment-prior sweeps re-ran at reporting tier and attached; the floored P/N release grid re-ran (`tau_prior_sensitivity.py --outcomes P N`); itt-010 was refit with `--rli-randomised-archive` so its full-cohort missingness bundle exists; did-007's sweep cells (6 and 3 divergences at the widened dose-prior scales under two seeds) converged under the documented `--cell-target-accept 0.99` escalation and attached; did-104 — first fitted in this batch and immediately prior-dominant on `tau_t2`, as a model that softens the allocation prior would be — was added to the sweepable set (#629) and its bundle attached.
- **Plan-persistence repair (#630).** The batch's own currency sweep caught 13 freshly refit fits (10 mediation, mm-002/102, adj-065) that were immediately "plan-stale": their pipelines deliberately overwrote the stored `resolved_run_plan` with a loader-filtered copy when a constant covariate dropped (a #585-era record), which the later #623 currency check can never accept. `config.json` now always carries the resolver's own plan (the active plan still drives the factory and the recipe prose; the removals stay recorded in `extra`), a pipeline-boundary AST test enforces it — and immediately caught the same pattern latent in both `joint_mechanism` entry points — and a metadata-only tail refit of those 13 fits plus two mediation fits carrying mid-batch provenance blemishes (med-276 fitted from a momentarily dirty tree, med-387 from an off-main commit) was **started at the merged commit `9f9cc192` and stopped on request at 0/15 complete** — the posteriors would be byte-identical (seeded, code-identical models); only the persisted `config.json` provenance/plan would change. Until that tail is run, those 15 fits remain the known plan-currency/provenance residual; everything they publish is numerically final.
- **Reports.** All 269 reports rendered; one crash (`lrp-rli-lf-206`, the exempt two-wave B window comparator, whose "ready, no link pairing" status carries no cards) was fixed in the shared partial (#632) and the report re-rendered — 269/269.

## Final state

- Release decisions: **263 of 269 publishable** (`ok` at the robustness stage). The six withheld are exactly the #338 Byrne/RLM ports blocked at the **inputs** stage by the unresolved `basspel`/`woco`/`basnum` score bounds — the release contract working as designed, resolvable only by external documentation.
- All eight phoneme-blending link pairs (#608/#619) are fully publishable on both halves: itt-008/108, gf-006/306, lf-006/106, did-003/103, al-006/306, ca-007/307, dose-084/384, med-087/387.
- The #623 plan-currency check passes for all eight gated link pairs. It is **not** registry-clean: the #630 tail never ran (the execution record above says it stopped at 0/15), so thirteen fits still store the loader-filtered plan and two carry provenance blemishes. An earlier revision of this bullet claimed the opposite; that claim was wrong and is corrected here. The tail was run on 2026-08-27 — see `notes/202608271200-closing-584-588-residuals.md`.

## What moved scientifically

The per-model record is `notes/202608262100-refit-batch-movement-record.md`. The short version:

1. **Randomised-effect headlines are stable**: no ITT, DiD, joint or gain-family causal term changed direction or interval zero-crossing; the gain treatment marginals are byte-identical through the #575 fixes.
2. **Mediation firmed systematically** under the #585 common pre-exposure-vector fix: six totals (med-059/078, 064, 080, 086/186) now exclude zero, driven by larger letter-sound NIEs; med-062's composite-route NIE weakened to include zero; med-092 now headlines the supported period-1 window (+2.6 items, 89 % +1.0 to +4.3) with the all-period average flagged as extrapolation.
3. **The level family's arm-free standardised AME (#584 decision 1)** moved cards by ≤ 0.05 items except lf-002 (+0.17), lf-010 (+0.09) and lf-003 (+0.05); lf-010 (taught expressive vocabulary) now excludes zero. All eleven two-wave comparators (decision 3) pass and track the four-wave estimates; lf-006 is released for the first time via the lf-106 link pair (+0.6 ordinary vs +0.5 floor-link items).
4. **mech-191's sessions→word-reading association collapses** once the 28 rows without observed attendance are excluded per its #586 population decision (interquartile contrast +0.17 items, 89 % −0.8 to +1.4, from a previous +2.2 that excluded zero) — the batch's strongest single reversal: the apparent dose signal was carried by the excluded zero-attendance rows.
5. **The gain family's new period-1-only refit sensitivity** (#575 finding 2) shows the stacked likelihood's borrowing modestly inflates most language and taught-vocabulary estimates; gf-006 (B), gf-007 (F) and the taught-vocabulary pair gf-012/013 lose their 89 % zero-exclusion under period-1-only fitting, while word reading and letter sounds are robust either way. This is the evidence the deferred `beta_p1` redesign question (#575 decisions note) should be argued from.
6. **mech-157/090/188's headline changes are presentation-only** — the #602 interquartile-estimand switch; each fit's `secondary_observed_range` row reproduces its old headline exactly.
7. **mech-301 (the new Mundlak split)** is a standing caveat for the letter-sounds→word-reading mechanism story: the within-child slope alone is inconclusive (+0.35 items, 89 % −1.15 to +1.82).

`notes/202608182200-findings-by-question.md` has been updated in place against the fresh artefacts; the 2026-08-18 per-family findings series otherwise remains the narrative record of the pre-batch fits and should be read with the movement record beside it.

## Addendum, 2026-08-27 — what this batch left undone

> [!NOTE]
> Added by a LLM-based AI tool (Claude Code/Opus 5).

Two actions this note's movement record already listed as open, plus one it did not:

1. **`scripts/compare_statistical_models.py` was never re-run.** Every artefact in `output/statistical_models/comparison/` still dated from 2026-08-20/23 and reported superseded numbers — `mediation_family.csv` gave med-059's total as 2.061 where the refit gives 2.319 — and the per-fit copies of the mechanism LOO comparison, which the batch's directory resets removed, were never rewritten.
2. **`lrp-rlm-jc-002`'s stale qualification could not be fixed by re-evaluation.** The movement record correctly noted that jc-002 still called the now-publishable jc-102 withheld, but re-evaluating only moved the complaint: the binding also requires `data_sha256` on both fits, and `LongitudinalPanel` recorded none, so it could never open.
3. **The #630 tail.** Thirteen plan-stale fits plus `med-276` (dirty tree) and `med-387` (off-main commit).

All three are closed in the 2026-08-27 pass.
