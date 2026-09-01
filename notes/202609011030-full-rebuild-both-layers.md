# Full rebuild of both model layers, 2026-09-01

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

This batch is not a refresh of stale fits. The repository's `output/` tree, its `.venv` and its `node_modules` were all **absent** at the start — no stored fits, no artefacts, and no backup anywhere on the machine (searched). Every number below therefore comes from a fit performed in this batch, and there is no "before" on disk to diff against, so this note carries no movement record. What it carries instead is one independent reproduction check (see **Validation**).

The batch resolves #635 by construction: every fit reads the post-#631 ERB quarantine, so no stored fit predates it.

## Execution record

- **Environment rebuilt** — `uv sync`, `npm ci`; full `pytest` suite green at `0819e59d` before any compute.
- **Gradient boosting**: 50/50 at `reporting`, one resumable per-model driver stream, 1 h 26 m, 0 failures.
- **Statistical**: 269/269 at `reporting`, two disjoint driver streams (6 chains x 6 cores each), 4 h 59 m and 3 h 55 m.
- The tree was held clean for the whole sweep, so **no fit records `dirty: true`**. The two code fixes below were developed in separate worktrees precisely to keep it so.

## Two code defects, found by the sweep

Both were latent on `main` and neither is a sampling problem. Together they accounted for all 122 apparent sweep failures.

**1. `odds_string` (PR #649, 114 reports).** PR #641's reporting split left `statistical_models.reporting` a re-export facade. It kept `evidence_label` and `favoured_direction` but dropped `odds_string`, which five result partials import, so every `itt`, `did`, `block_exposure`, `gain_factors` and `level_factors` report died with `ImportError`. **Fits were unaffected** — the trace, CSVs, diagnostics and release decision are all written before the render step — so this needed a re-render, not a refit.

**2. Eleven undeclared inline priors (PR #650, 8 models).** PR #647 missed eleven priors across four factories. Since #640 removed the name-and-scale classifier, an undeclared free RV is a hard stop, so `lcsm-081/082/091/181`, `mech-302/303`, `med-092` and `mm-101` could not fit **at all**.

Every miss shares a shape: the RV is created as a bare expression — in a dict, a comprehension, or an `else:` branch — behind an opt-in the rest of its family does not take. So each family's other members fit and nothing looked broken.

Both survived review for the same reason, and it is worth naming: **the guards existed but their fixtures did not exercise the real registered configurations.** `test_every_free_variable_carries_a_prior_descriptor` built `build_lcsm_model(panel)` with bare defaults, so none of the eleven sites was ever constructed; and no import-graph check over `src/` sees the report templates at all, because they are not Python modules. Both PRs replace the fixture list with checks derived from the real tree, and both regression tests were verified to fail without their fix.

## Remediations, each following recorded precedent

| Model      | Problem                                                                              | Resolution                                                                                    |
| ---------- | ------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- |
| `lcsm-081` | R-hat 1.0119 on five per-child latent offsets (`z1_W[...]`); 0 divergences, ESS 1302 | refit at `target_accept` 0.99 -> R-hat 1.0078, passes                                         |
| `mech-204` | 1 divergence in 36 000 draws, everything else clean                                  | refit at `target_accept` 0.98 -> 0 divergences                                                |
| `did-007`  | sweep cells unconverged                                                              | `--cell-target-accept 0.99`, the escalation `notes/202608262100` records for the same fit     |
| `itt-010`  | mandatory full-cohort missingness bundle absent                                      | UKDS ReShare archive re-installed (checksum-pinned) and refit with `--rli-randomised-archive` |

No escalation was applied blanket-wise; each is strictly above that fit's own recorded value, and each is recorded in its `config.json`.

## Evidence loop

Rebuilt from scratch, since the sweep directories went with `output/`: the DiD, gain-factor and level-factor treatment-prior sweeps (all cells converged, attached); the floored P/N release grid for `itt-009`/`itt-011`; the trace-backed phoneme-blending link archive from the fresh `itt-008`/`itt-108` traces; the dispersion sweep (12 cells); the horseshoe ranking sweep (20 cells).

`compare_statistical_models.py` wrote all 18 artefacts; `compare_gb_vs_statistical.py` and the four `compare_horseshoe_vs_gb.py` pairs followed. The horseshoe/GB pairings are the ones each `hs-*` module names for itself (`hs-001`/`gbg-012`, `hs-002`/`gbl-012`, `hs-003`/`gbg-009`, `hs-004`/`gbl-009`); their top-3 construct overlaps are 2/3, 2/3, 1/3 and 2/3, so the Bayesian sparse-regression ranking broadly corroborates the boosting ranking without matching it term for term. The L x N nested LOO (`joint_readiness_lxn_w_loo_compare.csv`) is again `comparison_valid=True` via `psis+reloo`, verdict inconclusive (|elpd_diff| < 4) — the same answer the 2026-08-27 run reached.

## Final state

- **269/269 statistical fits, 50/50 GB models**, all with config, trace, key findings and a rendered report.
- **0 convergence-gate failures. 0 fits with any divergence. 0 fitted from a dirty tree.**
- **263 publishable / 6 withheld.** The 6 are exactly the #338 Byrne/RLM ports blocked at the `inputs` stage on unconfirmed `basspel` / `woco` / `basnum` denominators — the release contract working as designed, and the same 6 as the 2026-08-26 batch.
- Rendered output corresponds exactly to the decisions: precisely 6 reports show "Findings withheld", and they are those 6. No rendered report contains an import error or traceback.
- **Commit split: 261 fits at `0819e59d`, 8 at `96f3b966`.** The 8 are the models PR #650 unblocked. The difference between the two commits is a reporting-facade re-export and prior _descriptor_ metadata; neither changes any posterior. Signed off as an accepted two-commit batch rather than re-sweeping 261 fits that would return byte-identical results.

## Validation

`lrp-rli-med-059`'s total effect comes out at **2.3195 words**, matching the **2.319** published by the 2026-08-27 closing pass. #635 had measured `med-059` as invariant under the ERB quarantine, so it is exactly the model that should reproduce — and it does, from an artefact tree that was destroyed and rebuilt. This is the batch's evidence that the seeded samplers reproduce prior results on unchanged code paths.

## Open residuals

1. **14 fits record no `data_sha256`** — `surv-009/011`, the six `rlm-adj-*`, `rlm-ca-001/002`, `rlm-hs-001/002/003`, `rlm-mm-001`. Nothing consumes the field for these families and all are publishable, so this is a provenance gap, not a correctness one. Note this is a _different_ set from the one `notes/202608271200` predicted: its `LongitudinalPanel` fix worked, and the nine `historical_growth` fits and all three `jc` fits now carry a checksum, closing that residual.
2. **5 of 20 horseshoe prior-sensitivity cells are unconverged** (1-33 divergences, on `hs-002`, `hs-004`, `rlm-hs-001`). The four RLI horseshoe primaries remain publishable; this limits the ranking cross-check, not the primaries. There is no prior batch record to compare against.
3. **The gain reports cite a ranking view that is never produced for them.** `gbg-017/018/020/021` tell readers a same-skill-excluded ranking (`ranking_excluding_same_skill.csv`) separates same-instrument from cross-domain signal, but `SAME_SKILL_SIBLINGS` is keyed on the _target_ and gain targets are deliberately absent, with the reasoning recorded in the table ("the baseline level is the regression-to-the-mean anchor, not contamination"). The two concerns differ — the table addresses contamination of the target, the prose addresses same-instrument _predictors_ dominating the ranking, which is a live concern for a gain model too. Left for a decision rather than silently patched. The 7 level models do get the view; all 11 cited ranking directories were regenerated.
4. **The ERB source-archive question remains open** (#631 follow-up). The t4 record for `ID_FDCBDCF29AC0BF03` is still unverified against the source archive, so this batch fits against the **quarantined (missing)** values, which is what current code does. If the archive is corrected, these fits should be re-run against corrected values.

Refs #635, #637; PRs #649, #650.
