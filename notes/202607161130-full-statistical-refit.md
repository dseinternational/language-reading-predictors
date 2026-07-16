# Full statistical-model refit sweep (reporting config)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

**Date:** 2026-07-16
**Driver:** coordinated refit of the entire Bayesian (Step 2) suite, requested to close out the reporting/interpret step of several tracked issues.
**Refs:** #340 (DiD re-specification), #312 (concurrent-associations family), #322 (undergraduate-readable reports), #338 (Byrne/RLM comparable-model suite).

## Why this sweep

Each of the four referenced issues lands code that only becomes _reportable_ once the models are re-fitted at the production (`reporting`) preset and their reports re-rendered:

- **#340** re-specified the waitlist-crossover / DiD family (`did-001`…`013`, `107`): arm-by-wave estimands, dose variants that separate treatment presence from attendance intensity, corrected DID-013 random-slope marginal, recalibrated intercept prior, and synchronised report/equation/catalogue prose. Its final open acceptance box is _"reporting/rep-lite fits are rerun after the changes; only convergence-gate-passing results are interpreted."_ This sweep is that step.
- **#312** added the `concurrent` family (`ca-001`…`006`): per-wave mutually adjusted conditional associations between contemporaneous skill levels. Purely descriptive (`Status.ASSOCIATION`); needs a gate-passing reporting fit and adoption of the #322 display conventions.
- **#322** made every report findings-first and data-space-led (predicted-scores panels, trajectory plots, labelled PPC overlays, computed key-findings box, gate badge). Those figures/prose are generated **at fit time**, so they only reach the public reports on a reporting sweep.
- **#338** shipped the Byrne/RLM Phase A window extension + ceiling sign-off and the Phase B/D models (`rlm-hg-*`, `rlm-jc-001`, `rlm-mm-001`, `rlm-adj-001`, `rlm-hs-001`). These are a separate (non-randomised) study — every term is descriptive/adjusted-associational.

## Scope

**159 registered models across 16 families** (the skill doc's "89" is stale). Families: `itt`, `joint`, `gain_factors`, `level_factors`, `did`, `mechanism`, `mediation`, `mediation_multi`, `aligned`, `dose_response`, `adjusted`, `corr_factor`, `horseshoe`, `lcsm`, `growth`, `historical_growth`, plus the new `concurrent` (`ca`) family and the RLM ports. (`bx`, `surv`, `lcf` also present.)

## Commands run

```bash
conda activate dse-language-reading-predictors
export QUARTO_PYTHON=<env python>            # else Quarto picks system python3 → No module named 'arviz'
python scripts/fit_statistical_model.py all --config reporting --render
python scripts/compare_statistical_models.py --config reporting
```

- **Config:** `reporting` = 6 chains × 6000 draws × 6000 tune, `target_accept = 0.95`.
- **Machine:** 16 cores, 336 GB free disk. Output root = repo-local `output/` (default).
- Run backgrounded with a log + exit-code marker; the sweep renders reports only after **all** fits finish (`fit_all --render` batches renders — an interrupted sweep leaves un-rendered dirs).

## Convergence gate (checked before interpreting anything)

`diagnostics_summary.json` → `passed` + `checks` = {rhat ≤ 1.01, ess ≥ 400, bfmi ≥ 0.30, divergences == 0}. The divergences check fails on _any_ divergence. Triage per METHODS / the skill:

- Divergence-only flags ≤ ~0.5 % of draws on GP/HSGP/dose geometries — within the ≤ 1 % guidance, usable with a noted caveat.
- Genuine concern = divergences **plus** sub-threshold BFMI (latent-factor funnel, e.g. `mm-001`) — structural legs held pending reparameterisation; correlations still fine.

Gate results are tabulated in the per-family findings notes; only gate-passing results are interpreted.

## Upload

Public research site (`dseresearch` container, anonymously readable) via the `AzureCliCredential` wrapper over the already-fitted dirs — the VM managed-identity path 403s. One `run_id` for the whole batch, traces excluded (default). Frank's `az login` (frank@dsegroup.net) carries the write role. Scope confirmed with the user (public + preliminary).

## Findings documentation

One dated findings note per family (16), each reading out the headline estimand from the family's result CSVs, following the METHODS reporting standard and the #179 evidence ladder (inconclusive / suggestive / moderate / strong / very strong at P ≥ 0.75 / 0.91 / 0.97 / 0.99), written to be accessible to undergraduate science and mathematics students.

## Bug found and fixed mid-sweep — `did-007` missing-dimension crash

`lrp-rli-did-007` (the period-varying DiD dose companion) **sampled fine** (14 divergences / 36 000 draws = 0.04 %, gate REVIEW — usable) but then **crashed** in the #320 key-findings natural-scale dose marginal:

```
ValueError: Dimensions {'phase'} do not exist. Expected one or more of ('dose_phase', 'sample')
```

`_write_dose_slope_summary` (`pipeline.py`) hardcoded `.transpose("phase", "sample")` on `beta_dose_phase`. That variable's period dimension is named **`phase`** in the `dose_response` family but **`dose_phase`** in the DiD dose companions — so the hardcoded spelling worked for `dose-*` but hard-crashed `did-007` before it could write its report. This is why `did-007` had a stale `diagnostics_summary.json` but no `.qmd`, and why #340's last acceptance box ("rerun reporting fits") had stayed unchecked. **Fix:** derive the period dim name instead of hardcoding it (`next(d for d in stacked.dims if d != "sample")`). Lint-clean. `did-007` re-fit individually after the sweep. Its pooled comparator `did-107` was unaffected (0 divergences, passed, rendered).

## Convergence-gate triage (interim, first 128 fits)

Only divergence-only flags, all far under the ≤ 1 % METHODS guidance, healthy R-hat/ESS/BFMI — usable with a caveat:

| model    | divergences | % of draws | min BFMI |
| -------- | ----------- | ---------- | -------- |
| did-007  | 14          | 0.039 %    | 0.53     |
| dose-077 | 17          | 0.047 %    | 0.86     |
| dose-083 | 4           | 0.011 %    | 0.51     |
| dose-084 | 2           | 0.006 %    | 0.68     |
| dose-177 | 13          | 0.036 %    | 0.86     |
| hs-001   | 2           | 0.006 %    | 0.72     |

(The `mm-001` corr_factor funnel case — the one expected genuine concern — had not yet fitted at this checkpoint; checked on completion.)

## Final outcome

- **159 / 159 models fitted and rendered** at `reporting` (all have `diagnostics_summary.json` + `index.html`).
- **Gate: 149 pass, 10 flagged.** Flags split into two groups:
  - **Divergence-only, usable with a caveat** (≤ 0.05 % of draws, healthy R̂/ESS/BFMI): `did-007`, `dose-077`, `dose-083`, `dose-084`, `dose-177`, `hs-001`.
  - **Latent-factor funnels — correlations robust, structural legs held** pending reparameterisation: `mm-001`, `mm-002` (R̂ 1.048, ESS 64 — worst), `mm-101`, `rlm-mm-001`.
- **Cross-model comparison** written to `output/statistical_models/comparison/`. The #340 discipline worked: gate-flagged fits (`did-007`, `dose-077`) were **excluded** from their LOO comparisons, so those nested comparisons were skipped by design.
- **Public upload:** all 160 dirs (159 models + comparison) published to the `dseresearch` **public** container via the `AzureCliCredential` wrapper (`AZURE_CLIENT_ID` unset). **run_id `019f6be6-5b26-7399-b95c-81690837e08b`**, traces excluded, 160/160 OK, 0 failures. Verified `200 text/html` (reports) and `200 image/png` (figures). Base URL: `…/public/projects/language-reading-predictors/output/019f6be6-5b26-7399-b95c-81690837e08b/<model>-reporting/index.html`.

## Findings notes (per family, undergraduate-accessible)

- [ITT](202607161800-findings-itt.md) — the headline causal analysis (27 models)
- [joint](202607161800-findings-joint.md) — all outcomes together + taught-vs-not-taught contrasts (4)
- [gain_factors](202607161800-findings-gain_factors.md) — ANCOVA second causal read (19)
- [level_factors](202607161800-findings-level_factors.md) — score levels at each wave (11)
- [did](202607161800-findings-did.md) — waitlist-crossover / difference-in-differences (14)
- [mechanism](202607161800-findings-mechanism.md) — skill→skill dose-response associations (12)
- [mediation](202607161800-findings-mediation.md) — g-formula decomposition, incl. two-mediator (16)
- [aligned](202607161800-findings-aligned.md) — onset-aligned per-protocol (9)
- [dose_response](202607161800-findings-dose_response.md) — outcome vs session dose (5)
- [adjusted](202607161800-findings-adjusted.md) — baseline predictors of gain (2)
- [concurrent](202607161800-findings-concurrent.md) — per-wave conditional associations, #312 (6)
- [measurement-models](202607161800-findings-measurement-models.md) — corr_factor + long_corr_factor (5)
- [horseshoe](202607161800-findings-horseshoe.md) — regularised predictor-ranking cross-check (5)
- [lcsm](202607161800-findings-lcsm.md) — latent change-score / cross-lagged (5)
- [growth](202607161800-findings-growth.md) — multivariate growth curves (3)
- [Byrne/RLM growth](202607161800-findings-byrne-rlm-growth.md) — historical_growth + historical_joint, #338 (10)
- [exploratory](202607161800-findings-exploratory-block-survival.md) — block_exposure + survival (6)

## Code change shipped in this sweep

`src/language_reading_predictors/statistical_models/pipeline.py` — `_write_dose_slope_summary` now derives the period dimension name instead of hardcoding `"phase"`, fixing the `did-007` crash (the DiD dose models name it `dose_phase`). Lint-clean. This is the correctness fix that unblocked #340's final acceptance box.

## Status log

- 2026-07-16 11:26 — first `all --render` run launched; reaped by the harness after 64 fits (background-job lifecycle limit).
- 2026-07-16 12:15 — relaunched as a **resumable** per-model driver (renders already-fitted dirs, fits the rest, skips completed). Ran to completion.
- 2026-07-16 ~12:18 — `did-007` crash diagnosed and fixed; re-fit cleanly post-sweep (exit 0, rendered).
- 2026-07-16 18:06 — cross-model comparison written; public upload of all 160 dirs completed (run_id `019f6be6…`).
- 2026-07-16 — 17 per-family findings notes written.
