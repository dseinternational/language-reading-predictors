<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

# Waitlist-crossover arm-by-wave family: reporting refit

- **Date:** 2026-07-15
- **Status:** run record
- **Issue:** #340 (acceptance criterion 8)

## What this closes

The estimand re-specification, dose redesign, DID-013 re-scoping, report/catalogue synchronisation, LOO gating and the new statistical-validation tests all landed in #343 (commit `ce73d29`) and were signed off as acceptance criteria 1–7. Criterion 8 — "reporting/rep-lite fits are rerun after the changes; only convergence-gate-passing results are interpreted" — was left open because #343 only produced dev smoke fits, whose convergence gates correctly reported REVIEW. This note records the production refit that closes it.

## Run

- **Config:** `reporting` (6 chains × 6000 draws × 6000 tune, `target_accept = 0.95`) for all 14 models; `lrp-rli-did-007` re-fit at `target_accept = 0.99` (see below).
- **Scope:** the full waitlist-crossover family — `lrp-rli-did-001`–`013` and `107`.
- **Fitted:** 14 / 14. **Convergence gate: 14 / 14 PASS** (R-hat ≤ 1.01, ESS ≥ 400, BFMI ≥ 0.30, 0 divergences).
- On the first pass `lrp-rli-did-007` (period-resolved letter-sound session dose) **failed** the convergence gate. The gate requires **zero** divergences to PASS, and this fit produced 25 (of 36 000 draws), with R-hat, ESS and BFMI all passing — the same geometry the issue flagged from the 13 July snapshot (28 divergences). Because criterion 8 interprets only gate-passing fits, this divergent fit is discarded, not interpreted or reported as within-policy. Re-fitting at `target_accept = 0.99` removed every divergence, so the fit now clears the 0-divergence gate; that refit is the one summarised below.

## Headline causal contrasts (`tau_t2`, the clean randomised t2 arm gap)

`tau_t2` is the only causal quantity in each arm-by-wave model — the immediate-minus-waitlist contrast at t2, before the waitlist arm crosses over. Positive favours the intervention. Item-scale figures are standardised over the documented fitted rows; direction labels follow the issue #179 evidence ladder.

| Model   | Outcome                         | `tau_t2` (logit, 95% CrI) | P(τ>0) | Item scale        | Evidence (direction) |
| ------- | ------------------------------- | ------------------------- | ------ | ----------------- | -------------------- |
| did-002 | L letter-sound knowledge        | +0.60 (+0.11, +1.07)      | 0.991  | +3.5 of 32 items  | very strong          |
| did-003 | B phoneme blending              | +0.42 (−0.06, +0.89)      | 0.956  | +0.9 items        | moderate             |
| did-004 | TE taught expressive vocabulary | +0.32 (−0.07, +0.71)      | 0.949  | +1.5 items        | moderate             |
| did-001 | W word reading                  | +0.34 (−0.13, +0.81)      | 0.920  | +2.2 items        | moderate             |
| did-013 | W word reading (het catch-up)   | +0.34 (−0.14, +0.80)      | 0.918  | +2.2 items        | moderate             |
| did-008 | TR taught receptive vocabulary  | +0.23 (−0.12, +0.57)      | 0.902  | +1.2 items        | suggestive           |
| did-010 | F basic concept knowledge       | +0.16 (−0.21, +0.54)      | 0.809  | +0.6 items        | suggestive           |
| did-012 | N nonword reading (off-floor)   | +0.30 (−0.45, +1.03)      | 0.788  | +0.058 risk diff. | suggestive           |
| did-011 | P phonetic spelling (off-floor) | +0.16 (−0.64, +0.96)      | 0.652  | +0.023 risk diff. | inconclusive         |
| did-009 | E standardised expressive vocab | +0.03 (−0.20, +0.25)      | 0.608  | +0.8 items        | inconclusive         |
| did-005 | R receptive vocabulary          | −0.00 (−0.21, +0.21)      | 0.492  | −0.1 items        | inconclusive         |

The pattern is coherent with the randomised ITT and the gain-factor ANCOVA: strong-to-very-strong signal on the phonics-proximal outcomes (L, B, W) and taught expressive vocabulary (TE), fading to inconclusive on broad standardised vocabulary (R, E). `did-013`'s `tau_t2` reproduces `did-001` as expected — the heterogeneity variant adds only a waitlist-only catch-up deviation and shares the arm-by-wave core. Agreement of `tau_t2` with the ITT is a shared-data parameterisation check, not independent replication, exactly as the reports now state. The floored P/N models report off-floor **prevalence** risk differences, not floor-exit transitions.

The `arm_gap_t3` and `delta_crossover` quantities are reported alongside but flagged associational (post-crossover histories differ by ~20 weeks of exposure); `arm_gap_t1` is a pre-treatment balance quantity.

## Dose associations (observational)

`did-006` (pooled session dose, W), `did-007` (period-resolved dose, L) and `did-107` (pooled-slope comparator, L) all separate the treatment-presence term from treated-centred attendance intensity and adjust for randomised arm, the pre-randomisation t1 outcome and t1 age. All slopes are weakly positive with 95% CrIs spanning zero — e.g. `did-007` letter-sound dose: overall +0.13 logit (P = 0.77), period-1 +0.16 (P = 0.84), period-2 +0.11 (P = 0.87). They remain labelled adjusted associations (latent ability and time-varying attendance confounding), never causal.

## DiD-dose LOO comparison

`did_dose_loo_compare.csv` (did-107 vs did-007) was regenerated with `scripts/compare_statistical_models.py --config reporting`. The gate roll-call printed **14/14 PASS**, so no gate-failing fit entered any comparison. The comparison records `row_identity = obs_id` and `comparison_valid = True` (identical, identically ordered analysis rows, verified pointwise rather than by row count). The pooled slope (did-107) ranks first, but `|elpd_diff| < 4` (elpd_diff ≈ −1.0, dSE ≈ 0.74): the two dose parameterisations are indistinguishable in predictive fit. The comparison honestly flags 4–5 observations with Pareto k̂ > 0.70, so the ELPD is unreliable and this is read as "no meaningful difference between dose parameterisations", not a model-selection verdict. The artefact is copied beside both model reports so the model partial renders it.

## Cell-stratified posterior-predictive checks

Every fit wrote `did_cell_ppc.csv` and a cell-stratified PPC plot over the six arm × wave cells. The redesigned model reproduces the descriptive means the issue cited (immediate t1/t2/t3 ≈ 5.86 / 10.50 / 14.86; waitlist ≈ 6.88 / 8.92 / 13.36) with no mean tail-flags. A few word-reading cells show zero-rate tail-flags (the Beta-Binomial slightly under-/over-predicts exact-floor counts at t1/t2 for the immediate arm) — a floor feature surfaced by the check, not a convergence problem.

## Incidental fix

The `--render` step of both fit scripts invoked `quarto render` with the inherited environment, so Quarto resolved its own `python3` from PATH. When the conda env is not the first interpreter on PATH (it is under the documented `conda activate` workflow, but not in every runner), the report cells failed with `ModuleNotFoundError: No module named 'arviz'` even though the fit and gate had passed. Both `scripts/fit_statistical_model.py` and `scripts/fit_model.py` now pin `QUARTO_PYTHON` to `sys.executable` for the render subprocess, so the report is always executed by the same interpreter that ran the fit. Artefacts (trace, CSVs, diagnostics) were unaffected — they are written before the render — so this changed only report generation, not any estimate.

## Publication

Artefacts remain local (`output/statistical_models/`, gitignored). Publishing to the public research site is preliminary-scope and deferred to a separate, explicitly confirmed step.
