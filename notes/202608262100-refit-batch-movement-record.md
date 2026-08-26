> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5).

Companion to `notes/202608262100-full-refit-batch.md` — the per-model numeric movement record behind that note's summary. OLD fits are preserved locally at `output/statistical_models/models.pre-batch-20260826/` (not committed); NEW fits are the current `output/statistical_models/models/` artefacts at the batch commits.

# Pre/post refit-batch movement report (2026-08-26)

OLD = `output/statistical_models/models.pre-batch-20260826/<id>-reporting/`, NEW = `output/statistical_models/models/<id>-reporting/`. 248 reporting fits exist on both sides; 21 are NEW-only first fits (did-103..106, lf-201..211, mech-301..305, rlm-jc-102); none are OLD-only. Reading key: seeded samplers make unchanged code paths reproduce **byte-identical** posteriors, so any movement below is code-driven (estimand/data/leg fixes), not MCMC noise — confirmed by exact old=new matches (gf treatment marginals, med-387, lf-006 card, mech-090/157/188 secondary rows).

## 1. GAIN (#575) — period-1 sensitivity, marginals

**Period-1-only refit vs stacked primary** (NEW `period1_sensitivity.csv`; stacked 153–161 rows, period-1-only 50–54; every period-1 refit `converged=True`). Items for gf-005/011 are off-floor probability differences, not counts.

| Model (sym)               | stacked β_trt [89%]    | stacked items | p1-only β_trt [89%]    | p1-only items [89%]    | items shift |
| ------------------------- | ---------------------- | ------------- | ---------------------- | ---------------------- | ----------- |
| gf-001 (W)                | 0.394 [0.135, 0.650]   | 2.63          | 0.506 [0.202, 0.805]   | 3.47 [1.40, 5.44]      | +0.84       |
| gf-002 (R)                | -0.069 [-0.200, 0.061] | -2.02         | -0.015 [-0.170, 0.144] | -0.42 [-4.96, 4.19]    | +1.60       |
| gf-003 (E)                | 0.037 [-0.089, 0.162]  | 0.97          | 0.004 [-0.155, 0.164]  | 0.10 [-4.00, 4.22]     | -0.87       |
| gf-004 (L)                | 0.568 [0.267, 0.869]   | 3.33          | 0.592 [0.228, 0.944]   | 3.36 [1.28, 5.38]      | +0.04       |
| gf-005 (P, off-floor)     | -0.098 [-0.758, 0.555] | -0.012        | -0.061 [-0.720, 0.597] | -0.009 [-0.114, 0.094] | +0.002      |
| gf-006 (B)                | 0.394 [0.041, 0.750]   | 0.84          | 0.325 [-0.081, 0.727]  | 0.66 [-0.16, 1.48]     | -0.17       |
| gf-007 (F)                | 0.257 [0.003, 0.515]   | 0.99          | 0.221 [-0.093, 0.524]  | 0.78 [-0.33, 1.84]     | -0.22       |
| gf-008 (T)                | 0.082 [-0.100, 0.264]  | 0.61          | 0.108 [-0.096, 0.313]  | 0.80 [-0.71, 2.32]     | +0.19       |
| gf-009 (TR)               | 0.180 [-0.038, 0.401]  | 0.96          | 0.161 [-0.088, 0.410]  | 0.85 [-0.46, 2.16]     | -0.11       |
| gf-010 (TE)               | 0.227 [-0.019, 0.474]  | 1.07          | 0.188 [-0.093, 0.469]  | 0.88 [-0.43, 2.18]     | -0.19       |
| gf-011 (N, off-floor)     | 0.119 [-0.515, 0.749]  | 0.020         | 0.083 [-0.582, 0.742]  | 0.014 [-0.095, 0.123]  | -0.006      |
| gf-012 (TR + broad-vocab) | 0.221 [0.005, 0.437]   | 1.16          | 0.189 [-0.061, 0.440]  | 0.99 [-0.32, 2.31]     | -0.17       |
| gf-013 (TE + broad-vocab) | 0.255 [0.017, 0.499]   | 1.19          | 0.226 [-0.059, 0.514]  | 1.05 [-0.27, 2.38]     | -0.13       |
| gf-306 (B floor-link)     | 0.470 [-0.056, 1.01]   | 0.50          | 0.344 [-0.230, 0.920]  | 0.34 [-0.23, 0.91]     | -0.15       |

Verdict sentence: borrowing is mildly unfavourable (period-1-only median smaller than stacked) for E, B, F, T-excluded taught/language outcomes gf-006/007/009/010/012/013, N and gf-306, and four of those — **gf-006 (B), gf-007 (F), gf-012 (TR), gf-013 (TE)** — see their stacked zero-excluding 89% β_trt interval come to include zero under period-1-only fitting; medians move little, so the crossings are driven mostly by the ~3× smaller n, while W, R and L move the other way (period-1-only larger/less negative) and W and L stay firmly positive either way.

**OLD vs NEW `treatment_marginal.csv`**: numerically unchanged. All 14 models' trt_items_median and 89% CI are byte-identical except gf-005, which moved by -0.00027 off-floor probability (post-mask refilter recomputation); e.g. gf-001 stays 2.63 [0.899, 4.30], P(>0) 0.992. NEW adds ESS/MCSE columns (bulk ESS ~1.2e4). The post-mask refilter and marginals-arithmetic changes did **not** move any headline treatment estimate.

**Association marginal labels (gf-001 example)**: OLD used one "+5 items" step for every cross-measure; NEW uses per-measure steps with rescaled values — own "+8 items" 4.93 [4.36, 5.51] (was "+5" 3.25), TR/TE "+2", R/E "+17", L "+3", N/B "+1". Most informative fix: the N row was "+5 items" 7.4 [-0.9, 17.7] (an absurd extrapolation for a mostly-floored predictor) and is now "+1 item" 0.30 [-0.04, 0.65]. sd_items also changed (25.46 → 25.78) from the refiltered scaling rows.

## 2. MEDIATION (#585) — 16 mediation + 4 mediation_multi (060/064/066/075)

Words/items scale, median [89%], P(>0). Zero-crossing changes marked ★. med-078/186/187 tables carry only the `total` decomposition row on both sides (interventional relabellings).

| Model   | row      | OLD                       | NEW                      | note                                                       |
| ------- | -------- | ------------------------- | ------------------------ | ---------------------------------------------------------- |
| med-059 | total    | 2.06 [-0.18, 4.23] 0.93   | 2.32 [0.15, 4.53] 0.95   | ★ stops including zero — **the #585 prediction, verified** |
| med-059 | NIE      | 1.79 [0.66, 3.37] 1.00    | 2.10 [0.86, 3.72] 1.00   | firmed                                                     |
| med-060 | total    | 1.72 [-0.82, 4.18] 0.86   | 2.33 [-0.23, 4.87] 0.93  | still crosses                                              |
| med-062 | NIE      | 0.86 [0.013, 2.14] 0.95   | 0.82 [-0.13, 2.17] 0.92  | ★ now includes zero (weakened)                             |
| med-062 | total    | 1.81 [-0.41, 3.99] 0.90   | 1.97 [-0.22, 4.11] 0.93  | crosses both                                               |
| med-064 | total    | 2.17 [-0.25, 4.65] 0.92   | 2.54 [0.09, 5.02] 0.95   | ★ stops including zero                                     |
| med-066 | total    | 1.95 [-0.43, 4.25] 0.91   | 2.21 [-0.10, 4.51] 0.94  | still crosses (barely)                                     |
| med-068 | total    | 2.29 [0.10, 4.47] 0.95    | 2.39 [0.20, 4.56] 0.96   | excl. both                                                 |
| med-074 | total    | 2.87 [0.92, 4.83] 0.99    | 2.89 [0.82, 4.88] 0.99   | stable                                                     |
| med-075 | total    | 1.90 [-0.45, 4.19] 0.90   | 2.22 [-0.10, 4.52] 0.94  | still crosses                                              |
| med-076 | total    | 1.22 [-2.30, 4.69] 0.72   | 1.71 [-1.65, 5.06] 0.79  | NIE 3.07→3.57                                              |
| med-078 | total    | 2.06 [-0.18, 4.23] 0.93   | 2.32 [0.15, 4.53] 0.95   | ★ (relabel of 059)                                         |
| med-079 | total    | 2.23 [0.41, 4.00] 0.98    | 2.19 [0.40, 4.03] 0.97   | stable                                                     |
| med-080 | total    | 2.28 [-0.014, 4.53] 0.94  | 2.60 [0.17, 5.08] 0.96   | ★ stops including zero                                     |
| med-086 | total    | 0.092 [-0.006, 0.19] 0.93 | 0.107 [0.006, 0.21] 0.96 | ★ (pp scale) stops incl. zero                              |
| med-087 | total    | 0.68 [-0.16, 1.50] 0.90   | 0.68 [-0.15, 1.50] 0.90  | stable                                                     |
| med-092 | total    | 3.10 [1.04, 5.04] 0.99    | 2.63 [0.95, 4.28] 0.99   | headline re-scoped (below)                                 |
| med-176 | total    | 1.84 [-0.13, 3.76] 0.93   | 1.83 [-0.23, 3.81] 0.92  | stable                                                     |
| med-186 | total    | 0.092 [-0.006, 0.19] 0.93 | 0.107 [0.006, 0.21] 0.96 | ★ (relabel of 086)                                         |
| med-187 | total    | 0.67 [-0.15, 1.50] 0.91   | 0.68 [-0.15, 1.50] 0.90  | stable                                                     |
| med-276 | total    | 1.13 [-0.96, 3.21] 0.81   | 0.91 [-1.29, 3.09] 0.75  | weakened                                                   |
| med-387 | all rows | identical                 | identical                | already fitted post-fix (#619 pair)                        |

Net: six headline totals (059/078, 064, 080, 086/186) **lose** their zero-crossing after the common pre-exposure-vector fix — a systematic firming, driven by larger letter-sound NIEs; one NIE (med-062, Gaussian reading-route composite) **gains** a crossing. NDEs barely move.

**MED-092**: NEW headline is confirmed the period-1 window — `key_findings` headline "+2.6 items (89% +1.0 to +4.3)" matches the period-1 `mediation_summary.csv` total 2.63 [0.95, 4.28], P(>0) 0.994; `config.extra.supported_periods=[1]` (period 1: 28 treated / 25 untreated; periods 2–3: treated-only) with `period_treatment_support` recorded. The all-period average is written separately (`mediation_summary_all_periods.csv`): total 3.22 [1.15, 5.09], P(>0) 0.993 — flagged as extrapolation.

## 3. LEVEL (#584) — lf-001..011 AME card, two-wave comparators

**Four-wave card, OLD (typical-ability conditional) vs NEW (arm-free standardised AME)** — items at t2, [89%], pd:

| Model (sym) | OLD                        | NEW                        | change in median                               |
| ----------- | -------------------------- | -------------------------- | ---------------------------------------------- |
| lf-001 (W)  | 2.30 [0.26, 4.32] 0.96     | 2.28 [0.30, 4.38] 0.97     | -0.02                                          |
| lf-002 (R)  | 0.23 [-4.22, 4.65] 0.54    | 0.40 [-3.52, 4.44] 0.57    | **+0.17 (> 0.05)**                             |
| lf-003 (E)  | 0.14 [-3.89, 4.17] 0.52    | 0.19 [-3.11, 3.52] 0.54    | **+0.05 (borderline)**                         |
| lf-004 (L)  | 2.84 [0.82, 4.86] 0.99     | 2.86 [0.80, 4.87] 0.99     | +0.01                                          |
| lf-005 (P)  | 0.002 [-0.082, 0.086] 0.51 | 0.005 [-0.066, 0.077] 0.55 | +0.004                                         |
| lf-006 (B)  | 0.64 [-0.15, 1.42] 0.90    | 0.64 [-0.15, 1.42] 0.90    | 0.000 (identical fit)                          |
| lf-007 (F)  | 0.80 [-0.25, 1.85] 0.89    | 0.79 [-0.27, 1.83] 0.89    | -0.005                                         |
| lf-008 (T)  | 0.64 [-1.04, 2.28] 0.73    | 0.63 [-1.00, 2.28] 0.73    | -0.002                                         |
| lf-009 (TR) | 1.16 [-0.26, 2.57] 0.90    | 1.19 [-0.09, 2.46] 0.93    | +0.04                                          |
| lf-010 (TE) | 1.30 [-0.05, 2.65] 0.94    | 1.39 [0.08, 2.74] 0.95     | **+0.09 (> 0.05); interval now excludes zero** |
| lf-011 (N)  | 0.029 [-0.084, 0.14] 0.66  | 0.031 [-0.078, 0.14] 0.67  | +0.002                                         |

Three exceed the #584 note's ~0.05-item expectation: lf-002 (+0.17), lf-010 (+0.09), lf-003 (+0.05, borderline) — all vocabulary-family outcomes where ability moderation is largest; and lf-010 (taught expressive vocab) changes its zero-crossing (headline now "+1.4 items, 89% +0.1 to +2.7"). lf-006's card is numerically unchanged; what changed is its release status (see §6).

**NEW two-wave (t1/t2) comparators beside four-wave** (all 11 pass the gate):
lf-201 2.46 [0.49, 4.55] vs 2.28 | lf-202 0.53 [-4.05, 5.26] vs 0.40 | lf-203 -0.07 [-3.41, 3.30] vs 0.19 | lf-204 2.89 [0.91, 4.87] vs 2.86 | lf-205 0.011 [-0.087, 0.11] vs 0.005 | lf-206 0.76 [-0.03, 1.55] vs 0.64 | lf-207 0.73 [-0.42, 1.84] vs 0.79 | lf-208 0.60 [-1.01, 2.22] vs 0.63 | lf-209 1.26 [-0.07, 2.56] vs 1.19 | lf-210 1.33 [-0.05, 2.73] vs 1.39 | lf-211 0.040 [-0.077, 0.16] vs 0.031. Two-wave estimates sit within a few tenths of an item of the four-wave ones everywhere (largest gap E, -0.07 vs +0.19 — both null); no comparator overturns a four-wave conclusion, though lf-210's two-wave interval crosses zero where the four-wave lf-010 no longer does.

**lf-006 + lf-106 (both links)**: ordinary logit +0.6 items [-0.1, +1.4], pd 0.90; guessing-floor link +0.5 items [-0.1, +1.0], pd 0.92. Same direction, floor link slightly smaller/tighter; released as a pair.

## 4. MECHANISM (#586)

**mech-063/163 (L→W GP readiness, 151→155-row refits, both gates pass).** OLD's single summary row was the observed-range contrast; NEW adds the interquartile headline:

- mech-063: OLD obs-range (2→32 L) 5.19 [0.74, 10.4] P 0.975 → NEW obs-range 5.72 [1.22, 10.7] P 0.984; NEW IQR headline (17→28 L) 2.07 [0.05, 4.03] P 0.95.
- mech-163: OLD obs-range 4.83 [0.52, 9.59] P 0.970 → NEW obs-range 5.29 [0.99, 9.90] P 0.982; NEW IQR headline 2.34 [0.39, 4.22] P 0.975.

**L×N joint-readiness comparison: not yet valid.** `comparison/joint_readiness_lxn_w_loo_compare.csv` is dated Aug 23 (pre-batch) and still records `comparison_valid=False`: Pareto-k 0.90/0.97 and "rebuilt frame has 155 rows but the stored trace has 151 … refusing to refit". That blocker is now removed (stored traces have 155 rows), but the comparison has not been re-run; NEW fits still carry one Pareto-k ≈ 0.96–0.97 observation each, so a re-run will need the now-available exact-refit (reloo) repair. Action: re-run `scripts/compare_statistical_models.py`. (The L×B analogue, mech-061/161, is valid via psis+reloo: |elpd_diff| = 2.0 ± 1.3, inconclusive.)

**mech-158 vs mech-058**: resolved run plans now differ **only** in `model_id` and the declared missing-data fields — `require_observed`: 058 `[]` vs 158 `['hs', 'deapp_c']`. Confirmed as intended.

**mech-191 (sessions→W GP knee test)**: population changed from 156 rows / 53 children, exposure 0–94 (mean 54.3) to **128 rows / 52 children, exposure 10–94 attend raw-score units (mean 66.2)** — the NEW resolved plan adds `exposure_positive_only: true`, excluding the 28 zero-attendance rows. The association collapses: OLD headline (obs-range) 2.18 [0.25, 4.24] P 0.972 → NEW obs-range 0.75 [-0.72, 3.97] P 0.77; NEW IQR headline 0.17 [-0.83, 1.43] P 0.64. The apparent dose signal was carried by the excluded rows.

**mech-301..305 (first fits; all gates pass, key_findings ok):**

- mech-301 — Between/within (Mundlak) split, L→W: within-child IQR contrast **+0.35 items [-1.15, 1.82], P 0.65** (n=156) — the within-child slope alone is inconclusive.
- mech-302 — Phase-varying slope, L→W: +2.68 [1.65, 3.67], P 1.00 (n=156).
- mech-303 — Phase-varying slope, L→R: +3.76 [2.03, 5.53], P 1.00 (n=159).
- mech-304 — Dispersion-prior sensitivity, L→W: +2.56 [1.60, 3.51], P 1.00 — matches 302.
- mech-305 — Dispersion-prior sensitivity, L→R: +4.02 [2.32, 5.74], P 1.00 — matches 303.

## 5. JOINT (#588)

**rlm-jc-102 (first fit) vs rlm-jc-002**: numerically the sensitivity **confirms** jc-002's within-scale verdict — per-measure sigma_within summaries are near-identical (basread 0.315 vs 0.316, resolvable True both; bpvs 0.032 both, P(>min) 0.28 vs 0.29, unresolvable both; basdig 0.044 both, P(>min) 0.44 both, unresolvable both). Both gates pass; both headline "no measure pair had both wave-specific residual SDs supported above 0.05 logits". **However** jc-002's `release_decision.publication_qualification` still reads: "the registered within-scale prior sensitivity (lrp-rlm-jc-102) is not release-ready beside this fit (its own release decision withholds publication), so which measures clear the resolvability threshold — and therefore which correlations may be read at all — is a conclusion under this fit's prior alone. This fit's own power scaling already flags that prior: the largest sigma_within prior sensitivity is 0.62, against ArviZ's 0.05 flag threshold." That cross-reference is **stale batch-ordering**: jc-102's own final `release_decision.json` says `publishable: true` (stage robustness, ok) — jc-002 was fitted before jc-102 finished. Re-evaluate jc-002's release decision (`release.evaluate_publication`, no refit needed) so the qualification reflects the now-publishable companion.

**itt-215/216/315 dependence notes**: the point-invariance claim is gone. OLD: "…the point estimate should agree; the interval and P(>0) may move…". NEW: "…this fit's per-child logistic-normal offset makes its average marginal effect a latent-conditional estimand rather than the parent's, so agreement of the point estimates is an empirical finding (medians move by 0.0001-0.0011 on the proportion-correct scale), not a mathematical invariant…" (same wording pattern in all three).

**jm-001/002**: `joint_mechanism_wave_eligibility.csv` is present in jm-001 (absent from OLD): all four waves fitted (`fitted=True`, none skipped), 53/53/53/52 fitted rows at t1–t4, one row dropped by wave eligibility at t1–t3 and none at t4, t3 hosting the fit artefacts. Wave verdict (jm-001 headline): Δ = β(LS→N) − β(LS→W) is t1 -0.47, t2 -0.17, t3 -0.26, t4 +0.06 logit/SD, all four waves reported with no single headline wave. jm-002 (phase-stacked ANCOVA variant) writes no wave-eligibility artefact — nothing per-wave to gate — and its identified contrast is Δ = +0.81 [89% +0.49, +1.13] logit/SD.

## 6. CROSS-CUTTING — key_findings.json, 248 models on both sides

- Status **ok on both sides: 241 / 248**. Status changed: **1** — lf-006 `robustness_unresolved → ok` (the #619 B-pair with lf-106 now satisfies the release gate; its numbers are unchanged). The other 6 non-ok are `inputs_unresolved → inputs_unresolved`, all Byrne/RLM ports awaiting historical inputs (rlm-adj-001, rlm-hg-002/003/008, rlm-hs-001, rlm-mm-001).
- No OLD artefact/schema breakages: every model had a readable key_findings.json on both sides (older CSV schemas — e.g. mechanism_summary without `contrast`/`n_obs`, treatment_marginal without ESS columns — were handled; noted, not errors).
- Headline **direction or zero-crossing changes (12 models)** — mech-090/157/188's are presentation-only (#602 IQR headline switch; each NEW `secondary_observed_range` row reproduces the OLD headline exactly):
  - **med-059, med-078**: "+2.1 items (-0.2 to +4.2)" → "+2.3 items (+0.1 to +4.5)" — crossing lost (real, #585).
  - **med-064**: "+2.2 (-0.2 to +4.6)" → "+2.5 (+0.1 to +5.0)" — crossing lost (real).
  - **med-080**: "+2.3 (-0.0 to +4.5)" → "+2.6 (+0.2 to +5.1)" — crossing lost (real).
  - **med-086, med-186**: "+9.2 pp (-0.6 to +19.1)" → "+10.7 pp (+0.6 to +21.1)" — crossing lost (real).
  - **lf-010**: "+1.3 items (-0.1 to +2.7)" → "+1.4 items (+0.1 to +2.7)" — crossing lost (real, #584 AME).
  - **lf-006**: empty OLD headline → "+0.6 items (-0.1 to +1.4)" (release-gate unblock, not a numeric change).
  - **mech-191**: "+2.2 (+0.2 to +4.2)" → "+0.2 (-0.8 to +1.4)" — crossing gained (**real**: population fix, §4).
  - **mech-157** (EV→W): "+0.7 (-1.0 to +5.8)" → "-0.1 (-1.7 to +1.0)" — direction flip **in the headline only** (estimand switch; posterior unchanged).
  - **mech-090** (erbto→W): "+3.0 (+0.1 to +5.9)" → "+1.4 (+0.0 to +2.8)" — boundary-touching either way; estimand switch only.
  - **mech-188** (TR→W): "+3.6 (-0.0 to +8.4)" → "+1.9 (+0.2 to +3.7)" — estimand switch only.
- No randomised-effect (τ) family — itt, did, joint, gain_factors causal terms — changed direction or crossing status.

## Executive summary

1. Randomised-effect headlines (ITT/DiD/joint/gain) are numerically stable through the batch; gain-family treatment marginals are byte-identical.
2. The #585 leg-contract fix systematically firmed mediation: six totals (med-059/078, 064, 080, 086/186) now exclude zero, driven by larger letter-sound NIEs; med-062's composite-route NIE weakened to include zero; med-092 correctly headlines the period-1 window (+2.6 [1.0, 4.3]) with the all-period +3.2 flagged as extrapolation.
3. The #584 arm-free AME moved level cards ≤0.05 items except lf-002 (+0.17), lf-010 (+0.09), lf-003 (+0.05); lf-010 (taught expressive vocab) now excludes zero — the one level-family conclusion that strengthens.
4. mech-191's sessions→W "knee" signal disappears once the 28 rows without observed attendance are excluded (2.18 → 0.17 items IQR, interval spans zero) — the strongest single scientific reversal in the batch.
5. mech-157/090/188's headline changes are the #602 interquartile-estimand switch only; their posteriors are unchanged.
6. New gain period-1 sensitivity: stacked borrowing modestly inflates most language/taught-vocab estimates; B, F and both taught-vocab variants lose their zero-exclusion under period-1-only fitting; W and L are robust either way.
7. lf-006 unblocked (robustness_unresolved → ok) via the lf-106 pairing; both links agree (+0.6 vs +0.5 items).
8. Two open actions: re-run the cross-model comparison so the L×N (mech-063/163) nested LOO can validate against the 155-row traces (reloo now possible), and re-evaluate rlm-jc-002's release decision, whose qualification still calls the now-publishable rlm-jc-102 withheld.
9. mech-301 (Mundlak split) is a caveat for the L→W mechanism story: the within-child slope alone is inconclusive (+0.35 [-1.15, 1.82]).
10. All 21 first fits pass their gates; the 6 Byrne/RLM ports remain inputs_unresolved on both sides.
