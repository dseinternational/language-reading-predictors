<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5).

# Statistical-model suite review: methodology and implementation (16 families)

Date: 2026-07-13

Related: #119 (floor rule), #141 (prior audit), #247 (revised-DAG factor adjustments), #257 (DiD off-floor own-baseline), #258 (covariate-timing P1 review), #84 (mediation t3 sensitivity), #116 (horseshoe), #265 (HSGP mechanism reparameterisation), #274 (gate over free RVs).

## Scope and method

A read-through of every Bayesian family in `src/language_reading_predictors/statistical_models/` — the shared core (`likelihood.py`/`dse_research_utils`, `priors.py`, `mediation.py`, `hsgp.py`, `floor.py`, `diagnostics.py`, `preprocessing.py`) plus the 16 family factories (`factories.py`) and fit pipelines (`pipeline.py`) — against the methodology `METHODS.md` sets out. The shared core was read directly; the families were reviewed in depth and every load-bearing finding below was then re-verified against the source. This note is the durable record of that review so the decisions a future reader might question are written down with their file anchors.

## Verdict

The suite is unusually rigorous and faithfully implements a sophisticated methodology. No family has a numerical-correctness defect in its headline causal estimand: the randomised effects (`tau`, `beta_trt`, the level-factor t2 group contrast, DiD `delta`) are constructed, signed and marginalised correctly; the empty-adjustment-set identification is genuinely enforced (every ITT spec sets `cross_symbols=()`); the g-formula, the regularised horseshoe and the CFA marginalisation are textbook-correct; and the convergence gate is stricter and better targeted than most production Bayesian code (unrounded R-hat/ESS over all free RVs, non-finite BFMI treated as failure, output-dir reset against stale artefacts). The findings cluster into three buckets: **(A)** modelling-assumption choices that could move a reported conclusion (all documented, some debatable); **(B)** reporting/audit-trail defects where the fitted numbers are right but a persisted artefact or flag is wrong or missing; **(C)** latent, robustness or cosmetic items. Nothing found means a currently-reported number is wrong.

## Group A — assumptions that could move a conclusion

### A1. State confounders read at the post (contemporaneous) wave risk post-treatment adjustment

`preprocessing.py:203-209` (`INTERVAL_COVARIATES = {"attend"}`); consumed by `fit_mechanism`, `fit_gain_factors`, `fit_level_factors`. Every adjuster except `attend` is a "state" covariate read from the transition's **post** row. For the identifying period-1 (t1→t2) transition, t2 is post-treatment for the immediate arm. If the intervention moves speech production (`deapp_c`) or phonological memory (`erbto`) within a single period — plausible for a language/reading programme — conditioning on their t2 value is conditioning on a treatment-affected variable, biasing the sole causal coefficient (`beta_trt`, `b_grp_time[1]`) toward the null. The documented rationale (`factories.py:2842-2844`, #258) is that the revised DAG asserts no `IG → HS/SP/RW` edge, which makes the contemporaneous read correct — very defensible for hearing, load-bearing and debatable for two language-proximal measures. There is no per-covariate control to read a plausibly-endogenous confounder at baseline instead, and the mediation family made the **opposite** choice (baseline t1) citing this exact affected-confounder risk (`lrp_rli_med_059.py:33-36`) — a real cross-family inconsistency. **Severity: medium. Confidence: timing confirmed; bias contingent on the SP/RW exogeneity claim, which is a substantive team judgement worth re-confirming for a language intervention.** **Resolved 2026-07-13 (team decision):** move SP/RW to the pre-randomisation baseline (t1) for the causal-term factor families, keeping hearing contemporaneous. Implemented via `preprocessing.split_confounders_by_timing` (a named policy over `BASELINE_CONFOUNDER_BASES = {deapp_c, erbto}`), routed into `baseline_covariates` by `fit_gain_factors` / `fit_level_factors`; the mechanism family (an avowed adjusted association) keeps the contemporaneous read. SP/RW are ~94–98% complete at t1, so the baseline read is well-populated and the missing-indicator method covers the remainder.

### A2. The graded DiD conditions on a treatment-affected baseline

`factories.py:1425-1427`. The graded Beta-Binomial DiD (DID-001–010) adds `gamma_own * own_pre`; for the immediate arm's P2 rows the period-start score is the post-P1-treatment t2 score. That mixes a lagged-dependent-variable/ANCOVA adjustment into what the docstring bills as "a difference-in-differences estimate of the ITT effect" (`factories.py:1294-1296`). The off-floor sibling (DID-011/012) deliberately drops `gamma_own` for exactly this reason (Rosenbaum 1984, DOI [10.2307/2981697](https://doi.org/10.2307/2981697); #257), so the graded family is internally inconsistent with its own floored branch. Partially mitigated — `delta` is over-identified and the pre-randomisation P1 between-arm contrast is uncontaminated — but the "pure ITT replication" reading of the graded `delta` is imprecise. **Severity: medium. Confidence: mechanism confirmed.** **Resolved 2026-07-13:** `gamma_own` dropped from the graded DiD too, unifying it with the off-floor branch — neither branch conditions on the treatment-affected period-start score, `pre_required=()` for both, and they now differ only in likelihood (Beta-Binomial vs Bernoulli). `delta` is identified by the period × treated DiD structure plus the child random intercept (each child their own control).

### A3. The LCSM headline configuration is the least-identified rung

`factories.py:3421-3443, 3390`. The default LRP67 (`use_process_noise=True, shared_process_noise=False`) gives each child-measure four latent degrees of freedom against four observed waves — an effectively saturated latent state — so the dynamic `sigma_proc` and the measurement `kappa` are only weakly separable and the headline cross-couplings `g_L`/`g_E` are prone to being prior-dominated at n≈54. Separately, `b_self ~ Normal(0, 0.5)` centres the level AR(1) on a unit root with roughly half the prior mass on explosive dynamics. This is inherent to the model at this n and is honestly hedged in the docstrings; the practical safeguard is that the convergence gate catches the resulting funnels. Treat a clean gate as the binding precondition before reading any LCSM coupling, and expect wide intervals. **Severity: medium. Confidence: construction confirmed; impact plausible.**

**Resolved 2026-07-13 (diagnose-first, team decision):** applied the low-risk half — `b_self` shifted to the mean-reverting `Normal(-0.3, 0.2)` (φ = 1 + `b_self` now centred ~0.7, ~7% explosive mass vs ~50%) — then ran a rep-lite LRP67 diagnostic to decide de-saturation on evidence. Result: the default config (`use_process_noise=True, shared_process_noise=False`) **converges cleanly** (0 divergences, max R-hat 1.007, min ESS 967, BFMI 0.58–0.65) and the headline couplings are **data-informed, not prior-dominated** — `g_L` = +0.14 [0.03, 0.26] (posterior SD 19% of the 0.30 prior), `g_E` = +0.29 [0.07, 0.53] (SD 39% of prior), both excluding zero; `b_self[W]` = −0.22 [−0.31, −0.14] (φ ≈ 0.78, data-supported mean-reversion). The saturation concern therefore does **not** translate into a convergence failure or prior-domination in practice, so **de-saturation (shared/no process noise) was not adopted** — the `b_self` prior fix alone suffices. The `use_process_noise` / `shared_process_noise` knobs remain available as a documented fallback if a future data revision re-triggers funnels.

### A4. Two floored-outcome asymmetries in the factor families

The gain-factor off-floor path keeps `gamma_own` with the graded mean-1 prior (`factories.py:2948`), unlike the ITT floored specs (`use_own_baseline=False`) and the DiD off-floor branch (drops it). Because P/N are heavily floored at baseline (N is 72% zero at t1), that informative prior shifts the operating point of the probability-scale off-floor risk difference (the reported headline), though not the logit `beta_trt`. And the level-factor off-floor outcomes (P, N) get no probability/items-scale effect card at all (`pipeline.py:2722` has no `off_floor` branch), so they are missing their interpretable headline and δ-sweep. **Severity: low–medium. Confidence: confirmed.** **Resolved 2026-07-13:** (a) `gamma_own` dropped from the gain-factor off-floor path, matching the ITT floored specs (`use_own_baseline=False`) and the DiD off-floor branch; (b) the level-factor off-floor outcomes now emit the off-floor risk-difference ROPE card + 10/15/20 pp δ-sweep — `expit(eta) = Pr(off-floor)`, so `level_t2_marginal_effect`'s probability-scale AME at `n_trials=1` is the risk difference, mirroring the gain-factor off-floor path.

## Group B — reporting and audit-trail fidelity (fitted numbers unaffected)

### B1. Sub-fit convergence verdicts are computed then discarded

`subfit_convergence` is wired correctly for the floor-rule secondary (`pipeline.py:1296-1298`: captures the dict and sets the summary's `converged` field) but is called as a bare statement with its return dropped at `pipeline.py:2104` (mediation t3 temporal-ordering sensitivity) and `pipeline.py:3051` (`_sample_model`, used by the adjusted family's bivariate + prior-sweep + SES sub-fits). So `mediation_summary_t3.csv`, `predictor_associations.csv`, `prior_sensitivity.csv` and `ses_sensitivity.csv` are published with no `converged` flag — the red console warning fires during the run, but the rendered report shows the numbers unflagged. This re-opens, at the artefact level, the exact "sub-fit reported without a flag" hazard the helper was written to close. **Severity: medium. Confidence: confirmed. Fix being applied in this change (mirror the 1296-1298 pattern at each site).**

### B2. `_effective_adjustment` omits the block-design ability covariate

`pipeline.py:139-147`. The function persists the "adjustment set the model actually fitted" to `config.json` and exists specifically to end a #258 misdescription — yet it has no ability parameter, while every gain/level-factor model fits `gamma_ability` on `blocks` (`factories.py:2809`). So the authoritative audit record understates the conditioning set by one term across the whole factor family. Mitigated only because the coefficient still appears in `factor_summary.csv`. **Severity: medium. Confidence: confirmed (signature + empirical reproduction). Fix being applied in this change.**

### B3. `b_E`/`b_B` prior-table rows are self-contradictory in single-mediator models

`priors.py:436-437`. The global map `_RV_TO_CTOR["b_E"]="b_path"` is correct where E/B are mediators (LRP64/66), but in LRP59/62/78 E is a confounder built with `gamma_cross` (Normal(0,0.3)). The `priors_table.csv` `b_E` row then shows distribution `Normal(0,0.3)` (correct, read off the RV) but rationale "b-path ~ Normal(0,1)" and panel `prior_b_path.png` — a contradictory row in three shipped models. Fix by adding `ctor_overrides` in the mediation `_prior_table_overrides` so `b_{confounder}` routes to `gamma_cross`. **Severity: low (reporting-only). Confidence: confirmed.** **Resolved 2026-07-13:** `_prior_table_overrides` now routes every single-mediator (`kind == "mediation"`) confounder `b_X` — any `b_*` other than the structural `{M, G, GM, W, A}` — to the `gamma_cross` constructor, so the `b_E` row's rationale + panel match its Normal(0, 0.3) distribution; `mediation_multi` is untouched (there `b_E`/`b_B` genuinely are the mediator b-paths).

### B4. CFA structural-slope prior and panel mismatch

`factories.py:2444`. `beta_factor`/`beta_age` use `Normal(0, 0.5)`, unreconciled with the 0.3 the 2026-07-07 prior review applied to `predictor_slope`, and the report's `prior_predictor_slope.png` panel plus rationale render at 0.3 while the RV is 0.5. Either reconcile or document why the latent-factor scale warrants 0.5. **Severity: low. Confidence: confirmed.** **Resolved 2026-07-13:** `build_correlated_factor_model`'s `predictor_slope_sigma` default reconciled 0.5 → 0.3, matching the shared `predictor_slope_prior` default (applies to `beta_factor`, `beta_age` and the structural-covariate slopes); the built RV now agrees with the `prior_predictor_slope` panel drawn at 0.3, clearing the panel/RV mismatch too.

## Group C — latent, robustness and cosmetic

- Floor-rule PRIMARY prior-predictive plot overlays Bernoulli {0,1} draws on graded counts (`diagnostics.py:701-729`): pass the `y_offfloor` node. Diagnostic plot only, guarded.
- Joint `alpha` untiered for distal outcomes (`factories.py:590`): known documented follow-up. Joint `tau_k` being untiered is intentional and documented in `METHODS.md`, not a defect.
- Joint `pm.Data("A_std")` node is dead — the age terms read the raw array (`factories.py:574` vs `:624`): a latent `set_data` hazard; `G` is wired correctly.
- Joint / random-effect LOO is leave-one-cell-out, correctly labelled conditional; comparability caveat holds and `historical_growth` is missing from the caveat set in `_footer.qmd`.
- Horseshoe `tau0=0.1` drops the logit pseudo-σ from the Piironen–Vehtari global-scale heuristic (`priors.py:305`): mildly aggressive shrinkage; tolerable for an ordinal cross-check.
- Horseshoe/CFA gate ESS-checks the heavy-tailed local scales (`diagnostics.py:360`): errs safe (false failure, not false pass).
- Missing-predictor mean-imputation attenuates patchier predictors in the horseshoe ranking (`factories.py:2303`): add a caveat.
- `beta_dose` maps to the `tau`→"causal" constructor and is only relabelled "association" by an override (`priors.py:414`); `fit_dose_response` passes the collider dose slope as `causal_term` (`pipeline.py:1963`) — focuses plots only, no artefact mislabelled, but a smell in a family whose thesis is "dose is not causal".
- `dropped_rows` undercounts structural (missing-wave) attrition — the inner merge precedes the count (`preprocessing.py:461`); scales with wave dropout.
- Covariate/age standardisation is over the loader rows, not the factory-subset rows (`factories.py:352`): a centring imprecision on the intercept, not leakage, and it cannot bias a randomised effect.
- ITT raises `KeyError` on a constant adjuster (`factories.py:316`) while the loader drops-and-continues — two modules disagree; "fail loud" is defensible.
- `drop_missing_pre=False` bypasses all masking (`preprocessing.py:463`): no spec sets it; add a guard.
- Missing-indicator fill uses the global column mean, not the fitted-wave mean (`preprocessing.py:247`): the `_missing` indicator absorbs the offset so estimates stay unbiased under randomisation — a docstring inaccuracy, not a bias.
- Naming nits: `_beta_summary(hdi=...)` computes equal-tailed quantiles; `sigma_dose` is summarised with a meaningless `p_pos≡1.0`; a dead `save_shared_prior_panel` call at `pipeline.py:4008` is superseded by `_emit_priors`.

## Per-family verdicts

| Family                            | Verdict                                                                                                                                                                                     |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ITT / floored ITT                 | Clean; empty adjustment set enforced; floor rule correctly targets the baseline-floored (pre-randomisation) subgroup, off-floor-post hurdle honestly flagged as selection.                  |
| Joint                             | Correct off-diagonal masking, per-cell denominators, non-zero-imputed missing cells, LKJ double-scaling fixed; distal α-tiering pending (C).                                                |
| Gain / level factors              | Period-1 causal isolation and t2 isolation correct and leak-free; concerns are A1, A4, B2.                                                                                                  |
| Mechanism / dose-response         | Reparameterisation correctly scoped to `f_mech`; A1 applies; a near-flat curve under `target_accept=0.999` + `HalfNormal(0.3)` may be prior/step-size-conservative rather than data-driven. |
| Mediation (single & multi)        | Factory↔g-formula name/scale contract disciplined and clean; LOO correctly skipped, gate intact; only B3 and B1 (t3 flag).                                                                  |
| DiD / aligned                     | Aligned is exactly to spec (no causal flags, correct onset windows); graded DiD has A2.                                                                                                     |
| Adjusted / horseshoe / CFA        | Horseshoe is a correct Piironen–Vehtari implementation; CFA identification and its documented forward-simulation caveat are sound; B1 (sub-fit flags), B4 (prior panel).                    |
| LCSM / growth / historical-growth | Historical study cleanly isolated (no RLI↔RLM leakage); growth causal-hygiene clean; LCSM has A3.                                                                                           |

## Strengths worth preserving

The separation of causal / precision / association / nuisance / GP roles is enforced in code (only `tau_prior` carries the "causal" role), not just prose; priors are scale-calibrated to item counts and anchored on external reliabilities, never on the fitted data. The g-formula computes NDE/NIE by simulation with common random numbers and states its non-identification honestly (exposure-induced dose confounder, latent general ability, cross-world assumptions, interventional relabelling). The convergence gate evaluates unrounded R-hat/ESS over all free RVs and treats non-finite BFMI as failure. The floor rule's pre- vs post-randomisation conditioning distinction is exactly right. This is code that has clearly already been through serious methodological critique.

## Recommended actions

1. B1 and B2 — small, mechanical, restore audit/convergence integrity (done this change).
2. A1 — SP/RW confounder timing moved to baseline for the factor families (done this change, team decision 2026-07-13).
3. A2 / A4 — estimand clarity in the DiD and floored-factor paths (done this change).
4. B3 / B4 — prior-table label + CFA-slope reconciliation (done this change). Group-C cleanups as time permits.
