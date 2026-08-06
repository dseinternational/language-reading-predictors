# Two default priors recalibrated: `sigma_subject` and the CFA loading geometry (#383)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

**Date:** 2026-08-06. **Issue:** #383, from the critical prior-analysis review (`notes/202607211500-prior-critical-review.md`, PR #380). **Scope:** the two default priors that review found in genuine tension with the data. The core effect priors (`tau`, `gamma_cross`, `alpha`, `kappa`, GP `eta`) were confirmed correctly scaled and are untouched.

## Decision 1 — `sigma_subject`: HalfNormal(0.5) → HalfNormal(1.0) in `rlm-hg` / `rlm-jc`

The between-child random-intercept SD prior in the Byrne historical-growth family (`lrp-rlm-hg-001…009`) and the joint correlated-trajectories model (`lrp-rlm-jc-001`) was `HalfNormal(0.5)`, set by the 2026-07-07 prior review's reconciliation (recommendation 3). The 2026-07-21 review found it in genuine prior-data conflict for the high-variance Down-syndrome measures, and the fitted reporting posteriors confirm it precisely. Under `HalfNormal(0.5)` the 99th percentile is 1.288; the fitted Down-syndrome `sigma_subject` posteriors were:

| fit      | measure   | posterior mean | 89% ETI      | vs prior            |
| -------- | --------- | -------------- | ------------ | ------------------- |
| `hg-001` | `basread` | 1.267          | [1.03, 1.55] | at the 99th pct     |
| `hg-002` | `basspel` | 1.255          | [0.98, 1.58] | at the 99th pct     |
| `hg-003` | `woco`    | 1.388          | [1.10, 1.72] | beyond the 99th pct |
| `hg-004` | `bpvs`    | 0.354          | [0.25, 0.49] | comfortably inside  |
| `hg-005` | `trog`    | 0.603          | [0.45, 0.79] | inside              |
| `hg-006` | `basdig`  | 0.500          | [0.36, 0.67] | inside              |
| `hg-007` | `bassim`  | 1.328          | [1.05, 1.65] | beyond the 99th pct |
| `hg-008` | `basnum`  | 0.769          | [0.58, 1.00] | inside              |
| `hg-009` | `basmat`  | 0.368          | [0.11, 0.62] | inside              |
| `jc-001` | `basread` | 1.287          | [1.06, 1.56] | at the 99th pct     |
| `jc-001` | `bpvs`    | 0.290          | [0.18, 0.41] | inside              |
| `jc-001` | `basdig`  | 0.470          | [0.35, 0.62] | inside              |

A posterior mean sitting at its prior's 99th percentile is not a tail coincidence repeated five times — it is the prior truncating the likelihood, which mildly biases the reported between-child spread downward exactly where heterogeneity is the substantive story (word reading, spelling, orthographic choice, and the `bassim` ability proxy in the Down-syndrome group). One detail the review's "verbal/reading" framing did not anticipate: `bassim` (similarities) conflicts too, so the conflict is not confined to literacy measures.

**Decision: widen family-wide to `HalfNormal(1.0)` (99th pct 2.58), not per-measure.** A selective per-measure widening would tune the prior to each posterior — the wrong direction of travel. `HalfNormal(1.0)` is weakly informative for the low-heterogeneity measures (their posteriors, 0.29–0.77, are data-dominated and should barely move — verified below) while releasing the conflicted ones. This deliberately reverses the 2026-07-07 reconciliation on the later review's evidence; both decisions are now recorded, and the module comments in `lrp_rlm_hg_001.py` carry the lineage. The factory defaults (`build_historical_growth_model`, `build_rlm_joint_growth_model`) move with the specs — their only registered consumers are these ten models — and the fallback-defaults test locks the reviewed value.

## Decision 2 — CFA loading geometry: the communality scale becomes the RLI default

The review's second finding: `HalfNormal(1)` on **both** the loadings and residuals of the cross-sectional CFA (`lrp-rli-mm-001/002`) "ignores the λ² + σ² ≈ 1 budget (~32% prior mass on loadings > 1, Heywood-adjacent)". The indicators are standardised, so each free indicator's model-implied marginal variance is λ² + σ², and the data pipeline pins the observed variance at 1.

There is a sharper way to state the defect. Squaring two independent `HalfNormal(1)` scales gives two iid χ²₁ variables, so the legacy pair implies

**communality = λ²/(λ² + σ²) ~ Beta(½, ½)** — the arcsine distribution:

U-shaped, with mass piled on _both_ singular corners — the λ→0 neck (which drove the original funnel) and the Heywood-adjacent c→1 boundary (which inflates disattenuated factor correlations). P(c > 0.8) = 0.295 and P(c < 0.2) = 0.295 under it. The one defensible property is its median of 0.5, which the LRPMM101 ablation explicitly defended when it rejected the #261 "recalibrated" pair (median communality 0.79).

**Decision: port the communality-scale parameterisation of `build_rlm_corr_factor_model` (#409 item B, already the Byrne default) into `build_correlated_factor_model` as the RLI default.** The free parameter is `communality ~ Beta(2, 2)` per indicator, with `lambda = sqrt(c)` and `sigma = sqrt(1 − c)` derived:

- λ² + σ² = 1 **exactly** — Heywood configurations have zero prior mass, and the loading-residual ridge (one parameter per indicator fewer) is removed;
- the prior median communality stays **0.5** — the geometry changes, the central commitment does not;
- Beta(2, 2) has zero density at both corners, regularising precisely the two pathologies the arcsine prior favours;
- the node names (`lambda_load`, `sigma_indicator`, `communality`) are unchanged — only which is the free RV differs — so every downstream consumer (summaries, gate, psense, report partials) is untouched.

The rlm-mm builder documents this same design as "the mm-001 gate rescue"; the #381 indicator-scale prior check had already quantified the calibration difference on real artefacts before this change: `rlm-mm-001` (communality scale) prior-predictive/observed SD ratio ≈ **1.01**, `mm-001` (legacy pair) ≈ **1.36–1.48**. After the refit the RLI fits should sit at ≈ 1.0 by construction; verified below.

### What happens to LRPMM101

Its registered question — does the #261 positive-mode recalibration (`TruncatedNormal(0.6, 0.5)` / `HalfNormal(0.5)`) change the posterior? — was answered (no, while costing a 0.50 → 0.79 median-communality commitment; `notes/202607101638-mm-001-convergence-reparameterisation.md`) and is settled. Leaving it registered as-was would have preserved a sensitivity contrast against a rejected prior nobody uses, while confounding parameterisation and prior values — exactly the confounding #261 was criticised for. LRPMM101 is therefore re-registered as the **geometry-sensitivity companion**: `loading_prior="free"` at the legacy knob defaults (the old `HalfNormal(1)` pair), identical to LRPMM01 in everything else. The live contrast isolates the single changed thing (bounded mid-mass Beta(2,2) vs boundary-loving unbounded pair, same 0.5 median), and the pair demonstrates — or falsifies — the claim that the reported quantities are data-dominated under the geometry change.

### Not extended to `lcf`

`lrp-rli-lcf-001` (the longitudinal CFA) keeps its free `HalfNormal(1)` pair. The same budget argument applies — the #381 indicator check found it the loosest family (SD ratios 1.47–2.26) — but #383 names only `mm`/`rlm-mm`, the lcf model passed its gate, and its wave-invariant loading structure with per-block missingness patterns makes the port a larger change than a default swap. Deferred; if the lcf model graduates from its #338 "fragile, defer" status the same parameterisation should be applied then.

## Implementation

- `build_correlated_factor_model`: new `loading_prior` (`"communality"` default / `"free"` legacy), `comm_alpha`, `comm_beta` knobs; validation mirrors the rlm builder.
- `fit_correlated_factor`: settings-coherence guard **before** `make_context` resets the output directory (#455 principle) — free-pair knobs under the communality parameterisation (or Beta shapes under the free pair) raise instead of being silently ignored.
- `sigma_subject_prior_sigma: 0.5 → 1.0` in the ten `rlm-hg`/`rlm-jc` specs and both factory defaults; the pipeline's literal jc fallback updated to match.
- Tests: factory defaults locked (`test_pipeline_fallback_defaults`), both parameterisations' RV structure and the exact λ² + σ² = 1 budget (`test_factories`), the coherence guard, and the legacy pair's non-enforcement of the budget.
- Report templates (`mm-001`, `mm-002`, `mm-101`) updated to describe the new geometry; `mm-101`'s Overview rewritten for its new role with the ablation history preserved.

### A #484 wiring defect found on the way

The dev smoke fit of the new parameterisation exposed a fresh-fit ordering defect in the #381/#484 indicator-scale prior check: `_write_indicator_prior_check` ran **before** `save_trace`, which is the step that attaches the `prior`/`prior_predictive` groups to the in-memory trace on a fresh fit — so every fresh CFA fit skipped the check silently ("no indicator nodes found"). The #484 re-emit verification never saw this because `--reuse-trace` loads a DataTree whose groups are already on disk. The call now runs after `save_trace` at all three sites (mm, rlm-mm, lcf); the reporting refits below all carry the CSV.

## Refit verification (reporting tier, 2026-08-06)

**All 13 refits pass the full convergence gate with 0 divergences** (`hg-001…009`, `jc-001`, `mm-001/002/101`).

**`sigma_subject` released where truncated, unmoved where not.** Down-syndrome posteriors, before → after: `basread` 1.267 → 1.408, `basspel` 1.255 → 1.447, `woco` 1.388 → 1.623, `bassim` 1.328 → 1.563, jc `basread` 1.287 → 1.440 — the most-truncated measure (`woco`) moved most, the dose-response pattern truncation relief predicts, with upper interval limits now reaching 1.77–2.10 where the old prior pinned them at ≈1.5–1.7. The data-dominated measures barely moved: `bpvs` 0.354 → 0.363, `basdig` 0.500 → 0.515, `basmat` 0.368 → 0.399, jc `bpvs` 0.290 → 0.311. The families' reported deliverable — the items-scale wave-to-wave growth quantities — is untouched: max |Δ| ≤ 0.10 items across every `hg` model and all quantiles, concentrated in interval endpoints.

**The mm posteriors are data-dominated under the geometry change, with one small systematic shift.** Communalities move at most ±0.036 (L 0.404 → 0.427, B 0.390 → 0.419 in `mm-001`; the weak code indicators gain slightly as Beta(2, 2) trims the arcsine prior's λ→0 corner); structural slopes move ≤ 0.008 (e.g. `mm-002` `beta_code` 0.340 → 0.345). The factor correlations shift **up by ≈ +0.02 uniformly** (`mm-001`: vocab~~code 0.763 → 0.785, vocab~~grammar 0.802 → 0.826, code~grammar 0.673 → 0.695). The cleanest isolation of the geometry is the new `mm-001` vs the new `mm-101` (same data, same sampler, geometry the only difference): +0.022 on each correlation. Small against 89% interval half-widths of ≈ 0.15–0.20, but systematic and in the direction higher code communalities imply; the family's standing "fragile and prior-dependent at n ≈ 51" caveat is doing exactly its job on these ~0.02s.

**The indicator-scale check confirms the calibration by construction.** `mm-001` and `mm-002` prior-predictive/observed SD ratios 1.00–1.01 (all "well scaled"); `mm-101` (legacy pair) 1.36–1.48. One labelling caveat: 1.36–1.48 sits below the coarse 1.5 "loose" threshold, so `mm-101`'s verdict column also reads "well scaled" — quote the ratios, not the labels, when comparing the geometries.
