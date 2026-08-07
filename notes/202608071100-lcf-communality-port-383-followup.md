<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# The lcf loading geometry moves to a pooled-budget communality parameterisation (#383 follow-up)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5).

**Date:** 2026-08-07. **Context:** the one engineering follow-up #383 left on record. Its decision note (`notes/202608061500-default-prior-recalibration-383.md`, PR #485 "Not covered") ported the communality-scale loading geometry to the cross-sectional CFA builders but deferred `lrp-rli-lcf-001`, noting the same budget argument applies — the #381 indicator-scale check found the lcf family the loosest, prior-predictive/observed SD ratios 1.47–2.26 — while "its wave-invariant loading structure with per-block missingness patterns makes the port a larger change than a default swap". This note records the port, and why the deferral's "larger change" warning was right on the mathematics, not just the plumbing. A second, smaller follow-up — retiring the stale `rlm-mm-001` psense exemption — is recorded at the end.

## Why a verbatim port would have been wrong

The cross-sectional `mm` builders standardise each indicator at a single occasion, so the unit sample variance is all within-occasion and the exact budget `λ² + σ² = 1` is the correct constraint. The lcf model **pooled-standardises across the four waves** — deliberately, so wave-to-wave level change survives standardisation and is carried by the per-wave factor means. That puts part of the unit variance _between_ waves. Measured on the fitted panel, the between-wave share runs **5% (B) to 18% (TE/L)** of each indicator's pooled unit variance, and the fitted free-pair posterior confirms the model uses that freedom: `λ² + σ²` sits at **0.75–0.88**, not 1. Forcing `λ² + σ² = 1` verbatim would overstate the within-wave variance by exactly that share, with nowhere for the likelihood to absorb it — a systematic downward bias on communalities dressed up as a constraint.

## The pooled-budget parameterisation

The budget the lcf data pipeline actually imposes is on the **pooled** variance, and the model can enforce exactly that. Per indicator `j` with domain `d`, let `V_j` be the observed-cell-weighted variance of the model's own wave means `λ_j μ_{d,t}` — computable in-graph from `factor_mean` and the fixed per-indicator wave weights (missingness differs by indicator, so each indicator weighs the domain means with its own cell counts). With the within-wave communality `c_j ~ Beta(2, 2)` free:

```
λ_j = sqrt(c_j / (1 + c_j V_j))        σ_j = sqrt((1 − c_j) / (1 + c_j V_j))
```

Then in every prior draw and at every posterior point:

- **pooled implied variance** `λ²(1 + V) + σ² = 1` exactly — the constraint the standardisation imposes;
- **communality** `λ² / (λ² + σ²) = c` exactly — the free parameter is the quantity the loadings table reports;
- **within-wave budget** `λ² + σ² = 1 / (1 + cV)` — exposed as a new `within_share` Deterministic in both modes;
- Heywood configurations have **zero prior mass**, and the `Beta(2, 2)` keeps the defended prior median communality of 0.5;
- the cross-sectional exact budget is recovered as the `V = 0` special case, so this is a generalisation of #383's geometry, not a different idea.

The legacy free pair (`TruncatedNormal(0, 1, lower=0)` loadings, `HalfNormal(1)` residuals — jointly an arcsine `Beta(½, ½)` communality prior, #383's sharp statement of the defect) is retained behind `loading_prior="free"` so a geometry-only sensitivity contrast remains constructible, exactly as `LRPMM101` does for the cross-sectional family. No companion is registered here: the lcf model keeps its #338 "fragile, defer" status, and its docstring already lists a prior-sensitivity companion among the follow-ups if it graduates.

## Implementation

- `build_longitudinal_corr_factor_model`: `loading_prior` (`"communality"` default / `"free"` legacy), `comm_alpha`, `comm_beta`; validation mirrors the cross-sectional builder; `factor_mean` is declared before the measurement parameters (the derived scales now depend on it); node names `lambda_load` / `sigma_indicator` / `communality` unchanged in both modes.
- `fit_longitudinal_corr_factor`: the #455-style settings-coherence guard **before** `make_context` (free-pair knobs under the communality parameterisation raise, and conversely); `within_share` joins the gated summary set; the prior-predictive variable list names the derived nodes explicitly (as Deterministics they leave the free-RV listing) and dedupes `communality`; `communality` joins the power-scaled psense set.
- The shared `communality` priors-table rationale now states both budgets (exact `λ² + σ² = 1` cross-sectionally; `1/(1 + cV)` pooled-budget here) — the dev smoke exposed it describing the mm budget on the lcf report.
- Tests: factory-default rows locked (`test_pipeline_fallback_defaults`), the pooled-budget identity `λ²(V+1) + σ² = 1` verified to 1e-8 against independently recomputed weights, the legacy pair's non-enforcement, validation and coherence-guard rejections.

## Refit verification (reporting tier, 2026-08-07)

**The refit passes the full gate with 0 divergences** (max R-hat 1.0006, min ESS 6,112 vs the legacy fit's 5,157, min BFMI 0.78), and `key_findings.json` releases (`ok`).

**The calibration defect is repaired by construction.** The indicator-scale prior check moves from the legacy **1.47–2.26 with 30 of 32 cells "loose"** — the loosest family in the #381 sweep — to **0.91–1.38 with 29 of 32 "well scaled"**, centred near 1. The residual per-cell spread around 1 is real single-cell sampling variation (each cell's observed SD is an n ≈ 54 draw), not mis-scaling: the model-side pooled budget is exact. The fitted `within_share` posteriors (0.84–0.92) land exactly on the data-side within-wave variance shares (0.83–0.97 measured directly from the panel), which is the budget doing its job rather than a coincidence.

**The posteriors move the way the mm geometry change predicted, slightly more strongly.** Communalities rise by +0.014 (L) to +0.041 (TR) — the weak indicators gain most as the arcsine prior's λ→0 corner mass goes — and every latent correlation shifts **up systematically**: `vocabulary~code` +0.038, `vocabulary~grammar` +0.021, `code~grammar` +0.048 (medians, essentially identical at all four waves; e.g. wave-1 `code~grammar` 0.559 → 0.608, 89% [0.46, 0.72]). This is the longitudinal echo of `mm-001`'s +0.022 uniform correlation shift, larger here because the legacy lcf prior was the further from calibrated. All shifts are small against 89% interval half-widths of ≈ 0.10–0.13, but systematic and in the direction higher communalities imply — the family's standing "fragile and prior-dependent at n ≈ 54" caveat is carrying exactly this. `trait_share` is essentially unmoved (0.93–0.95 → 0.94–0.96), so the "read it as one matrix, not four" guidance stands, with the report's fit-anchored numbers updated (`vocabulary~grammar` ≈ 0.88).

**Power-scaling** flags the same members as before, informationally: the four per-wave `vocabulary~grammar` correlations and the code/grammar trait shares at "potential prior-data conflict" (likelihood sensitivities 0.20–0.29), consistent with the template's standing statement that the vocabulary–grammar correlation is the most prior-influenced quantity and is to be read as a conservative lower bound.

## The `rlm-mm-001` psense exemption is retired (the second follow-up)

The #383 issue thread recorded that `rlm-mm-001` "is no longer a legitimate psense exemption: #381 lists it as the one family exempt 'on account of its non-converged posterior', and that ground is gone". The code side was already closed (#480 wired psense into `fit_rlm_corr_factor`; the current reporting fit passes the full gate and carries a measured `psense_summary.csv` — 22 parameters, 4 informational `factor_corr_pairs` conflict flags). What remained stale was the record: `notes/202607261700-psense-coverage-backfill.md` still named `rlm-mm-001` "the one true exemption". That note now carries a dated supersession warning and a closing section, so both exemptions it recorded are closed with measurements, not waivers.
