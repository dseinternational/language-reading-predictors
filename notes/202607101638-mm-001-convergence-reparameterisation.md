# mm-001 convergence fix: marginalising the factor scores out of the measurement likelihood

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

Date: 2026-07-10. This records why the correlated-domain-factor measurement model `lrp-rli-mm-001` (`kind="corr_factor"`, `statistical_models/factories.py::build_correlated_factor_model`) failed its convergence gate, what was changed to fix it, and what the fixed fit says. Read alongside the model's design note `notes/202606291700-correlated-domain-factor-measurement-model.md` and `METHODS.md`.

## The problem

The only reporting-tier fit of `lrp-rli-mm-001` **failed the project's convergence gate**: `diagnostics_summary.json` showed `passed: false` with **422 divergences (1.17 % of 36 000 draws)** and a **failing energy diagnostic (BFMI ≈ 0.21–0.36 on all six chains, below the 0.30 floor)**. R-hat and ESS passed, so this was a geometry problem, not a mixing problem — the signature of a latent-variable **funnel**. Under the project's own reporting rule (`METHODS.md`, `CLAUDE.md`: check convergence _before_ interpreting) the fit's headline output — the vocabulary/code/grammar factor **correlations** — could not be interpreted, yet several documents had begun citing those "0.65–0.80 domain correlations" as evidence (see the reconciliation list below).

## Diagnosis

The original build was a textbook funnel. It sampled a **per-child latent factor score** for each of the three domains (`factor_z`, 51 children × 3 domains = 153 latent parameters) and conditioned **both** the standardised indicators **and** the Beta-Binomial structural outcome on those scores. Two things then interact badly at n ≈ 51:

1. **The measurement funnel.** For a standardised indicator, the model-implied variance `λ_j² + σ_j²` is pinned to ≈ 1 by the data, so the free loading `λ_j` (`HalfNormal(1)`, mode at 0) and the free residual `σ_j` (`HalfNormal(1)`) share a curved ridge. Near the `λ_j → 0` corner the per-child factor scores become unconstrained by that indicator, so the conditional variance of the latent scores changes sharply across the posterior — the classic Neal funnel that wrecks the energy (BFMI) diagnostic.
2. **Small n, two indicators per domain.** Each domain factor is identified by only two indicators, so the loadings are weakly determined and the funnel neck carries real posterior mass.

## The fix

The key observation is that the **measurement model is Gaussian in the factors**, so the factor scores can be integrated out analytically, and they are only genuinely _needed_ for the (non-Gaussian) structural leg. The rewrite therefore does three things, none of which changes the model's meaning:

1. **Marginalise the factor scores out of the measurement likelihood.** The indicators now enter as a single multivariate normal per child, `Z_i ~ MVN(0, Λ Corr Λ' + diag(σ²))`, with no per-child latent variable. This is a standard confirmatory-factor-analysis identification (over-identified here: 15 parameters against 21 unique covariance entries) and removes the 153-dimensional funnel from the measurement part entirely.
2. **Reintroduce the factor scores for the structural leg via their conjugate Gaussian conditional.** Because the measurement model and the factor prior are jointly Gaussian, `p(factors | Z, params)` is `MVN(m_i, V)` in closed form, with `V = (Corr⁻¹ + Λ' diag(σ⁻²) Λ)⁻¹` and `m_i = V Λ' diag(σ⁻²) Z_i`. The scores are then written **non-centred** around that data-informed conditional mean, `factors_i = m_i + chol(V) · z_i` with `z_i ~ N(0, I)` (the RV still named `factor_z`), so the standard-normal offset is decoupled from the loading/residual scales. This is a **measure-preserving reparameterisation**: `p(params) p(factors | params) p(Z | factors) = p(params) p(Z | params) p(factors | Z, params)` by conjugacy, so the posterior over loadings, residuals, factor correlations, factor scores and structural slopes is **unchanged** — only the sampler geometry is. The structural Beta-Binomial leg still sees genuine per-child factor scores with full uncertainty; nothing is plugged in.
3. **Prior recalibration to move mass off the ridge** (kept as free per-indicator RVs, so the loadings/communalities table is unchanged): a positive-mode `TruncatedNormal(loading_mu=0.6, loading_sigma=0.5, lower=0)` loading prior (mass off the `λ → 0` neck instead of a `HalfNormal` peaked at 0) and a tighter `HalfNormal(residual_sigma=0.5)` residual prior (a residual SD cannot exceed the ~unit total variance of a standardised indicator). The `test_pipeline_fallback_defaults` lock was updated for this deliberate recalibration.

Finally, the reporting fit lifts `target_accept` to **0.999** (via `spec.extra`, mirroring the horseshoe fit's funnel handling) to clear the last handful of boundary divergences — the strict gate requires **exactly zero**. This is a legitimate small-steps response for a model whose energy geometry is otherwise healthy; it is not papering over a funnel (the BFMI is now ≈ 0.9).

## Before → after (reporting config, 6 000 draws × 6 chains)

| Diagnostic       | Original build | Fixed build |
| ---------------- | -------------: | ----------: |
| Gate `passed`    |        `false` |      `true` |
| Divergences      |   422 (1.17 %) |           0 |
| BFMI (min chain) |           0.21 |        0.89 |
| min ESS          |            710 |       2 800 |
| max R-hat        |           1.00 |        1.00 |

## What the fixed fit says

The reparameterisation is measure-preserving, so — reassuringly — the substantive result is unchanged. The three latent domains are strongly correlated: **vocabulary↔grammar 0.80 [0.59, 0.94], vocabulary↔code 0.72 [0.43, 0.92], code↔grammar 0.63 [0.28, 0.89]** (all P > 0.999). These match the earlier (non-converged) 0.80/0.74/0.65 to within rounding, so the qualitative reading — a correlated skill system in which a single latent general ability `GA` is a defensible first approximation but the domains are distinguishable — now rests on a fit that **passes** the gate rather than one that failed it. The intervals remain **wide** — the honest result at n ≈ 51, exactly as the model's design note and the closed LRP66 flagged. The structural factor → gain slopes stay **adjusted associations, not causal** (locked DAG ID-2): vocabulary +0.04 [−0.55, +0.64], code +0.26 [−0.31, +0.84], grammar +0.23 [−0.29, +0.76], all inconclusive; age −0.35 [−0.57, −0.12] (P(<0) ≈ 1.00).

## Files changed

- `statistical_models/factories.py` — `build_correlated_factor_model`: marginalised MVN measurement likelihood + conjugate-conditional factor scores; `TruncatedNormal` loadings + tighter residual prior; new `loading_mu` / `residual_sigma` parameters.
- `statistical_models/pipeline.py` — `fit_correlated_factor`: wire `loading_mu` / `residual_sigma`, and honour a `spec.extra["target_accept"]` override.
- `statistical_models/lrp_rli_mm_001.py` — `SPEC.extra["target_accept"] = 0.999`.
- `tests/statistical_models/test_pipeline_fallback_defaults.py` — locked the recalibrated prior defaults.

## Reconciled documents

Two notes had cited the failed fit's correlations as interpretable and were corrected to disclose the failed gate and point here: `notes/202607082140-statistical-models-full-reporting-fit.md` (§ convergence table and the measurement/summary bullets) and `notes/202607091430-dag-critical-review-td-atypical-literature.md` (the single-`GA`-vs-correlated-factors rows). The DAG review-draft note `notes/202607101444-dag-explanation-review-draft.md` had already been corrected to disclose the failure; its disclosure now reads against a fit that has since been fixed.

## Reference

- M. Betancourt (2017), _A Conceptual Introduction to Hamiltonian Monte Carlo_, arXiv:1701.02434 — the energy/BFMI diagnostic and funnel geometry this fix addresses.
