# mm-001 convergence fix: marginalising the factor scores out of the measurement likelihood

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

Date: 2026-07-10, **revised 2026-07-12** after the review of #261. This records why the correlated-domain-factor measurement model `lrp-rli-mm-001` (`kind="corr_factor"`, `statistical_models/factories.py::build_correlated_factor_model`) failed its convergence gate, what was changed to fix it, what the fixed fit says, and — following the review — which of those changes actually did the work. Read alongside the model's design note `notes/202606291700-correlated-domain-factor-measurement-model.md` and `METHODS.md`.

## The problem

The only reporting-tier fit of `lrp-rli-mm-001` **failed the project's convergence gate**: `diagnostics_summary.json` showed `passed: false` with **422 divergences (1.17 % of 36 000 draws)** and a **failing energy diagnostic (BFMI ≈ 0.21–0.36 on all six chains, below the 0.30 floor)**. R-hat and ESS passed, so this was a geometry problem, not a mixing problem — the signature of a latent-variable **funnel**. Under the project's own reporting rule (`METHODS.md`, `CLAUDE.md`: check convergence _before_ interpreting) the fit's headline output — the vocabulary/code/grammar factor **correlations** — could not be interpreted, yet several documents had begun citing those "0.65–0.80 domain correlations" as evidence (see the reconciliation list below).

## Diagnosis

The original build was a textbook funnel. It sampled a **per-child latent factor score** for each of the three domains (`factor_z`, 51 children × 3 domains = 153 latent parameters) and conditioned **both** the standardised indicators **and** the Beta-Binomial structural outcome on those scores. Two things then interact badly at n ≈ 51:

1. **The measurement funnel.** For a standardised indicator, the model-implied variance `λ_j² + σ_j²` is pinned to ≈ 1 by the data, so the free loading `λ_j` (`HalfNormal(1)`, mode at 0) and the free residual `σ_j` (`HalfNormal(1)`) share a curved ridge. Near the `λ_j → 0` corner the per-child factor scores become unconstrained by that indicator, so the conditional variance of the latent scores changes sharply across the posterior — the classic Neal funnel that wrecks the energy (BFMI) diagnostic.
2. **Small n, two indicators per domain.** Each domain factor is identified by only two indicators, so the loadings are weakly determined and the funnel neck carries real posterior mass.

## The fix

The key observation is that the **measurement model is Gaussian in the factors**, so the factor scores can be integrated out analytically, and they are only genuinely _needed_ for the (non-Gaussian) structural leg.

1. **Marginalise the factor scores out of the measurement likelihood.** The indicators now enter as a single multivariate normal per child, `Z_i ~ MVN(0, Λ Corr Λ' + diag(σ²))`, with no per-child latent variable. This is a standard confirmatory-factor-analysis identification (over-identified here: 15 parameters against 21 unique covariance entries) and removes the 153-dimensional funnel from the measurement part entirely.
2. **Reintroduce the factor scores for the structural leg via their conjugate Gaussian conditional.** Because the measurement model and the factor prior are jointly Gaussian, `p(factors | Z, params)` is `MVN(m_i, V)` in closed form, with `V = (Corr⁻¹ + Λ' diag(σ⁻²) Λ)⁻¹` and `m_i = V Λ' diag(σ⁻²) Z_i`. The scores are then written **non-centred** around that data-informed conditional mean, `factors_i = m_i + chol(V) · z_i` with `z_i ~ N(0, I)` (the RV still named `factor_z`), so the standard-normal offset is decoupled from the loading/residual scales. This is a **measure-preserving reparameterisation**: `p(params) p(factors | params) p(Z | factors) = p(params) p(Z | params) p(factors | Z, params)` by conjugacy, so the posterior over loadings, residuals, factor correlations, factor scores and structural slopes is **unchanged** — only the sampler geometry is. The structural Beta-Binomial leg still sees genuine per-child factor scores with full uncertainty; nothing is plugged in.
3. **Lift `target_accept` to 0.999** (via `spec.extra`, mirroring the horseshoe fit's funnel handling) to clear the residual boundary divergences — the strict gate requires **exactly zero**. This is a legitimate small-steps response for a model whose energy geometry is otherwise healthy; it is not papering over a funnel (BFMI is now ≈ 0.9).

### What was tried and then reverted: the prior recalibration

The **first** version of this fix also **recalibrated the priors** — a positive-mode `TruncatedNormal(0.6, 0.5, lower=0)` loading prior and a tighter `HalfNormal(0.5)` residual prior, replacing the original `HalfNormal(1)` pair — and reported the result as "the posterior is unchanged; only the sampler geometry". The review of #261 correctly rejected that: the conjugate rewrite is measure-preserving **only with the priors held fixed**, so changing the priors in the same step both invalidates the claim and confounds the two changes, leaving neither assessable.

`lrp-rli-mm-101` was added to separate them. It is LRPMM01 with the **priors as the single free variable** — same data, same likelihood, same `target_accept = 0.999`. A 2×2 over {original, recalibrated} priors × {0.95, 0.999} `target_accept`, all at reporting tier (6 chains × 6000 draws):

| priors                   | `target_accept` = 0.95                             | `target_accept` = 0.999                            |
| ------------------------ | -------------------------------------------------- | -------------------------------------------------- |
| original `HalfNormal(1)` | **FAIL** — 571 divergences, min ESS 370, BFMI 0.84 | **PASS** — 0 divergences, min ESS 2 200, BFMI 0.87 |
| recalibrated             | **FAIL** — 528 divergences, min ESS 850, BFMI 0.91 | **PASS** — 0 divergences, min ESS 1 700, BFMI 0.89 |

Three conclusions, and they overturn the original write-up:

- **The prior recalibration is neither necessary nor sufficient for convergence.** At `target_accept = 0.95` both prior sets fail by a comparable margin (571 vs 528 divergences); at 0.999 both pass. It is the **marginalisation** (which repairs BFMI: 0.21 → ≈ 0.87 — note BFMI is healthy in _every_ cell above, including the failing ones) plus the **raised `target_accept`** (which clears the boundary divergences) that carry the fix.
- **It does not move the posterior.** Factor correlations agree within 0.02 and every structural slope to the third decimal (below). The data dominate the prior — which is the reassuring result, but it had to be demonstrated, not asserted.
- **It is not free.** The recalibration moves the prior-implied **median communality from 0.50 to 0.79**, and `P(communality > 0.8)` from **0.29 to 0.49** (400 000 draws; `communality = λ²/(λ² + σ²)`). With only two indicators per factor at n ≈ 51 that is a substantive prior commitment about how well the indicators measure their domain — exactly the quantity the model exists to estimate. A further claim in the original code comment, that `HalfNormal(0.5)` "caps the residual SD below the unit total variance of a standardised indicator", is simply **false**: a half-normal is unbounded and merely makes `σ > 1` unlikely (≈ 5 % of prior mass).

Buying nothing and costing that, **the recalibration was reverted**. `lrp-rli-mm-001` now uses the original `HalfNormal(1)` priors, so its "measure-preserving, posterior unchanged" claim is finally **true as stated**, and its numbers stay comparable to the closed LRP66. `lrp-rli-mm-101` retains the recalibrated priors as a **prior-sensitivity companion**.

## Before → after (reporting config, 6 000 draws × 6 chains)

| Diagnostic       | Original build | Fixed build |
| ---------------- | -------------: | ----------: |
| Gate `passed`    |        `false` |      `true` |
| Divergences      |   422 (1.17 %) |           0 |
| BFMI (min chain) |           0.21 |        0.87 |
| min ESS          |            710 |       2 200 |
| max R-hat        |           1.00 |        1.00 |

The gated parameter set was also **widened**. The original gate listed only loadings, residuals and structural coefficients — it did **not** include `factor_corr` or `factor_z`, so the reported minimum ESS and maximum R-hat said nothing about the factor correlations, which are the model's headline output. (`factor_corr` cannot be gated directly: its constant unit diagonal has undefined R-hat and zero variance, so a check over the full matrix passes vacuously.) The factory now exposes the three unique off-diagonals as a dedicated `factor_corr_pairs` vector, and both it and `factor_z` are in the gated set. The numbers above are therefore certified for the quantities actually released.

## What the fixed fit says

The three latent domains are strongly correlated: **vocabulary↔grammar 0.80 [0.58, 0.94], vocabulary↔code 0.73 [0.45, 0.93], code↔grammar 0.65 [0.30, 0.91]** (all P > 0.999). The intervals remain **wide** — the honest result at n ≈ 51, exactly as the model's design note and the closed LRP66 flagged.

The structural factor → gain slopes stay **adjusted associations, not causal** (locked DAG ID-2): vocabulary +0.04 [−0.55, +0.66], code +0.27 [−0.33, +0.86], grammar +0.22 [−0.32, +0.76], all inconclusive; age −0.35 [−0.58, −0.12] (P(<0) ≈ 1.00).

### What this does _not_ establish

The earlier draft of this note read the high correlations as "a correlated skill system in which a single latent general ability `GA` is a defensible first approximation **but the domains are distinguishable**." The second half of that claim is **withdrawn**, per the review.

High positive correlations _within a specified three-factor model_ show that the three factors co-vary. They do **not** show that three factors are preferable to one. Perfect correlation is a boundary value excluded a priori by the continuous LKJ prior, so upper credible limits below 1 are **not** evidence of discriminant validity — the more so with only two indicators per domain. Establishing that the domains are distinguishable would need a **one-factor or bifactor comparator** with appropriate predictive model checks, which has not been run. (It is not a trivial addition here: the `corr_factor` pipeline skips PSIS-LOO, because two observed nodes — the indicator matrix and the structural outcome — make a single-target LOO ambiguous, so a comparator needs a model-comparison criterion chosen first.)

The claim this fit supports, and the only one that should be cited, is the narrow one:

> **Within the specified three-factor model, the vocabulary, code and grammar factor correlations are positive and high (0.65–0.80), with wide intervals at n ≈ 51.**

## A caveat on the predictive checks

The pipeline draws both observed nodes (`Z_obs`, the indicators; `y_post`, the structural outcome) in the prior and posterior predictive. These are **two separate checks, not a joint predictive draw**, and the code and reports now say so. The factor scores condition on the **observed** indicator data (`Z_d`), not on the replicated `Z_obs`, so a replicated indicator is statistically independent of the replicated factor it nominally loads on. Read `Z_obs` as a marginal check of the measurement covariance and `y_post` as a check of the structural leg _conditional on the observed indicators_; together they do not certify the joint model. The same caveat applies to the prior predictive, and more sharply — the `y_post` prior draws condition on observed data, so they are not a prior predictive of the outcome in the usual, data-free sense. A coherent joint simulation would need separate generative nodes (`factors ~ MVN(0, Corr)`; `Z | factors`; `y | factors`) alongside the inferential ones; that is not done, and the labelling above is the honest description of what the pipeline emits.

## Files changed

- `statistical_models/factories.py` — `build_correlated_factor_model`: marginalised MVN measurement likelihood + conjugate-conditional factor scores; new `factor_corr_pairs` deterministic (the gate-able unique off-diagonals); `loading_mu` / `loading_sigma` / `residual_sigma` parameters, defaulting to the **original** `HalfNormal(1)` priors; the predictive-simulation caveat documented at the node.
- `statistical_models/pipeline.py` — `fit_correlated_factor`: gate `factor_corr_pairs` and `factor_z`; accurate labelling of the two PPC nodes; `_apply_spec_target_accept` with explicit **CLI > model-specific > preset** precedence (previously a `max()` let a spec's 0.999 silently override an explicit `--target-accept 0.95`, so ablations and diagnostic reproductions did not run at the requested setting).
- `statistical_models/lrp_rli_mm_001.py` — `SPEC.extra["target_accept"] = 0.999`; prior overrides removed (uses the restored defaults).
- `statistical_models/lrp_rli_mm_101.py` — **new**: the prior-sensitivity companion.
- `docs/models/lrp-rli-mm-101/index.qmd` — **new**: its report.
- `tests/statistical_models/test_pipeline_fallback_defaults.py` — locked the prior defaults at the **original** values.

## Reconciled documents

Two notes had cited the failed fit's correlations as interpretable and were corrected to disclose the failed gate and point here: `notes/202607082140-statistical-models-full-reporting-fit.md` (§ convergence table and the measurement/summary bullets) and `notes/202607091430-dag-critical-review-td-atypical-literature.md` (the single-`GA`-vs-correlated-factors rows). The DAG review-draft note `notes/202607101444-dag-explanation-review-draft.md` had already been corrected to disclose the failure; its disclosure now reads against a fit that has since been fixed.

Any document repeating the withdrawn "the domains are distinguishable" reading should be narrowed to the boxed claim above.
