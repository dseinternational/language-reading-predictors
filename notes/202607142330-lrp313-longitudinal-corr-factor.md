<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).
>
> Substantially edited by a LLM-based AI tool (Codex/GPT-5).

# Longitudinal correlated-domain-factor model (LRP-RLI-LCF-001, #313)

Date: 2026-07-14

Related: #313, #314 (descriptive-association workstream, final item), #312/#328 (the concurrent-associations regression anchor this cross-checks against), #134 (`lrp-rli-mm-001`, the cross-sectional `corr_factor` precedent this extends across waves), #67/LCSM and #69/70/`growth` (the existing `WavePanel` longitudinal infrastructure this reuses), `dag/dag-language-reading.dagitty` (2026-07-10 revision).

## What this is

A four-wave latent measurement instrument: correlated domain factors (vocabulary / code / grammar) measured at every timepoint, delivering **per-wave latent skill correlation matrices** with posterior uncertainty and conditional latent slopes derived from them. It is a latent-measurement companion to the #312 regression family: #312 gives directed, covariate-conditional observed-score slopes; this gives a latent-domain correlation web while modelling indicator-specific residual variation. The derived conditional slopes are also directional and generally asymmetric, despite coming from a symmetric correlation matrix. These are different estimands, with no required magnitude ordering. Every quantity is a **descriptive association** — no causal reading is licensed anywhere in this model, so the level-factors family's mediator-conditioning prohibition does not apply (it protects a causal group×time contrast this model does not claim).

## Decisions settled before implementation

Two choices were signed off by Frank on 2026-07-14 (the rest follow the issue's stated defaults):

1. **New sibling kind, not the `mm` space.** The proposed id `lrp-rli-mm-002` is already taken (Ethan's merged errors-in-variables model, #315). More decisively, a four-wave latent model needs the rectangular child×wave `WavePanel` (`load_wave_panel`), whereas the `corr_factor` factory hard-requires one row per child (`phase_mode ∈ {span, itt}`) and is carefully tuned for the single-wave n≈51 geometry. Grafting a panel path onto it would bloat and risk destabilising the shared factory behind `mm-001`/`mm-002`/`mm-101`. So this is the "sibling kind if the pipeline diverges too far" branch the issue explicitly sanctioned:
   - **kind** `long_corr_factor`, **family code** `lcf` (distinct from `lf` level-factors and `lcsm`; the id regex sorts longest-first so no shadowing), **id** `lrp-rli-lcf-001` (legacy `lrplcf01`, module `lrp_rli_lcf_001.py`), an embedded-family id.
2. **Vocabulary folds in taught vocabulary.** The three factors are **vocabulary {R, E, TR, TE}**, **code {L, B}**, **grammar {F, T}**. Taught receptive/expressive vocabulary join the vocabulary domain rather than standing as a fourth factor. Coverage supports this: TR/TE are observed at all four waves (54/54/54/53 cells, same as R/E — block-1 taught words were assessed even at baseline t1), so the vocabulary factor is measured by four indicators at every wave. Heavily-floored P and N stay excluded, as in `mm-001`.

## Data (54 children × 4 waves, near-complete)

Reuse `load_wave_panel` with `outcomes=("R","E","TR","TE","L","B","F","T")`. Per-wave observed-cell counts confirmed: R/E/TR/TE/B/T = 54,54,54,53; L/F = 54,54,54,52. Indicators enter on the **Haldane-corrected logit** of the count (the panel's `logit` field; trial denominators R/E 170, TR/TE 24, L/F 32/18, B 10, T 32). Missing cells are **masked, not dropped** — a child with one missing wave still contributes its other cells, which matters at this n. This is the same Gaussian-on-logit measurement approximation `mm-001` uses; the Beta-Binomial likelihood is reserved for outcome legs, and this model has none (see below).

## Model structure (the constrained starting model)

A longitudinal reflective CFA, **measurement-only** — no structural Beta-Binomial outcome leg. `mm-001`'s unique output was a factor→W-gain slope; here the deliverable is the correlation web itself, and the latent slopes are a post-processing of it, so no outcome leg is needed. (An optional factor→reading structural leg is noted as a future extension, not built now.)

**Standardisation.** Each indicator's logit is standardised by its **pooled** (all-waves) mean and SD, so wave-to-wave level change is preserved in the data and carried by per-wave factor means with an exact zero-sum-over-waves constraint, not erased by per-wave re-centring.

**Measurement — wave-invariant parameters (the constrained start).** For indicator `j` of domain `d` at wave `t`:

`z[i,j,t] = λ[j] · f[i,d,t] + ε[i,j,t]`, `ε ~ Normal(0, σ[j]²)`

with loadings `λ[j]` and residual SDs `σ[j]` **invariant across waves** (the factors mean the same thing at every t), positive loadings (orientation fixed), intercept 0 (pooled centring removes the grand mean; per-wave level lives in factor means whose four wave values sum exactly to zero for each domain). Report the standardised loading / **communality** `λ²/(λ²+σ²)` per indicator, exactly as `mm-001`.

**Factor scale and per-wave correlation.** Factor variance is fixed to 1 at every wave (the wave-invariant loadings anchor the cross-wave scale). The reported per-wave **within-wave factor-correlation matrix** `Corr_t` (3×3) does not get its own LKJ prior. Instead, a trait correlation matrix and each wave's state correlation matrix receive LKJ priors, and the reported correlation between domains `d` and `e` is induced as `sqrt(π[d]π[e]) · Corr_trait[d,e] + sqrt((1−π[d])(1−π[e])) · Corr_state,t[d,e]`. The shared trait component therefore couples the four reported matrices. Their time ordering is t1 (both arms untreated), t2 (immediate arm treated and waitlist arm untreated), and t3/t4 (both arms treated); comparisons describe changes across waves, not intervention effects. Exact-zero-sum per-wave factor means `μ[d,t]` carry average level differences across waves.

**Across-wave dependence — trait/state decomposition (recommended).** The same 54 children recur across waves, so the repeated-measures dependence must be modelled or the per-wave correlations get dishonest intervals. Recommended structured form:

`f[i,d,t] = sqrt(π[d]) · τ[i,d] + sqrt(1−π[d]) · s[i,d,t]`

with a stable per-child **trait** `τ[i] ~ MVN(0, Corr_trait)` (cross-factor correlated, shared across waves) and a wave-specific **state** `s[i,t] ~ MVN(0, Corr_state,t)` (cross-factor correlated, independent across waves), `π[d] ∈ [0,1]` the trait share. This keeps unit factor variance exactly (`π + (1−π) = 1`), lets the state contribution differ by wave, induces across-wave dependence through the shared trait, and is **PSD by construction**. It gives compound symmetry across waves: the same-domain correlation between any two waves is `π[d]`, while a cross-domain, cross-wave correlation is `sqrt(π[d]π[e]) · Corr_trait[d,e]`. Genuine AR(1) decay is a possible relaxation only if the equal-lag assumption misfits. A VAR(1) latent dynamic is another principled alternative, but it would replace rather than supplement this trait/state covariance construction.

**Small-n geometry (the `mm-001` funnel fix, carried over).** The measurement model is Gaussian in the factors, so the per-child factor scores are **marginalised out**: each child's observed indicator cells are an `MvNormal` with the trait/state factor covariance folded into the indicator covariance (`Λ Σ_f Λ' + diag(σ²)`, sliced to that child's observed cells). This is measure-preserving and repairs the energy funnel that sinks the sampled-score parameterisation at this n. Expose the per-wave unique off-diagonal correlations as their own 1-D `factor_corr_pairs[t]` vectors so the strict gate evaluates exactly the released numbers (a full matrix's constant unit diagonal has undefined R-hat and silently passes).

## Deliverables

- **Per-wave factor correlation matrices** (`factor_corr` per wave) with posterior mean + 95% CrI + P(>0), and the off-diagonal pairs as gated vectors.
- **Derived latent slopes**: from `Corr_t`, the multiple-regression coefficient for each latent target on one domain while holding the remaining domain fixed, plus pairwise **items-scale translations** for selected indicator pairs (through the loadings and trial counts). These slopes are directional and generally asymmetric because reversing target and predictor changes the conditional regression denominator. They can be placed beside #312 but are not the same estimand or directly magnitude-comparable.
- **#312 triangulation section** (the honesty anchor). Two levels: (a) a self-contained comparison of each latent factor correlation with the mean same-wave correlation across the corresponding observed indicator pairs; and (b) the reproducible 32-row `lcf_concurrent_comparison.csv`, aligning target, predictor and wave for `L` against `R`/`E`/`TR`/`TE` and `TR`/`TE` against `L`/`B`. Each directed row places the LCF mean-operating-point target-item translation for a `+1 same-wave SD` predictor change beside #312's adjusted average marginal effect for the same change. Neither comparison has a required ordering: the LCF side conditions on the third latent domain at one operating point, whereas #312 conditions on the other observed tests plus age and group and averages its nonlinear marginal effect over rows. The latent correlation estimate carries a credible interval, but the mean observed indicator-pair comparator is a point estimate without its own interval, so that separate output does not quantify uncertainty in their gap. Treat both comparisons as descriptive triangulation, not pass/fail rules.
- Report template (thin) + a new `_results_long_corr_factor.qmd` partial, catalogue row in `docs/models/README.md`, this dated note.

## Reporting, honesty, and convergence plan

- **Descriptive only.** No causal claims; the current reporting estimates are nominal and exploratory at n≈54, not final magnitudes. A prior-sensitivity sweep is required before substantive interpretation (an `lcf-101`-style companion varying the component LKJ `eta`, loading, residual-scale and trait-share priors, mirroring `mm-101`).
- **Priors** follow the `corr_factor` conventions and `docs/models/PRIORS.md`: `λ ~ HalfNormal(1)` (implemented as the equivalent positive-truncated Normal), `σ ~ HalfNormal(1)`, `π ~ Beta(1.5, 1.5)`, exact-zero-sum factor means with scale 1, and LKJ `eta=2` for the trait and four state correlation matrices; every sampled prior is recorded in `priors_table.csv`. The conditional latent slopes are derived from the correlation draws and have no separately sampled slope prior.
- **PSIS-LOO is included here** (unlike the LOO-exempt `mm-001`), pointwise per child. Fit and check the wave-invariant measurement, compound-symmetry starting model first. Fit wave-varying loadings or AR(1) dependence only if model checks indicate the corresponding assumption misfits; if an alternative is fitted, compare it using the exact per-child log likelihood followed by PSIS-LOO.
- **Sampling.** `rep-lite` first (portable, ESS-binding); marginalised geometry + lifted `target_accept` (spec-level, as `mm-001` at 0.999) to clear the strict zero-divergence gate. Expect real convergence effort — this is the most parameter-rich model in the descriptive workstream.

## Reporting-tier validation (2026-07-15)

The wave-invariant measurement LCF was fitted with the `reporting` configuration (6 chains × 6000 retained draws, `target_accept = 0.999`, seed 47). The posterior passes the convergence gate: maximum R-hat 1.00102, minimum effective sample size 5157, minimum per-chain BFMI 0.800 and zero divergences. The report renders successfully. These checks validate sampling and report generation for this fit; the numerical estimates remain nominal and exploratory until prior sensitivity is run.

The first reporting run exposed a real pipeline defect: the per-child log likelihood needed for PSIS-LOO was silently skipped because PyMC 6.1's transform-free `LKJCorr` value-variable names gained an extra `_cholesky` suffix that did not match the saved posterior variables. The repair does not resample or approximate the likelihood. It evaluates each observed-pattern multivariate-normal log density directly from the posterior `mean_z` and `Sigma_z`, in bounded chunks, validates exactly one contribution per used child, and matches an independent SciPy calculation to `1e-10` across multiple missingness patterns. PSIS is then applied to that exact pointwise log likelihood; PSIS-LOO itself remains an importance-sampling approximation. The recovered 54-child PSIS-LOO is elpd −1663.99 (SE 55.70), effective parameter count 40.83; one child has 0.7 < Pareto k ≤ 1 (maximum 0.883) and none exceeds 1. Any future model comparison must inspect that influential child rather than relying on the aggregate elpd alone.

The same naming mismatch had also left the saved parameter-prior group empty and omitted `log_prior`, breaking the suite's prior-overlay and power-scaling contract. The repaired reporting trace now contains 1000 parameter-prior draws and exact constrained-scale log-prior terms for all nine free-RV groups. The resulting `psense_summary.csv` flags potential prior–data conflict for all 12 per-wave factor-correlation pairs and all three trait shares (prior sensitivity 0.054–0.104; likelihood sensitivity 0.202–0.336). This diagnostic is evidence that the pre-specified alternative-prior refit remains necessary; it is not a substitute for that refit.

The pipeline now materialises the directed #312 comparison as `lcf_concurrent_comparison.csv` rather than relying on a manual summary. Its 32 rows align the selected target/predictor pairs and four waves, with both quantities expressed for a `+1 same-wave SD` predictor change. Both are directional, but they remain different estimands: the LCF translation aggregates indicators into domains, holds the third latent domain fixed and evaluates one mean operating point; #312 holds the other five observed tests plus age and group fixed and averages its nonlinear marginal effect over analysed rows. The report therefore presents the artefact without a pass/fail ordering. Legacy correlation cross-check artefact and field names remain for compatibility only.

## Completion and remaining validation

The model, report, production convergence evidence, parameter-prior/log-prior groups, exact per-child log likelihood and resulting PSIS-LOO validate the implemented #313 software and output contract. This does **not** establish that #314 is scientifically complete or ready for substantive reporting: no alternative-prior companion has yet been fitted, and the power-scaling diagnostic itself flags potential prior–data conflict, so the current LCF values remain nominal and exploratory rather than prior-robust conclusions. Wave-varying loadings or an alternative across-wave structure are not unconditional deliverables; they should be fitted only if checks indicate that the wave-invariant measurement specification or compound-symmetry starting assumption misfits, then assessed with per-child PSIS-LOO. The remaining scientific validation should stay explicitly tracked before publication use.
