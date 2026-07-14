<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

# Longitudinal correlated-domain-factor model (LRP-RLI-LCF-001, #313)

Date: 2026-07-14

Related: #313, #314 (descriptive-association workstream, final item), #312/#328 (the concurrent-associations regression anchor this cross-checks against), #134 (`lrp-rli-mm-001`, the cross-sectional `corr_factor` precedent this extends across waves), #67/LCSM and #69/70/`growth` (the existing `WavePanel` longitudinal infrastructure this reuses), `dag/dag-language-reading.dagitty` (2026-07-10 revision).

## What this is

A four-wave latent measurement instrument: correlated domain factors (vocabulary / code / grammar) measured at every timepoint, delivering **per-wave latent skill correlation matrices** with posterior uncertainty and the disattenuated pairwise/conditional latent slopes derived from them. It is the symmetric, measurement-error-corrected counterpart to the #312 regression family: #312 gives directed, covariate-conditional observed slopes; this gives the underlying latent correlation web with binomial counting noise on both sides modelled out. Every quantity is a **descriptive association** — no causal reading is licensed anywhere in this model, so the level-factors family's mediator-conditioning prohibition does not apply (it protects a causal group×time contrast this model does not claim).

## Decisions settled before implementation

Two choices were signed off by Frank on 2026-07-14 (the rest follow the issue's stated defaults):

1. **New sibling kind, not the `mm` space.** The proposed id `lrp-rli-mm-002` is already taken (Ethan's merged errors-in-variables model, #315). More decisively, a four-wave latent model needs the rectangular child×wave `WavePanel` (`load_wave_panel`), whereas the `corr_factor` factory hard-requires one row per child (`phase_mode ∈ {span, itt}`) and is carefully tuned for the single-wave n≈51 geometry. Grafting a panel path onto it would bloat and risk destabilising the shared factory behind `mm-001`/`mm-002`/`mm-101`. So this is the "sibling kind if the pipeline diverges too far" branch the issue explicitly sanctioned:
   - **kind** `long_corr_factor`, **family code** `lcf` (distinct from `lf` level-factors and `lcsm`; the id regex sorts longest-first so no shadowing), **id** `lrp-rli-lcf-001` (legacy `lrplcf01`, module `lrp_rli_lcf_001.py`), an embedded-family id.
2. **Vocabulary folds in taught vocabulary.** The three factors are **vocabulary {R, E, TR, TE}**, **code {L, B}**, **grammar {F, T}**. Taught receptive/expressive vocabulary join the vocabulary domain rather than standing as a fourth factor. Coverage supports this: TR/TE are observed at all four waves (54/54/54/53 cells, same as R/E — block-1 taught words were assessed even at baseline t1), so the vocabulary factor is measured by four indicators at every wave. Heavily-floored P and N stay excluded, as in `mm-001`.

## Data (54 children × 4 waves, near-complete)

Reuse `load_wave_panel` with `outcomes=("R","E","TR","TE","L","B","F","T")`. Per-wave observed-cell counts confirmed: R/E/TR/TE/B/T = 54,54,54,53; L/F = 54,54,54,52. Indicators enter on the **Haldane-corrected logit** of the count (the panel's `logit` field; trial denominators R/E 170, TR/TE 24, L/F 32/18, B 10, T 32). Missing cells are **masked, not dropped** — a child with one missing wave still contributes its other cells, which matters at this n. This is the same Gaussian-on-logit measurement approximation `mm-001` uses; the Beta-Binomial likelihood is reserved for outcome legs, and this model has none (see below).

## Model structure (the constrained starting model)

A longitudinal reflective CFA, **measurement-only** — no structural Beta-Binomial outcome leg. `mm-001`'s unique output was a factor→W-gain slope; here the deliverable is the correlation web itself, and the latent slopes are a post-processing of it, so no outcome leg is needed. (An optional factor→reading structural leg is noted as a future extension, not built now.)

**Standardisation.** Each indicator's logit is standardised by its **pooled** (all-waves) mean and SD, so wave-to-wave level change is preserved in the data and carried by free per-wave factor means, not erased by per-wave re-centring.

**Measurement — scalar invariance (the constrained start).** For indicator `j` of domain `d` at wave `t`:

`z[i,j,t] = λ[j] · f[i,d,t] + ε[i,j,t]`, `ε ~ Normal(0, σ[j]²)`

with loadings `λ[j]` and residual SDs `σ[j]` **invariant across waves** (the factors mean the same thing at every t), positive loadings (orientation fixed), intercept 0 (pooled centring removes the grand mean; per-wave level lives in the factor means). Report the standardised loading / **communality** `λ²/(λ²+σ²)` per indicator, exactly as `mm-001`.

**Factor scale and per-wave correlation.** Factor variance fixed to 1 at every wave (the invariant loadings anchor the cross-wave scale). Per-wave **within-factor correlation** `Corr_t` (3×3) gets its own **LKJ** prior — these four matrices are the headline deliverable: does the vocabulary–code coupling tighten once everyone is on intervention (t1 mixed → t2–t4 all treated)? Free per-wave factor means `μ[d,t]` (soft zero-sum over waves per factor, consistent with pooled centring) carry growth.

**Across-wave dependence — trait/state decomposition (recommended).** The same 54 children recur across waves, so the repeated-measures dependence must be modelled or the per-wave correlations get dishonest intervals. Recommended structured form:

`f[i,d,t] = sqrt(π[d]) · τ[i,d] + sqrt(1−π[d]) · s[i,d,t]`

with a stable per-child **trait** `τ[i] ~ MVN(0, Corr_trait)` (cross-factor correlated, shared across waves) and a wave-specific **state** `s[i,t] ~ MVN(0, Corr_state,t)` (cross-factor correlated, independent across waves), `π[d] ∈ [0,1]` the trait share. This keeps unit factor variance exactly (`π + (1−π) = 1`), lets the within-wave correlation differ by wave (via `Corr_state,t`), induces positive across-wave autocorrelation through the shared trait, and is **PSD by construction** (sum of two independent Gaussians) — which a free per-wave LKJ plus a bolted-on AR block is not. It gives compound-symmetry across waves (all lags equally correlated); genuine AR(1) decay is the first relaxation if the equal-lag assumption misfits. A VAR(1) latent dynamic is the alternative principled form but it couples `Corr_t` to the recursion (can't also set per-wave LKJ freely), so trait/state is the cleaner start.

**Small-n geometry (the `mm-001` funnel fix, carried over).** The measurement model is Gaussian in the factors, so the per-child factor scores are **marginalised out**: each child's observed indicator cells are an `MvNormal` with the trait/state factor covariance folded into the indicator covariance (`Λ Σ_f Λ' + diag(σ²)`, sliced to that child's observed cells). This is measure-preserving and repairs the energy funnel that sinks the sampled-score parameterisation at this n. Expose the per-wave unique off-diagonal correlations as their own 1-D `factor_corr_pairs[t]` vectors so the strict gate evaluates exactly the released numbers (a full matrix's constant unit diagonal has undefined R-hat and silently passes).

## Deliverables

- **Per-wave factor correlation matrices** (`factor_corr` per wave) with posterior mean + 95% CrI + P(>0), and the off-diagonal pairs as gated vectors.
- **Derived latent slopes**: from `Corr_t`, the implied conditional-normal pairwise slope between factors (on unit-variance factors, the conditional slope of factor a on factor b is `ρ_ab`), plus **items-scale translations** for selected indicator pairs (through the loadings and trial counts) so the numbers are directly comparable with #312.
- **#312 cross-check section** (the honesty anchor). Two levels: (a) self-contained disattenuation check — the latent factor correlation should be **≥** the observed same-wave correlation between the two indicators in magnitude (measurement-error correction can only inflate); a latent correlation _below_ the raw observed one is a red flag for the latent model. (b) Directed reference — where #312 (`lrp-rli-ca-001`) supplies a concurrent partial slope for a matching predictor pair, the latent-derived items-scale slope should be systematically at least as large. (`ca-001`'s only focal is W, which is not a factor indicator here, so the directed comparison is available only for pairs #312 fits; extending `ca-001` to more focal measures to sharpen this is optional and out of scope for #313.)
- Report template (thin) + a new `_results_long_corr_factor.qmd` partial, catalogue row in `docs/models/README.md`, this dated note.

## Reporting, honesty, and convergence plan

- **Descriptive only.** No causal claims; wide intervals are the honest result at n≈54, exactly as `mm-001` and the closed LRP66 reported. Prior-sensitivity sweep required before any reported conclusion (an `lcf-101`-style companion varying the LKJ `eta`, loading, and trait-share priors, mirroring `mm-101`).
- **Priors** follow the `corr_factor` conventions and `docs/models/PRIORS.md`: `λ ~ HalfNormal(1)`, `σ ~ HalfNormal(1)`, LKJ `eta=2`, association-scale slopes at `sigma=0.3`; every prior in `priors_table.csv`.
- **LOO is included here** (unlike the LOO-exempt `mm-001`), pointwise per child, because the invariance decision is a LOO comparison: fit the scalar-invariant model first; relax to configural (per-wave loadings) or add AR(1) across-wave **only if** it misfits, and justify by LOO.
- **Sampling.** `rep-lite` first (portable, ESS-binding); marginalised geometry + lifted `target_accept` (spec-level, as `mm-001` at 0.999) to clear the strict zero-divergence gate. Expect real convergence effort — this is the most parameter-rich model in the descriptive workstream.

## Sequencing

Last item of #314. Unblocked now #312 (PR #328) is merged — its regression AMEs are the sanity anchor the cross-check needs. Implementation order: extend `load_wave_panel` coverage check → factory → pipeline + reporting → dev fit + gates + tests → PR against #313/#314.
