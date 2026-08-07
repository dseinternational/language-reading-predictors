<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Level factors: identified anchored intercepts, plan-owned contracts and the family treatment-prior sweep (#389, remaining findings)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5).

**Date:** 2026-08-07. **Issue:** #389, the level-factors critical review. Findings 1 (the t2 estimand — resolved and labelled via #464) and 3 (psense interlock — resolved by the #482 robustness release gate) were closed earlier; this note records the remaining work: **finding 2** (the intercept parameterisation, the one refit-bearing change), the verification trail for **findings 4 and 5** (largely landed with the typed run plan and the report partial), the closure of **finding 6's** last two acceptance criteria (data guards; plan-owned names), and the **family treatment-prior sweep** the release gate demanded for its withheld fits (criterion 6).

## Finding 2 — anchored, zero-sum intercepts (the methodological decision)

**The defect, restated.** The level model carried a free global `alpha` (zero-centred, tier scale) plus a free four-element `alpha_time ~ Normal(0, 0.5)`. Five location parameters serve four wave means, so only the sums were likelihood-identified — the review measured posterior correlations between `alpha` and `alpha_time` elements of −0.62 to −0.96 — and both pieces were centred at logit zero, implying prior-predictive scores near half each instrument's maximum against a population observed far below it (prior-predictive means W 39.1 / R 86.1 / E 84.1 items vs observed 11.8 / 41.1 / 35.0).

**The decision.** Both defects are repaired jointly, and each half follows an existing reviewed idiom rather than inventing one:

- `alpha_time` becomes an **exact zero-sum** wave-deviation vector (`ZeroSumNormal`), removing the translation ridge by construction — the identification half of the review's own recommendation ("a global intercept plus reference-coded/sum-to-zero time deviations"). Its scale is 0.75, sized so the largest observed wave deviation from the across-wave mean (±0.84 logits, phoneme segmenting; the others run ±0.19–0.65) sits within ~1.3 marginal prior SD rather than at ~1.9 as the old 0.5 implied — the same posterior-at-the-99th-percentile pattern #383 just spent an issue removing elsewhere.
- `alpha` becomes a Deterministic: a **pooled, arm-blind empirical-Bayes anchor at the observed pre-randomisation t1 logit** (Haldane-smoothed; the off-floor outcomes anchor on the t1 off-floor rate) plus a free zero-centred `alpha_offset` at the outcome-tier alpha scale. This is the DiD `alpha_anchor` idiom the #390 option-B ruling accepted, and the #481 labelling machinery applies by name: the priors table marks `alpha_offset` as empirical Bayes with the standard disclosure sentence. Anchoring at t1 rather than the all-wave mean deliberately uses **only pre-randomisation data** — the anchor under-centres the across-wave mean level by the (smaller) growth increment, a conservative trade against importing treatment-affected waves into the prior.

Observed t1 logits run −2.44 (W) to +0.03 (F), so the recentring is material exactly where the review said it was: the dev smoke's fitted W intercept is −2.54 (anchor −2.44 + offset −0.10), where the old prior centred it at 0.

**What this does not touch.** The t2 contrast `b_grp_time[1]` keeps its tier prior; the estimand and its at-mean-ability definition (#464) are unchanged; the anchor is arm-blind so the randomised contrast is not conditioned on any treatment-affected quantity.

## Finding 6's last criteria — data guards and plan-owned names

The typed `LevelFactorsSettings` / run plan landed with #394 pillar 4 and already carried the audit metadata (design, estimand, causal status, analysis population, missing-data assumption → `config.json` + `model_recipe.md`). Two acceptance criteria remained open and are closed here:

- **Fail before fitting** (criterion): the plan now owns `validate_prepared`, called after loading and before the output-affecting build: it rejects a panel whose t2 rows do not contain both randomised arms (the declared contrast would be unidentified) and any non-finite ability value on a fitted row (which previously would have propagated NaN silently into the likelihood — the factory's row filter only screens the outcome and requested adjusters).
- **Single source of truth** (criterion): `coefficient_names`, `diag_vars`, `causal_vector` and `causal_terms` are now plan methods; the pipeline's separate `_lf_coef_names` / `_lf_diag_vars` reconstructions are deleted. A pooled `beta_grp` (`group_by_time=False`) yields no causal term by construction — a pooled group coefficient mixes post-crossover waves.

## Findings 4 and 5 — verified, with the evidence trail

- **Finding 4** (available-case metadata and prose): the run plan records and persists the analysis population ("available-case children … the randomised interpretation applies to the t2 contrast on this available-case population") and the ignorable-missingness assumption in `config.json`/`model_recipe.md`, and the key-findings causal sentence renders it ("limited to the fitted available-case t2 population and assumes outcome and required-covariate observation do not depend jointly on arm and potential outcomes"). The gain-factor and DiD families carry the same metadata through their own typed plans.
- **Finding 5** (SP/RW timing): the shared factors partial carries a term-gated baseline-timing note — speech accuracy (`deapp_c`) and phonological memory (`erbto`) are read at the child's baseline t1 value and their coefficients described as **baseline-predictor associations**, with hearing named as the contemporaneous contrast where present.

## Criterion 6 — the level-family treatment-prior sweep

The #482 robustness release gate withholds a prior-dominant randomised claim unless "a `tau_prior_sensitivity.csv` treatment-prior sweep, computed from this fit's own trace" shows one sign across the grid — and no non-ITT runner existed, which is why `lf-001` (W) and `lf-005` (P) shipped `robustness_unresolved`. New `scripts/level_factors_prior_sensitivity.py`: for each of the review's five outcomes (W, L, P, B, N — all proximal-tier) it rebuilds the registered primary from its typed run plan with `b_grp_time`'s prior moved across the proximal grid (0.25 / 0.5 / 0.75), gates every cell on the full convergence criteria, emits the standard sweep schema (hash-bound to the primary's `config.json`/`trace.nc`, content-addressed cell traces, level-family provenance stamped as `model_kind: level_factors`), and — with `--attach` — writes the per-outcome rows beside each primary whose cells all converged. A levels model has no own-baseline term, so `gamma_own_sigma` is recorded as NaN; the off-floor outcomes report the risk difference through `n_trials = 1`.

## Refit verification (reporting tier, 2026-08-07)

_To be completed when the eleven refits, the sweep, and the cross-model comparison land._
