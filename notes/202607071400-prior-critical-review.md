<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8).

# Critical review of the statistical-model priors — starting with ITT, working forward

Date: 2026-07-07

## Purpose and relation to #141

Issue #141 audited the priors, fixed the `priors_table` completeness gaps, and shipped
the **two-tier τ** (proximal `Normal(0, 0.5)` / distal `Normal(0, 0.3)`) with a
sensitivity sweep. That work is sound and this review builds on it rather than
repeating it. The value added here is threefold:

1. **Interrogate the shared priors #141 largely accepted** — the intercept `alpha`,
   the own-baseline coupling `gamma_own`, and the dispersion `kappa` — with an
   **empirical prior-predictive pushforward** (not just the analytic items-scale
   translation #141 did for τ).
2. **Extend the audit to families #141 did not critically assess** — the bespoke
   LCSM / correlated-factor / two-mediator / adjusted (`lrp65`) priors, and everything
   that postdates #141: horseshoe (`lrphs`), the growth curves (`lrp69`/`lrp70`), and
   historical growth (`rlmhg`) — with **per-family pushforwards** of the bespoke
   coefficient priors (horseshoe shrinkage, mediation/mechanism paths, factor loadings).
3. **Cross-family consistency** — do the shared priors mean the same thing
   everywhere, and are the newer families coherent with the ITT baseline?

Scope: this covers every prior-bearing **Bayesian** family (priors are set per family,
not per model, so `lrpitt01–24` etc. are one review each). The 50 Layer-1
gradient-boosting models have LightGBM hyperparameters, not Bayesian priors; they are
outside a prior review but reviewed in a delineated appendix on request.

Provenance constraint from #141 still binds: `data/rli_data_long.csv` **is** the
Burgoyne et al. (2012) trial dataset, so that publication cannot set the _scale_ of
any effect prior (double-counting); it may only fix measurement facts. Priors are
Bayesian; the frequentist-reader bridge is in
`notes/202606261304-evidence-strength-and-rope-reporting.md`.

## Method

For each family: (a) read the registered priors off a built model (role + scale, via
`priors.py`); (b) translate each coefficient to the **items scale** through each
outcome's real Beta-Binomial denominator; (c) **prior-predictive pushforward** — draw
from the prior and push through the actual likelihood on the real baseline data,
comparing the implied post-score distribution to what DS children actually score.
A prior that is weakly-informative on the logit scale can be wildly informative on the
items scale of a long test — the recurring theme.

## ITT suite (`build_itt_model`) — the LRPITT family

Registered priors (LRPITT config: own baseline + linear age, empty cross/adjust set):
`alpha ~ Normal(0, 1.5)`, `tau ~ Normal(0, 0.5|0.3)` [tiered, #141], `gamma_own ~
Normal(1, 0.5)`, `gamma_A ~ Normal(0, 0.3)`, `kappa ~ HalfNormal(50)`.

### Prior-predictive pushforward (dev data, n = 53, 1000 draws)

Implied post-count distribution vs what children actually score:

| Outcome         | n_trials | observed post (min/med/max, SD) | prior-pred post (p05/med/p95, SD) | frac at floor 0 |
| --------------- | -------: | ------------------------------- | --------------------------------- | --------------: |
| W word reading  |       79 | 0 / 5 / 56, SD 12.6 (≤71%)      | 0 / 2 / 62, **SD 20.4**           |        **0.40** |
| R receptive voc |      170 | 15 / 36 / 68, SD 12.0 (≤40%)    | 0 / 37 / **148**, **SD 48.2**     |            0.08 |
| L letter-sounds |       32 | 2 / 22 / 31, SD 8.5 (**≤97%**)  | 0 / 15 / 32, SD 11.0              |            0.11 |

### Finding 1 (headline) — the _intercept_ `alpha ~ Normal(0, 1.5)` is not scale-invariant, and #141 tiered only τ

On the 170-item receptive-vocabulary test the ITT prior implies post-scores spanning
**0 to 148 items (SD 48)**, where DS children occupy 15–68 (SD 12). The dominant
driver is **not** τ (which #141 already tightened) but the intercept: near the
operating point (p ≈ 0.25 on R) the item count changes ≈ 36 items per logit, so
`alpha`'s ±1.5-logit (1 SD) prior alone sweeps ≈ ±50 items — essentially flat across
the whole plausible range. This is exactly the non-invariance #141 diagnosed for τ,
but for the intercept, and it was left at a common `Normal(0, 1.5)` for every outcome.

**Recommendation.** Two coherent options, both defensible:

- **Anchor the intercept** at the observed baseline logit and make `alpha` a _tight
  deviation_ prior — which is precisely what the newer `build_growth_model` and
  `build_lcsm_model` already do (`intercept_anchor`/`w1_anchor`). This also removes a
  **cross-family inconsistency**: the ITT/mechanism/DiD/mediation families leave the
  intercept free around 0, while the growth/LCSM families anchor it. Reconciling on
  the anchored form is the cleaner fix.
- Or **tier `alpha`'s scale** for the high-denominator distal outcomes as τ was tiered
  (e.g. `Normal(0, 1.0)` for R/E), keeping the probability-scale coverage sensible
  without the 0–170-item sweep.

The distal outcomes are already ~null (sweep in #141 §5a), so this cannot suppress a
real effect — it only stops the prior asserting implausibly dispersed item-scale
scores, and it makes the prior-predictive checks (which the report should show) look
sane on the long tests.

### Finding 2 — `gamma_own ~ Normal(1, 0.5)` is the one prior informative in its _mean_, and its SD is loose

`gamma_own` is centred at 1 (post-logit tracks pre-logit 1:1, i.e. **no regression to
the mean**) with SD 0.5 (95% ≈ 0 to 2: from no tracking to double). It is a _precision_
term, so it cannot bias τ — but it materially inflates the item-scale prior spread on
long tests (Finding 1), and #141's own recommendation (open decision 3) to anchor it
against published test–retest reliabilities (typically r ≈ 0.8–0.95 at these ages)
remains open. A retest r ≈ 0.9 supports keeping the mean near 1 but **tightening
the SD to ≈ 0.25**. This is the single most-informative coefficient prior in the suite
and the cheapest calibration win: the reliabilities are an admissible external source
(test manuals), unlike the trial data.

### Finding 3 — the graded proximal outcomes carry heavy prior floor mass

40% of the W prior-predictive mass sits exactly at 0. For genuinely floored outcomes
(P, N) the suite already switches to the off-floor Bernoulli estimand (good — the
item-scale concern does not apply there). But W is modelled graded, and a prior that
puts 40% of children at exactly zero word-reading is a strong floor assertion baked in
before the data. It is _defensible_ for DS word reading, and it is downstream of
Finding 1/2 (fix those and the floor mass falls), but it is worth a prior-predictive
panel in the report so the floor concentration is visible, not implicit.

### Finding 4 — `kappa ~ HalfNormal(50)` is permissive but under-anchored, not a red flag

The prior-predictive over-dispersion above is location-driven (α + baseline coupling),
not `kappa`: median prior κ ≈ 33 (near-binomial to moderate over-dispersion), tail to
≈ 98. HalfNormal(50) does allow small κ (real over-dispersion) so it is not obviously
mis-scaled, but its scale (50) is arbitrary and #141 flagged the normative raw-score
SDs as the admissible, still-unused anchor for it. Lower priority than α/`gamma_own`;
revisit only if a posterior-predictive dispersion check shows misfit.

### Finding 5 — ceiling censoring on the short tests (L, and T)

Observed letter-sounds uses **97%** of its 32-item scale (max 31/32); grammar (T) and
blending (B) are similar. A Beta-Binomial has no ceiling-censoring term, so children at
or near the maximum compress the measured effect — a measurement caveat that touches
τ, the growth-curve slopes (already flagged for `lrp69`/`lrp70`), and any item-scale
translation. Not a prior mis-specification per se, but the priors interact with it
(the floor/ceiling both truncate the pushforward), so it belongs in the same audit and
should be stated in the report's measurement caveats.

### What ITT gets right

The **two-tier τ** (#141) is well-judged and sensitivity-checked; the **role
discipline** (only τ causal; `gamma_own`/`gamma_A` precision; adjusters association) is
clean and machine-checked (`test_prior_inventory`); the **empty adjustment set** under
the locked DAG is correct; the **HSGP amplitude tightening** to `HalfNormal(0.3)` is a
sound, documented response to the LRP52 funnel (it trades flexibility for
identifiability at n ≈ 54 — a reasonable call, though worth a one-line note that it
_does_ cap the age nonlinearity the GP can express). The intercept and `gamma_own`
calibration (Findings 1–2) are the substantive gaps.

## Families that reuse the shared priors (joint, factors, DiD, aligned, adjusted, mechanism, mediation)

These inherit `alpha`, `gamma_own`, `kappa` unchanged, so **Findings 1–2 apply to all
of them** — the item-scale non-invariance of the free intercept and the loose
`gamma_own` is a _suite-wide_ property, not an ITT quirk. Family-specifics:

- **Joint (LRPITT12/15):** per-outcome `alpha`/`tau`/`gamma_own` vectors; keeps the
  **common** `tau ~ Normal(0, 0.5)` (deliberately _not_ per-outcome-tiered — #141
  cross-checks it against the tiered single-outcome τ's), so R/E get 0.5 in the joint
  fit vs 0.3 single-outcome (documented, minor). The LKJ residual correlation
  (`eta = 4`, SD `HalfNormal(0.5)`) and the age-GP are **off by default** — LRP55 found
  them prior-dominated (correlation CIs spanning zero); a sound default.
- **Gain/level factors:** `beta_trt` / `b_grp` / `beta_grp` correctly carry the
  **tiered** τ; `alpha_phase`/`alpha_time ~ Normal(0, 0.5)`; `sigma_child ~
HalfNormal(0.5)`; associations at 0.3. Clean reuse.
- **DiD:** `delta` / `beta_period` / `beta_dose` on the τ prior; `gamma_own`,
  `gamma_A`, `sigma_child`. Standard.
- **Aligned (per-protocol):** `beta_cohort` uses the **untiered** `Normal(0, 0.5)` —
  deliberately, since it is an _association_ (onset-aligned gain, not randomised) and
  so exempt from the τ tier (#141). Correct role call; but on a distal outcome a 0.5
  association carries the same item-scale spread flagged for α, so it wants the same
  anchoring.
- **Mechanism:** `beta_G` reuses the τ constructor but is **role-overridden to
  association** (a DAG backdoor, not randomised — a correct #141 fix); the mechanism
  slope `beta_mech ~ Normal(0, 1)` is one of the two loosest coefficient priors.
- **Mediation (+ two-mediator):** a-path (`a_G`, τ); b-path (`b_M`/`b_L`/`b_E` ~
  Normal(0, 1)) — the other loosest priors; `sigma_M ~ HalfNormal(1.0)`.
- **Adjusted (`lrp65`, between-child):** `eta = alpha + gamma_own·logit(W_pre) + Σ_k
βₖ·z_k` with `βₖ ~ Normal(0, predictor_slope_sigma = 0.5)` on each standardised
  baseline predictor, plus the shared `alpha`/`gamma_own`/`kappa`. Findings 1–2 apply
  (the free intercept and loose `gamma_own`); W is the outcome (n = 79, moderate
  occupancy) so the item-scale spread is milder than R/E. The `0.5` slope prior is a
  **third instance of the 0.3-vs-0.5 association drift** (§ below) — looser than
  `gamma_cross` 0.3, and #141 already lists `predictor_slope` for a `{0.3, 0.7}` sweep.

→ **`beta_mech` / `b_path` at `Normal(0, 1)`** remain the loosest coefficient priors
(a +1 SD move ≈ ×2.7 odds; empirically, at a mid-scale baseline the induced outcome
shift is Δp ≈ ±0.16 median, ±0.38 at p95 — see the per-family pushforward below) —
#141's flagged second sensitivity candidate after τ, still open.

## Families that postdate #141 (never critically audited)

- **Growth (`lrp69`/`lrp70`) — the exemplar for Finding 1.** It **anchors** the
  intercept at the grand-mean observed logit (`alpha ~ Normal(intercept_anchor, 1.5)`),
  exactly the fix recommended for ITT. But its association/slope priors sit at **0.5**
  (`assoc_prior_sigma`, `slope_prior_sigma`) — looser than the shared `gamma_cross`
  0.3 (see drift below); RE-intercept SD 1.0; positive-constrained loadings
  (identification).
- **LCSM (`lrp67`):** anchors the initial latent `mu1` at the observed wave-1 logit
  (good), but its couplings + age term use `coupling_prior_sigma` /
  `covariate_prior_sigma = 0.5` — again looser than the shared 0.3.
- **Historical growth (`rlmhg`):** `eta_group_wave ~ Normal(0, 1.5)` is **UNANCHORED**,
  and here the group×wave grid _is_ the estimand, on the `basread` 87-item test — so
  the item-scale logit-prior sweep of Finding 1 applies directly to the descriptive
  means. The clearest new "should anchor at the pooled grand-mean logit" case; also
  `sigma_subject ~ HalfNormal(1.0)` differs from the `HalfNormal(0.5)` used elsewhere.
- **Horseshoe (`lrphs`):** the regularized-horseshoe shrinkage (`tau0 = 0.1` global
  HalfCauchy, `lambda ~ HalfCauchy(1)`, slab `c² ~ InverseGamma(2, 8)`) is textbook
  Piironen–Vehtari and well-justified for a ranking cross-check (`tau0 = 0.1` encodes
  "few relevant predictors"). The induced coefficient prior (pushforward below)
  confirms the intended sparsity — **61 % of the mass falls within ±0.1 logit, median
  |β| ≈ 0.06** — so null predictors are shrunk hard. One nuance: the slab tail is
  heavier than the "≈2 logit" the slab scale suggests (p99.9 |β| ≈ 6 logit), because
  `c² ~ InverseGamma(2, 8)` has a fat right tail; an escaped coefficient can be
  implausibly large. Harmless for a _relative_ ranking cross-check, but if `lrphs`
  coefficients are ever read on the item scale, a tighter slab (smaller `slab_scale`,
  or a lighter-tailed `c²`) would cap escaped effects at a plausible magnitude.
- **Correlated factor (`lrpmm01`):** LKJ factor correlation (`eta = 2`), positive
  `HalfNormal(1)` loadings (identification), `sigma_indicator ~ HalfNormal(1.0)` —
  standard CFA. The pushforward confirms it is reasonable: `LKJ(2)` mildly regularises
  the inter-factor correlation toward 0 (median |r| ≈ 0.31, p95 ≈ 0.76), and the
  loading/residual priors imply within-domain indicator correlations of median ≈ 0.36
  (p95 ≈ 0.92) — loose but admissible for indicators of one latent. Fine as is; only
  tighten `loading_sigma` if a-priori near-unity indicator correlations are implausible.

## Cross-family consistency — three scale drifts

1. **Intercept anchoring is inconsistent.** Only growth + LCSM anchor the intercept;
   the count-outcome families (ITT, factors, joint, DiD, mediation, historical growth)
   leave it free at `Normal(0, 1.5)`. This only _bites_ for the high-denominator,
   low-occupancy outcomes (see the empirical extension below) — but the inconsistency
   is real, and the fix already exists in the codebase (growth/LCSM `*_anchor`).
2. **Association scale: 0.3 vs 0.5.** Established families use `gamma_cross ~
Normal(0, 0.3)`; growth + LCSM use 0.5 for their couplings/associations. Pick one
   intended association scale and reconcile — the newer families drifted looser without
   a documented rationale.
3. **Random-effect SD: 0.5 vs 1.0.** `sigma_child ~ HalfNormal(0.5)`
   (mechanism/DiD/factors/horseshoe) vs `sigma_subject` / `re_intercept ~
HalfNormal(1.0)` (historical growth, growth). Reconcile or document.

Plus: **`kappa ~ HalfNormal(50)` is universal** (good consistency) but universally
under-anchored — the normative raw-score SDs (#141's admissible, still-unused source)
would calibrate it once, everywhere.

## Empirical extension — all-families pushforward (anchored vs unanchored)

Running the pushforward across families sharpens Finding 1 and **corrects** one earlier
claim. Two contrasting families (dev data, 400 draws):

| Family (intercept)                           | scale            | observed (med, SD, occupancy) | prior-pred (med, p95, SD)           |
| -------------------------------------------- | ---------------- | ----------------------------- | ----------------------------------- |
| Growth `lrp69` (**anchored**)                | pooled R/E/T/W/L | med 22, p95 55                | med **18** (tracks), p95 126, SD 38 |
| Historical growth `rlmhg` (**unanchored** η) | basread, n = 87  | med 32, SD 26, **0–100%**     | med 43, p95 85, **SD 27**           |

- **The blow-out is conditional on occupancy, not universal.** The unanchored `rlmhg`
  η-grid is empirically _well-calibrated_ on `basread` (prior-pred SD 27 ≈ observed 26;
  p95 85 ≈ observed max 87) — because the Byrne mixed-reading-group cohort genuinely
  spans the whole 87-item scale, so a wide logit prior is _appropriate_ there. The
  severe mis-calibration is specific to **high-denominator, low-occupancy** outcomes —
  RLI R/E cluster in 9–40% of a 170-item scale, so the same wide prior sweeps far past
  the plausible range. This **corrects** the earlier reading of `rlmhg` as a
  "should-anchor" case: on `basread` it is fine; the target is the _occupancy mismatch_,
  not the unanchored intercept per se.
- **Anchoring fixes location, not spread.** Growth anchors the intercept and its
  prior-pred median (18) duly tracks the observed (22) — but the pooled p95 (126) is
  still wide (R/E-driven): the `Normal(anchor, 1.5)` deviation SD + the RE-intercept
  `HalfNormal(1.0)` + the age-slope prior keep the spread loose. So anchoring is
  **necessary but not sufficient** for the high-denominator outcomes — the _deviation_
  SD (and the RE scale) need tightening there too.

### Per-family pushforward — the bespoke coefficient priors

The three families with priors _not_ shared with ITT (their inherited
`alpha`/`gamma_own`/`kappa` are already covered above) were checked by sampling each
bespoke prior in isolation (200 k draws; LKJ via a standalone `sample_prior_predictive`):

| Prior (family)                               | quantity pushed forward                         | result                                                                      |
| -------------------------------------------- | ----------------------------------------------- | --------------------------------------------------------------------------- |
| Horseshoe `β` (`lrphs`)                      | induced coefficient (shrinkage profile)         | P(\|β\| < 0.1) = **0.61**, median \|β\| 0.06, p95 1.45, **p99.9 ≈ 6 logit** |
| `beta_mech` / `b_path` `Normal(0, 1)`        | Δp per +1 SD at mid-scale baseline (`p₀ = 0.5`) | Δp median **±0.16**, p95 **±0.38**                                          |
| — vs shared `gamma_cross` `Normal(0, 0.3)`   | (same)                                          | Δp median ±0.05, p95 ±0.14                                                  |
| Corr-factor `LKJ(2)` + `HalfNormal(1)` loads | inter-factor r ; implied indicator r            | factor \|r\| med 0.31 / p95 0.76 ; indicator r med 0.36 / p95 0.92          |

- **Horseshoe shrinks correctly** (61 % of mass within ±0.1) — the sparsity the ranking
  cross-check wants — but its **slab tail is heavier than the nominal ≈2 logit**
  (p99.9 ≈ 6); fine for a relative ranking, worth a lighter slab if ever read on the
  item scale.
- **`beta_mech` / `b_path` are ≈2.7× looser than `gamma_cross` on the probability
  scale** — a single mediation/mechanism path at its p95 shifts the outcome by more
  than a third of the probability range. This is the empirical substance behind #141's
  "second sensitivity candidate": try `Normal(0, 0.5)` (which would roughly halve the
  p95 Δp) unless a wider path prior is deliberately justified.
- **Corr-factor priors are reasonable** — `LKJ(2)` regularises the inter-factor
  correlation, and the implied indicator correlations stay admissible.

## Prioritised recommendations

1. **For high-denominator, low-occupancy outcomes (R/E, and any distal standardised
   test where DS scores cluster low): anchor the intercept _and_ tighten the deviation
   / RE scales.** Anchoring alone centres but does not narrow (growth shows this). This
   is **surgical, not a blanket suite-wide anchor** — outcomes that span their scale
   (`basread`, L) are already fine. (Finding 1, as refined by the pushforward. **See
   the provenance addendum below: this recommendation re-splits into a safe 1a and a
   double-dip-prone 1b.**)
2. **Tighten `gamma_own` SD to ≈ 0.25** against published test–retest reliabilities
   (r ≈ 0.8–0.95) — cheap, admissible external source, #141 open decision 3.
   (Finding 2.)
3. **Reconcile the drifts** — the 0.3-vs-0.5 association scale (now **three** families
   at 0.5: growth, LCSM, and adjusted `lrp65`'s `predictor_slope`) and the 0.5-vs-1.0
   RE-SD. (`rlmhg`'s unanchored η is deliberately _not_ on this list — the pushforward
   shows it is well-calibrated for `basread`.)
4. **Sensitivity-check `beta_mech` / `b_path`** at `Normal(0, 0.5)` vs 1 — #141's
   remaining recommended sweep, now empirically motivated: the per-family pushforward
   shows the `Normal(0, 1)` path prior shifts the outcome by ±0.38 (p95) at mid-scale,
   ≈2.7× the shared association scale; `Normal(0, 0.5)` roughly halves that.
5. **Anchor `kappa`** once against normative SDs; add **prior-predictive panels**
   (floor/ceiling mass, item-scale spread) to the reports so the pushforward is
   visible, not implicit.

None of these change the current substantive conclusions (distal outcomes null,
proximal effects keep direction — #141 §5a); they improve item-scale plausibility,
cross-family coherence, and defensibility to a critical reader. Each is a self-
contained follow-up that can land behind the existing shared-constructor seam.

## Prior provenance and anti-double-dipping

This is an exploratory **reanalysis of already-reported data** — `data/rli_data_long.csv` _is_ the Burgoyne et al. (2012) trial dataset — so the recommendations above must not launder the data at hand back in as a prior. #141 already closed the first double-dipping channel (Burgoyne's published SDs / effect sizes cannot set the scale of τ, the estimand). This addendum names the _second_ channel, which some recommendations above brush against, and states the discipline that keeps them clean.

**Two distinct risks.** (1) _Effect-prior double-counting_ — using this dataset (or its published summaries) to scale the causal estimand τ. Already handled: τ is tiered analytically from the items-scale translation, never from any observed effect. (2) _Self-calibration_ — setting a nuisance/scale prior (`alpha`, its deviation SD, RE scales, `kappa`) by looking at this sample's _realised_ statistics. Recommendation 1 as written ("tighten... so the item-scale spread matches observed") sits on this line.

**The bright line — two questions.** (a) _Does the prior touch the estimand?_ τ's prior must come from outside this dataset, full stop; `alpha`/`gamma_own`/`kappa`/RE-SDs cannot bias τ's mean but can distort its calibration and the item-scale predictions. (b) _Pre-treatment or outcome?_ Under randomisation the **baseline (t1, pre-randomisation) marginal is independent of treatment assignment**, so anchoring the intercept on the _arm-pooled baseline_ logit carries no information about τ; anchoring or tightening from **post-scores** (the outcome) does.

**Check vs. fit — the distinction that resolves most of it.** A prior-predictive pushforward is legitimate _diagnostics_ and illegitimate _calibration_ depending only on what it is compared against:

- **Check (legitimate).** "Does this prior imply item-scale predictions that are _a priori_ absurd — 0–148 items on a 170-item test where no DS child plausibly scores?" The comparison is to an **externally-justified plausible range** (test length, DS trajectories from _other_ cohorts, logical bounds). This is standard Bayesian-workflow prior-predictive checking, and it is what should catch the α blow-out (Finding 1).
- **Fit (double-dipping).** "Tune the prior SD until the pushforward reproduces _this sample's_ median/SD." Now the prior encodes the data about to be fitted. The observed-post columns of the pushforward tables are a diagnostic backdrop, **not** a target to match.

**Re-split of recommendation 1.** Finding 1's fix conflates a safe move with a risky one; separate them:

- **1a (safe) — anchor the intercept on the _pre-treatment, arm-pooled_ baseline logit**, framed as a location/nuisance term. Randomisation makes the baseline marginal independent of treatment, so no information about τ leaks in. Prefer an external DS-norm anchor where one exists; baseline-from-this-sample is the justified fallback and should be documented as such. This is what growth/LCSM already do.
- **1b (double-dip-prone) — set the _deviation_ and RE spread from an external plausible range, never from the observed post-scores.** Use the test manual's normative raw-score SD, or an explicit domain statement about where DS children of this age fall, to pick a weakly-informative spread centred on the _midpoint of that external band_. Do **not** reference this sample's realised occupancy (e.g. R's 15–68) as the target. The aim is "within an externally-plausible band", not "matching the observed distribution".

**Where the other recommendations stand.** Recommendations 2 (`gamma_own` ← test–retest reliability) and 5's `kappa` anchor (← normative raw-score SDs) are **clean**: both trace to the tests' standardisation samples (the manuals #141 lists as admissible), not to Burgoyne. Recommendation 5's prior-predictive panels are actively _protective_ — they make the check visible and show the prior was not reverse-engineered from the outcome. Only ensure the actual `r` and normative-SD values cited come from the manuals, not from this dataset.

**The backstop, since the data are already reported.** We cannot pre-register blind. The substitutes are (i) **documented external provenance** for every data-adjacent prior scale — extend #141's admissible-sources table so anything justified only by "matches what we observed" is flagged for re-derivation; and (ii) **sensitivity analysis as an acceptance criterion**, not an afterthought — a prior change ships only with a sweep showing the substantive conclusion is invariant across a defensible external range (as already done for τ). Where the outcome is informative (α, κ) the data will dominate, making any residual self-calibration inconsequential _and demonstrably so_ rather than merely asserted. Report the prior-predictive band and the external plausible band side by side so a critical reader can see they were set independently.

## Appendix — Layer-1 gradient-boosting models: hyperparameters, not priors

The 50 LightGBM models (`lrpgbg01–22` gain, `lrpgbl01–28` level) are frequentist and
have **no Bayesian priors**, so they fall outside a _prior_ review. Reviewed here on
request because they are part of the modelling suite; the analogue of "prior
specification" is the **regularisation regime** — the committed LightGBM
hyperparameters. Each model is tuned _independently_ (per-model Optuna, MAE objective,
GroupKFold `cv ≈ 51–53`), so unlike the Bayesian side (priors set once per family)
there are 50 separate regimes. Observed spread across the 50: `num_leaves` 10–63
(median 38), `max_depth` 3–12 (median 6), `min_child_samples` 4–37 (median 10),
`learning_rate` 0.012–0.19 (median 0.064), `n_estimators` 5–580 (median 66),
`reg_alpha`/`reg_lambda` 0.001–9 (median ≈ 0.04).

- **Stale/borrowed hyperparameters — already tracked by #169.** Six modules
  (`lrpgbg03/04/11`, `lrpgbl03/04/11`) carry no per-model tuning block; their params are
  `retune-pending`, retained from the earlier _pruned_-predictor-set Optuna runs and no
  longer matched to the current full predictor sets. This is the GB analogue of a
  mis-specified prior — settings not matched to the data they now fit — and it is the
  subject of open issue **#169** (reviewed retune of all 50 onto the full sets, promote
  into the declarative model defs; guarded by `tests/test_borrowed_params.py`). Highest
  value, already owned; nothing to add here beyond flagging it.
- **`num_leaves > 2^max_depth` in 16/50 — a search-space coupling bug.** In roughly a
  third of the models the committed `num_leaves` can never be reached because
  `max_depth` binds the tree first, so that tuned value is partly inert and misleading.
  Fold a fix into the #169 retune: either constrain `num_leaves ≤ 2^max_depth` in the
  search space, or (LightGBM-idiomatic) set `max_depth = -1` and control complexity via
  `num_leaves` alone.
- **Tuning-to-noise risk at n ≈ 54 with near-LOSO CV.** `cv ≈ 53` GroupKFold on ~54
  subjects is near leave-one-subject-out: near-unbiased but high-variance per fold, so
  the Optuna objective is itself noisy and per-model optima (`n_estimators` as low as 5;
  `learning_rate` up to 0.19) may be fitting CV noise rather than signal. The #169
  retune should use repeated/nested CV (or report tuning-objective stability), and could
  compare each per-model tune against a single robust shared config as a floor.
- **Regularisation is coherent but heterogeneous, no red flag.** `min_child_samples`
  median 10 (a leaf ≈ 19 % of the sample — moderate); six models at ≥ 30 are throttled
  to near-additive ensembles; L1/L2 penalties are mostly non-trivial (median ≈ 0.04) but
  near-off in six. The heterogeneity is the expected consequence of 50 independent tunes
  on a tiny sample — which is exactly why #169's _reviewed, consistent_ retune is the
  right remediation.

**Bottom line for the GB layer:** there is no prior to critique; the analogues of
prior mis-specification are (a) the stale/borrowed hyperparameters and (b) the
`num_leaves`/`max_depth` coupling — (a) is already #169, and (b) plus a tuning-to-noise
guard should be folded into that same retune.
