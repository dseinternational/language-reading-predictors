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
   LCSM / correlated-factor / two-mediator priors, and everything that postdates
   #141: horseshoe (`lrphs`), the growth curves (`lrp69`/`lrp70`), and historical
   growth (`rlmhg`).
3. **Cross-family consistency** — do the shared priors mean the same thing
   everywhere, and are the newer families coherent with the ITT baseline?

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

## Families that reuse the shared priors (joint, factors, DiD, aligned, mechanism, mediation)

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

→ **`beta_mech` / `b_path` at `Normal(0, 1)`** remain the loosest coefficient priors
(a +1 SD move ≈ ×2.7 odds) — #141's flagged second sensitivity candidate after τ,
still open.

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
  HalfCauchy, `lambda ~ HalfCauchy(1)`, slab `c² ~ InverseGamma(2, 8)` capping at ≈2
  logit) is textbook Piironen–Vehtari, well-justified for a ranking cross-check
  (`tau0 = 0.1` encodes "few relevant predictors"). No concern.
- **Correlated factor (`lrpmm01`):** LKJ factor correlation, positive `HalfNormal`
  loadings (identification), `sigma_indicator ~ HalfNormal(1.0)` — standard CFA; fine.

## Cross-family consistency — three scale drifts

1. **Intercept anchoring is inverted.** Only the two newest per-outcome longitudinal
   families (growth, LCSM) anchor the intercept; every count-outcome family where the
   item-scale non-invariance bites hardest (ITT, factors, joint, DiD, mediation,
   _historical growth_) leaves it free at `Normal(0, 1.5)`. The fix already exists in
   the codebase (growth/LCSM `*_anchor`) — it just is not applied where it matters
   most.
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

## Prioritised recommendations

1. **Anchor the intercept** (or tier α for high-denominator outcomes) across the
   count-outcome families — the biggest item-scale-plausibility win, and it removes the
   growth/LCSM-vs-rest inconsistency. (Finding 1; suite-wide.)
2. **Tighten `gamma_own` SD to ≈ 0.25** against published test–retest reliabilities
   (r ≈ 0.8–0.95) — cheap, admissible external source, #141 open decision 3.
   (Finding 2.)
3. **Reconcile the drifts** — the 0.3-vs-0.5 association scale and the 0.5-vs-1.0
   RE-SD; anchor `rlmhg`'s `eta_group_wave`.
4. **Sensitivity-check `beta_mech` / `b_path`** at `Normal(0, 0.5)` vs 1 — #141's
   remaining recommended sweep.
5. **Anchor `kappa`** once against normative SDs; add **prior-predictive panels**
   (floor/ceiling mass, item-scale spread) to the reports so the pushforward is
   visible, not implicit.

None of these change the current substantive conclusions (distal outcomes null,
proximal effects keep direction — #141 §5a); they improve item-scale plausibility,
cross-family coherence, and defensibility to a critical reader. Each is a self-
contained follow-up that can land behind the existing shared-constructor seam.
