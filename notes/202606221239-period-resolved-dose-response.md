# Period-resolved dose-response + ITT restatement (#104, Phase 2, lean)

**Date:** 2026-06-22
**Scope:** The gated, **lean** Phase-2 follow-up to the period-resolved GB
diagnostic (`notes/202606221146-period-resolved-gb-diagnostic.md`, PR #106). Two
pieces: (1) a period-resolved Bayesian **dose-response** on word-reading
conditional change (LRP77, + a pooled comparator LRP77base and an ability-adjusted
sensitivity fit LRP77a); (2) a period-1 **ITT restatement** that makes the
dose-absorbs-group finding explicit (LRP52d). The within-control crossover and
any fan-out across other outcomes are out of scope. Fits at `--config reporting`
(max R-hat 1.002, min ESS ~4k, 0 divergences).

> **Provenance caveats (stacking).** Built on the #106 branch (the
> `period`/`on_intervention` schema is not yet on `main`) and cites shared DAG v5
> from a worktree note (not yet committed). Re-derive the adjustment set if v5
> changes on merge. See PR description.

## What was fitted

All models use the Beta-Binomial **conditional-change** likelihood: the word-
reading post-count modelled on its own baseline logit (`adjust_baseline_symbol`
= W, `n_trials` = 79) — never raw change scores. Sample: all three period
transitions stacked (n = 156 rows, 53 children), **including the waitlist
controls' period-1 zero-dose rows** (the anchor that identifies the slope).

- **LRP77** — period-varying dose-response. Dose = `attend` (per-period sessions)
  with partial-pooled per-period slopes; `attend_cumul` (prior cumulative dose) a
  control; adjustment `{G, A, W_pre}`; subject random intercept. +1 SD of dose =
  **+31 sessions** (mean 54, SD 31).
- **LRP77base** — identical but a single pooled dose slope (no period variation);
  the nested PSIS-LOO comparator.
- **LRP77a** — LRP77 + baseline-skill cluster (L, E, B) — the no-`g`->dose
  sensitivity fit.
- **LRP52d** — period-1 ITT for W entering both `group` and the period-1 dose
  (`fit_itt` + `adjust_for`), vs LRP52 (group only).

## Results

### Dose slope (logit per 1 SD dose ≈ 31 sessions)

| model | term | mean | 95% CI | P(>0) |
|---|---|---|---|---|
| LRP77 (period-varying) | overall (mu_dose) | 0.127 | [-0.127, 0.380] | 0.91 |
| LRP77 | period 1 | 0.127 | [-0.021, 0.274] | 0.96 |
| LRP77 | period 2 | 0.135 | [-0.028, 0.306] | 0.95 |
| LRP77 | period 3 | 0.124 | [-0.032, 0.279] | 0.95 |
| LRP77 | sigma (between-period) | 0.136 | [0.004, 0.560] | — |
| **LRP77base (pooled)** | **dose** | **0.127** | **[0.029, 0.227]** | **0.995** |
| LRP77a (ability-adjusted) | overall | 0.141 | [-0.122, 0.406] | 0.92 |

**The dose slope is small but credibly positive** — more sessions in a period
predict a larger conditional word-reading gain. The **pooled** estimate is
+0.127 logit per 31 sessions, 95% CI [0.03, 0.23], P(>0) = 0.995. The
period-varying model's *overall* slope has a wider CI only because the partial-
pooling hyper-prior inflates the hyper-mean's uncertainty; the three period
slopes are near-identical.

### Does the dose-gain relationship vary by period? — No

PSIS-LOO (`compare_statistical_models.py`, `dose_response_loo_compare.csv`):

| model | elpd_loo | p_loo | elpd_diff (se) |
|---|---|---|---|
| LRP77base (pooled) | -410.0 | 9.3 | 0.0 |
| LRP77 (period-varying) | -410.0 | 9.8 | -0.6 (0.13) |

The period-varying model does **not** improve fit (elpd_diff -0.6, within noise);
the between-period slope SD is small and the period slopes are indistinguishable.
**The dose-gain relationship is constant across periods** — consistent with the
Phase-1 finding that the signal sits on the dose axis, not the period-split axis.
LOO is read cautiously at this n (the LRP68 caveat).

### Does the dose effect survive ability adjustment? — Yes

LRP77a adds the baseline-skill cluster (letter-sounds, expressive vocabulary,
blending — the reflective indicators of latent ability `g`). The dose slope is
**essentially unchanged** (overall 0.141 vs 0.127). Baseline letter-sounds is
itself a credible predictor (`gamma_L_pre` 0.103, 95% CI [0.007, 0.201]), but the
dose signal does not run through it. So in this sample the **no-`ability`->dose
assumption is defensible** — the weak dose-response is not merely abler children
attending more. (The subject random intercept already absorbs *stable* child
ability; LRP77a adds the time-varying baseline-skill adjustment on top.)

### ITT restatement: dose absorbs the randomised group contrast

Group is coded **G = 1 = waitlist control, G = 0 = immediate intervention** (so a
negative tau means the immediate arm scores higher — see the coding flag below).

| model | tau (group), logit | 95% CI | gamma_attend (dose) | 95% CI |
|---|---|---|---|---|
| LRP52 (group only) | **-0.429** | [-0.785, -0.080] | — | — |
| LRP52d (group + period-1 dose) | -0.132 | [-0.813, 0.550] | 0.184 | [-0.18, 0.55] |

The period-1 ITT shows a **credible treatment benefit**: LRP52 tau = -0.43, CI
excludes 0 (immediate > waitlist; probability-scale AME -0.036, i.e. the immediate
arm ~3.6 percentage points higher; P(immediate higher) = 0.99). When the period-1
**dose** is entered alongside group (LRP52d), tau attenuates to -0.13 (CI now
straddles 0) and the positive dose term (gamma_attend +0.18) takes up the slack:
under the period-1 collinearity (controls have `attend == 0`) the continuous dose
absorbs the binary randomised contrast. **The randomised group tau (LRP52) is the
primary causal estimand; the dose is the mechanism it runs through** — they are
two expressions of one contrast, not independent effects.

## Decision / reading

The Phase-1 signal is confirmed and quantified with honest uncertainty: a
**small, credibly positive, period-invariant dose-response** on word reading
(+0.13 logit per ~31 sessions, pooled CI [0.03, 0.23]) that **survives ability
adjustment**, and a **credible randomised period-1 ITT benefit** that the dose
term absorbs. This is **adjusted association** for the dose-response (only the ITT
is randomised) and remains a **weak effect** — Phase-1 strata reached only R^2
0.1-0.3; the deliverable is a calibrated effect with credible intervals, not a
strong predictor. No period-resolved structure was found, so there is **no case
for a richer period-interaction model**; the within-control crossover (the
remaining Phase-2 option) stays deferred unless a second design-based
triangulation is specifically wanted.

## Flags for review

- **Group-coding hazard (pre-existing, not introduced here).**
  `preprocessing.py` maps dataset group 1 [Initial intervention] -> G=0 and group
  2 [Wait] -> G=1, but its inline comment says "0 = control, 1 = intervention" —
  inverted. So across the ITT family (LRP52-55, LRP60) `tau` is the coefficient on
  the **waitlist** indicator and `P(tau > 0)` is P(waitlist higher), **not**
  P(treatment helps). Verified against `data_variables.GROUP` and the raw attend
  pattern (group 1 has attend > 0 at t1). Likely relevant to the
  `fix/lrp78-itt-tau-ame` branch; flagged, not fixed here (out of scope, shared
  file in flight).
- **No raw change scores / confirmed ceilings only** (W = 79, confirmed) — both
  honoured.
- **Reproduce:** `python scripts/fit_statistical_model.py {lrp77,lrp77base,lrp77a,lrp52d} --config reporting --render`
  then `python scripts/compare_statistical_models.py --config reporting`.
