# Period-resolved dose-response + ITT restatement (#104, Phase 2, lean)

**Date:** 2026-06-22
**Scope:** The gated, **lean** Phase-2 follow-up to the period-resolved GB
diagnostic (`notes/202606221146-period-resolved-gb-diagnostic.md`, PR #106). Two
pieces: (1) a period-resolved Bayesian **dose-response** on word-reading
conditional change (LRP77, + a pooled comparator LRP77base and an ability-adjusted
sensitivity fit LRP77a). A period-1 ITT-plus-dose restatement was **dropped on
review** — it conditions on the `IS` dose collider (see the caution below). The within-control crossover and
any fan-out across other outcomes are out of scope. Fits at `--config reporting`
(max R-hat 1.002, min ESS ~4k, 0 divergences).

> **Provenance.** Re-synced onto current `main` (the `period`/`on_intervention`
> schema and #106's diagnostic are merged) and cites the **locked DAG**
> (`notes/202606231600-dag-revision-consolidated.md`). The reporting-tier figures
> below predate the `#117` sign flip and need a re-fit under `G = 2 − group` to
> refresh (the report template reads them from CSV, so re-rendering refreshes them).

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

> **Why there is no fitted "ITT + period-1 dose" model here (dropped on review).**
> Entering the period-1 dose alongside `group` in the ITT (the former LRP52d) is a
> DAG violation, not a restatement: dose is the locked DAG's `IS` node — both a
> *mediator* of `IG → WR` and a `GA`-collider (`IG → IS ← GA`). ID-1 is explicit:
> the ITT's minimal adjustment set is ∅ and conditioning on `IS` *"would bias even
> the ITT."* The period-1 collinearity that makes the dose appear to "absorb" the
> randomised contrast (controls have `attend == 0`) **is** that bias, not an
> insight. The randomised word-reading ITT is now `lrpitt10` (positive τ =
> intervention benefit under `G = 2 − group`); only the DAG-legal ID-3
> **observational** dose-response (LRP77, below) is kept as a fitted deliverable.

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

- **Sign convention (corrected by `#117`).** `main`'s
  `preprocessing.load_and_prepare` uses `G = 2 − group`, so `G = 1` is the
  immediate-intervention arm and `G = 0` the wait-list control; **positive `tau` =
  intervention benefit**. (An earlier draft carried a "group-coding hazard" flag
  describing an inverted mapping — that bug was fixed in `#117`, so the flag is
  removed.) The reporting-tier figures above predate the flip and need a re-fit to
  refresh under the corrected sign.
- **No raw change scores / confirmed ceilings only** (W = 79, confirmed) — both
  honoured.
- **Reproduce:** `python scripts/fit_statistical_model.py {lrp77,lrp77base,lrp77a} --config reporting --render`
  then `python scripts/compare_statistical_models.py --config reporting`.
