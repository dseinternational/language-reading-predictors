# Period-resolved / intervention-aligned GB diagnostic (#104, Phase 1)

**Date:** 2026-06-22
**Scope:** The cheap, pre-committed gradient-boosting diagnostic from #104,
Phase 1 — does resolving the near-noise gain models by **period** or by
**intervention status** concentrate signal that the all-periods pool dilutes, or
are gains genuinely near-unpredictable regardless of resolution? Builds on the
pruned `selection_steps` from #102 (`notes/202606201500-gb-replication-findings.md`,
`notes/202606211200-uniform-gb-fs.md`). Phase-2 Bayesian models are **not** built
here — they are gated on this result (see the decision below).

## Verdict (one line)

**Not the flat null, but not a predictability rescue either:** period-*splitting*
does not help (it dilutes via n ~ 50); pooling across **intervention status** was
a minor culprit (the intervention-aligned pool is consistently, if modestly, the
best stratum); and a weak, positive, period-1 / on-treatment-concentrated **dose
(`attend`)** signal survives for word-reading gains. This motivates a **targeted,
expectations-managed** Phase-2 (intervention-status / dose models, *not*
per-period-split models) — see the decision gate.

## Method

For each of the 11 gain primaries (registry models whose `target_var` ends in
`_gain`) the model's **committed pruned predictor set** (3–8 predictors from its
`selection_steps`, *not* the 32–34-predictor full sets) was augmented with the
study-design terms needed to read the dose and group contrasts in every stratum:

- `group` — the randomised contrast (immediate vs waitlist);
- `attend` — the per-period intervention dose (sessions that period);
- `attend_cumul` — cumulative *prior* dose (a dose-stage signal, so a dose-stage
  effect is not misread as a period effect);
- `age` — the maturation covariate (groups differ in age by period);
- `period` — the period index (skipped where the set already keeps `time`, which
  is numerically identical).

Each augmented model was fitted under five stratifications — `all` (the #102
baseline, for reference), `period1`/`period2`/`period3` (n ~ 50 each), and
`intervention` (on-treatment pool: immediate periods 1–3 + waitlist periods 2–3,
n ~ 131) — reusing the project's `EstimatorPipeline` / `LGBMPipeline` machinery
and each model's committed hyperparameters. **GroupKFold by `subject_id`**
throughout; a fixed **k = 10** for every stratum (so the comparison is internally
consistent and the n ~ 50 strata still leave several rows per validation fold —
near-leave-one-subject-out would give degenerate 1-row folds). Absolute R²
therefore differs from the #102 near-LOSO numbers; the comparison of interest is
*relative across strata*, on the same augmented set.

`period` and `on_intervention` are derived once in `data_utils.load_data`
(`Variables.PERIOD` / `Variables.ON_INTERVENTION`): `period = time`;
`on_intervention` is off iff `(group == 2) & (period == 1)`.

Reproduce: `python scripts/period_resolved_gb_diagnostic.py` (env: the ML stack +
`dse_research_utils`; writes `output/replication/period_resolved/`).

### Caveat that conditions every importance reading

The stepped design makes design terms collinear *within* some strata, so
permutation importance splits between them — do not read them as independent
magnitudes:

- **Period 1:** `attend` ≈ `group` (the waitlist controls have `attend == 0`),
  so the continuous dose absorbs the binary assignment.
- **Period 2:** `attend_cumul` ≈ `group` (only the immediate group has prior dose).

## Results

### Pooled out-of-fold R² by stratum

| model | target | all | period1 | period2 | period3 | intervention |
| --- | --- | --- | --- | --- | --- | --- |
| lrp01 | ewrswr_gain | 0.081 | 0.079 | -0.055 | -0.014 | **0.156** |
| lrp03 | eowpvt_gain | 0.077 | -0.055 | 0.084 | 0.049 | 0.061 |
| lrp05 | yarclet_gain | 0.260 | -0.010 | -0.035 | -0.033 | **0.287** |
| lrp07 | rowpvt_gain | 0.039 | 0.010 | -0.004 | -0.019 | **0.066** |
| lrp09 | celf_gain | 0.222 | 0.065 | 0.331 | 0.089 | 0.241 |
| lrp11 | trog_gain | 0.131 | 0.003 | 0.142 | 0.051 | **0.184** |
| lrp13 | nonword_gain | 0.168 | -0.018 | 0.032 | -0.041 | **0.175** |
| lrp15 | blending_gain | 0.174 | 0.026 | 0.145 | 0.027 | **0.183** |
| lrp17 | aptgram_gain | 0.121 | -0.018 | 0.205 | -0.115 | 0.143 |
| lrp19 | aptinfo_gain | 0.099 | -0.002 | -0.018 | 0.030 | 0.016 |
| lrp21 | deappfi_gain | 0.113 | 0.001 | 0.058 | 0.038 | **0.114** |

(Bold = single best stratum.) Per-fold R² mean/std are uninformative here — with
k = 10 over n ~ 50 the folds hold ~5 rows each, so per-fold R² is wildly negative
(e.g. lrp01 period2 fold-mean −1.69 ± 2.65). Pooled R² is the honest summary, as
in #102. Tellingly, the intervention-aligned pool is the **only** stratum where
the *fold-level* R² for lrp01 is even positive (+0.07 ± 0.20).

### `attend` (dose) permutation importance by stratum

| model | all | period1 | period2 | period3 | intervention | P1 SHAP sign |
| --- | --- | --- | --- | --- | --- | --- |
| lrp01 | 0.127 | **0.188** | -0.012 | 0.002 | 0.146 | +0.96 |
| lrp03 | -0.012 | -0.014 | -0.003 | -0.015 | -0.010 | +0.94 |
| lrp05 | 0.027 | 0.027 | 0.000 | 0.000 | -0.024 | +0.94 |
| lrp07 | 0.077 | 0.000 | 0.000 | 0.000 | 0.068 | n/a |
| lrp09 | 0.007 | 0.077 | 0.044 | 0.047 | -0.001 | +0.04 |
| lrp11 | -0.031 | 0.012 | -0.005 | -0.033 | -0.024 | +0.89 |
| lrp13 | -0.007 | 0.000 | 0.000 | 0.000 | -0.000 | n/a |
| lrp15 | 0.004 | -0.011 | 0.105 | -0.022 | -0.018 | -0.95 |
| lrp17 | -0.032 | 0.061 | 0.048 | 0.068 | 0.014 | -0.40 |
| lrp19 | -0.010 | -0.002 | 0.014 | -0.004 | 0.015 | -0.92 |
| lrp21 | 0.001 | 0.000 | 0.000 | 0.000 | -0.004 | n/a |

All importances sit within the fragility band #102 flagged (no predictor reaches
1 SD above zero). The one clear exception is **lrp01 (word reading)**: dose
importance ~0.19 in period 1 and ~0.15 on-treatment, ~0 in periods 2–3, with a
consistent positive SHAP sign (more dose → larger gain).

### `group` permutation importance by stratum

`group` is at the floor in **every** stratum, including period 1 (its randomised
contrast): the per-model period-1 importances are all within ±0.01 of zero. This
is not "no treatment effect" — under the period-1 collinearity the continuous
dose (`attend`) carries whatever treatment signal exists, so the binary indicator
adds nothing on top.

## Answers to the pre-committed questions

**Q1 — Does resolving by period or intervention status *materially* raise pooled
R² over the all-periods pool?**
*Period:* No. Per-period R² is mostly ≤ the all-periods pool and often negative;
only 1/11 (lrp09 period 2, 0.33 vs 0.22) clears +0.10, and it is a lone,
fold-unstable outlier (period 1 and 3 of the same model are 0.07 / 0.09). Small n
dilutes more than period homogeneity concentrates.
*Intervention status:* Weakly yes. The intervention-aligned pool is the single
best stratum for **7/11** models and ≥ the all-periods pool for **9/11** (median
delta ≈ +0.02, never large). Dropping the waitlist group's untreated period-1
gains — pure maturation / regression-to-the-mean with zero dose — removes
off-treatment heterogeneity that the all-periods pool mixes in. This is the one
place the "pooling dilutes signal" hypothesis is (mildly) confirmed.

**Q2 — Does the `attend` (dose) contribution concentrate in period 1 or vary by
period?**
Mostly at the noise floor. Where it has any signal it is positive (more dose →
larger gain) and leans to period 1 / the on-treatment pool — clearest for
word-reading gains (lrp01), and weakly for letter-sounds (lrp05) and concept
knowledge (lrp09). Period-1 dose importance exceeds 2× the period-2/3 mean in
3/11 models. It is *not* a broad period-varying dose effect across outcomes.

**Q3 — Does the `group` contribution concentrate in period 1?**
No — `group` is ≈ 0 in every stratum, period 1 included. Its period-1 randomised
signal is absorbed by the collinear continuous dose term, so the binary
assignment carries no additional importance.

## Decision gate

The strata are **not** "all equally flat", so this is not the stop-and-record-the-
null outcome — but neither did period resolution rescue predictability. The
structure that survives is on the **intervention-status / dose** axis, not the
period-split axis. That motivates a **targeted, expectations-managed Phase-2**,
listed here as a conditional follow-up (**not built in this PR**):

1. **Period-1 ITT** — the clean randomised contrast (immediate-on vs control-off),
   the primary causal estimand (anchor family LRP52–54/60). This is where the
   only randomisation lives; the GB diagnostic shows the treatment signal there is
   carried by dose, so estimate it directly with proper uncertainty.
2. **Within-control pre/post crossover** — waitlist controls off in period 1 vs
   on in periods 2–3, each child as its own baseline; a second, design-based
   estimate to triangulate with the ITT.
3. **Hierarchical period-resolved dose-response** — pool all periods with `period`
   random slopes and/or a `group × period` interaction (mechanism family
   LRP56–58/71–73), with the dose (`attend`) and dose-stage (`attend_cumul`)
   terms, so "does the dose–gain relationship vary by period?" is answered with
   partial pooling instead of paying the n ~ 50 penalty. Fit on **conditional
   change** (the existing Beta-Binomial on post-counts), **not** raw change
   scores (Lord's paradox / regression to the mean).

Prioritise word-reading (and possibly letter-sound / concept) outcomes; most
others showed negligible dose structure regardless of resolution.

## Manage expectations

Even in the best strata, pooled R² is 0.1–0.3 and the dose/group importances are
small and fragile. Phase-2's value is a **design-clean, interpretable effect
estimate with honest uncertainty** (the ITT τ; the within-control crossover
contrast; a period-resolved dose slope), **not** improved prediction of who gains
how much. Gains remain near-noise; resolving them did not change that — it only
showed *where* the small, real treatment signal sits (in active-treatment periods,
via dose) and where pooling buried it (mixing the controls' untreated period-1
gains into the all-periods pool).
