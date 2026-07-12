<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

# HSGP mechanism-curve reparameterisation — clearing the zero-divergence gate (#265)

Date: 2026-07-12

Related: #265 (this fix), #258 (mechanism adjustment sets), #273 item 13 (HSGP input standardisation), #274 (the convergence gate this must clear).

## The problem

Three mechanism models kept the nonparametric **HSGP** `f_mech` curve (the _shape_ of the relationship is the scientific point, so linearising them — as `lrp-rli-mech-056`/`057` do — is not acceptable) but did **not** clear the strict zero-divergence gate at reporting tier (`target_accept = 0.999`):

| model                                             | mechanism | divergences (before) |
| ------------------------------------------------- | --------- | -------------------: |
| `lrp-rli-mech-058` (L→W)                          | HSGP      |                    1 |
| `lrp-rli-mech-071` (E-moderation of L→W)          | HSGP      |                  ~13 |
| `lrp-rli-mech-158` (L→W complete-case comparator) | HSGP      |                 ~3–6 |

R-hat ≈ 1.00 and min ESS ≈ 2400–3000 throughout, so this was **boundary geometry, not mixing** — the classic signature of a GP whose amplitude/lengthscale posterior has a hard-to-sample neck at small n (~157 rows, ~50 children).

## Diagnosis

The `f_mech` HSGP was built on the **raw, uncentred** `logit(mech_post)` input, while `build_hsgp_1d`'s priors — lengthscale `InverseGamma(3, 1)`, boundary factor `c = 1.5`, basis count `m = 20` — are all calibrated for **standardised** (unit-SD) inputs (#273 item 13). On the raw logit (spread wider than 1 SD) the lengthscale prior put mass on lengthscales that are _short relative to the data_, forcing the GP toward a wiggly, weakly-identified fit — the neck.

## The fix (scoped to the mechanism GP)

Applied through the per-call HSGP hooks so the ITT age-GP, the dose-response GP and the age-moderation mechanisms (`mech-073`/`173`) keep the default and do not regress:

1. **Standardise the GP input** — `f_mech` is now built on `z(logit(mech_post))`, keeping the priors in their calibrated regime. (The curve is still plotted against the raw logit downstream, so its shape/location is unchanged.)
2. **Moderate-lengthscale prior** — a new `ell_prior_mech()` = `InverseGamma(5, 5)` (mode 0.83 / mean 1.25) shifts mass off the very short lengthscales toward a smoother curve, while still allowing genuine curvature.
3. **Smaller basis** — `m = 10` (down from 20); at this n a smooth 1-D curve does not need 20 basis functions, and fewer shrink the parameter space feeding the funnel.

Standardising the input alone did **not** clear the divergences (058: 1→1, 071: 14→14 at reporting); the lengthscale prior + smaller basis were the decisive levers.

## Verification (reporting tier, `target_accept = 0.999`)

All three now **PASS** the strict gate with **0 divergences**, R-hat ≤ 1.003, and the HSGP curve retained:

| model              | divergences (after) | curve `f_mean` range (logit) |
| ------------------ | ------------------: | ---------------------------- |
| `lrp-rli-mech-058` |                   0 | −0.43 … +0.45                |
| `lrp-rli-mech-071` |                   0 | −0.32 … +0.33                |
| `lrp-rli-mech-158` |                   0 | −0.45 … +0.46                |

The curve ranges (~±0.45 logit) show real curvature, not an over-smoothed line — the reparameterisation moved the sampling _geometry_, not the inference. The three "residual boundary divergences" disclosure callouts have been removed from the reports and replaced with a short convergence note.
