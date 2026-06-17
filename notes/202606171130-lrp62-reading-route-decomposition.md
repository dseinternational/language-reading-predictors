# LRP62 — reading-route decomposition (phonics-route composite mediation)

> [!WARNING]
> AI-assisted analysis. Numbers are reproducible from
> `python scripts/fit_statistical_model.py lrp62 --config reporting`; the
> interpretation should be reviewed by the study team.

Date: 2026-06-17

## Context

LRP62 is the robust replacement for LRP59's single-mediator estimate. LRP59 asked
how much of the intervention's word-reading effect runs specifically through
**letter-sound** knowledge and reported a proportion mediated of ~62% — but as a
single measure standing in for "the pathway", with a very wide interval. Following
the project review (`notes/2026-05-12-project-review.md`, §124/§167), LRP62 instead
asks how much of the effect is **phonics-route-compatible** — flowing through the
decoding skills *together* — versus a residual **lexical / whole-word-compatible**
path. This is the team's stated next step and the honest "use the measures together"
framing.

## Method

- **Phase 0 only** (`phase_mode="itt"`, t1→t2): the single randomised contrast,
  `n = 53` (one child dropped for a missing baseline, as in LRP59).
- **Mediator = phonics-route composite.** Equal-weight, standardised-logit mean of
  **letter-sound (L)** and **blending (B)**, conditioned on the baseline composite.
  **Phonetic spelling (P) excluded** — the measurement-sensitivity audit
  (`notes/202606171000-measurement-sensitivity-audit.md`) found it floored (78%→64%
  at zero, ~36% movers), so it carries little usable signal. Because the composite is
  continuous, the mediator leg is **Normal**; the **outcome leg and the
  counterfactual NDE/NIE g-formula are identical to LRP59** (see
  `factories._build_route_composite_model`, `mediation.decompose` gaussian branch).
- **Adjustment {G, A, E, R, W_pre}** (+ baseline composite, internal), confounders at
  baseline (cross-world assumption). Sign convention as LRP59: positive =
  intervention helps.

## Results (reporting config; intervention-helps, words out of 90)

| Quantity | mean (words/90) | 95% CI | P(>0) |
| --- | --- | --- | --- |
| **Total** | **+2.72** | [0.40, 5.07] | 0.99 |
| NDE (direct / residual) | +1.67 | [−0.56, 3.94] | 0.93 |
| NIE (phonics route) | +1.06 | [−0.04, 2.56] | 0.97 |
| Phonics-route share (NIE/Total) | 0.38 (median) | [−0.07, 1.48] | 0.99 (P(Total>0)) |

Diagnostics: **0 divergences**, max R-hat 1.00, min ESS_bulk ≈ 18.9k.

## Interpretation

- **The Total reconciles** with LRP52's τ_W and with LRP59's Total (~+2.9 words/90) in
  sign and rough magnitude — the decomposition is anchored to the same real effect.
- **Mixed routes.** The phonics route carries a credibly-near-positive indirect
  effect (NIE +1.06, P>0 = 0.97), but at the posterior mean the **direct/residual
  path is actually larger** (NDE +1.67). The route share median is ~38% with a very
  wide interval ([−7%, 148%]). The robust reading — as in LRP59 — is the **NDE/NIE
  split, not the ratio**: a substantial part of the word-reading gain is
  phonics-route-compatible, and a residual lexical-compatible path **cannot be ruled
  out** (and may dominate).
- **vs LRP59.** Broadening from letter-sound-only (LRP59: ~62%) to the L+B route
  lowers the central share (~38%) and widens it; the intervals overlap heavily. LRP62
  **supersedes** the single-mediator LRP59 as the way to talk about "how much is
  phonics" — it stops treating one measure as the pathway and reports the honest,
  wider uncertainty.

## Caveats (carried from LRP59)

- `n = 53`: posteriors are wide; do not over-read the point share.
- **Contemporaneous** mediator/outcome measurement (both at t2) — no temporal
  precedence within the window; conditioning on baselines mitigates but does not
  remove this.
- The binding, unverifiable assumption is **no unmeasured route→reading
  confounding** — the principal threat, not the sample size.
- The composite is a deliberately simple equal-weight summary of "the phonics route",
  not a latent-variable measurement model.

## Reproduce

```
python scripts/fit_statistical_model.py lrp62 --config reporting --render
# -> output/statistical_models/models/lrp62-reporting/mediation_summary.csv
# -> docs/models/lrp62/index.qmd  (rendered report)
```
