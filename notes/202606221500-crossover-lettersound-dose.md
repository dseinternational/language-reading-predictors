# Within-control crossover + letter-sound dose-response (#104, Phase 3)

**Date:** 2026-06-22
**Scope:** The two Phase-3 follow-ups agreed after Phase 2: (1) a **within-control
crossover** (LRP83) — the waitlist-as-own-control design from the meeting notes
(Snowling / Gertz / Bayliss et al.) — as a third, design-based estimate of the
treatment effect on word reading; and (2) **extending the dose-response to
letter sounds** (LRP86 period-varying + LRP86base pooled), to test whether the
small word-reading dose effect generalises to a decoding skill. Conditional
change only (Beta-Binomial on post-counts); reporting config, all three converge
(max R-hat 1.002, min ESS > 3k, 0 divergences).

> **Stacked PR caveat.** Built on the Phase-2 branch (`feat/lrp77-...`), which is
> on the unmerged #107 → #106. 3-deep stack; merge #106 → #107 → this in order.
> Re-derive against the (still-uncommitted) shared DAG v5 if it changes.

## Piece 1 — within-control crossover (LRP83)

Waitlist controls only (25 children, ~74 rows), off in period 1 and on from
period 2; within-child effect of `on` on word-reading conditional change
(`beta_on`, same logit scale as the ITT `tau`), with a subject random intercept,
a linear age maturation control, and **no period intercepts** (for controls `on`
≡ "period ≥ 2", so a per-period intercept would absorb the effect).

| term | mean | 95% CI | P(>0) |
|---|---|---|---|
| `beta_on` (logit) | **0.16** | [−0.11, 0.44] | 0.88 |
| AME (probability) | +0.015 | [−0.011, 0.041] | 0.88 |

**Positive, in the treatment-benefit direction, but individually uncertain** (CI
straddles 0; only 25 controls). On its own this is weak; its value is
**triangulation** — see below.

**Maturation caveat:** because `on` and period coincide for controls, `beta_on`
is the intervention effect *only* under no-strong-(differential)-maturation
beyond the linear age trend. It is the least rigorous of the three estimates.

## The word-reading triangulation (the payoff)

Three *independent identifications*, three different assumption sets, **all point
the same way** (treatment/dose helps word reading):

| design | model | estimate | credible? |
|---|---|---|---|
| Randomised ITT (between-arm, period 1) | LRP52 | τ = −0.43 [−0.79, −0.08] (immediate higher; `tau<0` = benefit) | **yes** |
| Dose-response (all periods, adjusted) | LRP77base | +0.13 logit / ~31 sessions [0.03, 0.23] | **yes** |
| Within-control crossover (within-child) | LRP83 | `beta_on` +0.16 [−0.11, 0.44], P=0.88 | directional |

The randomised ITT is the anchor; the dose-response and the crossover — each
leaning on *different* assumptions (selection-on-observables for dose;
no-strong-maturation for the crossover) — agree in sign. **Convergence across
methods is the strength of the word-reading conclusion**, not any single fit.

## Piece 2 — letter-sound dose-response (LRP86 / LRP86base)

Same machinery as LRP77, outcome swapped to letter-sound knowledge (L, N = 32).

| model | term | mean | 95% CI | P(>0) |
|---|---|---|---|---|
| LRP86 (period-varying) | overall | 0.13 | [−0.21, 0.45] | 0.86 |
| **LRP86base (pooled)** | **dose** | **0.14** | **[−0.001, 0.284]** | **0.97** |

**This came back more positive than expected.** Phase 1's GB permutation
importance hinted dose mattered *less* for letter sounds; the Bayesian
conditional-change model instead finds a letter-sound dose slope **comparable in
size to word reading** (+0.14 vs +0.13 logit/SD), borderline-credible (95% CI
lower bound ≈ 0, P(>0) = 0.97 vs word reading's clean [0.03, 0.23]). The likely
reason: the Bayesian model isolates the standardised dose term net of
`{G, A, L_pre}` + a subject RI, whereas GB permutation importance split the dose
signal across correlated predictors and buried it. So **the dose effect appears
to generalise to letter sounds (decoding), not just word reading** — a more
useful positive result than the null I'd flagged.

As for word reading, it **does not vary by period** — LOO prefers the pooled
model (`elpd_diff` −1.0); the period-varying fit carries a Pareto-k warning, so
its LOO is unreliable and only the pooled slope should be quoted.

## Bottom line

- The **word-reading treatment effect is now triangulated** across three designs
  (randomised ITT + adjusted dose-response + within-control crossover), all
  positive — the strongest the data support.
- The **dose effect generalises to letter sounds** (≈ same small magnitude,
  borderline-credible), updating the Phase-1 expectation.
- Everything stays **small and honestly bounded** (per-SD logit effects ~0.1–0.2;
  the GB ceiling was R² 0.1–0.3). The deliverables are calibrated effects, not
  strong predictors.
- The crossover and letter-sound dose are **adjusted / design-based**, not
  randomised — only the period-1 ITT is. Report direction + uncertainty.

## What is NOT pursued

Cross-lagged panel models (meeting point 4) remain out — LRP67/68 found them
degenerate at this n, and the repo holds a no-reciprocal-claims stance. With the
crossover done and the dose effect generalised, the #104 programme is complete.

## Flags (carried from Phase 2)

- `preprocessing.py` group coding: `G=0` immediate, `G=1` waitlist, but the inline
  comment is inverted — ITT `tau<0` = immediate benefits. Pre-existing; likely
  overlaps `fix/lrp78-itt-tau-ame`.
- Reproduce: `python scripts/fit_statistical_model.py {lrp83,lrp86,lrp86base} --config reporting --render`
  then `python scripts/compare_statistical_models.py --config reporting`.
