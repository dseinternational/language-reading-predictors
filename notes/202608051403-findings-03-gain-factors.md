<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Findings 03 — the gain-factor family (period-stacked ANCOVA)

Reports every model in the `gain_factors` family from the 2026-08-04/05 `reporting` refit. **21 models, all passing the convergence gate.** Reading conventions in note 00. Preliminary research data — all estimates provisional.

## What these models do

Two jobs at once, and it is important to keep them apart.

1. **Re-estimate the intervention effect from more data than the ITT suite uses.** The ITT models use only the randomised t1→t2 window — one row per child. These stack **every** adjacent-wave transition, on- and off-intervention, giving roughly three times the rows, with a child random intercept to handle repeated observations from the same child.
2. **Describe who progresses.** Each model carries the child's own baseline, upstream skills, age and non-verbal ability, so it reports which characteristics travel with larger gains.

**Design.** ANCOVA on the period's post-score given its own pre-score — conditional change, never raw change scores, which avoids Lord's paradox and regression to the mean. Beta-Binomial likelihood on a logit predictor, non-centred child random intercept.

**The single most important caveat in this family.** Only the **on-intervention term is causal**, and only because its marginal effect is averaged over the **period-1** transition — the randomised, all-untreated-baseline one. Every other coefficient — own baseline, age, non-verbal ability, upstream skills, the interactions — is an **adjusted association**, confounded by latent general ability. The child random intercept partially pools stable between-child differences; it is a shrunken stand-in, **not** a control for latent ability, and treating it as one is the main way to misread this family.

SES is deliberately excluded: it is not a node in the study's causal diagram and is statistically redundant given the covariates already present.

**Floored outcomes.** Phonetic spelling (PS) and nonword reading (NW) use the floor rule — a Bernoulli on the off-the-floor-at-post indicator, with the treatment effect as an **off-floor risk difference**, not items.

## Results — the thirteen effect-estimating models

Treatment effect is the period-1 on-intervention marginal, median with 89% range.

| Model    | Outcome                             | On-intervention effect (89%)  | P(>0) | Evidence     | ITT counterpart (note 01) |
| -------- | ----------------------------------- | ----------------------------- | ----: | ------------ | ------------------------- |
| `gf-004` | Letter sounds (LS)                  | **+3.3** items (+1.6 to +5.0) | 0.999 | very strong  | +3.5                      |
| `gf-001` | Word reading (WR)                   | **+2.6** items (+0.9 to +4.3) | 0.991 | very strong  | +2.4                      |
| `gf-013` | Taught expressive (TE) †            | +1.3 items (+0.2 to +2.4)     | 0.967 | moderate     | +1.5                      |
| `gf-012` | Taught receptive (TR) †             | +1.3 items (+0.1 to +2.4)     | 0.962 | moderate     | +1.4                      |
| `gf-010` | Taught expressive (TE)              | +1.2 items (−0.0 to +2.3)     | 0.944 | moderate     | +1.5                      |
| `gf-009` | Taught receptive (TR)               | +1.1 items (−0.1 to +2.2)     | 0.927 | moderate     | +1.4                      |
| `gf-007` | Basic concepts (LF)                 | +1.1 items (+0.1 to +2.0)     | 0.962 | moderate     | +0.9                      |
| `gf-003` | Expressive vocabulary (EV)          | +1.1 items (−2.1 to +4.3)     | 0.708 | inconclusive | +0.2                      |
| `gf-006` | Phoneme blending (PA)               | +0.8 items (+0.1 to +1.6)     | 0.961 | moderate     | +1.0 (link-sensitive)     |
| `gf-008` | Receptive grammar (RG)              | +0.8 items (−0.5 to +2.2)     | 0.835 | suggestive   | +0.7                      |
| `gf-002` | Receptive vocabulary (RV)           | −1.5 items (−5.3 to +2.4)     | 0.263 | inconclusive | +0.2                      |
| `gf-011` | Nonword reading (NW) — floor rule   | +2.5 pp (−8.0 to +13.0)       | 0.650 | inconclusive | +10 pp                    |
| `gf-005` | Phonetic spelling (PS) — floor rule | −2.0 pp (−11.1 to +6.8)       | 0.360 | inconclusive | +4 pp                     |

† `gf-012`/`gf-013` add broad vocabulary as an adjuster to the taught-word models.

**This is an independent replication of the ITT result, and it holds.** Letter sounds and word reading come out at +3.3 and +2.6 items against the ITT's +3.5 and +2.4 — agreement well inside the uncertainty, from a different set of rows and a different identification argument. That is the most important thing this family contributes.

The ordering is the same too: letter sounds strongest, then word reading, then the taught word sets and basic concepts, with broad vocabulary flat.

**Where it differs from the ITT suite, the differences are inside the noise.** RV moves from +0.2 to −1.5 and EV from +0.2 to +1.1 — both inconclusive in both families, with intervals spanning ±3–5 items. The floored outcomes are noisier here than in the ITT models (NW +10 pp → +2.5 pp) and inconclusive in this family; the ITT floor-rule branch is the registered estimand for those two, and this family's version should be read as a supporting check, not a competing number.

## The eight treated-only companions

`gf-101`–`gf-108` (WR, RV, EV, LS, PS, PA, LF, RG) refit each model on **only the periods when a child was receiving the intervention**. Because the treatment indicator is then constant, these models estimate **no treatment effect at all** — the run plan records both the declared and the active coefficient set so `config.json` never names a coefficient the posterior lacks.

Their purpose is purely descriptive: what travels with progress _during_ the intervention. Every result in them is an adjusted association. In six of the eight the clearest association is the same one: **the child's own starting point on the measure**, at direction probability 1.000. Children higher on a skill at the start of a period were higher at the end of it — which is autocorrelation, not a finding about what helps.

Two do not follow that pattern, and both for structural reasons. `gf-105` (phonetic spelling) uses the floor rule and so carries **no own-baseline term at all**; its clearest association is baseline letter sounds (P = 0.9997). In `gf-108` (receptive grammar) the own-baseline term is present but weaker than usual (P = 0.996) and is edged out by baseline receptive vocabulary (P = 0.9998) — grammar is the measure whose own past predicts its future least well in this family.

These companions are the right place to look for "who progresses" questions and the wrong place to look for "what works" questions.

## The recurring association result: age

Across this family the notable non-treatment association is that **age is negatively associated with gain, conditional on baseline**. This is not an artefact of the difficulty ladder or the likelihood: it was tested directly and survives (`notes/202607261405-binomial-exchangeability-item-difficulty-review.md` §3, which simulated Rasch-type ladders under a zero true age effect and found every ladder produces a _positive_ bias, the opposite direction).

The honest reading is that this cannot separate developmental timing from trajectory selection. Being older at the same score _means_ having grown more slowly historically, and no cross-sectional adjustment distinguishes "older children gain less" from "slower-progressing children are older when they reach any given score". Both readings are consistent with the data.

## Caveats

- **The random intercept is not an ability control.** It partially pools stable between-child variation. Latent general ability confounds every non-treatment coefficient regardless.
- **Only period-1 anchors the causal claim.** Later periods are post-crossover; the marginal effect is deliberately averaged over period 1 alone.
- **Available-case.** About 53–54 children at the period-1 transition, under ignorable missingness given the modelled covariates.
- **Predictive calibration.** 50% bands cover about 70% of observations. As elsewhere, substantially mechanical rather than a likelihood defect (note 00).
- **Items are not comparable across measures.**

## Where this leads

The gain-factor family answers "does the effect replicate with more rows?" — yes. The DiD crossover family (note 05) answers it again from the within-child crossover structure, and the aligned per-protocol family (note 06) from onset-aligned windows. The mechanism (note 07) and mediation (note 08) families take up the question this family cannot answer: what carries the effect.
