# Findings — joint family (all outcomes together, and taught-vs-not-taught contrasts)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8). Numbers from the `reporting`-config refit under the median + inner-50% + outer-89% equal-tailed credible-interval standard (2026-07-18; see [the credible-interval standard note](202607172359-credible-interval-standard.md) and [process note](202607161130-full-statistical-refit.md)); reviewed and extended on 2026-07-17 to cover all models in the family. Only the credible-interval brackets changed when we moved from 95% to 89% — medians, direction probabilities and evidence labels are unchanged. Preliminary.

## What these models ask

The [ITT models](202607161800-findings-itt.md) fit one outcome at a time. The **joint** family fits several outcomes **inside one model** so that we get a single, internally-consistent picture across outcomes and can form **contrasts between outcomes** — e.g. "did the programme help _taught_ vocabulary more than _not-taught_ vocabulary?". As with ITT, each outcome's treatment effect is **causal** (it comes from random assignment to the immediate-start arm); the taught-vs-not-taught contrasts are causal comparisons of two randomised effects. Two things about how these particular models are built matter for reading them, and both were wrong in the first draft of this note: the fitted models **do not** estimate a residual correlation between outcomes — they factorise by outcome (this is the documented design in `docs/models/README.md`), so there are no estimated cross-outcome residual correlations to report. And because within-child covariance is not modelled, the contrast intervals below **omit** that covariance and are therefore likely a little too narrow (anti-conservative); each contrast CSV's `dependence_note` says a paired child-level confirmation (randomisation-inference/permutation, bootstrap, sandwich, or a dependence-model sensitivity fit) is still required. Treat the contrast probabilities as provisional pending that check.

A note on the Bayesian quantities, for readers more used to frequentist output. An **89% credible interval** is the range within which the true effect lies with 89% probability given the data and priors — a direct probability statement about the parameter, unlike a confidence interval. **P(effect > 0)** ("prob_pos" below) is the posterior probability the true effect is positive; we grade it on the project evidence ladder (inconclusive < 0.75 ≤ suggestive < 0.91 ≤ moderate < 0.97 ≤ strong < 0.99 ≤ very strong). The **ROPE** ("region of practical equivalence") is a band around zero deemed too small to matter; the "big enough to matter" column below is the posterior probability the benefit clears a pre-set smallest-important-difference bar (`prob_benefit_ge_delta`) — a statement about the effect's _size_, not just its direction.

## Convergence gate

All 4 models **passed**. For the headline joint fit (ITT-012): maximum R-hat 1.0007, minimum ESS 13,206, 0 divergences, per-chain BFMI 0.97–1.01 (all comfortably above 0.3). ITT-015, ITT-115 and ITT-016 all report `passed = true`. No flagged models.

## Results — all models

### ITT-012 — the full ten-outcome suite, jointly (causal per outcome)

Every row is a **causal** intention-to-treat effect for one outcome, expressed on the items scale (how many more items, out of the test, a treated child gets right on average). The "big enough to matter" column is the probability the benefit clears that outcome's smallest-important-difference bar (Δ = 1 item for the vocabulary and Braille-type outcomes, Δ = 2 items for R, E, L; P has no bar). Own-baseline and linear-age terms in the model are **precision covariates**, not causal effects, and there are no skill-to-skill couplings in this family at all.

| Outcome                              | Best est. (items) | 89% credible range | P(effect > 0) | Evidence     | Big enough to matter (Δ) |
| ------------------------------------ | ----------------- | ------------------ | ------------- | ------------ | ------------------------ |
| **L** — letter sounds                | **+3.53**         | [+1.67, +5.35]     | 0.999         | very strong  | 0.904 (Δ=2) — clears     |
| **W** — word reading                 | **+2.37**         | [+0.68, +4.08]     | 0.987         | strong       | 0.903 (Δ=1) — clears     |
| **P** — (heavily floored)            | +1.82             | [−1.14, +4.82]     | 0.838         | suggestive   | no bar defined           |
| **TE** — taught expressive vocab     | +1.57             | [+0.44, +2.71]     | 0.986         | strong       | 0.789 (Δ=1) — clears     |
| **TR** — taught receptive vocab      | +1.32             | [+0.12, +2.49]     | 0.961         | moderate     | 0.667 (Δ=1) — clears     |
| **R** — (receptive language)         | +1.01             | [−3.11, +5.06]     | 0.655         | inconclusive | 0.349 (Δ=2) — no         |
| **B** — Braille-type                 | +0.91             | [+0.12, +1.67]     | 0.968         | moderate     | 0.426 (Δ=1) — no         |
| **UR** — not-taught receptive vocab  | +0.69             | [−0.01, +1.39]     | 0.942         | moderate     | 0.239 (Δ=1) — no         |
| **UE** — not-taught expressive vocab | +0.38             | [−0.35, +1.10]     | 0.799         | suggestive   | 0.087 (Δ=1) — no         |
| **E** — (expressive language)        | −0.08             | [−3.44, +3.34]     | 0.483         | inconclusive | 0.164 (Δ=2) — no         |

Reading this: best estimates run from a near-zero, very wide E (−0.08 items, interval spanning ±4) up to a tight, clearly-positive letter-sounds effect (+3.53, interval well clear of zero). The **interval widths matter as much as the medians** — R, E and P are wide and uncertain, whereas L, W, TE are narrow and pointed away from zero. Of the nine outcomes that have a "big enough to matter" bar (P has none), **four are more likely than not to clear it**: L (0.904), W (0.903), TE (0.789) and TR (0.667). The second-clearest effect after letter sounds is **word reading (W): +2.37 items, 89% [0.68, 4.08], 98.7% probability positive (strong)** — this was omitted from the first draft, which reported only letter sounds.

The fit also writes a descriptive `tau_contrast_matrix.csv`: a full 10×10 table of the posterior probability that each outcome's effect exceeds each other's. Letter sounds dominates every other outcome (probabilities 0.62–0.998). These between-outcome comparisons are **descriptive orderings**, not the pre-registered taught/not-taught contrasts, and are not causal claims about one outcome versus another.

### The three pre-registered contrasts (causal comparisons of two randomised effects)

These are separate two-outcome fits. Each reports both marginals (both causal) and a contrast. The contrast is on the risk-difference (proportion-correct) scale; positive = the first-named outcome gained more. All three contrast intervals omit within-child covariance (see `dependence_note`), so read the probabilities as provisional.

| Fit         | Contrast                            | Marginals (items, P>0, evidence)                       | Contrast estimate | 89% range     | P(contrast > 0) | Verdict          |
| ----------- | ----------------------------------- | ------------------------------------------------------ | ----------------- | ------------- | --------------- | ---------------- |
| **ITT-016** | TE − TR (modality)                  | TE +1.54, 0.986 (strong); TR +1.35, 0.966 (moderate)   | +0.8 pp           | [−6.0, +7.6]  | 0.572           | **inconclusive** |
| **ITT-015** | TE − UE (expressive generalisation) | TE +1.55, 0.985 (strong); UE +0.36, 0.787 (suggestive) | +3.4 pp           | [−4.3, +11.1] | 0.762           | **suggestive**   |
| **ITT-115** | TR − UR (receptive generalisation)  | TR +1.37, 0.968 (moderate); UR +0.73, 0.955 (moderate) | −0.3 pp           | [−8.0, +7.1]  | 0.471           | **inconclusive** |

On the logit scale the same three contrasts are +0.069 [−0.25, +0.39] (016), +0.186 [−0.18, +0.55] (015) and −0.048 [−0.41, +0.31] (115). Reading them:

- **Modality (016).** Taught expressive and taught receptive vocabulary benefit by almost the same amount; the direct contrast is essentially null (0.8 pp, P = 0.57). Both marginals are positive and both are more likely than not to clear their 1-item bar (TE 0.782, TR 0.685).
- **Expressive generalisation (015).** The only contrast that leans anywhere: taught expressive words gained a little more than not-taught expressive words (+3.4 pp, P = 0.76, suggestive). Of the two marginals only the taught side clears its 1-item bar (TE 0.785; UE 0.081). But note the CSV's own `transfer_interpretation`: a "limited generalisation" conclusion **cannot be read off the contrast alone** — it needs the not-taught (UE) marginal judged against a substantive negligible-effect threshold, which we have not set.
- **Receptive generalisation (115).** The direct taught-vs-not-taught contrast is **null and if anything slightly favours the not-taught side** (−0.3 pp, P = 0.47). Taught receptive vocabulary is larger only in the marginal medians (1.37 vs 0.73 items), not in the estimated contrast. The earlier draft's claim that "the taught side is the stronger" here is not supported by the contrast.

(The "97%" that a casual reader might attach to TR is the raw 0.968 — that sits below the 0.97 strong cut, so TR is graded **moderate**, not strong.)

## The one-paragraph story

Fitting the outcomes together tells the same story as the single-outcome ITT models: the effect is largest and clearest on **letter sounds** (+3.53 items, very strong) and **word reading** (+2.37 items, strong), then the **directly-taught vocabulary** (TE strong, TR moderate). Four outcomes — L, W, TE, TR — are more likely than not to clear a "matters in practice" bar; the rest are either genuinely uncertain (R, E, P have wide intervals) or positive-but-small (B, UR moderate direction, but unlikely to clear the bar). On the pre-registered generalisation question the picture is softer than the first draft implied: only the **expressive** contrast leans towards the taught words (suggestive, P = 0.76), while the **modality** and **receptive** contrasts are statistically null (P = 0.57 and 0.47). So the data are consistent with a targeted teaching effect concentrated on taught words, but the contrasts do **not** on their own establish that generalisation to untaught words was limited — the untaught marginals are merely inconclusive-to-suggestive (UE 0.799, UR 0.942 positive), and confirming any of the contrasts needs the paired within-child sensitivity analysis the CSVs flag.

## What is causal

Each outcome's treatment marginal (`tau`) is **causal** — it is identified by random assignment. The three taught-vs-not-taught contrasts (016 modality, 015/115 generalisation) are **causal comparisons of two randomised effects**. Everything else is not a causal claim: the own-baseline and linear-age terms are precision covariates; the 10×10 `tau_contrast_matrix` dominance probabilities are **descriptive orderings**. There are **no** residual-correlation parameters in these fits (the model factorises by outcome), and because within-child covariance is not modelled the contrast intervals are anti-conservative pending a paired child-level randomisation-inference / bootstrap / sandwich / dependence-model check.
