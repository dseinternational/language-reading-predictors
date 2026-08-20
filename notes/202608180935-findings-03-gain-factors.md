> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).
>
> Substantially corrected by a LLM-based AI tool (Codex/GPT-5).

# Findings: the `gain_factors` family — change from each child's own starting point

**Refreshed from the 2026-08-20 full refit** (`notes/202608200800-full-refit-both-layers-2026-08.md`); the hearing-composite recoding moved several values in the first decimal, and the numbers below are read from the new artefacts.

**Read `findings-00-overview` first.** This note covers the 32 models in the `gain_factors` family, a third model-based specification of the intervention contrast using the same trial.

## The data

**RLI trial only.** Unlike the `itt` family, these models use **every transition between consecutive timepoints** — timepoint 1→2, 2→3 and 3→4 — stacked into one dataset. Each row is one child in one period: their score at the end of the period, with their score at the start of that period as a covariate. A typical fit has 53–54 children contributing about 153–161 rows.

The data are pooled across periods, and one shared `beta_trt` coefficient enters every stacked transition. The reported items-scale marginal contrast is nevertheless standardised over **period-1 rows only** — the randomised, all-untreated-baseline transition. Later rows can influence the shared posterior through the fitted coefficient and the model's other parameters, but after crossover they contain no untreated comparison. The causal reading therefore belongs only to the period-1-standardised marginal and relies on the model's constant-treatment-effect structure; it is not an independently estimated period-1-only coefficient.

## What the model is for

This is a **period-stacked post-score ANCOVA conditional on pre-score**, not a regression on a literal difference score: after accounting for the child's score at the start of a period, is the ending score higher during intervention periods under the shared-effect model?

Two features distinguish it from the `itt` approach. First, **stacking periods** means the model learns the general relationship between a starting score and an ending score from three times as much data, which makes the baseline adjustment more reliable. Second, each child gets a **random intercept**, partially pooling their stable tendency to score high or low.

That random intercept needs a caution attached. It is a shrunken summary of a child's stable level — it is emphatically **not** a control for latent general ability. A child's random intercept absorbs whatever is stable about them, including things you would want to adjust for and things you would not.

Every covariate other than the treatment indicator — the child's own baseline, age, cognitive ability, upstream skills, hearing, speech and phonological memory — is an **adjusted association**. The covariate sets were chosen from a causal diagram to make the treatment estimate clean, not to make those coefficients interpretable.

The headline models contain **no interaction with treatment**. Whether the effect varies by ability or starting point is asked separately in the `201`–`211` variants, and those answers are explicitly not causal.

## How to read the results

The reported quantity is the fitted effect of switching the shared on-intervention term, averaged over the observed period-1 covariate profiles and translated into items. Positive means the intervention helped under this model.

| Measure                           | Effect (items) | 89% range    |
| --------------------------------- | -------------- | ------------ |
| Letter-sound knowledge (L)        | **+3.3**       | +1.6 to +5.1 |
| Word reading (W)                  | **+2.6**       | +0.9 to +4.3 |
| Taught expressive vocabulary (TE) | +1.1           | −0.1 to +2.2 |
| Taught receptive vocabulary (TR)  | +1.0           | −0.2 to +2.1 |
| Basic concept knowledge (F)       | **+1.0**       | +0.0 to +2.0 |
| Expressive vocabulary (E)         | +1.0           | −2.3 to +4.2 |
| Phoneme blending (B)              | **+0.8**       | +0.1 to +1.6 |
| Receptive grammar (T)             | +0.6           | −0.8 to +2.0 |
| Nonword reading (N), off-floor    | +2 pp          | −9 to +12 pp |
| Phonetic spelling (P), off-floor  | −1 pp          | −9 to +7 pp  |
| Receptive vocabulary (R)          | **−2.0**       | −5.8 to +1.8 |

## What was found

**This specification gives a similar result on the main targeted outcomes.** Letter-sound knowledge is +3.3 items (against +3.5 from `itt` and +3.5 from `did`); word reading +2.6 (+2.4 and +2.2); phoneme blending +0.8 (+1.0 and +0.9). These fits reuse the same children and randomised window, so they are not independent evidence, but agreement across their different adjustment and repeated-measures assumptions is a useful robustness check.

**Receptive vocabulary comes out negative here (−2.0 items), and this needs stating precisely rather than waved away.** The 89% range runs from −5.8 to +1.8 and therefore includes zero — but that is not how direction is judged in this project. The posterior probability that the effect is negative is **0.80**, which on the evidence ladder is _suggestive_ evidence of a negative effect, not "inconclusive". Reading it as inconclusive because the interval crosses zero would be the significance-testing habit the overview explicitly warns against.

Suggestive is the weakest rung above inconclusive — roughly 3:1 odds — and the 89% interval still leaves appreciable probability on either side of zero. The project's fixed vocabulary therefore treats this as weak evidence in the harmful direction, not as established harm. The `itt` family gives +0.2 items and `did` gives −0.1; all three intervals overlap. The defensible summary is that broad standardised vocabulary has no well-resolved benefit in these data, with this specification leaning weakly negative. It should be reported neither as a flat zero nor as a firm harmful effect.

Taught vocabulary is weaker here (+1.0 receptive and +1.1 expressive) than in the `itt` family (+1.4 and +1.5), with intervals that include zero. The alternative specifications `gf-012` and `gf-013` give +1.2 each, with intervals just clearing zero. The estimates are consistently positive and modest, but their evidential strength depends on the specification — worth knowing before quoting a single number.

## The two companion sets

**Treated-only companions (`gf-101`–`108`)** restrict to children while they were receiving the intervention. Because everyone in that subset is treated, there is no comparison group and **no treatment effect is estimated at all**. Every number in them is an adjusted association. They exist to describe progress during intervention, not to evaluate it.

**Moderation variants (`gf-201`–`211`)** ask whether the on-intervention association varies with a child's starting point or general cognitive ability. These estimate the moderation terms **across all periods, including after crossover**, so their answers are not randomised. The project's release gate deliberately does not apply its causal checks to them, and they must not be reported as "the intervention worked better for children with higher ability". They can only say that, in these data, the association differed — with all the confounding that implies.

## The floored measures

Phonetic spelling and nonword reading use the off-floor rule: the outcome is whether the child scored above zero at all, and the effect is a change in that probability. Both are inconclusive (−1 and +2 percentage points, intervals 16 and 21 percentage points wide). For these two the model uses a **binary** indicator of whether the child was off the floor at the start of the period, rather than their graded starting score, because a graded score on a mostly-zero measure is nearly all zeros and carries almost no information.

Both were initially withheld for prior-dominance and cleared for publication only after a prior-sensitivity sweep confirmed the direction is stable; their release status is the qualified `qualify` tier (prior-informed and exploratory), not an ordinary release. As with the DiD family, that certifies direction, not magnitude — and for these two the direction is inconclusive anyway.

## What these models cannot tell you

**Only the treatment marginal standardised to period 1 is given a causal reading**, under the shared-effect model and the available-case assumptions. Every other coefficient describes which children progressed.

**The random intercept is not an ability control.** Any reading of the ability coefficient as "ability causes progress" is unsupported.

**The moderation variants are not causal**, even though they sit in the same family and share its machinery.

## Model inventory

All 32 pass the convergence gate with zero divergences and are publishable. Primaries: `gf-001` (W), `002` (R), `003` (E), `004` (L), `005` (P), `006` (B), `007` (F), `008` (T), `009`/`012` (TR), `010`/`013` (TE), `011` (N). Treated-only companions: `101`–`108`. Moderation variants: `201`–`211`.
