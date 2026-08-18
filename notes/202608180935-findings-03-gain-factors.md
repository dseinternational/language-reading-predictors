> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Findings: the `gain_factors` family — change from each child's own starting point

**Read `findings-00-overview` first.** This note covers the 32 models in the `gain_factors` family, the third independent route to the intervention effect.

## The data

**RLI trial only.** Unlike the `itt` family, these models use **every transition between consecutive timepoints** — timepoint 1→2, 2→3 and 3→4 — stacked into one dataset. Each row is one child in one period: their score at the end of the period, with their score at the start of that period as a covariate. A typical fit has 54 children contributing about 160 rows.

So the data **are** pooled across periods, but the causal quantity is not. The treatment effect is deliberately read off **period 1 only** — the randomised, all-untreated-baseline transition. Later periods contribute to estimating the covariate relationships and the residual variation, which sharpens the model, but they cannot contribute to the causal comparison because by then both arms have been treated.

## What the model is for

This is an ANCOVA on change: does a child who was receiving the intervention during a period end that period higher than their own starting point predicts?

Two features distinguish it from the `itt` approach. First, **stacking periods** means the model learns the general relationship between a starting score and an ending score from three times as much data, which makes the baseline adjustment more reliable. Second, each child gets a **random intercept**, partially pooling their stable tendency to score high or low.

That random intercept needs a caution attached. It is a shrunken summary of a child's stable level — it is emphatically **not** a control for latent general ability. A child's random intercept absorbs whatever is stable about them, including things you would want to adjust for and things you would not.

Every covariate other than the treatment indicator — the child's own baseline, age, cognitive ability, upstream skills, hearing, speech and phonological memory — is an **adjusted association**. The covariate sets were chosen from a causal diagram to make the treatment estimate clean, not to make those coefficients interpretable.

The headline models contain **no interaction with treatment**. Whether the effect varies by ability or starting point is asked separately in the `201`–`211` variants, and those answers are explicitly not causal.

## How to read the results

The reported quantity is the average marginal effect of being on the intervention during period 1, in items. Positive means the intervention helped.

| Measure                           | Effect (items) | 89% range    |
| --------------------------------- | -------------- | ------------ |
| Letter-sound knowledge (L)        | **+3.3**       | +1.6 to +5.0 |
| Word reading (W)                  | **+2.6**       | +0.9 to +4.3 |
| Taught expressive vocabulary (TE) | +1.0           | −0.1 to +2.2 |
| Taught receptive vocabulary (TR)  | +1.0           | −0.2 to +2.1 |
| Basic concept knowledge (F)       | **+1.0**       | +0.0 to +2.0 |
| Expressive vocabulary (E)         | +0.9           | −2.4 to +4.1 |
| Phoneme blending (B)              | **+0.8**       | +0.1 to +1.6 |
| Receptive grammar (T)             | +0.6           | −0.7 to +2.0 |
| Nonword reading (N), off-floor    | +2 pp          | −9 to +12 pp |
| Phonetic spelling (P), off-floor  | −1 pp          | −9 to +7 pp  |
| Receptive vocabulary (R)          | **−1.8**       | −5.6 to +2.0 |

## What was found

**The third route reaches the same destination.** Letter-sound knowledge +3.3 items (against +3.5 from `itt`, +3.5 from `did`); word reading +2.6 (+2.4, +2.2); phoneme blending +0.8 (+1.0, +0.9). Three models with different data windows, different adjustment strategies and different assumptions agree on both direction and rough magnitude for the outcomes the intervention targeted.

**Receptive vocabulary comes out negative here (−1.8 items), and this needs stating precisely rather than waved away.** The 89% range runs from −5.6 to +2.0 and therefore includes zero — but that is not how direction is judged in this project. The posterior probability that the effect is negative is **0.78**, which on the evidence ladder is _suggestive_ evidence of a negative effect, not "inconclusive". Reading it as inconclusive because the interval crosses zero would be the significance-testing habit the overview explicitly warns against.

That said, suggestive is the weakest rung above inconclusive — roughly 3:1 odds — and it sits against two other estimates of the same quantity that do not agree: the `itt` family gives +0.2 items for receptive vocabulary and `did` gives −0.1. Three routes to the same effect produce −1.8, +0.2 and −0.1, so the disagreement is itself the finding. The defensible summary is that broad standardised vocabulary shows no reliable movement, with one of the three models leaning weakly negative. It is emphatically not evidence that the intervention harmed vocabulary, and equally it should not be reported as a flat zero.

Taught vocabulary is weaker here (+1.0) than in the `itt` family (+1.4 and +1.5), with intervals that include zero. The alternative specifications `gf-012` and `gf-013` give +1.1 and +1.2 with intervals just clearing zero. So the taught-vocabulary effect is real but modest, and its apparent strength depends somewhat on how the model is set up — worth knowing before quoting a single number.

## The two companion sets

**Treated-only companions (`gf-101`–`108`)** restrict to children while they were receiving the intervention. Because everyone in that subset is treated, there is no comparison group and **no treatment effect is estimated at all**. Every number in them is an adjusted association. They exist to describe progress during intervention, not to evaluate it.

**Moderation variants (`gf-201`–`211`)** ask whether the on-intervention association varies with a child's starting point or general cognitive ability. These estimate the moderation terms **across all periods, including after crossover**, so their answers are not randomised. The project's release gate deliberately does not apply its causal checks to them, and they must not be reported as "the intervention worked better for children with higher ability". They can only say that, in these data, the association differed — with all the confounding that implies.

## The floored measures

Phonetic spelling and nonword reading use the off-floor rule: the outcome is whether the child scored above zero at all, and the effect is a change in that probability. Both are inconclusive (−1 and +2 percentage points, intervals about 20 points wide). For these two the model uses a **binary** indicator of whether the child was off the floor at the start of the period, rather than their graded starting score, because a graded score on a mostly-zero measure is nearly all zeros and carries almost no information.

Both were initially withheld for prior-dominance and released only after a prior-sensitivity sweep confirmed the direction is stable. As with the DiD family, that certifies direction, not magnitude — and for these two the direction is inconclusive anyway.

## What these models cannot tell you

**Only the period-1 treatment marginal is causal.** Every other coefficient describes which children progressed.

**The random intercept is not an ability control.** Any reading of the ability coefficient as "ability causes progress" is unsupported.

**The moderation variants are not causal**, even though they sit in the same family and share its machinery.

## Model inventory

All 32 pass the convergence gate with zero divergences and are publishable. Primaries: `gf-001` (W), `002` (R), `003` (E), `004` (L), `005` (P), `006` (B), `007` (F), `008` (T), `009`/`012` (TR), `010`/`013` (TE), `011` (N). Treated-only companions: `101`–`108`. Moderation variants: `201`–`211`.
