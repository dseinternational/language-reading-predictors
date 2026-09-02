> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5.1).

# Findings: the `gain_factors` family — change from each child's own starting point

**Read `findings-00-overview` first.** This note covers the 33 models in the `gain_factors` family: a third model-based specification of the intervention contrast using the same trial, plus the family's covariate associations, which are the project's main source of _predictor_ evidence on the gains scale. All 33 pass the convergence gate with zero divergences and all are publishable (2026-09-01 rebuild).

## The data

**RLI trial only.** These models use **every transition between consecutive timepoints** — 1→2, 2→3 and 3→4 — stacked into one dataset. Each row is one child in one period: the score at the end of the period, with the score at the start as a covariate. A typical primary fit has 53–54 children contributing 153–161 rows; the treated-only companions 130–135 rows.

One shared on-intervention coefficient enters every stacked transition, but the items-scale contrast is standardised over **period-1 rows only** — the randomised, all-untreated-baseline transition. Later rows inform the shared posterior; after crossover they contain no untreated comparison. Since #575 every model of record also refits itself on **period 1 alone** as a mandatory sensitivity, so the borrowing from later periods is measured rather than assumed.

## What the model is for

A **period-stacked post-score ANCOVA conditional on pre-score**: after accounting for the child's score at the start of a period, is the ending score higher during intervention periods under the shared-effect model? Each child gets a random intercept — a shrunken summary of their stable level, **not** a control for latent general ability. Every covariate other than treatment (own baseline, age, cognitive ability, upstream skills, hearing, speech, phonological memory) is an **adjusted association**. The headline models carry no treatment interaction; moderation is asked separately in the `201`–`211` variants, which are explicitly not causal.

## The randomised contrast

Period-1-standardised marginal contrast, in items (percentage points for the off-floor outcomes), with the period-1-only refit's treatment coefficient beside the stacked one on the model's logit scale:

| Measure                          | Effect (items) | 89% range    | P(>0) | Evidence     | Stacked β (logit)   | Period-1-only β     |
| -------------------------------- | -------------- | ------------ | ----- | ------------ | ------------------- | ------------------- |
| Letter-sound knowledge (L)       | **+3.3**       | +1.6 to +5.1 | 0.999 | very strong  | 0.57 [0.27, 0.87]   | 0.59 [0.23, 0.94]   |
| Word reading (W)                 | **+2.6**       | +0.9 to +4.3 | 0.992 | very strong  | 0.39 [0.13, 0.65]   | 0.51 [0.20, 0.80]   |
| Taught expressive (TE, `gf-013`) | **+1.2**       | +0.1 to +2.3 | 0.95  | moderate     | 0.25 [0.01, 0.50]   | 0.23 [−0.06, 0.51]  |
| Taught receptive (TR, `gf-012`)  | **+1.1**       | +0.0 to +2.3 | 0.95  | moderate     | 0.22 [0.00, 0.43]   | 0.19 [−0.06, 0.44]  |
| Taught expressive (TE, `gf-010`) | +1.1           | −0.1 to +2.2 | 0.93  | moderate     | 0.22 [−0.02, 0.47]  | 0.19 [−0.10, 0.47]  |
| Taught receptive (TR, `gf-009`)  | +1.0           | −0.2 to +2.1 | 0.91  | suggestive   | 0.18 [−0.04, 0.40]  | 0.16 [−0.09, 0.41]  |
| Basic concept knowledge (F)      | **+1.0**       | +0.0 to +2.0 | 0.95  | moderate     | 0.26 [0.00, 0.52]   | 0.22 [−0.09, 0.52]  |
| Expressive vocabulary (E)        | +1.0           | −2.2 to +4.2 | 0.69  | inconclusive | 0.04 [−0.09, 0.16]  | 0.01 [−0.15, 0.17]  |
| Phoneme blending (B), ordinary   | **+0.8**       | +0.1 to +1.6 | 0.96  | moderate     | 0.39 [0.03, 0.75]   | 0.33 [−0.09, 0.73]  |
| Phoneme blending (B), floor link | +0.5           | −0.1 to +1.0 | 0.92  | moderate     | 0.48 [−0.05, 1.01]  | 0.35 [−0.23, 0.93]  |
| Receptive grammar (T)            | +0.6           | −0.7 to +2.0 | 0.76  | suggestive   | 0.08 [−0.10, 0.26]  | 0.11 [−0.10, 0.31]  |
| Nonword reading (N), off-floor   | +2 pp          | −9 to +12    | 0.62  | inconclusive | 0.12 [−0.51, 0.75]  | 0.08 [−0.58, 0.74]  |
| Phonetic spelling (P), off-floor | −1 pp          | −9 to +7     | 0.60  | inconclusive | −0.10 [−0.76, 0.57] | −0.06 [−0.72, 0.59] |
| Receptive vocabulary (R)         | −2.0           | −5.8 to +1.8 | 0.81  | suggestive   | −0.07 [−0.20, 0.06] | −0.02 [−0.17, 0.15] |

For receptive vocabulary the evidence label refers to the **negative** direction (P(effect < 0) = 0.81).

**On the targeted skills this specification agrees with the others.** Letter sounds +3.3 (against +3.5 in `itt` and `did`), word reading +2.6 (+2.4, +2.2), blending +0.8 (+1.0, +0.9). The cross-family triangulation table (`triangulation_consistency.csv`) records direction agreement and overlapping intervals for W, E, L, TR, TE and F across the three designs; receptive vocabulary is the one outcome where direction does not agree.

**Receptive vocabulary leans negative here (−2.0 items, P(negative) = 0.81)** — suggestive evidence in the harmful direction under this specification, not inconclusive and not established harm. The other families give +0.2 (`itt`), −0.1 (`did`) and +0.3 (`level_factors`), all with intervals that overlap this one. The period-1-only refit moves the coefficient from −0.07 to −0.01 logits, so most of the lean is carried by the stacked post-crossover rows. The defensible summary is unchanged from August: broad vocabulary has no well-resolved benefit, and this family's lean is a property of its stacking.

**What the period-1-only refits show.** Word reading and letter sounds are robust either way, and for word reading the stacked model if anything shrinks the effect (0.39 against 0.51). For blending, basic concepts and the taught-vocabulary variants the period-1-only coefficient is a little smaller and its interval, on roughly a third of the rows, includes zero. Medians move little; the crossings are mostly the smaller sample. Stacking therefore inflates the language and taught-vocabulary estimates modestly and the reading estimates not at all.

Taught vocabulary is weaker here (+1.0 to +1.2 items) than in `itt` (+1.4 and +1.5), with intervals touching zero; the `gf-012`/`013` specifications that add broad vocabulary as descriptive associates clear zero narrowly. Consistently positive and modest, with evidential strength that depends on the specification.

## The predictors of gain

Every covariate below is an adjusted association: which children progressed, not what would move them. Items per +1 SD of the covariate (the hearing flag is a 0→1 toggle), from the thirteen primaries, grouped by the posterior probability of the stated direction (the model is named where an outcome has two primaries; the probability follows each entry in the weaker columns):

| Covariate                | Strong or better (P ≥ 0.97)                                                                                                                                                           | Moderate (0.91 ≤ P < 0.97)                                          | Weaker or flat                                                                                 |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Own starting score       | every outcome except nonword reading (e.g. W +13.9 items per SD, R +6.1, E +7.1, L +4.8; starting off the floor +0.36 for P)                                                          | —                                                                   | N off-floor −0.03 (P(neg) = 0.65)                                                              |
| Age (negative)           | W −1.0 [−1.7, −0.4]; T −0.9 [−1.4, −0.3]; TR-012 −0.7 [−1.1, −0.2]; TE-013 −0.6 [−1.0, −0.1]                                                                                          | L −0.8 [−1.5, −0.0] (0.95); TR-009 −0.4 (0.95); TE-010 −0.4 (0.93)  | F −0.3 (0.89); N −0.04 (0.88); B −0.2 (0.84); E −0.5 (0.74); R +0.5 (P = 0.70); P +0.01 (0.71) |
| Non-verbal ability       | R +2.3 [+0.6, +4.1]; E +2.1 [+0.5, +3.8]; L +0.8 [+0.2, +1.5]; F +0.7 [+0.3, +1.1]; T +0.8 [+0.3, +1.4]; TR-009 +1.0 [+0.5, +1.4]; TE-010 +0.7 [+0.2, +1.3]; TR-012 +0.7 [+0.2, +1.1] | P off-floor +0.04 [+0.00, +0.08] (0.96)                             | TE-013 +0.3 (0.82); B +0.1 (0.74); N +0.01 (0.66); W +0.1 (0.57)                               |
| Phonological memory      | R +1.8 [+0.3, +3.4]; TR-009 +0.8 [+0.4, +1.2]; TR-012 +0.5 [+0.1, +0.9]; P off-floor +0.05 [+0.01, +0.09]; N off-floor +0.07 [+0.01, +0.13]                                           | TE-013 +0.8 [+0.1, +1.6] (0.96); TE-010 +0.7 [−0.0, +1.5] (0.94)    | B +0.3 (0.84); E −0.6 (P(neg) = 0.69)                                                          |
| Taught receptive vocab   | R +3.6 [+1.8, +5.5]; T +1.4 [+0.7, +2.1]; F +0.6 [+0.1, +1.1]; TE-010 +0.9 [+0.3, +1.4]                                                                                               | TE-013 +0.5 [−0.0, +1.1] (0.93)                                     | E +0.9 (0.82); W −0.1 (P(neg) = 0.57)                                                          |
| Taught expressive vocab  | E +3.0 [+1.0, +5.0]                                                                                                                                                                   | B +0.4 [+0.0, +0.7] (0.96); W +1.2 [−0.0, +2.4] (0.94)              | —                                                                                              |
| Broad vocabulary (R / E) | T (R) +1.1 [+0.5, +1.8]; F (R) +0.6 [+0.2, +1.0]; TR-012 (R) +1.0 [+0.5, +1.4]; TE-013 (E) +1.0 [+0.4, +1.6]                                                                          | E (R) +1.8 [+0.1, +3.6] (0.96); TE-013 (R) +0.6 [+0.0, +1.1] (0.95) | W (R +0.2, E +0.1); B (E +0.1); TR-012 (E +0.1)                                                |
| Letter sounds            | N off-floor +0.14 [+0.08, +0.20]; P off-floor +0.08 [+0.03, +0.14]                                                                                                                    | W +1.0 [−0.1, +2.1] (0.92)                                          | B +0.3 (0.90)                                                                                  |
| Speech production        | —                                                                                                                                                                                     | N off-floor +0.05 [−0.0, +0.1] (0.93)                               | E +1.5 (0.88); TE-013 −0.4 (P(neg) = 0.80); B, L, TE-010 flat                                  |
| Hearing-risk flag (0→1)  | —                                                                                                                                                                                     | —                                                                   | E +1.6 (0.86); TR-009 +0.6 (0.87); TR-012 +0.6 (0.90); others flat                             |

Three patterns stand out. **Age is negative for the reading, grammar and taught-vocabulary gains**, conditional on the starting score: older children gained less on the items scale. **Non-verbal ability tracks gains in every oral-language and vocabulary outcome and in letter sounds, but not in word reading or nonword decoding**, close to the split the `mechanism` family's ability panel found conditional on baseline. **Phonological memory tracks vocabulary gains and both off-floor outcomes**, while taught receptive vocabulary at the start of a period tracks later receptive-vocabulary, grammar, basic-concept and taught-expressive gains, and taught expressive vocabulary tracks later expressive-vocabulary and, at moderate evidence, blending gains. None of these identifies a lever.

## The two companion sets

**Treated-only companions (`gf-101`–`108`)** restrict to periods while children were receiving the intervention, so no treatment effect exists in them; every number is an adjusted association describing progress during intervention.

**Moderation variants (`gf-201`–`211`)** ask whether the on-intervention association varies with the starting point or with general cognitive ability, estimated across all periods including after crossover. The own-starting-point moderation is negative in ten of eleven outcomes and clearly so for receptive vocabulary (−0.13 logits, P = 0.97), receptive grammar (−0.24, P = 0.99), basic concepts (−0.25, P = 0.95) and both taught-vocabulary measures (−0.25, P = 0.97 each): lower starters show a larger on-intervention association. The ability moderation is positive for receptive vocabulary (+0.14, P = 0.96), expressive vocabulary (+0.13, P = 0.95) and receptive grammar (+0.22, P = 0.99) and negative for basic concepts (−0.30, P = 0.98). These are model-dependent adjusted associations partly informed by post-crossover data, released outside the causal gate, and regression to the mean is one obvious source of the starting-point pattern. They must not be reported as "the intervention worked better for" anyone.

## The floored measures

Phonetic spelling and nonword reading use the off-floor rule with the binary off-floor-at-pre indicator as the baseline term. Both are inconclusive (−1 and +2 percentage points, intervals 16 and 21 points wide) and both lean on the treatment prior, so they are released at the qualified tier after the family's treatment-prior sweep confirmed directional stability. That certifies direction, and for these two the direction is inconclusive anyway.

## What changed since the August notes

The treatment marginals match the August figures to within rounding; only the three phonetic-spelling fits (`gf-005`, `105`, `205`) read the quarantined ERB cell, and their reported numbers did not change at the published precision. New since then: the period-1-only sensitivity on every model of record, the per-measure items steps in the association tables (a fixed +5 items was replaced by a step scaled to each test), the guessing-floor pair for blending, and the `gf-012`/`013` broad-vocabulary terms recorded as descriptive associates.

## What these models cannot tell you

**Only the period-1-standardised treatment marginal is given a causal reading**, under the shared-effect model and the available-case assumptions. **The random intercept is not an ability control.** **The moderation variants are not causal.** **Covariate slopes on time-varying baselines blend within- and between-child variation**; the `pooled_levels` family carries the between/within split for the levels question.

## Model inventory

All 33 pass the convergence gate with zero divergences and are publishable. Primaries: `gf-001` (W), `002` (R), `003` (E), `004` (L), `005` (P), `006` (B), `007` (F), `008` (T), `009`/`012` (TR), `010`/`013` (TE), `011` (N). Treated-only companions: `101`–`108`. Moderation variants: `201`–`211`. Guessing-floor companion: `306` (B), released as a pair with `006`.
