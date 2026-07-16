# Findings — gain_factors family (ANCOVA on each period's gain; a second causal read of the effect)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8). Numbers from the `reporting`-config refit of 2026-07-16 (see [process note](202607161130-full-statistical-refit.md)). Preliminary.

## What these models ask

These models look at each child's **post-period score given their pre-period score** (an ANCOVA — analysis of covariance) and stack every study period together, with a child random intercept. The one **causal** number is the **on-intervention effect during the randomised first period** (period 1, when the immediate arm is treated and the waiting-list arm is still an untreated control). It is reported as an **items-scale risk difference**, and it is a genuinely independent second look at the [ITT](202607161800-findings-itt.md) effect using a different model structure. Every other coefficient (own baseline, age, ability, upstream skills) is an **adjusted association** describing _who_ progresses — not a lever you could pull.

There are 11 primary models (`gf-001–011`, one per outcome) and 8 "treated-only" companions (`gf-101–108`) that look only at children while on the programme and therefore estimate **no** treatment effect — they are association-only checks.

## Convergence gate

All 19 gain_factors models **passed**.

## Headline on-intervention effects (period-1, causal), gf-001–011

| Outcome                          | On-intervention effect | 95% range    | P(>0)   | Evidence                                          |
| -------------------------------- | ---------------------- | ------------ | ------- | ------------------------------------------------- |
| **Letter sounds (L)**            | **+3.3 items**         | +1.2 to +5.4 | 99%     | **very strong**                                   |
| **Word reading (W)**             | **+2.6 items**         | +0.5 to +4.7 | 99%     | **very strong**                                   |
| Basic concepts (F)               | +1.1 items             | −0.1 to +2.2 | 86%     | suggestive                                        |
| Taught expressive vocab (TE)     | +1.2 items             | −0.3 to +2.6 | 82%     | suggestive                                        |
| Taught receptive vocab (TR)      | +1.1 items             | −0.4 to +2.5 | 63%     | inconclusive                                      |
| Expressive vocab, std (E)        | +1.1 items             | −2.8 to +5.0 | 58%     | inconclusive                                      |
| Phoneme blending (B)             | +0.8 items             | −0.1 to +1.7 | 90%     | suggestive                                        |
| Receptive grammar (T)            | +0.8 items             | −0.8 to +2.5 | 66%     | inconclusive                                      |
| Receptive vocab, std (R)         | −1.4 items             | −6.2 to +3.3 | 83% neg | suggestive (toward harm, but ROPE 52% negligible) |
| Nonword reading (N), off-floor   | +2 pp                  | −11 to +15   | 62%     | inconclusive                                      |
| Phonetic spelling (P), off-floor | −3 pp                  | −15 to +8    | 62% neg | inconclusive                                      |

## The one-paragraph story

This model, built quite differently from the ITT model, **reproduces the ITT headline**: **very strong** evidence of benefit on **letter sounds** (+3.3 items) and **word reading** (+2.6 items), and the same fade-out for broad standardised vocabulary. Getting the same answer from a period-stacked ANCOVA as from the simple randomised comparison is the kind of triangulation that makes the core finding credible.

## What is causal

Only the **period-1 on-intervention marginal** is causal (it rests on the randomised first period). Own-baseline, age, and upstream-skill coefficients are adjusted associations — consistently, the child's **own starting score** is the most strongly resolved of these (≈99.9% for "higher start → higher finish"), which is expected and not a treatment effect. The `gf-101–108` treated-only companions estimate no effect at all and exist purely as association checks.
