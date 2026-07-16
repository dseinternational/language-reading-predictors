# Findings — level_factors family (score levels at each wave)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8). Numbers from the `reporting`-config refit of 2026-07-16 (see [process note](202607161130-full-statistical-refit.md)). Preliminary.

## What these models ask

The companion "levels" view of the [gain_factors](202607161800-findings-gain_factors.md) family: instead of modelling each period's _gain_, it models the **score itself at each timepoint**, with group×time and ability×time terms. Only the **t2 group contrast** (`b_grp_time[1]`) is a **clean randomised effect** — at t2 the immediate arm has been treated and the waiting-list arm has not. Later timepoints are **after the waiting-list children have crossed over**, so those group differences are associations, not randomised effects.

## Convergence gate

All 11 **passed**.

## Headline randomised t2 effects

| Outcome                          | t2 effect      | 95% range    | P(>0)   | Evidence                  |
| -------------------------------- | -------------- | ------------ | ------- | ------------------------- |
| **Letter sounds (L)**            | **+2.5 items** | −0.4 to +5.4 | 96%     | moderate                  |
| Word reading (W)                 | +1.4 items     | −1.8 to +4.6 | 81%     | suggestive                |
| Phoneme blending (B)             | +0.4 items     | −0.6 to +1.4 | 80%     | suggestive                |
| Taught receptive vocab (TR)      | +0.4 items     | −1.3 to +2.0 | 66%     | inconclusive              |
| Taught expressive vocab (TE)     | +0.3 items     | −1.4 to +2.0 | 64%     | inconclusive              |
| Nonword reading (N), off-floor   | +1 pp          | −13 to +15   | 53%     | inconclusive              |
| Phonetic spelling (P), off-floor | −1 pp          | −11 to +9    | 57% neg | inconclusive              |
| **Receptive vocab, std (R)**     | **−3.8 items** | −9.2 to +1.5 | 92% neg | moderate (toward lower)   |
| Expressive vocab, std (E)        | −2.4 items     | −7.4 to +2.8 | 82% neg | suggestive (toward lower) |
| Basic concepts (F)               | —              | —            | —       | (level model; see report) |
| Receptive grammar (T)            | —              | —            | —       | (level model; see report) |

## The one-paragraph story

A **levels** model is a blunter instrument than a gain model — it does not use each child as their own baseline, so its confidence intervals are wider and its t2 signal is weaker than the [ITT](202607161800-findings-itt.md) or [gain_factors](202607161800-findings-gain_factors.md) views. Even so, **letter sounds** shows moderate evidence of a positive randomised t2 effect (+2.5 items), consistent with the sharper analyses. The negative-leaning receptive/expressive standardised-vocabulary levels at t2 are noisy and should be read alongside the ITT verdict that those effects are **inconclusive and probably negligible** — a levels model is the least reliable place to read a small vocabulary effect.

## What is causal

**Only the t2 contrast.** Everything at t3/t4 is post-crossover association. Unlike the gain models, this family carries **no** other-skill adjusters on purpose: conditioning a levels model on another skill's contemporaneous level would condition on a post-treatment mediator and corrupt the group×time reading (the reasoning recorded in #247).
