# Findings — aligned family (onset-aligned per-protocol comparison)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8). Numbers from the `reporting`-config refit of 2026-07-16 (see [process note](202607161130-full-statistical-refit.md)). Preliminary.

## What these models ask

Instead of comparing arms _as randomised_, these models **align both arms by when they actually started** the intervention (immediate arm t1→t3, waiting-list arm t2→t4) and compare the two cohorts' ~40-week gains in one cross-sectional ANCOVA per child. This "per-protocol" framing answers "what did roughly 40 weeks of programme look like?" — but the cohort comparison is **not randomised**: age-at-onset and cohort/timing differences can confound it. So **no coefficient here is causal** — every one is an association.

## Convergence gate

All 9 (`al-001–008` + the dose-sensitivity variant `al-101`) **passed**.

## Cohort differences (immediate minus waiting-list, onset-aligned)

| Outcome                                 | Difference | 95% range    | P(>0)   | Evidence     |
| --------------------------------------- | ---------- | ------------ | ------- | ------------ |
| Receptive vocab, std (R)                | +2.7 items | −2.9 to +8.2 | 83%     | suggestive   |
| **Letter sounds (L)**                   | +2.2 items | −0.3 to +4.7 | 96%     | moderate     |
| Word reading (W)                        | +2.1 items | −1.1 to +5.3 | 91%     | suggestive   |
| Word reading (W), dose variant (al-101) | +2.1 items | −1.0 to +5.4 | 91%     | suggestive   |
| Phoneme blending (B)                    | +0.3 items | −0.8 to +1.4 | 71%     | inconclusive |
| Phonetic spelling (P)                   | +0.0 items | −0.1 to +0.2 | 70%     | inconclusive |
| Basic concepts (F)                      | −0.6 items | −1.9 to +0.7 | 82% neg | suggestive   |
| Receptive grammar (T)                   | −1.4 items | −3.5 to +0.7 | 91% neg | suggestive   |
| Expressive vocab, std (E)               | −3.1 items | −7.8 to +1.6 | 90% neg | suggestive   |

## The one-paragraph story

Aligning the two cohorts by intervention onset gives a positive-leaning read for **letter sounds** (moderate) and **word reading** (suggestive), broadly consistent with the randomised analyses — but the intervals are wide and several outcomes wander in both directions, exactly as you would expect from a **non-randomised** cohort contrast that timing and age can distort. This family is a sensitivity view, not evidence of record.

## What is causal

**Nothing.** The cohort contrast is confounded by design; dose (a partial collider) enters only the `al-101` sensitivity variant. The child's own baseline is, as everywhere, the most strongly resolved adjusted association (≈99.9% for "higher start → higher finish"). Read the [ITT](202607161800-findings-itt.md) / [gain_factors](202607161800-findings-gain_factors.md) / [did](202607161800-findings-did.md) families for the causal claims; this one only triangulates them.
