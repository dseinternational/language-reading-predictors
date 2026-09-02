> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5.1).

# Findings: the `dose_response` family — does more intervention mean more progress?

**Read `findings-00-overview` first.** This note covers the 6 models in the `dose_response` family. **Nothing here is causal**, and the family was substantially re-specified by the #587 review, so the numbers are not comparable with the August series. All 6 pass the convergence gate with zero divergences and all are publishable (2026-09-01 rebuild).

## The data

**RLI trial only.** All transitions are stacked — 53–54 children contributing 156–160 rows — with the number of intervention sessions a child actually attended during each period. The dose headline is averaged over the **on-intervention rows only** (128–131 of them); the wait-list arm's zero-dose period-1 rows enter through a separate "on the intervention at all" indicator.

## What the model is for

The treatment families ask whether being _assigned_ to the intervention helped. This family reports the **intensive margin**: among children who were receiving the intervention, did attending more sessions go with more progress? Since #587 sessions are centred and standardised over the on-intervention rows, a separate `theta_treated` indicator carries the extensive margin (whether a child was being taught at all), and the exposure is split Mundlak-style into each child's study-average attendance and their within-child deviation, because a single slope over a child random intercept blends the two. The headline is the outcome change across an interquartile step in sessions within a period.

## Why this cannot be causal

**Children were randomised to the intervention; they were not randomised to attend more of it.** Attendance depends on health, family circumstances, engagement and how the sessions were going, and the study's causal diagram has age, latent general ability and assigned group all pointing into it. A dose–outcome association is therefore an observational association, and attendance is a collider on some paths. The one randomised quantity in these fits is `theta_treated` read in period 1, where every immediate-arm child attended and every waiting-list child attended none.

## What was found

| Outcome                             | Interquartile step (sessions) | Dose association (items) | P(>0) | Evidence     | On-intervention indicator (logit) |
| ----------------------------------- | ----------------------------- | ------------------------ | ----- | ------------ | --------------------------------- |
| Word reading (`dose-077`)           | about 24                      | **+1.9** [+0.0, +4.0]    | 0.95  | moderate     | +0.25 [+0.02, +0.49]              |
| Word reading, ability-adjusted      | about 24                      | +1.9 [+0.1, +4.0]        | 0.95  | moderate     | +0.25 [+0.02, +0.50]              |
| Word reading, pooled slope          | about 24                      | +1.9 [+0.0, +3.9]        | 0.95  | moderate     | +0.24 [+0.02, +0.47]              |
| Letter-sound knowledge (`dose-083`) | about 25                      | **+1.6** [+0.1, +2.9]    | 0.96  | moderate     | +0.44 [+0.14, +0.73]              |
| Phoneme blending, ordinary link     | about 25                      | +0.0 [−0.7, +0.7]        | 0.51  | inconclusive | +0.26 [−0.10, +0.61]              |
| Phoneme blending, guessing floor    | about 25                      | −0.0 [−0.7, +0.6]        | 0.47  | inconclusive | +0.23 [−0.26, +0.74]              |

**Periods with more attended sessions were associated with better word reading and letter-sound knowledge, at moderate evidence, and with no resolved blending difference.** The word-reading gradient is unchanged by adding the baseline-skill cluster and by pooling the slope across periods (the nested predictive comparison between the period-varying and pooled versions is inconclusive), so it is not an artefact of either choice. The on-intervention indicator, which is where the randomised evidence sits, is positive for word reading and letter sounds and unresolved for blending, consistent with the treatment families.

**Blending is a released pair.** `dose-084` and `dose-384` (the three-choice guessing-floor link) agree that the dose association is zero, and both are publishable together. The key-findings box of each fit now reports them as a pair, quoting the two links' interquartile-step dose associations side by side so that neither is read as the answer on its own. Earlier copies of both boxes carried a caveat sentence saying the guessing-floor companion "has not been built for this family", a hard-coded remnant of #587 finding 6 that predated the pair; the builder was corrected in PR #655 and the key findings and reports of both fits were regenerated.

## How to read this honestly

**Periods with more recorded sessions went with better outcomes, and the design cannot say how much of that is the sessions.** Children who were doing well may have attended more, and children with complicating circumstances less. The magnitudes are compatible with the randomised estimates without adding to them: the randomised word-reading effect is about +2.4 items and an interquartile step of attendance is associated with about +1.9. That consistency is what you would expect whether or not the dose relationship is causal.

The `mechanism` family's sessions-to-reading curve (`mech-191`) is the natural comparison and now reads +0.2 words across its interquartile range once rows without observed attendance are excluded; the two fits define their exposure populations differently (all treated rows here; positive-attendance rows there) and neither supports a dose recommendation.

## What these models cannot tell you

**They cannot support "more sessions would produce more progress"**, and **they cannot be used to recommend a dose.** **The between/within split is descriptive**: the child-average and within-child parts of attendance are both associations.

## Model inventory

All 6 pass the convergence gate with zero divergences and are publishable: `dose-077` (W, period-resolved), `083` (L), `084` (B), `177` (W, ability-adjusted sensitivity), `277` (W, pooled comparator), `384` (B, guessing-floor link companion, released as a pair with `084`).
