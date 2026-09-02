> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5.1).

# Findings: the `level_factors` family — scores at each timepoint

**Read `findings-00-overview` first.** This note covers the 23 models in the `level_factors` family: the eleven four-wave primaries, the guessing-floor blending companion and the eleven randomised-window comparators registered by the #584 review. All 23 pass the convergence gate with zero divergences and all are publishable (2026-09-01 rebuild).

## The data

**RLI trial only.** Each child contributes one row per timepoint: 53–54 children and 207–215 rows over four waves in the primaries, 103–108 rows in the two-wave comparators. Unlike `itt` and `gain_factors`, these models do **not** use the child's own baseline as a covariate.

## What the model is for

Where `gain_factors` asks whether the post-period score is higher given the pre-period score, this family asks **how high the score was at each timepoint, and whether the arms differed**. The arm-by-time vector is centred on the timepoint-1 gap: `arm_gap_t1` is the covariate-adjusted arm difference before anyone was treated, and `d_grp_time[t]` the change in that gap at each later wave. **Only the t2 change** — a difference-in-differences of adjusted levels — is the randomised treated-versus-untreated effect. The t3 and t4 changes are also identified by the original randomisation, but as early-start-versus-delayed-start schedule contrasts (role `regime`), with no mechanistic reading. Ability enters as per-timepoint coefficients plus one time-invariant group-by-ability term; hearing, speech and phonological memory are adjusters; no other skill measure enters, because a same-wave skill level is a post-treatment mediator of the group-by-time effect.

Since #584 the items card is the **arm-free standardised average marginal effect**: each fitted t2 child evaluated at their own arm-free profile, with the moderation increment held at centred ability. The dispersion prior sits on the inverse square root of the concentration and the child-intercept scale is wider than in the gain family, both calibrated for a levels model that must carry the whole between-child spread.

## What was found

The randomised t2 change in the adjusted arm gap, in items, positive favouring the immediate arm, with the two-wave comparator beside it:

| Measure                           | Four-wave (model of record) | P(>0) | Evidence     | Two-wave comparator (`lf-2xx`) |
| --------------------------------- | --------------------------- | ----- | ------------ | ------------------------------ |
| Letter-sound knowledge (L)        | **+2.9** [+0.8, +4.9]       | 0.99  | strong       | +2.9 [+0.9, +4.9]              |
| Word reading (W)                  | **+2.3** [+0.3, +4.4]       | 0.97  | moderate     | +2.5 [+0.5, +4.6]              |
| Taught expressive vocabulary (TE) | **+1.4** [+0.1, +2.8]       | 0.96  | moderate     | +1.3 [−0.1, +2.7]              |
| Taught receptive vocabulary (TR)  | +1.2 [−0.1, +2.5]           | 0.93  | moderate     | +1.3 [−0.1, +2.6]              |
| Basic concept knowledge (F)       | +0.8 [−0.3, +1.8]           | 0.89  | suggestive   | +0.7 [−0.4, +1.8]              |
| Phoneme blending (B), ordinary    | +0.6 [−0.2, +1.4]           | 0.90  | suggestive   | +0.8 [−0.0, +1.6]              |
| Phoneme blending (B), floor link  | +0.5 [−0.1, +1.0]           | 0.92  | moderate     | —                              |
| Receptive grammar (T)             | +0.6 [−1.0, +2.3]           | 0.73  | inconclusive | +0.6 [−1.0, +2.2]              |
| Receptive vocabulary (R)          | +0.3 [−3.5, +4.3]           | 0.56  | inconclusive | +0.5 [−4.0, +5.2]              |
| Expressive vocabulary (E)         | +0.2 [−3.1, +3.5]           | 0.53  | inconclusive | −0.1 [−3.4, +3.4]              |
| Nonword reading (N), off-floor    | +3 pp [−8, +14]             | 0.67  | inconclusive | +4 pp [−8, +16]                |
| Phonetic spelling (P), off-floor  | +0 pp [−7, +8]              | 0.54  | inconclusive | +1 pp [−9, +11]                |

**The family agrees with the rest of the suite.** For word reading the four designs now read +2.4 (`itt`), +2.6 (`gain_factors`), +2.2 (`did`) and +2.3 here; for letter sounds +3.5, +3.3, +3.5 and +2.9. Taught expressive vocabulary is the one level-family estimate whose interval clears zero only since the arm-free standardisation (+1.4, 89% +0.1 to +2.8).

**The two-wave comparators say the longitudinal working model costs little.** Every comparator sits within a few tenths of an item of its four-wave estimate; the largest gap is expressive vocabulary (−0.1 against +0.2, both null), and taught expressive vocabulary is the one outcome whose four-wave interval clears zero (lower bound +0.1) while its two-wave comparator does not (−0.1). No comparator overturns a four-wave conclusion, so the post-crossover waves are not manufacturing the t2 answer through the shared balance term, child intercept or dispersion.

**Balance and the vocabulary story.** The pre-treatment gap `arm_gap_t1` is a balance quantity, never an effect. On the logit scale the immediate arm started slightly lower on receptive vocabulary (−0.17, 89% −0.31 to −0.02, moderate), expressive vocabulary (−0.13, suggestive), taught expressive (−0.21, suggestive) and taught receptive (−0.16) vocabulary, and at parity on word reading (+0.03) and letter sounds (+0.07). The full t2 arm gap the family used to report (`b_grp_time[1]`) still carries those starting gaps — receptive vocabulary −0.15, expressive −0.12 — while the randomised change over the window is +0.01 and +0.01. As August established arithmetically, the negative vocabulary levels are the gap the arms began with, not movement.

**The schedule contrasts.** The t3 and t4 changes compare an earlier with a later start of the same teaching. Three clear the ladder's moderate rung: word reading's t3 change (+0.25 logits, P = 0.92), blending's t4 change (+0.36, P = 0.93) and, more strikingly, receptive vocabulary's t4 change (+0.20, 89% +0.07 to +0.33, very strong) — by timepoint 4 the immediate arm's receptive-vocabulary standing had improved relative to the waiting arm's, after starting behind. Duration, carryover, maturation and ceilings are inseparable inside such a contrast, so it is reported and not interpreted. The time-invariant group-by-ability term is moderate for receptive vocabulary (−0.14) and receptive grammar (+0.18), suggestive for phonetic spelling (+0.21) and basic concepts (+0.20), and inconclusive elsewhere; it is estimated mostly off the non-randomised waves.

**Prior sensitivity.** The t2 term is likelihood-dominated in nine primaries. Phonetic spelling's focal term carries the "strong prior / weak likelihood" flag and is released at the qualified prior-informed tier after the family's treatment-prior sweep; nonword reading's is flagged for potential prior–data conflict and released; both are inconclusive regardless.

## What changed since the August notes

The four-wave cards match the 2026-08-26 movement record to within rounding except where a fit reads the quarantined ERB cell: receptive vocabulary moved from +0.40 to +0.35 items and expressive vocabulary from +0.19 to +0.17, and the other three ERB-affected primaries by less than 0.01. New at reporting tier since the August findings series: the arm-free standardised card (which, per that record, moved taught expressive vocabulary to +1.4 and receptive vocabulary by +0.17 items, and nothing else by more than 0.05), the eleven two-wave comparators, the blending link pair, the re-calibrated nuisance priors, and the `regime` labelling of the t3/t4 changes.

## What these models cannot tell you

**Only the t2 change in the adjusted gap is randomised treated-versus-untreated.** The t3/t4 changes are randomised schedule contrasts without a mechanism. **Ability and interaction terms are adjusted associations.** **These are levels, not changes** — a child can be lower in level and still have gained more. **Do not treat this family as a refutation of the others**; it is the same children with less statistical leverage.

## Model inventory

All 23 pass the convergence gate with zero divergences and are publishable. Four-wave primaries: `lf-001` (W), `002` (R), `003` (E), `004` (L), `005` (P), `006` (B), `007` (F), `008` (T), `009` (TR), `010` (TE), `011` (N). Guessing-floor companion: `106` (B), released as a pair with `006`. Two-wave comparators: `201`–`211`, one per primary; the blending comparator `206` carries the ordinary link only.
