> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5.1).

# Findings: the `concurrent` family — which skills go together at each timepoint

**Read `findings-00-overview` first.** This note covers the 14 models in the `concurrent` family. **Nothing here is causal**, and the reason is unusually stark: everything is measured at the same moment. All 14 pass the convergence gate with zero divergences and all are publishable (2026-09-01 rebuild). Ten of the twelve RLI fits (`ca-001`–`009` and `ca-307`) read the quarantined ERB cell, so their coefficients moved slightly from the August values; the blending pair `ca-007`/`ca-307` is complete.

## The data

**Both studies.** Twelve models use the RLI trial (51–54 children per wave) and two the historical Reading, Language and Memory cohort (96, 88, 78 and 61 children at waves 1–4 for word reading; 88, 88, 78, 61 for vocabulary). Each model is fitted **separately at each wave**, one row per child, everything measured at the same moment. The historical wave-4 results are an available-case, attrition-sensitive extension beyond the source paper's audited waves.

## What the model is for

At a single moment, which skills go with which? The focal outcome is regressed on the standardised same-wave levels of the other core skills, plus age, a group nuisance term and the background covariates (ability, hearing, speech, phonological memory, with missingness indicators). Effects are reported per standard deviation of the predictor, translated into outcome items, with a single-skill comparator beside every adjusted coefficient.

## The interpretive problem, stated plainly

Simultaneous measurement removes even the weak temporal ordering the other families have, and conditioning on other same-wave skills introduces a second hazard: holding fixed something on the path between two variables, or a common effect of both, can shrink a real relationship or manufacture a spurious one. These are **conditional** associations that change with the adjustment set, as the family itself demonstrates.

## What was found

The strongest mutually adjusted same-wave associate of each outcome, wave by wave (items per +1 SD of the predictor):

| Outcome                   | t1                                  | t2                       | t3                        | t4                        |
| ------------------------- | ----------------------------------- | ------------------------ | ------------------------- | ------------------------- |
| Word reading (W)          | L +2.5 [+0.8, +4.6]                 | **L +9.0 [+6.3, +11.7]** | L +4.2 [+1.4, +7.2]       | L +5.8 [+3.0, +8.6]       |
| Letter sounds (L)         | W +2.9 [+1.3, +4.4]                 | **W +4.2 [+3.3, +5.1]**  | W +2.5 [+1.2, +3.6]       | W +1.6 [+0.4, +2.6]       |
| Taught receptive (TR)     | R +0.9 (P = 0.88)                   | TE +1.0 [+0.0, +1.9]     | TE +1.3 [+0.2, +2.2]      | TE +0.9 (P = 0.93)        |
| Taught expressive (TE)    | E +1.3 [+0.3, +2.3]                 | E +1.6 [+0.5, +2.7]      | **E +2.4 [+1.2, +3.5]**   | E +1.2 (P = 0.94)         |
| Receptive vocabulary (R)  | E +5.3 [+1.2, +9.7]                 | E +3.3 (P = 0.88)        | **W +5.8 [+2.0, +9.9]**   | E +3.3 (P = 0.87)         |
| Expressive vocabulary (E) | R +4.1 [+0.6, +7.9]                 | TE +4.4 [+0.9, +8.1]     | **TE +6.4 [+2.7, +10.5]** | L +3.4 (P = 0.93)         |
| Phoneme blending (B)      | W +0.6 [+0.1, +1.1]                 | TR +0.7 [+0.1, +1.2]     | TE −0.7 [−1.3, −0.2]      | **W +0.8 [+0.4, +1.2]**   |
| Basic concepts (F)        | **TR +1.5 [+0.7, +2.3]**            | E +0.9 (P = 0.91)        | R +0.9 [+0.2, +1.6]       | W +0.8 [+0.0, +1.5]       |
| Receptive grammar (T)     | R +1.7 [+0.2, +3.2]                 | R +1.3 [+0.0, +2.6]      | **TR +2.0 [+0.8, +3.1]**  | TE +1.8 [+0.5, +3.2]      |
| BAS word reading (Byrne)  | **similarities +7.5 [+3.8, +11.3]** | similarities +8.7        | similarities +8.3         | digits +5.5 [+0.5, +10.2] |
| BPVS vocabulary (Byrne)   | grammar +1.2 [+0.3, +2.2]           | grammar +1.1             | grammar +1.4 [+0.3, +2.5] | similarities +1.3         |

**Letter sounds and word reading are each other's strongest correlate at every wave**, the clearest illustration of why these coefficients cannot be read directionally. **Word reading is the strongest correlate of receptive vocabulary at wave 3** (+5.8 items), which shows what these coefficients are: reverse direction, general progress and common causes remain compatible with it. **The vocabulary measures cluster**: taught and broad vocabulary track each other across waves, and grammar tracks vocabulary rather than the code skills. In the historical cohort, verbal reasoning (BAS similarities) is the strongest same-wave correlate of word reading at three of four waves — consistent with the general-ability reading of these associations, since it is about as close to a general-ability marker as that battery has.

**Adjustment changes the answer.** Three models estimate the letter-sound-to-word-reading relationship at timepoint 2 with different covariates:

| Model    | Adjustment                                     | Letter sounds, per +1 SD  |
| -------- | ---------------------------------------------- | ------------------------- |
| `ca-010` | Letter sounds + minimal background set         | +11.0 items [+8.2, +13.6] |
| `ca-001` | Six-skill + broader background set             | +9.0 items [+6.3, +11.7]  |
| `ca-011` | Letter sounds + nonword decoding + minimal set | +7.2 items [+4.4, +10.2]  |

Adding same-wave nonword decoding to the minimal specification reduces the letter-sound estimate from +11.0 to +7.2 items while decoding itself carries +2.9 (+1.4 to +4.6); the six-skill model is a different adjustment set, not the middle step. No single number is _the_ answer.

**Phoneme blending under the two links.** `ca-307` re-fits the blending outcome holding its expected score at or above chance: the adjusted associates shrink by a fifth to a third (word reading at t4 +0.5 against +0.8; the t3 letter-sound association +0.5 against +0.7) and keep their direction. Released as a pair.

## What these models cannot tell you

**No direction at all.** **No causal claim, and no mechanism** — see `mechanism` and `lcsm` for designs with temporal ordering and `itt` for the randomised question. **The coefficients depend on the adjustment set.** **These are not predictions**; an association at one wave does not forecast a later outcome.

## Model inventory

All 14 pass the convergence gate with zero divergences and are publishable. Trial cohort: `ca-001` (word reading), `002` (letter sounds), `003`/`004` (taught vocabulary), `005`/`006` (broad vocabulary), `007`/`307` (blending, ordinary and guessing-floor links), `008` (basic concepts), `009` (grammar), `010`/`011` (letter-sound to word-reading adjustment variants). Historical cohort: `rlm-ca-001` (BAS word reading), `rlm-ca-002` (BPVS receptive vocabulary); neither records a data checksum.
