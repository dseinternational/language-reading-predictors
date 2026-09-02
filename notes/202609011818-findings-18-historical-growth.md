> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5.1).

<!-- cspell:ignore basnum basspel woco -->

# Findings: the `historical_growth` family — how three groups of children progressed

**Read `findings-00-overview` first.** This note covers the 9 models in the `historical_growth` family. **Nothing here is causal**: there was no intervention and no randomisation. All 9 pass the convergence gate with zero divergences; 6 are publishable and 3 are withheld at the inputs stage (2026-09-01 rebuild).

## The data

**The Reading, Language and Memory cohort only** — a separate, older, observational study (Byrne, MacDonald and Buckley), not the RLI trial. 97 children were followed across five annual waves in three groups: children with Down syndrome, average readers, and reading-matched comparison children. Individual models use 58–77 children. Waves 1 to 3 are the complete-case core; wave 4 for all groups and wave 5 for the Down syndrome group only enter as an available-case extension, so later-wave quantities rest on the children still observed then (16–20 of the 23–24 in the Down syndrome group). BAS matrices was introduced later, so its core covers waves 3 to 4.

## What the model is for

Description: for each measure, how did each group's expected score change from wave to wave? Each supported group-wave cell gets its own mean, child-level variation and overdispersion are estimated per group, and the headline quantities are **within-group changes** across wave intervals, summarised on the children observed at both endpoints. Between-group comparisons are secondary because the reading-matched group was selected on the outcome.

## What was found

Within-group growth, in items, for the Down syndrome group across the six publishable measures:

| Measure (maximum)              | 1→2                   | 2→3                   | 3→4                    | 4→5                   | Cumulative (1→5, or 3→5) |
| ------------------------------ | --------------------- | --------------------- | ---------------------- | --------------------- | ------------------------ |
| BAS word reading (90)          | **+5.4** [+3.3, +7.4] | **+4.4** [+2.2, +6.6] | **+8.6** [+6.1, +11.1] | +2.0 [−0.9, +4.8]     | **+21.2** [+18.6, +23.8] |
| BPVS receptive vocabulary (32) | +0.7 [−0.5, +2.0]     | **+1.4** [+0.1, +2.7] | −0.1 [−1.5, +1.2]      | **+1.5** [+0.0, +3.0] | **+3.5** [+2.1, +4.9]    |
| TROG receptive grammar (20)    | **+1.2** [+0.3, +2.2] | +0.7 [−0.3, +1.7]     | +0.3 [−0.7, +1.3]      | +0.6 [−0.6, +1.8]     | **+2.8** [+1.7, +3.9]    |
| BAS recall of digits (34)      | +0.6 [−0.5, +1.7]     | +1.0 [−0.2, +2.1]     | +1.3 [−0.0, +2.5]      | −0.1 [−1.6, +1.3]     | **+2.8** [+1.5, +4.1]    |
| BAS similarities (21)          | +0.0 [−0.7, +0.7]     | **+1.0** [+0.3, +1.8] | **+1.4** [+0.5, +2.3]  | +0.3 [−0.8, +1.5]     | **+2.9** [+2.0, +3.9]    |
| BAS matrices (28), from wave 3 | —                     | —                     | **+0.8** [+0.0, +1.5]  | +0.2 [−0.6, +1.1]     | **+1.0** [+0.2, +1.8]    |

**Growth is clear for most group-and-interval combinations, but not all.** Across the six publishable models there are 70 within-group interval quantities, and 52 have an 89% interval entirely above zero: 19 of 21 for average readers, 18 of 21 for the reading-matched group, and **15 of 28 for the Down syndrome group**. Growth in that group is clearest on word reading (four of five intervals) and least clear on digit recall (one of five) and grammar (two of five); two medians are slightly negative (vocabulary from wave 3 to 4, digit recall from wave 4 to 5).

The headline for this project's purposes is that **children with Down syndrome showed clear, measurable word-reading growth**: +5.4, +4.4 and +8.6 items over the first three intervals and +21.2 items (89% +18.6 to +23.8) cumulatively across waves 1 to 5, against +37.1 for average readers and +33.2 for the reading-matched group over waves 1 to 4 (the Down syndrome group's common-window growth over waves 1 to 4 is +18.5). The final wave-4-to-5 step is suggestive rather than clear (P = 0.87).

The reference rows the key findings select for the other groups: average readers +16.5 words from wave 1 to 2 (89% +13.8 to +19.1), +1.7 grammar items, +2.2 digits and +1.9 similarities items over the same interval; the reading-matched group +2.2 vocabulary items from wave 2 to 3 and +4.0 matrices items from wave 3 to 4. As a reproduction check on the complete-case core, the largest fitted-minus-observed cell mean gap is 0.4 items (word reading) and 0.0–0.1 elsewhere.

## Why between-group comparisons need care

The reading-matched group was chosen _because_ its measured reading level matched the Down syndrome group's at the start, so selection on a noisy baseline can induce regression to the mean and contribute to apparent growth differences; the groups also differ in age and in everything correlated with membership. The models report the group contrasts over the common window (average readers minus Down syndrome +18.6 words, 89% +15.1 to +22.2; reading-matched minus Down syndrome +14.8, +9.7 to +19.8; vocabulary +2.6 and +3.2 items; grammar +1.2 and +3.2), but this note does not lead with them.

## The three withheld models

`rlm-hg-002` (BAS spelling), `rlm-hg-003` (WORD reading comprehension) and `rlm-hg-008` (BAS number skills) are withheld at the inputs stage because **the maximum score of each instrument is not known**: the 2002 source paper analysed raw scores without stating the test maxima, so the models use the highest observed score as a stand-in — a guess about the instrument. The August 2026 denominator and participant-bootstrap sensitivities supported many of their principal within-group patterns but did not license the registered bounded-count fits (BAS spelling received a strict `no_go` on one near-zero between-group contrast). Clearing them requires the administered test manuals or a decision to adopt a denominator-free analysis with different estimands. **They are withheld for want of a measurement fact, not because the results were unfavourable.**

## What these models cannot tell you

**Nothing causal.** **Group differences are not effects of group membership.** **These children are not the trial children.** **Growth here is not attributable to anything measured.**

## Model inventory

Six of nine are publishable: `rlm-hg-001` (BAS word reading), `004` (BPVS receptive vocabulary), `005` (TROG receptive grammar), `006` (BAS recall of digits), `007` (BAS similarities), `009` (BAS matrices, waves 3–4 core plus the Down syndrome wave-5 extension). Withheld at the inputs stage: `002`, `003`, `008`. All nine pass the convergence gate with zero divergences, and all nine now record a data checksum (the provenance gap the 2026-08-27 pass predicted for this family was closed by the loader fix).
