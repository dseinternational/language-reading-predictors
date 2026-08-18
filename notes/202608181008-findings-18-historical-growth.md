> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

<!-- cspell:ignore basnum basspel woco -->

# Findings: the `historical_growth` family — how three groups of children progressed

**Read `findings-00-overview` first.** This note covers the 9 models in the `historical_growth` family. **Nothing here is causal**, and the reason is simple: there was no intervention and no randomisation.

## The data

**The Reading, Language and Memory cohort only** — a separate, older, observational study (Byrne, MacDonald and Buckley), not the RLI trial. This is the most common misreading of this family, so it is worth stating twice: **these are different children from the trial**.

97 children were followed across **five annual assessment waves** in three groups: children with Down syndrome, average readers, and reading-matched comparison children. Individual models use 58–77 children depending on which measures a child has. Data are stacked by wave, with one row per child per wave.

The wave coverage differs by measure. Most models cover waves 1 to 4 with a fifth wave for the Down syndrome group only — that group was followed longer. One model (BAS matrices) covers only waves 3 to 4, because the measure was introduced later.

## What the model is for

The purpose is **description**: for each measure, how did each group's expected score change from wave to wave?

Each supported group-wave combination gets its own mean, so the model does not force a straight line through time — it lets each group's trajectory take whatever shape the data show. Child-level variation and overdispersion are estimated separately per group, because the groups differ in how spread out they are.

The headline quantities are **within-group changes** across wave intervals. Between-group comparisons are secondary, and for a good reason given below.

## What was found

| Measure                   | Group           | Interval | Change                       |
| ------------------------- | --------------- | -------- | ---------------------------- |
| BAS word reading          | Down syndrome   | wave 3→4 | **+8.5 items** [+5.7, +11.3] |
| BAS recall of digits      | average readers | wave 1→4 | **+5.0 items** [+3.6, +6.4]  |
| BPVS receptive vocabulary | average readers | wave 1→4 | **+4.6 items** [+3.2, +5.9]  |
| BAS matrices              | reading-matched | wave 3→4 | **+4.0 items** [+2.4, +5.6]  |
| TROG receptive grammar    | average readers | wave 1→4 | **+3.4 items** [+2.6, +4.2]  |
| BAS similarities          | Down syndrome   | wave 1→5 | **+2.9 items** [+1.9, +3.9]  |

**Growth is clear for most group-and-interval combinations, but not all.** Across the six publishable models there are 88 within-group growth quantities, and 58 of them have an 89% interval entirely above zero. The remaining 30 include zero, and they are not evenly spread: 19 of 21 intervals clear zero for average readers and 16 of 21 for the reading-matched group, but only **12 of 28 for the Down syndrome group**. Growth in that group is clearest on word reading and least clear on vocabulary and grammar in the later waves, where several intervals include zero and one median is slightly negative.

The headline result for this project's purposes is that children with Down syndrome showed clear, measurable word-reading growth throughout: +5.4 items from wave 1 to 2, +4.4 from wave 2 to 3 and +8.5 from wave 3 to 4, all with intervals well above zero, giving **+21.3 items across waves 1 to 5** (89% +18.4 to +24.1). Only the final wave-4-to-5 step falls short of that, at +2.0 (89% −1.1 to +5.1) — suggestive rather than clear, with a posterior probability of 0.84 that it is positive.

## Why between-group comparisons need care

It is tempting to line the three groups up and compare their rates. The design makes that treacherous.

**The reading-matched group was selected on the outcome.** These children were chosen _because_ their reading level matched the Down syndrome group's at the start. Selecting a group on a measured value guarantees regression to the mean on that measure — they will drift back towards their own population average regardless of anything else. Their apparent reading growth is partly a selection artefact.

**The groups differ in age and in everything correlated with group membership.** Nothing was randomised.

So within-group growth is the defensible quantity, and the models are built to report it. Cross-group contrasts are secondary summaries, and this note does not lead with them.

## The three withheld models

**`rlm-hg-002` (BAS spelling), `rlm-hg-003` (WORD reading comprehension) and `rlm-hg-008` (BAS number skills) are withheld** at the inputs stage.

The reason is a measurement fact, not a statistical one. These models treat scores as "so many correct out of a maximum", but for these three measures **the maximum is not known**. The 2002 source paper analysed raw scores without stating the test maxima, so the models are currently using the highest score anyone actually achieved as a stand-in — a guess about the instrument, not a property of it.

A dedicated sensitivity analysis stress-tested this in August 2026: it refitted with denominators two and four times larger and with a likelihood requiring no maximum at all. **The growth directions were robust to all of it** — every median kept its direction and all intervals overlapped. The project still chose to withhold the results, on the grounds that "robust to our guess" is not the same as "we know the scale", and a bounded-count model with an invented bound cannot be published as a model of record.

That decision is recorded in `notes/202608161900-byrne-denominator-likelihood-sensitivity.md`. Clearing these three requires the administered test manuals or records. **These are withheld for want of a measurement fact, not because the results were unfavourable** — the sensitivity work suggests they would look much like their published siblings.

## What these models cannot tell you

**Nothing causal.** No intervention, no randomisation, no treatment.

**Group differences are not effects of group membership.**

**These children are not the trial children.** Results here neither support nor contradict the trial findings; they describe a different cohort in a different era.

**Growth here is not attributable to anything measured.** It is what happened, not why.

## Model inventory

Six of nine pass and are publishable: `rlm-hg-001` (BAS word reading), `004` (BPVS receptive vocabulary), `005` (TROG receptive grammar), `006` (BAS recall of digits), `007` (BAS similarities), `009` (BAS matrices, waves 3–4). Withheld at the inputs stage: `002` (BAS spelling), `003` (WORD reading comprehension), `008` (BAS number skills). All nine pass the convergence gate with zero divergences — the withholding is about measurement provenance, not computation.
