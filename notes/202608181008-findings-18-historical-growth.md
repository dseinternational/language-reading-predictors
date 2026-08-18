> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).
>
> Substantially corrected by a LLM-based AI tool (Codex/GPT-5).

<!-- cspell:ignore basnum basspel woco -->

# Findings: the `historical_growth` family — how three groups of children progressed

**Read `findings-00-overview` first.** This note covers the 9 models in the `historical_growth` family. **Nothing here is causal**, and the reason is simple: there was no intervention and no randomisation.

## The data

**The Reading, Language and Memory cohort only** — a separate, older, observational study (Byrne, MacDonald and Buckley), not the RLI trial. This is the most common misreading of this family, so it is worth stating twice: **these are different children from the trial**.

97 children were followed across **five annual assessment waves** in three groups: children with Down syndrome, average readers, and reading-matched comparison children. Individual models use 58–77 children depending on which measures a child has. Data are stacked by wave, with one row per child per wave.

The wave coverage differs by measure. Most models treat waves 1 to 3 as the complete-case core, with wave 4 for all groups and a fifth wave for the Down syndrome group only — that group was followed longer — entering as an available-case extension, so the later-wave quantities (including the Down-syndrome word-reading headline for waves 3 to 4 and the cumulative total) rest on the children still observed then, roughly 16–19 of the 23. BAS matrices was introduced later, so its common three-group core covers waves 3 to 4; the model also reports a Down-syndrome-only wave-5 extension.

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

**Growth is clear for most group-and-interval combinations, but not all.** Across the six publishable models there are 70 within-group growth quantities, and 47 of them have an 89% interval entirely above zero. The remaining 23 include zero, and they are not evenly spread: 19 of 21 intervals clear zero for average readers and 16 of 21 for the reading-matched group, but only **12 of 28 for the Down syndrome group**. Growth in that group is clearest on word reading (one interval of five includes zero) and least clear on vocabulary and digit recall (four of five intervals include zero for each, vocabulary's from the first interval onwards) and grammar (three of five), with two medians slightly negative — vocabulary from wave 3 to 4 and digit recall from wave 4 to 5.

The headline result for this project's purposes is that children with Down syndrome showed clear, measurable word-reading growth through wave 4: +5.4 items from wave 1 to 2, +4.4 from wave 2 to 3 and +8.5 from wave 3 to 4, all with intervals well above zero. Cumulative growth across waves 1 to 5 was **+21.3 items** (89% +18.4 to +24.1). The final wave-4-to-5 step alone was +2.0 (89% −1.1 to +5.1) — suggestive rather than clear, with a posterior probability of 0.84 that it was positive.

## Why between-group comparisons need care

It is tempting to line the three groups up and compare their rates. The design makes that treacherous.

**The reading-matched group was selected on the outcome.** These children were chosen _because_ their measured reading level matched the Down syndrome group's at the start. Selection on a noisy baseline can induce regression to the mean and may therefore contribute to an apparent difference in growth. It does not guarantee the sign or size of that contribution, and these descriptive group-wave models do not identify how much of the observed pattern is selection rather than development.

**The groups differ in age and in everything correlated with group membership.** Nothing was randomised.

So within-group growth is the defensible quantity, and the models are built to report it. Cross-group contrasts are secondary summaries, and this note does not lead with them.

## The three withheld models

**`rlm-hg-002` (BAS spelling), `rlm-hg-003` (WORD reading comprehension) and `rlm-hg-008` (BAS number skills) are withheld** at the inputs stage.

The reason is a measurement fact, not a statistical one. These models treat scores as "so many correct out of a maximum", but for these three measures **the maximum is not known**. The 2002 source paper analysed raw scores without stating the test maxima, so the models are currently using the highest score anyone actually achieved as a stand-in — a guess about the instrument, not a property of it.

A dedicated sensitivity analysis stress-tested these three historical-growth models in August 2026: it refitted with denominators two and four times larger and with a likelihood requiring no maximum at all. Across those four likelihood variants every reported growth median kept its direction and every set of 89% intervals overlapped. A later denominator-free participant Bayesian bootstrap appended a fifth method: WORD comprehension and BAS number skills passed the pre-specified five-method rule, while BAS spelling received a strict `no_go` because one near-zero **between-group** contrast changed median sign (+0.09 to −0.19 items). Every interval for that contrast spanned zero, so this is not a substantively reversed within-group developmental pattern, but it prevents a blanket claim that every quantity was direction-stable under all five methods.

Those decisions are recorded in `notes/202608161900-byrne-denominator-likelihood-sensitivity.md` and `notes/202608161945-byrne-participant-bayesian-bootstrap.md`. Neither analysis identifies the administered maxima or makes the registered bounded-count fits publishable. Clearing these three requires the administered test manuals or records, or prior approval of a denominator-free raw-score model with different estimands and predictive limitations. **They are withheld for want of a measurement fact, not because the results were unfavourable.** The sensitivity work supports many principal raw-growth patterns, but it does not license the withheld models.

## What these models cannot tell you

**Nothing causal.** No intervention, no randomisation, no treatment.

**Group differences are not effects of group membership.**

**These children are not the trial children.** Results here neither support nor contradict the trial findings; they describe a different cohort in a different era.

**Growth here is not attributable to anything measured.** It is what happened, not why.

## Model inventory

Six of nine pass and are publishable: `rlm-hg-001` (BAS word reading), `004` (BPVS receptive vocabulary), `005` (TROG receptive grammar), `006` (BAS recall of digits), `007` (BAS similarities), `009` (BAS matrices, waves 3–4 core plus the Down-syndrome wave-5 extension). Withheld at the inputs stage: `002` (BAS spelling), `003` (WORD reading comprehension), `008` (BAS number skills). All nine pass the convergence gate with zero divergences — the withholding is about measurement provenance, not computation.
