> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).
>
> Substantially corrected by a LLM-based AI tool (Codex/GPT-5).
>
> Post-run additions summarised by a LLM-based AI tool (Claude Code/Opus 5).

<!-- cspell:ignore basnum basspel woco readgrp Kruschke -->

# Findings overview: how to read this series

**This is the introduction to a series of notes reporting what the statistical models in this project found.** Each companion note covers one model family. This note explains the studies, the data, the vocabulary and the reasoning style the series uses, so the family notes can get on with their results.

The series assumes undergraduate science or mathematics — you are comfortable with regression, standard deviations and the idea of a sampling distribution — but it does not assume you have worked in a Bayesian framework before. Where the Bayesian output differs from the frequentist output you may be used to, this note says so explicitly rather than quietly substituting one for the other.

All results are preliminary and exploratory. This is work in progress, not a published trial report.

## The two studies

The series draws on two different sets of children. Confusing them is the easiest mistake to make, so every family note states which one it uses at the top.

**The RLI trial** is a randomised controlled trial of a reading and language intervention. 57 children were randomised, 29 to receive the intervention immediately and 28 to a waiting-list control. Three children were lost to follow-up after randomisation, leaving **54 children** in the analysis files (28 immediate, 26 waiting). Children were assessed at **four timepoints**, giving three intervening periods. Analyses that need an observed outcome at a particular wave often work with slightly fewer children again — the word-reading analysis, for instance, has 53.

The trial uses a **waiting-list crossover** design, and one consequence of that design governs almost everything in this series. The immediate group begins the intervention after timepoint 1; the waiting group begins after timepoint 2. So at **timepoint 2, and only at timepoint 2**, one randomised group has received the intervention and the other has not. That single comparison is the one place where randomisation is doing its work and a genuine cause-and-effect claim is available. By timepoint 3 both groups have been treated, just for different lengths of time, so every later comparison is between groups that are no longer randomly different in the relevant respect. Family notes flag this constantly, and it is why so many quantities in this project are labelled "association" rather than "effect".

**The Reading, Language and Memory study** (Byrne, MacDonald and Buckley) is a completely separate, older, observational cohort: **97 children across five annual assessment waves**, in three reading groups — children with Down syndrome, average readers, and reading-matched comparison children. Nobody was randomised to anything. Families built on this cohort can describe how skills track together and how groups differ over time, but they contain **no causal quantity at all**, and the notes say so plainly rather than hedging.

## Why the scores are modelled as "so many out of so many"

Most outcomes here are test scores of the form "23 correct out of 32 items". Two features of such scores drive the modelling choices, and both matter when reading results.

First, the scores are **bounded counts**, not free-floating numbers. A child cannot score 33 out of 32, and a child near the ceiling has less room to improve than one in the middle. Ordinary linear regression ignores both facts. These models instead use a **Beta-Binomial** likelihood, which is the natural distribution for "k successes out of n attempts" with the added allowance that children vary more than pure coin-flipping would predict. Results are usually reported twice: on the model's internal log-odds scale, and translated back into **items** ("about +2.4 words out of 79"). The items translation is the one to quote — it is what the score actually means.

Second, several measures have a **floor**: a large share of children score zero or near it. Phonetic spelling and nonword reading are the clearest cases. A floored measure carries very little information — if most children score 0 out of 6, the data cannot say much about how those children differ. In the `itt` family, a post-hoc, arm-blind rule switches the exploratory headline to the arm difference in moving above zero among children observed at zero before treatment. Other families use different off-floor estimands: `gain_factors` retains all eligible stacked periods and models post-period off-floor status, while `level_factors` estimates off-floor prevalence at each wave. Those are not baseline-floor subgroup exit contrasts. Phoneme blending is different again: its primary model remains Beta-Binomial, but because it is a three-choice task a companion model explicitly allows for the one-third score expected from random guessing.

## Reading the numbers if you are used to p-values

The models are Bayesian, so the output is a **posterior distribution**: for each quantity, a full picture of which values are plausible given the data and the model. Four practical differences from a frequentist write-up:

**There are no p-values, and no significance tests.** Nothing here is "significant" or "non-significant".

**The interval means what you probably always wanted a confidence interval to mean.** When a note says "+2.4 items, 89% credible range +0.7 to +4.1", that is a direct statement: given this model and these data, there is an 89% probability the true value lies between +0.7 and +4.1. There is no repeated-sampling story. (A frequentist 95% confidence interval does _not_ license that reading, though it is almost universally read that way.)

**The interval is 89%, not 95%, and that is deliberate.** 95% is an arbitrary convention inherited from significance testing, and its 2.5% and 97.5% endpoints are the least numerically stable parts of a simulated posterior. The project reports a deliberately non-round **89%** outer range, usually with an inner **50%** range as well, so that no reader mistakes the interval for a significance test in disguise. The reasoning follows Kruschke's reporting guidelines and is recorded in `notes/202607172359-credible-interval-standard.md`.

**Direction is read from a probability, not from whether the interval clears zero.** Each result carries a number like "P(effect > 0) = 0.99" — the posterior probability the effect is positive. That is the direct answer to "which way does this go, and how sure are we?", and it does not degrade into a yes/no verdict at an arbitrary threshold.

The project attaches a fixed vocabulary to those probabilities so that different notes mean the same thing by the same word:

| Posterior probability | Label        | Rough odds |
| --------------------- | ------------ | ---------- |
| ≥ 0.75                | suggestive   | 3:1        |
| ≥ 0.91                | moderate     | 10:1       |
| ≥ 0.97                | strong       | 30:1       |
| ≥ 0.99                | very strong  | 100:1      |
| below 0.75            | inconclusive | —          |

Three rules about these labels are worth stating, because they are easy to misread. The label describes **how strong the evidence is for a stated claim**, never how big the effect is — "strong evidence the intervention helps" says nothing about whether the help is large. The label is oriented to whichever direction the data favour, so a clearly negative result is strong evidence of _harm_, not "inconclusive". And a flat result is **inconclusive**, which means the study could not tell — it never means "no effect". Where it matters, the notes also report the probability that the benefit exceeds a practically meaningful size. That threshold was agreed after the initial results review and is therefore a transparent post-hoc judgement, always read with its threshold-sensitivity analysis rather than as a prospectively specified decision rule. The rule behind it — half the waiting-list arm's natural gain over the untreated first period, floored at one item — was confirmed as the project's standing rule on 2026-08-19, so later analysis rounds will apply it prospectively; the values used in this run remain post-hoc in timing (`notes/202608191130-practical-difference-rule-confirmed.md`). "Almost certainly positive" and "almost certainly big enough to matter" are different claims and can diverge.

## Cause versus association

This distinction does more work in this series than any other, so it is worth being blunt.

**Only a handful of quantities in this entire project support a causal reading**, and they are alternative model-based versions of the same randomised timepoint-2 arm comparison: the single- and multi-outcome available-case modified intention-to-treat contrasts, the timepoint-2 arm contrasts in the `did` and `level_factors` families, the period-1-standardised treatment marginal in interaction-free primary `gain_factors` fits, and the fitted first-window assigned-arm latent-change contrasts in `lcsm`. They are not independent experiments or separate sources of identification. Everything else — every coefficient on age, ability, hearing, memory, vocabulary, every relationship between one skill and another, every dose-response slope — is an **adjusted association**. It describes which children progressed, not which levers move outcomes.

This is not excessive caution. Reading a covariate coefficient from a model built to estimate a treatment effect as though it were itself a causal effect is a well-known error (sometimes called the Table 2 fallacy): the model is not designed to control the right things for that second question, and the coefficient absorbs whatever else it happens to stand in for. The covariate sets here were chosen from a causal diagram to make the _treatment_ estimate as clean as possible, which generally makes the other coefficients less interpretable, not more. When a family note says "association, not a lever", it means that literally.

One more scope limit applies to the trial results. Because the analysis starts from the children present in the archive and requires an observed outcome, the headline trial estimate is an **available-case modified intention-to-treat estimate**, not the effect for all 57 randomised children. It supports a causal reading only if being in the analysis set is unrelated to how a child would have responded. The word-reading model quantifies how much that assumption matters, and the answer is: enough to take seriously.

## Withheld is not the same as null

Every fit passes through an automatic gate before any number reaches a reader, and the gate can withhold a result for reasons that have nothing to do with the finding being uninteresting.

The gate runs in four stages. **Inputs**: are the measurement facts the model relies on actually confirmed? **Computation**: did the simulation converge properly? **Artefacts**: did every required sensitivity analysis actually run? **Robustness**: do the required checks preserve the scientific quantity being released? For treatment estimates this includes prior sensitivity; for the historical growth influence refit it requires every named coefficient to retain its median direction and overlapping 89% intervals. A failure at any stage withholds the result.

As of the 2026-08-20 full refit, in which every registered model was refitted together from one code state and one data state, **236 of 242 models are publishable** (`notes/202608200800-full-refit-both-layers-2026-08.md`). The 6 withheld all fail at the _inputs_ stage, and all for the same reason: three measures in the historical cohort (BAS spelling, BAS number skills, and WORD reading comprehension) have no confirmed maximum score. The original 2002 paper analysed raw scores without stating the test maxima, so the models are currently using the highest observed score as a stand-in — which is a guess, not a fact about the instrument. Likelihood and participant-bootstrap checks applied only to the three affected `historical_growth` fits and supported many principal raw-growth patterns; they did not establish universal robustness, and the bootstrap returned a strict `no_go` for BAS spelling after one near-zero between-group contrast changed median sign. Those checks also do not repair the affected `adjusted`, `horseshoe` or measurement model. All 6 remain withheld because "robust under some alternative analyses" is not the same as "we know the scale"; clearing them requires the actual test records or prior approval of a different raw-score analysis.

So: a withheld model is one whose result **we are declining to state**, not one that found nothing. The relevant family notes name each withheld model and the reason.

## What passed, and how far to trust the computation

These models are fitted by simulation, and simulation can fail quietly, so the computation is checked before any result is read. The standard requires the independent simulation runs to agree with each other (R-hat ≤ 1.01), enough effectively independent samples to pin down the answer (≥ 400), a healthy exploration diagnostic, and **zero divergences** — a specific warning that the simulation may have failed to reach part of the answer.

In the 2026-08-20 refit **all 242 models pass, and not one has a single divergence**. One model needed a better simulation setting to get there (`mech-104`, one divergence in 36,000 draws, refitted at a higher target acceptance), and its estimates were essentially unchanged — the reassuring outcome, since the problem was with the computation rather than the conclusion. The full record is in the companion run note.

One honest caveat carried over from that work: fixing a simulation problem improves how reliably we have measured the posterior, but it cannot narrow a posterior that is genuinely wide because the data are thin. Where a note says a quantity remains weakly determined, that is a limit of the measurements, not of the arithmetic.

## What was added after the main run

Three sets of models were fitted after the sweep, all because reviewing the notes exposed a gap rather than an error. They are described in full in their family notes; the summary matters because two of them change what the series covers.

**Expressive language is now measured.** The Action Picture Test (Renfrew 1997) has two scales — grammar and information — and the project's causal diagram has always treated both as things the intervention could affect. Neither had ever been estimated, because the test's maximum scores were not on file, and a bounded-count model cannot be fitted without them. The maxima were confirmed from the manual on 2026-08-18 (grammar 37, information 40), and both scales were fitted in the `itt` and `did` families.

The result is worth knowing. **Expressive grammar shows the strongest lean of any outcome the intervention did not teach directly**: about **+1 mark out of 37** in the headline trial estimate, with an 0.85 probability of being positive, and about **+1.6 marks** in the crossover model at 0.93 — suggestive and moderate respectively. Neither interval clears zero, so this is a lean rather than a finding, but it is a consistent one across two designs, and it sits on a measure of everyday expressive language rather than a taught word list. Expressive information is inconclusive, and the crossover model suggests the immediate group may have started slightly ahead on it, which is a reason for caution rather than a result.

Two caveats travel with these two outcomes.

The information score awards half marks on some items, which a whole-number counting model cannot accept. The score is therefore doubled and treated as a count out of 80. That is exact — every half mark becomes a whole one — but it tells the model there are eighty separate chances where really there are forty items scored generously, which in principle makes the data look more informative than they are. A companion model that rounds to whole marks instead was fitted to check, and the two agree almost exactly, because the model's own "children vary more than chance" parameter adjusts to compensate. The encoding is safe here, and it was tested rather than assumed.

The second point concerned the practical-difference thresholds. Every outcome is judged against a threshold for what counts as a practically meaningful difference, and the two Action Picture Test thresholds were derived by applying the project's existing rule rather than agreed in advance. For expressive information the rule landed almost exactly on a rounding boundary, so the value was escalated rather than settled quietly. **Both thresholds were ratified on 2026-08-18** — grammar 1 mark of 37, information 1 whole mark — so the size statements for these outcomes now stand on the same footing as the rest of the suite. Like every threshold here they remain post-hoc rather than pre-specified; ratification settles who chose them, not when. The decision is recorded in `notes/202608182015-apt-delta-threshold-ratification.md`.

**The general-ability explanation was tested.** The `mechanism` family attributes part of its skill-to-skill associations to children simply differing in general ability. A measured ability score existed but that family had never adjusted for it, while every other family that appeals to ability does. Seven models now do: an ability-adjusted mirror of the six-model specificity panel, and an ability-adjusted version of the headline letter-sound curve. The short answer is that measured ability explains some of the pattern but not most of it, and it leaves the letter-sound-to-decoding result untouched — the detail is in the `mechanism` note.

**One association is now estimated across all four waves at once.** The suite asked how one skill's level tracks another's at each wave separately (`concurrent`) and how a skill's level tracks a later change (`mechanism`), but never the plain pooled question. The new `pooled_levels` family answers it for letter-sound knowledge and the two reading measures, and — because pooling repeated measures on the same children invites a well-known trap — splits the answer into the part that separates children from the part that moves within a child. Those two parts turn out to be very different, which is the finding; the `pooled_levels` note explains it.

## What the 2026-08-20 re-read changed

Every model in the project was refitted on 2026-08-20 from one code state and one data state (`notes/202608200800-full-refit-both-layers-2026-08.md`), and this series was re-read against the new artefacts rather than carried forward. Three things moved.

**One family now reports a different quantity.** `level_factors` was re-parameterised (#552) so that it reports the _change_ in the arm gap over the randomised window rather than the arm gap at timepoint 2. That change resolved the vocabulary disagreement this series used to describe: the negative vocabulary contrasts turned out to be the gap the arms started with, and the model now separates the two explicitly. Its note has been rewritten.

**One family gained four members.** `pooled_levels` now covers expressive vocabulary, receptive vocabulary, phonological memory and speech production as well as letter sounds (#553), and the joint family gained three dependence-aware companions (#551) that discharge the "dependence-limited" caveat its contrasts used to carry.

**Everything that carries the hearing flag moved slightly.** The composite is now derived with a three-valued OR (25 flagged, 20 clear, 9 unknown), which reclassifies one child. Coefficients in `mechanism`, `gain_factors`, `mediation`, `concurrent`, `lcsm`, `corr_factor`, `joint_mechanism`, `block_exposure`, `adjusted` and `pooled_levels` moved in the second or third decimal. No direction changed, and only one evidence label did (the latent-change model's hearing coefficient, 0.90 to 0.85). The notes for those families have been updated to the new figures.

The families that carry no hearing term — `itt`, `did`, `dose_response`, `aligned`, `survival`, `horseshoe`, `growth` and the historical-cohort families — reproduce their August numbers exactly, as they should: same code, same data, same seed. Their notes were checked against the new artefacts and left unchanged.

## The families

Twenty-three model families are reported, each with its own note. They answer different questions and are not interchangeable.

**Asking whether the intervention worked** (RLI trial; the causal estimates): `itt` — the headline per-outcome trial estimate; `joint` — the same outcome-specific marginal models placed in one factorised graph, with provisional cross-outcome contrasts that do not account for within-child residual covariance; `did` — an arm-by-wave repeated-level model whose timepoint-2 arm gap is the randomised quantity; `gain_factors` — a period-stacked post-score ANCOVA conditional on pre-score whose reported treatment marginal is standardised to the randomised first period; `level_factors` — scores and arm gaps at each timepoint rather than post-score-given-pre-score transitions; `lcsm` — latent changes whose fitted window-1 assigned-arm contrasts inherit randomisation, although their cross-process couplings and later-window contrasts do not.

**Asking how the intervention might work, or how skills relate** (RLI trial; associations): `mechanism` — how one skill tracks another over a period; `joint_mechanism` — two outcomes at once, to test whether a route is specific to decoding; `mediation` and `mediation_multi` — how the fitted decomposition allocates the reading contrast through letter-sound knowledge and other routes; `dose_response` — whether more intervention sessions track larger gains; `block_exposure` — the staggered second teaching block; `concurrent` — which skills track together at each wave; the cross-process couplings in `lcsm` — prior levels or changes associated with later changes; `corr_factor` and `long_corr_factor` — the correlation structure between skill domains, correcting for measurement error; `horseshoe` — a many-predictors ranking that cross-checks the machine-learning analysis; `survival` — how quickly floored children first move off the floor; `growth` — trajectories across waves; `aligned` — a per-protocol view aligning both arms by when their intervention started; `pooled_levels` — one level association across all four waves at once, split into the part that separates children from the part that moves within a child.

**Describing observational development and prediction:** `historical_growth` and `historical_joint` use only the separate historical cohort; `adjusted` spans that cohort and one RLI model, but all of its baseline-predictor coefficients are associations rather than treatment effects.

Each note states its data and timepoints, what the model was for, how to read its numbers, what it found, and what it cannot support.
