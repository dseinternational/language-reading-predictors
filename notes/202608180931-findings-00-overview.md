> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

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

Second, several measures have a **floor**: a large share of children score zero or near it. Nonword reading and phoneme blending are the clearest cases. A floored measure carries very little information — if most children score 0 out of 6, the data cannot say much about how those children differ. For these outcomes the project switches the primary question from "how much did the score change?" to "did the child move off the floor at all?", which is a question the data can actually answer. Phoneme blending has a further wrinkle: it is a three-choice task, so a child guessing at random scores about one third correct, and a companion model accounts for that guessing floor explicitly.

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

Three rules about these labels are worth stating, because they are easy to misread. The label describes **how strong the evidence is for a stated claim**, never how big the effect is — "strong evidence the intervention helps" says nothing about whether the help is large. The label is oriented to whichever direction the data favour, so a clearly negative result is strong evidence of _harm_, not "inconclusive". And a flat result is **inconclusive**, which means the study could not tell — it never means "no effect". Where it matters, the notes also report the probability that the benefit exceeds a pre-agreed practically-meaningful size, because "almost certainly positive" and "almost certainly big enough to matter" are different claims and can diverge.

## Cause versus association

This distinction does more work in this series than any other, so it is worth being blunt.

**Only a handful of quantities in this entire project support a causal reading**, and they all trace back to the randomisation at timepoint 2: the trial's intention-to-treat contrast, the difference-in-differences contrast at that wave, and the on-intervention marginal effect in the gain-factor models. Everything else — every coefficient on age, ability, hearing, memory, vocabulary, every relationship between one skill and another, every dose-response slope — is an **adjusted association**. It describes which children progressed, not which levers move outcomes.

This is not excessive caution. Reading a covariate coefficient from a model built to estimate a treatment effect as though it were itself a causal effect is a well-known error (sometimes called the Table 2 fallacy): the model is not designed to control the right things for that second question, and the coefficient absorbs whatever else it happens to stand in for. The covariate sets here were chosen from a causal diagram to make the _treatment_ estimate as clean as possible, which generally makes the other coefficients less interpretable, not more. When a family note says "association, not a lever", it means that literally.

One more scope limit applies to the trial results. Because the analysis starts from the children present in the archive and requires an observed outcome, the headline trial estimate is an **available-case modified intention-to-treat estimate**, not the effect for all 57 randomised children. It supports a causal reading only if being in the analysis set is unrelated to how a child would have responded. The word-reading model quantifies how much that assumption matters, and the answer is: enough to take seriously.

## Withheld is not the same as null

Every fit passes through an automatic gate before any number reaches a reader, and the gate can withhold a result for reasons that have nothing to do with the finding being uninteresting.

The gate runs in four stages. **Inputs**: are the measurement facts the model relies on actually confirmed? **Computation**: did the simulation converge properly? **Artefacts**: did every required sensitivity analysis actually run? **Robustness**: is the treatment estimate driven by the data rather than by the prior? A failure at any stage withholds the result.

In this run, **214 of 220 models are publishable**. The 6 withheld all fail at the _inputs_ stage, and all for the same reason: three measures in the historical cohort (BAS spelling, BAS number skills, and WORD reading comprehension) have no confirmed maximum score. The original 2002 paper analysed raw scores without stating the test maxima, so the models are currently using the highest observed score as a stand-in — which is a guess, not a fact about the instrument. A dedicated sensitivity analysis showed the growth conclusions do not depend on that guess, and the project still chose to withhold them, because "robust to our guess" is not the same as "we know the scale". Those results stay withheld until someone can supply the actual test records.

So: a withheld model is one whose result **we are declining to state**, not one that found nothing. The relevant family notes name each withheld model and the reason.

## What passed, and how far to trust the computation

These models are fitted by simulation, and simulation can fail quietly, so the computation is checked before any result is read. The standard requires the independent simulation runs to agree with each other (R-hat ≤ 1.01), enough effectively independent samples to pin down the answer (≥ 400), a healthy exploration diagnostic, and **zero divergences** — a specific warning that the simulation may have failed to reach part of the answer.

In this run **all 220 models pass, and not one has a single divergence.** Four models initially failed and were refitted with better simulation settings; in every case the estimates were essentially unchanged, which is the reassuring outcome — the problem was with the computation, not the conclusion. The full record is in the companion run note.

One honest caveat carried over from that work: fixing a simulation problem improves how reliably we have measured the posterior, but it cannot narrow a posterior that is genuinely wide because the data are thin. Where a note says a quantity remains weakly determined, that is a limit of the measurements, not of the arithmetic.

## The families

Twenty-two model families are reported, each with its own note. They answer different questions and are not interchangeable.

**Asking whether the intervention worked** (RLI trial; the only causal estimates): `itt` — the headline per-outcome trial estimate; `joint` — the same outcomes fitted together to compare effects across them; `did` — difference-in-differences using the crossover structure; `gain_factors` — change from each child's own starting point; `level_factors` — scores at each timepoint rather than changes.

**Asking how the intervention might work, or how skills relate** (RLI trial; associations): `mechanism` — how one skill tracks another over a period; `joint_mechanism` — two outcomes at once, to test whether a route is specific to decoding; `mediation` and `mediation_multi` — how much of the reading gain runs through letter-sound knowledge and other routes; `dose_response` — whether more intervention sessions track larger gains; `block_exposure` — the staggered second teaching block; `concurrent` — which skills track together at each wave; `lcsm` — coupled changes across four waves; `corr_factor` and `long_corr_factor` — the correlation structure between skill domains, correcting for measurement error; `horseshoe` — a many-predictors ranking that cross-checks the machine-learning analysis; `survival` — how quickly floored children first move off the floor; `growth` — trajectories across waves; `aligned` — a per-protocol view aligning both arms by when their intervention started.

**Describing the separate historical cohort** (observational; no causal quantity): `historical_growth`, `historical_joint`, `adjusted`.

Each note states its data and timepoints, what the model was for, how to read its numbers, what it found, and what it cannot support.
