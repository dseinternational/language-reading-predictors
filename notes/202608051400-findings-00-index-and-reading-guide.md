<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Findings index and reading guide — full statistical-model suite (2026-08-05)

Entry point for a set of dated notes reporting, model by model, the findings from the complete re-fit of every Bayesian statistical model in the study — **194 models across 22 families**, production `reporting` configuration, fit 2026-08-04/05. There is one note per family, each covering **every** model in it and reporting adjusted **associations** as fully as **causal** effects. This note explains the study, the conventions and — importantly — how to read the numbers. Written for a broad audience, including readers who do not work with Bayesian statistics day to day.

**Preliminary research data — all estimates provisional.**

## Why this series replaces the previous one

The previous series (`notes/2026072109*-findings-*`) was marked superseded in part on 2026-08-02: any finding drawn from a gate-failed fit was withheld under the divergence-qualification policy, and at that time a number of fits were failing.

**All 194 models in this run pass the convergence gate, with zero divergent transitions across the whole suite.** This is the first run in which every model clears computation. A separate robustness gate adopted on 2026-08-05 — after these fits were made — then withholds **ten** of them from release. It covers the five families whose findings box publishes a randomised causal claim (ITT, joint, DiD, gain factors, level factors: 70 fits) and asks whether the coefficient carrying that headline responds to the prior but not to the data. Two ITT fits are withheld for a missing treatment-prior grid (note 01); the other eight — four DiD, two gain-factor, two level-factor — are withheld because their causal coefficient is prior-dominant (notes 03, 04, 05). None is a letter-sound model, so the suite's strongest result is untouched. Three things also changed materially:

- **Phoneme blending is reported for the first time.** Its results were withheld pending a mandatory paired-link sensitivity bundle that could not be built; that is now in place, and the result is link-sensitive in a way that matters (note 01).
- **The four measurement models are reported from clean fits.** Their earlier failures were misdiagnosed as intrinsic geometry; the cause was an unidentified nuisance parameter, and fixing it moved the domain correlations slightly (note 14).
- **Every model now emits a prior-predictive check**; 24 previously emitted none.

The run record — what was fitted, what failed, what was repaired and how — is `notes/202608050649-reporting-refit-predictive-checks.md`.

## The study in one paragraph

An exploratory study of what predicts progress in language and reading for children with Down syndrome, run by Down Syndrome Education International. A reading and phonics intervention was evaluated in a **waitlist-crossover randomised design**: children were randomly assigned to receive it immediately or after a wait, and were measured at four timepoints. About 54 children contribute, so samples are small and estimates correspondingly uncertain — a recurring theme. A separate **historical Byrne cohort** (the "reading-language-memory" or `rlm` study, note 18) provides natural-history context but was not randomised.

## The outcome measures

Each skill is a test scored as a count of correct items out of a maximum. Effects are modelled on a proportion-correct scale and translated back into **items**, because "+3 of 32 letter sounds" is easier to grasp than a logit coefficient. Symbols are the node labels from the study's causal diagram.

| Symbol  | Skill (test)                                          | Max items |
| ------- | ----------------------------------------------------- | --------- |
| WR      | Word reading (EWRSWR)                                 | 79        |
| RV / EV | Receptive / expressive vocabulary (ROWPVT, EOWPVT)    | 170 each  |
| LS      | Letter-sound knowledge (YARC-LSK)                     | 32        |
| PS      | Phonetic spelling (SPPHON)                            | 92        |
| PA      | Phoneme blending                                      | 10        |
| LF      | Basic concepts (CELF)                                 | 18        |
| RG      | Receptive grammar (TROG-2)                            | 32        |
| NW      | Nonword reading                                       | 6         |
| TR / TE | Taught receptive / expressive vocabulary, block 1     | 24 each   |
| UR / UE | Not-taught receptive / expressive vocabulary, block 1 | 12 each   |

"Taught" words are the specific vocabulary the intervention teaches; "not-taught" are matched words it does not. RV and EV are broad standardised tests, distinct from the taught word sets.

Non-outcome variables use the same diagram labels: **A** age, **GA** general (latent) ability, **HS** hearing, **SP** speech production, **RW** phonological memory, **IG** intervention group, **IS** intervention sessions (attendance).

## How to read the numbers

- **Point estimate = the posterior median**, not the mean — unchanged by rescaling, so the same point on the logit and probability scales.
- **Uncertainty = the 89% equal-tailed credible interval**: "an 89% posterior probability the value lies in this range". 89% rather than 95% on purpose — 95% is an arbitrary convention and its 2.5/97.5% limits are the least stable to estimate.
- **Direction = the tail probability**, e.g. P(effect > 0) = 0.97, read directly. Not from whether an interval excludes zero, and never as a p-value — there are none here.
- **Evidence ladder** — fixed labels attached to the tail probability: **inconclusive** (< 0.75), **suggestive** (≥ 0.75), **moderate** (≥ 0.91), **strong** (≥ 0.97), **very strong** (≥ 0.99), i.e. round odds of 3:1, 10:1, 30:1, 100:1. The label qualifies evidence for a **directional claim**, is oriented to the favoured direction, is stated after the probability, and **never** describes effect size.
- **Direction and size are separate claims.** A high tail probability says an effect is probably positive, not that it is large. `P(benefit ≥ δ)` is the size claim against a minimally-important difference; ROPE mass quantifies "probably negligible". A flat result is **inconclusive**, never "null" or "no effect".
- **Small samples inflate winners.** At n ≈ 54 any estimate that just clears a threshold is on average too big, so lead with the interval.
- **δ thresholds are post-hoc**, agreed after the first results review — read `P(≥δ)` beside the threshold-sensitivity curve, not as a pre-registered test.

## Causal versus association — read this before any coefficient

Randomisation is what licenses a causal claim. In this suite only randomisation-anchored contrasts are causal:

1. the **ITT effect τ** (effect of being _assigned_ to the intervention) — notes 01, 02;
2. the **DiD t2 contrast** — note 05;
3. the **gain-factor on-intervention marginal**, averaged over the randomised period-1 transition — note 03;
4. the **t2 group contrast** in the level-factor family — note 04.

**Everything else in every family is an adjusted association**, including every skill-to-skill coupling, every baseline predictor and every dose slope. Reading those as levers is the Table-2 fallacy. Two specific traps recur:

- **A child random intercept is not a control for general ability.** It partially pools stable between-child variation. Latent general ability confounds the non-treatment coefficients regardless, and the measurement models (note 14) show why: the latent domains correlate at 0.82–0.95 in the historical cohort, so "adjusting for" one removes much of the others.
- **Temporal ordering is not identification.** Prior level predicting later change (note 10) rules out reverse simultaneity, nothing more.

Even the causal estimates are **available-case**: typically 53–54 of 57 randomised children, requiring that missingness does not depend jointly on arm and potential outcomes.

## A note on items

Items are **not equal-interval units of learning** — three items at the hard end of a test represent more progress than three at the easy end — and they are **not comparable across measures**. "+3.5 of 32 letter sounds" and "+2.4 of 79 words" cannot be ranked against each other.

## The notes

| #                                                      | Family                        | Models | What it answers                                                    |
| ------------------------------------------------------ | ----------------------------- | -----: | ------------------------------------------------------------------ |
| [01](202608051401-findings-01-itt-suite.md)            | ITT                           |     28 | Did the intervention work, per outcome?                            |
| [02](202608051402-findings-02-joint.md)                | Joint                         |      4 | Cross-outcome consistency; taught vs not-taught contrasts          |
| [03](202608051403-findings-03-gain-factors.md)         | Gain factors                  |     21 | The effect re-estimated from all periods; who progresses           |
| [04](202608051404-findings-04-level-factors.md)        | Level factors                 |     11 | The levels companion to note 03                                    |
| [05](202608051405-findings-05-did-crossover.md)        | DiD crossover                 |     14 | The effect from the crossover design; what happened after catch-up |
| [06](202608051406-findings-06-aligned-per-protocol.md) | Aligned per-protocol          |      9 | Onset-aligned 40-week windows                                      |
| [07](202608051407-findings-07-mechanism.md)            | Mechanism (+ joint mechanism) |     36 | Which skills travel together; decoding specificity                 |
| [08](202608051408-findings-08-mediation.md)            | Mediation (+ two-mediator)    |     19 | What carries the word-reading gain                                 |
| [09](202608051409-findings-09-dose-response.md)        | Dose-response                 |      5 | Does more attendance go with more progress?                        |
| [10](202608051410-findings-10-lcsm.md)                 | Latent change score           |      5 | Lead–lag: does one skill precede change in another?                |
| [11](202608051411-findings-11-concurrent.md)           | Concurrent                    |     11 | Which skills go together at a point in time                        |
| [12](202608051412-findings-12-adjusted.md)             | Adjusted                      |      2 | Which baseline characteristics predict gain                        |
| [13](202608051413-findings-13-horseshoe.md)            | Horseshoe                     |      5 | Shrinkage cross-check on predictor ranking                         |
| [14](202608051414-findings-14-measurement.md)          | Measurement models            |      5 | What the tests measure; how distinct the domains are               |
| [15](202608051415-findings-15-growth.md)               | Growth curves                 |      3 | Does ability predict trajectory shape?                             |
| [16](202608051416-findings-16-block-exposure.md)       | Block exposure                |      4 | Block-2 teaching specificity (exploratory)                         |
| [17](202608051417-findings-17-survival.md)             | Survival                      |      2 | Time to come off the floor                                         |
| [18](202608051418-findings-18-historical.md)           | Historical Byrne cohort       |     10 | Natural history without intervention                               |
| [19](202608051419-findings-19-cross-model-summary.md)  | **Cross-model summary**       |      — | What the whole suite says together                                 |

## Suggested reading order

**For the findings:** 00 → 01 (the effects) → 08 (what carries them) → 19 (synthesis).

**For "who progresses":** 12, 13 and 15 together, then 14 for why they agree.

**For methods and computation:** the run record, `notes/202608050649-reporting-refit-predictive-checks.md`.
