# Findings index and reading guide — full statistical-model suite (2026-07-20)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

This is the entry point for a set of dated notes that report, model by model, the findings from the full re-fit of every Bayesian statistical model in the study (176 models, production `reporting` configuration, fit 2026-07-19). There is one companion note per model family; each covers **every** model in its family, and reports adjusted **associations** as fully as **causal** effects. This note explains the study, the families, and — importantly — how to read the numbers the family notes quote.

## The study in one paragraph

An exploratory study of what predicts progress in language and reading for children with Down syndrome, run by Down Syndrome Education International. A reading/phonics intervention was evaluated in a **waitlist-crossover randomised design**: children were randomised to receive the intervention immediately or after a wait, and were measured at four timepoints. Roughly 54 children contribute, so samples are small and estimates are correspondingly uncertain — this is a recurring theme. All data and models are preliminary.

## Two-step methodology

1. **Exploratory gradient boosting** (LightGBM, SHAP) learns which predictors matter for each outcome. (Reported elsewhere; not these notes.)
2. **Bayesian models (PyMC)** — these notes — estimate interactions and, where the causal diagram (DAG) supports it, causal effects, with interpretable estimands and honest uncertainty.

## The outcome measures (and their maxima)

Each skill is a test scored as a count of correct items out of a maximum. Effects are modelled on a probability (proportion-correct) scale and then translated back into **items** (probability × maximum), because "+3 of 32 letter sounds" is far easier to grasp than a logit coefficient.

| Symbol                | Skill (test)                                          | Max items         |
| --------------------- | ----------------------------------------------------- | ----------------- |
| W                     | Word reading (EWRSWR)                                 | 79                |
| R                     | Receptive vocabulary (ROWPVT)                         | 170               |
| E                     | Expressive vocabulary (EOWPVT)                        | 170               |
| L                     | Letter-sound knowledge (YARC-LSK)                     | 32                |
| P                     | Phonetic spelling (SPPHON)                            | 92                |
| B                     | Phoneme blending                                      | 10                |
| F                     | Basic concepts (CELF)                                 | 18                |
| T                     | Receptive grammar (TROG-2)                            | 32                |
| N                     | Nonword reading                                       | 6                 |
| TR / TE               | Taught receptive / expressive vocabulary, block 1     | 24 each           |
| UR / UE               | Not-taught receptive / expressive vocabulary, block 1 | 12 each           |
| TR2 / TE2 / UR2 / UE2 | The same, block 2                                     | 24 / 24 / 12 / 12 |

"Taught" words are the specific vocabulary the intervention teaches; "not-taught" are matched words it does not — the gap between them measures teaching-specific versus general gains. R and E are broad standardised vocabulary tests (general language), distinct from the taught/not-taught word sets.

## How to read the numbers (house standard)

A Bayesian model returns a **posterior**: a full probability distribution for each quantity, given the data and priors. We summarise it as follows, and the family notes follow this exactly:

- **Point estimate = the median** of the posterior (not the mean). The median is preferred because it is unchanged by re-scaling (the same point on the logit or the probability scale). Where the mean differs noticeably from the median, the posterior is skewed — the notes flag this but lead with the median.
- **Uncertainty = the 89% equal-tailed credible interval.** A credible interval is the Bayesian counterpart of a confidence interval: "there is an 89% posterior probability the value lies in this range." We use **89%**, not the customary 95%, on purpose — 95% is an arbitrary convention, and its extreme (2.5%/97.5%) limits are the least stable to estimate from a finite sample. Notes also give the inner **50%** interval as the "most-of-the-mass" range.
- **Direction = the tail probability**, e.g. P(effect > 0) = 0.97, read directly — _not_ from whether an interval excludes zero, and never as a p-value (there are none here).
- **Evidence ladder** — fixed, claim-oriented labels attached to the tail probability: **inconclusive** (P < 0.75), **suggestive** (≥ 0.75), **moderate** (≥ 0.91), **strong** (≥ 0.97), **very strong** (≥ 0.99), i.e. round odds of 3:1, 10:1, 30:1, 100:1. The label describes the _strength of evidence for a directional claim_, oriented to the favoured direction; it is stated after the probability and **never** describes the size of the effect.
- **Direction and size are separate claims.** A big tail probability says an effect is _probably positive_; it does not say the effect is _large_. Where a note quotes it, `P(benefit ≥ δ)` is the probability the effect clears a pre-specified minimally-important difference (the size claim), and ROPE mass (region of practical equivalence around zero) quantifies a "probably negligible" reading. A flat result is **inconclusive**, never "null" or "no effect".
- **Small samples inflate winners.** With ~54 children, any estimate that just clears a threshold is on average too big (the "winner's curse"), so the notes lead with the interval, not the point.

## Causal versus association — read this before any coefficient

Randomisation is what licenses a causal claim. In this suite only three quantities are **causal**:

1. the **ITT effect τ** (intention-to-treat: the effect of being _assigned_ to the intervention);
2. the **difference-in-differences t2 contrast** (the randomised gap at the first post-baseline wave); and
3. the **gain-factor on-intervention treatment marginal**.

Everything else — every covariate coefficient (age, hearing, speech, baseline score, cognitive ability, other skills), every mechanism dose-response slope, every mediator→outcome path, every latent correlation, every dose slope, every aligned-cohort contrast, and any difference-in-differences or level quantity at _post-crossover_ waves — is an **adjusted association**. An association describes _who progresses_ ("children higher on X tend to gain more on Y, holding the adjustment set fixed"), not a lever you can pull. Reading a control variable's coefficient as if it were causal is the **Table-2 fallacy**, and the notes call it out. Residual confounding by latent general ability remains for all associations; the child random intercept pools stable individual differences but does not stand in for general ability. Adjustment sets are fixed in advance from the DAG, so a skill missing from a model was excluded by the diagram, not found unimportant.

## Convergence gate

Before any interpretation, each fit is checked against a gate: R-hat ≤ 1.01 (chains agree), effective sample size ≥ 400 (enough independent draws), BFMI ≥ 0.3 (the sampler explored well), and zero divergences. Each family note states each model's gate. Two failure types recur: **divergence-only** flags (everything else passes; a handful of divergences well under the 1% guidance) are usable with a note; the **correlation/measurement models** fail on R-hat/ESS (a latent-factor "funnel" geometry) — their correlations are usable but their structural coefficients are held pending reparameterisation.

## The families and the questions they answer

| Note | Family                                                 | Models | Question it answers                                                                                      | Causal?                          |
| ---- | ------------------------------------------------------ | ------ | -------------------------------------------------------------------------------------------------------- | -------------------------------- |
| 01   | **ITT suite** (+ joint, block-exposure)                | 35     | Did _assignment_ to the intervention change each skill? (the headline effect)                            | **Yes** (τ)                      |
| 02   | **Gain-factors** (ANCOVA)                              | 19     | Same effect via a period-stacked gain model; plus which baseline/ability/skills predict gains            | Yes (trt marginal) + assoc.      |
| 03   | **Level-factors**                                      | 11     | How each skill's _level_ differs by group over time                                                      | t2 only; rest assoc.             |
| 04   | **Difference-in-differences**                          | 14     | The within-person randomised gap using the crossover design                                              | Yes (t2 δ) + assoc.              |
| 05   | **Aligned (per-protocol)**                             | 9      | Onset-aligned dose/response once both arms have been treated                                             | Association only                 |
| 06   | **Mechanism**                                          | 27     | Shape of one skill's association with another (e.g. letter-sounds → word reading); interactions; "knees" | Association only                 |
| 07   | **Mediation (g-formula)**                              | 18     | How much of the reading gain runs _through_ letter-sounds vs other routes                                | Association (decomposition)      |
| 08   | **Dose-response**                                      | 5      | Whether more sessions attended goes with more progress                                                   | Association (dose is a collider) |
| 09   | **Latent change-score (LCSM)**                         | 5      | Do earlier levels of one skill predict later _change_ in another?                                        | Association (cross-lagged)       |
| 10   | **Growth & historical**                                | 13     | Growth trajectories, and comparison against a historical cohort                                          | Association                      |
| 11   | **Horseshoe (selection)**                              | 5      | Which predictors survive sparse Bayesian variable selection (cross-check of the boosting ranking)        | Association                      |
| 12   | **Correlation & measurement** (+ concurrent, adjusted) | 13     | Latent domain correlations, measurement structure, and adjusted between-child associations               | Association                      |
| 13   | **Survival**                                           | 2      | Time-to-first success on heavily floored skills (spelling, nonword reading)                              | Association                      |

## The coherent story to expect (spoiler)

Reading the notes together: strong-to-very-strong evidence of an ITT benefit on the code-related and directly-taught skills — letter-sound knowledge (L), phoneme blending (B), word reading (W), taught expressive vocabulary (TE) — tapering to moderate/suggestive on transfer measures, and **inconclusive and probably negligible on broad standardised vocabulary (R, E)**. The word-reading gain is **mediated by letter-sound knowledge**, not by the vocabulary route. The randomised result is echoed by the difference-in-differences and gain-factor re-analyses, and the associational families describe the developmental scaffolding (which skills track which) without over-claiming cause.
