# Findings index and reading guide — full statistical-model suite (2026-07-21)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

This is the entry point for a set of dated notes that report, model by model, the findings from the full re-fit of every Bayesian statistical model in the study (**179 models**, production `reporting` configuration, fit 2026-07-20/21). There is one companion note per model family; each covers **every** model in its family, and reports adjusted **associations** as fully as **causal** effects. This note explains the study, the families, and — importantly — how to read the numbers the family notes quote. It is written for a broad audience, including readers who do not work with Bayesian statistics day to day.

## The study in one paragraph

An exploratory study of what predicts progress in language and reading for children with Down syndrome, run by Down Syndrome Education International. A reading and phonics intervention was evaluated in a **waitlist-crossover randomised design**: children were randomly assigned to receive the intervention immediately or after a wait, and were measured at four timepoints. Roughly 54 children contribute, so samples are small and estimates are correspondingly uncertain — this is a recurring theme. A separate **historical Byrne cohort** (the "reading-language-memory", or `rlm`, study; see note 18) provides natural-history context but was not randomised. All data and models are preliminary.

## Two-step methodology

1. **Exploratory gradient boosting** (LightGBM, SHAP) learns which predictors matter for each outcome. (Reported elsewhere; not these notes.)
2. **Bayesian models (PyMC)** — these notes — estimate interactions and, where the causal diagram (DAG) supports it, causal effects, with interpretable estimands and honest uncertainty.

## The outcome measures (and their maxima)

Each skill is a test scored as a count of correct items out of a maximum. Effects are modelled on a probability (proportion-correct) scale and then translated back into **items** (probability × maximum), because "+3 of 32 letter sounds" is far easier to grasp than a logit coefficient. The symbols below are the **node labels used in the study's causal diagram** (the DAG, `dag/dag-language-reading.dagitty`), so the notes, their tables, and the DAG all name each variable the same way (e.g. WR = word reading, LS = letter-sound knowledge, PA = phoneme blending / phonological awareness).

| Symbol                | Skill (test)                                          | Max items         |
| --------------------- | ----------------------------------------------------- | ----------------- |
| WR                    | Word reading (EWRSWR)                                 | 79                |
| RV                    | Receptive vocabulary (ROWPVT)                         | 170               |
| EV                    | Expressive vocabulary (EOWPVT)                        | 170               |
| LS                    | Letter-sound knowledge (YARC-LSK)                     | 32                |
| PS                    | Phonetic spelling (SPPHON)                            | 92                |
| PA                    | Phoneme blending                                      | 10                |
| LF                    | Basic concepts (CELF)                                 | 18                |
| RG                    | Receptive grammar (TROG-2)                            | 32                |
| NW                    | Nonword reading                                       | 6                 |
| TR / TE               | Taught receptive / expressive vocabulary, block 1     | 24 each           |
| UR / UE               | Not-taught receptive / expressive vocabulary, block 1 | 12 each           |
| TR2 / TE2 / UR2 / UE2 | The same, block 2                                     | 24 / 24 / 12 / 12 |

"Taught" words are the specific vocabulary the intervention teaches; "not-taught" are matched words it does not — the gap between them measures teaching-specific versus general gains. RV and EV are broad standardised vocabulary tests (general language), distinct from the taught/not-taught word sets. (The not-taught sets UR/UE have no separate DAG node; they are the matched comparison words for the taught TR/TE nodes.)

The notes use the same DAG labels for the **non-outcome variables** they adjust for or discuss: **A** age, **GA** general (latent) ability, **HS** hearing status, **SP** speech production, **RW** phonological memory (word/nonword repetition), **IG** intervention group (the randomised arm), and **IS** intervention sessions (attendance dose). Where these are clearer spelled out ("hearing", "age"), the notes do so; where a symbol is used, it is the DAG symbol.

## How to read the numbers (house standard)

A Bayesian model returns a **posterior**: a full probability distribution for each quantity, given the data and priors. We summarise it as follows, and the family notes follow this exactly:

- **Point estimate = the median** of the posterior (not the mean). The median is preferred because it is unchanged by re-scaling (the same point on the logit or the probability scale). Where the mean differs noticeably from the median, the posterior is skewed — the notes flag this but lead with the median.
- **Uncertainty = the 89% equal-tailed credible interval.** A credible interval is the Bayesian counterpart of a confidence interval: "there is an 89% posterior probability the value lies in this range." We use **89%**, not the customary 95%, on purpose — 95% is an arbitrary convention, and its extreme (2.5%/97.5%) limits are the least stable to estimate from a finite sample. Notes also give the inner **50%** interval as the "most-of-the-mass" range where the fit reports it.
- **Direction = the tail probability**, e.g. P(effect > 0) = 0.97, read directly — _not_ from whether an interval excludes zero, and never as a p-value (there are none here).
- **Evidence ladder** — fixed, claim-oriented labels attached to the tail probability: **inconclusive** (P < 0.75), **suggestive** (≥ 0.75), **moderate** (≥ 0.91), **strong** (≥ 0.97), **very strong** (≥ 0.99), i.e. round odds of 3:1, 10:1, 30:1, 100:1. The label describes the _strength of evidence for a directional claim_, oriented to the favoured direction; it is stated after the probability and **never** describes the size of the effect. (Because the label attaches to the un-rounded probability, two effects both displayed as, say, "99%" can carry different labels when their exact probabilities fall either side of a rung.)
- **Direction and size are separate claims.** A big tail probability says an effect is _probably positive_; it does not say the effect is _large_. Where a note quotes it, `P(benefit ≥ δ)` is the probability the effect clears a pre-specified minimally-important difference (the size claim), and ROPE mass (region of practical equivalence around zero) quantifies a "probably negligible" reading. A flat result is **inconclusive**, never "null" or "no effect".
- **Small samples inflate winners.** With ~54 children, any estimate that just clears a threshold is on average too big (the "winner's curse"), so the notes lead with the interval, not the point.

## Causal versus association — read this before any coefficient

Randomisation is what licenses a causal claim. In this suite only a contrast anchored in randomisation is causal — in practice:

1. the **ITT effect τ** (intention-to-treat: the effect of being _assigned_ to the intervention; notes 01–02);
2. the **difference-in-differences t2 contrast** (the randomised gap at the first post-baseline wave; note 05); and
3. the **gain-factor on-intervention treatment marginal** (note 03).

The latent change-score family (note 10) also writes a randomised "window-1" contrast that is causally interpretable, but reports it only as a **consistency check**, not as its headline.

Everything else — every covariate coefficient (age, hearing, speech, baseline score, cognitive ability, other skills), every mechanism dose-response slope, every mediator→outcome path, every latent correlation, every dose slope, every aligned-cohort or block-exposure contrast, and any difference-in-differences or level quantity at _post-crossover_ waves — is an **adjusted association**. An association describes _who progresses_ ("children higher on X tend to gain more on Y, holding the adjustment set fixed"), not a lever you can pull. Reading a control variable's coefficient as if it were causal is the **Table-2 fallacy**, and the notes call it out. Residual confounding by latent general ability remains for all associations; the child random intercept pools stable individual differences but does not stand in for general ability. Adjustment sets are fixed in advance from the DAG, so a skill missing from a model was excluded by the diagram, not found unimportant.

## Convergence gate

Before any interpretation, each fit is checked against a gate: R-hat ≤ 1.01 (chains agree), effective sample size ≥ 400 (enough independent draws), BFMI ≥ 0.3 (the sampler explored well), and zero divergences. Of the 179 models, **162 pass the gate** and 17 are flagged for review. The 17 split into two clean groups, and each family note states each model's status:

- **Divergence-only (13 models: `did-007`, `dose-077/083/084/177`, `hs-001`, `mech-095/156/157/188/189/190/191`).** Everything else passes (R-hat ≤ ~1.003, ESS in the thousands); they fail only the zero-divergence criterion, with at most 31 divergences out of 36,000 draws (≤ 0.09%), well inside the ≤ 1% working guidance. These are the expected geometries — dose-slope and horseshoe funnels, and Gaussian-process mechanism surfaces — and are **usable with a note**.
- **Correlated-factor measurement models (4 models: `mm-001/002/101`, `rlm-mm-001`; note 14).** These fail R-hat and ESS — a latent-factor "funnel" geometry. Their **domain correlations are usable**, but their **structural coefficients are held** pending a non-centred reparameterisation.

## The families and the questions they answer

| Note | Family                                    | Models | Question it answers                                                                               | Causal?                        |
| ---- | ----------------------------------------- | ------ | ------------------------------------------------------------------------------------------------- | ------------------------------ |
| 01   | **ITT suite**                             | 27     | Did _assignment_ to the intervention change each skill? (the headline effect)                     | **Yes** (τ)                    |
| 02   | **Joint (multivariate ITT)**              | 4      | The suite outcomes fitted together; taught-vs-not-taught generalisation contrasts                 | **Yes** (τ per outcome)        |
| 03   | **Gain-factors (ANCOVA)**                 | 19     | Same effect via a period-stacked gain model; plus which baseline/ability/skills predict gains     | Yes (trt marginal) + assoc.    |
| 04   | **Level-factors**                         | 11     | How each skill's _level_ differs by group over time                                               | t2 only; rest assoc.           |
| 05   | **Difference-in-differences (crossover)** | 14     | The within-person randomised gap using the crossover design                                       | Yes (t2 δ) + assoc.            |
| 06   | **Aligned (per-protocol)**                | 9      | Onset-aligned ~40-week gain once both arms have been treated                                      | Association only               |
| 07   | **Mechanism**                             | 27     | Shape of one skill's association with another (e.g. letter-sounds → word reading); interactions   | Association only               |
| 08   | **Mediation (g-formula)**                 | 18     | How much of the reading gain runs _through_ letter-sounds vs other routes                         | Association (decomposition)    |
| 09   | **Dose-response**                         | 5      | Whether more sessions attended goes with more progress                                            | Association (dose = collider)  |
| 10   | **Latent change-score (LCSM)**            | 5      | Do earlier levels of one skill predict later _change_ in another?                                 | Association (+ window-1 check) |
| 11   | **Concurrent associations**               | 9      | Which contemporaneous skills co-vary with a focal outcome, mutually adjusted                      | Association                    |
| 12   | **Adjusted baseline predictors**          | 2      | Which baseline characteristics predict standing, adjusted vs one-at-a-time                        | Association                    |
| 13   | **Horseshoe (selection)**                 | 5      | Which predictors survive sparse Bayesian variable selection (cross-check of the boosting ranking) | Association                    |
| 14   | **Correlated-factor measurement**         | 5      | Latent domain correlations and measurement structure                                              | Association                    |
| 15   | **Growth curves**                         | 3      | Verbal/reading trajectories, and whether baseline ability predicts trajectory shape               | Association                    |
| 16   | **Block-2 exposure**                      | 4      | Staggered second-block exposure contrasts (parallel-trends assumption)                            | Association                    |
| 17   | **Survival (off-floor)**                  | 2      | Time-to-first success on heavily floored skills (spelling, nonword reading)                       | Association                    |
| 18   | **Historical (Byrne cohort)**             | 10     | Natural-history growth and domain structure in a separate, non-randomised cohort                  | Association                    |

Note 19 is a **cross-model summary** that reads the families together. The total is 179 models.

## The coherent story to expect (spoiler)

Reading the notes together: **strong-to-very-strong evidence of an ITT benefit on the code-related and directly-taught skills** — letter-sound knowledge (LS), phoneme blending (PA), word reading (WR), taught expressive vocabulary (TE) — tapering to moderate/suggestive on transfer measures, and **inconclusive and probably negligible on broad standardised vocabulary (RV, EV)**. The word-reading gain is **mediated by letter-sound knowledge**, not by the vocabulary route (note 08). The randomised result is echoed by the difference-in-differences and gain-factor re-analyses (notes 03, 05) and survives adjustment for ability, socio-economic status and study site (note 01). The associational families (notes 06–18) describe the developmental scaffolding — which skills track and appear to lead which — without over-claiming cause. Where an _unadjusted-level_ view (note 04) or a _non-randomised_ per-protocol view (note 06) shows a negative blip on standardised vocabulary, it is most consistent with residual baseline imbalance on those noisy 170-item tests, not with harm — the baseline-differencing families are the authoritative read.
