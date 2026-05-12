---
title: "Predictors of progress in language and reading skills for children with Down syndrome"
subtitle: "Project review briefing"
author: "Frank Buckley, Down Syndrome Education International"
date: 2026-05-12
format:
  pdf:
    documentclass: scrartcl
    papersize: a4
    colorlinks: true
    fontsize: 11pt
    mainfont: "Source Sans 3"
    sansfont: "Source Sans 3"
    monofont: "Monaspace Neon"
    monofontoptions: "Scale=0.75"
    linestretch: 1.2
    toc: true
    toc-depth: 2
    number-sections: true
    number-depth: 2
    geometry:
      - top=25mm
      - left=22mm
      - right=22mm
      - bottom=25mm
---

::: {.callout-warning}
This was drafted by digital assistants using LLMs that may make mistakes in ways that differ from other inference machines.
:::

**For:** project review meeting
**Date:** 2026-05-12
**Status:** work in progress; all numbers preliminary

## The question

We have data from a longitudinal study of children with Down syndrome who took part in the **Reading and Language Intervention (RLI)** trial. The long dataset currently contains **54 children**, each assessed at **four timepoints**, on a battery of language and reading measures. Individual model sample sizes are smaller where particular measures are missing.

We want to answer two related questions:

- **What predicts progress** in language and reading for these children — which baseline skills, demographics, or programme features matter, and how much?
- **What did the intervention itself change** — for which skills, by how much, and is the evidence convincing given the small sample?

## How we've approached it (plain English)

We use a deliberate **two-step method**. The two steps answer different questions, and neither is sufficient on its own.

**Step 1 — Discovery: which predictors matter?**
We fit *gradient-boosting models* — flexible "learn-the-pattern" models that automatically combine many possible predictors and rank them by how much each one helps the model's predictions. Think of it as an automated sieve over ~30 candidate predictors per outcome. For the more mature LRP01-LRP10 suite this has already pruned models down to 6–17 informative predictors; the newer LRP11-LRP22 models are currently full-predictor, MAE-tuned baselines that still need the same feature-selection pass. We are careful to test the models on children the model hasn't seen, so we don't fool ourselves with within-sample fit.

This step gives us **rankings and importances**, but it does *not* tell us how certain we should be about any one effect, and it can't cleanly separate causes from correlations.

**Step 2 — Inference: how certain are we, and what is the size of the effect?**
We take the predictors that survived Step 1 and fit *Bayesian statistical models*. These produce, for every quantity we care about, a **range of plausible values** ("we are 95% sure the true effect lies between X and Y") rather than a single point estimate. They also let us ask cleaner causal questions — for example, whether letter-sound knowledge is a plausible pathway from intervention to word reading — by being explicit about which other variables we adjust for. A formal mediation decomposition is a separate future analysis.

We currently have **22 discovery models** (LRP01–LRP22) covering eleven outcome pairs — level (the score at a timepoint) and gain (change between timepoints) — and **seven Bayesian inference models** (LRP52–LRP58).

## Headline findings — effect of the intervention

For each outcome, the Bayesian model gives us a probability that the intervention raised the score. We can read this as a continuous measure of evidence, not a yes/no verdict. The conventional "credible" bar is 95%; effects sitting below that bar can still lean strongly in one direction.

| Outcome | Best estimate of effect | 95% credible interval | P(intervention raised it) |
|---|---:|---|---:|
| **Letter-sound knowledge** | **+0.63 logit** | (+0.26, +0.98) | **≈ 99.9%** |
| **Word reading** | **+0.41 logit** | (+0.05, +0.76) | **≈ 98.7%** |
| Phoneme blending | +0.42 logit | (−0.03, +0.86) | ≈ 96.7% |
| Basic concepts (CELF) | +0.23 logit | (−0.19, +0.64) | ≈ 86% |
| Receptive grammar (TROG) | +0.12 logit | (−0.16, +0.40) | ≈ 81% |
| Expressive vocabulary | +0.07 logit | (−0.11, +0.24) | ≈ 78% |
| Receptive vocabulary | +0.06 logit | (−0.12, +0.25) | ≈ 75% |
| Phonetic spelling | +0.09 logit | (−0.61, +0.78) | ≈ 60% |

(Sign convention: numbers in this table are flipped relative to the raw model output so that **positive numbers mean the intervention raised the score**. The repository's internal convention is the opposite; the magnitudes and intervals are unchanged.)

**A note on "logit" units.** Outcomes here are bounded counts (so many items correct out of so many). We model them on the *logit* scale to respect the bounds, but there is no fixed conversion from logit units to percentage points. The same logit shift is largest in percentage-point terms for a child near 50% correct and smaller near the floor or ceiling. For orientation, a +0.4 logit shift is about +3 percentage points if the starting score is near 8% correct, but about +10 percentage points near 50% correct. Letter-sound knowledge is near the middle of its scale in this sample, so its +0.63 logit estimate is materially larger on the raw-score scale than the word-reading estimate. The comparison that matters is whether the intervention arm gained more than the wait-list arm over the same period, not a single universal percentage-point conversion.

### How to read the table for a meeting audience

- Only **letter-sound knowledge** and **word reading** clear the conventional 95% bar by a comfortable margin.
- **Phoneme blending** sits just below the bar (97%) and would normally be reported alongside the two credible effects, with the caveat that its interval just touches zero.
- Four further outcomes — basic concepts, receptive grammar, expressive and receptive vocabulary — all lean toward "intervention helped" with **75–86% posterior probability**. With only 54 children in the long data and slightly smaller effective samples in some fits, an effect in this range is exactly what we would expect to be unable to confirm even if it were genuine. The right reading is **"plausibly helpful but unconfirmed"**, not "no effect".
- **Phonetic spelling** is the only outcome where the data are genuinely close to silent (60% : 40% with a very wide interval). Treat this as "we do not yet know".

### Important caveats

- **The intervention does target vocabulary and grammar** as well as reading. The non-credible outcomes are about whether the measured scores moved enough to be confirmed in this dataset, not about what the intervention was designed to do.
- The sample is small (54 children in the long data; effective n varies by model). Effect sizes are estimated with substantial uncertainty across the board.

## Why letter-sound knowledge stood out

Letter-sound knowledge only emerged as the *largest* treatment effect once we fit a **joint model** that estimates all eight outcomes simultaneously. Looking at outcomes one at a time underplayed it.

We then asked whether letter-sound knowledge is a plausible **candidate pathway** for the word-reading effect — i.e. whether the intervention raises letter-sound knowledge, and whether letter-sound knowledge is independently associated with word reading. A mechanism model (LRP58) supports this interpretation: letter-sound knowledge has a credibly positive association with word reading, even after adjusting for other relevant skills. This is compatible with mediation, but it is not yet a formal mediation decomposition of how much of the word-reading gain runs *through* letter-sound knowledge versus other paths.

## Phonics skills, word learning and whole-word-compatible reading

One of the key substantive questions is whether the intervention's word-reading gains look mainly **phonics-mediated** — children learn letter-sound and blending skills, which then support word reading — or whether some children learn to read words through more **lexical / whole-word-compatible strategies**.

The current design cannot isolate "phonics teaching" from "whole-word teaching" causally, because the reading strand combined book reading, sight-word learning and revision, letter-sound teaching, phonological-awareness games, and linking letters with sounds. The language strand also explicitly taught vocabulary. The ingredients were not independently randomised.

Even so, the existing results can say more than the current briefing has said:

- The Bayesian models support a phonics-route signal: the intervention credibly raised **letter-sound knowledge** and **word reading**, blending is suggestive, and LRP58 shows letter-sound knowledge is independently associated with word reading.
- The word-reading discovery model also points toward a decoding route: phonetic spelling (`spphon`) and letter-sound knowledge (`yarclet`) are the top two predictors of word-reading level, with nonword reading (`nonword`) also in the top five.
- The word-reading gain model is mixed: letter-sound knowledge and blending remain in the final predictor set, but so do attendance and taught expressive vocabulary (`b1exto`), suggesting that reading gains may reflect both phonics-related and broader language/lexical learning.
- Burgoyne et al. reported intervention gains on single-word reading, letter-sound knowledge, phoneme blending and taught expressive vocabulary, but little transfer to nonword reading or spelling. That pattern is compatible with word learning moving ahead of fully generalised decoding.
- A simple descriptive profile check tells the same story: at _t_3, 48 / 54 children scored above zero on real-word reading, but 23 of those still scored zero on nonword reading and 25 scored zero on phonetic spelling. That does **not** prove whole-word strategy use, but it is compatible with real-word reading emerging before measurable decoding and spelling for many children.

The most defensible interpretation is therefore **mixed routes**. Some of the word-reading effect probably runs through letter-sound / decoding skills; some may reflect item-specific or lexical learning. The next phase should quantify that split rather than describe the intervention effect as purely phonics-mediated.

## What the discovery (Step 1) work added

- Using pooled out-of-fold CV R² (rather than the unstable mean of per-subject-fold R²), the best-predicted level outcomes are now **APT expressive information** (`aptinfo`, ≈ 0.80), **expressive vocabulary** (`eowpvt`, ≈ 0.69), **APT expressive grammar** (`aptgram`, ≈ 0.67), **letter-sound knowledge** (`yarclet`, ≈ 0.64), **receptive vocabulary** (`rowpvt`, ≈ 0.63) and **word reading** (`ewrswr`, ≈ 0.60). Several word-reading level variants are similar or slightly stronger (≈ 0.63–0.65).
- Predicting *gains* (change between timepoints) is intrinsically much harder than predicting levels — the discovery models explain roughly 7–24% of variation in gains on the same pooled out-of-fold basis. This is partly substantive (gains are noisy) and partly a feature of the small sample. The Bayesian models still find treatment signal where the discovery models cannot — they trade off model complexity for a single clean question.
- A recurring structural finding: when both halves of a paired test (e.g. expressive grammar information / grammar production, or DEAP initial/final articulation) are available as predictors, the partner score often absorbs much of the variance. This argues for treating these as joint constructs rather than independent predictors in future modelling.

## What we have *not* yet done

These are explicit, named gaps — flagged so the meeting can decide which are priorities:

1. **SES adjustment beyond a first word-reading sensitivity.** LRP60 now adds mother's education, father's education and age first read to the word-reading ITT model. The adjusted estimate still supports an intervention benefit (+0.62 logit, 95% interval +0.10 to +1.14; P ≈ 99.0%), but it is a complete-case analysis with only 33 children because 21 / 54 rows are dropped for missing pre-scores, SES/home-literacy covariates or group. We still need a matched complete-case unadjusted comparison and/or principled imputation before treating SES robustness as closed.
2. **Intervention-fidelity outcomes.** Specifically-taught vocabulary items (Block 1 expressive/receptive taught) are a "positive control": if the intervention does not visibly raise the words it explicitly taught, that would prompt careful checks of the data, scoring, and pipeline. We have not yet modelled these.
3. **Reading-route analysis.** We have not yet quantified how much of the word-reading effect looks phonics-mediated versus lexical / whole-word-compatible. This should use letter-sound knowledge, blending, nonword reading, phonetic spelling, taught vocabulary and the repetition measures together rather than treating any single measure as the pathway.
4. **Formal mediation decomposition.** Splitting the word-reading effect into a direct part and a part that flows through letter-sound knowledge. Out of scope for the current Bayesian models by design; ready to be picked up next.
5. **Teaching-practice heterogeneity and sequencing.** We have not yet asked whether different baseline profiles imply different teaching routes: for example, which children show the strongest phonics-linked response, whether vocabulary/language supports real-word learning beyond decoding, whether floor/ceiling position limits observed gain, or whether reading progress feeds back into later language progress.
6. **Data-dictionary confirmations.** A handful of "what is the maximum possible score on this measure?" questions need confirming before some of the planned models can be reported on the probability scale.

## Options for next steps

Six broad directions, listed in roughly increasing order of ambition. They are not exclusive: A and B are near-term defensibility work, C is the central substantive reading-route question, D and E deepen the mechanism story, and F is a deliberately selective queue of teaching-practice questions to keep in view.

### A. Consolidate what we already have

**Why.** The three small Bayesian-suite defects noted in the April write-up have now largely been closed out in code and notes. LRP55 now defaults to no LKJ residual block and no age Gaussian process, eliminating the earlier divergence pathology; the mechanism comparison plot now uses each model's real estimated grid; and the treatment-effect forest plot has been implemented. Consolidation is therefore less about new modelling and more about making sure the visible artefacts and reports all reflect the current default fits.

The remaining housekeeping is still worth doing before an external write-up:

- Regenerate or restore the `output/statistical_models/` artefacts in the current working tree, since the notes refer to reporting-run outputs that are not present locally.
- Re-render the Bayesian model reports and comparison outputs so the published artefacts use the current LRP55 effect sizes and the current mechanism-model defaults.
- Check the open dependency-bump PR (#56), which relaxes the diagnostics library constraint from ArviZ 0.x to allow 1.x. Library calls in the diagnostic pipeline need to be checked against the new API before merging.

**What is gained.** A clean, internally consistent set of reports; one fewer "we know about this issue" caveat to carry into any external write-up; confidence that the toolchain is current.

**What is involved.** A short consolidation pass: regenerate the statistical artefacts, re-render the reports and comparison plots, update any stale summary text, and run the relevant tests/lints against the ArviZ upgrade. No new modelling decisions and no new data assumptions.

### B. Socio-economic-status (SES) robustness check (LRP60)

**Why.** Across the discovery (Step 1) models, parental education (mother's and father's years of post-16 education) and home literacy environment (the age at which the child was first read to) are repeatedly selected as informative predictors. Randomisation should break this confounding on average, but with only 54 children even mild arm imbalance could matter, and SES adjustment is a useful robustness check.

**Current first pass.** LRP60 has now been implemented as a complete-case word-reading ITT sensitivity using linear standardised adjustment terms for `mumedupost16`, `dadedupost16` and `agebooks`. The intervention effect remains clearly in the beneficial direction and is larger than the unadjusted full-sample LRP52 estimate (+0.62 versus +0.40 logit, using the positive-is-benefit sign convention). The SES coefficients themselves are all close to zero with intervals spanning zero.

The caveat is substantial: requiring all three covariates drops 21 / 54 children, leaving 33. LRP60 therefore answers "does the word-reading effect survive SES adjustment among complete cases?", not yet "is the original full-sample estimate fully robust to SES?". LOO also flags 2 influential points (Pareto _k_ > 0.70), which is not surprising at this sample size but argues against over-reading the exact magnitude.

**What is gained.**

- A direct check on whether the credible effects on word reading and letter-sound knowledge survive adjustment for SES. The first word-reading check is reassuring on direction, but not yet final because of complete-case missingness.
- Potentially tighter credible intervals on the surviving effects, because SES may absorb some residual between-child variation.
- A model the journal-reviewer-in-our-heads will ask for anyway.

**What is involved.**

- Add a matched complete-case unadjusted comparator so we can separate the effect of SES adjustment from the effect of dropping children with missing covariates.
- Decide whether to keep complete-case adjustment as a sensitivity only, or add simple Bayesian imputation for the SES / home-literacy covariates.
- Extend the same adjustment pattern to letter-sound knowledge and the other ITT outcomes only after the missing-data decision is made.

### C. Reading-route analysis: phonics-mediated versus whole-word-compatible reading

**Why.** This is the key substantive question for the reading strand: did children learn words mainly because phonics skills improved, or did some children learn real words through lexical / whole-word-compatible strategies even when decoding remained weak?

The existing evidence is mixed in an informative way. Letter-sound knowledge stands out in the joint ITT model, and the word-reading level model gives high importance to phonetic spelling, letter-sound knowledge and nonword reading. At the same time, many children can read at least some real words while scoring zero on nonword reading or phonetic spelling, and the original trial report found little transfer to those more decoding-dependent outcomes.

**What is gained.**

- A clearer answer to whether the reading effect is mainly carried by phonics-related skills, item-specific word learning, or both.
- A child-level description of **whole-word-compatible profiles**: children whose real-word reading is stronger than their decoding measures would predict.
- A more honest bridge from the intervention result to mechanism claims. Instead of saying "letter-sound knowledge mediates word reading" too broadly, we can ask how far letter-sound knowledge, blending, nonword reading and spelling account for real-word gains.

**What is involved.**

- Start with a descriptive profile analysis: by timepoint, count children with real-word reading above zero but nonword reading / phonetic spelling at floor; then track whether those profiles persist or resolve.
- Fit a "phonics-route" model for word reading using letter-sound knowledge, blending, nonword reading, phonetic spelling and the Early Repetition Battery scores (`erbword`, `erbnw`), with baseline word reading and age included.
- Fit a complementary "lexical / whole-word-compatible" model: predict the part of word reading not explained by decoding measures, then ask whether taught vocabulary, expressive vocabulary, receptive vocabulary, age, attendance or other child characteristics explain that residual.
- Keep the causal caveat explicit: because phonics, sight-word practice and book reading were bundled in the intervention, this can identify route-compatible evidence, not a causal effect of phonics teaching versus whole-word teaching.

### D. Intervention-fidelity and broader mechanism phase (LRP61, LRP63)

Two connected model families support the reading-route question without replacing it.

**LRP61 — Did the intervention raise the words it specifically taught?**

*Why.* The RLI intervention includes a defined vocabulary teaching component with a specific list of target words. Whether the children who received the intervention scored higher on *those specific words* — separately from the broader vocabulary measures we've already looked at — is a different question and an important one. A null result on the taught words would be a serious prompt to check the data, scoring, and pipeline; a clear positive result is the "positive control" that strengthens everything else in the suite. It also quantifies *how steep* the teaching gradient is, which is the upstream input to all the broader vocabulary outcomes.

*What is gained.* A methodological reassurance about the pipeline; a quantified teaching gradient that contextualises the broader vocabulary nulls; potentially a clean strong-effect headline on a fidelity outcome.

*What is involved.* Confirm the maximum possible score on the two taught-word scales from the data dictionary (this is a small but real blocker), register them in the measures table, then re-use the existing ITT model machinery. The result should feed into the lexical / whole-word-compatible side of the reading-route analysis.

**LRP63 — Does speech articulation predict reading, after accounting for other language skills?**

*Why.* Articulation (measured by picture-naming sub-tests) was repeatedly selected by the discovery models, and the newer LRP21/LRP22 models show the DEAP articulation measures are tightly interrelated. That is a striking importance signal that does not yet have a Bayesian counterpart. The theoretical case is weaker than for phonological memory, but the empirical signal from Step 1 is strong enough that it should be examined explicitly rather than left implicit.

*What is gained.* Either a substantive new mechanism finding, or a clean null that quietens a question the discovery work raises. Either is useful.

*What is involved.* Articulation measures are percentage-scale rather than straightforward bounded counts, so the existing Beta-Binomial mechanism factory probably needs either a Gaussian-likelihood variant or a deliberate bounded-scale recoding. That is a contained piece of work, but it is real new model-class infrastructure, not just a configuration change.

**Standing caveat for the mechanism work.** The current mechanism models do not estimate a "pure" direct effect of the intervention on word reading because both arms of the RLI study are receiving the intervention during phases 1 and 2 — the group indicator is no longer a treatment contrast once you pool across phases. This is documented in each individual report and does not affect the mechanism association itself. It does, however, become central in option E.

### E. Formal mediation analysis (LRP64+)

**Why.** The current findings together suggest a story: the intervention raises letter-sound knowledge most; letter-sound knowledge is independently associated with word reading even after adjusting for other skills; the intervention also raises word reading. The natural follow-up question — *how much of the word-reading effect runs through letter-sound knowledge, and how much is direct?* — is not answered by any single model we have. It is what a paper-quality treatment of "how does this intervention work" would need.

**What is gained.** A defensible decomposition of the headline treatment effect on word reading into a **direct** path (the part of the effect that doesn't run through letter-sound knowledge) and an **indirect** path (the part that does). This is the analysis that earns the right to make causal-pathway statements rather than mediation-flavoured association statements.

**What is involved.** This is the methodologically most demanding option, for three reasons:

1. **Identification.** Formal mediation analysis requires explicit causal-graph assumptions about which variables confound the mediator-outcome relationship (and these are harder than the assumptions needed for the treatment effect alone). The DAG work done for LRP56–LRP58 is a start but does not directly identify a mediation parameter.
2. **Phase structure of the RLI trial.** As noted above, the group-assignment indicator stops being a treatment contrast once you leave the wait-list phase. A clean mediation decomposition needs to be restricted to the period where the two arms genuinely differ, which is a deliberate data sub-setting decision that has not yet been made formally.
3. **Estimand and estimator.** The standard mediation quantities ("natural direct effect" / "natural indirect effect") are not computed by the existing pipeline. Implementation is typically Bayesian g-formula or a similar counterfactual-style calculation. It is well-trodden methodologically but new to this codebase.

Realistically 2–4 weeks of focused work, with a substantial proportion of that spent on the identification assumptions rather than the code. The right framing for a meeting is "this is the option that turns the current results into a publishable mechanism story; it should not be started until A, B, C and D are in place."

### F. Additional teaching-practice questions for later Bayesian work

These are worth carrying in the model backlog, but they should be treated as **selective follow-ons**, not a licence to fit every possible interaction. With 54 children, each question needs a small, theory-led model rather than a broad fishing exercise.

| Question | Why it matters for theory / teaching | Sensible Bayesian analysis |
|---|---|---|
| **Which children benefit most from phonics-linked teaching?** | This is the differentiation question: some children may need more explicit phonics / phonological support, while others may profit first from vocabulary, sight-word or book-language work. | A small moderation model for the treatment effect on word reading and letter-sound knowledge, pre-specifying only a few baseline moderators: initial word reading, letter-sound knowledge, blending, oral vocabulary / grammar, age and perhaps speech or repetition scores. |
| **Is vocabulary a separate route into word learning?** | Whole-word-compatible learning is more plausible if broader oral vocabulary or taught vocabulary explains real-word reading beyond decoding. | Extend option C by modelling the part of word reading not explained by letter-sound knowledge, blending, nonword reading and spelling, then ask whether taught expressive/receptive vocabulary and broader vocabulary measures predict that residual. |
| **Do speech and repetition measures constrain the route, or just the measurement?** | Word repetition, nonword repetition, blending and articulation tasks all involve spoken responses. A child may look weak on "phonological" measures partly because of speech-production demands. | Jointly model repetition (`erbword`, `erbnw`), articulation, blending and reading outcomes, with a clear caveat that these are mixed memory/speech/phonological indicators rather than pure latent constructs. |
| **Are some gains hidden by floor or ceiling position?** | A teaching programme can look ineffective if many children are at floor on the measured skill, or if higher-scoring children have little room to gain. | Prefer bounded count models or baseline-conditioned growth models over raw change-score summaries; report predicted gains for low-, middle- and high-baseline children. |
| **Does reading progress support later language progress?** | Teaching practice often assumes language drives reading, but reading may also strengthen vocabulary, grammar and expressive information through text exposure. | Exploratory cross-lagged / multivariate growth models across the four timepoints, restricted to a few theoretically linked pairs such as word reading ↔ taught vocabulary and word reading ↔ expressive grammar / information. |
| **Does dose or attendance change the pattern of gains?** | If attendance mainly strengthens taught vocabulary or sight-word learning, that would suggest a different teaching gradient from a phonics-only account. | Dose-response models using `attend_cumul`, treated as observational rather than randomised evidence, and interpreted as sensitivity / implementation analysis rather than a causal treatment contrast. |

## Status note for the meeting

- The intense modelling burst that produced the current results ran 2026-04-12 to 2026-04-19. Since then, repository activity has been mostly infrastructure and dependency maintenance rather than new research-content modelling.
- One open dependency-bump pull request (#56) relaxes the ArviZ constraint to allow version 1.x and is worth checking before merging.
- All data, models, and conclusions remain **preliminary**. Nothing in this briefing has been peer-reviewed or externally validated.
