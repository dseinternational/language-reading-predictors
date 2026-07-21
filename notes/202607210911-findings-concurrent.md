# Concurrent associations findings (2026-07-21)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

This is one of a set of per-family notes from the full 2026-07-21 re-fit of every Bayesian statistical model in the study (production `reporting` configuration, 6 chains × 6000 draws, 89% credible intervals). For the study background, the outcome measures and their item maxima, and — importantly — how to read Bayesian medians, credible intervals, tail probabilities and the evidence ladder, see the index and reading guide, [`notes/202607210900-findings-00-index-and-reading-guide.md`](202607210900-findings-00-index-and-reading-guide.md). This note is self-contained on the numbers but assumes that reading guide for the conventions. The concurrent-associations family has **9 models** (ca-001 to ca-009), and **all 9 pass the convergence gate**. Preliminary — small sample, exploratory study.

## What this family probes

These models describe the **concurrent skill structure**: within a single measurement wave, among children who are alike on the other skills measured at that same wave, does a child who scores higher on one skill also tend to score higher on the focal skill? In plain terms, "holding the other same-wave skills fixed, how does each skill relate to the focal outcome, at each wave?" It is the regression-style, mutually-adjusted version of the study's own correlation tables — a picture of which skills co-vary at a timepoint. There are nine focal skills, one model each, and each model estimates a separate slope at each of the four timepoints (t1–t4). The levels predictors used here are aligned with the gradient-boosting gains panel, and three focal skills — phoneme blending (PA), basic concepts (LF) and receptive grammar (RG) — were added to the original six.

Each model uses a Beta-Binomial likelihood on the bounded item counts through a logit-linear predictor, with no interaction, spline or nonlinearity term — so there is no "knee" or threshold to report, only per-wave slopes. Every coefficient is reported **adjusted** (holding every other same-wave skill fixed) and put on the raw-items scale as an average marginal effect: the expected change in the focal skill's item count per **+1 SD** (one standard deviation, i.e. one typical between-child step) of the predictor, computed separately at each wave.

**What it controls for, and why it is not causal.** Each adjusted coefficient conditions on the other contemporaneous skills at the wave. Because all children are on the programme from t2 onwards, those same-wave skills are themselves post-treatment quantities: conditioning on them would wreck a causal reading, but it is exactly right for pure description. So **every coefficient in this family is an association, by design** — there is no causal term anywhere in it. Two standard warnings apply throughout. First, the **Table-2 fallacy**: each adjusted coefficient answers its own "holding-the-rest-fixed" question, so the set of coefficients is not a single ranking of importance, and no control coefficient may be read as a lever. Second, **regression dilution and collinearity**: the predictors are noisy observed scores rather than the latent skills themselves, and the skills are mutually correlated, so adjusted associations are pulled toward zero relative to the underlying truth, and a coefficient can even flip sign once a collinear partner is held fixed — a suppression artefact, not an effect. Residual confounding by latent general ability remains for all of these associations.

## How to read these numbers

Every headline below is an **adjusted association**, not a causal effect: it says "children higher on the partner skill tend to be higher on the focal skill at this wave, holding the other same-wave skills fixed," and nothing about what drives what. The estimand is on the **items** scale — the expected change in the focal skill's raw item count per +1 SD of the partner skill.

- The point estimate is the posterior **median**; the uncertainty is the **89% equal-tailed credible interval** (a direct probability statement: given the data and priors, there is an 89% probability the true association lies in that range).
- Direction is read from the **tail probability** P(association > 0), never from whether an interval excludes zero and never as a p-value.
- The **evidence ladder** grades that probability: inconclusive (< 0.75), suggestive (≥ 0.75), moderate (≥ 0.91), strong (≥ 0.97), very strong (≥ 0.99). The label describes strength of evidence for the direction, not the size of the effect.
- Because this family is purely descriptive, **no region of practical equivalence (ROPE)** is defined and no "big enough to matter" verdict is issued — evidence is graded on direction only.
- Direction is not size: a high probability means the association is _probably positive_, not that it is large. With roughly 54 children the samples are small, so lead with the interval, not the point estimate.

For each model the ground truth surfaces the single **clearest** adjusted same-wave partner — the partner-and-wave combination with the highest posterior probability of a positive association (chosen by certainty of direction, not by size of coefficient). That is the headline reported per model below.

## Per-model findings

All nine focal models pass the convergence gate (R-hat ≤ 1.01, effective sample size ≥ 400, BFMI ≥ 0.3, zero divergences). The table gives each model's clearest adjusted same-wave partner, the wave at which it is clearest, the items-scale association per +1 SD with its 89% credible interval, the posterior probability of a positive association, and the evidence label.

| Model  | Focal skill                  | Clearest adjusted same-wave partner (wave) | Association per +1 SD (89% CI) | P(> 0) | Evidence    | Gate |
| ------ | ---------------------------- | ------------------------------------------ | ------------------------------ | ------ | ----------- | ---- |
| ca-001 | Word reading (WR)            | Letter sounds (t2)                         | +8.5 items (+5.9 to +11.1)     | 99.9%  | very strong | pass |
| ca-002 | Letter sounds (LS)           | Word reading (t2)                          | +4.1 items (+3.1 to +5.1)      | 99.9%  | very strong | pass |
| ca-003 | Taught receptive vocab (TR)  | Taught expressive vocab (t3)               | +1.3 items (+0.3 to +2.2)      | 98%    | strong      | pass |
| ca-004 | Taught expressive vocab (TE) | Expressive vocab (t3)                      | +2.3 items (+1.1 to +3.4)      | 99.9%  | very strong | pass |
| ca-005 | Receptive vocab (RV)         | Word reading (t3)                          | +4.6 items (+0.7 to +8.5)      | 97%    | strong      | pass |
| ca-006 | Expressive vocab (EV)        | Taught expressive vocab (t3)               | +5.8 items (+2.2 to +9.6)      | 99.5%  | very strong | pass |
| ca-007 | Phoneme blending (PA)        | Letter sounds (t3)                         | +0.7 items (+0.3 to +1.2)      | 99%    | very strong | pass |
| ca-008 | Basic concepts (LF)          | Taught receptive vocab (t1)                | +1.6 items (+0.8 to +2.5)      | 99.8%  | very strong | pass |
| ca-009 | Receptive grammar (RG)       | Taught receptive vocab (t3)                | +2.1 items (+1.0 to +3.2)      | 99.8%  | very strong | pass |

### The reading pair — word reading and letter sounds (ca-001, ca-002)

**ca-001 (word reading, WR).** The clearest adjusted same-wave partner is **letter-sound knowledge at t2**: per +1 SD of letter sounds, word reading is higher by **+8.5 items** (89% credible range +5.9 to +11.1), holding the other same-wave skills fixed. The posterior probability of a positive association is **99.9% — very strong evidence** that the two same-wave skills tend to be higher together. Association only; it conditions on post-treatment skills and describes co-occurrence, not a pathway. Gate: pass.

**ca-002 (letter sounds, LS).** Reciprocally, the clearest partner is **word reading at t2**: per +1 SD of word reading, letter sounds is higher by **+4.1 items** (+3.1 to +5.1). Probability positive **99.9% — very strong**. So word reading and letter-sound knowledge form the family's tightest, mutually clearest within-wave pair. Association only. Gate: pass.

### The vocabulary cluster (ca-003 to ca-006)

**ca-003 (taught receptive vocabulary, TR).** Clearest partner **taught expressive vocabulary at t3**: **+1.3 items** per +1 SD (+0.3 to +2.2), probability positive **98% — strong evidence**. This is the most weakly-associated focal skill of the family; the headline is a modest, direction-clear link within the taught-word pair. Association only. Gate: pass.

**ca-004 (taught expressive vocabulary, TE).** Clearest partner **broad expressive vocabulary at t3**: **+2.3 items** per +1 SD (+1.1 to +3.4), probability positive **99.9% — very strong**. The taught expressive words track the general expressive-vocabulary measure at the same wave. Association only. Gate: pass.

**ca-005 (receptive vocabulary, RV).** Clearest partner **word reading at t3**: **+4.6 items** per +1 SD (+0.7 to +8.5), probability positive **97% — strong evidence**. The interval is wide (the vocabulary measures are mutually collinear and the sample is small), which is why a several-item association only reaches "strong" rather than "very strong". Association only. Gate: pass.

**ca-006 (expressive vocabulary, EV).** Clearest partner **taught expressive vocabulary at t3**: **+5.8 items** per +1 SD (+2.2 to +9.6), probability positive **99.5% — very strong**. The general and taught expressive-vocabulary measures co-vary strongly within the wave. Association only. Gate: pass.

### The added focal skills — blending, basic concepts, receptive grammar (ca-007 to ca-009)

**ca-007 (phoneme blending, PA).** Clearest partner **letter sounds at t3**: **+0.7 items** per +1 SD (+0.3 to +1.2), probability positive **99% — very strong evidence**. Blending is a 10-item measure, so the items translation is small in absolute terms, but the direction is very clear: blending sits with letter-sound knowledge inside the same code-related cluster as word reading. Association only. Gate: pass.

**ca-008 (basic concepts, LF).** Clearest partner **taught receptive vocabulary at t1**: **+1.6 items** per +1 SD (+0.8 to +2.5), probability positive **99.8% — very strong**. Basic concepts (a CELF subtest) tracks receptive taught vocabulary at the baseline wave. Association only. Gate: pass.

**ca-009 (receptive grammar, RG).** Clearest partner **taught receptive vocabulary at t3**: **+2.1 items** per +1 SD (+1.0 to +3.2), probability positive **99.8% — very strong**. Receptive grammar co-varies with receptive vocabulary within the wave, as the language-comprehension cluster would predict. Association only. Gate: pass.

## What to take away

Within any single wave the skills cluster exactly as reading and language science would predict, and the concurrent structure is clean: eight of the nine focal skills have a very-clear same-wave partner (very strong evidence of a positive association), and the ninth (receptive vocabulary, ca-005) is only a step weaker at strong. **Word reading and letter-sound knowledge are the tightest, mutually clearest pair** (ca-001/ca-002); the **vocabulary measures cluster among themselves** — general with taught expressive vocabulary, and the taught receptive/expressive pair together (ca-003 to ca-006); and the three added focal skills fall where their theory places them — **phoneme blending with letter sounds** in the code-related cluster (ca-007), and **basic concepts and receptive grammar with receptive vocabulary** in the language-comprehension cluster (ca-008/ca-009). But every one of these is a **cross-sectional, mutually-adjusted association**, still confounded by latent general ability; the family models no growth or change and licenses no causal claim. It describes which skills co-occur in the cohort — useful for understanding the developmental scaffolding — and says nothing about what drives what. For the randomised intervention effects see the ITT findings; for the "which skill carries the effect" decomposition see the mediation findings.
