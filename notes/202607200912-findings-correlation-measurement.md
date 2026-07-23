# Correlation & measurement-structure findings (2026-07-20)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8). Direction of the adj-065 hearing association corrected 2026-07-22 (Claude Code/Fable 5).

This is one of the per-family companion notes to the findings index and reading guide (`notes/202607200900-findings-00-index-and-reading-guide.md`) — read that first for the study, the outcome measures and their maxima, and the house rules for reading a posterior. This note covers the **correlation and measurement-structure** family: 13 models in four sub-groups. **Every quantity in this note is an association or a description of how the skills go together — none is a causal effect.** The study's causal content lives in the randomised ITT/DiD/gain-factor families, which these models never touch.

## What this family does and the question it answers

The other families ask "what does the intervention cause?" and "who progresses?". This family asks a prior, structural question: **how do the skills relate to one another once we take measurement seriously?** Reading tests are noisy proxies for underlying skills, and children who are good at one thing tend to be good at others. These 13 models describe that web of relationships four different ways:

- **A. Latent correlation factors** treat each skill _domain_ (e.g. "vocabulary") as an unobserved trait measured by several tests, then report how strongly the domains correlate once test-specific noise is stripped out.
- **B. A longitudinal correlated-factor model** does the same across all four timepoints, and additionally splits each domain into a _stable trait_ part and a _wave-specific state_ part.
- **C. Concurrent-association models** ask, at a single timepoint, how one skill co-varies with the others _at that same moment_ (adjusted, and on its own).
- **D. Adjusted-association models** ask, one row per child, which baseline characteristics go with more word-reading gain.

A quick glossary for the terms used below (fuller versions in the index note):

- **Posterior / credible interval.** A Bayesian fit returns a full probability distribution (the _posterior_) for each quantity. We summarise it by its **median** and an **89% equal-tailed credible interval** — "89% posterior probability the value lies in this range" — plus, where available, the inner **50%** interval.
- **Direction and the evidence ladder.** Direction is read from the **tail probability** P(>0), not from whether an interval excludes zero and never from a p-value. Labels: **suggestive** ≥ 0.75, **moderate** ≥ 0.91, **strong** ≥ 0.97, **very strong** ≥ 0.99 (round odds 3:1, 10:1, 30:1, 100:1); below 0.75 is **inconclusive**. The label describes _evidence for a directional claim_, never the size of the effect. Orient it to the favoured direction: P(>0) = 0.03 is _strong evidence of a negative_ association.
- **Logit vs probability vs items.** Bounded test scores are modelled as a proportion-correct through a **logit** link (log-odds). A logit coefficient per standard deviation (SD) is hard to read, so we translate to a change in **items** (proportion × the test's maximum), e.g. "+0.38 logit/SD ≈ +4.2 of 79 words".
- **Association, not cause.** An adjusted coefficient answers "for two children alike on the adjustment set, how much does the outcome tend to differ per unit of this predictor?" Reading it as a lever is the **Table-2 fallacy**. Residual confounding by latent general ability remains throughout; where a model has a per-child **random intercept** (a child-specific baseline offset that pools stable individual differences) it soaks up stable heterogeneity but does _not_ stand in for general ability.

---

## A. Latent correlation factors (`kind=corr_factor`) — 4 models

**What a latent-factor correlation is.** Imagine "vocabulary" is a real but unmeasured trait. We observe two noisy tests of it (receptive R and expressive E). A _factor model_ posits the shared trait behind both, treats what each test does not share as measurement noise, and reports the correlation between the _cleaned-up traits_ rather than between the raw tests. Each test's **loading** is how tightly it tracks its factor; its **communality** is the share of the test's variance the factor explains (0 = all noise, 1 = pure signal). Because the noise is removed, latent correlations are usually _higher_ than raw test-to-test correlations (this is **disattenuation** — see sub-group B).

> [!WARNING]
> **All four of these models FAIL the convergence gate.** A latent-factor model at ~50–75 children produces a sampling "funnel" the sampler struggles to explore, so the chains do not mix well (r̂ above 1.01 and/or effective sample size ESS below 400). Following the project's standing rule for measurement models: the **domain correlations reported below remain usable as exploratory descriptions**, but the **structural/latent regression coefficients (the `beta_*` slopes onto an outcome) are HELD as not-yet-reliable pending reparameterisation.** Do not quote the structural slopes. The gate figures per model are tabulated so the fragility is explicit.

| Model          | Domains (indicators)                                                        | N   | Gate     | max r̂ | min ESS | divergences |
| -------------- | --------------------------------------------------------------------------- | --- | -------- | ----- | ------- | ----------- |
| lrp-rli-mm-001 | vocab (R,E) / code (L,B) / grammar (F,T) — word-reading-gain structural leg | 51  | **FAIL** | 1.019 | 354     | 1           |
| lrp-rli-mm-002 | same domains — errors-in-variables code→word-reading mechanism              | 51  | **FAIL** | 1.048 | 64      | 0           |
| lrp-rli-mm-101 | same domains — prior-sensitivity re-fit of mm-001                           | 51  | **FAIL** | 1.021 | 260     | 57          |
| lrp-rlm-mm-001 | Byrne historical wave 3: reading / language / memory / ability              | 75  | **FAIL** | 1.028 | 64      | 143         |

**Factor correlations (the usable headline), median [89% CI], P(>0):**

_lrp-rli-mm-001, mm-002, mm-101_ — three views of the same RLI vocabulary/code/grammar structure (mm-001 the reference; mm-002 recasts the code factor as an errors-in-variables predictor of word reading; mm-101 re-fits mm-001 under recalibrated loading/residual priors). The three agree closely, which is reassuring given the gate failures:

| Domain pair        | mm-001            | mm-002            | mm-101            |
| ------------------ | ----------------- | ----------------- | ----------------- |
| vocabulary–grammar | 0.80 [0.62, 0.92] | 0.79 [0.60, 0.91] | 0.80 [0.62, 0.92] |
| vocabulary–code    | 0.74 [0.50, 0.90] | 0.75 [0.53, 0.91] | 0.73 [0.49, 0.89] |
| code–grammar       | 0.67 [0.37, 0.87] | 0.67 [0.39, 0.87] | 0.65 [0.36, 0.86] |

Every pair has P(>0) ≈ 1.00. Read plainly: the three underlying skill domains are **strongly and positively intertwined**, with vocabulary–grammar the tightest coupling and code (letter-sounds + blending) the most distinct — but even code shares roughly two-thirds of a correlation with the language domains. The loadings behind mm-001 are strong for the vocabulary tests (communality ≈ 0.79–0.82) and grammar (≈ 0.54–0.57), and weaker for the code tests (L ≈ 0.41, B ≈ 0.39), i.e. letter-sounds and blending carry more test-specific noise than the vocabulary measures. **The `beta_*` structural slopes (code/grammar/vocabulary → word-reading gain, and the mm-002 code→word mechanism slope) are held and not reported here.**

_lrp-rlm-mm-001_ — the Byrne historical dataset (a separate cohort, 75 children, wave 3), four domains: reading (3 tests), language (2), memory (1), ability (3). Correlations are **very high across the board**, all P(>0) = 1.00:

| Domain pair      | r [89% CI]        |
| ---------------- | ----------------- |
| language–ability | 0.93 [0.87, 0.97] |
| memory–ability   | 0.91 [0.83, 0.96] |
| reading–ability  | 0.88 [0.82, 0.93] |
| language–memory  | 0.86 [0.76, 0.93] |
| reading–memory   | 0.80 [0.69, 0.88] |
| reading–language | 0.79 [0.70, 0.86] |

These say the four domains are nearly a single general-ability dimension in this historical cohort — ability correlates ≥ 0.88 with everything. As with the RLI models the structural leg is held; only the correlation structure is reported, and only as description.

---

## B. Longitudinal correlated-factor model (`kind=long_corr_factor`) — 1 model

**lrp-rli-lcf-001** — the same three RLI domains (vocabulary now R,E,TR,TE; code L,B; grammar F,T) fitted across **all four waves at once**, 216 rows from 54 children. **This model PASSES the gate cleanly** (r̂ = 1.001, min ESS ≈ 5,157, 0 divergences), so unlike sub-group A its estimates are usable — subject to the usual small-n / prior-dependence caution the report attaches to any four-wave latent model at this size.

**Per-wave latent skill correlations (the headline).** One correlation per domain pair per wave; all twelve have P(>0) = 1.00:

| Domain pair        | t1                | t2                | t3                | t4                |
| ------------------ | ----------------- | ----------------- | ----------------- | ----------------- |
| vocabulary–grammar | 0.86 [0.78, 0.92] | 0.85 [0.77, 0.91] | 0.86 [0.77, 0.91] | 0.87 [0.78, 0.92] |
| vocabulary–code    | 0.66 [0.53, 0.77] | 0.66 [0.52, 0.77] | 0.66 [0.52, 0.76] | 0.67 [0.53, 0.77] |
| code–grammar       | 0.56 [0.40, 0.69] | 0.57 [0.41, 0.70] | 0.56 [0.40, 0.69] | 0.56 [0.40, 0.69] |

The couplings are **essentially flat across the four waves** — the vocabulary–grammar bond stays ~0.86, vocabulary–code ~0.66, code–grammar ~0.56 throughout. There is no evidence the skills tighten or loosen their relationship as the study proceeds. The ordering matches sub-group A (vocabulary–grammar tightest, code most distinct).

**Trait vs state (what this model adds).** A child's score at any wave is split into a **trait** part (their stable standing, the same across all four waves) and a **state** part (wave-specific wobble). The `trait_share` is the fraction that is stable:

| Domain     | Trait share [89% CI] |
| ---------- | -------------------- |
| vocabulary | 0.95 [0.92, 0.98]    |
| grammar    | 0.95 [0.89, 0.99]    |
| code       | 0.93 [0.85, 0.99]    |

In plain terms: **~93–95% of each domain is a stable trait** — a child who is high on vocabulary at t1 stays high, and only ~5–7% of the variation is wave-to-wave state. These are highly persistent skills over the study window. (The model assumes equal dependence across all wave gaps — "compound symmetry"; genuine decay over time would be a reason to try an AR(1) alternative, but the diagnostics did not force that here.)

**Disattenuated correlations.** A **disattenuated** correlation is the latent (noise-removed) correlation set beside the plain observed correlation between the two domains' actual test scores. Classical measurement error usually _shrinks_ an observed correlation toward zero, so the latent value should be the larger. That is exactly what we see: in **all 12 wave/pair cells the latent correlation exceeds its observed counterpart** (gaps of about +0.10 to +0.26 — e.g. at t1 vocabulary–grammar is 0.86 latent vs 0.59 observed, +0.26). This is the signature of attenuation by measurement noise, and it is descriptive triangulation, not a pass/fail test (the observed comparator carries no interval).

**Items-scale latent slopes** (a tangible restatement of the couplings, per +1 item of the predictor test, at the average operating point; all P(>0) = 1.00, near-identical across waves):

- grammar→vocabulary (F→R): ≈ **+1.39 R-items per +1 F-item** [1.15, 1.66]
- grammar→code (F→L): ≈ **+0.81 L-items per +1 F-item** [0.55, 1.08]
- code→vocabulary (L→R): ≈ **+0.54 R-items per +1 L-item** [0.40, 0.68]

These are latent, adjusted associations (each conditions on the third domain) evaluated at one operating point — a comparison anchor for the sub-group C items-scale marginals, not a caused gain.

---

## C. Concurrent associations (`kind=concurrent`) — 6 models

**What a concurrent association is.** At one timepoint, for two children the same age and alike on the other skills, how much does the focal outcome tend to differ with a little more of each predictor skill _measured at that same moment_? It is a same-wave co-occurrence, **not** a claim that one skill causes another and **not** a change over time. Each predictor gets an **adjusted** figure (holding the other skills, age and an arm nuisance term fixed — the headline) and a **bivariate** figure (the predictor on its own, no age). The gap between them shows _sensitivity to what you condition on_, not a decomposition of shared variance (Table-2 fallacy applies).

All six models **PASS the gate** (max r̂ ≤ 1.001, min ESS ≥ ~24,000, 0 divergences), and for each model **all 28 published sub-fits converged** (0 failed). The pipeline nominates the wave with the largest complete sample as the diagnostic anchor — **t3** for all six — which is the wave reported below (per +1 SD of the predictor; items = adjusted average marginal effect per +1 SD, translated to the outcome's item scale).

**lrp-rli-ca-001 — outcome W (word reading, /79), t3:**

| Predictor             | adj logit/SD [89%]  | P(>0) | Label        | items/+1 SD [89%] |
| --------------------- | ------------------- | ----- | ------------ | ----------------- |
| L letter-sounds       | +0.38 [0.13, 0.62]  | 0.992 | strong       | +4.2 [1.4, 7.3]   |
| R receptive vocab     | +0.28 [0.02, 0.53]  | 0.959 | moderate     | +3.0 [0.2, 6.1]   |
| E expressive vocab    | +0.27 [−0.04, 0.57] | 0.918 | moderate     | +2.9 [−0.4, 6.6]  |
| TE taught expr. vocab | +0.19 [−0.08, 0.47] | 0.869 | suggestive   | +2.1 [−0.8, 5.4]  |
| B blending            | +0.17 [−0.07, 0.40] | 0.875 | suggestive   | +1.8 [−0.7, 4.5]  |
| TR taught rec. vocab  | −0.03 [−0.26, 0.19] | 0.402 | inconclusive | −0.3 [−2.5, 2.0]  |

Letter-sound knowledge is the standout same-wave correlate of word reading. Adjustment shrinks it from a bivariate +0.76 (logit/SD, P ≈ 1.0) to +0.38 — the other skills absorb roughly half the raw link — but it survives as the strongest adjusted concurrent partner of word reading.

**lrp-rli-ca-002 — outcome L (letter sounds, /32), t3:**

| Predictor             | adj logit/SD [89%]  | P(>0) | Label        | items/+1 SD [89%] |
| --------------------- | ------------------- | ----- | ------------ | ----------------- |
| W word reading        | +0.37 [0.12, 0.63]  | 0.991 | strong       | +2.1 [0.7, 3.3]   |
| TE taught expr. vocab | +0.29 [0.02, 0.57]  | 0.959 | moderate     | +1.7 [0.1, 3.0]   |
| B blending            | +0.28 [0.07, 0.50]  | 0.982 | strong       | +1.6 [0.4, 2.7]   |
| TR taught rec. vocab  | +0.09 [−0.15, 0.33] | 0.715 | inconclusive | +0.5 [−0.9, 1.8]  |
| E expressive vocab    | +0.07 [−0.22, 0.36] | 0.645 | inconclusive | +0.4 [−1.3, 2.0]  |
| R receptive vocab     | +0.03 [−0.22, 0.27] | 0.565 | inconclusive | +0.1 [−1.3, 1.5]  |

Word reading and blending are letter-sound's strongest adjusted partners — the code cluster (L, B) and word reading hang together, while broad vocabulary (R, E) adds little once code is in the model.

**lrp-rli-ca-003 — outcome TR (taught receptive vocab, /24), t3:**

| Predictor             | adj logit/SD [89%]  | P(>0) | Label        | items/+1 SD [89%] |
| --------------------- | ------------------- | ----- | ------------ | ----------------- |
| TE taught expr. vocab | +0.28 [0.09, 0.47]  | 0.990 | strong       | +1.3 [0.4, 2.2]   |
| R receptive vocab     | +0.19 [0.03, 0.36]  | 0.968 | moderate     | +0.9 [0.1, 1.7]   |
| B blending            | +0.14 [−0.02, 0.29] | 0.922 | moderate     | +0.7 [−0.1, 1.4]  |
| E expressive vocab    | +0.12 [−0.10, 0.33] | 0.806 | suggestive   | +0.6 [−0.5, 1.6]  |
| L letter-sounds       | +0.04 [−0.13, 0.22] | 0.655 | inconclusive | +0.2 [−0.7, 1.0]  |
| W word reading        | −0.05 [−0.23, 0.14] | 0.340 | inconclusive | −0.2 [−1.2, 0.7]  |

Taught receptive vocabulary tracks the other vocabulary measures (its taught-expressive twin TE, and broad receptive R) rather than the code/reading skills.

**lrp-rli-ca-004 — outcome TE (taught expressive vocab, /24), t3:**

| Predictor            | adj logit/SD [89%]   | P(>0) | Label               | items/+1 SD [89%] |
| -------------------- | -------------------- | ----- | ------------------- | ----------------- |
| E expressive vocab   | +0.48 [0.28, 0.68]   | 1.000 | very strong         | +2.5 [1.4, 3.5]   |
| TR taught rec. vocab | +0.27 [0.11, 0.43]   | 0.997 | strong              | +1.4 [0.6, 2.2]   |
| L letter-sounds      | +0.19 [0.02, 0.37]   | 0.964 | moderate            | +1.0 [0.1, 1.9]   |
| W word reading       | +0.07 [−0.11, 0.25]  | 0.743 | inconclusive        | +0.4 [−0.5, 1.3]  |
| R receptive vocab    | −0.04 [−0.22, 0.14]  | 0.350 | inconclusive        | −0.2 [−1.1, 0.7]  |
| B blending           | −0.19 [−0.35, −0.02] | 0.032 | moderate (negative) | −0.9 [−1.7, −0.1] |

Taught expressive vocab is anchored by broad expressive vocab (E, very strong) and its taught-receptive twin. Note **blending shows moderate evidence of a _negative_ adjusted association** here (P(>0) = 0.032, i.e. P(<0) = 0.968): among children matched on the other skills, more blending goes with slightly less taught-expressive vocab. This is a conditional curiosity to flag, not a causal signal — the classic Table-2 trap.

**lrp-rli-ca-005 — outcome R (receptive vocab, /170), t3:**

| Predictor             | adj logit/SD [89%]  | P(>0) | Label        | items/+1 SD [89%] |
| --------------------- | ------------------- | ----- | ------------ | ----------------- |
| E expressive vocab    | +0.18 [0.05, 0.32]  | 0.982 | strong       | +6.0 [1.4, 10.9]  |
| TR taught rec. vocab  | +0.11 [0.02, 0.21]  | 0.966 | moderate     | +3.7 [0.5, 7.0]   |
| W word reading        | +0.10 [−0.01, 0.21] | 0.924 | moderate     | +3.2 [−0.4, 7.0]  |
| L letter-sounds       | +0.00 [−0.11, 0.11] | 0.517 | inconclusive | +0.1 [−3.2, 3.6]  |
| B blending            | −0.01 [−0.11, 0.09] | 0.445 | inconclusive | −0.3 [−3.3, 2.9]  |
| TE taught expr. vocab | −0.03 [−0.16, 0.10] | 0.362 | inconclusive | −0.9 [−4.7, 3.3]  |

**lrp-rli-ca-006 — outcome E (expressive vocab, /170), t3:**

| Predictor             | adj logit/SD [89%]  | P(>0) | Label        | items/+1 SD [89%] |
| --------------------- | ------------------- | ----- | ------------ | ----------------- |
| TE taught expr. vocab | +0.21 [0.11, 0.32]  | 0.999 | very strong  | +6.4 [3.1, 9.8]   |
| R receptive vocab     | +0.12 [0.02, 0.23]  | 0.975 | strong       | +3.6 [0.6, 6.8]   |
| W word reading        | +0.08 [−0.03, 0.19] | 0.880 | suggestive   | +2.3 [−0.8, 5.5]  |
| B blending            | +0.05 [−0.05, 0.14] | 0.786 | suggestive   | +1.3 [−1.3, 4.1]  |
| L letter-sounds       | +0.03 [−0.07, 0.14] | 0.682 | inconclusive | +0.9 [−2.0, 4.0]  |
| TR taught rec. vocab  | +0.02 [−0.08, 0.12] | 0.645 | inconclusive | +0.6 [−2.1, 3.4]  |

For both broad vocabulary outcomes (R, E) the strongest adjusted partner is the _other_ broad vocabulary measure and its taught twin — the two vocabulary systems co-move, largely independently of the code skills. **A caution specific to R and E:** their _bivariate_ (unadjusted, age-free) associations are near zero or mildly negative (P(>0) ≈ 0.20–0.35 across predictors), then turn positive once the adjusted fit conditions on age. This sign flip is driven by adjustment (age in particular) — a reason to read the adjusted column, and a textbook reminder that a bare bivariate co-movement can mislead.

---

## D. Adjusted associations (`kind=adjusted`) — 2 models

**Concurrent vs adjusted — the difference.** Sub-group C is _within a timepoint_ (skills measured at the same moment). Sub-group D is **one row per child, between children**: it relates a child's _baseline_ characteristics to how much word reading they _gain_, holding the other baseline predictors fixed. Both are adjusted associations; C answers "which skills sit together right now?", D answers "which starting characteristics go with more subsequent gain?". Neither is causal.

Both models **PASS the gate** (r̂ ≤ 1.001, min ESS ≥ ~9,600, 0 divergences).

**lrp-rli-adj-065 — word-reading gain (W, /79), 51 children.** Baseline predictors, per +1 SD, adjusted (holding the rest fixed); `words` = translation to word items:

| Predictor           | adj logit/SD [89%]   | P(>0) | Label                 | words [89%]       |
| ------------------- | -------------------- | ----- | --------------------- | ----------------- |
| Hearing status      | +0.21 [0.05, 0.39]   | 0.978 | strong                | +2.4 [0.5, 4.5]   |
| Age                 | −0.32 [−0.50, −0.14] | 0.003 | strong (negative)     | −2.9 [−4.3, −1.3] |
| Language composite  | +0.16 [−0.10, 0.41]  | 0.840 | suggestive            | +1.7 [−1.0, 4.9]  |
| Letter sounds       | +0.16 [−0.05, 0.37]  | 0.888 | suggestive            | +1.7 [−0.5, 4.3]  |
| Behaviour           | −0.17 [−0.36, 0.00]  | 0.059 | suggestive (negative) | −1.7 [−3.3, 0.0]  |
| Non-verbal MA       | +0.04 [−0.15, 0.24]  | 0.647 | inconclusive          | +0.5 [−1.5, 2.6]  |
| Speech production   | +0.05 [−0.21, 0.30]  | 0.628 | inconclusive          | +0.5 [−1.9, 3.5]  |
| Blending            | +0.02 [−0.14, 0.19]  | 0.590 | inconclusive          | +0.2 [−1.4, 2.1]  |
| Phonological memory | −0.01 [−0.26, 0.24]  | 0.473 | inconclusive          | −0.1 [−2.4, 2.7]  |

(Two missingness-indicator terms — hearing-missing and speech-missing — are nuisance controls, not substantive predictors.) The clearest patterns: **younger children gain more** (age, strong negative — an age-at-baseline association, not a claim that ageing reduces learning), and **the baseline hearing-status flag goes with more word-reading gain** (strong positive). Mind the coding: `hs` = 1 marks a child with **impaired hearing or repeated ear infections** (0 = clear), so the positive sign means the _risk flag_ — not better hearing — accompanies more measured gain, a counter-intuitive direction (plausibly the coarse flag, resolving ear infections, extra support, or small-sample noise) that should not be read as hearing difficulty helping. Baseline letter sounds and the language composite lean positive but only suggestively once mutually adjusted; the bivariate letter-sound link (+0.29, P ≈ 0.99) is roughly halved by adjustment, so much of its raw signal is shared with the other predictors. Behaviour leans negative (suggestive). The hearing association is **robust to adding SES** (in the 39-child SES-adjusted re-fit hearing stays +0.27, P = 0.990, while the SES term itself is small and negative, P = 0.12), so hearing is not merely proxying for socioeconomic status.

**lrp-rlm-adj-001 — Byrne historical word reading (basread), waves 1–3, 69 children.** Baseline predictors, per +1 SD, adjusted; `words` translation:

| Predictor                       | adj logit/SD [89%]   | P(>0) | Label             | words [89%]        |
| ------------------------------- | -------------------- | ----- | ----------------- | ------------------ |
| Age (months)                    | −0.38 [−0.58, −0.18] | 0.001 | strong (negative) | −8.4 [−12.6, −4.0] |
| Recall of digits (memory)       | +0.21 [−0.04, 0.47]  | 0.908 | suggestive        | +4.6 [−0.9, 10.1]  |
| Verbal reasoning (similarities) | +0.10 [−0.14, 0.34]  | 0.744 | inconclusive      | +2.2 [−3.2, 7.6]   |
| Number skills                   | +0.02 [−0.27, 0.30]  | 0.531 | inconclusive      | +0.3 [−6.1, 6.6]   |
| Receptive vocab (BPVS)          | +0.00 [−0.20, 0.21]  | 0.512 | inconclusive      | +0.1 [−4.4, 4.7]   |
| Receptive grammar (TROG)        | −0.09 [−0.33, 0.15]  | 0.263 | inconclusive      | −2.1 [−7.3, 3.2]   |

The dominant pattern is the same **younger-gain-more** age association (strong negative, ≈ −8 words). Among the cognitive predictors, short-term memory (recall of digits) is the only one leaning positive (suggestive, P = 0.91); verbal reasoning is weakly positive and the rest are inconclusive. This historical cohort echoes the RLI finding that age is the clearest between-child correlate of gain, with an additional hint that verbal short-term memory matters.

---

## What this family concludes

The measurement models draw a consistent picture of a **tightly inter-correlated skill web**: in the RLI cohort vocabulary and grammar are the closest pair (latent r ≈ 0.80–0.86), the code skills (letter-sounds, blending) are the most distinct but still strongly linked (r ≈ 0.56–0.74), and in the Byrne historical cohort the four domains collapse almost onto a single general-ability dimension (r up to 0.93). The longitudinal model adds that these domains are **highly stable traits** (≈ 93–95% trait share) whose couplings barely change across the four waves, and that raw test correlations understate the true skill correlations by roughly 0.10–0.26 (attenuation).

The concurrent and adjusted models sharpen _which_ skills sit together: **letter-sound knowledge is the strongest concurrent partner of word reading**, the code cluster (letter-sounds + blending + word reading) coheres somewhat apart from the vocabulary cluster (broad + taught vocabulary co-moving), and **between children, younger age and the baseline impaired-hearing/ear-infection flag are the clearest correlates of word-reading gain** — the hearing link counter-intuitively positive (the risk flag, not better hearing, goes with more gain; see sub-group D) and surviving SES adjustment.

How this triangulates with the rest of the suite: this family supplies the _descriptive backdrop_ against which the causal families are read. The strong skill correlations warn that any single-outcome effect sits inside a correlated system (a caution the joint/ITT models handle explicitly), and the concurrent letter-sound → word-reading association is the observational shadow of the decoding pathway that the mechanism and mediation families try to probe causally. **None of these numbers is a lever** — they say who progresses and what goes with what, not what to change.

## Caveats and convergence

- **Sub-group A (all four `corr_factor` models) fails the gate** (r̂ up to 1.048, ESS as low as 64, up to 143 divergences). Per project policy the **domain correlations are reported as usable exploratory descriptions**, but every **structural/latent regression slope is held** as not-yet-reliable pending reparameterisation (a non-centred / funnel-robust rewrite of the latent factors). Treat even the correlations as provisional and prior-sensitive at n = 51–75.
- **Sub-groups B, C, D all pass the gate** cleanly, and every one of the concurrent sub-fits converged. The longitudinal factor model, though it passes, remains fragile and prior-dependent at n = 54 across four waves; its trait/state split assumes equal dependence across wave gaps.
- **Small samples (~51–75 children)** mean point estimates that just clear a threshold are on average inflated (winner's curse) — lead with the intervals, which are wide.
- **All quantities are associations.** Residual confounding by latent general ability is unaddressed; the strong skill inter-correlations mean adjusted coefficients are especially sensitive to the conditioning set (the R/E bivariate-to-adjusted sign flip and the ca-004 negative blending coefficient are concrete reminders). Do not read any coefficient here as an effect of changing the predictor.
