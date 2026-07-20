# Intention-to-treat (ITT) suite findings (2026-07-20)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

This note reports the full re-fit (production `reporting` configuration: 6000 draws × 6000 tune × 6 chains) of the **intention-to-treat (ITT) suite** — 35 models in three related groups: 25 single-outcome ITT models (11 core + 14 robustness/adjusted variants, of which two add new outcomes), 4 joint models, and 4 block-2 exposure models. It is a companion to the shared reading guide `notes/202607200900-findings-00-index-and-reading-guide.md`; read that first for the study background, the outcome measures, and the house reporting standard. Every figure here is traceable to a model's own output CSVs in `output/statistical_models/models/<id>-reporting/`. Work in progress; all data are preliminary.

## What this family does, and the question it answers

The study is a **waitlist-crossover randomised trial**: children with Down syndrome were randomly assigned either to receive a reading/phonics intervention immediately or to receive it after a wait, and were tested at four timepoints. Because assignment was random, the difference in outcomes between the assigned arms is a genuine **causal** effect of _being offered the intervention_ — that quantity is the **ITT effect**, written τ (tau).

The ITT suite answers the headline question directly and one outcome at a time: **did assignment to the intervention change this skill, and by how much?** Each single-outcome model estimates τ from the first randomised comparison, using each child's own baseline score and a linear age term only as _precision_ terms (they sharpen the estimate; they are not part of what identifies the effect — the causal identification comes from randomisation, so the adjustment set is empty). This is what the other families do not give you in one clean number: the difference-in-differences and gain-factor families re-derive the same effect through the crossover and through period-stacked gain models, and the mechanism/mediation families ask _how_ the effect propagates, but the ITT τ is the primary randomised estimate.

Three extensions sit alongside the core. **Robustness variants** re-estimate τ with an extra covariate added — general cognitive ability (a block-design score), family socio-economic status (SES), or study site — to check that the headline effect survives; the point of these is agreement, not novelty. **Joint models** fit several outcomes together in one graph, giving per-outcome effects on a common probability scale plus taught-versus-not-taught contrasts. **Block-2 exposure models** look at a second, later-taught block of vocabulary for which there is no randomised contrast — so those are _associations_, not causal effects.

## How to read these numbers (recap)

Full conventions are in the shared reading guide. In brief:

- The point estimate is the posterior **median** (the "posterior" is the full probability distribution for the quantity given data and priors; the median is preferred because it does not change when we re-express the effect from the model's logit/log-odds scale to the probability scale). Lead with the **interval**, not the point, because with ~54 children an estimate that just clears a threshold is on average inflated.
- Uncertainty is the **89% equal-tailed credible interval** ("89% posterior probability the value lies in this range"), with the inner **50%** interval as the "most-of-the-mass" range. 89%, not 95%, is the deliberate house standard.
- Direction is the **tail probability** — P(intervention helps) — read directly, never as a p-value. Its **evidence label** is fixed by round odds: _inconclusive_ (< 0.75), _suggestive_ (≥ 0.75), _moderate_ (≥ 0.91), _strong_ (≥ 0.97), _very strong_ (≥ 0.99). The label grades the _evidence for a directional claim_; it never describes the _size_ of the effect.
- **Size is a separate claim.** `P(benefit ≥ δ)` is the probability the effect clears a pre-specified minimally-important difference δ (in items); `P(in ROPE)` is the posterior mass inside a "practically equivalent to zero" band. A flat result is **inconclusive** and quantified by ROPE mass — never "null".
- Effects are modelled on a probability (proportion-correct) scale and translated to **items** by multiplying by the test maximum (given per row below). For the heavily-floored outcomes P and N the estimand is instead a **risk difference in percentage points** (see below).
- **Causal versus association.** Only the ITT τ (and the joint per-outcome τ, which is the same estimand) is causal here. Every covariate coefficient — baseline, age, ability, SES, site — is an _adjusted association_ and must not be read as a lever (the Table-2 fallacy). The block-2 exposure `delta` is an association under a parallel-trends assumption. Residual confounding by latent general ability remains for all associations.

## Per-model results

**Convergence gate.** All 35 models **pass** the convergence gate (R-hat ≤ 1.01, effective sample size ≥ 400, zero divergences): every model has 0 divergences, maximum R-hat ≤ 1.002, and minimum effective sample size in the thousands (lowest ≈ 4,600, for a block-exposure model). One important qualification applies only to the two floored outcomes P and N and is set out in the caveats.

### A1. Core single-outcome ITT suite (τ is CAUSAL)

| ID      | Outcome (max items)                     | N   | τ, probability scale — median [89% CI] | τ, items — median [89% CI]  | P(helps) | Evidence     | Gate  |
| ------- | --------------------------------------- | --- | -------------------------------------- | --------------------------- | -------- | ------------ | ----- |
| itt-001 | TR — taught receptive vocab (24)        | 54  | +0.057 [+0.008, +0.105]                | +1.37 [+0.19, +2.53]        | 0.968    | moderate     | pass  |
| itt-002 | TE — taught expressive vocab (24)       | 54  | +0.064 [+0.018, +0.111]                | +1.55 [+0.42, +2.67]        | 0.985    | strong       | pass  |
| itt-003 | UR — not-taught receptive vocab (12)    | 54  | +0.050 [−0.002, +0.103]                | +0.60 [−0.03, +1.24]        | 0.937    | moderate     | pass  |
| itt-004 | UE — not-taught expressive vocab (12)   | 54  | +0.026 [−0.029, +0.080]                | +0.31 [−0.35, +0.97]        | 0.773    | suggestive   | pass  |
| itt-005 | R — standardised receptive vocab (170)  | 54  | +0.001 [−0.022, +0.025]                | +0.23 [−3.75, +4.26]        | 0.539    | inconclusive | pass  |
| itt-006 | E — standardised expressive vocab (170) | 54  | +0.001 [−0.018, +0.020]                | +0.18 [−3.08, +3.48]        | 0.534    | inconclusive | pass  |
| itt-007 | L — letter-sound knowledge (32)         | 54  | +0.110 [+0.053, +0.166]                | +3.52 [+1.68, +5.32]        | 0.999    | very strong  | pass  |
| itt-008 | B — phoneme blending (10)               | 54  | +0.099 [+0.022, +0.174]                | +0.99 [+0.22, +1.74]        | 0.980    | strong       | pass  |
| itt-009 | P — phonetic spelling (off-floor RD)    | 41  | +4.1 pp [−7.1, +15.5] pp               | risk difference — see below | 0.724    | inconclusive | pass* |
| itt-010 | W — word reading (79)                   | 53  | +0.030 [+0.009, +0.051]                | +2.37 [+0.68, +4.07]        | 0.986    | strong       | pass  |
| itt-011 | N — nonword reading (off-floor RD)      | 36  | +10.0 pp [−3.8, +23.7] pp              | risk difference — see below | 0.877    | suggestive   | pass* |

\* passes the convergence gate but trips a separate floor-specific _release_ gate — see caveats.

**Reading the core suite, outcome by outcome.** The pattern is coherent: the intervention most clearly moves the **code-related and directly-taught skills**, with the signal tapering across transfer measures and effectively vanishing on the broad standardised vocabulary tests.

- **Letter-sound knowledge, L (itt-007)** is the strongest result in the suite: median τ = +0.110 on the probability scale, about **+3.5 of 32 letter sounds** (89% credible interval +1.7 to +5.3; inner 50% +2.8 to +4.3), with P(helps) = 0.999 — _very strong_ evidence of benefit. The size claim also lands: P(benefit ≥ 2 items) = 0.91, with only 9% of the posterior inside the ROPE.
- **Phoneme blending, B (itt-008)**: median +0.099, about **+1.0 of 10** (89% +0.22 to +1.74), P = 0.980, _strong_. Directionally very clear; the magnitude is more equivocal — P(benefit ≥ 1 item) = 0.49, ROPE mass 0.51 — so a benefit is well supported but its practical size is uncertain (a small test amplifies uncertainty in items).
- **Word reading, W (itt-010)**: median +0.030, about **+2.4 of 79 words** (89% +0.7 to +4.1), P = 0.986, _strong_; P(benefit ≥ 1 word) = 0.90. A modest but well-supported gain in words actually read.
- **Taught vocabulary, TR and TE (itt-001, itt-002)**: taught expressive TE is +1.55 of 24 (89% +0.4 to +2.7), P = 0.985, _strong_; taught receptive TR is +1.37 of 24 (89% +0.2 to +2.5), P = 0.968, _moderate_. The words the programme teaches are learned.
- **Not-taught vocabulary, UR and UE (itt-003, itt-004)**: the transfer measures are weaker. Not-taught receptive UR is +0.60 of 12 (89% −0.03 to +1.24), P = 0.937, _moderate_ — some generalisation, but P(benefit ≥ 1 item) = only 0.16 and ROPE mass 0.84, so any transfer is small. Not-taught expressive UE is +0.31 of 12 (89% −0.35 to +0.97), P = 0.773, _suggestive_ and practically negligible (ROPE mass 0.95).
- **Standardised vocabulary, R and E (itt-005, itt-006)**: both are flat. Receptive R median +0.001 (about +0.2 of 170, 89% −3.8 to +4.3), P = 0.539; expressive E median +0.001 (+0.2 of 170), P = 0.534. Both are firmly **inconclusive** on direction and their wide item intervals reflect the broad-instrument scale; this is a genuinely uninformative-about-direction result, not evidence of "no effect". Broad standardised vocabulary was not moved over this window.
- **Phonetic spelling, P (itt-009)** and **nonword reading, N (itt-011)** are heavily floored (most children score zero), so the **primary estimand is a binary off-the-floor risk difference**: among children observed at the floor at baseline, the change in the probability of moving above zero by the first post-baseline wave (this is a post-hoc, arm-blind floor rule, so it is exploratory). For **P**, the median risk difference is **+4.1 percentage points** (89% −7.1 to +15.5), P(helps) = 0.724, _inconclusive_; the raw movers were 7/24 = 29% (intervention) versus 2/17 = 12% (control). For **N**, the median is **+10.0 percentage points** (89% −3.8 to +23.7), P(helps) = 0.877, _suggestive_; movers were 10/21 = 48% versus 2/15 = 13%. The graded item-count secondaries (reported for completeness, detection-limited by the floor) are essentially uninformative (P near 0.5) and are not the headline.

### A2. Robustness and adjusted variants (τ is CAUSAL)

These re-estimate the same causal τ with one extra adjustment, or add a further outcome. The extra covariate enters as an _adjusted association_ (not causal); the τ remains the causal quantity. The headline finding is **agreement**: no adjustment overturns the direction or the rough magnitude of the matching core estimate.

| ID      | What it adds                                    | Outcome (max)              | N   | τ items — median [89% CI] | P(helps) | Evidence     | Matches core?     |
| ------- | ----------------------------------------------- | -------------------------- | --- | ------------------------- | -------- | ------------ | ----------------- |
| itt-017 | + ability (block-design)                        | TR (24)                    | 54  | +1.27 [+0.09, +2.44]      | 0.957    | moderate     | ≈ itt-001 (+1.37) |
| itt-018 | + ability                                       | TE (24)                    | 54  | +1.47 [+0.33, +2.59]      | 0.982    | strong       | ≈ itt-002 (+1.55) |
| itt-019 | + ability                                       | UR (12)                    | 54  | +0.56 [−0.06, +1.20]      | 0.924    | moderate     | ≈ itt-003 (+0.60) |
| itt-020 | + ability                                       | UE (12)                    | 54  | +0.32 [−0.35, +0.98]      | 0.777    | suggestive   | ≈ itt-004 (+0.31) |
| itt-021 | + ability                                       | R (170)                    | 54  | +0.33 [−3.71, +4.40]      | 0.551    | inconclusive | ≈ itt-005 (flat)  |
| itt-022 | + ability                                       | E (170)                    | 54  | +0.21 [−3.11, +3.51]      | 0.542    | inconclusive | ≈ itt-006 (flat)  |
| itt-023 | + ability                                       | L (32)                     | 54  | +3.54 [+1.69, +5.34]      | 0.999    | very strong  | ≈ itt-007 (+3.52) |
| itt-024 | + ability                                       | W (79)                     | 53  | +2.23 [+0.51, +3.96]      | 0.980    | strong       | ≈ itt-010 (+2.37) |
| itt-013 | + SES (parent education, book age)              | W (79)                     | 33  | +2.54 [+0.33, +4.70]      | 0.967    | moderate     | ≈ itt-010         |
| itt-014 | SES complete-case, unadjusted (matches itt-013) | W (79)                     | 33  | +2.37 [+0.48, +4.28]      | 0.977    | strong       | ≈ itt-010         |
| itt-113 | + SES                                           | L (32)                     | 34  | +3.41 [+1.20, +5.60]      | 0.993    | very strong  | ≈ itt-007         |
| itt-114 | SES complete-case, unadjusted (matches itt-113) | L (32)                     | 34  | +3.93 [+1.85, +5.96]      | 0.998    | very strong  | ≈ itt-007         |
| itt-027 | + study site (area)                             | W (79)                     | 53  | +2.57 [+0.84, +4.32]      | 0.992    | very strong  | ≈ itt-010         |
| itt-028 | + study site (area)                             | L (32)                     | 54  | +3.51 [+1.66, +5.30]      | 0.998    | very strong  | ≈ itt-007         |
| itt-025 | new outcome                                     | F — basic concepts (18)    | 54  | +0.87 [−0.26, +1.98]      | 0.891    | suggestive   | new               |
| itt-026 | new outcome                                     | T — receptive grammar (32) | 54  | +0.65 [−0.83, +2.12]      | 0.760    | suggestive   | new               |

- **General-ability adjustment (itt-017–024)** adds a block-design cognitive score to the six vocabulary outcomes and the two reading anchors (L, W). Every adjusted τ lands within a whisker of its core counterpart — L stays _very strong_ at +3.5 items, W stays _strong_ at +2.2 items, TE _strong_, TR/UR _moderate_, UE _suggestive_, and R/E stay flat and _inconclusive_. The headline effects are not an artefact of ability imbalance between arms.
- **SES adjustment (itt-013 for W, itt-113 for L)** adds mother's and father's post-16 education and the age the child first had books, on the smaller complete-case subset (N = 33–34). Each is paired with a **matched unadjusted comparator on the identical subset** (itt-014, itt-114) so the effect of the adjustment can be separated from the effect of the sample restriction. W is +2.54 items adjusted versus +2.37 unadjusted-on-subset; L is +3.41 adjusted versus +3.93 unadjusted-on-subset. The effects survive; the modest label softening for itt-013 (to _moderate_) reflects the halved sample, not the adjustment.
- **Site adjustment (itt-027 for W, itt-028 for L)** adds study area; both effects are unchanged and if anything sharpen slightly (W _very strong_, +2.6 items; L _very strong_, +3.5 items).
- **Two additional outcomes**, fitted with the same empty-adjustment-set core design, extend the suite: **basic concepts F (itt-025)**, +0.87 of 18 (89% −0.26 to +1.98), P = 0.891, _suggestive_; and **receptive grammar T (itt-026)**, +0.65 of 32 (89% −0.83 to +2.12), P = 0.760, _suggestive_. Both lean positive but are directionally soft and practically small — consistent with the tapering seen on the broader language measures. (Neither carries the ROPE/δ machinery in its outputs, so items are translated as probability × maximum.)

### B. Joint models (per-outcome τ is CAUSAL)

**itt-012** fits ten outcomes (TR, TE, UR, UE, R, E, L, B, P, W) in one PyMC graph. In this fit it is a **product of outcome-specific marginals** — it does _not_ estimate residual correlation between outcomes (despite the family's optional LKJ-correlation capability). Its value is a single common-scale ranking and formal pairwise contrasts, not a change to any single effect. Each outcome's headline is its average marginal effect (probability scale); the per-outcome τ tracks the single-outcome models closely, confirming the ITT-vs-joint consistency check.

| Outcome      | AME probability — median [89% CI] | AME items — median [89% CI] | P(helps) | Evidence     |
| ------------ | --------------------------------- | --------------------------- | -------- | ------------ |
| L            | +0.110 [+0.052, +0.167]           | +3.53 [+1.67, +5.35]        | 0.999    | very strong  |
| B            | +0.091 [+0.012, +0.167]           | +0.91 [+0.12, +1.67]        | 0.968    | moderate     |
| W            | +0.030 [+0.009, +0.052]           | +2.37 [+0.68, +4.08]        | 0.987    | strong       |
| TE           | +0.065 [+0.018, +0.113]           | +1.57 [+0.44, +2.71]        | 0.986    | strong       |
| TR           | +0.055 [+0.005, +0.104]           | +1.32 [+0.12, +2.49]        | 0.961    | moderate     |
| UR           | +0.058 [−0.001, +0.116]           | +0.69 [−0.01, +1.39]        | 0.942    | moderate     |
| P (graded †) | +0.020 [−0.012, +0.052]           | +1.82 [−1.14, +4.81]        | 0.838    | suggestive   |
| UE           | +0.032 [−0.029, +0.092]           | +0.38 [−0.35, +1.10]        | 0.799    | suggestive   |
| R            | +0.006 [−0.018, +0.030]           | +1.01 [−3.11, +5.06]        | 0.655    | inconclusive |
| E            | −0.0005 [−0.020, +0.020]          | −0.08 [−3.44, +3.34]        | 0.483    | inconclusive |

† For P the joint model reports the **graded** item effect, which is _not_ the floored off-floor headline (itt-009); treat it as exploratory.

The three **contrast models** ask whether teaching-specific gains generalise, using the same joint machinery to difference two outcomes' effects on the probability scale. Because the fit factorises outcomes, these contrasts omit within-child covariance and are a **factorised sensitivity result**, not a calibrated paired contrast.

- **itt-015 — expressive generalisation (TE minus UE):** median difference +0.034 (89% −0.043 to +0.111), P(difference > 0) = 0.762, _suggestive_. The taught expressive words gained somewhat more than the not-taught ones — a hint of teaching-specificity in the expressive modality, but the interval spans zero, so the amount of expressive generalisation is not cleanly resolved. (TE itself is _strong_, P = 0.985; UE only _suggestive_, P = 0.787.)
- **itt-115 — receptive generalisation (TR minus UR):** median difference −0.003 (89% −0.080 to +0.071), P(difference > 0) = 0.471, _inconclusive_. Essentially no gap: taught and not-taught receptive vocabulary moved together, so receptive gains **generalise** to untaught words. (Both TR and UR are _moderate_, P = 0.968 and 0.955.)
- **itt-016 — modality (TE minus TR):** median difference +0.008 (89% −0.060 to +0.076), P(difference > 0) = 0.572, _inconclusive_. Taught expressive and taught receptive vocabulary gained by indistinguishable amounts.

### C. Block-2 exposure models (`delta` is an ASSOCIATION, not causal)

Block 2 is a second set of taught/not-taught words introduced later, with **no randomised contrast** (it is absent at baseline and equal across arms at the first post-baseline wave). Identification borrows the staggered roll-out: the immediate arm reaches block 2 while the wait-list arm is still on block 1. The focal `delta` is the jump in block-2 word knowledge attributable to block-2 teaching being **active** — but only under a **parallel-trends** assumption (that the arms' block-2 trajectories would otherwise have moved together), so it is a "block-2-active versus block-1-active" association, never treated-versus-untreated, and age-at-block-2 remains confounded. Each model carries a child random intercept and adjusts for ability, hearing and speech/phonological-memory covariates (all associations). N = 158–159 observations across 54 children.

| ID     | Outcome (max)                             | `delta` items — median [89% CI] | P(delta > 0) | Evidence (oriented)   |
| ------ | ----------------------------------------- | ------------------------------- | ------------ | --------------------- |
| bx-001 | TE2 — taught expressive, block 2 (24)     | +0.72 [−0.46, +1.91]            | 0.834        | _suggestive_ positive |
| bx-002 | TR2 — taught receptive, block 2 (24)      | −0.74 [−1.87, +0.43]            | 0.155        | _suggestive_ negative |
| bx-003 | UE2 — not-taught expressive, block 2 (12) | −0.28 [−0.91, +0.38]            | 0.251        | inconclusive          |
| bx-004 | UR2 — not-taught receptive, block 2 (12)  | +0.15 [−0.51, +0.82]            | 0.640        | inconclusive          |

The specificity check (a positive jump on _taught_ words but not on _not-taught_ words) is only partly borne out and is much weaker than the block-1 randomised picture. Taught expressive block-2 words (bx-001) show a _suggestive_ positive association (+0.7 of 24), and the two not-taught outcomes are inconclusive as hoped — but taught receptive block-2 words (bx-002) lean _negative_ (favoured-direction P = 0.845), which does not fit a clean teaching-specificity story. Given the parallel-trends dependence and the confounding by age-at-block-2, these are best read as weak, mixed associational signals, not as a second randomised confirmation.

## What the family concludes

The randomised evidence is internally consistent and points one way. The intervention produces **strong-to-very-strong** benefits on the code-related and directly-taught skills — **letter-sound knowledge (L, +3.5 of 32, very strong)**, **word reading (W, +2.4 of 79, strong)**, **phoneme blending (B, +1.0 of 10, strong)** and **taught expressive vocabulary (TE, +1.6 of 24, strong)** — with **moderate** support for taught receptive vocabulary (TR) and a tapering, **suggestive-to-moderate** signal on the transfer measures (UR moderate, UE suggestive). It is **inconclusive and practically negligible on the broad standardised vocabulary tests (R, E)**, and only _suggestive_ on the softer language outcomes (basic concepts F, receptive grammar T). On the floored code skills the direction is favourable (nonword reading N _suggestive_, +10 pp off the floor; phonetic spelling P _inconclusive_), but both are exploratory.

These conclusions are robust: adjusting τ for general ability, SES, or study site leaves every headline effect essentially unchanged, and the joint model reproduces the single-outcome ranking. The contrasts add nuance — receptive gains generalise to untaught words, while expressive gains show a hint of teaching-specificity. This ITT picture is the anchor the other family notes triangulate against: the difference-in-differences and gain-factor re-analyses recover the same randomised effects through different designs, and the mediation family attributes the word-reading gain to the letter-sound route rather than the vocabulary route — which is exactly what a code-related benefit that spares broad vocabulary would predict.

## Caveats and convergence

- **All 35 models pass the convergence gate** (0 divergences, R-hat ≤ 1.002, effective sample sizes in the thousands). No divergence-only or funnel-geometry flags arise in this family.
- **Floored outcomes P (itt-009) and N (itt-011) fail a separate floor-specific _release_ gate.** Their τ power-scaling diagnostic flags a _potential prior-data conflict_, and the six-cell estimand-matched sensitivity grid that the floor rule requires to clear that flag was **not** produced at this `reporting` fit. Both rendered reports therefore carry "Release gate failed: prior-data conflict is unresolved — this analysis is not ready for scientific interpretation." Treat the P and N off-floor headlines as **provisional** pending that grid; the N _suggestive_ signal in particular should not be quoted as settled.
- **Small samples inflate winners.** With ~54 children (and only 33–36 for the SES subsets and the floored subgroups), point estimates are on average magnitude-inflated; lead with the intervals. The item intervals on R and E are very wide because those tests have 170 items.
- **Treatment-prior sensitivity** is also flagged (_potential prior-data conflict_, an informational diagnostic only for the continuous outcomes — it does not trip a release gate there) on several models: itt-003, 007, 008, 013, 019, 023, 025, 028, 113, 114. For these, the interval — not the point — is the honest summary. This mostly affects the strong-signal models where the likelihood is highly informative.
- **Floor rule is post-hoc.** The 40% off-floor gate for P and N was chosen after inspecting the score distribution, so those estimands are exploratory, not prospectively specified primaries.
- **Modified ITT, not full ITT.** Each model analyses observed complete cases (54 of 57 randomised children are in the repository, and each model also needs its outcome and baseline observed); extending the causal reading to missing children assumes their exclusion is ignorable given observed pre-treatment information.
- **Associations are not levers.** Every covariate in these models (baseline, age, ability, SES, site, hearing, speech) is an adjusted association; the block-2 `delta` is an association under parallel trends. Residual confounding by latent general ability remains, and the child random intercept pools stable individual differences but does not stand in for general ability. Only the ITT/joint τ is causal.
