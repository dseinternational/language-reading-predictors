> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

# Full reporting fit of all statistical (Bayesian) models — findings

Date: 2026-07-08 (revised 2026-07-09 for readability and to expand the gain- and level-factor sections). A clean full fit of the entire Bayesian statistical-model suite from `main` at commit `8ec2089`, under the production sampling configuration, with every Quarto report rendered and all artefacts uploaded to Azure Blob Storage. This note records what was run, how well it converged, and what the fits say — read alongside each model's own report and `METHODS.md`.

## How to read the numbers in this note

This section is for readers new to Bayesian reporting (for example, undergraduate project students). The study is a randomised trial of a reading-and-language intervention for children with Down syndrome, measured at four timepoints; one group started the intervention immediately and a wait-list group started it later, which is what lets several different analyses estimate the same effect.

- **Posterior, credible interval.** Each model produces a probability distribution over the effect size (the _posterior_), given the data and the model. We summarise it as a median and a **95 % credible interval** — the range within which the effect lies with 95 % probability. Read "τ = +0.11 [+0.04, +0.18]" as: _our best estimate is +0.11, and the effect is very probably somewhere between +0.04 and +0.18_. Unlike a frequentist confidence interval, this is a direct probability statement about the effect.
- **P(effect > 0)** is the posterior probability the effect is positive. 0.5 means "no idea about direction"; 0.98 means "98 % sure the intervention helps on this measure". There are no p-values anywhere in this suite.
- **Evidence labels.** The project's fixed ladder (issue #179; see `notes/202606261304-evidence-strength-and-rope-reporting.md`) qualifies the evidence for a **named claim** — "strong evidence that the intervention helps" — at round-odds boundaries on the direction probability: **inconclusive** (< 0.75) / **suggestive** (≥ 0.75) / **moderate** (≥ 0.91) / **strong** (≥ 0.97) / **very strong** (≥ 0.99). The label is oriented to the favoured direction (a clearly negative effect is evidence of harm, not "inconclusive"), always follows the probability itself, and never describes the effect's _size_ — direction and magnitude are separate questions, so a "very strong" direction can sit on a small effect. Where a result looks like "no effect", the honest label is _inconclusive_, optionally quantified by how much of the posterior falls inside a region of practical equivalence (ROPE) around zero.
- **Scales.** Most outcomes are counts of correct items on a test with a fixed maximum, so effects are reported as a **risk difference (RD)** — the change in the proportion of items a child gets right (e.g. +0.03 = 3 percentage points) — and, where available, on the **items scale** (e.g. "+2.9 words on a 79-word reading test"), which is the easiest form to picture.
- **Causal vs association.** Only effects identified by the trial's randomisation are labelled _causal_: the intention-to-treat effect τ, the difference-in-differences δ, and the gain-factor on-intervention term. Every other coefficient — baselines, ages, skill-to-skill couplings, mediator paths — is an _adjusted association_: informative about how skills travel together, but never to be read as "X drives Y".
- **Floored outcomes.** Two tests (phonetic spelling P, nonword reading N) are so hard for this age group that most children score zero (they sit "on the floor"). A graded effect estimate is nearly meaningless there, so the pre-specified primary estimand is binary: does the intervention change the probability of scoring _above_ zero?
- **Convergence.** Before believing any posterior we check that the sampler worked: R-hat ≈ 1.00 (chains agree), effective sample size (ESS) large enough, few or no _divergences_ (numerical warnings that the sampler hit difficult geometry), and healthy energy diagnostics (BFMI). A model that fails these checks gets flagged before anyone reads its estimates.

## What was run

`python scripts/fit_statistical_model.py all --config reporting --render`, then `scripts/compare_statistical_models.py --config reporting`, then upload.

- **89/89 models fitted, 0 failed. 89/89 Quarto reports rendered.** Wall-clock **1 h 44 m** on a 16-core VM.
- Sampling preset `reporting`: **6000 draws × 6000 tune × 6 chains, `target_accept = 0.95`** per model (36 000 posterior draws each).
- The 16 model families, by count: itt 23, gain_factors 16, aligned 9, did 8, level_factors 8, mechanism 8, dose_response 3, joint 3, growth 2, horseshoe 2, mediation 2, plus one each of lcsm, mediation_multi, corr_factor and historical_growth (the last a separate historical-cohort study).
- The Quarto renders emit non-fatal "problem with a fenced div" warnings (a callout-syntax quirk); all 89 `index.html` built regardless.

## Artefacts and upload

Uploaded to Azure Blob Storage under `<outputs container>/language-reading-predictors/output/statistical_models/`:

- `models/{model_id}-reporting/` — **7228 files, 479 MB, 0 failed** (config, diagnostics, priors, result CSVs, PNG/SVG figures, and the rendered `index.html`).
- `comparison/` — 9 files (τ forests, ITT-vs-joint consistency, mechanism/dose/DiD nested-LOO tables).

Trace files (`trace.nc`, ~16.8 GB across the 89 fits) were **excluded** from the upload by design, matching the fit script's default. Note that the built-in `--upload` flag could not be used: it authenticates through the Azure Python SDK against the `public` container, for which the VM's managed identity has no write permission (`AuthorizationPermissionMismatch`). The upload was done instead with `azcopy` (managed identity) to the `outputs` container, the same path the gradient-boosting artefacts use. Output is git-ignored, so nothing beyond this note is committed.

## Convergence and diagnostics

Every model was checked against the pass/fail gate (`r̂ ≤ 1.01`, ESS ≥ 400, BFMI ≥ 0.30, and zero divergences). **79/89 passed outright.** All 10 that did not have **`r̂ = 1.00` and pass the ESS check** — they are flagged purely on divergences, not on non-convergence:

| Model            | Family                     | Divergences | of 36 000 | Other flags          |
| ---------------- | -------------------------- | ----------: | --------: | -------------------- |
| lrp-rli-mm-001   | corr_factor                |         422 |    1.17 % | **BFMI 0.21 < 0.30** |
| lrp-rli-mech-056 | mechanism (R→W)            |         145 |    0.40 % | —                    |
| lrp-rli-mech-058 | mechanism (L→W)            |          43 |    0.12 % | —                    |
| lrp-rli-mech-057 | mechanism (E→W)            |          39 |    0.11 % | —                    |
| lrp-rli-mech-071 | mechanism (L→W, moderated) |          34 |    0.09 % | —                    |
| lrp-rli-mech-073 | mechanism (L→W × age)      |          30 |    0.08 % | —                    |
| lrp-rli-mech-173 | mechanism (L→W baseline)   |          28 |    0.08 % | —                    |
| lrp-rli-dose-077 | dose_response              |          13 |    0.04 % | —                    |
| lrp-rli-did-007  | did (period dose)          |          10 |    0.03 % | —                    |
| lrp-rli-dose-177 | dose_response              |          10 |    0.03 % | —                    |

Nine of the ten sit at **≤ 0.4 % divergences** — a handful of transitions from the funnel geometry of the GP/HSGP mechanism surfaces and the session-dose slopes, comfortably inside the ≤ 1 % guidance in `METHODS.md`. Their τ/slope posteriors are trustworthy; the divergences are worth a note, not a veto. **Only `lrp-rli-mm-001` (the correlated-domain-factor measurement model) is a genuine sampling concern**: 1.17 % divergences _and_ sub-threshold BFMI (0.21), the signature of a latent-factor funnel that the current parameterisation samples inefficiently. Its factor **correlations** are stable and interpretable (below), but its **structural** coefficients should be read cautiously pending a non-centred reparameterisation or a higher `target_accept`.

## Headline causal findings — the randomised ITT effect (`itt` suite)

The intention-to-treat (ITT) models ask the trial's primary question: _did being randomised to start the intervention immediately improve each outcome at the end of the randomised phase, compared with waiting?_ Because assignment was random, the two groups differ (on average) only in the intervention, so the group coefficient τ is causal with no adjustment needed; the models include each child's own baseline score and age purely to sharpen precision. τ is reported on the risk-difference scale (median, 95 % credible interval, P(τ>0)); positive = the intervention helps.

| Outcome                        | Model   |     τ (RD) | 95 % CrI         | P(τ>0) | Evidence of benefit | ≈ items                 |
| ------------------------------ | ------- | ---------: | ---------------- | -----: | ------------------- | ----------------------- |
| Letter-sound knowledge (L)     | itt-007 | **+0.110** | [+0.040, +0.179] |  0.999 | very strong         | +3.5 of 32 sounds       |
| Phoneme blending (B)           | itt-008 | **+0.099** | [+0.004, +0.192] |  0.980 | strong              | +1 of 10 items          |
| Taught expressive vocab (TE)   | itt-002 | **+0.064** | [+0.006, +0.122] |  0.985 | strong              | +1.5 of 24 taught words |
| Word reading (W)               | itt-010 | **+0.030** | [+0.004, +0.057] |  0.986 | strong              | +2.4 of 79 words        |
| Taught receptive vocab (TR)    | itt-001 |     +0.057 | [−0.003, +0.117] |  0.968 | moderate            | +1.4 of 24 taught words |
| Untaught receptive vocab (UR)  | itt-003 |     +0.050 | [−0.014, +0.116] |  0.937 | moderate            | +0.6 of 12 words        |
| Untaught expressive vocab (UE) | itt-004 |     +0.026 | [−0.041, +0.093] |  0.773 | suggestive          | +0.3 of 12 words        |
| Receptive vocab, ROWPVT (R)    | itt-005 |     +0.001 | [−0.027, +0.030] |  0.539 | inconclusive        | +0.2 of 170 words       |
| Expressive vocab, EOWPVT (E)   | itt-006 |     +0.001 | [−0.022, +0.025] |  0.534 | inconclusive        | +0.2 of 170 words       |
| Phonetic spelling (P)          | itt-009 |          — | —                |      — | inconclusive        | floored (see below)     |
| Nonword reading (N)            | itt-011 |          — | —                |      — | inconclusive        | floored                 |

Direction and magnitude are separate claims: for letter sounds the evidence of _some_ benefit is very strong (P(τ>0) = 0.999), while the evidence that the benefit clears the pre-specified 2-item minimally-important difference is suggestive (P(benefit ≥ 2 sounds) = 0.91). For the two flat vocabulary outcomes the ROPE quantifies the "no effect" reading: 58 % (R) and 67 % (E) of the posterior lies within ±2 items of zero, so the honest label is inconclusive-and-probably-negligible, not merely "not significant".

**The coherent story: there is strong-to-very-strong evidence that the intervention improves the reading/phonics-proximal and directly-taught skills — letter-sound knowledge, phoneme blending, word reading, and taught expressive vocabulary — while the evidence for any effect on broad standardised vocabulary (ROWPVT, EOWPVT) is inconclusive and probably negligible.** The gradient runs exactly as a phonics-and-taught-words intervention predicts: strongest on letter sounds, present on blending and word reading, present for _taught_ words but fading to inconclusive for _untaught_ words (the generalisation contrast itt-015 puts P(taught-expressive > untaught-expressive) at 0.79 — suggestive, not decisive), and absent on the broad norm-referenced vocabulary tests. The two heavily-floored outcomes (spelling, nonword reading) carry too little off-floor movement to estimate — for spelling the off-floor rate is essentially identical between arms (35.7 % vs 36.0 %).

## The effect is robust across adjustment and across three independent estimators

The word-reading and letter-sound effects survive both robustness adjustments and reappear, at consistent magnitude, in designs that do not share the ITT model's assumptions. In plain terms: three different ways of slicing the same trial — comparing the randomised groups (ITT), comparing each child with themself before and during the intervention (DiD), and pooling every on- and off-intervention period into one adjusted gains model (gain factors) — all land on the same answer.

- **General-ability adjustment** (block-design covariate, itt-017–024): L +0.110, W +0.028, TE +0.061 — essentially unchanged.
- **SES adjustment + matched complete-case comparators** (itt-013/113/014/114): L +0.107 to +0.123, W +0.030 — unchanged (the SES-adjusted W in itt-013 slips to moderate as its interval widens just across 0, but its complete-case sibling itt-014 stays strong).
- **Within-person waitlist-crossover DiD** (`did` family): the wait-list group's untreated first period is compared with its own treated second period, using the immediate group to anchor how much children improve anyway with time. Each child is their own control, so stable child-level differences cancel. W δ +0.367 [+0.050, +0.685], P(δ>0) = 0.99 (strong); L δ +0.560 [+0.183, +0.934], P = 1.00 (very strong); B δ +0.437 [+0.012, +0.851], P = 0.98 (strong); TE +0.300, P = 0.97 (moderate); R inconclusive (P = 0.50) — the same ranking, on the logit scale, from a completely different identification strategy.
- **Gain-factor ANCOVA and level-factor models** — detailed in their own sections below — agree again: gains W +0.037 / L +0.097 with R/E inconclusive, and the levels view shows the group gap opening exactly at the end of the randomised phase for letter sounds.

Three estimators built on different assumptions converge on the **same causal ordering: letter sounds ≫ blending ≈ word reading > taught vocabulary ≫ broad vocabulary ≈ 0.**

## The gain-factor analysis (`gf` family) — what predicts progress, and what the intervention adds

The eight gain-factor models (gf-001–008, one per outcome W/R/E/L/P/B/F/T) are the suite's workhorse "what predicts progress?" models. Each is an ANCOVA on a period's **post-score given its own pre-score**, stacking every on-intervention and untreated period from both arms, with a child-level random intercept to absorb stable individual differences. One coefficient — being on-intervention during the period, which randomisation made ignorable — is causal; **every other term (own baseline, age, general ability, upstream skills, interactions) is an explicit adjusted association.** SES is excluded by design (not a DAG node, statistically redundant). Heavily-floored P uses the suite floor rule (a Bernoulli model on the off-floor indicator).

**The causal term.** The marginal on-intervention effect, on the risk-difference and items scales:

| Outcome               | Model  | τ (RD) | 95 % CrI         | Items scale             | P(τ>0) | Evidence of benefit |
| --------------------- | ------ | -----: | ---------------- | ----------------------- | -----: | ------------------- |
| Letter sounds (L)     | gf-004 | +0.097 | [+0.029, +0.168] | **+3.1** of 32 sounds   |  0.994 | very strong         |
| Word reading (W)      | gf-001 | +0.037 | [+0.006, +0.065] | **+2.9** of 79 words    |  0.995 | very strong         |
| Phoneme blending (B)  | gf-006 | +0.072 | [−0.012, +0.160] | +0.7 of 10 items        |  0.911 | moderate            |
| Basic concepts (F)    | gf-007 | +0.049 | [−0.010, +0.111] | +0.9 of 18 items        |  0.864 | suggestive          |
| Phonetic spelling (P) | gf-005 | +0.031 | [−0.054, +0.105] | (off-floor probability) |  0.753 | suggestive          |
| Receptive grammar (T) | gf-008 | +0.012 | [−0.041, +0.066] | +0.4 of 32 items        |  0.680 | inconclusive        |
| Expressive vocab (E)  | gf-003 | +0.002 | [−0.023, +0.027] | +0.4 of 170 words       |  0.544 | inconclusive        |
| Receptive vocab (R)   | gf-002 | −0.005 | [−0.038, +0.025] | −0.9 of 170 words       |  0.365 | inconclusive        |

In plain words: a child on the intervention gains, over one period, about **three more letter sounds and three more sight words** than an equivalent child not yet on it — a small effect, but with very strong evidence it is real (P(τ>0) ≥ 0.99) — while the evidence for any movement on the broad vocabulary tests is inconclusive. This reproduces the ITT ordering from a model with completely different structure.

**The adjusted associations — what travels with progress.** These coefficients answer a different question from the treatment term: not "does the intervention work?" but "_which children_ move most, whoever they are?". Two things to hold on to while reading them. First, the covariate set is **pre-specified from the DAG**, not searched: each outcome's model carries its own baseline, linear age, non-verbal ability (block design), and only its DAG-upstream skills (word reading ← letter sounds + receptive vocabulary; blending ← letter sounds; spelling ← letter sounds + blending; expressive vocabulary, basic concepts and grammar ← receptive vocabulary; letter sounds and receptive vocabulary ← none), plus three focal interactions (treatment × ability, treatment × own baseline, age × ability). A skill absent from a model was excluded by the causal diagram, not found unimportant. Second, these are textbook **"Table 2 fallacy"** territory: the model was built to identify the treatment effect, so the covariate coefficients are conditional associations along unmodelled paths — they describe who progresses, and must not be read as levers ("teach X and Y will follow"). Coefficients are on the logit scale — own baseline and upstream skills enter as the period-start score (logit), age and ability per standard deviation.

The main-effect grid (medians, evidence labels per the #179 ladder, oriented to the favoured direction):

| Outcome               | Own baseline       | Age                | Non-verbal ability  | Upstream skill(s)                    |
| --------------------- | ------------------ | ------------------ | ------------------- | ------------------------------------ |
| Word reading (W)      | +0.86, very strong | −0.12, strong      | +0.08, suggestive   | L +0.09, strong; R +0.16, suggestive |
| Receptive vocab (R)   | +0.70, very strong | ≈ 0, inconclusive  | −0.04, inconclusive | — (none in DAG)                      |
| Expressive vocab (E)  | +0.69, very strong | ≈ 0, inconclusive  | −0.04, inconclusive | R +0.21, very strong                 |
| Letter sounds (L)     | +0.70, very strong | −0.10, suggestive  | +0.24, moderate     | — (none in DAG)                      |
| Phonetic spelling (P) | +0.85, very strong | ≈ 0, inconclusive  | +0.30, suggestive   | B +0.47, strong; L +0.33, moderate   |
| Phoneme blending (B)  | +0.75, very strong | ≈ 0, inconclusive  | +0.19, suggestive   | L +0.16, very strong                 |
| Basic concepts (F)    | +0.61, very strong | −0.08, suggestive  | +0.49, very strong  | R +0.41, very strong                 |
| Receptive grammar (T) | +0.60, very strong | −0.13, very strong | −0.05, inconclusive | R +0.46, very strong                 |

- **A child's own baseline dominates everywhere.** `gamma_own` is the largest term in every model (+0.60 to +0.86, all P ≈ 1.00): where a child starts on a skill is by far the best predictor of where they end the period. There is even a readable gradient in _how_ dominant: the most constrained, floor-adjacent skills are stickiest (word reading +0.86, spelling +0.85), while the broad language measures hold the loosest autoregression (grammar +0.60, concepts +0.61) — more of their period-to-period movement is left for other factors to explain. This is the Bayesian mirror of the gradient-boosting finding that each outcome's own baseline tops every predictor ranking, and (because the standardised coefficients sit well below 1) it also encodes ordinary regression to the mean: high starters stay high but drift back toward the pack.
- **The upstream couplings sort into two routes, matching the structure the reading literature would predict** (a decoding route and a language-comprehension route, as in the "simple view of reading"). On the **code route**, baseline letter-sound knowledge is associated with gains in word reading (+0.09, P = 0.97, strong), blending (+0.16, P = 1.00, very strong) and — alongside blending itself (+0.47, P = 0.99, strong) — phonetic spelling (+0.33, P = 0.96, moderate). On the **language route**, baseline receptive vocabulary is associated with gains in expressive vocabulary (+0.21, P = 1.00, very strong), basic concepts (+0.41, P = 1.00, very strong) and grammar (+0.46, P = 1.00, very strong). One honest qualification: because the covariates are DAG-pre-specified, most models never _tested_ the cross-route candidate — the one that did (word reading, offered both L and R) found the code-route term better supported (L strong vs R suggestive). That within-model comparison, the mediation result (the intervention's reading effect runs via letter sounds, not vocabulary) and the LCSM couplings all point the same way, which is what makes the two-route reading more than a modelling artefact.
- **Non-verbal ability adds little once the own baseline is in the model** — with two exceptions: letter-sound progress (+0.24, P = 0.95, moderate) and, most clearly, basic-concept progress (+0.49, P = 1.00, very strong). The CELF basic-concepts task is itself the most "general-cognitive" measure in the battery, so ability earning its keep exactly there is coherent rather than surprising.
- **Age is negatively associated with progress** in word reading (−0.12, P(<0) = 0.99, strong) and grammar (−0.13, P(<0) = 1.00, very strong): at the same starting score, older children gain slightly less over a period. The same signal appears independently in the adjusted-association model (adj-065: age −0.26, P(<0) = 0.99, the only baseline predictor of word-reading gain with very strong evidence) and in the latent change-score model (age → reading change −0.15, P(<0) = 1.00). Three different model shapes agreeing makes this the most dependable _predictor_ finding outside the baselines — though as an association it says nothing about why (candidate explanations range from developmental timing to older children in this cohort having already banked their fastest-moving years on these measures).
- **The pre-specified interactions tell three different stories, and the differences matter.** (1) **Treatment × ability is genuinely mixed**: positive for receptive vocabulary (+0.16, P = 0.98, strong), grammar (+0.24, P = 0.99, very strong) and expressive vocabulary (+0.12, P = 0.95, moderate), _negative_ for basic concepts (−0.30, P(<0) = 0.98, strong), and ≈ 0 for the flagship reading outcomes W and L. Sign flips within the same domain family are the signature of noise at this sample size, so we do not read a moderation story here — and importantly, for the outcomes the intervention actually moves, there is **no evidence it works better for more able children**. (2) **Treatment × own baseline is individually weak but directionally consistent**: the sign is negative in seven of eight models (moderate for blending −0.29, concepts −0.22 and grammar −0.19; weaker elsewhere). Taken at face value that hints the intervention slightly _flattens_ the baseline gradient — lower starters gaining a little more from being taught than high starters — which would be an equity-favourable property. With every individual interval spanning or nearly spanning zero this stays a hypothesis to carry forward, not a finding. (3) **Age × ability is the sleeper pattern**: positive in six of eight models, reaching very strong evidence for concepts (+0.20, P = 0.99) and grammar (+0.14, P = 1.00) and moderate for expressive vocabulary and letter sounds. Older-_and_-more-able children progress more than age and ability separately predict — a fan-spread (Matthew-effect-like) widening of the ability gradient with age on the language-side measures. It was pre-specified, it is the most consistent interaction in the grid, and it deserves a dedicated look in a future analysis rather than a passing mention.
- **The treated-only companions (gf-101–108)** refit each model on on-intervention periods alone, asking what predicts progress _while being taught_. They reproduce the same structure — own baseline dominant (+0.34 to +0.86), the L→W/B/P and R→E/F/T couplings at similar magnitudes (e.g. L→W +0.10, R→E +0.18, R→T +0.49), negative age for word reading and grammar, and the recurring positive age × ability term (concepts +0.25, grammar +0.15) — which says the predictor landscape is a property of how these children progress, not an artefact of mixing treated and untreated periods.

Two caveats frame all of the above. These are between-period associations in a small sample: a stable child-level trait or simple measurement noise can inflate "baseline skill → later gain" couplings (the child random intercept repairs some, not all, of this — see `METHODS.md` on cross-lagged structure), so magnitudes are softer than their labels suggest. And their proper scientific role is **hypothesis generation**: the code-route couplings here are exactly what motivated the mechanism and mediation models, which probe the same paths with more structure — the right next step for any pattern flagged above (the age × ability fan-spread, the treatment × baseline flattening) is a purpose-built model, not a stronger adjective.

## The level-factor analysis (`lf` family) — the same story told in levels

The eight level-factor models (lf-001–008) are the companion _levels_ view: instead of modelling gains, they model the **score itself at each of the four timepoints**, with the group difference and the ability association each allowed a separate coefficient per timepoint (group×time and ability×time vectors). The timing of the design makes exactly one of those coefficients a clean experiment: at t1 nobody has been treated (any group difference is chance imbalance at baseline); at **t2 the immediate group has been treated and the wait-list group has not — that contrast is randomised**; from t3 the wait-list group has crossed over, so both arms are treated and later contrasts are flagged associations.

The t2 (randomised) group contrast, on the logit scale:

| Outcome               | Model  | t2 contrast | 95 % CrI         | P(>0) | Evidence of benefit    |
| --------------------- | ------ | ----------: | ---------------- | ----: | ---------------------- |
| Letter sounds (L)     | lf-004 |  **+0.491** | [+0.009, +0.963] | 0.977 | strong                 |
| Phoneme blending (B)  | lf-006 |      +0.303 | [−0.188, +0.786] | 0.888 | suggestive             |
| Word reading (W)      | lf-001 |      +0.221 | [−0.263, +0.709] | 0.813 | suggestive             |
| Basic concepts (F)    | lf-007 |      +0.134 | [−0.221, +0.490] | 0.772 | suggestive             |
| Receptive grammar (T) | lf-008 |      +0.027 | [−0.232, +0.288] | 0.584 | inconclusive           |
| Expressive vocab (E)  | lf-003 |      −0.003 | [−0.200, +0.196] | 0.489 | inconclusive           |
| Phonetic spelling (P) | lf-005 |      −0.006 | [−0.823, +0.817] | 0.494 | inconclusive (floored) |
| Receptive vocab (R)   | lf-002 |      −0.037 | [−0.225, +0.150] | 0.349 | inconclusive           |

Same ordering again — strong evidence for letter sounds, suggestive for blending and word reading, inconclusive for vocabulary — though with wider intervals than the gain models, because a single-timepoint contrast throws away the within-child information the gains view exploits. That is itself an instructive methods point: **levels models answer "who is ahead right now?", gains models answer "who moved?", and the gains view is the more sensitive of the two here.**

The full group×time profiles carry a second, satisfying signature. For letter sounds the profile runs −0.08 (t1, baseline noise) → **+0.49 (t2, end of the randomised phase)** → +0.13 (t3) → +0.32 (t4); word reading and blending show the same rise-then-narrow shape. The t3 shrinkage is exactly what should happen if the intervention works: the wait-list group has just crossed over and is catching up, so the gap closes. The partial re-widening at t4 for letter sounds and blending (P ≈ 0.90, suggestive) hints that the head-start group holds an edge, but post-crossover contrasts are associations and are flagged as such. The vocabulary and grammar profiles are flat at every timepoint — the inconclusive vocabulary result is not an artefact of when we looked.

The ability×time vectors add one more association: non-verbal ability is most strongly tied to letter-sound levels early (t1 +0.35 [+0.07, +0.63], P = 0.99, very strong) and fades by t4 (+0.11, inconclusive) — as instruction accumulates, where a child sits on letter sounds is decreasingly about general ability. The single wait-list-specific ability interaction (`gamma_grp_ability`) is inconclusive throughout.

## Mechanism and mediation — how the reading gain arises

These couplings are **adjusted associations, not causal drivers**, but they are internally consistent and answer the "through what?" question the trial itself cannot.

- **Mediation (g-formula NDE/NIE decomposition):** the intervention's effect on word reading runs **through letter-sound knowledge**. Single-mediator (med-059): the indirect path via L is +0.023 [+0.006, +0.045], P = 0.998 (very strong), ≈ 62 % of the total effect, while the evidence for a direct effect on its own is only suggestive (P = 0.85). Two-mediator (med-064): the path via L is +0.025 [+0.006, +0.048], P = 0.998 (very strong), whereas the path via expressive vocabulary is ≈ 0 [−0.009, +0.007] (inconclusive). In plain words: the intervention improved word reading mainly _by_ improving letter-sound knowledge, not by improving vocabulary — the same conclusion the ITT ranking implies.
- **Mechanism GP slopes** (marginal, adjusted): E→W +0.271 [−0.001, +0.593] (the strongest, all but excluding 0), R→W +0.131, L→W +0.090 — all positive-leaning associations with word reading, credible intervals touching 0.
- **Latent coupled change-score model (lcsm-067):** prior letter-sound score → later reading change +0.135 [+0.025, +0.256], P = 0.99, and prior expressive-vocabulary score → later reading change +0.284 [+0.054, +0.521], P = 0.99 — very strong evidence for both cross-lagged couplings (as associations); reading shows negative self-feedback (regression to the mean).
- **Dose-response:** pooled cumulative-session slope +0.127 [+0.028, +0.227] P 0.99 (dose-277); period-resolved slopes ≈ +0.13 per period (P ≈ 0.95). More sessions track more word-reading gain (dose is a partial collider, so this is a sensitivity view, not a clean causal dose curve).

## Secondary and cross-check families

- **Regularised-horseshoe predictor ranking (cross-check of the gradient-boosting ranking):** for word-reading _level_ (hs-002), letter sounds (P 0.99) and expressive vocabulary (P 0.99) are selected decisively, then grammar and age. For word-reading _gain_ (hs-001) nothing is selected (top probability 0.59, age) — echoing the gradient-boosting result that change scores are near-noise. The Bayesian and gradient-boosting rankings agree.
- **Correlated-domain-factor measurement model (mm-001):** the three latent domains are strongly correlated — vocabulary↔grammar 0.80, vocabulary↔code 0.74, code↔grammar 0.65 (P ≈ 1.00, very strong) — supporting a correlated-skill-system reading of the battery. (Structural coefficients held back pending the reparameterisation noted above.)
- **Multivariate growth curves (gc-069/070):** between-child gamma associations are inconclusive for R/E/W; only grammar shows a positive association (T ≈ +0.11, P ≈ 0.97, moderate-to-strong).
- **Aligned per-protocol (`al` family):** cohort contrast +0.217 [−0.109, +0.544], correctly flagged **non-causal** (confounded by age-at-onset and cohort timing) — no term in this family is presented as causal, by design.
- **Adjusted between-child association (adj-065):** of the baseline predictors of word-reading gain, only age carries very strong evidence (−0.259 [−0.463, −0.057], P(<0) = 0.99; older children gaining less); the language composite (+0.225, P = 0.93, moderate) and letter sounds (+0.159, P = 0.89, suggestive) are weaker.
- **Historical-cohort growth reproduction (rlm-hg-001, separate study):** converged cleanly (r̂ 1.00, 0 divergences).

## Caveats

- **This is preliminary, small-sample research** (≈ 159 children after cleaning; some outcomes far fewer). Credible intervals are wide, and several of the "strong" and "very strong" direction calls above sit on 95 % intervals that only just exclude zero — and, per the design analysis in the evidence-labels note, small-sample point estimates that clear a threshold are on average magnitude-inflated (the winner's curse), so lead with the interval, not the point.
- **Only τ (and the DiD δ, and the gain-factor on-intervention marginal) is causal.** Every mechanism slope, mediation path, coupling, growth gamma, horseshoe coefficient and adjusted association is an adjusted association and must not be read as "X drives Y".
- **`lrp-rli-mm-001` needs a sampling fix** before its structural coefficients are quoted; its correlations are fine. The nine other divergence-flagged fits are usable as-is but the mechanism GP models would benefit from a re-run at a higher `target_accept` if any single slope becomes load-bearing.
