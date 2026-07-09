> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

# Full reporting fit of all statistical (Bayesian) models — findings

Date: 2026-07-08 (revised 2026-07-09 for readability and to expand the gain- and level-factor sections). A clean full fit of the entire Bayesian statistical-model suite from `main` at commit `8ec2089`, under the production sampling configuration, with every Quarto report rendered and all artefacts uploaded to Azure Blob Storage. This note records what was run, how well it converged, and what the fits say — read alongside each model's own report and `METHODS.md`.

## How to read the numbers in this note

This section is for readers new to Bayesian reporting (for example, undergraduate project students). The study is a randomised trial of a reading-and-language intervention for children with Down syndrome, measured at four timepoints; one group started the intervention immediately and a wait-list group started it later, which is what lets several different analyses estimate the same effect.

- **Posterior, credible interval.** Each model produces a probability distribution over the effect size (the _posterior_), given the data and the model. We summarise it as a median and a **95 % credible interval** — the range within which the effect lies with 95 % probability. Read "τ = +0.11 [+0.04, +0.18]" as: _our best estimate is +0.11, and the effect is very probably somewhere between +0.04 and +0.18_. Unlike a frequentist confidence interval, this is a direct probability statement about the effect.
- **P(effect > 0)** is the posterior probability the effect is positive. 0.5 means "no idea about direction"; 0.98 means "98 % sure the intervention helps on this measure". There are no p-values anywhere in this suite.
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

| Outcome                        | Model   |     τ (RD) | 95 % CrI         | P(τ>0) | Read as                                    |
| ------------------------------ | ------- | ---------: | ---------------- | -----: | ------------------------------------------ |
| Letter-sound knowledge (L)     | itt-007 | **+0.110** | [+0.040, +0.179] |  0.999 | credible, large (≈ +3.5 of 32 sounds)      |
| Phoneme blending (B)           | itt-008 | **+0.099** | [+0.004, +0.192] |  0.980 | credible (≈ +1 of 10 items)                |
| Taught expressive vocab (TE)   | itt-002 | **+0.064** | [+0.006, +0.122] |  0.985 | credible (≈ +1.5 of 24 taught words)       |
| Word reading (W)               | itt-010 | **+0.030** | [+0.004, +0.057] |  0.986 | credible, small (≈ +2.4 of 79 words)       |
| Taught receptive vocab (TR)    | itt-001 |     +0.057 | [−0.003, +0.117] |  0.968 | leans positive                             |
| Untaught receptive vocab (UR)  | itt-003 |     +0.050 | [−0.014, +0.116] |  0.937 | leans positive                             |
| Untaught expressive vocab (UE) | itt-004 |     +0.026 | [−0.041, +0.093] |  0.773 | inconclusive                               |
| Receptive vocab, ROWPVT (R)    | itt-005 |     +0.001 | [−0.027, +0.030] |  0.539 | null                                       |
| Expressive vocab, EOWPVT (E)   | itt-006 |     +0.001 | [−0.022, +0.025] |  0.534 | null                                       |
| Phonetic spelling (P)          | itt-009 |          — | —                |      — | floored; off-floor RD ≈ 0 (0.357 vs 0.360) |
| Nonword reading (N)            | itt-011 |          — | —                |      — | floored; inconclusive                      |

**The coherent story: the intervention credibly moves the reading/phonics-proximal and directly-taught skills — letter-sound knowledge, phoneme blending, word reading, and taught expressive vocabulary — while broad standardised vocabulary (ROWPVT, EOWPVT) is flat.** The gradient runs exactly as a phonics-and-taught-words intervention predicts: strongest on letter sounds, present on blending and word reading, present for _taught_ words but fading to inconclusive for _untaught_ words (the generalisation contrast itt-015 puts P(taught-expressive > untaught-expressive) at 0.79 — suggestive, not decisive), and absent on the broad norm-referenced vocabulary tests. The two heavily-floored outcomes (spelling, nonword reading) carry too little off-floor movement to estimate — for spelling the off-floor rate is essentially identical between arms (35.7 % vs 36.0 %).

## The effect is robust across adjustment and across three independent estimators

The word-reading and letter-sound effects survive both robustness adjustments and reappear, at consistent magnitude, in designs that do not share the ITT model's assumptions. In plain terms: three different ways of slicing the same trial — comparing the randomised groups (ITT), comparing each child with themself before and during the intervention (DiD), and pooling every on- and off-intervention period into one adjusted gains model (gain factors) — all land on the same answer.

- **General-ability adjustment** (block-design covariate, itt-017–024): L +0.110, W +0.028, TE +0.061 — essentially unchanged.
- **SES adjustment + matched complete-case comparators** (itt-013/113/014/114): L +0.107 to +0.123, W +0.030 — unchanged (the SES-adjusted W in itt-013 widens just across 0, but its complete-case sibling itt-014 stays credible).
- **Within-person waitlist-crossover DiD** (`did` family): the wait-list group's untreated first period is compared with its own treated second period, using the immediate group to anchor how much children improve anyway with time. Each child is their own control, so stable child-level differences cancel. W δ +0.367 [+0.050, +0.685], L δ +0.560 [+0.183, +0.934], B δ +0.437 [+0.012, +0.851], TE +0.300 (P 0.97), R null — the same ranking, on the logit scale, from a completely different identification strategy.
- **Gain-factor ANCOVA and level-factor models** — detailed in their own sections below — agree again: gains W +0.037 / L +0.097 with R/E null, and the levels view shows the group gap opening exactly at the end of the randomised phase for letter sounds.

Three estimators built on different assumptions converge on the **same causal ordering: letter sounds ≫ blending ≈ word reading > taught vocabulary ≫ broad vocabulary ≈ 0.**

## The gain-factor analysis (`gf` family) — what predicts progress, and what the intervention adds

The eight gain-factor models (gf-001–008, one per outcome W/R/E/L/P/B/F/T) are the suite's workhorse "what predicts progress?" models. Each is an ANCOVA on a period's **post-score given its own pre-score**, stacking every on-intervention and untreated period from both arms, with a child-level random intercept to absorb stable individual differences. One coefficient — being on-intervention during the period, which randomisation made ignorable — is causal; **every other term (own baseline, age, general ability, upstream skills, interactions) is an explicit adjusted association.** SES is excluded by design (not a DAG node, statistically redundant). Heavily-floored P uses the suite floor rule (a Bernoulli model on the off-floor indicator).

**The causal term.** The marginal on-intervention effect, on the risk-difference and items scales:

| Outcome               | Model  | τ (RD) | 95 % CrI         | Items scale             | P(τ>0) |
| --------------------- | ------ | -----: | ---------------- | ----------------------- | -----: |
| Letter sounds (L)     | gf-004 | +0.097 | [+0.029, +0.168] | **+3.1** of 32 sounds   |  0.994 |
| Word reading (W)      | gf-001 | +0.037 | [+0.006, +0.065] | **+2.9** of 79 words    |  0.995 |
| Phoneme blending (B)  | gf-006 | +0.072 | [−0.012, +0.160] | +0.7 of 10 items        |  0.911 |
| Basic concepts (F)    | gf-007 | +0.049 | [−0.010, +0.111] | +0.9 of 18 items        |  0.864 |
| Phonetic spelling (P) | gf-005 | +0.031 | [−0.054, +0.105] | (off-floor probability) |  0.753 |
| Receptive grammar (T) | gf-008 | +0.012 | [−0.041, +0.066] | +0.4 of 32 items        |  0.680 |
| Expressive vocab (E)  | gf-003 | +0.002 | [−0.023, +0.027] | +0.4 of 170 words       |  0.544 |
| Receptive vocab (R)   | gf-002 | −0.005 | [−0.038, +0.025] | −0.9 of 170 words       |  0.365 |

In plain words: a child on the intervention gains, over one period, about **three more letter sounds and three more sight words** than an equivalent child not yet on it — small but credibly real — while the broad vocabulary tests move not at all. This reproduces the ITT ordering from a model with completely different structure.

**The adjusted associations — what travels with progress.** Consistent patterns across the eight outcomes (all associations, none causal):

- **A child's own baseline dominates everywhere.** `gamma_own` is the largest term in every model (+0.60 to +0.86, all P ≈ 1.00): where a child starts on a skill is by far the best predictor of where they end the period. This is the Bayesian mirror of the gradient-boosting finding that each outcome's own baseline tops every predictor ranking.
- **Skill-to-skill couplings follow the reading DAG.** Baseline **letter-sound knowledge predicts gains in the code-related skills**: word reading (gamma_L +0.09, P 0.97), blending (+0.16, P 1.00) and — together with blending itself (+0.46, P 0.99) — phonetic spelling (+0.33, P 0.96). Baseline **receptive vocabulary predicts gains in the language-side skills**: expressive vocabulary (+0.21, P 1.00), basic concepts (+0.41, P 1.00) and grammar (+0.46, P 1.00). The two routes barely cross — vocabulary does not predict code gains and letter sounds do not predict vocabulary gains.
- **General (non-verbal) ability** is associated with faster letter-sound (+0.24, P 0.95) and basic-concept (+0.49, P 1.00) progress, but adds little elsewhere once the own baseline is in the model.
- **Age leans slightly negative** for word reading (−0.12, P(>0) = 0.01) and grammar (−0.13): older children gain marginally less at the same starting score, consistent with the adjusted-association model (adj-065) where age is the only credible baseline predictor of word-reading gain (−0.26).
- **The focal interactions are mostly inconclusive.** Treatment × ability and treatment × baseline interactions hover near zero for the flagship outcomes W and L — no evidence the intervention works better for more able children or for higher starters. The scattered exceptions (a positive treatment × ability term for R/E/T, a negative one for F) are small and not consistent enough across outcomes to interpret.
- **The treated-only companions (gf-101–108)** refit each model on on-intervention periods alone, asking what predicts progress _while being taught_. They reproduce the same structure — own baseline dominant (+0.34 to +0.86), L→W/B/P and R→E/F/T couplings, negative age for W — which says the predictor landscape is not an artefact of mixing treated and untreated periods.

## The level-factor analysis (`lf` family) — the same story told in levels

The eight level-factor models (lf-001–008) are the companion _levels_ view: instead of modelling gains, they model the **score itself at each of the four timepoints**, with the group difference and the ability association each allowed a separate coefficient per timepoint (group×time and ability×time vectors). The timing of the design makes exactly one of those coefficients a clean experiment: at t1 nobody has been treated (any group difference is chance imbalance at baseline); at **t2 the immediate group has been treated and the wait-list group has not — that contrast is randomised**; from t3 the wait-list group has crossed over, so both arms are treated and later contrasts are flagged associations.

The t2 (randomised) group contrast, on the logit scale:

| Outcome               | Model  | t2 contrast | 95 % CrI         | P(>0) | Read as        |
| --------------------- | ------ | ----------: | ---------------- | ----: | -------------- |
| Letter sounds (L)     | lf-004 |  **+0.491** | [+0.009, +0.963] | 0.977 | credible       |
| Phoneme blending (B)  | lf-006 |      +0.303 | [−0.188, +0.786] | 0.888 | leans positive |
| Word reading (W)      | lf-001 |      +0.221 | [−0.263, +0.709] | 0.813 | leans positive |
| Basic concepts (F)    | lf-007 |      +0.134 | [−0.221, +0.490] | 0.772 | leans positive |
| Receptive grammar (T) | lf-008 |      +0.027 | [−0.232, +0.288] | 0.584 | null           |
| Expressive vocab (E)  | lf-003 |      −0.003 | [−0.200, +0.196] | 0.489 | null           |
| Phonetic spelling (P) | lf-005 |      −0.006 | [−0.823, +0.817] | 0.494 | null (floored) |
| Receptive vocab (R)   | lf-002 |      −0.037 | [−0.225, +0.150] | 0.349 | null           |

Same ordering again — letter sounds clearest, blending and word reading leaning positive, vocabulary flat — though with wider intervals than the gain models, because a single-timepoint contrast throws away the within-child information the gains view exploits. That is itself an instructive methods point: **levels models answer "who is ahead right now?", gains models answer "who moved?", and the gains view is the more sensitive of the two here.**

The full group×time profiles carry a second, satisfying signature. For letter sounds the profile runs −0.08 (t1, baseline noise) → **+0.49 (t2, end of the randomised phase)** → +0.13 (t3) → +0.32 (t4); word reading and blending show the same rise-then-narrow shape. The t3 shrinkage is exactly what should happen if the intervention works: the wait-list group has just crossed over and is catching up, so the gap closes. The partial re-widening at t4 for letter sounds and blending (P ≈ 0.90) hints that the head-start group holds an edge, but post-crossover contrasts are associations and are flagged as such. The vocabulary and grammar profiles are flat at every timepoint — the null is not an artefact of when we looked.

The ability×time vectors add one more association: non-verbal ability is most strongly tied to letter-sound levels early (t1 +0.35, credible) and fades by t4 (+0.11) — as instruction accumulates, where a child sits on letter sounds is decreasingly about general ability. The single wait-list-specific ability interaction (`gamma_grp_ability`) is inconclusive throughout.

## Mechanism and mediation — how the reading gain arises

These couplings are **adjusted associations, not causal drivers**, but they are internally consistent and answer the "through what?" question the trial itself cannot.

- **Mediation (g-formula NDE/NIE decomposition):** the intervention's effect on word reading runs **through letter-sound knowledge**. Single-mediator (med-059): the indirect path via L is +0.023 [+0.006, +0.045], P 0.998, ≈ 62 % of the total effect, with a non-credible direct effect. Two-mediator (med-064): the path via L is +0.025 [+0.006, +0.048] P 0.998, whereas the path via expressive vocabulary is ≈ 0 [−0.009, +0.007]. In plain words: the intervention improved word reading mainly _by_ improving letter-sound knowledge, not by improving vocabulary — the same conclusion the ITT ranking implies.
- **Mechanism GP slopes** (marginal, adjusted): E→W +0.271 [−0.001, +0.593] (the strongest, all but excluding 0), R→W +0.131, L→W +0.090 — all positive-leaning associations with word reading, credible intervals touching 0.
- **Latent coupled change-score model (lcsm-067):** prior letter-sound score → later reading change +0.135 [+0.025, +0.256] P 0.99, and prior expressive-vocabulary score → later reading change +0.284 [+0.054, +0.521] P 0.99 — both credible cross-lagged couplings; reading shows negative self-feedback (regression to the mean).
- **Dose-response:** pooled cumulative-session slope +0.127 [+0.028, +0.227] P 0.99 (dose-277); period-resolved slopes ≈ +0.13 per period (P ≈ 0.95). More sessions track more word-reading gain (dose is a partial collider, so this is a sensitivity view, not a clean causal dose curve).

## Secondary and cross-check families

- **Regularised-horseshoe predictor ranking (cross-check of the gradient-boosting ranking):** for word-reading _level_ (hs-002), letter sounds (P 0.99) and expressive vocabulary (P 0.99) are selected decisively, then grammar and age. For word-reading _gain_ (hs-001) nothing is selected (top probability 0.59, age) — echoing the gradient-boosting result that change scores are near-noise. The Bayesian and gradient-boosting rankings agree.
- **Correlated-domain-factor measurement model (mm-001):** the three latent domains are strongly correlated — vocabulary↔grammar 0.80, vocabulary↔code 0.74, code↔grammar 0.65 (all credible) — supporting a correlated-skill-system reading of the battery. (Structural coefficients held back pending the reparameterisation noted above.)
- **Multivariate growth curves (gc-069/070):** between-child gamma associations are inconclusive for R/E/W and credibly positive only for grammar (T ≈ +0.11).
- **Aligned per-protocol (`al` family):** cohort contrast +0.217 [−0.109, +0.544], correctly flagged **non-causal** (confounded by age-at-onset and cohort timing) — no term in this family is presented as causal, by design.
- **Adjusted between-child association (adj-065):** of the baseline predictors of word-reading gain, only age is credible (−0.259, older children gaining less); language composite (+0.225, P 0.93) and letter sounds (+0.159, P 0.89) lean positive but are not decisive.
- **Historical-cohort growth reproduction (rlm-hg-001, separate study):** converged cleanly (r̂ 1.00, 0 divergences).

## Caveats

- **This is preliminary, small-sample research** (≈ 159 children after cleaning; some outcomes far fewer). Credible intervals are wide, and the "credible" calls above rest on 95 % intervals that in several cases only just exclude zero.
- **Only τ (and the DiD δ, and the gain-factor on-intervention marginal) is causal.** Every mechanism slope, mediation path, coupling, growth gamma, horseshoe coefficient and adjusted association is an adjusted association and must not be read as "X drives Y".
- **`lrp-rli-mm-001` needs a sampling fix** before its structural coefficients are quoted; its correlations are fine. The nine other divergence-flagged fits are usable as-is but the mechanism GP models would benefit from a re-run at a higher `target_accept` if any single slope becomes load-bearing.
