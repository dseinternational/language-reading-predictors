# Gain-factor (ANCOVA) findings (2026-07-20)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

This note reports every model in the **gain-factor** family of the language-and-reading-predictors study. It is one of a set of per-family notes; the shared study description, the outcome-measure table, and the full "how to read the numbers" guide live in the index note, [`notes/202607200900-findings-00-index-and-reading-guide.md`](202607200900-findings-00-index-and-reading-guide.md). Read that first if any term below is unfamiliar. All data and models are preliminary (a work in progress).

## What this family does, and the question it answers

The gain-factor models are an **ANCOVA** re-analysis of the reading/phonics intervention. ANCOVA (analysis of covariance) here means: model each skill's score at the _end_ of a period from that same skill's score at the _start_ of the period, plus a set of covariates. Because the study has four measurement points, each child contributes several "period transitions" (start → end), and the model **stacks** them all — every on-intervention period and every untreated period — into one regression, with a **child random intercept** (a per-child offset that lets children who are generally higher or lower sit at their own level, pooling that stable difference across their rows). The score is a count of correct items out of a fixed maximum, so it is modelled on a **logit (log-odds) scale** and reported back as a change in the expected **proportion correct**, then translated into **items** (proportion × the test maximum).

The family answers two questions the intention-to-treat (ITT) suite does not tackle in one model. **First**, it gives a second, differently-built estimate of the causal intervention effect: the **on-intervention treatment marginal** — the average change in proportion-correct caused by being on the intervention, _averaged over the period-1 transition only_ (the genuinely randomised, all-untreated-baseline transition, which is the same contrast the ITT effect rests on). This is the single quantity in each model that randomisation licenses us to call **causal**. **Second**, and just as important, it reports a whole panel of **adjusted associations** — how each child's own baseline, age, cognitive ability, and the baseline levels of _other_ skills relate to how much they gain — which describe _who progresses_, not what the intervention does.

There are 19 models: 11 primary models (`gf-001`–`gf-011`, one per outcome) and 8 **treated-only companions** (`gf-101`–`gf-108`). The companions drop the untreated period rows and keep only rows where the child was on the intervention; because everyone in that subset is treated, there is no on-intervention contrast, so the companions have **no causal term at all** — every coefficient in them is an adjusted association. They exist to check that the associational structure is stable when we look only at gains made while receiving the intervention.

## How to read these numbers (short recap)

Full detail is in the index note; briefly: the point estimate is the posterior **median**; uncertainty is the **89% credible interval** (the range the value plausibly occupies, house standard, with the inner **50%** interval as the "most-of-the-mass" range); direction is the **tail probability** P(effect > 0), read directly, never as a p-value. The **evidence ladder** attaches a claim-strength label to that probability — inconclusive (< 0.75), suggestive (≥ 0.75), moderate (≥ 0.91), strong (≥ 0.97), very strong (≥ 0.99) — oriented to the favoured direction and describing _evidence of direction_, never effect size. Size is a separate claim: `P(benefit ≥ δ)` is the probability the gain clears a pre-set minimally-important difference δ, and **ROPE** mass (region of practical equivalence, a band around zero of effects too small to matter) quantifies "probably negligible". With only ~54 children, estimates that just clear a threshold are on average inflated, so lead with the interval.

**The one causal quantity in this family is the treatment marginal.** Every covariate coefficient below — own baseline, age, cognitive ability, other skills, hearing/speech/phonological memory, the interaction terms — is an **adjusted association**. Reading a control variable's coefficient as if it were a lever is the **Table-2 fallacy**; these coefficients describe patterns, holding the adjustment set fixed. Residual confounding by latent general ability remains for all of them, and the child random intercept only pools stable between-child differences — it does **not** stand in for general ability. Adjustment sets are fixed in advance from the causal diagram (DAG), so a skill missing from a given model was excluded by the diagram, not found unimportant.

## Per-model results

The outcomes and their item maxima: W word reading (max 79), R receptive vocabulary (170), E expressive vocabulary (170), L letter-sound knowledge (32), P phonetic spelling (92), B phoneme blending (10), F basic concepts (18), T receptive grammar (32), TR taught receptive vocabulary (24), TE taught expressive vocabulary (24), N nonword reading (6). For the two heavily-floored outcomes (**P** and **N**) the model uses the **floor rule**: instead of an item count it models the probability of being _off the floor_ (scoring above zero) at period end, so the "items" figure for P and N is a **change in off-floor probability** (a risk difference in percentage-point terms), not a count of items.

### Summary table — the causal treatment marginal

All 19 models **pass** the convergence gate (R-hat ≤ 1.01, effective sample size ≥ 400, BFMI ≥ 0.3, zero divergences); every fit had 0 divergences and healthy sampling (minimum effective sample size ≥ 1,390 across the family). "N" is (observations / children).

| Model  | Outcome              | N      | Treatment marginal, proportion median [89% CI] | Items median [89% CI]            | P(>0) | Evidence (direction)      | Gate |
| ------ | -------------------- | ------ | ---------------------------------------------- | -------------------------------- | ----- | ------------------------- | ---- |
| gf-001 | W word reading       | 153/53 | +0.033 [+0.011, +0.054]                        | +2.60 words [+0.87, +4.30]       | 0.993 | very strong ↑             | pass |
| gf-002 | R receptive vocab    | 161/54 | −0.008 [−0.031, +0.014]                        | −1.45 words [−5.26, +2.37]       | 0.169 | suggestive ↓ (negligible) | pass |
| gf-003 | E expressive vocab   | 161/54 | +0.007 [−0.012, +0.025]                        | +1.13 words [−2.11, +4.32]       | 0.577 | inconclusive              | pass |
| gf-004 | L letter-sounds      | 160/54 | +0.102 [+0.049, +0.155]                        | +3.27 sounds [+1.55, +4.97]      | 0.991 | very strong ↑             | pass |
| gf-005 | P spelling (floored) | 159/53 | −0.034 [−0.128, +0.055]                        | −3.4 pts off-floor [−12.8, +5.5] | 0.381 | inconclusive              | pass |
| gf-006 | B phoneme blending   | 161/54 | +0.083 [+0.008, +0.157]                        | +0.83 items [+0.08, +1.57]       | 0.903 | suggestive ↑              | pass |
| gf-007 | F basic concepts     | 160/54 | +0.060 [+0.007, +0.112]                        | +1.07 items [+0.12, +2.01]       | 0.865 | suggestive ↑              | pass |
| gf-008 | T receptive grammar  | 161/54 | +0.026 [−0.017, +0.068]                        | +0.82 items [−0.54, +2.18]       | 0.662 | inconclusive              | pass |
| gf-009 | TR taught receptive  | 161/54 | +0.044 [−0.005, +0.092]                        | +1.06 words [−0.11, +2.20]       | 0.633 | inconclusive              | pass |
| gf-010 | TE taught expressive | 161/54 | +0.048 [+0.000, +0.096]                        | +1.16 words [+0.01, +2.30]       | 0.818 | suggestive ↑              | pass |
| gf-011 | N nonword (floored)  | 153/53 | +0.022 [−0.085, +0.127]                        | +2.2 pts off-floor [−8.5, +12.7] | 0.621 | inconclusive              | pass |
| gf-101 | W, treated-only      | 130/53 | — (no causal term)                             | —                                | —     | associations only         | pass |
| gf-102 | R, treated-only      | 135/54 | — (no causal term)                             | —                                | —     | associations only         | pass |
| gf-103 | E, treated-only      | 135/54 | — (no causal term)                             | —                                | —     | associations only         | pass |
| gf-104 | L, treated-only      | 134/54 | — (no causal term)                             | —                                | —     | associations only         | pass |
| gf-105 | P, treated-only      | 134/53 | — (no causal term)                             | —                                | —     | associations only         | pass |
| gf-106 | B, treated-only      | 135/54 | — (no causal term)                             | —                                | —     | associations only         | pass |
| gf-107 | F, treated-only      | 134/54 | — (no causal term)                             | —                                | —     | associations only         | pass |
| gf-108 | T, treated-only      | 135/54 | — (no causal term)                             | —                                | —     | associations only         | pass |

### The causal treatment marginals, outcome by outcome

**Word reading — gf-001 (W).** The strongest single causal result in the family: an on-intervention gain of about **+2.6 words** on the EWRSWR test (proportion median +0.033; 89% CI +0.9 to +4.3 words; 50% CI +1.9 to +3.3), with P(effect > 0) = 0.993, **very strong** evidence of benefit. The size claim is also favourable: P(gain ≥ 1 word) = 0.93 (a moderate benefit label), with only 7% of the posterior inside the ROPE.

**Letter-sound knowledge — gf-004 (L).** The other clear causal win: about **+3.3 letter sounds** (proportion +0.102; 89% CI +1.6 to +5.0; 50% CI +2.6 to +4.0 of 32), P(>0) = 0.991, **very strong**. P(gain ≥ 2 sounds) = 0.88 (suggestive), ROPE mass 0.12.

**Phoneme blending — gf-006 (B).** A positive effect of about **+0.8 of 10 blends** (proportion +0.083; 89% CI +0.08 to +1.57), P(>0) = 0.903, **suggestive**. Direction is fairly clear, but size is not: P(gain ≥ 1 blend) is only 0.36 and 64% of the posterior sits inside the ROPE, so the effect is probably real but small (below one whole blend).

**Basic concepts — gf-007 (F).** About **+1.1 of 18 concepts** (proportion +0.060; 89% CI +0.12 to +2.01), P(>0) = 0.865, **suggestive**. There is no pre-set MID/ROPE for F (it sits outside the ROPE-anchored ITT suite), so this is a direction-only read.

**Taught expressive vocabulary — gf-010 (TE).** About **+1.2 of 24 taught expressive words** (proportion +0.048; 89% CI +0.01 to +2.30), P(>0) = 0.818, **suggestive**. P(gain ≥ 1 word) = 0.59, ROPE 0.41 — leaning positive, size uncertain.

**Taught receptive vocabulary — gf-009 (TR).** About **+1.1 of 24 taught receptive words** (proportion +0.044; 89% CI −0.11 to +2.20), P(>0) = 0.633, **inconclusive** (the interval spans zero; P(gain ≥ 1 word) = 0.53, ROPE 0.47).

**Receptive grammar — gf-008 (T).** About **+0.8 of 32** (proportion +0.026; 89% CI −0.54 to +2.18), P(>0) = 0.662, **inconclusive**. No pre-set ROPE for T.

**Receptive and expressive vocabulary — gf-002 (R), gf-003 (E).** Both **flat and probably negligible**, the broad standardised vocabulary story seen throughout the suite. For R the median is very slightly _negative_ (−1.4 words; 89% CI −5.3 to +2.4; P(>0) = 0.169, i.e. a suggestive lean toward a small decline), but over half the posterior (ROPE 0.52) is practically negligible and the harm-size probability is only 0.41, so this is best read as **no detectable effect**, not evidence of harm — the winner's-curse caution applies. For E the median is +1.1 words (89% CI −2.1 to +4.3; P(>0) = 0.577, inconclusive; ROPE 0.61).

**Floored outcomes — gf-005 (P spelling), gf-011 (N nonword).** Both use the off-floor rule and both are **inconclusive**. For phonetic spelling (P) the treatment marginal is a −3.4 percentage-point change in the probability of scoring off the floor (89% CI −12.8 to +5.5 pts; P(>0) = 0.381; ROPE mass 0.86 — overwhelmingly negligible). For nonword reading (N) it is +2.2 percentage points (89% CI −8.5 to +12.7; P(>0) = 0.621; ROPE 0.85). With so few children off the floor these are the least informative causal estimates in the family.

### The adjusted associations — who progresses (not causal)

The following describe _patterns of progress_, holding each model's DAG-fixed adjustment set constant. They are **not** levers and are subject to the Table-2 fallacy warning above; residual confounding by latent general ability remains.

**Own baseline (autoregression).** In every model the child's own starting score is the strongest predictor of their end score — `gamma_own` is positive with P(>0) = 1.000 (very strong) throughout, logit medians roughly +0.58 to +0.78. On the items scale a +1 SD higher baseline is associated with, for example, **+13.8 more words** in word reading (gf-001, 89% +12.1 to +15.6), +5.9 in receptive vocab, +4.6 in letter-sounds, and +1.1 to +2.3 in the smaller-range skills. This is largely the ANCOVA structure itself (higher starters end higher).

**Age.** A consistent, mild **negative** association: within the adjustment set, older children show slightly _smaller_ proportion-scale gains — clearest for word reading (gf-001, +1 SD age ≈ −1.1 words, P(<0) = 0.99), receptive grammar (gf-008, P(<0) = 0.996), letter-sounds (gf-004, P(<0) ≈ 0.95) and taught receptive vocab (gf-009, P(<0) ≈ 0.96). Read this as a ceiling/regression pattern in the adjusted data, **not** as age causing slower progress.

**Cognitive ability (WPPSI Block Design, `gamma_ability`).** Positively associated with progress, and most clearly on the **language and grammar** outcomes: basic concepts (gf-007, P(>0) = 1.000; +1 SD ≈ +0.67 items), receptive grammar (gf-008, P = 0.988), receptive vocab (gf-002, +1 SD ≈ +2.5 words, P = 0.988), expressive vocab (gf-003, +2.0 words, P = 0.975), taught receptive vocab (gf-009, P = 1.000), letter-sounds (gf-004, P = 0.949) and off-floor spelling (gf-005, P = 0.879). It is essentially flat for word reading (gf-001) and blending (gf-006). Higher general ability tracks more vocabulary/grammar progress; it does not track the code skills as strongly.

**Cross-skill couplings (baseline of _other_ skills).** These trace the developmental scaffolding the DAG encodes, all as associations:

- Word-reading gains (gf-001) track baseline **taught-expressive vocabulary** (gamma_TE, moderate, P = 0.929), **letter-sounds** (gamma_L, moderate, P = 0.912) and **nonword reading** (gamma_N, moderate, P = 0.913).
- Off-floor spelling (gf-005) tracks baseline **letter-sounds** (gamma_L, very strong, P = 0.999), **blending** (gamma_B, strong, P = 0.988) and **phonological memory** (gamma_erbto, strong, P = 0.989) — the code route.
- Nonword reading (gf-011) tracks baseline **letter-sounds** (very strong, P = 1.000), **phonological memory** (moderate, P = 0.970) and **speech** (deapp_c, moderate, P = 0.931).
- Expressive vocab (gf-003) tracks **receptive vocab** (moderate, P = 0.969) and **taught-expressive vocab** (very strong, P = 0.994); basic concepts (gf-007) and grammar (gf-008) both track **receptive vocab** (P = 0.980 / 0.999) and **taught-receptive vocab** (P = 0.989 / 1.000); taught-expressive vocab (gf-010) tracks **taught-receptive vocab** (very strong, P = 0.997) and vice-versa; and blending (gf-006) tracks **taught-expressive vocab** (moderate) and **letter-sounds** (moderate).

**Exogenous confounders (hearing `hs`, speech `deapp_c`, phonological memory `erbto`).** Mostly small. The most consistent is **phonological memory** (Early Repetition Battery total, `erbto`), positively associated with gains in receptive vocab (gf-002, moderate, P = 0.964), taught-receptive vocab (gf-009, very strong, P = 0.999), off-floor spelling (gf-005, strong) and nonword reading (gf-011, moderate). Hearing and speech coefficients are generally inconclusive.

**Interaction (moderation) terms — associations only.** The models include group×ability, group×own-baseline and age×ability interactions. The clearest pattern is a **negative group×own-baseline** interaction in several outcomes (word reading, taught expressive/receptive vocab, blending; e.g. gf-010 gamma_int_trt_own −0.25, P(<0) = 0.97), i.e. the on-intervention advantage is associated with being _smaller_ for children who start higher. Off-floor spelling shows the opposite (gf-005, strongly positive, P = 1.000). Because the causal treatment marginal is averaged over the all-untreated-baseline period-1 transition, these moderation terms colour the description without changing the headline causal read.

### The treated-only companions (gf-101–gf-108)

These re-fit the eight main item-count outcomes on **on-intervention rows only** (dropping the waitlist arm's untreated period 1), so N falls from ~160 to ~130–135 observations. As designed, the on-intervention term becomes constant and is dropped, so **there is no causal coefficient** — every row is an adjusted association describing gains _made while on the intervention_. The associational picture is essentially unchanged from the primaries: own baseline very strong everywhere (P = 1.000), age mildly negative, cognitive ability positive and if anything a little _stronger_ on the vocabulary outcomes (R gf-102 P = 0.977 and E gf-103 P = 0.973 both reach "strong", versus more muted in the pooled fits), and the same cross-skill couplings (letter-sounds → spelling/blending, receptive/taught vocab → grammar and basic concepts). The stability of these associations across the full-sample and treated-only views is the companions' main contribution.

## Comparison with the ITT suite — does the ANCOVA replicate the randomised result?

The gain-factor treatment marginal and the ITT effect τ are built differently (a period-stacked ANCOVA versus a single-outcome intention-to-treat model) but target the same randomised, all-untreated-baseline contrast, so they should agree. Both are reported here on the proportion-correct scale (the ITT figures come from each outcome's `tau_summary.csv`; F and T have **no** model in the core ITT-001–011 suite, so there is no direct comparator).

| Outcome                | Gain-factor marginal, proportion [89% CI] (evidence) | ITT τ, proportion [89% CI] (evidence)  | Verdict                           |
| ---------------------- | ---------------------------------------------------- | -------------------------------------- | --------------------------------- |
| W word reading         | +0.033 [+0.011, +0.054] (very strong)                | +0.030 [+0.009, +0.052] (strong)       | replicates closely                |
| L letter-sounds        | +0.102 [+0.049, +0.155] (very strong)                | +0.110 [+0.053, +0.166] (very strong)  | replicates closely                |
| B blending             | +0.083 [+0.008, +0.157] (suggestive)                 | +0.099 [+0.022, +0.174] (strong)       | same direction, a touch weaker    |
| TE taught expressive   | +0.048 [+0.000, +0.096] (suggestive)                 | +0.064 [+0.018, +0.111] (strong)       | same direction, weaker            |
| TR taught receptive    | +0.044 [−0.005, +0.092] (inconclusive)               | +0.057 [+0.008, +0.105] (moderate)     | weaker / inconclusive in ANCOVA   |
| R receptive vocab      | −0.008 [−0.031, +0.014] (inconclusive)               | +0.001 [−0.022, +0.025] (inconclusive) | both flat / negligible            |
| E expressive vocab     | +0.007 [−0.012, +0.025] (inconclusive)               | +0.001 [−0.018, +0.020] (inconclusive) | both flat / negligible            |
| P spelling (off-floor) | −0.034 [−0.128, +0.055] (inconclusive)               | +0.041 [−0.071, +0.155] (inconclusive) | both inconclusive (floored)       |
| N nonword (off-floor)  | +0.022 [−0.085, +0.127] (inconclusive)               | +0.100 [−0.038, +0.237] (suggestive)   | both short of moderate; GF weaker |
| F basic concepts       | +0.060 [+0.007, +0.112] (suggestive)                 | — (no ITT model)                       | no comparator                     |
| T receptive grammar    | +0.026 [−0.017, +0.068] (inconclusive)               | — (no ITT model)                       | no comparator                     |

The re-analysis **broadly replicates the randomised result**. The two headline benefits — **word reading** (+2.6 words vs the ITT's +2.4 words) and **letter-sound knowledge** (+3.3 vs +3.5 sounds) — come out almost identically, and the two flat vocabulary outcomes (R, E) are flat and negligible in both. The systematic difference is that the gain-factor marginals are **slightly attenuated** on the taught-vocabulary and blending outcomes (TR moves from moderate in ITT to inconclusive here; TE and B drop one rung), consistent with the ANCOVA's heavier conditioning (own baseline + upstream skills + ability) absorbing some of the signal, and with the period-1-only averaging using fewer transitions than the full ITT contrast. The direction never reverses on any outcome where either model has more than inconclusive evidence.

## What the family concludes

The gain-factor ANCOVA delivers a **second, independent causal estimate** that confirms the intervention's clearest effects: strong-to-very-strong benefit on **word reading** and **letter-sound knowledge**, weaker-but-positive signals on **blending, basic concepts and taught expressive vocabulary**, and **no detectable effect on broad standardised vocabulary (R, E)**. This is the same gradient the ITT suite and the difference-in-differences family report, so the headline is robust to how the model is built. The heavily-floored spelling (P) and nonword (N) outcomes remain inconclusive in every framing.

Its distinctive contribution is the **who-progresses** panel. Beyond the intervention, progress tracks the child's own baseline (universally), cognitive ability (especially for vocabulary and grammar), and — as an associational scaffolding consistent with the mediation family's letter-sound route — baseline **letter-sounds, blending and phonological memory** predict gains in the **code skills** (spelling, nonword and word reading), while the **vocabulary skills** predict each other and predict grammar and basic concepts. These are descriptions of developmental structure, not causal levers.

## Caveats and convergence

All 19 fits **pass** the gate cleanly (0 divergences each; R-hat ≤ 1.005 on the primaries, ≤ 1.003 on the companions; minimum effective sample size comfortably above 400 throughout — the tightest being gf-003 at ~1,390), so there are no convergence flags to hold. The binding limits are substantive, not computational: (1) **small sample** — ~54 children and 130–161 stacked rows, so single-point estimates that just clear a threshold are on average inflated; lead with the intervals. (2) **Floor effects** — for P and N most children score at or near zero, so the off-floor estimates are weak and should not be over-read. (3) **Identification** — only the treatment marginal is causal; every covariate coefficient is an adjusted association liable to the Table-2 fallacy, and the child random intercept pools stable between-child differences but does **not** control latent general ability, so residual confounding remains for all associations. (4) **Adjustment sets are DAG-fixed** — a skill absent from a model (for example, no non-measure confounders for word reading) was excluded by the causal diagram, not found unimportant. F and T have no ROPE/MID anchor and no ITT comparator, so those two are direction-only reads.
