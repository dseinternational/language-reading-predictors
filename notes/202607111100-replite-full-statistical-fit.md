# Full statistical-model fit in `rep-lite` — run summary and findings (2026-07-11)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

## Run metadata

- **Command:** `python scripts/fit_statistical_model.py all --config rep-lite`
- **Sampling preset (`rep-lite`):** 4 chains × 4000 draws × 4000 tune, `target_accept = 0.95` — reporting-grade rigour (same acceptance target as `reporting`) but lighter draws, since ESS, not raw draws, is the binding convergence metric.
- **Scope:** all 89 registered models across 16 families (96 output directories, counting per-model sensitivity/companion sub-fits).
- **Wall time:** 35 min 49 s (16-core workstation); exit code 0, 0 fits failed. Reports were **not** rendered (`--render` omitted); findings below read the result CSVs directly.
- **Cross-model comparison:** `python scripts/compare_statistical_models.py --config rep-lite` ran cleanly (`itt_vs_joint_tau.csv`, `tau_forest.png`, `mechanism_forest.{png,csv}`, nested PSIS-LOO tables). Extended with a **mediation-family comparison** (`mediation_family.csv` + `mediation_family_forest.png`) — every g-formula route to word reading lined up on the shared words scale (decomposition summaries, not a nested-LOO test).
- Preliminary research data — all estimates provisional.

## Convergence gate (checked before interpreting)

**86 / 96 pass the full gate cleanly** (r̂ ≤ 1.01, ESS ≥ 400, BFMI ≥ 0.30, zero divergences). The 10 flags:

- **9 are divergence-only, all ≤ 0.41 %** of 16 000 draws — the HSGP/GP mechanism surfaces (`mech-056/057/058/071/073/173`) and the dose slopes (`did-007`, `dose-077/177`). r̂ = 1.00, min ESS 720–2400 throughout. Within the METHODS ≤ 1 % divergence guidance — **usable, noted**.
- **1 is a genuine concern — `mm-001` (corr_factor):** 0.94 % divergences **plus** sub-threshold BFMI (min 0.15 across chains) **and** two parameters below the ESS floor (`sigma_indicator[R]`, `sigma_indicator[E]`, ESS ≈ 320). This is the known latent-factor funnel. Its **domain correlations are trustworthy; the structural leg is held** pending a non-centred reparameterisation or higher `target_accept`.

Net: **95 / 96 fits usable**; `mm-001`'s structural coefficients set aside.

## Headline finding

The pattern is coherent across four independent identification strategies (randomised ITT, within-person DiD, gain-factor ANCOVA, and g-formula mediation): **strong-to-very-strong evidence that the reading/phonics intervention improves the directly-taught reading-precursor skills — letter-sound knowledge, word reading, phoneme blending, and taught expressive vocabulary — and inconclusive, probably-negligible effects on broad standardised vocabulary.** The word-reading gain runs through letter-sound knowledge, not through expressive vocabulary.

Only τ (ITT), the DiD δ, and the gain-factor on-intervention marginal are **causal**. Every covariate coefficient, mechanism slope, and level-factor post-crossover term is an **adjusted association** (Table-2-fallacy territory) — it describes _who progresses_, not a lever.

## ITT suite — causal τ (risk-difference / probability scale, sample-mean baseline)

| Outcome                        | τ (prob)   | 95 % CI          | P(τ>0)    | Direction evidence | Magnitude vs δ                 | ROPE mass |
| ------------------------------ | ---------- | ---------------- | --------- | ------------------ | ------------------------------ | --------- |
| **L** letter-sound knowledge   | **+0.111** | [+0.040, +0.179] | **0.998** | very strong        | moderate (≈ +3.5 of 32 sounds) | 0.09      |
| **W** word reading             | **+0.030** | [+0.004, +0.057] | **0.988** | strong             | suggestive (≈ +2.4 words)      | 0.10      |
| **B** phoneme blending         | **+0.099** | [+0.004, +0.190] | **0.979** | strong             | inconclusive (≈ +1.0 item)     | 0.51      |
| **TE** taught expressive vocab | **+0.065** | [+0.006, +0.121] | **0.986** | strong             | suggestive (≈ +1.6 items)      | 0.21      |
| **TR** taught receptive vocab  | +0.057     | [−0.004, +0.115] | 0.966     | moderate           | inconclusive                   | 0.30      |
| **N** (floored)                | +0.056     | [−0.108, +0.218] | 0.754     | suggestive         | inconclusive                   | 0.67      |
| **UR** untaught receptive      | +0.050     | [−0.014, +0.114] | 0.939     | moderate           | inconclusive                   | 0.85      |
| **UE** untaught expressive     | +0.026     | [−0.041, +0.093] | 0.771     | suggestive         | inconclusive                   | 0.95      |
| **R** receptive vocab (std.)   | +0.001     | [−0.028, +0.031] | 0.531     | inconclusive       | inconclusive                   | 0.58      |
| **E** expressive vocab (std.)  | +0.001     | [−0.022, +0.025] | 0.542     | inconclusive       | inconclusive                   | 0.67      |
| **P** (floored)                | −0.008     | [−0.172, +0.153] | 0.463     | inconclusive       | inconclusive                   | 0.77      |

Read direction from the tail probability, not from whether the band clears zero. The point estimates that clear a threshold are, on small n, on average magnitude-inflated (winner's curse) — lead with the interval.

**The taught/untaught vocabulary gradient is informative:** the effect concentrates on directly-taught items (TE strong, TR moderate) and fades on untaught and broad standardised vocabulary (UE/UR suggestive→moderate; R/E flat with high ROPE mass) — i.e. the intervention teaches what it teaches, with little evidence of broad transfer at this sample size. R and E are **not "null"** — they are _inconclusive and probably negligible_ (ROPE-quantified), a distinct claim.

## Robustness — L and W hold under every pre-specified adjustment

| Adjustment                                   | L: τ (P)       | W: τ (P)       |
| -------------------------------------------- | -------------- | -------------- |
| Base (ITT-007 / 010)                         | +0.111 (0.998) | +0.030 (0.988) |
| General ability / block-design (ITT-023/024) | +0.111 (0.998) | +0.028 (0.981) |
| Site/area (ITT-028/027)                      | +0.109 (0.998) | +0.032 (0.991) |
| SES complete-case, adjusted (ITT-113/013)    | +0.106 (0.991) | +0.032 (0.968) |
| SES complete-case, unadjusted (ITT-114/014)  | +0.124 (0.999) | +0.030 (0.976) |

The letter-sound and word-reading effects are unmoved by ability, geography, or SES adjustment.

## Within-person replication — DiD δ (waitlist crossover, items scale)

| Outcome                  | δ (items) | 95 % CI        | P(δ>0)    | Evidence     |
| ------------------------ | --------- | -------------- | --------- | ------------ |
| **L** letter-sound       | **+3.38** | [+1.10, +5.67] | **0.998** | very strong  |
| **W** word reading       | **+2.75** | [+0.42, +4.96] | **0.990** | strong       |
| **B** blending           | +0.95     | [+0.04, +1.85] | 0.980     | strong       |
| **TE** taught expressive | +1.47     | [−0.02, +2.93] | 0.974     | strong       |
| **R** receptive (std.)   | −0.13     | [−5.26, +4.90] | 0.485     | inconclusive |

The within-person DiD reproduces the randomised ITT ordering — each child acting as its own control, the ceiling respected by the Beta-Binomial logit.

## Second replication — gain-factor ANCOVA, on-intervention marginal (the only causal term)

W +0.037 (P = 0.996, ≈ +2.9 words) · L +0.096 (P = 0.996, ≈ +3.1 sounds) · B +0.072 (P = 0.908) · F +0.049 (P = 0.862) · P +0.031 (P = 0.752) · T +0.012 (P = 0.682) · E +0.002 (P = 0.547) · **R −0.006 (P = 0.348)**. Same signal: strong for W/L, moderate-to-suggestive for B/F, flat for standardised vocabulary.

## Mechanism of the word-reading gain — mediation (g-formula, words out of test length)

The two-mediator decomposition (`med-064`) cleanly separates the routes:

| Quantity                     | words     | 95 % CI        | P(>0)     |
| ---------------------------- | --------- | -------------- | --------- |
| Total effect on W            | +2.98     | [+0.59, +5.43] | 0.992     |
| **NIE via letter-sound (L)** | **+1.94** | [+0.52, +3.81] | **0.998** |
| NIE via expressive vocab (E) | −0.02     | [−0.65, +0.57] | 0.468     |
| Natural direct effect (NDE)  | +1.06     | [−0.96, +3.13] | 0.848     |

The word-reading benefit is **mediated by letter-sound knowledge** (very strong evidence for the indirect effect via L; ≈ 65 % of the total), while the expressive-vocabulary route is inconclusive and ≈ 0. Single-mediator (`med-059`, NIE via L: +1.78, P = 0.998) and reading-route composite (`med-062`, NIE: +1.01, P = 0.965) agree. Mediation paths are g-formula estimands under the DAG's no-unmeasured-confounding assumption, not a manipulation experiment.

## Internal consistency

Single-outcome vs joint-model τ agree closely (logit scale): L 0.586 vs 0.578, W 0.357 vs 0.356, R/E ≈ 0 in both — the joint LKJ-correlated fit reproduces the per-outcome estimates. Mechanism forest (associations, not causal): E→W slope +0.26 [−0.00, +0.59], R→W +0.13 [−0.03, +0.41], L→W +0.09 [−0.01, +0.27].

## Deep review — gain-factor, level-factor, and mediation predictor structure

Reminder on what is causal: in gain-factors **only `beta_trt`** (the on-intervention marginal) is causal; in level-factors **only `b_grp_time[1]`** (the t2 contrast, before crossover). Every `gamma_*`, every interaction, and every post-crossover timepoint is an **adjusted association** — it describes _who_ progresses, not a lever (Table-2-fallacy territory). Covariate sets are DAG-pre-specified, so a skill absent from a model was excluded by the diagram.

### Gain-factors (ANCOVA, GF-001–008) — a third independent replication

Causal on-intervention term (logit; items from the marginal):

| Outcome                    | `beta_trt` (logit)    | P(>0) | Evidence               | ≈ items     |
| -------------------------- | --------------------- | ----- | ---------------------- | ----------- |
| **L** letter-sound         | +0.519 [+0.12, +0.91] | 0.996 | very strong            | +3.1 sounds |
| **W** word reading         | +0.438 [+0.11, +0.78] | 0.996 | very strong            | +2.9 words  |
| **B** blending             | +0.303 [−0.14, +0.74] | 0.908 | suggestive             | +0.7        |
| **F** CELF concepts        | +0.179 [−0.14, +0.50] | 0.862 | suggestive             | +0.9        |
| **P** spelling (off-floor) | +0.287 [−0.55, +1.14] | 0.752 | suggestive             | ≈0          |
| **T** TROG grammar         | +0.054 [−0.17, +0.29] | 0.682 | inconclusive           | +0.4        |
| **E** expressive vocab     | +0.009 [−0.14, +0.16] | 0.547 | inconclusive           | +0.4        |
| **R** receptive vocab      | −0.032 [−0.20, +0.13] | 0.348 | inconclusive (leans −) | −1.0        |

Same ordering as ITT and DiD for a third time: L/W very strong, B/F/P suggestive, R/E/T flat (R nudges slightly negative — noise at this n). The three identification strategies converge.

Adjusted associations (who progresses — not causal):

- **`gamma_own` (own baseline): +0.60 to +0.86, P = 1.000 everywhere.** Baseline is the dominant predictor of the post-score, exactly as an ANCOVA expects — the precision workhorse, not a finding.
- **`gamma_A` (linear age) is negative for the reading/grammar outcomes** — W −0.123 (P(neg) = 0.99, strong), T −0.129 (very strong), L −0.101, F −0.076. Conditional on own baseline, older children in this cohort show smaller residual gains on these skills (a maturation/ceiling association, why age is in the adjustment set as a precision term).
- **`gamma_ability` (block-design) tracks the code/concept skills:** F +0.485 (very strong), P +0.301, L +0.234 (moderate); near-zero for R/E/T/W.
- **Upstream-skill couplings match the DAG (associations):** `gamma_L`→W +0.090 (strong), →B +0.157 (very strong), →P +0.323; `gamma_R`→E +0.209, →F +0.412, →T +0.457 (all very strong). Letter-sound tracks the reading-code skills; receptive vocabulary breadth tracks the language skills. Foreshadows the mediation result, correlational here.
- **Interactions are mostly inconclusive** with wide intervals; the few that flag (`trt×ability` on T +0.240 P = 0.992 vs on F −0.296 P = 0.017 point _opposite_ ways; `trt×own` on B −0.293 P = 0.044) are exploratory moderation on ~52 children — no claim should rest on them.
- **Treated-only companions (GF-101–108)** drop the contrast and refit on the treated arm; `gamma_own` and the ability/vocabulary couplings reproduce cleanly (ability→R +0.111 P = 1.000, ability→E +0.077 P = 0.993) — a reassuring sensitivity check.

### Level-factors (LF-001–008) — the levels view, and a passing placebo check

Clean randomised t2 contrast `b_grp_time[1]`:

| Outcome                | t2 contrast           | P(>0) | Evidence     |
| ---------------------- | --------------------- | ----- | ------------ |
| **L** letter-sound     | +0.483 [+0.01, +0.97] | 0.978 | strong       |
| **B** blending         | +0.304 [−0.18, +0.78] | 0.888 | suggestive   |
| **W** word reading     | +0.217 [−0.26, +0.69] | 0.813 | suggestive   |
| **F** concepts         | +0.132 [−0.22, +0.48] | 0.769 | suggestive   |
| **T** grammar          | +0.025                | 0.577 | inconclusive |
| **P** spelling         | −0.006                | 0.495 | inconclusive |
| **R** receptive vocab  | −0.036                | 0.353 | inconclusive |
| **E** expressive vocab | −0.003                | 0.486 | inconclusive |

Same sign and ranking as every other view (L strong; W/B/F suggestive-positive; R/E/P/T flat) but with **wider intervals and lower tail probabilities** — expected, because the levels parameterisation conditions on _no_ own-baseline, so it is less precise than the ANCOVA-style ITT/gain-factors. The two views agree on the story and differ only in precision, by design.

**Placebo check — pre-treatment `b_grp_time[0]` (t1):** all eight are statistically indistinguishable from zero (every CI spans 0). Where they lean, they lean slightly _negative_ (W −0.215, P −0.517, F −0.147, B −0.138) — the immediate arm started a touch lower if anything, so there is no positive baseline imbalance inflating the t2 effects; if real, the small gaps are conservative. Post-crossover `[2]`/`[3]` are associations (the waitlist arm is now treated too) — L stays positive; a lone R +0.150 (moderate) at t4 should not be over-read.

### Mediation (MED-059 / 062 / 064) — the word-reading effect runs through letter-sound knowledge

Primary decomposition (words out of test length; g-formula under the DAG):

| Model                        | Total          | NIE (indirect)                                  | NDE (direct)   | Prop. mediated |
| ---------------------------- | -------------- | ----------------------------------------------- | -------------- | -------------- |
| **059** single mediator L    | +2.85, P=0.992 | **+1.78 via L, P=0.998**                        | +1.07, P=0.856 | ≈0.62          |
| **062** code-route composite | +2.65, P=0.988 | +1.01, P=0.965                                  | +1.64, P=0.933 | ≈0.37          |
| **064** two-mediator L vs E  | +2.98, P=0.992 | **NIE_L +1.94, P=0.998** / NIE_E −0.02, P=0.468 | +1.06, P=0.848 | ≈0.64          |

The two-mediator model is decisive: the indirect effect is **entirely the letter-sound path** (NIE_L +1.94, very strong) with the **expressive-vocabulary route flat at ≈0** (NIE_E −0.02, inconclusive). ~60–64 % of the total word-reading effect is mediated by L.

**Temporal-ordering sensitivity (t3 — mediator measured _strictly before_ the outcome)** is the strongest panel:

| Model | NIE via L (t3)     | NDE (t3)               |
| ----- | ------------------ | ---------------------- |
| 059   | **+3.07, P=0.999** | −0.41, P=0.378 (≈0, −) |
| 062   | **+2.17, P=0.997** | +0.39, P=0.601 (≈0)    |

Once temporal ordering is enforced, the **direct effect collapses to essentially zero** and **all** of the word-reading benefit routes through letter-sound knowledge (proportion mediated ≈1.0). The mediation is not an artefact of contemporaneous measurement — tightening the timing strengthens the indirect path and removes the direct one.

**Caveat on the mediation claim:** the treatment→mediator and treatment→outcome legs are randomised, but the **mediator→outcome leg is observational** — the NDE/NIE split rests on sequential ignorability (no unmeasured mediator–outcome confounding), a DAG assumption, not an experiment (n = 53). Phrase as mechanism under stated assumptions, not proof of manipulation.

### Cross-family bottom line

The causal terms triangulate cleanly — gain-factors (`beta_trt`) and level-factors (t2 `b_grp_time[1]`) both reproduce the ITT/DiD ranking (L strong, W/B/F next, R/E/P/T flat), with level-factors simply noisier for want of a baseline covariate, and its placebo check passing. The associations tell a coherent developmental story (baseline tracks strongly; ability tracks code skills; upstream vocabulary tracks language skills) but are explicitly not levers. The mediation pins the reading mechanism: the word-reading gain flows through letter-sound knowledge, not vocabulary, with the direct path vanishing under strict temporal ordering.

## New models — closing two coverage gaps (added 2026-07-11)

Two models were drafted and fitted at `rep-lite` to answer questions the existing suite did not: (1) does word reading route through a phonics skill **other than letter sounds** — specifically phoneme blending — independently of letter sounds? and (2) does the intervention's effect differ between **taught-expressive (TE)** and **taught-receptive (TR)** vocabulary? Both required small infrastructure work: the two-mediator factory/decomposition/pipeline was hard-coded to the L/E mediator pair, so it was parameterised on the second mediator symbol (byte-identical for the existing `med-064` L/E build); the TE-vs-TR contrast reuses the generic joint pipeline. Both pass the full convergence gate (0 divergences, r̂ = 1.00, min ESS 7900 / 8900).

### LRP-RLI-MED-066 — two-mediator split: letter sounds (L) vs phoneme blending (B)

The phonics analogue of `med-064` (which split L vs expressive vocab E). Adjustment {G, A, E, R, W_pre, L_t1, B_t1}; g-formula NDE/NIE (words out of test length):

| Quantity                     | words     | 95 % CI        | P(>0)     |
| ---------------------------- | --------- | -------------- | --------- |
| Total effect on W            | +2.91     | [+0.59, +5.24] | 0.993     |
| NIE_joint (L + B block)      | +1.65     | [+0.15, +3.44] | 0.984     |
| **NIE via letter-sound (L)** | **+1.85** | [+0.47, +3.65] | **0.998** |
| NIE via **blending (B)**     | **−0.20** | [−1.13, +0.51] | 0.292     |
| Natural direct effect (NDE)  | +1.26     | [−0.83, +3.42] | 0.880     |

**Finding: blending adds no independent route to word reading beyond letter sounds.** The joint {L, B} indirect effect (+1.65) is essentially the letter-sound path alone (+1.85); the blending-specific path is ≈ 0 (point estimate slightly negative, interval spanning zero, inconclusive). This mirrors the L-vs-E result exactly: whichever second mediator is paired with letter sounds — vocabulary (E) or blending (B) — it contributes **nothing incremental**. The mediated portion of the word-reading effect is letter-sound knowledge, full stop. Caveat: the path-specific split is exploratory and ordering-dependent, and L and B are strongly correlated phonics skills (L is ordered first and absorbs shared variance), so the near-zero NIE_B is especially uncertain — but the substantive conclusion (no independent blending route) is the same under either the composite `med-062` or this decomposition.

### LRP-RLI-ITT-016 — modality contrast: taught-expressive (TE) vs taught-receptive (TR)

Joint Beta-Binomial over the randomised window; both τ are causal (randomised-window ITT), the contrast read on the logit scale (different item denominators). LKJ residual correlation off (convergence-safe fallback, matching the `itt-015` companions).

| Term                    | logit      | 95 % CI          | P(>0) | Evidence         |
| ----------------------- | ---------- | ---------------- | ----- | ---------------- |
| τ[TE] taught-expressive | +0.326     | [+0.036, +0.617] | 0.985 | strong           |
| τ[TR] taught-receptive  | +0.254     | [−0.014, +0.520] | 0.967 | moderate         |
| **TE − TR contrast**    | **+0.072** | [−0.320, +0.467] | 0.640 | **inconclusive** |

**Finding: the intervention moves the directly-taught words by a similar amount in both modalities.** Both taught outcomes benefit (expressive strong, receptive moderate — consistent with the marginal ITTs), and the expressive-minus-receptive difference is small and inconclusive (P = 0.64) — no meaningful evidence the programme favoured production over comprehension of the taught words. This complements the within-modality generalisation contrasts (`itt-015`/`115`): teaching-specificity showed up as taught-vs-untaught **within** the expressive modality, but there is no comparable expressive-vs-receptive gap **among the taught words**.

Together these close the two gaps flagged in the mediation/contrast review: the reading mechanism is letter-sound-specific (blending carries no independent route), and the taught-vocabulary effect is modality-symmetric.

## Mediation deep-dive — five new routes (Tier-1 & Tier-2, added 2026-07-11)

Five further mediation analyses were drafted and fitted at `rep-lite` to probe what carries the word-reading (W) gain beyond the single-mediator letter-sound story. All pass the full convergence gate (0 divergences, r̂ = 1.00, min ESS 5800–8500). All are **ID-2 adjusted associations** (GA-confounded; and, except #4, natural effects assume away the treatment-induced dose confounder) — decompositions under stated assumptions, never causal routes; n ≈ 50–53 → wide intervals. Words are out of test length.

| Model            | Route / estimand                   | Indirect effect                | P(>0) |
| ---------------- | ---------------------------------- | ------------------------------ | ----- |
| **MED-068** (#3) | via taught-expressive vocab (TE)   | NIE +0.64 [−0.36, +1.94]       | 0.897 |
| **MED-080**      | via taught-receptive vocab (TR)    | NIE +0.54 [−0.29, +1.88]       | 0.875 |
| **MED-074** (#1) | via nonword decoding (N)           | NIE +0.13 [−0.30, +0.78]       | 0.706 |
| **MED-075** (#2) | sequential L → B → W (joint block) | NIE_joint +1.64 [+0.16, +3.40] | 0.986 |
| **MED-076** (#5) | longitudinal L(t2) → W(t4)         | NIE +3.20 [+0.93, +5.97]       | 0.998 |
| **MED-078** (#4) | interventional (IIE) via L         | IIE +1.71 [+0.48, +3.34]       | 0.998 |

**#3 Taught-vocabulary route (MED-068 expressive; MED-080 receptive companion).** The DAG's taught → reading edges (`TE → WR` from decision 5; `TR → WR` predating it) each carry a route, but a **modest and inconclusive** one, and the two modalities are **symmetric**: taught-expressive NIE +0.64 (P=0.897, ~26% of the +2.44 total) and taught-receptive NIE +0.54 (P=0.875, ~22%), with most of the effect direct/residual in both (NDE +1.81 / +1.88). The intervention moves taught vocabulary through **production and comprehension alike**, but neither routes a decisive share into reading. Note the taught-receptive → reading _association_ is if anything the firmer of the two (b_M +0.45, P=0.998 vs TE +0.36, P=0.979) — so the inconclusive NIE reflects the diffuse total effect, not a weak mediator–outcome link. The lexical route is not zero, but neither is it established — the phonics route dominates.

**#1 Decoding route (MED-074).** The route through nonword decoding — the skill the DAG labels the "code-route mediator" — is **≈0 and inconclusive** (NIE +0.13, P=0.706), with the effect essentially all direct (NDE +2.59, P=0.987). This is **floor-limited** (N is 62% floored, n_trials=6): treatment barely moves measured decoding off the floor, so it cannot carry the reading gain — consistent with the documented DS whole-word/sight-word reliance and the `NW` floor, and with the DS-uncertain forward direction. Read as floor-limited, not as evidence against a decoding mechanism.

**#2 Sequential code route (MED-075).** Reframing L and B as a **chain** (adding the `L → B` edge) rather than parallel competitors: the `L → B` coupling is positive but only **suggestive** (aB_L +0.122, P=0.818), the joint `{L, B}` indirect (+1.64) is **identical to the parallel MED-066 joint** (+1.65), and the blending-beyond-L path stays ≈0 (NIE_B −0.21, P=0.288). Together with MED-066 this settles the picture: blending is (suggestively) _downstream_ of letter sounds and adds no independent route — the code-route mediation is letter-sound-driven whether B is modelled as parallel or sequential. (Per-path split is exploratory / convention-dependent; the joint block and the coupling are the interpretable outputs.)

**#5 Longitudinal ordering (MED-076).** With the mediator held at t2 and the outcome pushed to **t4** (two waves later, so the mediator strictly precedes the outcome), the letter-sound indirect effect is the **strongest of any mediation fit** (NIE +3.20, P=0.998) and the direct effect **vanishes** (NDE −0.77, P=0.313). Tightening temporal precedence does not weaken the letter-sound route — it removes the residual direct path entirely. Caveat: the t2→t4 increment is not randomised (both arms treated by then); this is a within-design triangulation, **not** the full wave-unrolled model (issue #250).

**#4 Interventional effects (MED-078).** The estimand-class repair for the DAG's own ID-2 point that dose is a treatment-induced mediator-outcome confounder (under which the natural NDE/NIE are _not_ identified). The randomised interventional indirect effect **IIE +1.71 (P=0.998) is within Monte-Carlo noise of the natural NIE +1.78** — so the natural-effect letter-sound decomposition was **not** materially distorted by the dose confounder. (The interventional effects fix only the treatment-induced-confounder leg; both remain GA-confounded associations.)

**Cross-cutting.** Across all seven mediators now tested — L, and as a _second_ route E (MED-064), B parallel (MED-066), B sequential (MED-075), TE (MED-068), TR (MED-080), N (MED-074) — **no measure carries an independent route to word reading beyond letter-sound knowledge.** The letter-sound route strengthens under stricter temporal ordering (t4) and survives the interventional estimand-class fix; every alternative route (decoding behaviour, taught expressive/receptive or standardised vocabulary, blending) is ≈0 or, for the taught-vocab pair, modest-and-inconclusive. Letter-sound knowledge is the mediator of record.

_Infrastructure note:_ these reused the single- and two-mediator g-formula machinery, generalised so the mediator symbol is no longer hard-coded to L/E (MED-064/066 remain byte-identical), plus new options for a lagged-outcome primary fit (`outcome_time`), an `L → B` chain edge, and randomised-interventional draws. See the accompanying PR.

## Interrogating the letter-sound story — causality, measures, heterogeneity (added 2026-07-11)

Three probes of whether the letter-sound mediation result can be over-read.

### (a) How far can we call letter sounds _causal_, not just associational?

The intervention→letter-sounds leg is already causal (randomised, $\tau_L$, P=0.998); the **letter-sounds→reading** leg is the confounded one (DAG ID-2 names latent general ability $GA$ and treatment-induced dose). Two new analyses bound how much of that leg could be confounding.

**Negative-control mediator (LRP-RLI-MED-079, rep-lite, gate PASS).** The DAG _severs_ grammar (TROG, `T`) from word reading — it has **no causal path** — yet grammar is strongly correlated with DS reading (Byrne et al. 2002). So the residual grammar→reading association after the _same_ adjustment set is a direct readout of leftover $GA$ confounding. Comparing the mediator→outcome slope $b_M$ (per SD of standardised mediator, logit scale):

| Mediator                       | $b_M$ (median) | 95 % CI        | P($b_M$>0) | E-value (point / CI) |
| ------------------------------ | -------------- | -------------- | ---------- | -------------------- |
| **L — letter sounds**          | **+0.60**      | [+0.29, +0.91] | **1.000**  | **2.04 / 1.59**      |
| T — grammar (negative control) | +0.25          | [−0.04, +0.53] | 0.956      | 1.52 / 1.16          |
| TE — taught expressive vocab   | +0.36          | [+0.02, +0.73] | 0.979      | 1.68 / 1.09          |
| TR — taught receptive vocab    | +0.45          | [+0.16, +0.75] | 0.998      | 1.82 / 1.38          |
| N — nonword decoding           | +0.27          | [+0.05, +0.49] | 0.992      | 1.55 / 1.19          |

The letter-sound association is **~2.4× the grammar negative-control** and unambiguous (P=1.000), where grammar's is half the size with a CI touching zero. Read as a calibration: even a causally-inert mediator carries a residual $b_M\approx0.25$, so the adjustment set does **not** fully close the $GA$ back-door — but letter sounds sits well above that confounding floor. (The full NIE _through_ grammar is ≈0 — total +2.23, NIE +0.14, P=0.712 — but that is uninformative because the intervention barely moves grammar; $b_M$ is the clean negative-control quantity.)

**E-value.** An unmeasured confounder ($GA$) would need to be associated with **both** letter-sound level and word reading by a risk ratio of at least **≈2.0** (point) — **≈1.6** to drag the CI to null — over and above the measured age/vocabulary/baseline adjustment, to explain away the letter-sound association. Grammar's placebo association only needs ≈1.5 / ≈1.2. So letter sounds is the more confounding-robust route, but an E-value of ~2 is a _moderate_ bar that a strong DS ability factor could plausibly meet — this is triangulation, not proof. (E-values use the logit→$d$→RR approximation; treat as a rough guide.)

**Verdict on (a):** together with the temporal-precedence (MED-076, direct effect vanishes at t4), the dose-confounder-robust interventional fit (MED-078), and the low-ability-benefits-most gradient below (inconsistent with pure ability-sorting), the letter-sound route is defensibly **more than a bare association** — but the grammar residual (0.25 ≠ 0) and a mid-range E-value mean we stop short of a causal claim. A within-person / $GA$-differencing design (removing stable ability) is the remaining lever.

### (2) Could vocabulary's weakness be a _measurement_ artefact? — partly, yes.

Effective resolution differs sharply across measures (phase-0 window, n=50):

| Measure                    | Denominator | Range actually used | Floor    |
| -------------------------- | ----------- | ------------------- | -------- |
| Letter sounds (L)          | 32          | **91 %**            | 0 %      |
| Taught vocab (TE / TR)     | 24          | 62–67 %             | 0 %      |
| Standardised vocab (R / E) | 170         | **31–36 %**         | 0 %      |
| Nonword decoding (N)       | 6           | —                   | **62 %** |

The **standardised vocabulary measures occupy only ~a third of their scale** in this cohort (normed on a broad population; DS children cluster low) — severe **range restriction**, which attenuates _both_ the detectable ITT effect and the mediator→outcome slope (regression dilution). And the intervention _did_ move vocabulary where it is well-measured (taught TE showed a strong ITT effect). So the honest reading is **"no detectable diffuse transfer to coarse standardised measures,"** not "no vocabulary role" — we cannot distinguish a true null from measure insensitivity. N is measurement-crippled outright (6 items, 62 % floored), so its ≈0 route is uninformative about decoding. Next step: reliability/errors-in-variables treatment (the MM-001 latent-factor model is the natural home once its funnel is fixed).

### (3) Is the effect carried by a capable minority? — no; if anything the opposite.

Gains are broad, and the **causal effect is largest in the least able children** (phase-0 window):

- **86 %** of treated children gained on letter sounds, **89 %** on word reading (11–14 % flat/declined); the **median** treated child gained +6.5 letter sounds / +3 words.
- Treatment effect (arm difference) by baseline block-design ability tertile:

| Ability tertile | L: treated − control | W: treated − control |
| --------------- | -------------------- | -------------------- |
| **Low**         | **+6.1**             | **+3.8**             |
| Mid             | +3.5                 | +2.1                 |
| High            | +1.3                 | +1.5                 |

The benefit **decreases** with ability for both outcomes — the opposite of a capable-minority story; letter-sound gain is also negatively correlated with baseline ability (−0.24) and own baseline (−0.47), i.e. room-to-grow / ceiling. Concentration is mild (top gain-quartile holds 46 % of L / 56 % of W total gain — normal right-skew). Caveats: n per tertile cell is 6–11 (noisy), and "low-ability gains more" on letter sounds is partly a ceiling artefact (high-ability children start nearer the 32-item cap). This gradient also feeds (a): a pure $GA$ confound would concentrate the effect in _high_-ability children, not the lowest.

## Caveats

- **Small sample** (52–54 children in the between-child fits; 33–34 in SES complete-case). Threshold-clearing point estimates are magnitude-inflated on average — the intervals are the honest summary.
- Covariate coefficients are DAG-pre-specified adjustments, **not** effect estimates. A skill absent from a model was excluded by the diagram, not found unimportant.
- `mm-001` structural coefficients are on hold (funnel geometry); its domain correlations are fine.
- The nine divergence-only flags (≤ 0.41 %) are within tolerance but worth a targeted `--target-accept 0.97` refit if those mechanism/dose surfaces become load-bearing for any claim.

## Artefacts

`output/statistical_models/models/<model_id>-rep-lite/` (per model) and `output/statistical_models/comparison/` (incl. `mediation_family.{csv,png}`). The original 89-model sweep was not rendered (`--render` omitted); the nine models added this day (MED-066/068/074/075/076/078/079/080, ITT-016) each have a thin `docs/models/<id>/index.qmd` template and were **rendered to HTML** in their output dirs. Traces (`trace.nc`) retained locally; not uploaded.
