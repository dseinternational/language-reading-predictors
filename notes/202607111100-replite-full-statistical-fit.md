# Full statistical-model fit in `rep-lite` — run summary and findings (2026-07-11)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

## Run metadata

- **Command:** `python scripts/fit_statistical_model.py all --config rep-lite`
- **Sampling preset (`rep-lite`):** 4 chains × 4000 draws × 4000 tune, `target_accept = 0.95` — reporting-grade rigour (same acceptance target as `reporting`) but lighter draws, since ESS, not raw draws, is the binding convergence metric.
- **Scope:** all 89 registered models across 16 families (96 output directories, counting per-model sensitivity/companion sub-fits).
- **Wall time:** 35 min 49 s (16-core workstation); exit code 0, 0 fits failed. Reports were **not** rendered (`--render` omitted); findings below read the result CSVs directly.
- **Cross-model comparison:** `python scripts/compare_statistical_models.py --config rep-lite` ran cleanly (`itt_vs_joint_tau.csv`, `tau_forest.png`, `mechanism_forest.{png,csv}`, nested PSIS-LOO tables).
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

## Caveats

- **Small sample** (52–54 children in the between-child fits; 33–34 in SES complete-case). Threshold-clearing point estimates are magnitude-inflated on average — the intervals are the honest summary.
- Covariate coefficients are DAG-pre-specified adjustments, **not** effect estimates. A skill absent from a model was excluded by the diagram, not found unimportant.
- `mm-001` structural coefficients are on hold (funnel geometry); its domain correlations are fine.
- The nine divergence-only flags (≤ 0.41 %) are within tolerance but worth a targeted `--target-accept 0.97` refit if those mechanism/dose surfaces become load-bearing for any claim.

## Artefacts

`output/statistical_models/models/<model_id>-rep-lite/` (per model) and `output/statistical_models/comparison/`. Traces (`trace.nc`) retained locally; not uploaded.
