# Difference-in-differences (waitlist-crossover) findings (2026-07-20)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

This is family note 04 in the suite. It reports every model in the **difference-in-differences (DiD)** family and should be read alongside the shared index and reading guide, `notes/202607200900-findings-00-index-and-reading-guide.md`, which explains the study, the outcome measures and their item maxima, and the house rules for reporting a posterior. All figures below are traced to each model's own output CSVs (`did_summary.csv`, `dose_slope_summary.csv`, `config.json`, `diagnostics_summary.json`) from the production `reporting` re-fit; all data are preliminary.

## What this family does and the question it answers

The trial is a **waitlist-crossover randomised design**: about 54 children with Down syndrome were randomly assigned either to start a reading/phonics intervention immediately or to start it after a wait, and were tested at four timepoints. The DiD family uses that crossover structure directly. Each model jointly fits the bounded post-scores at three waves (t1, t2, t3) for one skill, on a proportion-correct (logit) scale via a Beta-Binomial likelihood, and estimates a separate **immediate-minus-waitlist arm gap at each wave** plus a child random intercept (which partially pools stable between-child differences but does **not** stand in for latent general ability, and is not an exact fixed-effect control).

The question it answers that the others do not: it reads the intervention effect _through the within-person crossover trajectory_ — the immediate arm pulls ahead at t2, then the waitlist arm catches up once it too receives the intervention — rather than through a single post-baseline comparison (ITT) or a period-stacked gain model (gain-factors). It is a **longitudinal sensitivity analysis alongside the randomised ITT, not an independent experiment**: agreement between the DiD t2 contrast and its ITT sibling is a parameterisation cross-check on shared data, not a replication.

The family has four wave-contrasts, and telling them apart is the whole point:

- `arm_gap_t1` — the **pre-treatment** arm gap. Both arms are still untreated at t1, so this should sit near zero; it is a **balance check**, not an effect.
- `tau_t2` — the **t2 immediate-minus-waitlist contrast**. This is the first post-baseline wave, both arms are still as-randomised (the waitlist arm has not yet crossed over), so this is the **clean randomised (causal) contrast**. Positive favours the intervention.
- `arm_gap_t3` — the t3 arm gap, comparing two **already-treated** arms with different exposure histories. **Association**, not a randomised effect.
- `delta_crossover = tau_t2 − arm_gap_t3` — how much the immediate-arm advantage has **closed** by t3 once the waitlist arm has also been taught. Positive = catch-up. **Association**, not a second treatment effect.

Three dose companions (did-006, did-007, did-107) reframe the model around session attendance: they separate _being currently treated_ (`theta_treated`) from _treated-centred session intensity_ (`beta_dose` / the period-resolved `dose_*` slopes). Session count is **not randomised** (it reflects attendance and roll-out), so every dose coefficient is an **observational association**, never a proven lever.

## How to read these numbers (recap)

Full rules are in the index note; briefly: the point estimate is the posterior **median**; uncertainty is the **89% equal-tailed credible interval** (the house standard, with the inner **50%** interval as the "most-of-the-mass" range); direction is the **tail probability** P(contrast > 0), read directly and never as a p-value. The evidence ladder attaches fixed labels to that probability — **inconclusive** (< 0.75), **suggestive** (≥ 0.75), **moderate** (≥ 0.91), **strong** (≥ 0.97), **very strong** (≥ 0.99) — describing the strength of evidence for a _directional_ claim, never the size of the effect. Effects are modelled on a proportion-correct scale and translated to **items** (probability × the test maximum). For the two heavily-floored skills (P, N) the translated quantity is an **off-floor risk difference**: a difference in the probability of _being_ off the floor at that wave (reported here in percentage points, pp), not the probability of coming off it.

**Causal versus association for this family:** only `tau_t2` is causal (randomisation licenses it). `arm_gap_t1` is a balance quantity; `arm_gap_t3`, `delta_crossover`, and every dose coefficient are **adjusted associations**. Reading an association as a lever is the Table-2 fallacy. Residual confounding by latent general ability remains for all of them.

## Per-model results

### Binary arm-by-wave models (the randomised `tau_t2` is CAUSAL)

The table gives the headline `tau_t2` (t2 randomised contrast). Item columns are per each test's maximum; P and N are off-floor risk differences in pp. N is the number of children.

| ID      | Outcome (max items)                     | N   | τ_t2 probability — median [89% CI] | τ_t2 items — median [89% CI] | P(τ_t2 > 0) | Evidence     | Gate |
| ------- | --------------------------------------- | --- | ---------------------------------- | ---------------------------- | ----------- | ------------ | ---- |
| did-002 | L — letter-sound knowledge (32)         | 54  | +0.595 [+0.200, +0.980]            | +3.53 [+1.18, +5.81]         | 0.991       | very strong  | pass |
| did-003 | B — phoneme blending (10)               | 54  | +0.415 [+0.030, +0.798]            | +0.88 [+0.06, +1.69]         | 0.956       | moderate     | pass |
| did-004 | TE — taught expressive vocab (24)       | 54  | +0.323 [+0.009, +0.638]            | +1.51 [+0.04, +2.95]         | 0.949       | moderate     | pass |
| did-001 | W — word reading (79)                   | 53  | +0.340 [−0.047, +0.725]            | +2.22 [−0.31, +4.69]         | 0.920       | moderate     | pass |
| did-013 | W — word reading, catch-up variant (79) | 53  | +0.335 [−0.047, +0.717]            | +2.19 [−0.31, +4.64]         | 0.918       | moderate     | pass |
| did-008 | TR — taught receptive vocab (24)        | 54  | +0.231 [−0.055, +0.512]            | +1.22 [−0.29, +2.70]         | 0.902       | suggestive   | pass |
| did-010 | F — basic concepts (18)                 | 54  | +0.164 [−0.135, +0.468]            | +0.62 [−0.51, +1.77]         | 0.809       | suggestive   | pass |
| did-012 | N — nonword reading (off-floor)         | 53  | +0.298 log-odds [−0.310, +0.896]   | +5.8 pp [−6.0, +17.7]        | 0.788       | suggestive   | pass |
| did-011 | P — phonetic spelling (off-floor)       | 54  | +0.161 log-odds [−0.491, +0.815]   | +2.3 pp [−7.0, +11.6]        | 0.652       | inconclusive | pass |
| did-009 | E — standardised expressive vocab (170) | 54  | +0.032 [−0.153, +0.214]            | +0.84 [−3.97, +5.54]         | 0.608       | inconclusive | pass |
| did-005 | R — standardised receptive vocab (170)  | 54  | −0.002 [−0.175, +0.172]            | −0.06 [−5.11, +5.00]         | 0.492       | inconclusive | pass |

**Letter-sound knowledge, L (did-002).** The strongest DiD result. The randomised t2 contrast is τ_t2 = +0.595 on the logit scale, about **+3.5 of 32 letter sounds** (89% CrI +1.2 to +5.8; inner 50% +2.5 to +4.5), with P(τ_t2 > 0) = 0.991 — _very strong_ evidence of benefit. Baseline balance is good: arm_gap_t1 = −0.037 logit (−0.2 items, P > 0 = 0.433, inconclusive), i.e. no meaningful pre-treatment gap. Post-crossover, the immediate arm still leads at t3 (arm_gap_t3 = +0.242 logit ≈ +1.3 items, P > 0 = 0.833, _suggestive_ association) and the gap has closed only partially (delta_crossover = +0.353 logit ≈ +2.2 items, P > 0 = 0.922, _moderate_ association) — consistent with catch-up but not a second randomised effect.

**Phoneme blending, B (did-003).** τ_t2 = +0.415 logit ≈ **+0.88 of 10** (89% +0.06 to +1.69; 50% +0.54 to +1.22), P = 0.956, _moderate_ causal benefit. Baseline balanced (arm_gap_t1 = −0.073, P > 0 = 0.359). arm_gap_t3 = +0.171 (≈ +0.35 items, P > 0 = 0.761, suggestive) and delta_crossover = +0.246 (≈ +0.53 items, P > 0 = 0.821, suggestive) — the same partial-catch-up pattern, associational.

**Taught expressive vocabulary, TE (did-004).** τ_t2 = +0.323 logit ≈ **+1.5 of 24** taught expressive words (89% +0.04 to +2.95; 50% +0.89 to +2.11), P = 0.949, _moderate_ causal benefit. Baseline balanced (arm_gap_t1 = −0.058, P > 0 = 0.374). Uniquely in the family, at t3 the arms are level (arm_gap_t3 = −0.064, ≈ −0.33 items, P > 0 = 0.373, inconclusive) and the gap has fully closed — delta_crossover = +0.388 logit ≈ +1.8 items, P > 0 = 0.979 (_strong_ association for catch-up). This is exactly what a taught-word effect should do once both arms have been taught the same words.

**Word reading, W (did-001, and did-013 the catch-up-heterogeneity variant).** τ_t2 = +0.340 logit ≈ **+2.2 of 79 words** (89% −0.31 to +4.69; 50% +1.18 to +3.27), P = 0.920, _moderate_ causal benefit — the 89% interval just includes zero, so this is a clear-direction but not decisive signal. Baseline balance is adequate but slightly negative (arm_gap_t1 = −0.138 logit ≈ −0.66 items, P > 0 = 0.261, inconclusive — the immediate arm started marginally lower). arm_gap_t3 = +0.233 (≈ +2.0 items, P > 0 = 0.838, _suggestive_ association); delta_crossover = +0.107 (≈ +0.25 items, P > 0 = 0.705, inconclusive) — little detectable closure by t3, i.e. the immediate arm largely holds its lead. did-013 is a variant that adds an exploratory child-specific catch-up term: its fixed τ_t2 is essentially identical (+0.335 logit ≈ +2.19 items, P = 0.918, moderate), and its sample-average heterogeneous catch-up contrast is +0.103 logit (89% −0.218 to +0.419, over 25 children) — inconclusive, and explicitly _not_ an individual-causal-effect average.

**Taught receptive vocabulary, TR (did-008).** τ_t2 = +0.231 logit ≈ **+1.2 of 24** (89% −0.29 to +2.70; 50% +0.59 to +1.85), P = 0.902, _suggestive_ causal benefit. Baseline balanced (arm_gap_t1 = −0.050, P > 0 = 0.374). arm_gap_t3 = +0.079 (≈ +0.40 items, P > 0 = 0.671, inconclusive) and delta_crossover = +0.152 (≈ +0.83 items, P > 0 = 0.800, suggestive) — the arms have largely converged by t3.

**Basic concepts, F (did-010).** τ_t2 = +0.164 logit ≈ **+0.62 of 18** (89% −0.51 to +1.77; 50% +0.14 to +1.11), P = 0.809, _suggestive_ causal benefit. Note a baseline imbalance here: arm_gap_t1 = −0.175 logit ≈ −0.67 items, favoured-direction P = 0.176 negative (_suggestive_) — the immediate arm started somewhat _lower_ on basic concepts, so the t2 gain is estimated over a slightly lower starting point. arm_gap_t3 = +0.033 (inconclusive) and delta_crossover = +0.131 (P > 0 = 0.735, inconclusive).

**Nonword reading, N (did-012) and phonetic spelling, P (did-011) — heavily floored, off-floor scale.** These two model the probability of being _off the floor_. For N, the t2 off-floor risk difference is +5.8 pp (89% −6.0 to +17.7 pp; τ_t2 log-odds +0.298, P > 0 = 0.788, _suggestive_ favourable). For P it is +2.3 pp (89% −7.0 to +11.6 pp; log-odds +0.161, P > 0 = 0.652, _inconclusive_). Both show a mild negative baseline off-floor imbalance (arm_gap_t1 favoured-direction _suggestive_ negative: P −3.3 pp, N −3.4 pp — fewer immediate-arm children were off the floor at t1). Post-crossover contrasts are inconclusive for both. Treat these as exploratory: floored outcomes are low-information and the direction, not the magnitude, is the usable read.

**Broad standardised vocabulary, R (did-005) and E (did-009) — flat.** Neither shows a randomised effect. R: τ_t2 = −0.002 logit ≈ −0.06 of 170 (89% −5.11 to +5.00), P > 0 = 0.492 — _inconclusive_, centred on zero. E: τ_t2 = +0.032 logit ≈ +0.84 of 170 (89% −3.97 to +5.54), P > 0 = 0.608 — _inconclusive_. All their wave contrasts are inconclusive and near zero. This is a flat, uninformative result on the broad vocabulary tests, not evidence of "no effect".

### Session-dose companions (all coefficients are ASSOCIATIONS — dose is not randomised)

| ID      | Outcome                                  | Children / obs | Session-intensity slope — median [89% CI]                | P(> 0) | Evidence    | Gate                        |
| ------- | ---------------------------------------- | -------------- | -------------------------------------------------------- | ------ | ----------- | --------------------------- |
| did-006 | W — word reading, pooled dose            | 53 / 105       | `beta_dose` +0.228 [+0.098, +0.363] per 1 SD sessions    | 0.997  | very strong | pass                        |
| did-007 | L — letter-sound, period-resolved dose   | 54 / 107       | `dose_overall` +0.127 [−0.211, +0.491] per 1 SD sessions | 0.774  | suggestive  | **fail (divergences only)** |
| did-107 | L — letter-sound, pooled-dose comparator | 54 / 107       | `beta_dose` +0.118 [−0.035, +0.268] per 1 SD sessions    | 0.893  | suggestive  | pass                        |

**Word reading dose, W (did-006).** After adjusting for arm, period, current-treatment presence and the shared t1 outcome, higher treated-centred session intensity is associated with a higher period-end word-reading score: `beta_dose` = +0.228 logit per 1 SD of sessions (89% +0.098 to +0.363; 50% +0.173 to +0.285), P > 0 = 0.997, _very strong_ — but this is an **observational** intensive-margin association, not a randomised or proven dose effect (children who attend more may differ). The treatment-presence term `theta_treated` here is +0.091 (89% −0.203 to +0.380, P > 0 = 0.694, inconclusive), and the transition-period term `beta_period` = +0.520 (P > 0 = 1.000) simply reflects that everyone improves across periods; the post-crossover arm association `beta_group` = +0.357 (P > 0 = 0.976, strong association). So for word reading, the residual session-intensity gradient carries a clear signal.

**Letter-sound dose, L (did-007 period-resolved; did-107 pooled comparator).** Here the picture inverts relative to W: the signal sits in _being treated_ rather than in the intensive margin. `theta_treated` = +0.422 (89% +0.086 to +0.755, P > 0 = 0.978, _strong_ association) in both fits. The session-intensity slope is weaker: did-007 gives `dose_overall` = +0.127 (89% −0.211 to +0.491, P > 0 = 0.774, _suggestive_), resolved as period-1 +0.149 (P > 0 = 0.838) and period-2 +0.112 (P > 0 = 0.876), with a between-period SD of +0.179 (little evidence the slope differs by period); its items-scale marginal is +0.71 of 32 (89% −0.25 to +1.64, P > 0 = 0.883). did-107 is the **pooled-slope comparator** to did-007 (a single dose slope, no period variation), and it recovers essentially the same coefficients — `beta_dose` = +0.118 (89% −0.035 to +0.268, P > 0 = 0.893, _suggestive_), `theta_treated` = +0.424, `beta_group` = +0.186 — confirming the period-resolved fit adds no material structure. (The formal nested PSIS-LOO comparison table was not emitted at this fit; the substantive point is the close agreement of the shared coefficients.) All of these are observational associations.

## What the family concludes

The DiD re-analysis **triangulates cleanly with the randomised ITT** on the causal `tau_t2` contrast, outcome by outcome (ITT items in brackets):

- L: DiD +3.5 items, very strong ≈ **ITT +3.5, very strong** — near-identical.
- W: DiD +2.2 items, moderate ≈ **ITT +2.4, strong** — same magnitude, DiD interval a touch wider so one rung lower on the ladder.
- B: DiD +0.88 of 10, moderate ≈ **ITT +0.99, strong** — consistent.
- TE: DiD +1.5 of 24, moderate ≈ **ITT +1.6, strong** — consistent.
- TR: DiD +1.2 of 24, suggestive ≈ **ITT +1.4, moderate** — consistent.
- F: DiD +0.62 of 18, suggestive ≈ **ITT +0.87, suggestive** — consistent.
- N: DiD +5.8 pp off-floor, suggestive ≈ **ITT +10 pp, suggestive** — same favourable direction.
- P: DiD +2.3 pp off-floor, inconclusive ≈ **ITT +4.1 pp, inconclusive** — same (flat, floored).
- R and E: DiD flat/inconclusive ≈ **ITT flat/inconclusive** — agreement on the null-ish result.

The consistent theme is that the DiD `tau_t2` sits within a whisker of the ITT τ but is typically one evidence-rung softer, because a three-wave joint fit spends more of its precision than the single ITT comparison — which is exactly the expected behaviour of a sensitivity analysis. The **crossover picture** is coherent: the immediate arm pulls ahead at t2 on the code and taught skills, and by t3 the waitlist arm has partly (L, B) or fully (TE, TR) caught up once it too is taught — visible in the positive `delta_crossover` associations and the shrinking t3 arm gaps. The dose companions add an observational layer: for word reading the residual session-intensity gradient is strongly positive, whereas for letter-sounds the association is carried by treatment presence rather than by intensity — both associational and both consistent with a code-focused benefit.

## Caveats and convergence

- **Convergence gate.** Thirteen of the fourteen models **pass** cleanly (R-hat ≤ 1.003, minimum ESS in the thousands, zero divergences). The one exception is **did-007**, which **fails on divergences only** (14 divergences out of 36,000 draws ≈ 0.04%, well under the METHODS ≤ 1% guidance; R-hat, ESS and BFMI all pass). It is **usable with that caveat noted**, and its pooled comparator did-107 (which passes cleanly) reproduces the same coefficients, so the letter-sound dose conclusion does not hinge on the flagged fit.
- **Causal scope.** Only `tau_t2` is causal. `arm_gap_t3`, `delta_crossover`, all dose slopes and `theta_treated` are adjusted associations — do not read them as levers (Table-2 fallacy). The models deliberately do **not** condition on each period's start outcome (the t2 score is already treatment-affected for the immediate arm), and the child random intercept partially pools stable heterogeneity but is not an exact fixed-effect control and does not stand in for latent general ability.
- **Small sample and baseline imbalance.** With ~54 children, estimates are uncertain and any point estimate that just clears a threshold is on average magnitude-inflated (winner's curse) — lead with the intervals. Randomisation did not produce perfect balance in this small sample: mild negative `arm_gap_t1` gaps appear for F (−0.67 items, suggestive), and for the floored P and N (immediate arm slightly lower off the floor at t1). These are the reason the t2 contrast, not a raw t3 comparison, is the trustworthy quantity.
- **Floor effects.** P and N are heavily floored; their off-floor risk differences are low-information and exploratory — read direction, not magnitude.
- **Broad vocabulary.** R and E are flat and _inconclusive_; this quantifies "no detectable signal", not a demonstrated absence of effect.
