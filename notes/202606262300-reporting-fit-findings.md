# Reporting-config fit run: how the models fit and what they find (2026-06-26)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

The full statistical-model suite (all 66 registered models) was fit in **`reporting`
config** (6000 draws × 6 chains, `target_accept` 0.95), rendered, and uploaded to the
DSE research blob. This note records **how the models fit** (convergence) and **what
they find** (results), superseding the preliminary dev-config review
(`notes/202606261800-dev-fit-findings-review.md`) — these are the converged numbers.

## How the models fit

- **All 66 models converged cleanly.** Every model passes the convergence gate:
  **R-hat ≤ 1.01, ESS_bulk/tail ≥ 400, 0 divergent transitions, BFMI ≥ 0.3.** The
  reporting chains resolved the low-ESS REVIEW flags that the dev fits showed on the
  factor/level/joint families — at 6000 × 6 those models now mix well (e.g. the ITT
  singles reach ESS in the tens of thousands).
- **0 fit failures** across the suite (per-model `rc=0` for all 66).
- **64 of 66 reports rendered.** The two without a report — `lrp72base` and
  `lrp73base` — are the no-interaction _baseline companions_ of the LRP72/LRP73
  mechanism models, used only for nested PSIS-LOO comparison; they have no standalone
  report template by design (this is expected, not a failure). Their fits and CSVs
  uploaded normally.
- **Families:** ITT singles 23, joint 3, waitlist-crossover/DiD 6, gain-factors 16,
  level-factors 8, mechanism 8, mediation 2.
- Convergence detail per model: `output/reporting_convergence_summary.csv`.

### Reading conventions (recap)

Positive = intervention helps. Effects are reported as the **median** with the **95%
equal-tailed credible interval (CrI)** and **pd = P(effect > 0)**; magnitude is
judged against a minimally-important difference δ (ROPE) on the items scale, with
`P(benefit ≥ δ)` and the share of mass inside the ROPE. **Only randomised terms are
causal** — the ITT `τ`, the gain family's `beta_trt`, the level family's
`b_grp_time[1]` (t2), and the DiD `delta`. Own baseline, age, ability, cross-skills,
mechanism slopes and mediator paths are **adjusted associations** ("associated with",
never "drives"). At n ≈ 53 children even a converged point estimate can be
magnitude-inflated (Type-M), so lead with the interval.

---

## What they find

### Headline

The intervention **credibly improves print/decoding skills — letter-sound knowledge most of all, then word reading, then (smaller) phoneme blending — and the directly-taught vocabulary (especially expressive)**. It shows **no credible effect on broad standardised vocabulary (ROWPVT, EOWPVT), grammar (TROG), or the heavily-floored phonetic-spelling/nonword outcomes**. The effect concentrates where instruction is most direct and fades with distance from it (a clean proximal→distal gradient). The reading-skill effects are **robust to SES and general-ability adjustment** and **replicate across four independent identification strategies**.

### 1. Single-outcome ITT — the core randomised contrasts (τ, the only causal term)

Items-scale treatment effect (median, 95% CrI), pd = P(τ>0), and the practical-benefit verdict. ✶ = 95% CrI excludes 0.

| outcome                                       | items median (95% CrI)   | pd        | practical benefit              |
| --------------------------------------------- | ------------------------ | --------- | ------------------------------ |
| **L** letter sounds (lrpitt07)                | **+3.56 (1.25, 5.79)** ✶ | **0.998** | P(≥2)=0.91 — likely meaningful |
| **W** word reading (lrpitt10)                 | **+2.38 (0.27, 4.50)** ✶ | **0.986** | P(≥1)=0.90 — likely meaningful |
| **TE** taught expressive vocab (lrpitt02)     | **+1.54 (0.14, 2.92)** ✶ | **0.985** | P(≥1)=0.78 — suggestive        |
| B phoneme blending (lrpitt08)                 | +1.00 (0.03, 1.93) ✶     | 0.978     | P(≥1)=0.50 — size uncertain    |
| TR taught receptive vocab (lrpitt01)          | +1.35 (−0.12, 2.78)      | 0.964     | P(≥1)=0.68 — inconclusive      |
| UR not-taught receptive (lrpitt03)            | +0.70 (−0.15, 1.55)      | 0.948     | 75% in ROPE — negligible       |
| UE not-taught expressive (lrpitt04)           | +0.36 (−0.54, 1.25)      | 0.787     | 92% in ROPE — negligible       |
| R ROWPVT std. receptive (lrpitt05)            | +0.23 (−4.89, 5.34)      | 0.537     | null                           |
| E EOWPVT std. expressive (lrpitt06)           | +0.24 (−3.89, 4.46)      | 0.544     | null                           |
| P phonetic spelling — off-floor RD (lrpitt09) | −0.01 (−0.17, 0.16)      | 0.459     | precise null (76% in ROPE)     |
| N nonword — off-floor RD (lrpitt11)           | +0.056 (−0.11, 0.22)     | 0.746     | inconclusive (67% in ROPE)     |

- **Three outcomes have intervals excluding zero: L, W, TE.** A clear transfer gradient: directly-taught/decoding skills move; not-taught vocabulary is small-to-negligible; standardised vocabulary is flat-null; the floored spelling/nonword outcomes carry little power (P/N reported as off-floor _risk differences_; P is a precise null with identical 36% vs 36% off-floor rates, N leans positive but inconclusive at 43% vs 28%).
- **Floored-rule machinery worked as designed** — both P and N passed the proportion-at-zero posterior-predictive check (P p=0.42, N p=0.49). Their δ are provisional (0.10 risk difference, pending the education lead).

**Robustness (lrpitt13/13b/14/14b SES; lrpitt17–24 general ability).** The reading effects survive both adjustments. Letter sounds stays "very strong" everywhere (SES-adjusted +3.47 ✶; matched-unadjusted +4.00 ✶; ability-adjusted +3.58 ✶). Word reading stays positive/"strong" (+2.2 to +2.5; CrI excludes 0 on the full-sample ability-adjusted fit and the matched SES comparator, grazes 0 on the small n=33 SES-adjusted subset). The **matched SES comparators (14/14b) show SES adjustment barely moves the estimates — SES is not confounding the reading effects.** Standardised vocabulary stays null under ability adjustment (R/E pd ≈ 0.53). (SES variants run on small complete-case subsets, n=33–34.)

### 2. Joint model + generalisation contrasts

- **lrpitt12 (joint, all 10 outcomes)** reproduces the singles (consistency check holds) and shows the same gradient: credibly-positive **L (+0.576 logit, pd 0.998), W (+0.354, pd 0.986), TE (+0.324, pd 0.984)**; B/TR/UR positive but CrIs graze zero; **R near-zero, E essentially null (pd 0.49)**. Contrast matrix puts L as the largest effect (P(L>E)=0.997, P(L>R)=0.994).
- **Generalisation (lrpitt15 expressive, lrpitt15b receptive).** Expressive: taught (TE) credibly positive, not-taught (UE) uncertain, taught-minus-not-taught **+0.18 but CrI spans 0 (pd 0.79)** — limited far-transfer is _suggested, not confirmed_. Receptive: taught (TR) ≈ not-taught (UR), difference ≈ 0 — receptive gains generalise to untaught items as much as taught ones.

### 3. Waitlist-crossover / DiD — within-person replication (delta, causal)

Each child is their own control; `beta_period` (maturation) is not causal.

- **W word reading: +2.73 items (0.41, 4.95), pd 0.988** ✶; **dose-response confirms it** (lrpdid06 session-dose slope +0.165, pd 0.997 ✶).
- **L letter sounds: +3.39 items (1.08, 5.76), pd 0.998** ✶ (strongest).
- **B blending: +0.96 items (0.04, 1.85), pd 0.980** ✶.
- TE taught expressive: +1.48 items, pd 0.971 — borderline (CrI just touches 0).
- R receptive vocab: null (delta ≈ 0, pd 0.49).
- **Read:** the within-person design independently reproduces the ITT reading-skill benefits (with a dose-response for word reading) and the vocabulary null.

### 4. Gain-factor models (LRPGF01–08) — causal term `beta_trt`; everything else adjusted association

`beta_trt` is the randomised on-intervention contrast (pooled over on-intervention periods, child random intercept); every `gamma_*` is an adjusted association. Items column is the treatment average marginal effect (for P it is the off-floor **risk difference**, RD). ✶ = logit 95% CrI excludes 0. Ordered by strength.

| model       | outcome                         | `beta_trt` logit (95% CrI)  | pd        | items (95% CrI)           | verdict                   |
| ----------- | ------------------------------- | --------------------------- | --------- | ------------------------- | ------------------------- |
| **lrpgf01** | **W** word reading              | **+0.432 [0.107, 0.768]** ✶ | **0.996** | **+3.48 [0.91, 5.83]**    | credible; P(≥1)=0.97      |
| **lrpgf04** | **L** letter sounds             | **+0.529 [0.133, 0.917]** ✶ | **0.995** | **+3.06 [0.73, 5.44]**    | credible; P(≥2)=0.81      |
| lrpgf06     | B blending                      | +0.324 [−0.120, 0.767]      | 0.923     | +0.68 [−0.24, 1.61]       | suggestive; 75% in ROPE   |
| lrpgf07     | F CELF concepts                 | +0.235 [−0.121, 0.593]      | 0.902     | +0.86 [−0.43, 2.21]       | suggestive; CrI spans 0   |
| lrpgf08     | T TROG grammar                  | +0.075 [−0.170, 0.318]      | 0.728     | +0.57 [−1.28, 2.37]       | null                      |
| lrpgf03     | E expressive vocab              | +0.017 [−0.136, 0.171]      | 0.583     | +0.47 [−3.92, 4.67]       | null; 63% in ROPE         |
| lrpgf02     | R receptive vocab               | −0.032 [−0.198, 0.134]      | 0.351     | −1.01 [−6.42, 4.12]       | null                      |
| lrpgf05     | P phonetic spelling (off-floor) | +0.273 [−0.546, 1.105]      | 0.742     | RD +0.025 [−0.056, 0.098] | inconclusive; 98% in ROPE |

The **8 `b` companions (lrpgf01b–08b) are treated-only and carry NO causal term** (the on-intervention indicator is constant ⇒ no contrast). They document only within-treated _adjusted associations_ (own baseline dominant; receptive vocab a strong correlate of E/F/T) and say nothing about whether the intervention worked — by design.

### 5. Level-factor models (LRPLF01–08) — causal term is the t2 contrast `b_grp_time[1]`; later timepoints are post-crossover

The levels formulation carries no own baseline, so it is the least precise route to the causal contrast; only the t2 element is a clean randomised effect. Items column from the t2 ROPE summary (— where the floored/grammar models emit none; logit only). ✶ = logit 95% CrI excludes 0. Ordered by strength.

| model       | outcome                         | t2 `b_grp_time[1]` logit (95% CrI) | pd        | items (95% CrI)        | verdict                    |
| ----------- | ------------------------------- | ---------------------------------- | --------- | ---------------------- | -------------------------- |
| **lrplf04** | **L** letter sounds             | **+0.489 [0.009, 0.963]** ✶        | **0.977** | **+2.88 [0.05, 5.68]** | credible — family standout |
| lrplf01     | W word reading                  | +0.221 [−0.250, 0.695]             | 0.817     | +1.70 [−1.54, 4.86]    | suggestive; 29% in ROPE    |
| lrplf06     | B blending                      | +0.302 [−0.188, 0.786]             | 0.888     | +0.62 [−0.39, 1.61]    | suggestive; 77% in ROPE    |
| lrplf07     | F CELF concepts                 | +0.129 [−0.300, 0.560]             | 0.723     | —                      | null                       |
| lrplf08     | T TROG grammar                  | +0.019 [−0.267, 0.302]             | 0.552     | —                      | null                       |
| lrplf03     | E expressive vocab              | −0.006 [−0.218, 0.206]             | 0.475     | +0.09 [−5.42, 5.59]    | null                       |
| lrplf02     | R receptive vocab               | −0.035 [−0.236, 0.168]             | 0.366     | −1.09 [−7.01, 4.84]    | null                       |
| lrplf05     | P phonetic spelling (off-floor) | −0.005 [−0.823, 0.817]             | 0.494     | —                      | null                       |

Only letter sounds shows a credible t2 randomised effect; word reading and blending lean positive but inconclusive; vocabulary, grammar and floored spelling are null at t2. (Some t3/t4 group terms are positive but are post-crossover associations, not treatment effects.)

### 6. Mechanism + mediation — adjusted associations / exploratory decomposition

- **The one causal claim is the total effect:** in the randomised window the intervention raises word reading by **+2.6 to +2.9 words** (lrp59 +2.86 [0.52, 5.24], pd 0.99; lrp62 +2.65 [0.40, 4.90], pd 0.99) — consistent with ITT/DiD.
- **Mechanism (lrp56–58, 71–73): all adjusted associations** (`beta_G` is _not_ a treatment contrast — pooled phases — and sits at ≈0). The vocabulary/letter-sound → word-reading dose-response curves trend positive but their 95% bands span zero (letter-sound fits marginally best by LOO). The **clearest mechanism signal is the code-based route into nonword decoding (lrp72): letter sounds (+1.16) and blending (+0.45) each credibly associated with decoding, with a credibly _negative_ (sub-additive) interaction** — the two skills substitute rather than compound.
- **Mediation (lrp59, lrp62):** the decomposition points to most of the word-reading gain running through letter-sound knowledge (NIE credible), but the direct/indirect split rests on a **non-causal, unverified mediator path and is unstable** (`proportion_mediated` CrI exceeds 1) — **exploratory, not a causal mediation claim.**

---

## Cross-cutting synthesis

### Convergent validity — letter sounds and word reading replicate across designs

The two strongest effects are credibly positive under **four independent identification strategies**, which is the most compelling evidence in the suite:

| design (estimand)                 | letter sounds (LS) | word reading (WR)                   |
| --------------------------------- | ------------------ | ----------------------------------- |
| ITT, between-arm (τ)              | +3.56, pd 0.998 ✶  | +2.38, pd 0.986 ✶                   |
| DiD, within-person (delta)        | +3.39, pd 0.998 ✶  | +2.73, pd 0.988 ✶ (dose-response ✶) |
| Gain-factor (beta_trt)            | +3.06, pd 0.995 ✶  | +3.48, pd 0.996 ✶                   |
| Mediation total / Level-factor t2 | (LF t2 +2.88 ✶)    | +2.86 total, pd 0.99 ✶              |

All on the items scale. Between-arm randomisation, within-person crossover, the gain-factor ANCOVA and the mediation total effect all land on **L ≈ +3 items and W ≈ +2.4–3.5 words** — different designs, same answer.

### The transfer gradient

**Decoding/print skills (L, W, B) → directly-taught vocabulary (TE, then TR) → not-taught vocabulary (UR, UE) → broad standardised vocabulary & grammar (R, E, T: null).** Effects are largest where the intervention teaches most directly and shrink with distance; far-transfer to norm-referenced vocabulary is absent, and even near-transfer to untaught words is at best small and not practically distinguishable from zero. (Per project memory: the null standardised-vocabulary result is a generalisation/measurement story at n≈54, since taught vocabulary _does_ move — not "the intervention doesn't target vocabulary".)

### What is solid vs promising vs fragile

- **Solid (credible, replicated, robust):** letter-sound knowledge and word reading improve — intervals exclude zero across designs and survive SES + ability adjustment; word reading shows a dose-response. The standardised-vocabulary nulls (R, E) are consistent everywhere. The phonetic-spelling off-floor null (identical arm rates) is clean.
- **Promising but not decisive:** taught expressive vocabulary (TE, credible in ITT/joint, borderline in DiD); phoneme blending (credibly-signed, magnitude uncertain); the letter-sound → decoding mechanism association.
- **Fragile / not credible / exploratory:** taught receptive vocabulary magnitude; generalisation to untaught words; nonword-reading off-floor hint; CELF/TROG; the NDE/NIE mediation split (exploratory — non-causal path, unstable proportion-mediated); all `gamma_*`/mechanism/mediator quantities (adjusted associations, never "drives").

### Caveats

- Small samples (n ≈ 53–54 children; the SES subsets n ≈ 33–34) → wide intervals and Type-M (winner's-curse) inflation of large-looking points; lead with the interval.
- The floored P/N estimands are low-power by construction and use **provisional δ** (0.10 risk difference) — settle with the education lead before quoting their ROPE verdicts; F/T have no agreed δ.
- Only randomised terms are causal. Mechanism `beta_G` is pooled (not a contrast); mediation decompositions are exploratory.

### Recommended next steps

1. Lead any write-up with the **convergent-validity result** (L and W across four designs) and the **transfer gradient** — these are the robust, defensible findings.
2. Settle the **floored-outcome δ** (P/N) and decide on δ for F/T with the education lead.
3. Treat the **mediation/mechanism** results as hypothesis-generating (letter-sound route into reading), not causal mediation.
4. Per-model reports (HTML, with convergence banner, ROPE, forests) are on the DSE research blob; convergence detail in `output/reporting_convergence_summary.csv`.
