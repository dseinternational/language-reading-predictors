# Aligned-40-week per-protocol family (LRPAL) — design decisions

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

Date: 2026-06-26

## Scope

A statistical-model family (`kind="aligned"`, modules `lrpal01–08` + the `lrpal01d`
dose variant) that compares the two RLI arms on a **like-for-like intervention
dose** by aligning each child to its own intervention onset, rather than to
calendar/assessment wave. It answers "after ~40 weeks of the program _from
onset_, where does each arm sit, and what is that aligned outcome associated
with?" — a per-protocol companion to the randomised LRPITT suite and its
within-person replication (LRPDID). This note records the decisions a future
reader might question.

## The onset alignment (confirmed from the data)

Cumulative sessions (`attend_cumul`) by arm × wave fix the onset and window:

| Wave | Immediate (group 1) | Wait-list (group 2)   |
| ---- | ------------------- | --------------------- |
| t1   | 0                   | 0                     |
| t2   | ~73                 | **0 (still waiting)** |
| t3   | ~137                | ~70                   |
| t4   | ~190                | ~127                  |

So the immediate arm onsets at **t1** and the wait-list arm at **t2** (after its
wait). Two periods of intervention from onset (~the 40-week program, ~130 sessions)
lands at **t3** for the immediate arm and **t4** for the wait-list arm. The aligned
single gain is therefore:

- **Immediate:** pre = t1 (onset), post = t3.
- **Wait-list:** pre = t2 (onset), post = t4.

One row per child (54: 28 immediate, 26 wait-list; W gives 52 after dropping 2
incomplete). Data: Burgoyne et al. 2012, the RLI RCT
(doi:10.1111/j.1469-7610.2012.02557.x).

## Estimand and identification

### Decision 1 — per-protocol, so **nothing is a clean treatment effect**

Aligning by onset buys a like-for-like dose comparison but **spends the
randomisation**: the immediate-vs-wait-list contrast at the aligned endpoints
(`beta_cohort`) is no longer the ITT effect. The two arms reach their aligned
window at different calendar times and **different ages** (see Decision 2), so
`beta_cohort` is a confounded cohort/timing association. Accordingly the pipeline
flags **no** term causal (`causal_terms=()`): every coefficient — cohort, own
baseline, age-at-onset, ability — is reported as an _association_. The randomised
estimate of record stays in LRPITT/LRPDID.

### Decision 2 — age-at-onset is the headline confound

Because the wait-list arm onsets a wave later, it reaches its aligned window **~3–4
months older** (immediate onset ≈ 83.8 mo at t1; wait-list onset ≈ 87.4 mo at t2).
Age enters as **age-at-onset** (the age at each arm's own pre-wave), not a fixed
wave age. In the W exemplar `gamma_A` is negative and credible, so the cohort
contrast must be read net of this age gap — it is the main reason `beta_cohort`
(+0.19 logit, ~+1.9 words) is _weaker_ than the randomised ITT τ (~+0.44).

### Decision 3 — one row per child, **no random intercept**

Each child contributes a single aligned gain, so there are no repeated measures to
pool: the model is a cross-sectional Beta-Binomial ANCOVA with no child random
intercept (unlike the gain/level factor families, which stack periods/timepoints).

### Decision 4 — ability is merged from t1 for **both** arms

Cognitive ability (block design) is a t1-only baseline. For the wait-list arm the
onset row is t2, where `blocks` is not re-measured — so ability is taken from t1
for every child, never from the wait-list arm's t2 onset row. (Age-at-onset and the
own baseline _do_ come from the onset row; ability does not.)

### Decision 5 — dose is a collider → sensitivity variant only (`lrpal01d`)

Cumulative sessions are a **collider** on the DAG (a descendant of both group — the
immediate arm accrues sessions earlier — and ability — more able / available
children attend more), so conditioning on dose can open a back-door. Dose therefore
enters **only** the `lrpal01d` sensitivity variant, never the primary adjustment
set. As expected (the Phase-0b "dose ≈ null" check), `gamma_dose` is weak and
inconclusive once onset baseline, age-at-onset and ability are in the model
(W: +0.04, P ≈ 0.64, CrI −0.15..0.22) — the apparent dose signal is largely the
randomised contrast relabelled.

### Decision 6 — phonetic spelling (P) takes the floor rule

P is heavily floored, so `lrpal05` uses `likelihood="bernoulli_offfloor"` (a
Bernoulli on aligned post > 0, no `kappa`); its cohort marginal is an off-floor
risk difference. Same rule as the ITT suite and the gain/level factor families.

## What this is — and is not

- **Not randomised:** every LRPAL coefficient is an association; `beta_cohort` is a
  per-protocol cohort contrast, not the ITT effect.
- **Complements, does not replace:** triangulates the randomised LRPITT (between-arm)
  and LRPDID (within-person) effects with a per-protocol, dose-aligned view.
- **Dose is never a primary adjustment** — collider; sensitivity variant only.

## Reporting (ROPE conventions, and a deliberate deferral)

The ROPE-anchored evidence reporting adopted for the suite
(`notes/202606261304-evidence-strength-and-rope-reporting.md`) prefers the
**median**, **leads with the interval** rather than the point, separates
**direction** (`P(coef > 0)`) from **magnitude** (`P(|effect| ≥ δ)` against a
minimally-important difference δ / region of practical equivalence), and flags the
**Type-M / winner's-curse** inflation of point estimates at small samples. The LRPAL
reports adopt the **prose** side of this in full: direction is labelled as direction,
the interval leads, and the Type-M caveat is stated — per-arm _n_ is only ~26–28, so
the warning bites harder here than in the pooled suite.

The ROPE/δ **magnitude card** itself (`reporting.rope_summary` + `rope_summary.png`,
emitted by `fit_itt`) is **deliberately not wired into `fit_aligned`**, for three
reasons:

1. **It would mis-frame a confounded association as a treatment benefit.**
   `rope_summary` reports `P(benefit ≥ δ)` for the randomised ITT effect; LRPAL's
   `beta_cohort` is a per-protocol cohort association (Decision 1), not a treatment
   effect, so a "probability of a _meaningful benefit_" card on it would contradict
   the model's own causal stance.
2. **The plumbing is not there yet.** `rope_summary` is hard-wired to the ITT
   `tau`/`tau_i`/`eta` parameterisation; the aligned term is `beta_cohort`. The
   term-parameterised average-marginal-effect core (the `_itt_ame_draws` ↔
   `treatment_marginal_effect` fold flagged for merge in the ROPE note) is the
   prerequisite, and that is itself a post-merge follow-up.
3. **Half of LRPAL's outcomes have no agreed δ.** `measures.ROPE_DELTA` covers
   W/R/E/L/B but **deliberately omits F and T** (a lookup raises), and P (`lrpal05`)
   has only a placeholder `ROPE_DELTA_PROB = 0.10` (an off-floor risk-difference δ) —
   both pending the education lead.

Revisit when the ITT ROPE block is rolled out beyond the `lrpitt07` exemplar. If a
magnitude read is added to LRPAL then, it must be framed as the size of an
**association**, against an association-appropriate δ — never as a treatment benefit.
Separately, the tabulated point in `factor_summary` is still the posterior **mean**
(the shared helper has not been converted to median-first); that conversion is a
suite-wide change, out of scope here.

## Validation (dev config)

- **`lrpal01` (W):** `beta_cohort` = +0.19 (items ≈ +1.9 words, CrI −1.2..+5.0,
  P ≈ 0.87) — weaker than the ITT τ (~+0.44) because `gamma_A` = −0.17 (P ≈ 0.017)
  absorbs the age-at-onset gap. `gamma_own` = +1.21 (P ≈ 1).
- **`lrpal05` (P, off-floor):** all-association, `gamma_ability` = +0.45 (P ≈ 0.97)
  — ability tracks coming off the spelling floor.
- **`lrpal01d` (W dose):** `gamma_dose` ≈ +0.04 (P ≈ 0.64) — weak/inconclusive, as
  the collider reasoning predicts.

## Implementation

- **Preprocessing** (`preprocessing.py`): `load_and_prepare_aligned` builds the
  onset-aligned one-row-per-child frame (windows `{1:(1,3), 2:(2,4)}`; ability
  merged from t1; optional cumulative-session dose). `phase_mode="aligned"`,
  `n_phases = 1`.
- **Factories** (`factories.py`): `build_aligned_model` — Beta-Binomial ANCOVA on
  own onset baseline + age-at-onset + ability + cohort (+ optional dose), off-floor
  branch, no child random intercept.
- **Pipeline** (`pipeline.py`): `fit_aligned` + `_al_coef_names` / `_al_diag_vars`
  (no `sigma_child`; no `kappa` off-floor); writes a per-protocol `cohort_marginal`
  (labelled NOT randomised). `causal_terms=()`.
- **Modules / reports**: `lrpal01–08` (01=W, 05=P off-floor, …) + `lrpal01d`
  (W dose), each with a Quarto report foregrounding the non-randomised caveat.

## References

- Gain/level factor families: `notes/202606261230-gain-level-factors-design.md`.
- Waitlist-crossover structure / dosing: `notes/202606260702-did-crossover-design.md`.
- Consolidated DAG: `notes/202606231600-dag-revision-consolidated.md`.
- Data: Burgoyne et al. 2012, the RLI RCT (doi:10.1111/j.1469-7610.2012.02557.x).
