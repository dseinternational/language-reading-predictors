# DAG-focused gain- and level-factor families (LRPGF / LRPLF) — design decisions

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

Date: 2026-06-26

## Scope

Two statistical-model families that answer a different question from the ITT
suite. Where `LRPITT01–11` ask "did randomisation raise outcome X?", these ask
**"what is each outcome's progress associated with, and which part of that is
causal?"** — one interpretable per-outcome factor decomposition, fit eight times
(W/R/E/L/P/B/F/T):

- **`kind="gain_factors"`** (`lrpgf01–08`, plus a `…b` treated-only companion
  each) — an ANCOVA of a period's post-score on its **own** pre-score, stacked
  over every untreated and on-intervention period (`phase_mode="all"`), with a
  per-child random intercept.
- **`kind="level_factors"`** (`lrplf01–08`) — the companion *levels* view: the
  score **at each timepoint** (`phase_mode="levels"`, four rows per child, no own
  baseline), with group and ability entered as per-timepoint coefficient vectors.

This note records the decisions a future reader might question; several are
non-obvious, and a couple of tempting shortcuts are wrong.

## The factor sets (DAG-derived, per outcome)

Every model carries the **same causal skeleton** — randomised on-intervention
(the only causal term), own baseline, linear age, cognitive ability (block
design, `blocks`) — plus the **upstream DAG skills** specific to that outcome,
and three focal interactions (`trt×ability`, `trt×own`, `age×ability`):

| Sym | Outcome (measure)               | n_trials | Skill cross-predictors | Likelihood            |
| --- | ------------------------------- | -------: | ---------------------- | --------------------- |
| W   | Word reading (EWRSWR)           |       79 | L, R                   | Beta-Binomial         |
| R   | Receptive vocabulary (ROWPVT)   |      170 | — (core only)          | Beta-Binomial         |
| E   | Expressive vocabulary (EOWPVT)  |      170 | R                      | Beta-Binomial         |
| L   | Letter-sound knowledge (YARC)   |       32 | — (core only)          | Beta-Binomial         |
| P   | Phonetic spelling (SPPHON)      |       92 | L, B                   | **Bernoulli off-floor** |
| B   | Phoneme blending                |       10 | L                      | Beta-Binomial         |
| F   | Basic concept knowledge (CELF)  |       18 | R                      | Beta-Binomial         |
| T   | Receptive grammar (TROG-2)      |       32 | R                      | Beta-Binomial         |

Skills enter as the standardised pre-score logit of the upstream measure (e.g.
word reading is regressed on baseline letter sounds **L** and receptive
vocabulary **R**). They are reported as **adjusted associations**, never as
"X drives Y".

### Decision 1 — SES is excluded, and that is the DAG-faithful choice

The issue's first-draft "core" set listed SES (parental post-16 education +
age-appropriate books at home). It is **dropped from both families.** Three
reasons, all pointing the same way:

1. **SES is not a node in the consolidated DAG**
   (`notes/202606231600-dag-revision-consolidated.md`). The word-reading parent
   set is `{A, GA, IG, IS, LS, NW, PA, PS, RV, TR}`; the ITT effect is identified
   by the *empty* adjustment set. There is no arrow for an SES coefficient to
   estimate.
2. **It was found statistically redundant** in the gradient-boosting selection
   (its variance is absorbed by ability + baselines).
3. **Conditioning on it costs data:** SES is incomplete, so requiring it dropped
   the analysable sample from ~54 to ~34 children. Dropping SES is therefore both
   DAG-faithful *and* restores n (block design is complete, 54/54).

SES survives only as the dedicated `lrpitt13` SES-**robustness** companion, never
in a core factor set. Cognitive ability (`blocks`) is kept as the observed
general-ability proxy.

### Decision 2 — a per-child random intercept (latent-GA repair)

Like the mechanism and DiD families, both factor families add a non-centred
per-child random intercept. It absorbs the **time-invariant** part of latent
general ability (GA) shared across a child's measures, so the cross-lagged
"baseline X → later Y" associations are not inflated by a stable trait. The
time-*varying* part of GA is not repaired — hence the strict "association"
labelling on every non-randomised term.

### Decision 3 — gain is an own-baseline ANCOVA; treated-only is its honest sibling

The gain linear predictor is

```
eta = alpha + alpha_phase[phase] + beta_trt·OnIntervention
      + gamma_own·logit*(own_pre) + gamma_A·age_std + gamma_ability·z(blocks)
      + Σ gamma_skill·logit*(skill_pre) + Σ gamma_int·(interaction) + u_child
```

`OnIntervention = (G==1) | (phase>=1)`. The **treated-only** companion (`…b`)
restricts to on-intervention rows. There the treatment indicator is constant, so
it is **not identified**: `beta_trt` and every interaction that involves `trt`
are dropped automatically, leaving the within-treated adjusted associations. This
is a feature — it answers "among children receiving the programme, what tracks
progress?" without pretending the constant exposure is a contrast.

### Decision 4 — level uses per-timepoint vectors, and only t2 is randomised

The level model enters group and ability as **per-timepoint coefficient
vectors** (`b_grp_time[phase]`, `gamma_ability_time[phase]`), plus a `group×ability`
cross term. This is deliberate: the trial is a **waitlist crossover**, so a single
"group effect" would average a randomised contrast with a post-crossover one.

Only **`b_grp_time[1]` (t2)** is a clean randomised between-arm contrast — it is
the one post-baseline timepoint at which the immediate arm has been treated and
the waitlist has not. `b_grp_time[2]`/`[3]` (t3/t4) are **post-crossover** and are
flagged as associations (cohort/timing), not effects. The report and the
`factor_summary` `role` column carry this distinction explicitly.

### Decision 5 — phonetic spelling (P) takes the suite floor rule

P is heavily floored (most period post-scores are zero), so a graded
Beta-Binomial gain would be driven by a few dispersed tail values rather than the
factor contrasts. P therefore uses `likelihood="bernoulli_offfloor"`, identical in
spirit to the ITT suite's floor rule for P and N: a **Bernoulli on the off-floor
indicator** (`post > 0`). The linear predictor is the log-odds of coming off the
floor; there is **no `kappa`**; and the treatment marginal collapses to an
**off-floor risk difference** (`n_trials = 1`, so the items scale equals the
probability scale). The same branch is available to the level family.

## What this is — and is not

- **Causal:** only the randomised on-intervention term (gain `beta_trt`; level
  `b_grp_time[1]`). Everything else — own baseline, age, ability, skills,
  interactions, and every other timepoint — is an **adjusted association**,
  GA-confounded in its time-varying part and reported as such.
- **Never conditioned on:** intervention sessions / dose. Dose is a **collider**
  on the DAG (a descendant of both group and ability); conditioning on it would
  open a back-door. A dose variant is deferred and, if built, will use a Mundlak
  within/between split with a "within-child dose weak/inconclusive" headline.
- **Not a substitute for the ITT suite:** these triangulate and decompose; the
  randomised effect of record remains `LRPITT01–11`.

## Validation (dev config)

- **`lrpgf01` (W):** `beta_trt = +0.44` reproduces the ITT τ ≈ +0.35 — the gain
  ANCOVA recovers the randomised effect, as it should.
- **`lrpgf04` (L):** `beta_trt = +0.53` (P ≈ 0.997) — the strongest on-intervention
  signal, consistent with letter sounds being directly taught.
- **`lrpgf03` (E):** `beta_trt ≈ 0` with `gamma_R = +0.22` — expressive vocabulary
  tracks **ability/receptive vocabulary, not the intervention**, corroborating the
  Phase-0b reading-vs-vocabulary split.
- **`lrpgf05` (P, off-floor):** `beta_trt = +0.28` off-floor log-odds (risk
  difference ≈ +2.5pp, CrI −5.4..+9.7pp — weak, as the floor demands), with
  `gamma_B = +0.47` (P ≈ 0.99) and `gamma_L = +0.34` (P ≈ 0.97) recovering the
  phonics route.

Dev fits are under-tuned (R̂ ≈ 1.03–1.09); convergence is to be confirmed at the
reporting config before any estimate is read.

## Implementation

- **Factories** (`factories.py`): `build_gain_factors_model`,
  `build_level_factors_model`. Both take a `likelihood` switch
  (`beta_binomial` | `bernoulli_offfloor`) and reuse the shared priors so they
  cannot drift from the rest of the suite.
- **Pipeline** (`pipeline.py`): `fit_gain_factors`, `fit_level_factors`, with
  `_gf_/_lf_diag_vars` (kappa dropped under the off-floor branch) and
  `_gf_/_lf_coef_names` driving the factor table.
- **Preprocessing** (`preprocessing.py`): the new `phase_mode="levels"` (four
  per-timepoint rows) and a `baseline_covariates` argument that **broadcasts**
  t1-only baselines (ability/SES) across every row by subject merge. The
  broadcast is essential for the gain model's `all` mode — a t1-only baseline
  pulled per-phase would be NaN after t1 and collapse the fit to one phase.
- **Reporting** (`reporting.py`): a vector-aware `factor_summary` (emits
  per-element rows for the level vectors) tagging each term causal/association,
  and a `treatment_marginal_effect` items-scale AME (an off-floor risk difference
  under the Bernoulli branch).
- **Tests**: factory smoke tests for both families (including the treated-only
  drop, the off-floor branch, and a diag-vars guard) and preprocessing tests for
  the levels mode and the baseline broadcast.
- **Reports**: `docs/models/{lrpgf,lrplf}NN/index.qmd`, one per model.

## References

- Consolidated DAG: `notes/202606231600-dag-revision-consolidated.md`
  (and `notes/dag-language-reading.dagitty`).
- ITT suite design + floor rule: `notes/202606251321-lrpitt-suite-design.md`,
  `notes/202606251124-lrpitt-floored-outcomes-nonword-spelling.md`.
- Waitlist-crossover structure: `notes/202606260702-did-crossover-design.md`.
- Data: Burgoyne et al. 2012, the RLI RCT (doi:10.1111/j.1469-7610.2012.02557.x).
