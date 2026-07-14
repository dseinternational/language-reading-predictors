<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Model inventory

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

A catalogue of every model in this study — what it is, what outcome it targets, and
what question it answers. It is a map, not a results document: read the per-model
report (`docs/models/{model_id}/index.qmd`) and `METHODS.md` for findings, diagnostics,
and the full methodology.

The project uses a deliberate **two-step methodology** (see `METHODS.md`):

1. **Layer 1 — gradient-boosting discovery** (`src/language_reading_predictors/models/`,
   ids `lrp-rli-gbg-NNN` / `lrp-rli-gbl-NNN`). LightGBM models that _rank_ which predictors help out-of-sample
   prediction of each outcome, read with permutation importance and SHAP. Associational
   and exploratory — never causal.
2. **Layer 2 — Bayesian statistical models**
   (`src/language_reading_predictors/statistical_models/`, family-prefixed ids). PyMC
   models that estimate interpretable estimands with quantified uncertainty and, where
   the DAG supports it, a causal effect. All use a Beta-Binomial likelihood on bounded
   post-score counts via a logit linear predictor.

Both layers are built against the **revised causal DAG**
(`dag/dag-language-reading.dagitty`, revised 2026-07-10). The single most important reading
rule across the whole collection: **only the randomised contrast is causal.** In Layer 2
that is the intervention-group term (`τ`) in the ITT/DiD/gain-factor families. Every
skill→skill coupling, mechanism slope, mediator→outcome path, and dose–response is a
latent-ability-confounded **adjusted association**, never "X drives Y". Positive `τ` =
intervention benefit (`G = 2 − group`).

## At a glance

| Layer | Family (id prefix)                                                                | Count | Purpose                                                                                                                      |
| ----- | --------------------------------------------------------------------------------- | ----: | ---------------------------------------------------------------------------------------------------------------------------- |
| 1     | Gradient-boosting discovery (`lrp-rli-gbg` / `lrp-rli-gbl`)                       |    50 | Rank predictors of each outcome's gain and level                                                                             |
| 2     | ITT suite (`lrp-rli-itt`) + joint (`lrp-rli-itt-012`)                             |    31 | Randomised intervention effect on each outcome (+ joint, SES, ability & site robustness, generalisation)                     |
| 2     | Gain factors (`lrp-rli-gf`)                                                       |    19 | DAG-focused ANCOVA: randomised effect + adjusted associations on each outcome's gain                                         |
| 2     | Level factors (`lrp-rli-lf`)                                                      |    11 | Companion levels view: group×time and ability×time per timepoint                                                             |
| 2     | Waitlist-crossover / DiD (`lrp-rli-did`)                                          |    14 | Within-person replication of the ITT via the waitlist crossover (floored `P`/`N` off-floor) + treatment-effect heterogeneity |
| 2     | Aligned per-protocol (`lrp-rli-al`)                                               |     9 | Onset-aligned single 40-week gain per child (associational)                                                                  |
| 2     | Mechanism (`lrp-rli-mech-056–058`, `071–073`, `158` incl. `172`/`173`)            |     9 | Adjusted dose-response of one skill on another                                                                               |
| 2     | Mediation (`lrp-rli-med-059`–`080`; g-formula + interventional)                   |    11 | How much of a reading gain runs through a given skill                                                                        |
| 2     | Predictor / dynamics (`lrp-rli-adj-065`, `lcsm-067/081/082`, `dose-077` variants) |     8 | Baseline predictors, within-child change, lagged reverse couplings, and dose–response of word reading                        |
| 2     | Horseshoe ranking cross-check (`lrp-rli-hs-001`/`002`)                            |     2 | Regularised-horseshoe predictor ranking vs the gradient-boosting layer                                                       |
| 2     | Correlated-factor measurement model (`lrp-rli-mm-001`/`101`)                      |     2 | Correlated domain-factor measurement model of the skills                                                                     |
| 2     | Growth curves (`lrp-rli-gc-069`, `70`)                                            |     2 | Joint verbal/reading trajectories + whether baseline non-verbal ability predicts trajectory shape                            |
| 2     | Floor-sitter survival (`lrp-rli-surv-009`, `011`)                                 |     2 | Discrete-time hazard for _when_ a floored child (P / N) first comes off the floor                                            |
| 2     | Concurrent associations (`lrp-rli-ca`)                                            |     1 | Per-wave mutually-adjusted associations between contemporaneous skill levels and the focal outcome                           |
| 2     | Longitudinal correlated-factor model (`lrp-rli-lcf-001`)                          |     1 | Per-wave latent skill correlations (disattenuated counterpart to the concurrent regression family)                           |

Counts are of base models on `main` (137 statistical models in total, from `definitions.MODEL_REGISTRY`). Layer-2 selection variants (`…b` / `…base` / `…d`)
are included in the family counts and listed in the per-family tables below.

## Outcome symbols (Layer 2)

Layer-2 models refer to outcomes by short symbols; the bounded count maximum (`n`) is the
Beta-Binomial trial ceiling.

| Symbol      | Measure                                                | `n` | Notes                                  |
| ----------- | ------------------------------------------------------ | --: | -------------------------------------- |
| `W`         | Word reading (EWRSWR)                                  |  79 | Primary outcome of most analyses       |
| `R`         | Receptive vocabulary (ROWPVT)                          | 170 | Standardised (transfer) measure        |
| `E`         | Expressive vocabulary (EOWPVT)                         | 170 | Standardised (transfer) measure        |
| `L`         | Letter-sound knowledge (YARC-LSK)                      |  32 | Direct teaching target                 |
| `P`         | Phonetic spelling (SPPHON)                             |  92 | Heavily floored (~78 % at zero at t1)  |
| `B`         | Phoneme blending                                       |  10 | Direct teaching target                 |
| `F`         | Basic concept knowledge (CELF)                         |  18 |                                        |
| `T`         | Receptive grammar (TROG-2)                             |  32 |                                        |
| `N`         | Nonword reading                                        |   6 | Floored and post-only (no t1 baseline) |
| `TR` / `TE` | Taught receptive / expressive vocabulary (block 1)     |   — | Curated word set taught by RLI         |
| `UR` / `UE` | Not-taught receptive / expressive vocabulary (block 1) |   — | Generalisation comparators             |

---

## Layer 1 — Gradient-boosting discovery (`lrp-rli-gbg` / `lrp-rli-gbl`)

**Purpose.** For each outcome, fit a tuned LightGBM (GroupKFold by `subject_id`) and rank
predictors by out-of-fold permutation importance + mean |SHAP|, reading direction and
consistency from the SHAP beeswarm. This is the discovery layer that tells the Bayesian
work _which_ predictors are worth modelling. Two model families per outcome: **gain**
(predicting a `_GAIN` change score) and **level** (predicting a concurrent same-wave
level).

Gain-model rankings are near-noise (baseline-driven regression to the mean); level-model
rankings are largely concurrent same-construct correlation — read both under those
caveats (`notes/202606231100-gb-selected-features-tables.md`). This layer uses full-set
_ranking_ (`scripts/rank_predictors.py`, issue `#116`): hard feature selection was retired
in Phase D. The same-skill sibling contrast is exposed via the ranking's
`ranking_excluding_same_skill.csv` rather than per-model variants.

### Core outcomes (reading / language outcomes)

| Gain              | Level             | Outcome                                     |
| ----------------- | ----------------- | ------------------------------------------- |
| `lrp-rli-gbg-012` | `lrp-rli-gbl-012` | Word reading (`ewrswr`)                     |
| `lrp-rli-gbg-006` | `lrp-rli-gbl-006` | Expressive vocabulary (`eowpvt`)            |
| `lrp-rli-gbg-009` | `lrp-rli-gbl-009` | Letter-sound knowledge (`yarclet`)          |
| `lrp-rli-gbg-005` | `lrp-rli-gbl-005` | Receptive vocabulary (`rowpvt`)             |
| `lrp-rli-gbg-014` | `lrp-rli-gbl-014` | Basic concept knowledge (`celf`)            |
| `lrp-rli-gbg-015` | `lrp-rli-gbl-015` | Receptive grammar (`trog`)                  |
| `lrp-rli-gbg-013` | `lrp-rli-gbl-013` | Nonword reading (`nonword`)                 |
| `lrp-rli-gbg-010` | `lrp-rli-gbl-010` | Phoneme blending (`blending`)               |
| `lrp-rli-gbg-008` | `lrp-rli-gbl-008` | Expressive grammar (`aptgram`)              |
| `lrp-rli-gbg-007` | `lrp-rli-gbl-007` | Expressive information (`aptinfo`)          |
| `lrp-rli-gbg-016` | `lrp-rli-gbl-016` | DEAP fine articulation (`deappfi`)          |
| `lrp-rli-gbg-002` | `lrp-rli-gbl-002` | Taught expressive vocabulary (`b1extau`)    |
| `lrp-rli-gbg-001` | `lrp-rli-gbl-001` | Taught receptive vocabulary (`b1retau`)     |
| `lrp-rli-gbg-003` | `lrp-rli-gbl-003` | Not-taught receptive vocabulary (`b1rent`)  |
| `lrp-rli-gbg-004` | `lrp-rli-gbl-004` | Not-taught expressive vocabulary (`b1exnt`) |
| `lrp-rli-gbg-011` | `lrp-rli-gbl-011` | Phonetic spelling (`spphon`)                |

The last four rows are the #116 Phase-B additions completing the 13 priority
outcomes; their hyperparameters were MAE-tuned by Optuna on the full predictor
set (150 trials, seed 47; #169), and they do not yet have bespoke report
templates (Phase C).
`spphon` is heavily floored, so its gain ranking is expected to be near-noise.

### Speech, verbal-memory and language-sample measures (`lrp-rli-gbg`/`lrp-rli-gbl` 017–028)

Exploratory predictability discovery for measures that had only ever been predictors,
to inform the DAG's measurement side.
LSAM and `deapp_c` are level-only.

| Gain              | Level             | Outcome                                                 |
| ----------------- | ----------------- | ------------------------------------------------------- |
| `lrp-rli-gbg-017` | `lrp-rli-gbl-017` | Early Repetition Battery — nonword repetition (`erbnw`) |
| `lrp-rli-gbg-018` | `lrp-rli-gbl-018` | Early Repetition Battery — word repetition (`erbword`)  |
| `lrp-rli-gbg-019` | `lrp-rli-gbl-019` | Early Repetition Battery — total repetition (`erbto`)   |
| `lrp-rli-gbg-020` | `lrp-rli-gbl-020` | DEAP initial-consonant articulation (`deappin`)         |
| `lrp-rli-gbg-021` | `lrp-rli-gbl-021` | DEAP vowel articulation (`deappvo`)                     |
| `lrp-rli-gbg-022` | `lrp-rli-gbl-022` | DEAP average articulation (`deappav`)                   |
| —                 | `lrp-rli-gbl-023` | DEAP composite articulation (`deapp_c`)                 |
| —                 | `lrp-rli-gbl-024` | Language sample — mean length of utterance (`lsammlu`)  |
| —                 | `lrp-rli-gbl-025` | Language sample — maximum utterance length (`lsammax`)  |
| —                 | `lrp-rli-gbl-026` | Language sample — intelligibility (`lsamint`)           |
| —                 | `lrp-rli-gbl-027` | Language sample — unique words (`lsamun`)               |
| —                 | `lrp-rli-gbl-028` | Language sample — total words (`lsamto`)                |

---

## Layer 2 — Bayesian statistical models

One module per model, each defining a `SPEC = ModelSpec(...)` and a `fit(config)`. Eight
factory/pipeline families keyed by `ModelSpec.kind`. Shared priors, HSGP helpers, the
g-formula, and the floor rule live in the package; each fit writes `trace.nc`,
`diagnostics_summary.json` (the convergence gate), per-family CSVs, and diagnostic plots
to `output/statistical_models/models/{model_id}-{config}/`.

### ITT suite — `lrp-rli-itt-001–lrp-rli-itt-028` (`kind="itt"` / `"joint"`)

**Purpose.** The headline causal layer: the randomised intention-to-treat effect `τ` of
group assignment on each outcome. Under the revised DAG the ITT is identified by the
**empty adjustment set** (the own baseline and linear age enter as _precision_ terms
only); attendance/dose is never conditioned on (a collider). Heavily-floored outcomes
(`P`, `N`) take a pre-specified floor rule: a binary off-floor primary estimand plus a
flagged graded secondary. Design notes: `notes/202606251321-lrpitt-suite-design.md`,
`notes/202606251124-lrpitt-floored-outcomes-nonword-spelling.md`.

| Model                     | Outcome             | Purpose                                                                                        |
| ------------------------- | ------------------- | ---------------------------------------------------------------------------------------------- |
| `lrp-rli-itt-001`         | `TR`                | ITT on taught receptive vocabulary (block 1)                                                   |
| `lrp-rli-itt-002`         | `TE`                | ITT on taught expressive vocabulary (block 1)                                                  |
| `lrp-rli-itt-003`         | `UR`                | ITT on not-taught receptive vocabulary (block 1)                                               |
| `lrp-rli-itt-004`         | `UE`                | ITT on not-taught expressive vocabulary (block 1)                                              |
| `lrp-rli-itt-005`         | `R`                 | ITT on standardised receptive vocabulary                                                       |
| `lrp-rli-itt-006`         | `E`                 | ITT on standardised expressive vocabulary                                                      |
| `lrp-rli-itt-007`         | `L`                 | ITT on letter-sound knowledge                                                                  |
| `lrp-rli-itt-008`         | `B`                 | ITT on phoneme blending                                                                        |
| `lrp-rli-itt-009`         | `P`                 | ITT on phonetic spelling — floor-rule branch                                                   |
| `lrp-rli-itt-010`         | `W`                 | **ITT on word reading** (the primary effect; supersedes the former LRP52)                      |
| `lrp-rli-itt-011`         | `N`                 | ITT on nonword reading — floor-rule branch                                                     |
| `lrp-rli-itt-012`         | joint               | Joint model over all suite outcomes (optional LKJ residual correlation)                        |
| `lrp-rli-itt-013` / `13b` | `W` / `L`           | SES-adjusted ITT (mother's education etc.)                                                     |
| `lrp-rli-itt-014` / `14b` | `W` / `L`           | Unadjusted ITT on the SES complete-case subset — matched comparator to `lrp-rli-itt-013`/`13b` |
| `lrp-rli-itt-015` / `15b` | contrast            | Generalisation: taught vs not-taught vocabulary, expressive (`15`) and receptive (`15b`)       |
| `lrp-rli-itt-017–020`     | `TR`,`TE`,`UR`,`UE` | Ability-adjusted (block-design) robustness across the taught/untaught vocabulary family        |
| `lrp-rli-itt-021` / `22`  | `R` / `E`           | Ability-adjusted robustness, standardised vocabulary                                           |
| `lrp-rli-itt-023` / `24`  | `L` / `W`           | Ability-adjusted robustness, letter sounds and word reading                                    |
| `lrp-rli-itt-025`         | `F`                 | ITT on basic concepts (CELF) — effect only (no agreed δ, so no meaningful-benefit table)       |
| `lrp-rli-itt-026`         | `T`                 | ITT on receptive grammar (TROG-2) — effect only (no agreed δ)                                  |
| `lrp-rli-itt-027` / `28`  | `W` / `L`           | Site-adjusted (`area`, North/South) robustness — `area` complete, so no matched comparator     |

_(`lrp-rli-itt-016` is intentionally unused — reserved for a deferred descriptive floored-outcome
trajectory complement.)_

### Gain factors — `lrp-rli-gf-001–lrp-rli-gf-011` (+ `…b`) (`kind="gain_factors"`)

**Purpose.** A DAG-focused ANCOVA on each outcome's period gain (post-score given its own
pre-score), stacking every on-intervention and untreated period with a child random
intercept — a partial, shrunken stand-in for between-child heterogeneity, **not** a
control for latent ability. The randomised on-intervention term is the **only** causal
coefficient, and its probability/items-scale marginal effect is averaged over the
**period-1** (randomised) transition only; own baseline, age, cognitive ability (block
design), the upstream DAG skill baselines (`skill_symbols`), the revised-DAG non-measure
confounders hearing/speech/phonological memory (`adjust_for`), and focal interactions are
explicit _adjusted associations_. Adjustment sets were re-derived against the revised DAG
in #247. The `…b` variant is treated-only (gains while on intervention). Design note:
`notes/202606261230-gain-level-factors-design.md`; re-derivation:
`notes/202607122200-gf-lf-revised-dag-adjustments.md`.

**Naming note.** "Factors" here (and in the level-factors family below) carries its plain-English sense — the observed covariates _associated with_ gains or levels — not the factor-analysis sense: these are regression models with no latent variables. The latent measurement model is `lrp-rli-mm-001` (`kind="corr_factor"`).

| Model            | Outcome | Skill baselines (`skill_symbols`)         | Confounders (`adjust_for`) | Treated-only `…b` |
| ---------------- | ------- | ----------------------------------------- | -------------------------- | ----------------- |
| `lrp-rli-gf-001` | `W`     | `TR`, `TE`, `R`, `E`, `L`, `N`, `B`       | —                          | `lrp-rli-gf-101`  |
| `lrp-rli-gf-002` | `R`     | `TR`                                      | `HS`, `RW`                 | `lrp-rli-gf-102`  |
| `lrp-rli-gf-003` | `E`     | `R`, `TR`, `TE`                           | `HS`, `SP`, `RW`           | `lrp-rli-gf-103`  |
| `lrp-rli-gf-004` | `L`     | —                                         | `HS`, `SP`                 | `lrp-rli-gf-104`  |
| `lrp-rli-gf-005` | `P`     | `L`, `B` (off-floor Bernoulli likelihood) | `RW`                       | `lrp-rli-gf-105`  |
| `lrp-rli-gf-006` | `B`     | `L`, `E`, `TE`                            | `HS`, `SP`, `RW`           | `lrp-rli-gf-106`  |
| `lrp-rli-gf-007` | `F`     | `R`, `TR`                                 | —                          | `lrp-rli-gf-107`  |
| `lrp-rli-gf-008` | `T`     | `R`, `TR`                                 | —                          | `lrp-rli-gf-108`  |
| `lrp-rli-gf-009` | `TR`    | —                                         | `HS`, `RW`                 | —                 |
| `lrp-rli-gf-010` | `TE`    | `TR`                                      | `HS`, `SP`, `RW`           | —                 |
| `lrp-rli-gf-011` | `N`     | `L`, `B` (off-floor Bernoulli likelihood) | `SP`, `RW`                 | —                 |

### Level factors — `lrp-rli-lf-001–lrp-rli-lf-011` (`kind="level_factors"`)

**Purpose.** The companion _levels_ view of each outcome (the score at each timepoint, no
own baseline), with group×time and ability×time as per-timepoint coefficient vectors. Only
the t2 group contrast is a clean randomised effect; later timepoints are post-crossover and
flagged as associations. Each outcome carries the same revised-DAG exogenous confounders
(`adjust_for`: hearing/speech/phonological memory) as its gain-factor sibling, but **no**
measure-skill adjusters — in a levels model a skill's contemporaneous level is a
post-treatment mediator of the group×time effect (#247). Outcomes mirror the gain-factor
family: `lrp-rli-lf-001` `W`, `02` `R`, `03` `E`, `04` `L`, `05` `P` (off-floor), `06` `B`,
`07` `F`, `08` `T`, `09` `TR`, `10` `TE`, `11` `N` (off-floor).
The gain-factors naming note applies here too: "factors" means observed regression covariates, not latent factors.

### Waitlist-crossover / difference-in-differences — `lrp-rli-did-001–lrp-rli-did-013` (+ `lrp-rli-did-107`) (`kind="did"`)

**Purpose.** A within-person replication of the randomised ITT, using the waitlist arm's
crossover: each child is partly their own control, with the immediate arm anchoring the
time/maturation trend (Beta-Binomial logit so the ceiling is respected). A second,
non-randomised view that triangulates the ITT. The heavily floored outcomes (`P`, `N`)
instead take the suite's **off-floor floor rule** — a Bernoulli on the off-floor
indicator — so their `delta` is the within-person effect on the log-odds of coming off
the floor (an off-floor risk difference on the items scale), mirroring their ITT siblings.
Design note: `notes/202606260702-did-crossover-design.md`.

| Model             | Outcome | Purpose                                                                                                                                                   |
| ----------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lrp-rli-did-001` | `W`     | Within-person DiD effect on word reading                                                                                                                  |
| `lrp-rli-did-002` | `L`     | Within-person DiD effect on letter-sound knowledge                                                                                                        |
| `lrp-rli-did-003` | `B`     | Within-person DiD effect on phoneme blending                                                                                                              |
| `lrp-rli-did-004` | `TE`    | Within-person DiD effect on taught expressive vocabulary                                                                                                  |
| `lrp-rli-did-005` | `R`     | Within-person DiD effect on receptive vocabulary (the null control)                                                                                       |
| `lrp-rli-did-006` | `W`     | Word-reading DiD with a session **dose-response** term                                                                                                    |
| `lrp-rli-did-007` | `L`     | Letter-sound DiD with a **period-resolved** session dose slope (#135); `lrp-rli-did-107` is its pooled (no-period-variation) comparator                   |
| `lrp-rli-did-008` | `TR`    | Within-person DiD effect on taught receptive vocabulary                                                                                                   |
| `lrp-rli-did-009` | `E`     | Within-person DiD effect on standardised expressive vocabulary                                                                                            |
| `lrp-rli-did-010` | `F`     | Within-person DiD effect on basic concept knowledge                                                                                                       |
| `lrp-rli-did-011` | `P`     | Within-person DiD effect on phonetic spelling (off-floor)                                                                                                 |
| `lrp-rli-did-012` | `N`     | Within-person DiD effect on nonword reading (off-floor)                                                                                                   |
| `lrp-rli-did-013` | `W`     | Word-reading DiD with a per-child random on-intervention slope; reports `sigma_delta`, the between-child SD (treatment-effect heterogeneity, #230 §2/§4a) |

### Aligned per-protocol — `lrp-rli-al-001–lrp-rli-al-008` (+ `lrp-rli-al-101`) (`kind="aligned"`)

**Purpose.** An onset-aligned, per-protocol single gain: both arms aligned by intervention
onset (immediate t1→t3, waitlist t2→t4) into one cross-sectional Beta-Binomial ANCOVA per
child. The cohort contrast is **not** randomised (confounded by age-at-onset and timing),
so _no_ term is causal — every coefficient is an association. Design note:
`notes/202606261343-lrpal-aligned-design.md`. Outcomes: `lrp-rli-al-001` `W`, `02` `R`, `03` `E`,
`04` `L`, `05` `P` (off-floor), `06` `B`, `07` `F`, `08` `T`; **`lrp-rli-al-101`** adds a
cumulative-session dose sensitivity term (a collider — sensitivity only).

### Mechanism — `lrp-rli-mech-056–lrp-rli-mech-058`, `lrp-rli-mech-071–lrp-rli-mech-073`, `lrp-rli-mech-088–lrp-rli-mech-090` (`kind="mechanism"`)

**Purpose.** The adjustment-set dose-response of one measured skill on another across all
phases, with subject random intercepts and optional linear moderation. Every slope is an
**adjusted association** (latent-ability confounded), not a causal effect.

| Model                         | Path     | Purpose                                                                                                                   |
| ----------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------- |
| `lrp-rli-mech-056`            | `R → W`  | Receptive vocabulary → word reading                                                                                       |
| `lrp-rli-mech-057`            | `E → W`  | Expressive vocabulary → word reading                                                                                      |
| `lrp-rli-mech-058`            | `L → W`  | Letter-sound knowledge → word reading                                                                                     |
| `lrp-rli-mech-071`            | `L → W`  | Letter sounds → word reading, linear moderation by expressive vocabulary `E`                                              |
| `lrp-rli-mech-072` / `72base` | `L → N`  | Code-based route: letter sounds moderated by blending `B` → decoding (with / without the interaction)                     |
| `lrp-rli-mech-073` / `73base` | `L → W`  | Letter sounds → word reading, moderated by age (with / without the interaction)                                           |
| `lrp-rli-mech-088`            | `TR → W` | Taught receptive vocabulary → word reading (#311; linear, IS backdoor flagged not adjusted)                               |
| `lrp-rli-mech-089`            | `TE → W` | Taught expressive vocabulary → word reading (#311; linear, TR measure confounder, IS flagged)                             |
| `lrp-rli-mech-090`            | `RW → W` | Phonological memory (word/nonword repetition) → word reading (#311; covariate exposure, adjust `HS` only, no IS backdoor) |

### Mediation — `lrp-rli-med-059`, `lrp-rli-med-062`, `lrp-rli-med-064` (`kind="mediation"` / `"mediation_multi"`)

**Purpose.** g-formula NDE/NIE decomposition: how much of the intervention's word-reading
gain runs through a given skill. Not point-identified under the revised DAG (latent ability +
same-wave mediator/outcome) — reported as triangulation, leading with the robust quantity.

| Model             | Purpose                                                                                               |
| ----------------- | ----------------------------------------------------------------------------------------------------- |
| `lrp-rli-med-059` | Single-mediator: word-reading gain via letter-sound knowledge `L`                                     |
| `lrp-rli-med-062` | Reading-route decomposition: code-based-route (`L` + blending `B`) vs lexical share                   |
| `lrp-rli-med-064` | Two-mediator split: `L` vs expressive vocabulary `E` (joint indirect + path-specific `NIE_L`/`NIE_E`) |

### Predictor / within-child dynamics — `lrp-rli-adj-065`, `lrp-rli-lcsm-067/081/082`, `lrp-rli-dose-077` (+ variants)

**Purpose.** Complementary, explicitly **associational** views of skill progress that sit
outside the randomised families — including the time-lagged reverse-coupling suite built on
the wave-unrolled DAG (#250; design `notes/202607141030-time-lagged-model-designs.md`).

| Model              | Kind            | Purpose                                                                                                                                                                                                         |
| ------------------ | --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lrp-rli-adj-065`  | `adjusted`      | Between-child: which wave-1 baseline skills go with more subsequent word-reading gain, mutually adjusted                                                                                                        |
| `lrp-rli-lcsm-067` | `lcsm`          | Within-child latent change-score: prior-wave letter sounds `L` and vocabulary `E` as predictors of reading _change_                                                                                             |
| `lrp-rli-lcsm-081` | `lcsm`          | Lagged reverse coupling: prior word reading `W` predicting taught-vocabulary (`TE`/`TR`) change — crossover-aware arm × window intercepts + `hs`/`rw`/`sp` adjusters (the verified lagged-DAG backdoor set)     |
| `lrp-rli-lcsm-181` | `lcsm`          | No-reverse-coupling LOO comparator for `lcsm-081` ("does the reverse edge earn its place predictively")                                                                                                         |
| `lrp-rli-lcsm-082` | `lcsm`          | Reciprocal dominance (exploratory): blending `B` ↔ word reading `W` lagged cross-couplings with an SD-standardised dominance contrast; broadly confounded in both directions                                    |
| `lrp-rli-dose-077` | `dose_response` | Period-resolved observational dose-response of intervention sessions → word reading; `lrp-rli-dose-177` adds an ability-adjusted sensitivity, `lrp-rli-dose-277` is the pooled (no-period-variation) comparator |

`lrp-rli-dose-077`'s dose terms are observational (sessions = a DAG collider as exposure): an adjusted
within-child association, never "more sessions cause more gain". Only the randomised ITT/DiD
contrasts carry the causal claim — in the lagged suite that is solely `lcsm-081/082`'s window-1
arm contrast (`itt_window1_contrast.csv`), reported as an ITT-suite consistency check.

### Joint growth curves — `lrp-rli-gc-069`, `lrp-rli-gc-070` (`kind="growth"`)

**Purpose.** Characterise the **longitudinal trajectories** of the five verbal/reading
measures (`R`, `E`, `T`, `W`, `L`) across the four RLI waves and ask whether **baseline
non-verbal ability** (`blocks`, WPPSI Block Design, t1-only, complete for all 54 children)
predicts their _shape_ — the descriptive Q5 answer (issue #187). Each measure gets a
per-child latent logit intercept + linear age slope (masked Beta-Binomial); `gamma`
(non-verbal → growth _rate_) is the headline estimand, `delta` the effect on baseline
_level_.

| Model            | Kind     | Purpose                                                                                                                                                                                                        |
| ---------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lrp-rli-gc-069` | `growth` | Independent-core: per-measure trajectories; baseline non-verbal ability → each measure's growth rate (`gamma`) + level (`delta`)                                                                               |
| `lrp-rli-gc-070` | `growth` | Adds a rank-1 shared growth-tempo factor: do the measures grow together, and does non-verbal ability predict the common tempo? `LOO(lrp-rli-gc-069 vs lrp-rli-gc-070)` tests whether the factor earns its keep |

`gamma`/`delta` are **adjusted, `GA`-confounded associations, never causal** — block design
is an off-DAG ability proxy (revised DAG) and the child random intercept only _partially_
adjusts. Descriptive natural-history, `n≈54` (wide intervals). Byrne-cohort replication is a
gated follow-up (unconfirmed `bpvs`/`basmat` ceilings; `basmat` is wave-3+, so no baseline).

### Floor-sitter survival — `lrp-rli-surv-009`, `lrp-rli-surv-011` (`kind="survival"`)

**Purpose.** The four-wave generalisation of the floored P/N off-floor rule (siblings
`lrp-rli-itt-009`/`011`): instead of the single t1→t2 off-floor transition, a **discrete-time
survival** model for _when_ a child at the floor at baseline first comes off it (issue #230 §5).
The at-risk set is the children at the floor at t1; one person-period row per still-at-floor
interval (t1→t2, t2→t3, t3→t4); the event is the first score above zero. The hazard uses a
complementary-log-log link (logistic variant as sensitivity), with a per-interval baseline
hazard, baseline (t1) letter-sound knowledge and word reading as prognostic covariates, and an
intervention-aligned treatment hazard shift `tau` (immediate arm treated throughout, waitlist
arm from its crossover; `G = 2 − group`, positive = benefit).

| Model              | Kind       | Outcome | Purpose                                                      |
| ------------------ | ---------- | ------- | ------------------------------------------------------------ |
| `lrp-rli-surv-009` | `survival` | `P`     | time-to-off-floor hazard, phonetic spelling (base `itt-009`) |
| `lrp-rli-surv-011` | `survival` | `N`     | time-to-off-floor hazard, nonword reading (base `itt-011`)   |

**Prognostic, not causal.** By t4 both arms have been treated, so `tau` is an association
anchored on the immediate arm's randomised first interval, not a randomised effect of record;
concurrent letter sounds are excluded as a treatment-affected mediator. Descriptive companion:
`notes/…-persistent-floor-sitters-nonword-spelling.md` + `scripts/descriptive/floor_sitters.py`.

### Concurrent conditional associations — `lrp-rli-ca-001` (`kind="concurrent"`)

**Purpose.** The one family that describes how contemporaneous skill levels co-occur with the
focal outcome at each wave (#312, descriptive-association workstream #314). At every timepoint it
fits a between-child Beta-Binomial regression of the focal outcome's _level_ on the standardised
same-wave logits of a skill set (letter sounds, blending, taught receptive/expressive vocabulary,
receptive/expressive vocabulary), plus age and a group nuisance term — "at wave t, among children
alike on age and the other skills, +n of a predictor is associated with +m of the outcome". The
four waves are fitted separately (one row per child per wave) and reported side by side; the
best-powered wave carries the convergence gate and posterior diagnostics, the others are sub-fits.

| Model            | Kind         | Outcome | Purpose                                                                  |
| ---------------- | ------------ | ------- | ------------------------------------------------------------------------ |
| `lrp-rli-ca-001` | `concurrent` | `W`     | per-wave conditional associations of concurrent skills with word reading |

**Association only — three caveats.** Every coefficient is an adjusted association; conditioning
on contemporaneous (post-treatment) skill levels is intentional because nothing is read causally
(contrast the level-factors family, which omits cross-skill terms to protect a causal contrast).
Read with the **Table-2 fallacy** (each coefficient answers a different conditional question),
**regression dilution** (observed-score predictors attenuate the associations — the longitudinal
factor model #313 is the follow-on instrument), and **collinearity shrinkage** (n ≈ 53 with a
correlated predictor cluster, so the adjusted-vs-bivariate gap is itself informative; both are
reported). Group is a non-interpretable nuisance. Extension: letter sounds and taught vocabulary
as further focal outcomes once the word-reading model settles the pattern.

### Longitudinal correlated-domain-factor model — `lrp-rli-lcf-001` (`kind="long_corr_factor"`)

**Purpose.** The symmetric, disattenuated counterpart to the concurrent regression family (#313,
descriptive-association workstream #314) and the four-wave extension of the cross-sectional
`corr_factor` CFA (`lrp-rli-mm-001`). It estimates correlated **vocabulary {R,E,TR,TE} / code {L,B}
/ grammar {F,T}** domain factors at every timepoint over the child×wave panel and reports the
**per-wave latent skill correlation matrices** plus the conditional latent slopes derived from them
— the skill coupling with the binomial counting noise on both sides modelled out. Scalar-invariant
loadings, per-wave within-factor LKJ correlations, a trait/state across-wave decomposition, and the
factor scores marginalised out (the `mm-001` funnel fix); missing cells are masked, not dropped.

| Model             | Kind               | Outcome | Purpose                                                      |
| ----------------- | ------------------ | ------- | ------------------------------------------------------------ |
| `lrp-rli-lcf-001` | `long_corr_factor` | `W`     | per-wave latent skill correlations (vocabulary/code/grammar) |

**Measurement / triangulation only.** Every latent correlation and slope is a descriptive
association (ID-2), never causal. Fragile and prior-dependent at n ≈ 54 — read the wide intervals
as the honest result. A self-contained **disattenuation cross-check** (latent correlation ≥ observed,
since measurement-error correction can only inflate) anchors it against the #312 regression AMEs.
Follow-ups: a prior-sensitivity companion and the invariance relaxation (configural / AR-across-wave,
chosen by LOO). See `notes/202607142330-lrp313-longitudinal-corr-factor.md`.

---

## Conventions and pointers

- **Fit a model:** `python scripts/fit_statistical_model.py {model_id|all} --config dev|test|reporting [--render]` (Layer 2); `python scripts/fit_model.py {model_id|all} --config dev [--render]` (Layer 1).
- **Reports:** one per model at `docs/models/{model_id}/index.qmd`; thin templates that include shared partials from `docs/models/_partials/`. Selection variants fall back to their parent's template.
- **Cross-model comparisons:** `scripts/compare_statistical_models.py` (ITT-vs-joint `τ` consistency, `τ` and mechanism-slope forests, nested PSIS-LOO).
- **Interpreting results:** check convergence (R-hat ≈ 1.00, ESS, ≤ 1 % divergences) before interpreting; report the posterior (median + equal-tailed 95 % credible interval + tail probability, with an HPDI sensitivity interval alongside); positive `τ` = intervention helps; only `τ` is causal. Full guidance in `METHODS.md`.
- **Source of truth:** Layer-1 ids/outcomes live in each module + `models/registry.py`; Layer-2 in each module's `SPEC` (`statistical_models/`). Keep this inventory in step with those when models are added, renamed, or retired.
