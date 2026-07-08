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
   ids `lrpNN`). LightGBM models that _rank_ which predictors help out-of-sample
   prediction of each outcome, read with permutation importance and SHAP. Associational
   and exploratory — never causal.
2. **Layer 2 — Bayesian statistical models**
   (`src/language_reading_predictors/statistical_models/`, family-prefixed ids). PyMC
   models that estimate interpretable estimands with quantified uncertainty and, where
   the DAG supports it, a causal effect. All use a Beta-Binomial likelihood on bounded
   post-score counts via a logit linear predictor.

Both layers are built against the **locked causal DAG**
(`notes/202606231600-dag-revision-consolidated.md`). The single most important reading
rule across the whole collection: **only the randomised contrast is causal.** In Layer 2
that is the intervention-group term (`τ`) in the ITT/DiD/gain-factor families. Every
skill→skill coupling, mechanism slope, mediator→outcome path, and dose–response is a
latent-ability-confounded **adjusted association**, never "X drives Y". Positive `τ` =
intervention benefit (`G = 2 − group`).

## At a glance

| Layer | Family (id prefix)                                                        | Count | Purpose                                                                                           |
| ----- | ------------------------------------------------------------------------- | ----: | ------------------------------------------------------------------------------------------------- |
| 1     | Gradient-boosting discovery (`lrpgbg##` / `lrpgbl##`)                     |    50 | Rank predictors of each outcome's gain and level                                                  |
| 2     | ITT suite (`lrp-rli-itt`)                                                 |    26 | Randomised intervention effect on each outcome (+ joint, SES, ability robustness, generalisation) |
| 2     | Gain factors (`lrp-rli-gf`)                                               |    16 | DAG-focused ANCOVA: randomised effect + adjusted associations on each outcome's gain              |
| 2     | Level factors (`lrp-rli-lf`)                                              |     8 | Companion levels view: group×time and ability×time per timepoint                                  |
| 2     | Waitlist-crossover / DiD (`lrp-rli-did`)                                  |     6 | Within-person replication of the ITT via the waitlist crossover                                   |
| 2     | Aligned per-protocol (`lrp-rli-al`)                                       |     9 | Onset-aligned single 40-week gain per child (associational)                                       |
| 2     | Mechanism (`lrp-rli-mech-056–058`, `071–073` incl. `172`/`173`)           |     8 | Adjusted dose-response of one skill on another                                                    |
| 2     | Mediation (`lrp-rli-med-059`, `62`, `64`)                                 |     3 | How much of a reading gain runs through a given skill                                             |
| 2     | Predictor / dynamics (`lrp-rli-adj-065`, `67`, `77` incl. `77a`/`77base`) |     5 | Baseline predictors, within-child change, and dose–response of word reading                       |
| 2     | Growth curves (`lrp-rli-gc-069`, `70`)                                    |     2 | Joint verbal/reading trajectories + whether baseline non-verbal ability predicts trajectory shape |

Counts are of base models on `main`. Layer-2 selection variants (`…b` / `…base` / `…d`)
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

## Layer 1 — Gradient-boosting discovery (`lrpgbg##` / `lrpgbl##`)

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

| Gain       | Level      | Outcome                                     |
| ---------- | ---------- | ------------------------------------------- |
| `lrpgbg12` | `lrpgbl12` | Word reading (`ewrswr`)                     |
| `lrpgbg06` | `lrpgbl06` | Expressive vocabulary (`eowpvt`)            |
| `lrpgbg09` | `lrpgbl09` | Letter-sound knowledge (`yarclet`)          |
| `lrpgbg05` | `lrpgbl05` | Receptive vocabulary (`rowpvt`)             |
| `lrpgbg14` | `lrpgbl14` | Basic concept knowledge (`celf`)            |
| `lrpgbg15` | `lrpgbl15` | Receptive grammar (`trog`)                  |
| `lrpgbg13` | `lrpgbl13` | Nonword reading (`nonword`)                 |
| `lrpgbg10` | `lrpgbl10` | Phoneme blending (`blending`)               |
| `lrpgbg08` | `lrpgbl08` | Expressive grammar (`aptgram`)              |
| `lrpgbg07` | `lrpgbl07` | Expressive information (`aptinfo`)          |
| `lrpgbg16` | `lrpgbl16` | DEAP fine articulation (`deappfi`)          |
| `lrpgbg02` | `lrpgbl02` | Taught expressive vocabulary (`b1extau`)    |
| `lrpgbg01` | `lrpgbl01` | Taught receptive vocabulary (`b1retau`)     |
| `lrpgbg03` | `lrpgbl03` | Not-taught receptive vocabulary (`b1rent`)  |
| `lrpgbg04` | `lrpgbl04` | Not-taught expressive vocabulary (`b1exnt`) |
| `lrpgbg11` | `lrpgbl11` | Phonetic spelling (`spphon`)                |

The last four rows are the #116 Phase-B additions completing the 13 priority
outcomes; their hyperparameters are borrowed from the nearest analogue pending a
target-specific tune, and they do not yet have bespoke report templates (Phase C).
`spphon` is heavily floored, so its gain ranking is expected to be near-noise.

### Speech, verbal-memory and language-sample measures (`lrpgbg`/`lrpgbl` 17–28)

Exploratory predictability discovery for measures that had only ever been predictors,
to inform the DAG's measurement side.
LSAM and `deapp_c` are level-only.

| Gain       | Level      | Outcome                                                 |
| ---------- | ---------- | ------------------------------------------------------- |
| `lrpgbg17` | `lrpgbl17` | Early Repetition Battery — nonword repetition (`erbnw`) |
| `lrpgbg18` | `lrpgbl18` | Early Repetition Battery — word repetition (`erbword`)  |
| `lrpgbg19` | `lrpgbl19` | Early Repetition Battery — total repetition (`erbto`)   |
| `lrpgbg20` | `lrpgbl20` | DEAP initial-consonant articulation (`deappin`)         |
| `lrpgbg21` | `lrpgbl21` | DEAP vowel articulation (`deappvo`)                     |
| `lrpgbg22` | `lrpgbl22` | DEAP average articulation (`deappav`)                   |
| —          | `lrpgbl23` | DEAP composite articulation (`deapp_c`)                 |
| —          | `lrpgbl24` | Language sample — mean length of utterance (`lsammlu`)  |
| —          | `lrpgbl25` | Language sample — maximum utterance length (`lsammax`)  |
| —          | `lrpgbl26` | Language sample — intelligibility (`lsamint`)           |
| —          | `lrpgbl27` | Language sample — unique words (`lsamun`)               |
| —          | `lrpgbl28` | Language sample — total words (`lsamto`)                |

---

## Layer 2 — Bayesian statistical models

One module per model, each defining a `SPEC = ModelSpec(...)` and a `fit(config)`. Eight
factory/pipeline families keyed by `ModelSpec.kind`. Shared priors, HSGP helpers, the
g-formula, and the floor rule live in the package; each fit writes `trace.nc`,
`diagnostics_summary.json` (the convergence gate), per-family CSVs, and diagnostic plots
to `output/statistical_models/models/{model_id}-{config}/`.

### ITT suite — `lrp-rli-itt-001–lrp-rli-itt-024` (`kind="itt"` / `"joint"`)

**Purpose.** The headline causal layer: the randomised intention-to-treat effect `τ` of
group assignment on each outcome. Under the locked DAG the ITT is identified by the
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

_(`lrp-rli-itt-016` is intentionally unused — reserved for a deferred descriptive floored-outcome
trajectory complement.)_

### Gain factors — `lrp-rli-gf-001–lrp-rli-gf-008` (+ `…b`) (`kind="gain_factors"`)

**Purpose.** A DAG-focused ANCOVA on each outcome's period gain (post-score given its own
pre-score), stacking every on-intervention and untreated period with a child random
intercept (the partial latent-ability repair). The randomised on-intervention term is the
**only** causal coefficient; own baseline, age, cognitive ability (block design), upstream
DAG skills, and focal interactions are explicit _adjusted associations_. The `…b` variant
is treated-only (gains while on intervention). Design note:
`notes/202606261230-gain-level-factors-design.md`.

| Model            | Outcome | Cross-skill terms                         | Treated-only `…b` |
| ---------------- | ------- | ----------------------------------------- | ----------------- |
| `lrp-rli-gf-001` | `W`     | letter sounds `L`, receptive vocab `R`    | `lrp-rli-gf-101`  |
| `lrp-rli-gf-002` | `R`     | —                                         | `lrp-rli-gf-102`  |
| `lrp-rli-gf-003` | `E`     | `R`                                       | `lrp-rli-gf-103`  |
| `lrp-rli-gf-004` | `L`     | —                                         | `lrp-rli-gf-104`  |
| `lrp-rli-gf-005` | `P`     | `L`, `B` (off-floor Bernoulli likelihood) | `lrp-rli-gf-105`  |
| `lrp-rli-gf-006` | `B`     | `L`                                       | `lrp-rli-gf-106`  |
| `lrp-rli-gf-007` | `F`     | `R`                                       | `lrp-rli-gf-107`  |
| `lrp-rli-gf-008` | `T`     | `R`                                       | `lrp-rli-gf-108`  |

### Level factors — `lrp-rli-lf-001–lrp-rli-lf-008` (`kind="level_factors"`)

**Purpose.** The companion _levels_ view of each outcome (the score at each timepoint, no
own baseline), with group×time and ability×time as per-timepoint coefficient vectors. Only
the t2 group contrast is a clean randomised effect; later timepoints are post-crossover and
flagged as associations. Outcomes mirror the gain-factor family: `lrp-rli-lf-001` `W`, `02` `R`,
`03` `E`, `04` `L`, `05` `P` (off-floor), `06` `B`, `07` `F`, `08` `T`.

### Waitlist-crossover / difference-in-differences — `lrp-rli-did-001–lrp-rli-did-006` (`kind="did"`)

**Purpose.** A within-person replication of the randomised ITT, using the waitlist arm's
crossover: each child is partly their own control, with the immediate arm anchoring the
time/maturation trend (Beta-Binomial logit so the ceiling is respected). A second,
non-randomised view that triangulates the ITT. Design note:
`notes/202606260702-did-crossover-design.md`.

| Model             | Outcome | Purpose                                                             |
| ----------------- | ------- | ------------------------------------------------------------------- |
| `lrp-rli-did-001` | `W`     | Within-person DiD effect on word reading                            |
| `lrp-rli-did-002` | `L`     | Within-person DiD effect on letter-sound knowledge                  |
| `lrp-rli-did-003` | `B`     | Within-person DiD effect on phoneme blending                        |
| `lrp-rli-did-004` | `TE`    | Within-person DiD effect on taught expressive vocabulary            |
| `lrp-rli-did-005` | `R`     | Within-person DiD effect on receptive vocabulary (the null control) |
| `lrp-rli-did-006` | `W`     | Word-reading DiD with a session **dose-response** term              |

### Aligned per-protocol — `lrp-rli-al-001–lrp-rli-al-008` (+ `lrp-rli-al-101`) (`kind="aligned"`)

**Purpose.** An onset-aligned, per-protocol single gain: both arms aligned by intervention
onset (immediate t1→t3, waitlist t2→t4) into one cross-sectional Beta-Binomial ANCOVA per
child. The cohort contrast is **not** randomised (confounded by age-at-onset and timing),
so _no_ term is causal — every coefficient is an association. Design note:
`notes/202606261343-lrpal-aligned-design.md`. Outcomes: `lrp-rli-al-001` `W`, `02` `R`, `03` `E`,
`04` `L`, `05` `P` (off-floor), `06` `B`, `07` `F`, `08` `T`; **`lrp-rli-al-101`** adds a
cumulative-session dose sensitivity term (a collider — sensitivity only).

### Mechanism — `lrp-rli-mech-056–lrp-rli-mech-058`, `lrp-rli-mech-071–lrp-rli-mech-073` (`kind="mechanism"`)

**Purpose.** The adjustment-set dose-response of one measured skill on another across all
phases, with subject random intercepts and optional linear moderation. Every slope is an
**adjusted association** (latent-ability confounded), not a causal effect.

| Model                         | Path    | Purpose                                                                                               |
| ----------------------------- | ------- | ----------------------------------------------------------------------------------------------------- |
| `lrp-rli-mech-056`            | `R → W` | Receptive vocabulary → word reading                                                                   |
| `lrp-rli-mech-057`            | `E → W` | Expressive vocabulary → word reading                                                                  |
| `lrp-rli-mech-058`            | `L → W` | Letter-sound knowledge → word reading                                                                 |
| `lrp-rli-mech-071`            | `L → W` | Letter sounds → word reading, linear moderation by expressive vocabulary `E`                          |
| `lrp-rli-mech-072` / `72base` | `L → N` | Code-based route: letter sounds moderated by blending `B` → decoding (with / without the interaction) |
| `lrp-rli-mech-073` / `73base` | `L → W` | Letter sounds → word reading, moderated by age (with / without the interaction)                       |

### Mediation — `lrp-rli-med-059`, `lrp-rli-med-062`, `lrp-rli-med-064` (`kind="mediation"` / `"mediation_multi"`)

**Purpose.** g-formula NDE/NIE decomposition: how much of the intervention's word-reading
gain runs through a given skill. Not point-identified under the locked DAG (latent ability +
same-wave mediator/outcome) — reported as triangulation, leading with the robust quantity.

| Model             | Purpose                                                                                               |
| ----------------- | ----------------------------------------------------------------------------------------------------- |
| `lrp-rli-med-059` | Single-mediator: word-reading gain via letter-sound knowledge `L`                                     |
| `lrp-rli-med-062` | Reading-route decomposition: code-based-route (`L` + blending `B`) vs lexical share                   |
| `lrp-rli-med-064` | Two-mediator split: `L` vs expressive vocabulary `E` (joint indirect + path-specific `NIE_L`/`NIE_E`) |

### Predictor / within-child dynamics — `lrp-rli-adj-065`, `lrp-rli-lcsm-067`, `lrp-rli-dose-077` (+ variants)

**Purpose.** Three complementary, explicitly **associational** views of word-reading
progress that sit outside the randomised families.

| Model              | Kind            | Purpose                                                                                                                                                                                                         |
| ------------------ | --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lrp-rli-adj-065`  | `adjusted`      | Between-child: which wave-1 baseline skills go with more subsequent word-reading gain, mutually adjusted                                                                                                        |
| `lrp-rli-lcsm-067` | `lcsm`          | Within-child latent change-score: prior-wave letter sounds `L` and vocabulary `E` as predictors of reading _change_                                                                                             |
| `lrp-rli-dose-077` | `dose_response` | Period-resolved observational dose-response of intervention sessions → word reading; `lrp-rli-dose-177` adds an ability-adjusted sensitivity, `lrp-rli-dose-277` is the pooled (no-period-variation) comparator |

`lrp-rli-dose-077`'s dose terms are observational (sessions = a DAG collider as exposure): an adjusted
within-child association, never "more sessions cause more gain". Only the randomised ITT/DiD
contrasts carry the causal claim.

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
is an off-DAG ability proxy (locked DAG) and the child random intercept only _partially_
adjusts. Descriptive natural-history, `n≈54` (wide intervals). Byrne-cohort replication is a
gated follow-up (unconfirmed `bpvs`/`basmat` ceilings; `basmat` is wave-3+, so no baseline).

---

## Conventions and pointers

- **Fit a model:** `python scripts/fit_statistical_model.py {model_id|all} --config dev|test|reporting [--render]` (Layer 2); `python scripts/fit_model.py {model_id|all} --config dev [--render]` (Layer 1).
- **Reports:** one per model at `docs/models/{model_id}/index.qmd`; thin templates that include shared partials from `docs/models/_partials/`. Selection variants fall back to their parent's template.
- **Cross-model comparisons:** `scripts/compare_statistical_models.py` (ITT-vs-joint `τ` consistency, `τ` and mechanism-slope forests, nested PSIS-LOO).
- **Interpreting results:** check convergence (R-hat ≈ 1.00, ESS, ≤ 1 % divergences) before interpreting; report the posterior (median + equal-tailed 95 % credible interval + tail probability, with an HPDI sensitivity interval alongside); positive `τ` = intervention helps; only `τ` is causal. Full guidance in `METHODS.md`.
- **Source of truth:** Layer-1 ids/outcomes live in each module + `models/registry.py`; Layer-2 in each module's `SPEC` (`statistical_models/`). Keep this inventory in step with those when models are added, renamed, or retired.
