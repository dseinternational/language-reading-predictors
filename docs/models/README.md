<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).
>
> Substantially edited in the ITT, concurrent, longitudinal-factor and waitlist-crossover sections by a LLM-based AI tool (Codex/GPT-5).
>
> Substantially edited in the report-order documentation by a LLM-based AI tool (Codex/GPT-5).
>
> Divergent-transition qualification policy updated by a LLM-based AI tool (Codex/GPT-5).
>
> Phoneme-blending link-sensitivity documentation updated by a LLM-based AI tool (Codex/GPT-5).
>
> Participant-flow wording updated by a LLM-based AI tool (Codex/GPT-5).
>
> Available-case modified ITT terminology updated by a LLM-based AI tool (Codex/GPT-5).
>
> Byrne/RLM lagged-DAG status updated by a LLM-based AI tool (Codex/GPT-5).
>
> Byrne/RLM confirmed-input BPVS gain models updated by a LLM-based AI tool (Codex/GPT-5).
>
> Byrne/RLM confirmed-input grammar and memory gain models updated by a LLM-based AI tool (Codex/GPT-5).

# Model inventory

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
   the DAG supports it, a causal effect. Most bounded-score families use a Beta-Binomial
   working likelihood via a logit linear predictor: this respects score bounds and
   overdispersion but is not a literal claim that heterogeneous test items or stopping-rule
   scores are exchangeable Bernoulli trials.

Both layers are built against the **revised causal DAG**
(`dag/dag-language-reading.dagitty`, revised 2026-07-10). The single most important reading
rule across the whole collection: **only a contrast licensed by randomisation can be causal.** In Layer 2
that is `tau` in the randomised-window ITT family, `tau_t2` in the arm-by-wave crossover family and `beta_trt` in the gain-factor family, subject to each analysis's stated available-case missingness assumption. Every
skill→skill coupling, mechanism slope, mediator→outcome path, and dose–response is a
latent-ability-confounded **adjusted association**, never "X drives Y". Positive `τ` =
intervention benefit (`G = 2 − group`).

## At a glance

| Layer | Family (id prefix)                                            | Purpose                                                                                                                          |
| ----- | ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| 1     | Gradient-boosting discovery (`lrp-rli-gbg` / `lrp-rli-gbl`)   | Rank predictors of each outcome's gain and level                                                                                 |
| 2     | ITT suite (`lrp-rli-itt`) + joint (`lrp-rli-itt-012`)         | Available-case modified ITT estimate (+ joint graph, SES, ability/site robustness, generalisation)                               |
| 2     | Gain factors (`lrp-rli-gf`)                                   | DAG-focused ANCOVA: randomised effect + adjusted associations on each outcome's gain                                             |
| 2     | Level factors (`lrp-rli-lf`)                                  | Companion levels view: group×time and ability×time per timepoint                                                                 |
| 2     | Waitlist-crossover arm-by-wave (`lrp-rli-did`)                | Randomised t2 arm gap plus separate baseline/post-crossover gaps; observational dose and exploratory catch-up heterogeneity      |
| 2     | Aligned per-protocol (`lrp-rli-al`)                           | Onset-aligned single 40-week gain per child (associational)                                                                      |
| 2     | Mechanism (`lrp-rli-mech`)                                    | Adjusted dose-response of one skill on another                                                                                   |
| 2     | Joint bivariate mechanism (`lrp-rli-jm`)                      | One exposure → two outcomes jointly (LKJ cross-outcome dependence block); identified decoding-specificity contrast               |
| 2     | Mediation (`lrp-rli-med`; natural + interventional g-formula) | How much of an intervention-outcome contrast runs through a given skill                                                          |
| 2     | Predictor / dynamics (`lrp-rli-adj`, `lcsm`, `dose`)          | Baseline predictors, within-child change, lagged reverse couplings, change-on-change, and dose–response of word reading          |
| 2     | Horseshoe ranking cross-check (`lrp-rli-hs`)                  | Regularised-horseshoe predictor ranking vs the gradient-boosting layer                                                           |
| 2     | Correlated-factor measurement model (`lrp-rli-mm`)            | Correlated domain-factor measurement model of the skills                                                                         |
| 2     | Growth curves (`lrp-rli-gc`)                                  | Joint verbal/reading trajectories + whether baseline non-verbal ability predicts trajectory shape                                |
| 2     | Block-2 exposure (`lrp-rli-bx`)                               | Staggered block-2 exposure contrasts, reported as associations requiring a parallel-trends assumption                            |
| 2     | Floor-sitter survival (`lrp-rli-surv`)                        | Discrete-time hazard for _when_ a floored child (P / N) first comes off the floor                                                |
| 2     | Concurrent associations (`lrp-rli-ca`)                        | Per-wave mutually-adjusted associations between contemporaneous skill levels and the focal outcome                               |
| 2     | Longitudinal correlated-factor model (`lrp-rli-lcf`)          | Per-wave latent-domain correlations and directional conditional slopes from a longitudinal measurement model                     |
| 2     | Wave-pooled level association (`lrp-rli-pl`)                  | One skill-to-skill level association pooled over all four waves, split into between-child and within-child parts (associational) |
| 2     | Historical growth, Byrne cohort (`lrp-rlm-hg`)                | Descriptive group-by-wave natural-history growth per measure in the Byrne reading-language-memory study (`study_id="rlm"`)       |
| 2     | Byrne Phase B/D (`lrp-rlm-jc/mm/adj/hs/ca`)                   | Joint trajectories, measurement, baseline-predictor and confirmed-measure concurrent views                                       |

Layer-2 totals are generated from the code rather than maintained in prose: `definitions.MODEL_REGISTRY` is the RLI catalogue, while module auto-discovery adds the Byrne `lrp-rlm-*` models. The current checked snapshot is `docs/models/registry-counts.json`; CI runs `python scripts/check_statistical_documentation.py`, which fails if that snapshot differs from `definitions.KINDS`, `definitions.MODEL_REGISTRY`, or `registry.discover_models()`. Regenerate it with the same command plus `--write` after an intentional registry change. Layer-2 selection variants (`…b` / `…base` / `…d`) are included in the per-family tables below.

## Outcome symbols (Layer 2)

Layer-2 models refer to outcomes by short symbols; the bounded count maximum (`n`) is the
Beta-Binomial trial ceiling.

| Symbol      | Measure                                                | `n` | Notes                                                     |
| ----------- | ------------------------------------------------------ | --: | --------------------------------------------------------- |
| `W`         | Word reading (EWRSWR)                                  |  79 | Headline primary in this reanalysis                       |
| `R`         | Receptive vocabulary (ROWPVT)                          | 170 | Standardised (transfer) measure                           |
| `E`         | Expressive vocabulary (EOWPVT)                         | 170 | Standardised (transfer) measure                           |
| `L`         | Letter-sound knowledge (YARC-LSK)                      |  32 | Direct teaching target                                    |
| `P`         | Phonetic spelling (SPPHON)                             |  92 | Heavily floored (~78 % at zero at t1)                     |
| `B`         | Phoneme blending                                       |  10 | Direct teaching target                                    |
| `F`         | Basic concept knowledge (CELF)                         |  18 |                                                           |
| `T`         | Receptive grammar (TROG-2)                             |  32 |                                                           |
| `N`         | Nonword reading                                        |   6 | Heavily floored; t1 is missing for four archived children |
| `TR` / `TE` | Taught receptive / expressive vocabulary (block 1)     |   — | Curated word set taught by RLI                            |
| `UR` / `UE` | Not-taught receptive / expressive vocabulary (block 1) |   — | Generalisation comparators                                |

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

Converted families use immutable typed settings, resolved and validated before data loading as one run plan — `itt`, `gain_factors`, `level_factors`, `did`, `concurrent`, `aligned` and `growth` so far. The pipeline records the full plan in `config.json` and writes `model_recipe.md`, a plain-language description that students can read beside the fitted report; the recipe and executable paths are generated from the same plan so they cannot quietly disagree. Where a family drops a declared term at fit time — a treated-only gain-factor fit drops its treatment interactions, the indicator being constant — the plan records the declared and the fitted set separately, so the recipe never names a coefficient the posterior lacks.

One module per model, each defining a `SPEC = ModelSpec(...)` and a `fit(config)`. Eight
factory/pipeline families keyed by `ModelSpec.kind`. Shared priors, HSGP helpers, the
g-formula, and the floor rule live in the package; each fit writes `trace.nc`,
`diagnostics_summary.json` (the convergence gate), per-family CSVs, and diagnostic plots
to `output/statistical_models/models/{model_id}-{config}/`.

### ITT suite — `lrp-rli-itt-001–lrp-rli-itt-030` plus registered companions (`kind="itt"` / `"joint"`)

**Purpose.** The headline randomised layer estimates `τ`, the assigned-arm contrast during t1→t2. Randomisation identifies the full randomised-sample arm contrast before missing outcomes—the own baseline and linear age enter as _precision_ terms, not as an identification set—and attendance/dose is never conditioned on. Of 57 children randomised (29 immediate intervention, 28 waiting control), the published CONSORT diagram records three losses to follow-up: one intervention child who moved school and two waiting-control children, one who moved school and one whose reason is recorded as “refused to participate in testing, school withdrawn”. This left 54 children analysed (28 and 26), who are represented in the repository. Two additional children in each arm discontinued the intervention after moving school but were followed and retained in their assigned groups. Each model then applies outcome- and covariate-observation requirements, giving the sequence `57 randomised → 3 lost to follow-up → 54 analysed and available → model-specific fitted sample` (commonly 54, 53 where a t2 score is unavailable, and smaller in the floor subgroups). The suite is therefore an **available-case modified ITT**, not a full ITT of all randomised children: it handles observed non-adherence by assigned group, but it does not recover the three missing follow-up outcomes. A causal reading even among the fitted children requires that loss to follow-up and any further observed-data restriction do not induce an arm–potential-outcome association; extending that contrast to all 57 randomised children additionally requires a defensible missingness or transportability assumption. Every report must state fitted denominators and exclusions by arm.

**Outcome hierarchy and floor rule.** The published 2012 trial (DOI [10.1111/j.1469-7610.2012.02557.x](https://doi.org/10.1111/j.1469-7610.2012.02557.x)) described four primary outcomes: `W`, `L`, `B` and `TE`. This project designates `W` as the single headline primary for the current reanalysis; that is a transparent reanalysis hierarchy, not the original trial hierarchy. The floor branch for `P` and `N` uses an arm-blind threshold based on the observed t2 zero prevalence. It reports the resulting `Pr(post > 0 | observed pre = 0)` risk difference as an exploratory headline, because the rule and 40 % threshold were adopted after inspecting this trial's outcome distribution. It is therefore a **post-hoc, data-adaptive exploratory estimand**, not a prospectively pre-specified trial primary. Because observed baseline-floor status is pre-randomisation, the subgroup contrast retains randomised causal logic among children with observed floor status and post-score, subject to the same missingness assumption. The graded analyses remain flagged, detection-limited secondaries. Design notes: `notes/202606251321-lrpitt-suite-design.md`, `notes/202606251124-lrpitt-floored-outcomes-nonword-spelling.md`.

**Joint-model scope and contrasts.** The registered parent joint specifications set residual correlation off. With independent outcome-specific priors and likelihoods, they are factorised collections of marginal outcome models in one PyMC graph; they do not learn within-child residual covariance, so posterior differences between outcome effects omit that covariance. The current reports lead with contrasts between per-draw probability-scale average marginal effects, a common proportion-correct scale; raw `tau_i - tau_j` conditional-logit contrasts are supplementary. Neither interval preserves within-child covariance under the factorised model, so those contrasts are exploratory sensitivity results pending a dependence-model analysis. That analysis is now registered (#551): `lrp-rli-itt-215`, `-315` and `-216` are the dependence-aware companions of `015`, `115` and `016` — the same two-outcome fits with the per-child LKJ residual-correlation block on (`use_residual_correlation=True`, `joint_structure="residual_correlated"`), publishing each contrast as a posterior difference that carries the estimated within-child covariance, with `u_corr` / `sigma_outcome` reported against their priors so weak identification is stated where present. The point estimates should agree with the parents; the intervals and P(> 0) may move. A companion that fails the house gate withholds its result, and the follow-up is then a paired child-level randomisation-inference/permutation or bootstrap analysis outside the pipeline. The ten-outcome `lrp-rli-itt-012` is deliberately left factorised: its 10 × 10 block was prior-dominated at n = 53 in April 2026. In the taught-versus-not-taught models, a positive contrast establishes only that the taught effect is larger; limited transfer additionally requires the marginal not-taught effect to be small against a substantively defined negligible-effect threshold. `lrp-rli-itt-012` covers the ten baseline-bearing outcomes in the original ITT suite (`TR`, `TE`, `UR`, `UE`, `R`, `E`, `L`, `B`, `P`, `W`): post-only `N` is excluded, and `F`/`T` were later additions with single-outcome models rather than members of this joint scope.

**Artefact compatibility.** In refits produced from July 2026 onward, `tau_summary.csv` uses `prob_ame_pos` for the probability that the headline probability-scale average marginal effect is positive. `prob_tau_pos` is retained as an exact compatibility alias of that field; it no longer names the conditional logit-coefficient probability in moderated or varying-effect models. Use `prob_tau_logit_pos` for that secondary coefficient-scale quantity, and do not compare an old `prob_tau_pos` column across fit vintages without checking the generating code and `config.json`.

**Phoneme-blending response link.** `lrp-rli-itt-008` remains the primary phoneme-blending model and uses the suite's ordinary logit mean. Because its ten items each have three response alternatives, `lrp-rli-itt-108` is the mandatory registered robustness companion: it constrains the expected score to be at least the one-third guessing floor. Neither fit is release-ready on its own. Their fitted rows and run provenance must match, both traces must pass the clean convergence gate, and `scripts/blending_link_sensitivity.py` must build and validate the paired trace-backed bundle before either report's key findings are regenerated or published.

**Word-reading missing outcomes.** `lrp-rli-itt-010` remains the 53-outcome, t1-baseline model of record. Its release additionally requires a converged, trace-backed screening-baseline sub-fit supplied with the checksum-pinned 57-row UK Data Service archive. The sub-fit persists its pre-randomisation-screening-anchored prior and prior-predictive checks; reports a matched common-profile bridge over the same 53 observed outcomes; and standardises both treatment surfaces over all 57 screening profiles under conditional MAR. The no-benefit and complete item-delta grid `{-8, -4, 0, +4, +8}` instead complete the factual randomised arms, using denominators 29 and 28 and modifying only the one intervention and three control missing outcomes in their assigned arms. The zero-delta cell is the factual-arm MAR anchor; the intervention non-starter no-benefit row is a mean-surface restriction, not classical distributional reference-based imputation. The grid is a broad diagnostic stress test, not a fitted distribution over missing outcomes, and boundary clipping is reported beside model-free sharp bounds. These are secondary identifying-assumption sensitivities, not alternative primary results. The importer keeps the external archive gitignored because its ReShare item-level licence is blank, and the loader reconciles the 54 included rows to the repository across 71 fields. The local raw CSV retains upstream source identifiers; returned model data and emitted artefacts omit them, and no subject-ID crosswalk is persisted.

| Model                     | Outcome             | Purpose                                                                                                    |
| ------------------------- | ------------------- | ---------------------------------------------------------------------------------------------------------- |
| `lrp-rli-itt-001`         | `TR`                | Available-case modified ITT analysis of taught receptive vocabulary (block 1)                              |
| `lrp-rli-itt-002`         | `TE`                | Available-case modified ITT analysis of taught expressive vocabulary (block 1)                             |
| `lrp-rli-itt-003`         | `UR`                | Available-case modified ITT analysis of not-taught receptive vocabulary (block 1)                          |
| `lrp-rli-itt-004`         | `UE`                | Available-case modified ITT analysis of not-taught expressive vocabulary (block 1)                         |
| `lrp-rli-itt-005`         | `R`                 | Available-case modified ITT analysis of standardised receptive vocabulary                                  |
| `lrp-rli-itt-006`         | `E`                 | Available-case modified ITT analysis of standardised expressive vocabulary                                 |
| `lrp-rli-itt-007`         | `L`                 | Available-case modified ITT analysis of letter-sound knowledge                                             |
| `lrp-rli-itt-008`         | `B`                 | Available-case modified ITT analysis of phoneme blending                                                   |
| `lrp-rli-itt-108`         | `B`                 | Mandatory one-third guessing-floor response-link companion to `lrp-rli-itt-008`                            |
| `lrp-rli-itt-009`         | `P`                 | Available-case modified ITT analysis of phonetic spelling — floor-rule branch                              |
| `lrp-rli-itt-010`         | `W`                 | **Available-case modified ITT analysis of word reading** (headline primary; supersedes LRP52)              |
| `lrp-rli-itt-011`         | `N`                 | Available-case modified ITT analysis of nonword reading — floor-rule branch                                |
| `lrp-rli-itt-012`         | joint               | Factorised joint available-case modified ITT graph over ten baseline-bearing outcomes                      |
| `lrp-rli-itt-013` / `13b` | `W` / `L`           | SES-adjusted available-case modified ITT analyses                                                          |
| `lrp-rli-itt-014` / `14b` | `W` / `L`           | Unadjusted available-case modified ITT analyses on the matched SES complete-case subsets                   |
| `lrp-rli-itt-015` / `15b` | contrast            | Available-case modified ITT generalisation contrasts, expressive (`15`) and receptive (`15b`)              |
| `lrp-rli-itt-016`         | contrast            | Available-case modified ITT modality contrast: taught expressive versus receptive vocabulary               |
| `lrp-rli-itt-215` / `315` | contrast            | Dependence-aware companions of `015` / `115` (#551): the same contrasts with the LKJ residual block on     |
| `lrp-rli-itt-216`         | contrast            | Dependence-aware companion of `016` (#551): the modality contrast with the LKJ residual block on           |
| `lrp-rli-itt-017–020`     | `TR`,`TE`,`UR`,`UE` | Ability-adjusted available-case modified ITT robustness across the vocabulary family                       |
| `lrp-rli-itt-021` / `22`  | `R` / `E`           | Ability-adjusted available-case modified ITT robustness, standardised vocabulary                           |
| `lrp-rli-itt-023` / `24`  | `L` / `W`           | Ability-adjusted available-case modified ITT robustness, letter sounds and word reading                    |
| `lrp-rli-itt-025`         | `F`                 | Available-case modified ITT analysis of basic concepts (δ = 1 item under the ½-natural-maturation rule)    |
| `lrp-rli-itt-026`         | `T`                 | Available-case modified ITT analysis of receptive grammar (δ = 1 item under the ½-natural-maturation rule) |
| `lrp-rli-itt-027` / `28`  | `W` / `L`           | Site-adjusted available-case modified ITT robustness; `area` is complete                                   |
| `lrp-rli-itt-029`         | `EI`                | Available-case modified ITT analysis of APT expressive information (doubled half-mark scale, /80)          |
| `lrp-rli-itt-129`         | `EI40`              | Denominator-sensitivity comparator for `029`: the same score rounded to whole marks (/40)                  |
| `lrp-rli-itt-030`         | `EG`                | Available-case modified ITT analysis of APT expressive grammar (/37)                                       |

### Gain factors — `lrp-rli-gf-001–lrp-rli-gf-013` (+ `…b`, `…m`) (`kind="gain_factors"`)

**Purpose.** A DAG-focused ANCOVA on each outcome's period gain (post-score given its own
pre-score), stacking every on-intervention and untreated period with a child random
intercept — a partial, shrunken stand-in for between-child heterogeneity, **not** a
control for latent ability. The randomised on-intervention term is the **only** causal
coefficient, and its probability/items-scale marginal effect is averaged over the
**period-1** (randomised) transition only; own baseline, age, cognitive ability (block
design), the upstream DAG skill baselines (`skill_symbols`), the revised-DAG non-measure
confounders hearing/speech/phonological memory (`adjust_for`), and the `age × ability`
precision interaction are explicit _adjusted associations_. Adjustment sets were
re-derived against the revised DAG in #247. The causal headline is **interaction-free in
`trt`** (#391 finding 3 decision, 2026-07-22): the pre-specified `trt × ability` /
`trt × own` moderation questions live in the explicitly associational `…m` moderation
variants (`lrp-rli-gf-201`–`211`, one per outcome, anchored to the per-outcome primary),
whose interaction-aware marginal is model-dependent — partly informed by post-crossover
data — and never released as causal. On the off-floor fits the own baseline is the
**binary off-floor-at-pre indicator** (`gamma_own_offfloor`, #391 finding 2 decision):
the graded pre logit of a heavily-floored measure is a near-degenerate spike, so the
indicator is the honest functional form, in the main effect and any variant interaction
alike. The `…b` variant is treated-only (gains while on intervention). Design note:
`notes/202606261230-gain-level-factors-design.md`; re-derivation:
`notes/202607122200-gf-lf-revised-dag-adjustments.md`.

**Naming note.** "Factors" here (and in the level-factors family below) carries its plain-English sense — the observed covariates _associated with_ gains or levels — not the factor-analysis sense: these are regression models with no latent variables. The latent measurement model is `lrp-rli-mm-001` (`kind="corr_factor"`).

| Model            | Outcome | Skill baselines (`skill_symbols`)         | Confounders (`adjust_for`) | Treated-only `…b` | Moderation `…m`  |
| ---------------- | ------- | ----------------------------------------- | -------------------------- | ----------------- | ---------------- |
| `lrp-rli-gf-001` | `W`     | `TR`, `TE`, `R`, `E`, `L`, `N`, `B`       | —                          | `lrp-rli-gf-101`  | `lrp-rli-gf-201` |
| `lrp-rli-gf-002` | `R`     | `TR`                                      | `HS`, `RW`                 | `lrp-rli-gf-102`  | `lrp-rli-gf-202` |
| `lrp-rli-gf-003` | `E`     | `R`, `TR`, `TE`                           | `HS`, `SP`, `RW`           | `lrp-rli-gf-103`  | `lrp-rli-gf-203` |
| `lrp-rli-gf-004` | `L`     | —                                         | `HS`, `SP`                 | `lrp-rli-gf-104`  | `lrp-rli-gf-204` |
| `lrp-rli-gf-005` | `P`     | `L`, `B` (off-floor Bernoulli likelihood) | `RW`                       | `lrp-rli-gf-105`  | `lrp-rli-gf-205` |
| `lrp-rli-gf-006` | `B`     | `L`, `E`, `TE`                            | `HS`, `SP`, `RW`           | `lrp-rli-gf-106`  | `lrp-rli-gf-206` |
| `lrp-rli-gf-007` | `F`     | `R`, `TR`                                 | —                          | `lrp-rli-gf-107`  | `lrp-rli-gf-207` |
| `lrp-rli-gf-008` | `T`     | `R`, `TR`                                 | —                          | `lrp-rli-gf-108`  | `lrp-rli-gf-208` |
| `lrp-rli-gf-009` | `TR`    | —                                         | `HS`, `RW`                 | —                 | `lrp-rli-gf-209` |
| `lrp-rli-gf-010` | `TE`    | `TR`                                      | `HS`, `SP`, `RW`           | —                 | `lrp-rli-gf-210` |
| `lrp-rli-gf-011` | `N`     | `L`, `B` (off-floor Bernoulli likelihood) | `SP`, `RW`                 | —                 | `lrp-rli-gf-211` |
| `lrp-rli-gf-012` | `TR`    | `R`, `E`                                  | `HS`, `RW`                 | —                 | —                |
| `lrp-rli-gf-013` | `TE`    | `TR`, `R`, `E`                            | `HS`, `SP`, `RW`           | —                 | —                |

> `gf-012` and `gf-013` (#421) extend `gf-009`/`gf-010` by entering broad receptive/expressive vocabulary as **downstream descriptive associations** (the review's RV/EV → taught-vocabulary finding), _not_ DAG-parent baselines; as everywhere in this family, only the randomised on-intervention term is causal. Their moderation questions are carried by the per-outcome variants `gf-209`/`gf-210`, so they take no `…m` of their own.

### Level factors — `lrp-rli-lf-001–lrp-rli-lf-011` (`kind="level_factors"`)

**Purpose.** The companion _levels_ view of each outcome (the score at each timepoint, no
own baseline), with group×time and ability×time as per-timepoint coefficient vectors. The
arm-by-time vector is centred on the timepoint-1 arm gap (#552): `arm_gap_t1` is the
covariate-adjusted pre-randomisation balance quantity (reported, never an effect) and
`d_grp_time[t]` the change in that gap at each later wave, with the per-wave levels view
`b_grp_time[t]` kept as a derived quantity. Only the t2 change `d_grp_time[t2]` — a
difference-in-differences of adjusted levels — is a clean randomised effect; later
timepoints are post-crossover and flagged as associations. `arm_gap_reference="free"`
retains the pre-#552 free per-timepoint vector (focal `b_grp_time[1]`) as an explicit
comparator. Each outcome carries the same revised-DAG exogenous confounders
(`adjust_for`: hearing/speech/phonological memory) as its gain-factor sibling, but **no**
measure-skill adjusters — in a levels model a skill's contemporaneous level is a
post-treatment mediator of the group×time effect (#247). Outcomes mirror the gain-factor
family: `lrp-rli-lf-001` `W`, `02` `R`, `03` `E`, `04` `L`, `05` `P` (off-floor), `06` `B`,
`07` `F`, `08` `T`, `09` `TR`, `10` `TE`, `11` `N` (off-floor).
The gain-factors naming note applies here too: "factors" means observed regression covariates, not latent factors.

**Phoneme-blending response link (#584 decision 2).** Blending is scored from ten three-alternative forced-choice items, so an expected score cannot fall below about 3.3/10, but the ordinary inverse-logit mean permits it — and the `lrp-rli-lf-006` posterior uses that room (13.7% of row-by-draw expected proportions below one third; 16.0% at t2). `lrp-rli-lf-106` is the registered companion fitting the same data with the guessing-floor score mean $\mu = \tfrac{1}{3} + \tfrac{2}{3}\operatorname{expit}(\eta)$, with the empirical-Bayes intercept anchor mapped back through the link. The pair mirrors `lrp-rli-itt-008` / `lrp-rli-itt-108`: **neither fit releases without the other**, and the release gate enforces it.

**Randomised-window comparators (#584 decision 3).** `lrp-rli-lf-201`–`211` are the eleven primaries refitted on the **t1/t2 window alone**. In the four-wave model of record the post-crossover likelihood reaches the reported t2 change through parameters the waves share — the balance term `arm_gap_t1`, the child intercept, the dispersion and the time-invariant group×ability term — and the posterior correlation between `arm_gap_t1` and `d_grp_time[t2]` runs from −0.07 to −0.44 across the suite. The comparator removes that path entirely, so the difference between the two fits is the size of the longitudinal working model's contribution. The four-wave fit remains the model of record (showing all four waves on one scale is its purpose); the comparator is reported beside it, and is **not** a gate — a missing comparator leaves the question open rather than withholding the estimate. The blending comparator `lrp-rli-lf-206` carries the ordinary link only, so it answers the window question and not the response-link one; that headline stays with the four-wave link pair.

**Nuisance priors (#584 decision 4).** Two scales differ from the shared defaults, both calibrated in `notes/202608231930-level-factors-nuisance-prior-calibration.md`. The dispersion prior sits on `1/sqrt(kappa)` rather than on the concentration, so the near-Binomial limit — no extra-Binomial dispersion beyond the child random intercept — is reachable; `kappa` remains the reported Deterministic. The scale (0.25, the RLM constant, confirmed rather than re-derived) is calibration-preserving: it reproduces the old prior's median variance inflation at every level denominator to within 3%, while raising the near-Binomial region's prior mass from ~0 to 8–33%. `sigma_child` widens to `HalfNormal(1.0)`: a levels model has no own-baseline term, so the child intercept carries the whole between-child spread in level, and at the shared 0.5 scale the median asserted a middle-95% child range of 0.18–0.45 on a mid-difficulty measure. Registered sweep axes `--axis kappa` and `--axis sigma_child` measure both, each including the pre-decision scale.

### Waitlist-crossover arm-by-wave sensitivity — `lrp-rli-did-001–lrp-rli-did-015` (+ `lrp-rli-did-101`, `lrp-rli-did-102`, `lrp-rli-did-107`) (`kind="did"`)

**Purpose.** A longitudinal sensitivity analysis alongside the available-case modified ITT estimates. The binary-treatment models jointly model bounded t1/t2/t3 levels with a separate immediate-minus-waitlist gap at each wave: `arm_gap_t1` checks baseline balance, `tau_t2` is the randomised causal contrast, `arm_gap_t3` is a post-crossover association and `delta_crossover = tau_t2 - arm_gap_t3` describes closure of the arm gap rather than a second treatment effect. A child random intercept partially pools stable between-child differences but does not make every child a fixed-effect control. No model conditions on each period's start outcome: t2 is already treatment-affected for the immediate arm when used as the P2 baseline. The heavily floored outcomes (`P`, `N`) use a Bernoulli on wave-specific off-floor status, so their contrasts concern off-floor **prevalence**, not coming off the floor. Dose variants retain the P1/P2 transition frame, adjust for randomised arm, current treatment, t1 outcome and t1 age, and estimate observational treated-centred session-dose associations. The current design decision is `notes/202607151800-did-arm-wave-redesign.md`; it supersedes the historical restricted-model decision in `notes/202606260702-did-crossover-design.md`.

| Model             | Outcome | Purpose                                                                                                                                                        |
| ----------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lrp-rli-did-001` | `W`     | Arm-by-wave word-reading sensitivity; randomised t2 contrast plus post-crossover contrasts                                                                     |
| `lrp-rli-did-002` | `L`     | Arm-by-wave letter-sound sensitivity; randomised t2 contrast plus post-crossover contrasts                                                                     |
| `lrp-rli-did-003` | `B`     | Arm-by-wave phoneme-blending sensitivity; randomised t2 contrast plus post-crossover contrasts                                                                 |
| `lrp-rli-did-004` | `TE`    | Arm-by-wave taught-expressive-vocabulary sensitivity; randomised t2 contrast plus post-crossover contrasts                                                     |
| `lrp-rli-did-005` | `R`     | Arm-by-wave receptive-vocabulary sensitivity; randomised t2 contrast plus post-crossover contrasts                                                             |
| `lrp-rli-did-006` | `W`     | P1/P2 transition model with separate treatment-presence and pooled observational session-dose terms                                                            |
| `lrp-rli-did-007` | `L`     | P1/P2 transition model with observational period-resolved session-dose slopes; `lrp-rli-did-107` is its pooled comparator                                      |
| `lrp-rli-did-008` | `TR`    | Arm-by-wave taught-receptive-vocabulary sensitivity; randomised t2 contrast plus post-crossover contrasts                                                      |
| `lrp-rli-did-009` | `E`     | Arm-by-wave standardised-expressive-vocabulary sensitivity; randomised t2 contrast plus post-crossover contrasts                                               |
| `lrp-rli-did-010` | `F`     | Arm-by-wave basic-concepts sensitivity; randomised t2 contrast plus post-crossover contrasts                                                                   |
| `lrp-rli-did-011` | `P`     | Arm-by-wave phonetic-spelling sensitivity on period-end off-floor prevalence                                                                                   |
| `lrp-rli-did-012` | `N`     | Arm-by-wave nonword-reading sensitivity on period-end off-floor prevalence                                                                                     |
| `lrp-rli-did-013` | `W`     | Exploratory waitlist-t3 catch-up heterogeneity; the variance component conflates response, maturation, history and noise                                       |
| `lrp-rli-did-014` | `EI`    | Arm-by-wave APT expressive-information sensitivity (doubled half-mark scale); randomised t2 contrast plus post-crossover contrasts                             |
| `lrp-rli-did-015` | `EG`    | Arm-by-wave APT expressive-grammar sensitivity; randomised t2 contrast plus post-crossover contrasts                                                           |
| `lrp-rli-did-101` | `W`     | Independent-prior intercept sensitivity for `lrp-rli-did-001`: the empirical-Bayes t1 anchor replaced with a free zero-centred intercept (#390 P1 condition 1) |
| `lrp-rli-did-102` | `L`     | Wide-`tau_t2` prior sensitivity for `lrp-rli-did-002`: the causal contrast's prior widened from the tier Normal(0, 0.5) to Normal(0, 1) (#382 rec 3)           |

### Aligned per-protocol — `lrp-rli-al-001–lrp-rli-al-008` (+ `lrp-rli-al-101`) (`kind="aligned"`)

**Purpose.** An onset-aligned, per-protocol single gain: both arms aligned by intervention
onset (immediate t1→t3, waitlist t2→t4) into one cross-sectional Beta-Binomial ANCOVA per
child. The cohort contrast is **not** randomised (confounded by age-at-onset and timing),
so _no_ term is causal — every coefficient is an association. Design note:
`notes/202606261343-lrpal-aligned-design.md`. Outcomes: `lrp-rli-al-001` `W`, `02` `R`, `03` `E`,
`04` `L`, `05` `P` (off-floor: a Bernoulli on the off-floor indicator whose own-baseline term
is the binary off-floor-at-onset indicator — the #391 floor rule, adopted here by the
2026-08-21 aligned review), `06` `B`, `07` `F`, `08` `T`; **`lrp-rli-al-101`** adds a
cumulative-session dose sensitivity term (a collider — sensitivity only).

### Mechanism — `lrp-rli-mech-056–lrp-rli-mech-058`, `lrp-rli-mech-071–lrp-rli-mech-073`, `lrp-rli-mech-088–lrp-rli-mech-090`, `lrp-rli-mech-102–lrp-rli-mech-104` (`kind="mechanism"`)

**Purpose.** The adjustment-set dose-response of one measured skill on another across all
phases, with subject random intercepts and optional linear moderation. Every slope is an
**adjusted association** (latent-ability confounded), not a causal effect.

| Model                         | Path              | Purpose                                                                                                                                                  |
| ----------------------------- | ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lrp-rli-mech-056`            | `R → W`           | Receptive vocabulary → word reading                                                                                                                      |
| `lrp-rli-mech-057`            | `E → W`           | Expressive vocabulary → word reading                                                                                                                     |
| `lrp-rli-mech-058`            | `L → W`           | Letter-sound knowledge → word reading                                                                                                                    |
| `lrp-rli-mech-071`            | `L → W`           | Letter sounds → word reading, linear moderation by expressive vocabulary `E`                                                                             |
| `lrp-rli-mech-072` / `72base` | `L → N`           | Code-based route: letter sounds moderated by blending `B` → decoding (with / without the interaction)                                                    |
| `lrp-rli-mech-073` / `73base` | `L → W`           | Letter sounds → word reading, moderated by age (with / without the interaction)                                                                          |
| `lrp-rli-mech-061` / `161`    | `L → W`           | Joint readiness: letter sounds moderated by phoneme blending `B` → word reading (with / without the interaction; #404)                                   |
| `lrp-rli-mech-063` / `163`    | `L → W`           | Joint readiness: letter sounds moderated by nonword decoding `N` → word reading (with / without the interaction; #404)                                   |
| `lrp-rli-mech-088`            | `TR → W`          | Taught receptive vocabulary → word reading (#311; linear, IS backdoor flagged not adjusted)                                                              |
| `lrp-rli-mech-089`            | `TE → W`          | Taught expressive vocabulary → word reading (#311; linear, TR measure confounder, IS flagged)                                                            |
| `lrp-rli-mech-090`            | `RW → W`          | Phonological memory (word/nonword repetition) → word reading (#311; covariate exposure, adjust `HS` only, no IS backdoor)                                |
| `lrp-rli-mech-102`            | `RW → N`          | Phonological memory → nonword decoding (#421; alphabetic-route counterpart of mech-090; covariate exposure, adjust `HS`, linear/floored outcome)         |
| `lrp-rli-mech-103`            | `SP → N`          | Speech production → nonword decoding (#421; first SP-exposure mechanism; covariate exposure, adjust `HS`, linear/floored outcome)                        |
| `lrp-rli-mech-104` / `204`    | `L → W`           | Letter sounds → word reading, moderated by phonological memory `RW` (with / without the interaction; #421 Tier 2)                                        |
| `lrp-rli-mech-096` / `101`    | `L → N` / `L → W` | Tier-1 decoding-specificity positive controls (linear); their difference is the pre-specified nonword-minus-word contrast                                |
| `lrp-rli-mech-097–100`        | `L → R/E/T/F`     | Tier-1 negative controls: letter sounds → receptive / expressive vocabulary, grammar, basic concepts (linear)                                            |
| `lrp-rli-mech-196–201`        | as `096–101`      | Ability-adjusted mirror of the Tier-1 panel: identical rows and terms plus the t1 block-design proxy via the typed `ability_covariate` setting           |
| `lrp-rli-mech-258`            | `L → W`           | Ability-adjusted counterpart of the `058` HSGP curve, so shape can be compared curve against curve (`compare_statistical_models.py` writes the overlay)  |
| `lrp-rli-mech-301`            | `L → W`           | Between/within (Mundlak) split of the `101` slope, so the between-child and within-child associations are reported separately rather than blended (#603) |
| `lrp-rli-mech-302` / `303`    | `L → W` / `L → R` | Phase-stability sensitivity: partially-pooled per-period slopes against the pooled `101` / `097` comparators, with a nested PSIS-LOO test (#604)         |
| `lrp-rli-mech-304` / `305`    | `L → W` / `L → R` | Dispersion prior sensitivity: `1/sqrt(kappa) ~ HalfNormal(0.25)` in place of the shared `kappa ~ HalfNormal(50)`, at n = 79 and n = 170 (#605)           |

The table above is a selection; the family currently has 46 registered models (the curve tests `156–158` / `188–191`, the joint-readiness comparators `093–095` and the remaining companions are catalogued in `definitions.MODEL_REGISTRY` and the family findings note).

**One declared natural-scale estimand (#602).** Every mechanism fit reports the same headline: the predicted outcome difference between the **25th and 75th percentile of the fitted exposure**, standardised over the fitted rows — each row keeping its own period, covariates, baseline and its child's own fitted random intercept, with only the exposure moved. `mechanism_summary.csv` carries that row first and the full-observed-range contrast second as an explicitly labelled secondary, each with a machine-readable `estimand`; the items-scale curve plots the same standardised quantity, so the worked-example points lie on it. The steepest-interval table publishes both the latent-logit interval and, under the same reference population, the expected-items one.

### Mediation — `lrp-rli-med` (`kind="mediation"` / `"mediation_multi"`)

**Purpose.** g-formula decomposition of how much of an intervention-outcome contrast runs through a given skill. The natural-effect models report NDE/NIE; MED-078/186/187 report the interventional IDE/IIE analogues. Neither class is point-identified under the revised DAG because latent general ability confounds mediator→outcome paths; the interventional class additionally removes the recanting-witness cross-world obstacle, but does not turn the decomposition into a causal route.

| Model             | Purpose                                                                                                                                                                                             |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lrp-rli-med-059` | Single-mediator: word-reading gain via letter-sound knowledge `L`                                                                                                                                   |
| `lrp-rli-med-062` | Reading-route decomposition: code-based-route (`L` + blending `B`) vs lexical share                                                                                                                 |
| `lrp-rli-med-064` | Two-mediator split: `L` vs expressive vocabulary `E` (joint indirect + path-specific `NIE_L`/`NIE_E`)                                                                                               |
| `lrp-rli-med-066` | Two-mediator split: letter sounds `L` and phoneme blending `B` as parallel routes to word reading                                                                                                   |
| `lrp-rli-med-068` | Single-mediator: word-reading gain via taught expressive vocabulary `TE`                                                                                                                            |
| `lrp-rli-med-074` | Single-mediator: word-reading gain via nonword decoding `N` (floor-limited mediator)                                                                                                                |
| `lrp-rli-med-075` | Sequential code route: letter sounds `L` → blending `B` → word reading                                                                                                                              |
| `lrp-rli-med-060` | Sequential code route: letter sounds `L` → nonword decoding `N` → word reading, via an **off-floor** second-mediator leg (#421 Tier 3; the note's proposed `med-081` collided with live `lcsm-081`) |
| `lrp-rli-med-076` | Longitudinal-ordering companion: letter sounds at t2 → word reading at t4                                                                                                                           |
| `lrp-rli-med-078` | Interventional companion to MED-059: IDE/IIE for word reading via letter sounds                                                                                                                     |
| `lrp-rli-med-079` | Negative-control mediator: receptive grammar calibrating residual confounding                                                                                                                       |
| `lrp-rli-med-080` | Single-mediator: word-reading gain via taught receptive vocabulary `TR`                                                                                                                             |
| `lrp-rli-med-086` | Natural-effect decomposition: nonword off-floor risk via letter sounds                                                                                                                              |
| `lrp-rli-med-087` | Natural-effect decomposition: phoneme blending via letter sounds                                                                                                                                    |
| `lrp-rli-med-092` | Period-stacked companion (#229): the `med-059` design on the gain-factor scaffold — exposure = per-period on-intervention (ignorability, not randomisation); all-period + period-1 readouts         |
| `lrp-rli-med-186` | Interventional companion to MED-086: IDE/IIE for nonword off-floor risk via letter sounds                                                                                                           |
| `lrp-rli-med-187` | Interventional companion to MED-087: IDE/IIE for phoneme blending via letter sounds                                                                                                                 |

### Predictor / within-child dynamics — `lrp-rli-adj-065`, `lrp-rli-lcsm-067/081/082/091`, `lrp-rli-dose-077` (+ variants)

**Purpose.** Complementary, explicitly **associational** views of skill progress that sit
outside the randomised families — including the time-lagged reverse-coupling suite built on
the wave-unrolled DAG (#250; design `notes/202607141030-time-lagged-model-designs.md`).

| Model              | Kind            | Purpose                                                                                                                                                                                                                             |
| ------------------ | --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lrp-rli-adj-065`  | `adjusted`      | Between-child: which wave-1 baseline skills go with more subsequent word-reading gain, mutually adjusted                                                                                                                            |
| `lrp-rli-lcsm-067` | `lcsm`          | Within-child latent change-score: prior-wave letter sounds `L` and vocabulary `E` as predictors of reading _change_                                                                                                                 |
| `lrp-rli-lcsm-081` | `lcsm`          | Lagged reverse coupling: prior word reading `W` predicting taught-vocabulary (`TE`/`TR`) change — crossover-aware arm × window intercepts + `hs`/`rw`/`sp` adjusters (the verified lagged-DAG backdoor set)                         |
| `lrp-rli-lcsm-181` | `lcsm`          | No-reverse-coupling LOO comparator for `lcsm-081` ("does the reverse edge earn its place predictively")                                                                                                                             |
| `lrp-rli-lcsm-082` | `lcsm`          | Reciprocal dominance (exploratory): blending `B` ↔ word reading `W` lagged cross-couplings with an SD-standardised dominance contrast; broadly confounded in both directions                                                        |
| `lrp-rli-lcsm-091` | `lcsm`          | Lagged change-on-change (#229 spec 2, exploratory): prior letter-sound / vocabulary _change_ (`h_L`/`h_E`) alongside prior _level_ (`g_L`/`g_E`) predicting reading change; two usable transitions, direction-agreement deliverable |
| `lrp-rli-dose-077` | `dose_response` | Period-resolved observational attendance-response of intervention sessions → word reading; `lrp-rli-dose-177` adds a baseline-ability sensitivity, `lrp-rli-dose-277` is the pooled (no-period-variation) comparator                |
| `lrp-rli-dose-083` | `dose_response` | The same attendance-response on letter sounds (`L`)                                                                                                                                                                                 |
| `lrp-rli-dose-084` | `dose_response` | The same attendance-response on phoneme blending (`B`); qualified — the required guessing-floor link companion is not yet built for this family                                                                                     |

The `dose_response` family reports the **intensive margin**: among children who were receiving the intervention, did attending more sessions go with more progress? Sessions enter centred and standardised over the on-intervention rows only, a separate `theta_treated` indicator carries the extensive margin (whether a child was being taught at all), and the exposure is split Mundlak-style into each child's study-average attendance and their within-child deviation from it — a single slope over a child random intercept returns a precision-weighted **blend** of the two, so calling it a within-child association, as this catalogue previously did, was wrong. Attendance is not randomised and the DAG has `A`, latent `GA` and `IG` all pointing into `IS`, so every dose slope is an adjusted association, never "more sessions cause more gain". The one randomised quantity in these fits is `theta_treated` read in period 1, where every immediate-arm child attended and every waiting control attended none. Otherwise only the available-case modified ITT estimates and the randomised-window DiD contrasts carry a causal interpretation under their stated selection and missing-data assumptions — in the lagged suite that is solely `lcsm-081/082`'s window-1 arm contrast (`itt_window1_contrast.csv`), reported as an available-case modified ITT consistency check.

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
hazard, baseline (t1) letter-sound knowledge, word reading and age as prognostic covariates,
and a treatment hazard contrast `tau` fitted in the **randomised first interval only**
(`treatment_window="randomised"`, 2026-08-21 survival review, finding 1): every person-period
row after the wait-list crossover is treatment-on, so the later intervals carry no arm
contrast and fit their own both-arms-treated baseline hazards (`G = 2 − group`, positive =
benefit). The legacy pooled all-interval shift — whose split from the post-crossover
baselines was prior-mediated — remains available as the explicit
`treatment_window="pooled"` comparator.

| Model              | Kind       | Outcome | Purpose                                                      |
| ------------------ | ---------- | ------- | ------------------------------------------------------------ |
| `lrp-rli-surv-009` | `survival` | `P`     | time-to-off-floor hazard, phonetic spelling (base `itt-009`) |
| `lrp-rli-surv-011` | `survival` | `N`     | time-to-off-floor hazard, nonword reading (base `itt-011`)   |

**Prognostic, not causal.** `tau` is anchored on the randomised first interval among children
at the floor at t1 and is reported as a prognostic association, not a randomised effect of
record; concurrent letter sounds are excluded as a treatment-affected mediator. Descriptive companion:
`notes/…-persistent-floor-sitters-nonword-spelling.md` + `scripts/descriptive/floor_sitters.py`.

### Concurrent conditional associations — `lrp-rli-ca` (`kind="concurrent"`)

**Purpose.** The one family that describes how contemporaneous skill levels co-occur with the focal outcome at each wave (#312, descriptive-association workstream #314). At every timepoint it fits a between-child Beta-Binomial regression of the focal outcome's _level_ on the standardised same-wave logits of the other core skills, plus age and a group nuisance term — "at wave t, among children alike on age, the other skills and the effective trait covariates, +n of a predictor is associated with +m of the outcome". The family's core skill set is {`W`, `L`, `B`, `TR`, `TE`, `R`, `E`}; each core-set model conditions its focal outcome on the remaining six, so together they describe the conditional joint distribution of the same measure set from different sides. `ca-007` (phoneme blending, `B`) completes that core set as a focal outcome; `ca-008` (basic concepts, CELF `F`) and `ca-009` (receptive grammar, TROG-2 `T`) are focal-only scope extensions (#371) that condition on the full seven-measure core set without being added to the sibling models' predictor sets, so the levels panel of the association matrix covers all non-floored outcomes. `ca-001`–`009` carry the gains-panel trait covariates (non-verbal ability, hearing, speech, phonological memory) as t1 baselines (#371), with explicit paired missingness indicators for hearing, speech and phonological memory. A flag that is constant on a wave's outcome-complete rows is omitted rather than fitted as an intercept alias; any fitted `_missing` coefficient is a nuisance subgroup offset, never a skill association. The four waves are fitted separately (one row per child per wave) and reported side by side. The wave with the largest complete-outcome sample (ties → latest) is an operational diagnostic anchor for the standard trace and plots, not a claim that it is best-powered; every adjusted and single-skill comparator has its own complete convergence metrics. The single-skill comparator retains the same effective trait covariates while omitting age, group and the other skills; legacy output fields retain the `biv_*` name. `ca-010` and `ca-011` (#421) are a separate **minimal-adjustment** pair: instead of the seven-measure mutual adjustment they regress word reading on letter sounds — and, in `ca-011`, same-wave nonword decoding — adjusting only for age, hearing and non-verbal ability, giving the letter-sound → word-reading review's headline association and its holding-decoding-fixed decomposition.

| Model            | Kind         | Outcome | Purpose                                                                                                 |
| ---------------- | ------------ | ------- | ------------------------------------------------------------------------------------------------------- |
| `lrp-rli-ca-001` | `concurrent` | `W`     | per-wave conditional associations of concurrent skills with word reading                                |
| `lrp-rli-ca-002` | `concurrent` | `L`     | per-wave conditional associations of concurrent skills with letter sounds                               |
| `lrp-rli-ca-003` | `concurrent` | `TR`    | per-wave conditional associations of concurrent skills with taught receptive vocabulary                 |
| `lrp-rli-ca-004` | `concurrent` | `TE`    | per-wave conditional associations of concurrent skills with taught expressive vocabulary                |
| `lrp-rli-ca-005` | `concurrent` | `R`     | per-wave conditional associations of concurrent skills with standardised receptive vocabulary (ROWPVT)  |
| `lrp-rli-ca-006` | `concurrent` | `E`     | per-wave conditional associations of concurrent skills with standardised expressive vocabulary (EOWPVT) |
| `lrp-rli-ca-007` | `concurrent` | `B`     | per-wave conditional associations of concurrent skills with phoneme blending                            |
| `lrp-rli-ca-008` | `concurrent` | `F`     | per-wave conditional associations of concurrent skills with basic concepts (CELF)                       |
| `lrp-rli-ca-009` | `concurrent` | `T`     | per-wave conditional associations of concurrent skills with receptive grammar (TROG-2)                  |
| `lrp-rli-ca-010` | `concurrent` | `W`     | minimal-adjustment letter sounds → word reading, per wave (#421)                                        |
| `lrp-rli-ca-011` | `concurrent` | `W`     | letter sounds + nonword decoding → word reading, decoding held fixed (#421)                             |

**Association only — three caveats.** Every coefficient is an adjusted association; conditioning on contemporaneous (post-treatment) skill levels is intentional because nothing is read causally (contrast the level-factors family, which omits cross-skill terms to protect a causal contrast). Read with the **Table-2 fallacy** (each coefficient answers a different conditional question), **measurement error** (classical error often attenuates a simple association, but the size and direction of distortion are not guaranteed in a multivariable nonlinear model; longitudinal factor model #313 is a complementary latent-measurement analysis), and **collinearity plus regularisation** (n ≈ 53 with a correlated predictor cluster, so mutually adjusted and single-skill coefficients answer materially different questions). Their difference shows sensitivity to the conditioning set; it is not a decomposition of shared variance. Group and the missingness offsets are non-interpretable nuisances. Floored measures (`P`, `N`) are excluded as predictors and as focal outcomes; `TR` approaches its 24-item ceiling at later waves, which the Beta-Binomial respects but which compresses the resolution of `ca-003`'s later-wave associations. The 170-item standardised `R` and `E` measures do not have that focal-specific warning.

### Longitudinal correlated-domain-factor model — `lrp-rli-lcf-001` (`kind="long_corr_factor"`)

**Purpose.** A latent-measurement companion to the concurrent regression family (#313, descriptive-association workstream #314) and the four-wave extension of the cross-sectional `corr_factor` CFA (`lrp-rli-mm-001`). It estimates correlated **vocabulary {R,E,TR,TE} / code {L,B} / grammar {F,T}** domain factors at every timepoint over the child×wave panel and reports the **per-wave latent skill correlation matrices** plus directional, generally asymmetric conditional latent slopes derived from them; only the correlation matrices are symmetric. Indicator loadings and residual scales are wave-invariant, factor means have an exact zero-sum-over-waves constraint, and factor scores are marginalised out (the `mm-001` funnel fix); missing cells are masked, not dropped. A trait correlation matrix and one state correlation matrix per wave receive LKJ priors. Each reported within-wave correlation is their trait-share-weighted sum, so the reported matrices are induced and share a trait component rather than receiving independent per-wave LKJ priors.

| Model             | Kind               | Outcome | Purpose                                                      |
| ----------------- | ------------------ | ------- | ------------------------------------------------------------ |
| `lrp-rli-lcf-001` | `long_corr_factor` | —       | per-wave latent skill correlations (vocabulary/code/grammar) |

**Measurement / triangulation only.** Every latent correlation and slope is a descriptive association (ID-2), never causal. The current reporting estimates are nominal and exploratory at n ≈ 54, not final scientific magnitudes. A self-contained **latent-versus-observed comparison** places the factor correlations beside mean indicator-pair correlations as a triangulation diagnostic; no ordering is required because they are different estimands, and the observed comparator is a point estimate without its own uncertainty interval. The reproducible 48-row `lcf_concurrent_comparison.csv` then aligns LCF target-item translations with concurrent-family adjusted average marginal effects for `L` versus `R`/`E`/`TR`/`TE` and all four vocabulary indicators versus `L`/`B`, at each wave and for a `+1 same-wave SD` predictor change. Both sides are directional, but the LCF translation conditions on latent domains at a mean operating point while the concurrent family conditions on observed tests and averages a nonlinear marginal over rows, so no pass/fail ordering applies. Prior sensitivity is required before substantive interpretation. Wave-varying loadings or an AR across-wave structure should be fitted only if checks indicate that the wave-invariant measurement specification or compound symmetry misfits, with any fitted alternative compared using per-child PSIS-LOO. See `notes/202607142330-lrp313-longitudinal-corr-factor.md`.

### Joint bivariate mechanism — `lrp-rli-jm-001`, `lrp-rli-jm-002` (`kind="joint_mechanism"`)

**Purpose.** Turns two quantities the suite reports as _product-of-marginals sensitivities_ into identified posterior contrasts (#421 Tier 3 (1); letter-sound → word-reading review note #424, decoding-specificity note `202607172358`). Both are currently assembled by pairing draws from separate fits that share children, which imposes a cross-outcome covariance of zero the data do not have. Each model here fits word reading (`W`) and nonword decoding (`N`) together on the same standardised letter-sound exposure, with an **LKJ cross-outcome dependence block**, so the quantities become within-model deterministics. A single shared child intercept with a fixed loading of 1 on both logits would not do: conditional on such a scalar the two likelihood legs still factorise, so it yields no outcome-specific child effect, no residual correlation and no conditional slope.

`jm-001` is the **per-wave levels** design the issue specifies: one cross-sectional fit per timepoint, one row per child, with the within-wave residual correlation free. It reports both the identified contrast Δ = β(LS→N) − β(LS→W) and the identified **conditional-to-marginal slope ratio** — the letter-sound → word-reading slope holding _latent_ decoding fixed, `β_W − ρ (σ_W/σ_N) β_N`, over the unconditional slope. Its adjustment set and slope prior are matched to `ca-010` / `ca-011`, which makes the two conditional slopes comparable in construction but does **not** make it a nested replacement for their paired-draws ratio: this is a bivariate logistic-normal Binomial model conditioning on a latent skill, and `ca-011` is a Beta-Binomial fit conditioning on an observed count with mean-imputed missing predictors. The likelihood is Binomial, not Beta-Binomial: the bivariate residual already carries the extra-binomial variance, and a `kappa` alongside it is the second overdispersion mechanism that left the ITT joint's LKJ block prior-dominated in April 2026. Every published wave is trace-backed, convergence-scanned over its reported deterministics, given the new-child predictive check and a recorded power-scaling result, and release-gated as one bundle; no reporting path selects a wave after seeing its posterior (#591).

`jm-002` is the **phase-stacked ANCOVA** companion, matched term-for-term to `mech-096` / `mech-101` (own baselines, phase intercepts, the {G, A, HS, IS, SP} adjustment set, Beta-Binomial denominators of 79 and 6 never pooled). It re-reports the Tier-1 Δ on the parameterisation that contrast was originally computed on. Its dependence block is a **bivariate child random intercept**, i.e. a between-child covariance, so it reports Δ and `rho_outcome` but no conditional slope ratio. Its estimand is each outcome's post-level given its own baseline — an ANCOVA association pooling between-child and within-child information, not a within-child change effect.

**The two Δs are different estimands and need not agree in sign** — which is exactly why both models exist rather than the Tier-1 number being read off the levels fit. `jm-001`'s Δ asks how much higher each _score_ is per SD of letter sounds; `jm-002`'s asks how each score's post-level differs per SD of letter sounds among children who started at the same place. A shared reading-development / general-ability component lifts the letter-sound → word-reading _level_ association without lifting the ANCOVA one, and the 6-item nonword floor compresses nonword levels harder.

**Neither is a nested substitute for its marginal comparators.** `jm-002` requires both outcome baselines on every retained transition and standardises the exposure once over that joint union (153 rows, exposure SD 1.41), while `mech-096` and `mech-101` filter to their own outcome's rows and re-standardise there (152 rows at 1.39 and 156 rows at 1.43). One SD is a different raw increment in each, and the word-reading marginal keeps rows the joint fit excludes — so the gap between the joint contrast and the paired-marginal sensitivity is not attributable to cross-outcome covariance alone. `scripts/compare_statistical_models.py` publishes that row/cell/scaler reconciliation beside the two contrast rows and marks the comparison `comparable=False` when they diverge.

| Model            | Kind              | Outcome | Purpose                                                                                   |
| ---------------- | ----------------- | ------- | ----------------------------------------------------------------------------------------- |
| `lrp-rli-jm-001` | `joint_mechanism` | —       | per-wave bivariate levels: conditional slope ratio + decoding-specificity contrast        |
| `lrp-rli-jm-002` | `joint_mechanism` | —       | phase-stacked bivariate ANCOVA companion: identified Tier-1 decoding-specificity contrast |

**Descriptive only.** Every slope is an adjusted association — latent general ability is unobserved and neither dependence block stands in for it — never causal. Δ is a Campbell–Fiske convergent/discriminant argument, not identification of a causal decoding effect, and that argument assumes a cross-instrument measurement invariance neither model imposes: with unequal loadings on one general ability the two latent-scale slopes differ even with no causal letter-sound route, and different item counts, floors and link discrimination can each move the difference. `share_retained` is a conditional-to-marginal **slope ratio**, not a bounded pathway share: it is unbounded, governed by one prespecified stability rule covering both instability routes (the unconditional slope and the held-fixed outcome's residual scale must each be supported more than 0.05 logit from zero with at least 95% probability), withheld entirely where that fails — with the denominator-free `abs_slope_reduction` published in its place — and elsewhere published with the probability mass below zero / inside [0, 1] / above one, and never with a mean. It conditions on the **latent** nonword logit where `ca-011` conditions on the observed count; classical measurement-error intuition suggests it retains less, but that ordering is not guaranteed across two nonlinear models with different likelihoods and floors, so the two are not presented as bracketing. Nonword decoding is floored for 72 / 64 / 52 / 40 % of children at t1–t4, so its residual scale is the least well determined quantity in `jm-001`. `scripts/compare_statistical_models.py` (`tier1_decoding_specificity`) writes the identified Δ from `jm-002` alongside the product-of-marginals row, flagged `identified=True`, rather than replacing it — the sensitivity row stays as the historical comparator.

### Wave-pooled level association — `lrp-rli-pl-001–lrp-rli-pl-006` (+ `lrp-rli-pl-101`) (`kind="pooled_levels"`)

**Purpose.** The one skill-to-skill question neither `concurrent` (per-wave levels) nor `mechanism` (post-score given pre-score) asks: how does one skill's level track another's pooled over all four waves? One Beta-Binomial likelihood over every child-wave row with per-wave intercepts and a child random intercept, and a Mundlak split of the exposure into the child's study mean and the wave's deviation from it, because a single random-intercept coefficient returns a precision-weighted blend of the between-child and within-child associations. **Nothing here is causal**: exposure and outcome are contemporaneous, the arm term pools post-crossover waves, and latent ability is unblocked. Adjusters match `concurrent` (hearing, speech, the t1 block-design ability proxy, linear age). Two extensions (#553) carry the split to the other predictors of word reading: a **raw-score covariate exposure** (`mechanism_is_covariate` — `erbto`, `deapp_c`, whose documented maxima are recorded nowhere, enter as the standardised raw score, complete-case on the exposure via `require_observed`, with the raw-units SD recorded beside the fit) and **same-wave skill adjusters** (`skill_symbols` — the standardised logits of other measures at the row's wave, each a `gamma_<symbol>` adjusted association; a row is kept only when the outcome, the exposure and every skill are observed, and the dropped count is reported). Adjustment sets for the new fits mirror each exposure's `mechanism` model minus the own baseline; `attend` is omitted throughout (an interval dose, a transition covariate).

| Model            | Path          | Purpose                                                                                                                                      |
| ---------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `lrp-rli-pl-001` | `L → W`       | Letter sounds → word reading, between-child and within-child slopes with per-wave intercepts (primary)                                       |
| `lrp-rli-pl-002` | `L → N`       | Letter sounds → nonword decoding, the decoding counterpart on the same scale                                                                 |
| `lrp-rli-pl-003` | `E → W`       | Expressive vocabulary → word reading; same-wave `TR`, `TE`, `R` skill adjusters, hearing / phonological memory / speech, ability, age (#553) |
| `lrp-rli-pl-004` | `R → W`       | Receptive vocabulary → word reading; same-wave `TR` skill adjuster, hearing / phonological memory, ability, age (#553)                       |
| `lrp-rli-pl-005` | `erbto → W`   | Phonological memory (word/nonword repetition, raw-score covariate exposure, `require_observed`) → word reading; hearing, ability, age (#553) |
| `lrp-rli-pl-006` | `deapp_c → W` | Speech production (raw-score covariate exposure, `require_observed`) → word reading; hearing, phonological memory, ability, age (#553)       |
| `lrp-rli-pl-101` | `L → W`       | Comparator for `001` without wave intercepts: shows how much apparent within-child association secular co-movement alone manufactures        |

### Historical growth, Byrne cohort — `lrp-rlm-hg-001–lrp-rlm-hg-009` (`kind="historical_growth"`, `study_id="rlm"`)

**Purpose.** The Byrne, MacDonald & Buckley (2002) reading-language-memory comparable-model
suite's Phase A (#338): one descriptive group-by-wave growth model per measure for the three
reading groups (Down syndrome / average readers / reading-matched). Beta-Binomial on the
bounded count over the supported (group, wave) cells, with a group-centred per-child random
intercept whose SD — and the overdispersion `kappa` — are indexed by group. Each model's
complete-case **core** is the paper's waves 1–3 (the Table 2 audit subset; waves 3–4 for the
wave-3+ `basmat`), with waves 4–5 as **extension waves**: children in the core contribute
wherever the measure was observed, so later cells are an attrition-selected follow-up tail
with their own per-cell `n`. Wave 4 carries all three groups (the between-group window);
wave 5 exists only for the Down syndrome group. Interval growth is summarised on the
children observed at both endpoint waves.

| Model            | Kind                | Measure   | Window                |
| ---------------- | ------------------- | --------- | --------------------- |
| `lrp-rlm-hg-001` | `historical_growth` | `basread` | waves 1–4 + DS wave 5 |
| `lrp-rlm-hg-002` | `historical_growth` | `basspel` | waves 1–4 + DS wave 5 |
| `lrp-rlm-hg-003` | `historical_growth` | `woco`    | waves 1–4 + DS wave 5 |
| `lrp-rlm-hg-004` | `historical_growth` | `bpvs`    | waves 1–4 + DS wave 5 |
| `lrp-rlm-hg-005` | `historical_growth` | `trog`    | waves 1–4 + DS wave 5 |
| `lrp-rlm-hg-006` | `historical_growth` | `basdig`  | waves 1–4 + DS wave 5 |
| `lrp-rlm-hg-007` | `historical_growth` | `bassim`  | waves 1–4 + DS wave 5 |
| `lrp-rlm-hg-008` | `historical_growth` | `basnum`  | waves 1–4 (no wave 5) |
| `lrp-rlm-hg-009` | `historical_growth` | `basmat`  | waves 3–4 + DS wave 5 |

**Descriptive only — no causal quantity exists in this cohort.** `readgrp` is an
observational cohort factor (`causal_status="none"` throughout); there is no intervention,
so the four intervention-dependent RLI families (`itt`, `did`, `aligned`, `dose_response`)
have no Byrne counterpart. ("Intervention-dependent", not "randomised": only `itt` and the
`did` t2 contrast rest on randomisation — `aligned`'s cohort contrast is explicitly not
randomised, and `dose_response`'s attendance slopes are observational.) Six Beta-Binomial ceilings are researched and confirmed (`basread` 90, `bpvs` 32,
`trog` 20, `basdig` 34, `bassim` 21, `basmat` 28 — #338 sign-off, 2026-07-16); `basspel`,
`basnum` and `woco` keep **provisional observed-max ceilings** (`n_trials_confirmed=False`)
pending their instrument manuals. The primary paper identifies `basspel` as 1983 BAS
Spelling and `basnum` as BAS number-skills raw scores; its published number-score means
reproduce exactly from the prepared extract. Those identities are therefore confirmed,
but the paper does not state either instrument ceiling. The reading-matched group is _selected
on_ `basread` level, so between-group contrasts touching that group carry the selection
caveat. Roadmap: the phased Byrne suite is tracked in #338 and mapped in
`notes/202607131600-byrne-comparable-models-plan.md`.

### Byrne Phase B/D — `lrp-rlm-jc-001/002`, `lrp-rlm-mm-001`, `lrp-rlm-adj-001–006`, `lrp-rlm-hs-001–003`, `lrp-rlm-ca-001/002`, `lrp-rlm-gc-001` (`study_id="rlm"`)

**Purpose.** The Byrne suite's joint/measurement structure (Phase B) and predictor views (Phase D), ported from the RLI observational families per the plan and the 2026-07-16 decisions. The span models use the audited w1→w3 core window; the concurrent models fit waves 1–4 separately and restrict their battery to confirmed-denominator, confirmed-identity measures. All quantities are descriptive associations in an observational cohort — nothing causal exists here.

| Model             | Kind               | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ----------------- | ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lrp-rlm-jc-001`  | `historical_joint` | Joint correlated growth over `basread`/`bpvs`/`basdig`: per-measure supported-cell grids + LKJ-correlated per-child stable offsets; headline = the 3×3 between-child stable-level correlation matrix (n = 71, waves 1–3 core + extension waves)                                                                                                                                                                                  |
| `lrp-rlm-jc-002`  | `historical_joint` | Balanced waves 1–3 companion over the same confirmed measures: adds double-centred wave-specific latent deviations; target = the 3×3 within-child latent-logit correlation matrix, interpreted only where both residual scales resolve (n = 71)                                                                                                                                                                                  |
| `lrp-rlm-jc-102`  | `historical_joint` | Registered alternative-prior sensitivity for `lrp-rlm-jc-002`: the same fit with the within-scale prior widened to `HalfNormal(1.0)`. That scale decides which measures clear the resolvability threshold and therefore which correlation pairs are interpretable, so it is a conclusion-level assumption, not a nuisance (#588 finding 5). **Not yet fitted** — until it is, the parent's prior-robustness claim is preliminary |
| `lrp-rlm-mm-001`  | `corr_factor`      | Wave-3 correlated domain-factor measurement model: reading {`basread`,`basspel`,`woco`}, language {`bpvs`,`trog`}, memory {`basdig`, single indicator, fixed reliability 0.8}, ability {`bassim`,`basmat`,`basnum`}; measurement-only (n = 75)                                                                                                                                                                                   |
| `lrp-rlm-adj-001` | `adjusted`         | Wave-1 predictors of w1→w3 word-reading gain, mutually adjusted + bivariate comparison + prior-sensitivity sweep (pooled, n = 69)                                                                                                                                                                                                                                                                                                |
| `lrp-rlm-adj-002` | `adjusted`         | Pre-specified reduced Down-syndrome-only companion: `basdig`, `bpvs` and `bassim` predictors of w1→w3 word-reading gain, conditional on own baseline, without an age or group slope (22 of 24 children)                                                                                                                                                                                                                          |
| `lrp-rlm-adj-003` | `adjusted`         | Confirmed-input D1 companion: `basread`, `trog`, `basdig`, `bassim` and age predictors of w1→w3 BPVS receptive-vocabulary gain, conditional on own baseline; adjusted + bivariate + prior-sensitivity fits (pooled, n = 71)                                                                                                                                                                                                      |
| `lrp-rlm-adj-004` | `adjusted`         | Confirmed-input D1 companion: `basread`, `bpvs`, `basdig`, `bassim` and age predictors of w1→w3 TROG receptive-grammar gain, conditional on own baseline; adjusted + bivariate + prior-sensitivity fits (pooled, n = 69)                                                                                                                                                                                                         |
| `lrp-rlm-adj-005` | `adjusted`         | Confirmed-input D1 companion: `basread`, `bpvs`, `trog`, `bassim` and age predictors of w1→w3 BAS digit-recall gain, conditional on own baseline; adjusted + bivariate + prior-sensitivity fits (pooled, n = 71)                                                                                                                                                                                                                 |
| `lrp-rlm-adj-006` | `adjusted`         | Confirmed-input D2 stacked ANCOVA: pooled within-transition `bpvs`, `trog`, `basdig`, `bassim` and age predictors of annual BAS word-reading progress over w1→w5; child-level LOO, common-horizon w1→w4 and transition-specific sensitivities (225 transitions, 84 children)                                                                                                                                                     |
| `lrp-rlm-hs-001`  | `horseshoe`        | Regularised-horseshoe ranking cross-check over the identical frame; no GB comparison exists for this cohort — the partner is `lrp-rlm-adj-001`                                                                                                                                                                                                                                                                                   |
| `lrp-rlm-hs-002`  | `horseshoe`        | Regularised-horseshoe cross-check for the confirmed-input BPVS gain frame; no GB comparison exists for this cohort — the partner is `lrp-rlm-adj-003`                                                                                                                                                                                                                                                                            |
| `lrp-rlm-hs-003`  | `horseshoe`        | Regularised-horseshoe cross-check for the confirmed-input TROG gain frame; no GB comparison exists for this cohort — the partner is `lrp-rlm-adj-004`                                                                                                                                                                                                                                                                            |
| `lrp-rlm-ca-001`  | `concurrent`       | Per-wave concurrent correlates of BAS word reading over waves 1–4; confirmed five-measure subset, age and reading-group nuisance adjustment, plus single-skill comparators                                                                                                                                                                                                                                                       |
| `lrp-rlm-ca-002`  | `concurrent`       | Per-wave concurrent correlates of BPVS receptive vocabulary over waves 1–4; confirmed five-measure subset, age and reading-group nuisance adjustment, plus single-skill comparators                                                                                                                                                                                                                                              |
| `lrp-rlm-gc-001`  | `growth`           | Wave-1 BAS similarities and the BAS word-reading trajectory over paper-compatible waves 1–3; reading-group-specific nuisance trajectories with a shared within-group ability–growth association (87 children)                                                                                                                                                                                                                    |

**Notes.** `lrp-rlm-jc-001/002` compute no PSIS-LOO (one likelihood node per measure); the
correlation matrices are shared across groups (stated assumption) while the random-effect
scales stay group-indexed. RLMJC02 excludes the attrition-selected extension tail so each
child contributes exactly three waves. Its logistic-normal residual supplies the
extra-Binomial variance; a development fit that also retained Beta-Binomial
overdispersion was rejected as prior-dominated. A wider residual-scale prior remains
a required sensitivity before interpretation. `lrp-rlm-mm-001` states its single-indicator memory reliability
assumption and pooled-loadings (invariance) assumption up front. The pooled Phase D word-reading span model excludes `basmat` (no wave-1 value) and the reading-route `basspel`/`woco`. Its Down-syndrome-only companion is `lrp-rlm-adj-002`: three pre-specified confirmed-ceiling skills replace the prior-dominated seven-slope proposal, leaving 22 complete cases. The D1 adjusted models `lrp-rlm-adj-003–005` extend the span question to BPVS receptive-vocabulary, TROG receptive-grammar and BAS digit-recall gain using only confirmed inputs. BPVS and TROG have registered horseshoe cross-checks (`lrp-rlm-hs-002/003`); the proposed digit-recall horseshoe was rejected because the rep-lite fit retained divergences even at `target_accept=0.999`. The D2 model `lrp-rlm-adj-006` stacks the four annual word-reading transitions, standardises predictors within transition, and treats repeated children as the LOO unit. Its wave-5 transition is Down-syndrome-only, so the common-horizon-through-wave-4 and transition-specific-slope refits are required scope checks rather than optional embellishments. BAS word reading remains selection-sensitive because the reading-matched group was selected on reading level. BAS spelling and word completion remain outside this bounded-count matrix while their score ceilings are provisional. The concurrent pair also excludes provisional-denominator `basspel`, `woco` and `basnum`; confirmed source-native `basmat` remains outside it because it has no wave-1 value and is not part of the paper's reported three-wave battery. Wave 4 is an attrition-sensitive extension and predictor missingness is reported under the family's mean-imputation policy. Phase C's annual lagged working graph is adopted (`dag/dag-reading-language-memory-lagged.dagitty`), with prior word reading pointing only to later receptive vocabulary, receptive grammar and digit recall. No Phase C lagged model is registered: the pre-fit recovery study rejected both the Down-syndrome-only and three-group shared-coupling candidates because none recovered all three modest positive paths reliably (`notes/202608141812-byrne-lcsm-feasibility.md`). A formal temporal measurement-invariance model is also deferred: the confirmed five-measure battery leaves three of four proposed domains as fixed-reliability single indicators, while the wider battery introduces three provisional ceilings; separate per-wave fits would be a stability screen rather than a repeated-child invariance test. Immediate and delayed visual recall are absent from the retained source, so the paper's full memory battery cannot be reconstructed from this archive. Phase E remains gated on the #289/#324 sensitivity prerequisite.

`lrp-rlm-gc-001` completes #409 D4 over confirmed inputs. It excludes waves 4–5 from the primary trajectory question, requires wave-1 `bassim` and at least two observed `basread` waves, and treats the three reading groups as nuisance trajectory strata. Its common `gamma` is an adjusted within-group association, not evidence that verbal reasoning causes reading growth; the reading-matched selection caveat remains. Any high-Pareto observation cells trigger a trace-backed exclusion refit that tests `gamma`/`delta` stability without being mislabelled as exact or child-level LOO.

---

## Conventions and pointers

- **Fit a model:** `python scripts/fit_statistical_model.py {model_id|all} --config dev|test|reporting [--render]` (Layer 2); `python scripts/fit_model.py {model_id|all} --config dev [--render]` (Layer 1).
- **Reports:** one per model at `docs/models/{model_id}/index.qmd`; thin templates that include shared partials from `docs/models/_partials/`. Statistical reports use the findings-first order `_header` → `_setup` → `_gate_badge` → `_key_findings` → collapsed `_reading_guide` → model prose → family results → `_priors` → `_prior_predictive` → collapsed `_technical` (the full convergence banner, sampling diagnostics and posterior-predictive checks) → `_footer`. A failed gate is a prominent red badge; the key-findings interlock withholds result sentences, and the shared setup suppresses scientific result tables and figures while retaining diagnostic material. Selection variants fall back to their parent's template.
- **Cross-model comparisons:** `scripts/compare_statistical_models.py` (ITT-vs-joint `τ` consistency, `τ` and mechanism-slope forests, nested PSIS-LOO).
- **Interpreting results:** read the visible sampling-quality badge before interpreting; expand Technical checks for R-hat, ESS, divergences, BFMI and the full predictive diagnostics. Zero divergences is the only automatic clean pass; a future trace-bound divergence qualification is amber and explicitly **not passed**, under the narrow policy in `METHODS.md`. Report the posterior (median + inner 50 % and outer equal-tailed 89 % credible intervals + tail probability, with an 89 % HPDI sensitivity interval alongside); positive `τ` = intervention helps; only a contrast explicitly licensed by randomisation is causal.
- **Source of truth:** Layer-1 ids/outcomes live in each module + `models/registry.py`; Layer-2 in each module's `SPEC` (`statistical_models/`). Keep this inventory in step with those when models are added, renamed, or retired.
