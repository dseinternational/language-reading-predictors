# Methods

This document describes the methods used in this project. See [README.md](README.md) for an introduction to the project.

## Overview

We use two processes for exploring predictors of language and reading outcomes:

1. **Gradient boosting algorithms** to identify the most influential predictors for each
   outcome.
2. **Bayesian statistical models** to estimate interpretable quantities with quantified uncertainty — treatment effects, dose-response curves, mediation — answering specific questions.

## 1. Gradient-boosting models

**Purpose:** rank predictors by how much they help out-of-sample prediction, and read the direction and consistency of each effect.

**Fit and cross-validate.** LightGBM with an MAE objective, `GroupKFold` by `subject_id`. One module per problem; level and gain targets are separate models.

**Tune.** `scripts/tune_model.py` runs an Optuna TPE study under the same `GroupKFold`. Inside each training fold it carves an inner `GroupShuffleSplit` for early stopping, so the outer validation fold is never seen by `early_stopping` and the reported CV RMSE and best iteration stay independent; the mean best iteration becomes the tuned `n_estimators`. Results land in `output/tuning/{model_id}/`. Tuning does **not** mutate the registry — applying tuned parameters is a manual, reviewable edit, so the source stays the single source of truth.

**Select features.** Iterative selection prunes ~34 predictors to ~5–15 per model, dropping zero-importance noise while tracking CV MAE, R² and MedAE (median absolute error). Each Select step is a named `SelectionStep` plus a dated note. Per-fit diagnostics — Spearman correlation, a distance-correlation dendrogram, a mutual-information heatmap, and importance pairing — guard against cutting a predictor that is merely collinear with one being kept.

**Interpret.** Read the SHAP beeswarm (`output/models/{model_id}/shap_summary.png`) _together with_ the permutation-importance ranking: importance is _how much_ a feature contributes, the beeswarm is _which direction_ and _how consistently_. They often disagree in interpretively important ways — a top-ranked predictor can run opposite to every other, be non-monotonic or sit beside a similar-importance predictor of opposite sign. For each top predictor, check whether low (blue) and high (red) values separate cleanly across zero, how tight the effect is, and whether any tail acts against the dominant direction. Always state the direction; never let a reader infer it from importance alone. A predictive importance on a `_GAIN` or `_NEXT` variable is _not_ a causal claim.

## 2. Bayesian statistical models

**Purpose:** estimate interpretable quantities with full uncertainty. Eight families, each a factory in `factories.py` and a pipeline entry in `pipeline.py`:

- **ITT** — the randomised-phase treatment effect (τ) on one outcome (the uniform DAG-faithful **LRPITT01–11** suite; the own baseline and linear age are _precision_ terms only — the ITT effect is identified by the empty adjustment set — and no cross-baselines enter). Heavily-floored outcomes (phonetic spelling, nonword reading) use a pre-specified, arm-blind **floor rule** (≥ 40 % of post-scores at zero at t2): a binary "off-floor" primary estimand (`Pr(post > 0)`), with the graded effect demoted to a flagged, detection-limited secondary read only beside per-arm mover counts.
- **Joint** — one posterior over the suite outcomes' treatment effects, enabling cross-outcome contrasts such as "is τ_L more positive than τ_W?".
- **Mechanism** — a dose-response curve `f_mech` of one measure on another (e.g. letter-sound → word reading), conditioned on a DAG-derived adjustment set and a per-child random intercept.
- **Mediation** — a natural direct/indirect decomposition (NDE/NIE) through a mediator by counterfactual g-formula simulation from the posterior, not from coefficients.
- **DiD** — a within-person replication of the randomised τ (the **LRPDID01–06** family): the wait-list arm's own untreated → crossover transition, each child its own control, with the immediate arm anchoring the maturation trend.
- **Gain factors** — a DAG-focused ANCOVA of a period's post-score on its own pre-score (the **LRPGF01–08** family, one model per outcome, each with a treated-only companion), with a per-child random intercept repairing latent general ability. Only the randomised on-intervention term is causal; every other covariate (own baseline, age, cognitive ability, upstream DAG skills, focal interactions) is reported as an explicit _adjusted association_, never as "X drives Y". SES is excluded — not a DAG node, and statistically redundant. Phonetic spelling inherits the ITT floor rule (an off-floor `Pr(post > 0)` Bernoulli, the treatment marginal an off-floor risk difference).
- **Level factors** — the companion _levels_ view (the **LRPLF01–08** family): the score at each timepoint, with group-by-time and ability-by-time as per-timepoint coefficient vectors. Only the t2 group contrast is a clean randomised effect; later timepoints are post-crossover and reported as associations.
- **Aligned** — an onset-aligned per-protocol single gain (the **LRPAL01–08** family, plus a dose sensitivity variant): both arms are aligned by intervention onset (immediate t1 → t3, wait-list t2 → t4) into one cross-sectional ANCOVA per child. This trades randomisation for a like-for-like dose comparison, so the cohort contrast is **not** the ITT effect — it is confounded by age-at-onset and cohort/timing, and every coefficient is reported as an association. Cumulative sessions (dose) are a collider and enter only the sensitivity variant.

**Likelihood and priors.** A Beta-Binomial likelihood on the bounded post-score count via a logit linear predictor. Priors come from shared constructors in `priors.py` (so the factories cannot drift apart); smooth nonlinear terms use Hilbert-space Gaussian-process approximations (`hsgp.py`).

**Prior-setting protocol.** Priors are **weakly informative and scale-calibrated**, not "uninformative": on a bounded-count logit model a very broad coefficient prior implies implausibly large probability- and item-scale effects — a `Normal(0, 0.5)` on τ is a plausible ~4 readable items on the 32-item letter-sound test but a ~35-item 95 % swing on the 170-item vocabulary tests. Each prior is **role-specific** — the causal τ, the _precision_ covariates (own baseline, age), the adjusted _associations_ (`gamma_cross`, the mechanism and mediator slopes), the _nuisance_ terms (intercept, the dispersion `kappa`, random-effect scales) and the _GP_ amplitudes / lengthscales — and its role is documented per parameter in every report's `priors_table.csv` (a guard test refuses any prior the table cannot describe). The **treatment-effect prior is tiered by outcome** (`measures.DISTAL_OUTCOMES`): proximal directly-taught and decoding outcomes keep the wider `Normal(0, 0.5)`; the broad standardised-transfer outcomes (vocabulary, grammar, concepts) take a tighter `Normal(0, 0.3)`, because that is where a large effect is least expected and where the wide prior is worst on the item scale. What may inform a prior: **measurement and scale properties (ceilings, denominators, test–retest reliability), external independent-sample literature, expert judgement from the education team, and pre-specified substantive constraints — but never the posterior of the same fitted model.** That last exclusion bites here: the analysis dataset _is_ the Burgoyne et al. (2012) trial, so that trial's own published effects are off-limits as a τ prior (using them would condition the prior on the data we are about to fit). Every prior is checked by **prior-predictive simulation and an estimand-scale pushforward** — does it generate plausible scores, gains and floor / ceiling patterns? — and any tightening is justified by that reasoning and confirmed **stable under a prior-sensitivity sweep** (`scripts/tau_prior_sensitivity.py`), not adopted because an estimate is inconveniently uncertain. Full rationale and the per-parameter inventory live in `docs/models/PRIORS.md` and `notes/202607011600-issue-141-prior-audit.md`.

**Sign convention:** `G = 1` is the immediate-intervention arm after the `2 − group` recode (`group == 1` = immediate intervention, `group == 2` = wait-list control), so **positive τ means the intervention raises the outcome**.

**Fit.** NUTS via `nutpie`, with `dev` / `test` / `reporting` sampling presets (the last is 6 chains × 6000 draws); `--target-accept` overrides when a funnel needs it.

**Evaluate.** Check convergence _before_ reading any estimate: R-hat ≈ 1.00, adequate effective sample size (ESS), and divergent transitions at or near zero. A non-converged or divergent fit's posterior is not interpretable — fix the model, do not report it. The pipeline writes a `diagnostics_summary.json` (divergences, BFMI per chain, R-hat ≤ 1.01, ESS ≥ 400) and the report renders it as a **pass/fail convergence banner first**, before any τ; Pareto-k (pointwise LOO), rank, ESS-evolution and LOO-PIT plots back it up. Prior- and posterior-predictive checks confirm the likelihood can reproduce the data; the 1,000 prior draws are persisted onto `trace.nc` (the `prior` / `prior_predictive` / `log_prior` groups) so the report can show the prior-predictive check, the prior-vs-posterior overlay, the estimand-scale prior pushforward, and power-scaling sensitivity without refitting.

**Compare.** PSIS-LOO via ArviZ: prefer the higher-`elpd` model only when the difference clears its standard error (`elpd_diff` against `dse`). The interaction models are tested against their own no-interaction baselines as clean nested comparisons; `scripts/compare_statistical_models.py` collects the cross-model views.

**Interpret.** Report the **posterior**, not a point estimate: the **median** (preferred over the mean — it is transformation-invariant across the logit and probability scales) with its 95 % equal-tailed credible interval. A highest posterior density interval (HPDI) at the same coverage is reported alongside as a per-scale **sensitivity** check — useful when a bounded-scale posterior is skewed — but not as the primary interval, since the HPDI is not transformation-invariant. Posterior bands follow a fixed convention (issue #177): the **equal-tailed 95 % credible interval** is the headline uncertainty summary; the **central 50 %** interval (the middle half of the posterior mass) is a visual aid, not a decision threshold; and the **equal-tailed 90 %** interval may be shown as a sensitivity / compatibility band — never as a substitute headline. Read direction from the tail probability `P(τ > 0)` directly, not from whether a 90 % or 95 % band excludes zero. There are no p-values. Distinguish two questions: _direction_ — the tail probability `P(τ > 0)` — and _magnitude_ — `P(benefit ≥ δ)` against a pre-specified minimally-important difference δ (a region of practical equivalence), because a high `P(τ > 0)` can sit on a practically negligible effect. Give both the logit scale (the natural parameter) and the probability / items scale at sample-mean baseline. A 95 % credible interval that includes zero means the sign is not pinned down — avoid a decisive directional claim, and report where the posterior mass sits rather than collapsing to "significant / not". At this study's sample size a significant-looking point estimate is on average magnitude-inflated (a Type-M / winner's-curse effect), so lead with the interval, not the point. **Evidence labels** (inconclusive / suggestive / moderate / strong / very strong; issue #179) qualify the evidence for a **named claim** — "moderate evidence that the intervention helps", never "a moderate effect" — are reported _after_ the probability and odds, and are oriented to the **favoured direction**, so a clearly negative effect is reported as evidence of harm, not as inconclusive. See `notes/202606261304-evidence-strength-and-rope-reporting.md`.

## Causal interpretation and its limits

**What is causal:** the treatment effect τ, because randomisation during the trial phase balances the arms on everything, observed and unobserved. The own-baseline term is in the model for _precision_, not identification — say so when writing up an ITT result, so a reader does not mistake the baseline adjustment for the thing that licenses the causal claim.

**What is not:** every association we did not randomise — the `gamma_cross` "baseline X predicts later Y" terms, the mechanism `f_mech` slope, and the mediator → outcome arrow. These have cross-lagged-panel structure and can be inflated by a stable trait shared across measures or by ordinary measurement noise (which alone can manufacture a spurious "earlier → later" path). The mechanism models already include the recommended repair — a per-child random intercept, but two leaks remain: the mechanism is measured at the _same wave_ as the outcome, and we do not separately model measurement error. So report these as _adjusted associations_, never as "X drives Y".

## Reporting results

Write for a numerate reader who knows frequentist statistics but is newer to Bayesian methods — picture a science undergraduate. Lead with the finding and its uncertainty, then the "so what"; keep narratives as short as the content allows.

- **Translate the Bayesian parts.** Expand shorthand on first use (τ = treatment effect, `gamma_own` = baseline coupling, `f_mech` = mechanism curve). Read an equal-tailed 95% credible interval in plain words — "given the model and data, the value lies in [a, b] with 95% probability" — prefer a posterior tail probability (`P(τ > 0) = 0.97`) to anything that reads like a p-value, and restate the sign convention wherever τ appears.
- **Pair every estimate with its uncertainty** — never a bare posterior mean or a bare importance rank.
- **Avoid verbal evidence labels ("strong", "moderate") wherever possible** — report the odds and probabilities and let them speak. Where prose forces a label, append the word _evidence_ and name the claim ("strong evidence that the intervention helps"), show the odds beside it, and use the round-odds ladder in `notes/202606261304-evidence-strength-and-rope-reporting.md` — run separately for _direction_ (`pd`) and _meaningful benefit_ (`P(benefit ≥ δ)`). A label must never connote effect _size_.
- **Cite primary sources, verify them before committing them to text, and always include a DOI** when one exists.

## Conventions

- **No outlier exclusion** unless justified on the model page.
- **`GroupKFold` by `subject_id`** wherever longitudinal leakage would otherwise inflate performance.
- **Mechanism-model β_G is not a direct-effect estimate** — both arms are on intervention during phases 1 and 2, so the pooled β_G averages over phases where group is no longer a treatment contrast (documented in each mechanism report).
- **RLI intervention scope** — the intervention targets vocabulary and grammar as well as reading, so null effects on receptive/expressive vocabulary are substantive findings, not by-design predictions.
- **Decisions a future reader might question get a dated note in `notes/`** before they leave your head — the question, the choice, the rationale, the result. Mark AI-drafted notes with a `> [!NOTE]` admonition naming the tool and model (per the AI-authorship rule in `CLAUDE.md`).

## Glossary

- **Posterior** — the distribution of a parameter after combining prior and data; summarised by a mean/median, a credible interval, and tail probabilities.
- **Credible interval (CrI)** — a range holding the parameter with stated probability _given the model and data_ (the reading people wrongly attach to a confidence interval).
- **Posterior tail probability** — e.g. `P(τ > 0)`, the posterior mass on one side of a value; used in place of a p-value.
- **Probability of direction (`pd`)** — the same quantity as the posterior tail probability `P(τ > 0)`; answers existence/direction, not magnitude. For symmetric posteriors `p ≈ 2(1 − pd)`, so a `pd` threshold is a p-value threshold in disguise.
- **ROPE / minimally-important difference (δ)** — a band [−δ, +δ] around zero wide enough that an effect inside it is practically equivalent to none (the frequentist SESOI / clinical MID); `P(|effect| < δ)` is its coverage, `P(effect > δ)` the probability of a meaningful benefit.
- **Type-S / Type-M error** — among effects declared "significant", the rate of wrong _sign_ (S) and the average _magnitude_ exaggeration (M); both worsen at low power.
- **τ (tau)** — the ITT treatment effect (positive ⇒ intervention helps).
- **`gamma_own` / `gamma_cross`** — a measure's own-baseline coupling / its cross-baseline couplings, on the logit scale.
- **`f_mech`** — the nonparametric dose-response curve in a mechanism model.
- **ITT (intention-to-treat)** — analyse children by the arm they were randomised to.
- **Mediation / NDE / NIE** — how much of an effect flows through a mediator (natural direct / natural indirect effect).
- **PSIS-LOO** — Pareto-smoothed importance-sampling leave-one-out cross-validation, the Bayesian model-comparison score (`elpd`).
- **HSGP** — Hilbert-space approximation to a Gaussian process; a fast smooth nonlinear term.
- **Regression to the mean** — extreme scores drift toward the average on retest; visible here as `gamma_own` below 1.
