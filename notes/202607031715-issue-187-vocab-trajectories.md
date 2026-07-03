# Verbal & reading trajectories, and whether baseline non-verbal ability predicts their shape (issue #187, Q5)

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8).

**Question (Q5, collaborator request — issue #187):** characterise the longitudinal trajectories of the verbal and reading measures across the four RLI waves, and ask whether baseline non-verbal ability predicts the _shape_ of those trajectories.

**Short answer.** All five measures grow over the four waves, and their growth rates are positively coupled — a shared developmental tempo, strongest for reading. Baseline non-verbal ability (block design) strongly predicts where children _start_ (the baseline level of vocabulary and grammar) but predicts the growth _rate_ of only one measure — **receptive grammar** (moderate-to-strong positive); vocabulary and reading growth rates are inconclusive. And baseline non-verbal ability does **not** predict the common growth tempo (correlation ≈ 0). Everything here is an **adjusted, `GA`-confounded association, not a causal effect** (framing below). This complements #186: non-verbal ability's link to vocabulary was a marginal/level association that all but vanished once own baseline was conditioned on — here its link to vocabulary _growth_ is likewise near-null.

## Vehicle — two new joint growth-curve models

Q5 is a _trajectory-shape_ question, distinct from LRP67's latent change-score model (which asks what within-child _change_ drives reading change). So it takes a new joint multivariate **latent growth-curve** model (`kind="growth"`), reusing the four-wave `WavePanel` scaffold LRP67 introduced. For each of the five measures — receptive vocabulary (R = ROWPVT), expressive vocabulary (E = EOWPVT), receptive grammar (T = TROG-2), word reading (W = EWRSWR), letter-sound knowledge (L = YARC-LSK) — a per-child latent logit intercept and linear age slope enter a masked Beta-Binomial; baseline non-verbal ability (`blocks`, WPPSI Block Design, recorded at t1, complete for all 54 children) enters, standardised, as a per-child predictor of both:

- `gamma_k` — baseline non-verbal ability → growth **rate** of measure k (the headline Q5 estimand);
- `delta_k` — → baseline **level**.

Two layers (the approved design):

- **LRP69 (independent core):** per-measure trajectories, random intercept + slope independent across measures (the within-measure intercept–slope correlation is omitted at n ≈ 54, mirroring the joint ITT model's disabled LKJ residual correlation). The robust primary.
- **LRP70 (shared-tempo factor):** adds a rank-1 shared child-level growth-tempo factor loading (positively) on every slope — the genuinely _joint_ layer: do the measures grow together, and does non-verbal ability predict the common tempo?

Both converge cleanly at reporting tier (6 chains × 6000; 0 divergences, R-hat 1.0, min ESS ≥ 2600).

## Results (reporting tier)

### Baseline non-verbal ability → growth rate (`gamma`, logit, per +1 SD block design)

Direction is read from `P(>0)` per the evidence-language policy (#179).

| Measure                        | `gamma` median | 90% ETI          | P(>0) | evidence         |
| ------------------------------ | -------------: | ---------------- | ----: | ---------------- |
| T — receptive grammar (TROG-2) |         +0.119 | [+0.025, +0.213] | 0.980 | strong           |
| E — expressive vocabulary      |         −0.017 | [−0.081, +0.046] | 0.326 | inconclusive     |
| R — receptive vocabulary       |         −0.035 | [−0.095, +0.024] | 0.169 | suggestive (neg) |
| W — word reading (EWRSWR)      |         −0.050 | [−0.250, +0.138] | 0.331 | inconclusive     |
| L — letter-sound knowledge     |         −0.158 | [−0.361, +0.047] | 0.103 | suggestive (neg) |

The one clear signal is **grammar**: higher baseline non-verbal ability goes with faster grammar growth (the 90% interval excludes zero). Vocabulary (R, E) and reading (W) growth rates are inconclusive; letter-sound growth (L) leans negative — plausibly a ceiling artefact (L has a 32-item ceiling that several children approach, compressing measured growth for higher-ability children).

### Baseline non-verbal ability → baseline level (`delta`)

Non-verbal ability is a **very strong** positive correlate of the baseline _level_ of vocabulary and grammar — R +0.190, E +0.228, T +0.234 (all `P(>0) = 1.00`, 90% intervals exclude zero) — and suggestive for reading (W +0.290, `P` 0.90) and letter-sounds (L +0.155, `P` 0.79). So ability tracks _where children start_ broadly, but (above) tracks _how fast they grow_ only for grammar. This is the growth-curve echo of #186's marginal-vs-incremental split.

### Mean growth trajectory (`beta`, population slope, logit per +1 SD age)

Every measure grows (all `P(>0) = 1.00`): reading fastest (W +1.16, L +1.08), then vocabulary and grammar (E +0.29, R +0.25, T +0.22) — reading is the actively-acquired skill across this window.

### Shared growth tempo (LRP70)

- **Loadings** are all positive (`P(>0) = 1.00`): R +0.10, E +0.13, T +0.17, W +0.37, L +0.33 — the measures' growth rates track a **common developmental tempo**, most strongly the reading measures.
- **Does baseline non-verbal ability predict that common tempo?** No: the correlation between each child's latent tempo and their block-design score is **−0.01 (90% interval −0.28 to +0.27; `P(>0)` 0.48)** — essentially zero.
- **LOO(LRP69 vs LRP70):** `elpd_loo` −3137 (core) vs −3139 (factor) — the shared-tempo factor does **not** improve out-of-sample fit, so the independent-core LRP69 is preferred on parsimony. The factor is retained as the interpretable "do they grow together?" read-out (they do), reported alongside and flagged as the more structure-dependent view.

## How to read this — an adjusted association, not a cause

Per the locked DAG (`notes/202606231600-dag-revision-consolidated.md`), block design is an **off-DAG, pre-randomisation child covariate**, and latent general ability (`GA`) is the unobserved common cause of block design, vocabulary and the other skills. So `gamma_k` / `delta_k` are **adjusted associations — never "non-verbal ability drives growth"**: block design is essentially an ability proxy and the association is confounded by `GA` (not point-identified). The child random intercept only _partially_ adjusts (`METHODS.md` § "Causal interpretation and its limits"). This is descriptive natural-history evidence.

## Caveats

- **n = 54**, single cohort; most intervals are wide — directional statements, not decisive effects (grammar's `gamma` is the one interval to exclude zero at 90%).
- **Adjusted association**, `GA`-confounded, not causal (above).
- **Linear growth in age** is the identifiable choice at four waves; it cannot capture curvature.
- **Ceilings**: letter-sounds (L) and grammar (T) both have a 32-item ceiling that some children approach, which compresses measured growth and can bias slopes at high ability.

## Follow-ups (planned, gated)

- **Byrne-cohort replication:** the same growth vehicle can run on the Byrne five-wave panel, but is gated on #164 decision 3 — only `basread`'s ceiling is confirmed (`bpvs`/`basmat` unconfirmed → no responsible Beta-Binomial denominator), `basmat` (the non-verbal analogue) is **wave-3+** so there is no baseline non-verbal measure, and Byrne has no expressive-vocabulary measure.
- **Curvature / nonlinear trajectories** and a population-level age GP mean trend are possible enrichments beyond the linear identifiable core.

## Files

- `statistical_models/lrp69.py`, `lrp70.py` — the two growth models (`kind="growth"`, auto-discovered).
- `statistical_models/factories.py::build_growth_model`, `pipeline.py::fit_growth`, `reporting.py::growth_association_summary` — factory, pipeline, and the per-(coef, outcome) read-out.
- `statistical_models/preprocessing.py` — `load_wave_panel(baseline_covariates=…)` broadcasts a t1-only baseline covariate per child.
- `docs/models/lrp69/index.qmd`, `lrp70/index.qmd`, `docs/models/_partials/_results_growth.qmd` — reports.
- `tests/statistical_models/test_growth_models.py` — loader, summary read-out, factory-build.
