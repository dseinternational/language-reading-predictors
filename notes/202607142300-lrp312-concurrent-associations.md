<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5).

# Concurrent conditional-associations family (LRP-CA-001, #312)

Date: 2026-07-14

Related: #312, #314 (descriptive-association workstream), #310/#325 (gain-family items-scale marginals; the per-measure-k caveat), #313 (the longitudinal factor-model follow-on), `dag/dag-language-reading.dagitty` (2026-07-10 revision).

## What was added

A new statistical-model family, `kind="concurrent"` / family code `ca`, and its first model `lrp-rli-ca-001` (word reading focal). At each timepoint it fits a between-child Beta-Binomial regression of the focal outcome's _level_ on the standardised same-wave logits of a core skill set (L, B, TR, TE, R, E), plus age and a group nuisance term — the mutually-adjusted concurrent association "at wave t, +n of a predictor is associated with +m of the outcome". Every coefficient is an association; the family makes no causal claim, so conditioning on contemporaneous (post-treatment) skill levels is intentional (unlike the level-factors family, which omits cross-skill terms to protect a causal group×time contrast).

## Design decisions (issue #312 recommendations adopted)

- **Per-wave separate fits, not stacked.** Four cross-sectional fits (one row per child per wave), reported side by side, rather than one stacked model with a child random intercept. The random-intercept version borrows strength but partially converts the coefficients into within-child quantities, changing the estimand; the pure between-child cross-section is the cleaner descriptive object at each wave. Realised as: loop the waves, fit `build_concurrent_model` on each single-wave subset of the `phase_mode="levels"` frame.
- **Primary wave for the standard artefacts.** The standard pipeline (trace.nc, convergence gate, PPC, prior/posterior plots) assumes one model, so the best-powered wave (most rows; ties → latest) is designated the primary fit that carries them; the other three waves and every bivariate (single-predictor, unadjusted) fit are sub-fits with their own `subfit_convergence` flag recorded in the CSV — exactly the `fit_adjusted` sub-fit pattern. The primary wave is recorded in `config.json` as `primary_timepoint`.
- **Group as a flagged nuisance.** Included only to absorb arm composition at each wave (`beta_group_nuisance`, a wide `Normal(0, 1)`), not reported as an association; the report and `config.json` flag it non-interpretable.
- **Regularising priors are load-bearing.** `Normal(0, 0.3)` per-SD slopes: n ≈ 53 with a strongly inter-correlated predictor cluster means the mutually-adjusted coefficients are collinearity-shrunk. The report carries the Table-2-fallacy caveat (each coefficient answers a different conditional question) and the regression-dilution caveat (observed-score predictors attenuate associations toward zero relative to the latent truth — #313 addresses this).
- **Floored measures (P, N) excluded as predictors.** Off-floor indicators are a possible later extension.
- **Missing predictor values mean-imputed** (0 on the standardised scale — PyMC cannot take NaN inputs); a predictor's realised variance shrinks with its missingness, biasing that coefficient toward zero (flagged, as in the horseshoe level model). Rows missing the focal _outcome_ are dropped.

## Items-scale marginals: the standardised↔raw-logit bridge

The reader-facing answer ("+n predictor items ↔ +m outcome items") is produced by a new `reporting.concurrent_marginals` helper (with a `ConcurrentTerm` descriptor). Because the concurrent model has **no interaction terms**, the perturbation is a scalar shift per posterior draw — `Δη = β · Δz` — so the helper is a small, self-contained pushforward rather than the per-observation interaction machinery of the gain family's `association_marginals`. The predictor enters the model standardised (`β` per-SD, a well-calibrated regularising prior), and the `+k items` row maps the mean-point items increment into standardised units via `Δz = (logit(p̄ + k/N) − logit(p̄)) / sd_logit`, where `sd_logit` is the same standardisation the factory applied. `k` is set **per measure** as `max(1, round(N / 10))` (L→3, B→1, TR/TE→2, R/E→17), so a fixed `+5` does not span 3 %–50 % of predictor scales that differ tenfold — the caveat I raised on the gain family's #325 (there still a fixed `+5`), applied here from the outset.

## Ids / wiring

`ca` is registered as an **embedded** family (own number space from `ca-001`, legacy `lrpca01`), like the recent `bx` family — added to `FAMILY_BY_KIND` and `_EMBEDDED_FAMILIES` in `model_ids.py`, to `KINDS` and a new `_CA` list in `definitions.py`. There is no central kind→fit dispatch: the spec module calls `fit_concurrent` directly (the family convention). Dev smoke fit: 4 waves (n = 53/53/53/51), primary t3; letter sounds carry the clearest adjusted association with word reading at t3/t4 (per-SD ≈ 0.38/0.46, P(>0) ≈ 0.99/1.0), attenuated below their bivariate counterparts as expected under collinearity shrinkage.

## Follow-on

#313 (longitudinal latent factor model) is the instrument that corrects the regression dilution this regression-style family cannot. The family's own extension is additional focal outcomes (letter sounds, taught vocabulary) once the word-reading model settles the pattern.

## Known doc drift (pre-existing, not fixed here)

`docs/models/README.md`'s "At a glance" total ("120 statistical models") and some per-family counts are stale relative to `definitions.MODEL_REGISTRY` (now 132), and the `block_exposure` family is missing from the at-a-glance table entirely. I added the concurrent row and detailed section but did not re-audit the other counts (a separate docs-sync pass; also flagged on #327).
