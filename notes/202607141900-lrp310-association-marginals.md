<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8).

# Gain-factor per-covariate items-scale association marginals (#310)

Date: 2026-07-14

Related: #310, #127 (the LRPGF/LRPLF families), the treatment-marginal pushforward (`reporting.treatment_marginal_effect` / `_itt_ame_draws`), `docs/models/_partials/_results_factors.qmd`.

## What #310 asked

The gain-factor family reports its adjusted-association covariates (own baseline, age, cognitive ability, the `skill_symbols` baselines, the `adjust_for` confounders) only as logit-scale coefficients of standardised covariates in `factor_summary.csv`. Only the randomised treatment term got a probability/items-scale translation (`treatment_marginal.csv`). #310 asks for the same interpretable form for every covariate — "+1 SD in baseline letter sounds is associated with +m words of word-reading gain" — across all eleven outcomes. Reporting-only: no new models, no change to any likelihood or prior.

## What was added

A new `reporting.association_marginals` helper plus an `AssociationTerm` descriptor, wired into `fit_gain_factors`, writing `association_marginals.csv` on every gain-factor fit (the 11 stock models and the `…b` treated-only companions inherit it automatically). It reuses the treatment marginal's "net-out-and-toggle" pushforward idiom, specialised from a 0/1 treatment switch to a continuous covariate increment.

For each covariate, per posterior draw, the linear-predictor shift from a `+1 SD` perturbation is

    Δη_i = γ_c · (main_scale) + Σ_k γ_int_k · z_partner_{k,i},

and the probability-scale average marginal effect is `mean_i[ expit(η_i + Δη_i) − expit(η_i) ]`, reported on the probability and items (`n_trials` × probability) scales with 90 % and 95 % intervals.

**Scale bookkeeping (the one subtlety).** The own baseline and skill baselines enter the linear predictor on the **raw logit** scale, while their interactions use the **standardised** vector. So `main_scale` for those is the SD of the raw logit (a `+1 SD` move shifts the raw-logit input by that SD), whereas age and cognitive ability are standardised throughout (`main_scale = 1`). Because the interaction inputs are plain elementwise products of standardised vectors, a `+1` standardised shift changes each product by exactly the partner's standardised vector — so the interaction contribution is `γ_int · z_partner`. Treatment interactions are included in a covariate's marginal: the covariate marginal holds the treatment indicator fixed and perturbs the covariate, so a `trt × covariate` term does move with it. The pipeline reconstructs the exact standardised vectors the factory used from the fitted subset `built.prepared`, so the pushforward is on the same scale the model was built on.

**Per-k-items companion.** "+1 SD" is opaque, so for the bounded-count baselines a `+5 items` row is emitted, evaluated at the covariate's mean baseline proportion `p̄`: the raw-logit increment `logit(p̄ + k/N) − logit(p̄)` replaces the `+1 SD` shift, and the interaction scales by that increment divided by `main_scale` (the same shift in standardised units). Each `+1 SD` row also carries `sd_items`, how many items `+1 SD` is at the mean, so the two scales can be read against each other.

**Off-floor floor rule.** For the floored outcomes (P, N; `bernoulli_offfloor`) the items scale collapses to the off-floor probability delta (`n_trials = 1`), mirroring the treatment marginal, and the own baseline is dropped (it is not built on that path).

**Averaging population (pre-specified).** The covariate associations are descriptive, so the natural averaging population is **all fitted rows** (`row_mask=None`) — in contrast to the treatment marginal, which restricts to the randomised period-1 transition. That choice is recorded in `config.json` (`extra.association_marginals.averaging_population = "all_stacked_rows"`), not left implicit.

Every row carries `role = "association"` — none of these terms is causal, per the family's documented estimand structure. The `_results_factors` partial renders the CSV on the items scale under an "Items-scale associations — not causal" callout.

## Dev-fit reads (sanity, not reporting quality)

- `gf-004` (L, graded, `n_trials = 32`): own baseline `+1 SD` → **+4.6 letter-sounds** of gain (P > 0 = 1.00); ability `+1 SD` → +0.9 (0.98); age `+1 SD` → −0.8 (0.05). Every `+1 SD` sign matches its logit coefficient.
- `gf-005` (P, off-floor): items == off-floor probability delta exactly; own baseline row correctly absent; L `+1 SD` → +0.11 off-floor probability (1.00).
- `gf-101` (W, treated-only `…b`): `association_marginals.csv` written with no `beta_trt`; own `+1 SD` → +14.0 words (n_trials 79); the trt interactions are absent (dropped in treated-only), so no leakage.

Signs are consistent with the logit coefficients across all three; `items = n_trials × prob` holds; the guard test (`tests/statistical_models/test_reporting.py`) locks sign/scale consistency, the per-k-items mapping, the interaction pushforward against a hand-loop reference, and the off-floor path.

## Files

- `statistical_models/reporting.py` — `AssociationTerm` dataclass + `association_marginals`.
- `statistical_models/pipeline.py` — `_gf_association_terms` builder + the `fit_gain_factors` wiring (writes the CSV, records the averaging population in `config.json`).
- `docs/models/_partials/_results_factors.qmd` — the items-scale association table + caveat prose.
- `tests/statistical_models/test_reporting.py` — five guard tests.

## Scope note

A reporting enhancement plus a reporting-config refit sweep of the family to regenerate artefacts. No new models, no change to any likelihood or prior. This is preliminary, exploratory work in progress.
