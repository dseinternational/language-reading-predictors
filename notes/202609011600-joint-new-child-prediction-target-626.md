<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# The joint families predict a new child, and two estimators say how well

- **Date:** 2026-09-01
- **Status:** scientific decision, implemented
- **Issue:** #626 (deferred from #588 work items 7 and 9)

## Decision

1. **The declared out-of-sample prediction target for every joint family is a new child** — a child the model has never seen, in a replicate cohort of the realised size and group composition, at the observed covariate values. It is declared as `prediction_target` on `joint`, `joint_mechanism` and `historical_joint`, recorded in `config.json` under `resolved_run_plan`, and stated in `model_recipe.md`. The alternative target — a new occasion for an already observed child — is a different holdout with a different latent treatment; it is named in `PREDICTION_TARGETS` and **refused**, so a family cannot declare it and quietly receive the new-child machinery under the wrong label.

2. **Children are the sampling unit; waves are not.** The three families draw children from a population and observe them on a fixed, balanced wave design. Generalisation therefore means another child, not another occasion, and the child-aggregated LOO unit these families already declared was always claiming exactly this.

3. **A child-level latent must be integrated out, and any that is not declared fails the fit.** Each family declares which of its free random variables are child-indexed. `verify_child_latents` checks the declaration against the built model and refuses when a free variable carries a declared child dimension and was not declared. This is the defect the issue is about: leaving such a variable at its fitted value makes the answer conditional on the held-out child's own data while the label still says leave-one-child-out.

4. **Two estimators, one question.** Importance sampling on the integrated term where its diagnostics allow; grouped child-level K-fold refits where they do not. Both target the same declared quantity, and a fit publishes at most one.

5. **An estimate is withheld unless both of its error sources are small.** Pareto-$k$ above `good_k` withholds it, as #626 requires. So does a latent integral whose own Monte-Carlo error, summed over children, exceeds the ELPD's standard error — see the measurements below for why that second gate is not hypothetical.

6. **The calibration diagnostic's holdout unit is the child.** Each measure's per-child total gets a randomised PIT under the same new-child predictive. The existing per-outcome LOO-PIT plots stay, still labelled _conditional leave-one-cell-out_; the reports now say plainly which is which and that neither validates the other.

## Why the replicate-cohort framing settles `historical_joint`

The obstacle recorded in `loo_reason` was real: this family's subject offsets are group-centred and, in `jc-002`, its within-child departures double-centred, both **over the realised sample**. There is no closed-form marginal population distribution for one new child's latent, so "draw it from its population distribution" was not a well-defined instruction.

Declaring the target as a child in a _replicate cohort of the realised size_ dissolves that. The re-draw is performed by the model itself at that size, so the centring is applied to the fresh draw exactly as it is applied to the fitted one. Nothing has to be derived, and nothing is assumed.

Mechanically, the re-draw is forced by **removing** the latent from the posterior handed to `pm.sample_posterior_predictive`. Naming it in `var_names` is not enough — PyMC treats a variable it finds in the trace as given and returns the fitted values unchanged, which is the silently-conditional answer this work exists to prevent. The engine checks that every declared latent actually came back re-drawn and fails the run if one did not.

## The measurements behind decisions 4 and 5

Probes against the stored 2026-09-01 reporting traces, with the child latents integrated out as the new-child target requires.

**`joint` (RLI), `lrp-rli-itt-215`, LKJ residual block on.** At 7 200 posterior draws: ELPD −231.5 (SE 7.3), max Pareto-$k$ 0.32, half-split integration error 0.005 nats per child at 64 latent draws and 0.009 at 16. The two-dimensional residual settles quickly, and the estimate is publishable.

**`joint` (RLI), `lrp-rli-itt-012`, no residual block.** With no child latent the integrated term _is_ the conditional one, and the computation reproduces the stored fit's Pareto-$k$ maximum of 0.7005 exactly at the full 36 000 draws. That agreement is the engine's correctness check as much as it is a result. It also shows what thinning costs: the same fit's maximum $k$ read 0.98 at 1 500 draws, 0.65 at 4 000 and 0.769 at 8 000, while the ELPD moved by less than 0.4 across all of them. The shape estimate is a tail quantity; the ELPD is not.

That is why a latent-free design is **not thinned**. The integral over an empty set needs one pass, so the whole posterior costs a single log-likelihood evaluation (11 s here), and thinning would publish a _different_ Pareto-$k$ for the same estimator the fit already reports conditionally — 0.769 beside 0.7005, two numbers for one quantity in one report. The first reporting fit of this batch shipped exactly that before the policy was corrected; the identity now holds in the artefacts rather than only in this note. Where there **is** a latent the budget is real, because the cost is passes × draws, and the half-split diagnostic is what says whether the resulting integral is good enough.

**`historical_joint` (RLM), `lrp-rlm-jc-001`.** Unusable by importance sampling, and by a wide margin:

| latent draws | ELPD  | max Pareto-$k$ | children over `good_k` | half-split error |
| ------------ | ----- | -------------- | ---------------------- | ---------------- |
| 8            | −4904 | 18.3           | 71 / 71                | 2.28             |
| 64           | −2572 | 5.53           | 34 / 71                | 0.17             |
| 256          | −2448 | 2.09           | 9 / 71                 | 0.089            |

Both failure modes are present and independent. The integral is still moving at 256 draws — a naive draw from the population is a poor proposal for a child whose whole multi-measure profile is being predicted — and even granting a perfect integral, a child contributing four waves across three correlated measures puts the leave-one-child-out posterior far enough from the full one that no smoothing rescues the weights. This is what decision 5's second gate exists to catch: at 256 draws the $k$ values alone would already have been improving, while the estimate was still drifting by tens of nats.

A useful cross-check fell out of building both estimators. The five-fold refit route puts `jc-001`'s new-child ELPD at −2417 (SE 47), against −2448 from the importance-sampling route at 256 latent draws — the two agree once the integral is given enough draws, and diverge wildly (−4904) when it is not. That agreement is evidence the K-fold implementation is scoring the right quantity, and the divergence is evidence for decision 5's gate. Held-out PIT values across the three measures come out close to uniform (medians 0.49 to 0.53, quartiles near 0.25 and 0.75).

Hence K-fold for this family. Each fold refits on its training children through the family's own factory, transplants that fold's **free** global parameters into the full model, re-draws the child latents, and scores the held-out children. Only free random variables are transplanted; the deterministics are functions of them and are recomputed for the full child set. Getting that wrong is not theoretical — the first run refused all five folds because `subject_offset`, a per-training-child deterministic, was being carried into a model with a row per child.

## Consequences

- `historical_joint` no longer records `loo_unit="undeclared_prediction_target_not_implemented"`. It records `loo_unit="child"`, the declared target, and a `loo_reason` that says PSIS is refused _because_ its diagnostics refuse it rather than because nothing was defined.
- New per-fit artefacts: `new_child_loo.csv`, `new_child_pareto_k.csv`, `new_child_pit.csv` and one `new_child_pit_<measure>` figure per measure; for the K-fold route `new_child_kfold.csv`, `new_child_kfold_pointwise.csv`, `new_child_kfold_pit.csv` and its figures. The PIT and Pareto-$k$ tables are gate-visible diagnostics, like `pareto_k.csv`; the two ELPD summaries are withheld with the other results when the convergence gate fails.
- K-fold folds are recorded as `cross_validation` sub-fits in `subfit_provenance.csv`, so every refit behind a published ELPD is auditable with its own convergence verdict.
- A partial K-fold is reported but is **not** the declared estimate: which refits happened to work is a selection, not a smaller sample.
- Both `joint_mechanism` designs validate, including the `levels` one that computes no PSIS-LOO at all. Gating this on `compute_loo` would have withheld the validation from the design that needs it most: its saturated per-child residual is exactly what makes conditional importance sampling useless there, and exactly what this integrates away.
- Stored fits predating this work carry none of the new artefacts. The reports degrade to the declaration sentence alone, which is accurate for them. The numbers arrive on the next reporting refit of the twelve affected models — `lrp-rli-itt-012/015/016/115/215/216/315`, `lrp-rli-jm-001/002` and `lrp-rlm-jc-001/002/102`.

Two registered fits withhold their ELPD under this policy on first contact, which is the policy working rather than a problem to route around. `lrp-rli-jm-002` has two children over `good_k` — the same two its stored conditional PSIS-LOO already flagged. `lrp-rli-jm-001`, the saturated levels design, has twelve, which is what its own documentation has always said would happen and is now measured instead of asserted. Where the weights are refused the PSIS-weighted PIT inherits that, and the report says so on the figure rather than leaving a reader to carry the caveat down from the table. Extending the K-fold route to `joint_mechanism` would remove the caveat for those two fits; it is not done here because #626's scope is the declaration plus one working estimator per family, and the levels design's single-wave subset needs its own rebuild callback.

## Which estimator a family needs, measured

Integrating a child's latent out is not uniformly harder on importance sampling than leaving it fitted. Across the twelve reporting fits of the 2026-09-01 batch it ran in both directions, and what decides it is **how much of that latent the held-out child's own data was carrying**. Where a child contributes one row and a small residual, re-drawing that residual loses little, the leave-one-child-out posterior stays close to the full one, and PSIS gets _easier_. Where the latent is effectively a per-child parameter, it loses everything.

| design                                             | rows per child       | child latent                   | integrated $k$                   | conditional $k$           |
| -------------------------------------------------- | -------------------- | ------------------------------ | -------------------------------- | ------------------------- |
| `joint`, no residual block (`itt-012/015/016/115`) | 1 per outcome        | none                           | 0.31–0.70                        | identical by construction |
| `joint`, LKJ residual (`itt-215/216/315`)          | 1 per outcome        | 2-dim residual                 | **0.32–0.40**                    | 0.56–0.65                 |
| `joint_mechanism` transition (`jm-002`)            | 3                    | bivariate child intercept      | **1.14**                         | 0.98                      |
| `joint_mechanism` levels (`jm-001`)                | 1                    | residual _is_ the child effect | **6.12**                         | not computed              |
| `historical_joint` (`jc-001`)                      | 4 waves x 3 measures | stable (+ within) departures   | **2.09 at 256 draws, 18.3 at 8** | not computed              |

Read down the integrated column: it is the one belonging to the declared target, and the conditional column is shown only to give the direction of travel. The three LKJ-residual fits are the case where integration **helps**. `jm-002` is mildly worse. `jm-001` and `jc-001` are the cases where the child's own data was doing most of the work for that child's rows, and no smoothing survives removing it.

The practical guidance for a family added later: expect the importance-sampling estimator to serve a design whose child-level latent is low-dimensional and shared across few rows, and expect to need refits once the latent approaches one free parameter per child, or spans several waves and correlated measures. Pareto-$k$ and the half-split diagnostic between them say which case applies without anyone having to guess, and both fail closed.

## What the refit route delivered

All three `historical_joint` fits scored every one of their 71 children, with **every fold converged** (max R-hat 1.007, min ESS 1 203, zero divergences across all fifteen refits) and no fold refused:

| fit      | analysis rows | `elpd_kfold` | SE   | held-out PIT medians |
| -------- | ------------- | ------------ | ---- | -------------------- |
| `jc-001` | 284           | −2414.0      | 46.7 | 0.507, 0.523, 0.520  |
| `jc-002` | 213           | −1839.5      | 26.6 | 0.501, 0.544, 0.510  |
| `jc-102` | 213           | −1838.3      | 26.1 | 0.502, 0.544, 0.507  |

> [!WARNING]
> `jc-001` is **not** comparable with the other two. `within_correlation` requires one row per child and wave, so `jc-002` and `jc-102` fit 213 rows against `jc-001`'s 284. An ELPD is a sum over held-out units; different units, no comparison.

Three independent routes to `jc-001`'s number agree inside its own standard error — −2414 from the reporting K-fold, −2417 from the dev-config K-fold, −2448 from the integrated-PSIS probe at 256 latent draws — while the same probe at 8 draws said −4904. That spread is the argument for the half-split gate stated as a measurement: the estimator it refused was wrong by roughly 2 500 nats, and nothing in its Pareto-$k$ alone would have said so.

`jc-002` and `jc-102` **are** comparable — they differ only in `sigma_within_prior_sigma` (0.5 against 1.0) and share a fold partition. Paired over the 71 children, `elpd_diff = +1.13` with a **paired** standard error of 0.76 (the marginal standard errors, ~26, are the wrong yardstick for a contrast and are ~35x too large). That is inconclusive under the standing `|elpd_diff| < 4` rule, which is a pass: it is the predictive half of the sensitivity `jc-002`'s report has always promised, and it says the correlation conclusions do not detectably depend on that regularisation. Before this work the family had no out-of-sample quantity at all, so the comparison could not be made.

## What this does not claim

The new-child ELPD is a predictive summary, not a causal one. Nothing here changes an estimand, a prior, a likelihood, a fitted population or an adjustment set. `historical_joint` remains descriptive throughout, and a better predictive score for one joint specification over another is not evidence about a mechanism.

The importance-sampling estimator's latent integral uses draws from the population distribution, which is the honest but inefficient proposal. A proposal centred on the fitted latent would converge faster and might bring more families inside the PSIS route; that is a possible future improvement, not a defect in what is published, because the half-split gate withholds any estimate the current proposal cannot support.

Refs: #626, #588; `notes/202608240900-joint-588-remediation.md`, `notes/202608241500-joint-588-review.md`.
