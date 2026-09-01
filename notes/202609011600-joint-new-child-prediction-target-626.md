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

## What this does not claim

The new-child ELPD is a predictive summary, not a causal one. Nothing here changes an estimand, a prior, a likelihood, a fitted population or an adjustment set. `historical_joint` remains descriptive throughout, and a better predictive score for one joint specification over another is not evidence about a mechanism.

The importance-sampling estimator's latent integral uses draws from the population distribution, which is the honest but inefficient proposal. A proposal centred on the fitted latent would converge faster and might bring more families inside the PSIS route; that is a possible future improvement, not a defect in what is published, because the half-split gate withholds any estimate the current proposal cannot support.

Refs: #626, #588; `notes/202608240900-joint-588-remediation.md`, `notes/202608241500-joint-588-review.md`.
