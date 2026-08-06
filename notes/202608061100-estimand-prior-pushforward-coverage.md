<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Generalising the estimand-scale prior pushforward past the treatment families (2026-08-06)

Records how the second half of #381 was resolved — the `prior_pushforward.csv` half, which had stalled while the power-scaling half went in — and the four design decisions taken along the way. The power-scaling (`psense`) half is covered by [`202608050649-reporting-refit-predictive-checks.md`](202608050649-reporting-refit-predictive-checks.md) follow-up 10.

## Why it stalled

The existing helper is ITT-shaped. `reporting.prior_pushforward` takes a treatment indicator `G`, an item denominator and a coefficient name, and pushes the prior through the toggle-everyone average marginal effect. That is the right estimand for exactly the four families that already had the file — ITT (`tau`), gain factors (`beta_trt`), DiD (`tau_t2`) and level factors (`b_grp_time[1]`) — and no other family reports a binary treatment contrast. So "call the helper for the remaining families" was never a real option: there was nothing to pass as `G`, and no shared quantity for it to compute.

The generalisation is the one `level_prior_pushforward` had already established for the level family: **the prior pushforward is the family's own posterior-estimand transform, run on the `prior` group.** Not a common approximation applied everywhere. That principle decides everything else below.

## What "the estimand" is, per family

| Family                 | Reported estimand                                                              | Transform reused                                       |
| ---------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------ |
| `aligned`              | items-scale cohort contrast (`beta_cohort`)                                    | the ITT toggle-everyone core — the term is binary      |
| `concurrent`           | per-predictor `+1 SD` adjusted association                                     | `concurrent_marginals`' forward shift                  |
| `dose_response`, `did` | `+1 SD` session-dose step (`mu_dose` / `beta_dose`)                            | the dose marginal's forward shift                      |
| `block_exposure`       | block-active exposure association (`delta`, off `eta_base`)                    | `block_exposure_summary`'s forward shift               |
| `adjusted`             | per-predictor `+1 SD` adjusted association                                     | forward shift                                          |
| `mechanism`            | the worked contrast: predicted items difference between two exposure quantiles | `mechanism_items_curve`, threaded with `group="prior"` |
| `horseshoe`            | per-predictor shrunk coefficient behind the `P(\|β\| > δ)` ranking             | forward shift, one row per predictor                   |
| `joint`                | one `tau` per outcome, each with its own item ceiling                          | `_joint_ame_draws`                                     |
| `historical_growth`    | between-group total-growth contrasts over the common window                    | `historical.growth_summary`, threaded with `group`     |
| `historical_joint`     | the same, one set per measure                                                  | as above, prefixed by measure                          |

Two of those are worth stating plainly, because they are the ones a reader would otherwise assume were mechanical.

**Mechanism** is the largest family (34 fits) and the one the prior-analysis review named first. Its deliverable is not a coefficient but a curve, and 15 of the 34 fits are HSGP (`f_mech`) rather than linear (`beta_mech`). Pushing a coefficient prior through would therefore have checked the wrong thing for nearly half the family, and nothing at all for the curve's shape. Instead `mechanism_items.mechanism_items_curve` gained a `group` argument and the check runs the _whole_ curve reconstruction — reference constant, moderator net-out, quantile interpolation — on the prior. That is precisely the review's concern: the GP amplitude prior is deliberately tight, so a flat fitted curve and a curve the prior could never have bent look identical until this check separates them.

**Horseshoe** reports a ranking by `P(|β| > δ)` with `δ = 0.1` on the logit/per-SD scale. On `lrp-rli-hs-001` the prior's own 89% range for a single coefficient is about **−0.68 to +0.52** — an order of magnitude wider than `δ`. So the ranking statistic starts high a priori for every predictor, and the fitted ranking is a statement about _relative_ shrinkage, not about whether any coefficient clears a threshold. That is worth having in the report rather than inferred.

## Decision 1 — the numeric schema does not change, labels are added by the writer

`blending_sensitivity.py` validates the released phoneme-blending bundle by recomputing `prior_pushforward` from the trace and comparing it, key by key, against the saved CSV. The comparison intersects the CSV's columns with the _returned dict's_ keys and coerces each side with `float()`, so any string key present in both would compare `False` and fail the bundle. `output/statistical_models/blending_link_sensitivity/blending_link_sensitivity.csv` additionally records a `scientific_artifacts_sha256` over `prior_pushforward.csv` itself.

So the four label columns (`estimand`, `estimand_label`, `role`, `scale`) plus `status` / `reason` are attached by `reporting.labelled_pushforward` at the point the row is **written**, never by the transform. `prior_pushforward()` returns exactly the numeric keys it always did.

## Decision 2 — the 63 pre-existing rows are not rewritten

Following from decision 1, the ITT rows are hash-pinned in a release-gate-interlocked bundle, and rewriting them for a cosmetic column would invalidate it. Adding labels to DiD, gain-factor and level-factor rows but not ITT would be worse than adding them to none, so the pre-#381 rows stay byte-identical and `_priors.qmd` falls back to its original sentence for them.

That fallback is not a compromise: all four pre-existing families report a **treatment effect on the items scale**, which is exactly what the original sentence says. The sentence was only wrong for the families that had no row at all.

## Decision 3 — the renderer must name the estimand, not assume it

`_priors.qmd` previously printed "the prior on the **treatment effect** implies an items-scale **average marginal effect**" for whatever row it found. Extending coverage without changing that would have published, on nine aligned reports, a claim that the cohort contrast is a treatment effect — the one thing the aligned family's design note is emphatic that it is not. The partial now reads `estimand_label`, `role` and `scale` off the row, renders several quantities as a table rather than promoting one to headline, and keeps the old sentence only for rows that carry no labels.

## Decision 4 — an unavailable check is written down, not skipped

The meta-finding behind #381 is that a missing artefact reads as a clean one. Every previous pushforward call site was wrapped in `try/except` that printed a yellow warning to a terminal nobody re-reads and wrote no file, so the rendered report was identical to one whose prior had been checked and found harmless.

Every family now writes the file either way. A check that cannot run produces a row with `status = "unavailable"` and a reason, and the partial prints that reason in the report.

**No fit in the current suite produces one.** All 282 emitted rows across 150 fits are `status = "ok"` — every registered aligned model sets `use_cohort=True`, so the pooled-cohort branch that would emit the first real unavailable row never fires today. The branch is a guard for a future variant and for a transform that fails on a fit it should have handled; the unit test is what exercises it, not the suite. Worth stating plainly, because "the unavailable path is implemented" and "the unavailable path has been seen to work on real data" are different claims and only the first is true.

## What the check found on the first run

The point of #381 was to convert "no flags" from _unverified_ to _measured_, so two results are worth recording rather than leaving in 150 separate reports.

**Historical growth is overwhelmingly data-driven.** On `lrp-rlm-hg-001` the prior permits a between-group total-growth contrast anywhere in roughly **−77 to +73 items**; the posterior for the same contrast is **14.9 to 22.3**. The data narrow the interval by something like a factor of twenty, and the prior's own median sits near zero with no directional pull. The between-group growth findings in that family are not being carried by their priors.

**The horseshoe ranking threshold sits well inside its own prior.** `δ = 0.1` on the logit/per-SD scale, against a prior 89% range of roughly **−0.68 to +0.52** for a single coefficient. `P(|β| > δ)` therefore starts high a priori for every predictor, which means the ranking is informative about the _order_ of predictors and much less so about whether any one of them clears a threshold. That is a real qualification on how the horseshoe deliverable should be read, and it was not visible before this check existed.

The mechanism family's prior-versus-posterior comparison is genuinely per-fit — the worked contrast depends on each fit's own exposure distribution — so it is left to the reports, which now show both sides. For the HSGP fits the prior worked contrast runs to about ±3 items, which is not obviously wide next to the curve rises being estimated; that is the review's "deliberately tight GP amplitude" concern, now measurable per model instead of argued in general.

## Not in scope, and why

These are not "not done yet" — each needs an estimand defined that this change does not define, and one of them needs a different scale entirely:

- **`mediation` / `mediation_multi` (19 fits)** — the estimand is an NDE/NIE decomposition produced by g-formula counterfactual simulation, not a transform of a stored linear predictor. Pushing the prior through it means running the simulation on prior draws, which is a distinct piece of work in `mediation.py`.
- **`survival` (2)** — reports on the hazard scale; the items columns of this schema do not apply, and inventing an item denominator for it would be worse than leaving it uncovered.
- **`lcsm` (5), `growth` (3)** — latent-change and latent-trajectory quantities with no `eta` in the prior group; their estimands are on the latent scale.
- **`corr_factor` (4), `long_corr_factor` (1)** — #381 asks for an **indicator-scale** prior-predictive check for the measurement/CFA families specifically, which is a different artefact from this one.
- **`gain_factors` treated-only companions (8, `gf-101`–`108`)** — genuinely **not applicable**, on the same ground the robustness release gate excludes them (#482): the treatment indicator is constant in a treated-only fit, so `beta_trt` is not in the model and there is no causal term whose prior could be pushed anywhere.

The historical families (`historical_growth`, `historical_joint`) were nearly left here too, on the ground that `historical.growth_summary` computes its group intervals with panel subsetting rather than from a stored coefficient. They are in scope after all: threading `group` through that function is the same one-line change `mechanism_items` needed, and the resulting check — how much between-group total growth the priors alone permit — is worth having on a descriptive family whose whole deliverable is a growth comparison. `_summarize` gained `q25` / `q75` so those rows carry a real inner 50% band rather than a blank.

## Coverage

63 of 194 fits before; **150 after** — the 63 unchanged, plus 87 newly covered across eleven families. The remaining 44 are the four groups above, of which 8 are not applicable and 36 need an estimand this change does not define.
