> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# DiD family: #576 review remediation

**Date:** 2026-08-24. **Issue:** [#576](https://github.com/dseinternational/language-reading-predictors/issues/576). **Scope:** the eighteen registered `lrp-rli-did-*` declarations, the typed settings and run plan, both `build_did_model` branches, the family pipeline, the reporting and figure layers, the release gate, the prior-sensitivity runner and the family's tests.

The review's own verdict was that the core data coding, likelihood construction, indexing and posterior alignment are mechanically sound, and nothing here contradicts that. What the review found is that the family was not yet **publication**-correct: in several places the quantity gated, checked or described was not the quantity published.

## What changed

### 1. One named focal estimand per fit (finding 1)

`DiDRunPlan` now records `focal_estimand`, `focal_estimand_scale` and `focal_estimand_artifact`, persisted to `config.json` and printed in `model_recipe.md`. Every DiD headline is a natural-scale marginal, never a bare coefficient, so `focal_estimand_scale` is `"natural"` for the whole family.

The bite is in `release._standard_sweep_evidence`. Its sign-stability clause used to read `tau_logit_mean` — the swept coefficient — for every family. For LRPDID07 that is `mu_dose`, the hierarchical centre the per-period slopes `beta_dose_phase[p] = mu_dose + sigma_dose * z[p]` are drawn around, while the published headline applies each row's _realised_ slope through a nonlinear items transform averaged over treated rows with unequal per-period counts. Those are not the same estimand and their signs can diverge, so the fit could clear a robustness gate for one quantity and publish another. A fit whose plan declares a natural-scale focal estimand now has that clause read `items_mean`, the published estimand's own column. `tests/statistical_models/test_did_estimand_release.py::test_gate_follows_the_published_marginal_not_mu_dose` constructs both divergent cases and pins the direction.

Two supporting changes make "one estimand" true rather than aspirational. The sweep runner's items translation now calls the _same_ `dose_marginal_draws` the posterior marginal and prior pushforward use, instead of reimplementing the transform. And the pooled dose companions (LRPDID06, LRPDID107) now route through `write_dose_slope_summary` like LRPDID07, so `dose_marginal_summary.csv` means the same thing across the whole dose branch — before this they emitted only an inline prior pushforward and no marginal table at all.

A stored plan carries no `focal_estimand_scale`, so pre-#576 fits re-decide exactly as they did without a refit.

### 2. The phoneme-blending link pair, for this family (finding 2)

Phoneme blending is ten **three-alternative** forced-choice items, so chance is about 3.3 of 10; the ordinary inverse-logit score mean permits fitted means below that. The methodology has required a guessing-floor companion for any headline `B` interpretation since the ITT suite adopted it, and the ITT pairing roughly halved the item-scale estimate — this is material. The `did` family had no version of it, so LRPDID03 could publish an unqualified `B` headline.

Added: `score_mean_link` on `DiDModelSettings` and `build_did_model`'s arm-by-wave branch (with the empirical-Bayes intercept anchor inverted through the link, as the level family does — for a pooled t1 proportion near 0.49 the anchor moves from −0.03 to −1.16); the registered companion **LRPDID103**; `blending_sensitivity.evaluate_did_blending_link_pair`; and a `did` branch in `release._blending_pair_release_failures`. Requirement is derived from the registered pair **and** from the plan, so a future graded `B` arm-by-wave fit outside the pair fails closed rather than publishing unpaired.

The ITT companion does not stand in for this one: the arm-by-wave likelihood is longitudinal with a child random intercept, so t1 and t3 data inform the t2 posterior.

### 3. The t3 taxonomy corrected (finding 3)

The run plan, `METHODS.md`, `docs/models/README.md`, the report partial, the priors table, the key-findings box, the `contrast_status` metadata and the family docstrings all described `arm_gap_t3` and `delta_crossover` as **latent-ability-confounded associations**. That is wrong in both directions.

Original assignment is still randomised at t3. `arm_gap_t3` identifies the effect of assignment to the **early-start treatment history versus the delayed-start one**, under the same available-case selection and model assumptions as `tau_t2`. Latent ability does not become a confounder of a randomised assignment because the waiting list crossed over. What is genuinely unavailable is the **mechanism**: duration, carryover, maturation, ceiling effects and different taught blocks are inseparable in that one contrast. `delta_crossover` is correspondingly the change between two randomised regime contrasts, and cannot be attributed to waiting-list catch-up or read as the intervention wearing off.

The priors table gains a `regime` role for exactly this: identified by the design, but as a contrast between assigned schedules rather than between treated and untreated.

### 4. The t2 estimand signed off and tested under imbalance (finding 4)

Recorded separately in `notes/202608241100-did-t2-estimand-signoff.md`: the family keeps the arm-gap **level**, the soft prior-weighted baseline adjustment is documented, `arm_gap_t1_prior_sigma` / `sigma_child_prior_sigma` make the allocation variable, **LRPDID104** is the registered estimand-matched sensitivity, and three new tests exercise recovery under a material +0.45-logit realised baseline gap — where the previous recovery test simulated exactly zero, so the level and the change were indistinguishable.

### 5. LRPDID13's trajectory is fully marginalised (finding 5)

The population trajectory removed and integrated `u_child` but left the fitted `v_delta` waitlist-t3 deviations in `eta`, then labelled the result population-level. That contradicted `did_summary`, which correctly withholds LRPDID13's t3 items summaries for the same reason. `marginal_cell_probabilities` and `write_group_arm_trajectory` now take an optional second child-level random effect with its own row mask and scale, remove it and integrate over `Normal(0, sigma_delta)` — the two deviations being independent, the masked rows marginalise at `sqrt(sigma_child² + sigma_delta²)`. The written table records `marginalised_effects`, so "population-level" is checkable in the report rather than asserted.

### 6. Sensitivity bundles bound to the fitted equation (finding 6)

The sweep compared itself with the stored primary through model/outcome identity, the data digest, row counts and arm totals. None of those move when the likelihood, intercept anchor, age adjustment, random-effect choice or a prior width changes, so a primary fitted under an older plan could be released by a sweep generated under a newer one.

`DiDRunPlan.run_plan_digest` is a canonical SHA-256 over a fixed, closed list of **modelling** fields, taken with the same defaults a fresh resolution uses. Two consequences are deliberate: a stored plan written before a field existed digests identically to a fresh plan taking that field's default (so existing fits stay reproducible without a refit), and a prose revision — this change makes several — does not invalidate evidence for an unchanged equation. The digest is recorded on `PrimaryStandardReference`, on every sweep row, in the cell trace's stamped provenance, and re-checked by `attach_outcome_bundle`, `_validate_cell_trace` and `_standard_sweep_evidence`, so a stale bundle stops lifting the gate the moment the plan changes rather than only when someone re-runs the installer.

### 7. LRPDID07's report shows its own estimates (finding 7)

The partial displayed `did_summary.csv` whenever present, which suppressed `dose_slope_summary.csv` entirely — the table LRPDID07 exists to produce — then printed only its first row, using the posterior mean. Both tables now render, every dose row prints on the house median + 50 % + 89 % convention with a label saying what it is, and the published marginal prints beneath them, explicitly distinguished from the hierarchical centre.

### 8. No direction claim without posterior support (finding 8)

Every binary report asserted that "the immediate arm pulls ahead at t2 and the wait-list arm catches up", regardless of what the fit found. The sentence is now derived from `prob_tau_t2_pos` and `prob_delta_crossover_pos` and states the probabilities rather than the story.

### 9. Dose terms named for what the design supports (finding 9)

With `treated = (G == 1) OR (period == P2)` the four-cell fixed-effect design is saturated, so `theta_treated` at the mean treated dose is the crossover **cell** contrast `(waitlist P2 − waitlist P1) − (immediate P2 − immediate P1)`, not a separately identified current-treatment-presence effect. The P2 slope relates P2 sessions to the t3 period-end level conditional on t1 and is not a P2 _gain_ slope, because the treatment-affected t2 period-start score and prior P1 dose are omitted. Renamed in the run plan, the audit metadata, the priors table and the report.

### Lower-severity items

- DiD cell-PPC figure legends now render the configured `ci_prob` instead of a hard-coded 95 %.
- LRPDID07's audit metadata recorded `beta_dose`, which its posterior does not contain; it now records the parameters the fit actually has.
- Dose recipes printed the inherited `waves=(0, 1, 2)`; they now print `Periods: P1, P2`.
- A binary `waves != (0, 1, 2)`, a dose `waves` declaration (silently ignored), and an explicit `outcomes` tuple omitting the focal outcome now fail at settings/resolution time rather than after the output directory is reset and the panel is read.
- `tau_t2_prior_sigma=True` passed numeric validation and silently became `1.0`; every optional prior width now goes through one validator that rejects `bool` explicitly.
- Two-arm support in every arm-by-wave cell is checked at build time rather than surfacing during reporting. (Current data fill every cell.)
- The two DiD PPC figures now have one guard each, so the manifest cannot record a skip against the wrong filename.
- The findings note described LRPDID102 as "a variant with different adjustment"; it widens the `tau_t2` prior and nothing else.

### Material qualifications addressed

- **Dispersion prior.** `kappa ~ HalfNormal(50)` cannot reach the near-Binomial limit; at `n = 170` its prior median implies about 5.9× Binomial variance. `kappa_prior_family` / `kappa_prior_sigma` are now settings, and **LRPDID105** (18 items) and **LRPDID106** (170 items) are the registered low-/high-denominator sensitivities on the dispersion scale. The family **default is unchanged**, because every stored DiD fit was sampled under the concentration prior.
- **Repeated-measures covariance.** The cell PPC compares marginal cell means and zero rates, which a model with wrong within-child dependence can still reproduce. `did_within_child_ppc.csv` (new) compares observed within-child changes — their arm-specific means and their spread — and across-child wave-pair correlations with the same statistics recomputed on every posterior-predictive replicate, and the report says what a flag means.
- **Items-scale gap closure.** `did_summary` now also reports the gap change recomputed on the children observed at **both** t2 and t3, with a flag recording whether the two wave populations coincide at all; the report and the key-findings box prefer it, because the wave-specific version mixes the change over time with a change in composition when the row sets differ.

## Not done here, and why

- **A t2-only / alternative-covariance sensitivity.** The review offers "corresponding PPCs _or_ a t2-only or alternative-covariance sensitivity". The PPC is implemented. A two-wave DiD would need the factory's `waves == (0, 1, 2)` requirement relaxed, and the level-factor family already carries a registered t1→t2 window comparator (#584 decision 3) answering the same question — whether the randomised t2 conclusion survives without the longitudinal working model — for the same trial. Adding a second one here is a design decision, not a defect fix.
- **APT denominator sensitivity for `did`.** LRPDID14 doubles 40 partial-credit items to 80 half-mark units; its registered denominator sensitivity is ITT-specific. Recorded, unchanged.
- **Typed declarations for the existing eighteen modules.** The four new models declare `DiDModelSettings`; the eighteen existing ones still use the strict `extra` translator and record `settings_source="legacy_extra"`. Migrating them is mechanical, touches every module, and would obscure this change's diff. Safe as-is — a misspelled key fails before any I/O — and deferred.

## Refit implications

**No refits were run as part of this change.** Every code fix lands in `config.json`, `model_recipe.md`, the summaries and the report only on a future fit, and each fit directory's `_partials` copy refreshes at fit/render time. Specifically:

| What                                                                        | Fits affected                          |
| --------------------------------------------------------------------------- | -------------------------------------- |
| New registered models (never fitted)                                        | `lrp-rli-did-103`, `104`, `105`, `106` |
| Corrected t3 taxonomy in plan, metadata, key findings, report               | all 18 existing                        |
| Within-child PPC, common-population gap change, link field in `did_summary` | all 15 arm-by-wave                     |
| `run_plan_digest` (needed before any new sweep can attach)                  | all 18                                 |
| Focal-estimand record and the gate's estimand-scale sign column             | all 18                                 |
| `dose_marginal_summary.csv` newly written; dose term labels                 | `did-006`, `007`, `107`                |
| Fully marginalised trajectory                                               | `did-013`                              |
| Stale artefacts already outstanding from the 2026-08-20 review              | `did-007`, `did-013`                   |

`lrp-rli-did-003` cannot release until `lrp-rli-did-103` is fitted at the same tier and passes its own gate — the pairing is fail-closed, so the phoneme-blending DiD card is withheld in the meantime. Any existing `tau_prior_sensitivity.csv` bundle beside a DiD fit will stop lifting that fit's gate once the fit is refitted with a `run_plan_digest`, so the sweep must be re-run (`scripts/did_prior_sensitivity.py --config reporting --attach`) after the refits, not before.

## Verification

`uv run ruff check src/` passes. The family's own suites pass, including the real-NUTS recovery tests through the production factory, and 25 new tests in `tests/statistical_models/test_did_estimand_release.py` cover the estimand/gate, the blending pair, the run-plan digest and the trajectory marginalisation. All four new models resolve, load and build against the current data.
