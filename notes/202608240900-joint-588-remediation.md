<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Joint-family remediation: decisions taken for #588 (2026-08-24)

Preliminary research data and models — all conclusions remain provisional.

- **Tracking issue:** [#588](https://github.com/dseinternational/language-reading-predictors/issues/588)
- **Audit note:** `notes/202608231145-joint-statistical-model-audit.md` (Codex/GPT-5, PR #590)
- **Scope:** the `joint`, `joint_mechanism` and `historical_joint` families — `lrp-rli-itt-012/015/016/115/215/216/315`, `lrp-rli-jm-001/002`, `lrp-rlm-jc-001/002` — plus the shared release, reporting, influence, diagnostics and preprocessing code they run through.

The audit note records the diagnosis. This note records what was **done** and why, and where a decision differs from the audit's first recommendation.

## Independent verification first

Every finding was re-derived against the code and the stored reporting-tier fits before anything changed. The audit is accurate; nothing in it was a phantom. Reproduced exactly:

- **Finding 1.** The `dependence_note` written into `config.json` for all six contrast fits, and the `.qmd` prose for all six reports, still said the point estimate "should agree" / "is unaffected" / agrees "to Monte Carlo error". The 215/216/315 module **docstrings** had already been corrected by the 2026-08-22 ITT audit and now contradicted the machine-readable note beside them.
- **Finding 2.** More severe than stated. The stored `resolved_run_plan.contrast` for `015`, `016` and `115` has **no `dependence_companion` key at all** — every parent artefact predates the field — so `_joint_dependence_companion_note` short-circuited on all three and the pairing gate was **dormant on exactly the fits it was written for**. The binding it did apply checked five things (sibling directory, publishable, readable config, self-reported id, `data_sha256`); a companion fitted on different outcomes, the reversed contrast, different rows, different sampling settings or a different commit would have satisfied it.
- **Finding 3.** `_kf_build_joint_mechanism` selects the lead wave with `_kf_most_resolved_row` over every converged wave; the anchor is chosen by row count, and nothing labelled the selection.
- **Finding 5.** `sigma_within_prior_sigma` is `0.5` in `lrp_rlm_jc_002.py` and appears nowhere else at `1.0`. No registered companion, pipeline branch or sub-fit sets it.
- **Finding 7.** The dev fit reproduces the audit's numbers exactly: **153 rows / 53 children**, 152 observed cells each for `W` and `N`.
- **Finding 9.** `tau_difference_summary` had no `row_mask` parameter at all, so the influence path could not have recomputed the contrast even if asked.
- **Finding 11.** `df[wave_c].astype(int)` and `df[grp].astype(int)` run **before** any validation, the count guard tested only `max > n_trials`, and no check required a child to stay in one group. The factory then casts with `df[m].to_numpy(dtype=int)`.
- **Finding 12.** `_blending_pair_release_failures` returns `()` for `kind != "itt"` (bar `level_factors`), and `lrp-rli-itt-012` carries `B`.
- **Lower-priority.** `_prior_predictive.qmd` looked only for `prior_predictive_check.png` while historical joint writes only `prior_predictive_check_{measure}.png` — so it rendered nothing and then printed "_No prior-predictive plot available_". `_diagnostics.qmd` rendered only the unsuffixed PPC overlay (the calibration panels already globbed). `UR` carries `n_trials_confirmed=True` while `itt-115`/`315` still called it "not documented". `resolve_joint_run_plan` accepted a one-outcome LKJ declaration.

## Decisions

### Finding 1 — the correlated companions' estimand

**Decided: keep the hierarchical model, name the estimand, delete every invariance claim.** The audit offered a second route (a paired bootstrap or randomisation analysis targeting the parent's own marginal). That is a larger piece of new machinery, and the empirical case for needing it is weak: on the three registered pairs the contrast medians move by 0.00014, 0.00105 and 0.00063 on the proportion-correct scale. The defect is that the documentation asserted a mathematical guarantee where there is an empirical observation. So the run plan now records the companion's target explicitly — a **finite-sample latent-conditional standardisation**, not a new-child population marginal and not the parent's quantity with its covariance corrected — and states that the two are separate estimands under different hierarchical models. Six module `dependence_note`s and six report templates were corrected.

### Finding 2 — binding and the contrast's own robustness

**Decided: derive the requirement from the registered constant, bind nine fields, and assess the block through the contrast.**

The requirement now comes from `joint.JOINT_DEPENDENCE_COMPANIONS` first and the stored plan second — the same idiom `_blending_pair_release_failures` already uses with `BLENDING_LINK_MODELS`, and the only way to reach the three stored parents whose plans predate the field. A drift test in `test_joint_run_plan.py` fails if the constant and the module declarations disagree.

`_JOINT_PAIR_BINDING` checks ordered outcomes, contrast direction, the fitted equation's precision terms, the PSIS-LOO unit, the input checksum, the fitted-row identity, the fitted-data digest and observed denominators, the sampling configuration and the source commit — every one of which is **already persisted** in `config.json`, so no refit was needed. A field absent on either side fails closed: absence is not evidence of a match. The companion must also declare `use_residual_correlation=True`, so a factorised sibling cannot pose as the dependence model. Nine tamper tests plus four unrecorded-field tests cover it.

On robustness: the audit asked for the declared average-marginal-effect difference to be assessed rather than the component `tau` vector. Power scaling cannot reach it — the contrast is computed from draws post hoc, not sampled as a Deterministic — so requiring a psense row would have meant adding it to the model graph and refitting. Instead the pairing now measures **what the dependence model does to the declared contrast**, reading both fits' `tau_difference.csv` and recording median shift, direction-probability shift, interval-width ratio and a sign-flip flag into `release_decision.json` under `dependence_contrast`. A qualifier attaches only when the conclusion changes: a sign flip, or a shift in `P(> 0)` of at least 0.05. This is the audit's own preferred framing — assess the block through its consequence for the contrast rather than requiring every nuisance correlation to be sharply identified, which at n = 53 it never will be. Measured on the three stored pairs the shifts are 0.0061, 0.0062 and 0.0058, so the threshold flags a change of reading rather than the noise between two independently sampled fits.

All three stored parents now activate the check and **pass** it.

### Finding 3 — `jm-001` wave selection

**Decided: label, not refit.** The issue permits either option; giving every wave a persisted trace and a full diagnostic bundle means refitting `jm-001` and changing what the release gate covers. The findings box now carries an explicit **exploratory wave selection** note saying the lead wave was chosen after seeing all of them, that the across-wave comparison is exploratory, that the range rather than the lead value is the result, and that only the diagnostic-anchor wave receives the full primary-fit lifecycle — naming the sub-fit checks the others do get. Making the full bundle release-gating for every wave remains open under [#591](https://github.com/dseinternational/language-reading-predictors/issues/591).

### Finding 4 — joint-mechanism interpretation

**Decided: relabel everywhere the claim appears.** `delta_ls_decoding` is now "operational test-score slope contrast … not a construct-level decoding-specificity measure" in the term label, the key findings, the direction sentence and the report, with the reason stated beside it: the two tests differ in item count (79 vs 6), score distribution, discrimination, reliability and floor/ceiling behaviour, and the model calibrates them to no common latent outcome scale, so one shared ability loading differently on them produces a non-zero contrast on its own. `share_retained` becomes "ratio of adjusted associations"; the pathway language ("runs through the decoding channel") is gone. The `jm-001` module's claim that this fit and the `ca-010`/`ca-011` sensitivities bracket one underlying answer is replaced by a statement that they differ in likelihood, missing-predictor treatment, conditioning target, fitted rows and estimand, so bracketing is not supported.

### Finding 5 — the `jc-002` wider-prior sensitivity

**Decided: register the companion, and mark the claim preliminary until it is fitted.** Withdrawing the requirement was the other option, but the prior is not a nuisance here: `omega_m` decides whether each measure clears the 0.05-logit resolvability threshold and therefore which correlation pairs are interpretable at all. `lrp-rlm-jc-102` is `jc-002` built with `dataclasses.replace` so only `sigma_within_prior_sigma` can differ (verified: the two resolved plans differ in exactly `model_id` and that field). The parent's report no longer says the sensitivity "checks" anything — it names the companion, says it has not been run, and says the prior-robustness claim is preliminary until it has passed its own gate. The new report explains why power scaling is not a substitute and forbids pairing draws by draw number across two independently sampled fits.

A **dev-tier fit of `lrp-rlm-jc-102` completed end to end here** (exit 0, 23 artefacts), so the companion is runnable — the audit's own attempt aborted during Numba/PyTensor compilation and it treated that as environment-limited rather than a model defect, which this confirms. At dev tier the resolvability pattern under the wider prior matches the parent's (only `basread` clears the threshold), but a dev fit is not publishable evidence. **The fit has not been run at reporting tier**, and that is the one acceptance item this PR does not close.

### Finding 6 — LOO-PIT holdout unit

**Decided: label, not rebuild.** The audit's own first option. The figure title now reads "Conditional leave-one-cell-out PIT calibration (…) — the child's other cells remain observed", and both report partials carry a _Holdout unit_ paragraph saying these plots and the stored child-level PSIS-LOO answer different questions and neither validates the other. A genuine child-level calibration diagnostic needs exact leave-child-out predictive draws or an explicitly grouped construction; that is new machinery and is not in this PR.

### Finding 7 — the `jm-002` comparison population

**Decided: quantify and relabel, not re-fit the comparators.** `config.json` now records `comparator_population` — fitted rows, children, observed cells by outcome, the baseline rule, the exposure-standardisation consequence, a `comparison_status` of `approximate_not_like_for_like`, and the fitted-subject digest. The report's "like-for-like" bullet is replaced by the row counts and the explicit statement that a difference between this Δ and the separate-fit Δ is **not** attributable to the dependence treatment alone.

### Finding 8 — the historical-joint LOO reason

**Decided: correct the reason and the recorded unit; do not implement grouped K-fold here.** `loo_reason` no longer says multiple likelihood nodes make LOO undefined — it says no prediction target has been defined and implemented, notes that the nodes share an observation coordinate and could be summed per child-wave row, sets out what each candidate target would have to integrate, and records that an exploratory PSIS probe of older stored traces was unreliable and is not reported. `loo_unit` changes from `not_defined_multiple_likelihood_nodes` to `undeclared_prediction_target_not_implemented`. Implementing grouped child-level K-fold or exact refits is a separate piece of work.

### Finding 9 — influence and the declared contrast

**Decided: implement.** `tau_difference_summary` gains `row_mask`, and `_joint_contrast_influence` recomputes the declared contrast on all three influence populations — primary over all children, primary restandardised over the retained children, exact refit — with the same `total = composition + refit` decomposition the marginals use, plus a sign-flip flag. The columns are constant across the per-outcome rows because they describe one contrast per fit. Generic joint influence is not a publication-gate criterion and this does not make it one.

### Finding 10 — the `share_retained` ratio

**Decided: implement the rule, and add a denominator-free companion.** Two instability routes are checked on the posterior: the denominator `beta_mech[focal]`, and the held-fixed outcome's residual scale `sigma_u_resid[held]`, which divides the conditional slope. The rule mirrors the historical-joint residual-scale convention exactly — 0.05 logit, 95% support — because that convention is already established in this repository. The posterior **mean** is withheld unconditionally for this term: a ratio's mean is a property of the draws. When the rule fails, the summary is blanked in the CSV, the findings box says the ratio is withheld and why, and the report says so too. `abs_slope_reduction` — the per-draw `|beta_focal| - |beta_focal_given_held|` — is published either way, taken from the two slopes directly rather than reconstructed through the ratio so it inherits none of its behaviour. On the current data all four waves are **stable** (denominator support 0.998–1.000, scale support 1.000).

### Finding 11 — historical panel validation

**Decided: implement, pre-cast, and fail loudly.** `_require_finite_integers` rejects non-numeric, non-finite and fractional values before any integer cast; the count guard now tests `0 <= y <= n` rather than the ceiling alone; wave and group codes are validated before `astype(int)`; and a child appearing in more than one group is rejected. `NaN` remains the documented missing-data route and is untouched — two tests assert that permitted core and extension missingness still loads. Eight new failing-input tests cover fractional, negative, infinite, fractional-index and group-changing inputs.

### Finding 12 — the phoneme-blending policy's scope in a joint fit

**Decided: record the scope and verify it, rather than extend the withhold.** The audit offered both. Withholding a ten-outcome fit because one row's companion is stale would destroy nine sound results to protect a row that is not the model of record — the same reasoning the dependence pairing already applies. So `resolve_joint_run_plan` records `link_sensitivity_scope` whenever `B` is among the outcomes, stating that the 008/108 pairing governs the B model of record and that a joint `B` row is a secondary structural cross-check, not independently release-qualified and unable to supersede that conclusion. `_joint_blending_scope_note` **verifies** the sibling bundle — same input data, release-ready — and attaches a note saying the joint B row must not be read as a blending treatment claim when it cannot. Previously the findings box asserted the condition in prose and nothing checked it. On the current data the bundle is ready, so `itt-012` attracts no note.

### Lower-priority corrections

Rendering: `_prior_predictive.qmd` and `_diagnostics.qmd` now glob their measure-suffixed siblings, so the three historical-joint prior-predictive plots and the BPVS / digit-recall PPC overlays render. Ordering: `_results_historical_joint.qmd` leads with the within-child section when a within-child summary exists — `jc-002`'s declared estimand, which the Results section previously demoted to a "companion headline" while the Key Findings box led with it — and the between-child matrix follows as "(secondary)". Median convention: the pair guidance and the growth rows now point at `median` / `q50`; `mean` is retained for the correlation _matrix_ with the reason stated (entrywise averaging keeps it positive semidefinite, entrywise medians would not) and for cell-average calibration, where a mean is the quantity being calibrated. Prose: `_cell_values` no longer claims per-child offsets "cancel exactly" — matching children holds composition fixed, but this is an items-scale quantity through the inverse logit and in `jc-002` the wave-specific departure does not cancel even on the latent scale; the within-scale residual is described as carrying **extra-Binomial occasion-specific** measurement noise, since ordinary Binomial item-sampling variation is still represented by the likelihood. Selection: both historical-joint key-findings branches now carry an exploratory pair-selection note, without implying multiplicity adjustment. API: `resolve_joint_run_plan` rejects `use_residual_correlation` with fewer than two outcomes. Coverage: `ppc_summary.csv` gains per-outcome rows for joint-mechanism fits, rendered as their own table, with `ppc_coverage_markdown` excluding them from the pooled sum so totals are not doubled. Robustness: the joint-mechanism factory re-checks adjuster variance on the **final** fitted rows and raises rather than fitting an unidentified coefficient — it fails loudly instead of dropping the column, because a silent drop would make the recorded `effective_adjustment` wrong.

The `UR = 12` denominator prose in `itt-115`/`315` now says confirmed, matching `measures.py`.

## Verification

- Full suite: **2572 passed**, then the new tests added. `ruff`, `prettier` and `cspell` pass.
- Release decisions recomputed over the six stored reporting fits: `015`, `016` and `115` now activate the pairing, bind on all nine fields and pass, recording `dependence_contrast` with the measured shifts above; `215`/`216`/`315` keep their prior-dominance qualifier; `itt-012` attracts no blending note because the bundle is ready.
- Release decisions recomputed over **all 242 stored fits**: zero errors and **zero status or publishable changes** attributable to this work. One fit does differ from its stored decision — `lrp-rli-lf-006` now withholds because `lrp-rli-lf-106` has never been fitted — but that is the merged #595 level-family blending gate acting on a stale stored decision, and it reproduces identically with this branch's changes stashed.
- Dev fits of `lrp-rli-jm-001`, `lrp-rli-jm-002` and `lrp-rlm-jc-102` exercised the new pipeline paths end to end. `jc-102` also confirmed the rendering fixes: it writes three `prior_predictive_check_{measure}.png` and three posterior-predictive overlays, all of which the partials now render and none of which they rendered before. `jm-002` reproduces 153 rows / 53 children and 152 cells per outcome. The per-outcome coverage split is immediately informative: at the 50% level `W` covers 71% and `N` 83% against a pooled 77%, which is exactly the outcome-specific difference the pooled figure hides.

## What this PR does not close

1. `lrp-rlm-jc-102` has not been fitted at reporting tier, so the parent's prior-robustness claim stays preliminary — as its report and the catalogue now say.
2. A genuine child-level (new-child) predictive calibration diagnostic for the joint families, and a target-specific out-of-sample assessment for historical joint. Both are labelled honestly instead.
3. Simulation studies of the contrast estimator under zero, positive and negative within-child dependence.
4. Giving every `jm-001` wave a persisted trace and full release-gated bundle — overlaps [#591](https://github.com/dseinternational/language-reading-predictors/issues/591).
5. **No refits.** Every change here is either code, prose, or a release decision that recomputes over a stored fit. The affected stored fits should have `release_decision.json` and `key_findings.json` regenerated so they pick up the new binding record and the corrected prose; the `dependence_note` and `link_sensitivity_scope` in `config.json` need a refit to update, since they are written at fit time.
