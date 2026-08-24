<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Joint-mechanism remediation: verification and decisions for #591 (2026-08-24)

Preliminary research data and models — all conclusions remain provisional.

- **Tracking issue:** [#591](https://github.com/dseinternational/language-reading-predictors/issues/591)
- **Audit note:** `notes/202608231146-joint-mechanism-follow-up-review.md` (Codex/GPT-5, PR #592)
- **Scope:** `lrp-rli-jm-001`, `lrp-rli-jm-002`, the shared settings / factory / pipeline / release / reporting layers they use, the Tier-1 comparison and both reports

The audit note records the diagnosis. This note records what was independently verified, what was changed, and what deliberately was not.

## Overlap with the #588 joint-family batch

PR #609 (`notes/202608240900-joint-588-remediation.md`) landed first and addressed three of the same findings from the joint audit's side, choosing the _lighter_ option each time. This batch supersedes two of those and keeps the third:

- **Wave selection.** #609 chose "label, not refit": the findings box carried an "exploratory wave selection" note saying the lead wave was picked after seeing all four and that only the artefact-hosting wave got the full lifecycle. Since #591 every wave gets the full lifecycle and nothing selects a wave at all, so that note is removed rather than merged — it would now assert two things that are false.
- **The ratio.** #609 introduced one stability rule (`_jm_ratio_stability`: 0.05 logit, 95% support, checked on _both_ the denominator and the held-fixed residual scale), blanked an unstable ratio's whole row and added the denominator-free `abs_slope_reduction`. That rule is **kept as the family's only rule** — this batch's `conditional_slope_ratio.csv` reproduces its verdict rather than applying a second, competing one, and adds the three region probabilities beside it. A pull request that shipped two different stability thresholds for one quantity would have been a defect, not a merge.
- **Interpretation.** #609's "operational test-score slope contrast, not a construct-level decoding-specificity measure" wording is kept verbatim in the term labels and the headline; this batch's common-factor algebra (unequal loadings on one general ability produce a non-zero Δ with no causal route) is the mechanism behind that statement and sits beside it.

Two of #609's additions are complementary and untouched: per-outcome rows in the _conditional_ `ppc_summary.csv` (this batch adds them to the _marginal_ companion, on the same null-pooled-row convention), and `comparator_population` in `jm-002`'s `config.json` (this batch adds the exposure scaler and the cross-model `comparable` verdict).

## Independent verification first

Every checkable claim was re-derived against the code, the two stored reporting-tier fits and the current data (SHA-256 `dc8dda5780b705e902155372c135a993778506c547ef8ebb2b5b03668c11f043`, matching the audit). **No finding was a phantom.** In particular:

- **The comparator reconciliation reproduces exactly.** Re-running each model's own run plan and loader over the current CSV gives `lrp-rli-jm-002` 153 union rows (152 word-reading cells, 152 nonword cells, 53 children) with an exposure logit SD of **1.411770**; `lrp-rli-mech-096` 152 nonword rows at **1.385682**; `lrp-rli-mech-101` 156 word-reading rows at **1.433543**. The three stored `config.json` files agree on the row counts (153 / 152 / 156). One standard deviation of letter sounds is therefore a different raw increment in each fit, and the word-reading marginal keeps four rows the joint fit excludes.
- **The wave-scale spread reproduces.** `lrp-rli-jm-001` re-standardises within each wave: 1.588 / 1.382 / 1.385 / 1.436 logits at t1–t4 (the audit said ~1.59 / 1.38 / 1.39 / 1.44).
- **The data-selected headline is real.** In the stored reporting fit timepoint 3 is the artefact-hosting wave, while the key-findings box and the report both led with timepoint 1 — the wave whose `P(Δ > 0)` (0.072) sits furthest from 0.5. All four waves had passed their sampler checks, so this was a selection defect, not a convergence one.
- **The lifecycle gap is real.** Only the artefact-hosting wave went through `run_primary_fit`. The other three called `run_subfit` with no `trace_filename` and no posterior-predictive request, so they had no persisted trace, no informative predictive check and no recorded power-scaling result; `--reuse-trace` would have failed closed on them. `config.json` recorded `all_published_fits_converged`, and the release evaluator never read it.
- **The power-scaling diagnoses are material and absent from the box.** `lrp-rli-jm-001` flags "potential strong prior / weak likelihood" on `beta_mech[W]`, `beta_mech[N]`, `delta_ls_decoding` and `beta_mech_focal_given_held`; `lrp-rli-jm-002` flags "potential prior-data conflict" on `beta_mech[N]`, `delta_ls_decoding` and `rho_outcome`. Neither box mentioned any of it.
- **The LOO-PIT target mismatch is real.** `_joint_outcome_predictive_tree` subsets one outcome's flattened cells and keeps no child map, so each plot leaves out one cell while the same child's other transitions, other outcome and fitted intercept remain — whereas the main PSIS-LOO does aggregate by child.
- **The high Pareto-k values are real.** `lrp-rli-jm-002` has 2 of 53 children above 0.7, maximum 0.94.
- **`("N", "N")` did build.** The public factory checked only membership, so a duplicate contrast produced a `delta_ls_decoding` identically zero and a conditional slope partialling the focal outcome against itself.
- **`tier1_decoding_specificity` did return early.** `if not rows: return False` sat above the `jm-002` load, so a joint-only run discarded an otherwise valid identified contrast.

## Decisions

### D1 — Every published wave gets the whole lifecycle, and the bundle gates the release

The issue offered a choice: fully diagnose every wave, or publish one prespecified wave and label the rest exploratory. Taken: **diagnose every wave**, because four cross-sectional estimates are the deliverable and downgrading three of them would lose more than it saves.

Each fitted wave now gets its own persisted trace (`trace.nc` for the artefact-hosting wave, `trace_wave_tN.nc` otherwise), its own new-child predictive check (`ppc_summary_marginal_tN.csv`), its own power-scaling result (`psense_wave_tN_summary.csv`), a convergence scan covering its **reported deterministics** as well as its free variables, and — for a non-hosting wave — a sub-fit provenance row binding all of it together. `joint_mechanism_fit_diagnostics.csv` names each wave's three files, and `release._joint_mechanism_wave_release_failures` fails closed on a missing file, an inconsistent slope/diagnostics wave set, a missing provenance row, or any wave whose verdict failed or could not be taken.

Two supporting changes were needed in the shared layer. `run_subfit` gained `extra_var_names`, because a published ratio can mix far worse than the slopes it is built from and the scan had never seen it; the parameters actually scanned are now recorded in a new `convergence_vars` provenance column. `diagnostics.psense_artifacts` gained a `stem`, so a family publishing several posteriors from one fit can record a power-scaling result per posterior instead of letting the hosting fit's stand in for all of them.

The row-count anchor rule is kept but renamed throughout to what it is — an **artefact-hosting** rule about file placement, recorded as `artifact_hosting_timepoint`. It confers no scientific priority, which is now true rather than merely asserted, because nothing downstream reads one wave in preference to another.

### D2 — No reporting path selects a wave after seeing its posterior

The key-findings builder and the results partial both reported the whole set in wave order. The headline is now "Δ is t1 −0.47, t2 −0.17, t3 −0.26, t4 +0.06 … All fitted waves are reported; none is selected as a headline", and the direction sentence states whether the sign is stable across waves rather than quoting the clearest one. The plot marks a `[GATE-FAIL]` wave instead of showing it unflagged.

### D3 — Relabel the comparison rather than force a common contract

The issue offered "a common row, cell, prior, likelihood and exposure-scaling contract, **or** relabel the comparison". Taken: **relabel, and make the mismatch machine-checkable** — a genuine common-contract refit would mean registering new comparison-only marginal models, which is a scientific addition rather than a repair, and it would still not make `lrp-rli-jm-001` nested with `ca-010` / `ca-011` (a latent-conditioning logistic-normal Binomial model against observed-score Beta-Binomial fits with mean-imputed predictors).

Both designs now carry a `comparator_equivalence` statement in the run plan, the generated recipe, `config.json` and the rendered report. Each mechanism fit records its own `exposure_logit_sd`, and `scripts/compare_statistical_models.py` writes `tier1_1a_comparison_contract.csv` plus a `comparable` flag and reason on both 1A rows. A field a fit never recorded counts as **not proven**, never as agreement. On the current stored fits the verdict is `comparable=False` on rows alone; it will remain `False` on the scaler until `mech-096` / `mech-101` are refitted with the new metadata.

### D4 — `Delta` is an association contrast, not a decoding-mechanism test

The Campbell–Fiske argument is kept but is no longer presented as sufficient. With both outcomes and the exposure loading on one unobserved general ability, the latent-scale slopes stay proportional to their loadings, so their difference is proportional to the **loading difference** with no causal letter-sound route at all — and the two instruments differ in item count (79 against 6), floor compression and link discrimination, with no measurement invariance imposed. The key-findings direction sentence now says "not a decoding-use signature" explicitly, and the same qualification is in the factory docstring, both model modules, both reports, the results partial and `docs/models/README.md`.

### D5 — The slope ratio is governed, not classified

`share_retained` keeps its machine key (a stable CSV column is worth more than a tidier name) but its published label is the **conditional-to-marginal slope ratio**, and the report's "median below 0.5 means most of it runs through decoding" classification is gone — it was a mediation claim this observational model does not identify, and it was simply false for a negative ratio.

The ratio's `mean` cell is now written blank, because a ratio's mean is dominated by small-denominator draws and the model documentation had said never to report it while the generic table published one anyway. A new `conditional_slope_ratio.csv` carries `P(ratio < 0)`, `P(0 ≤ ratio ≤ 1)`, `P(ratio > 1)` and a **prespecified** denominator-stability rule: the unconditional slope's 89% interval must exclude zero and at most 5% of its mass may lie within 0.1 logit per SD of zero. Both numbers are fixed in advance so the rule cannot be tuned to a fit. The "the latent and observed versions bracket the answer" claim is withdrawn: that ordering is not guaranteed across two nonlinear models with different likelihoods, missing-data handling and floors.

### D6 — Surface the power-scaling diagnoses; direct alternative-prior fits deferred

Material diagnoses now appear in the key-findings box, per wave, read from each wave's own table. The issue also asks for **direct** alternative-prior fits (slope width 1.0 → 0.5 and own-baseline 0.25 → 0.5 for `jm-002`; 0.3 → 0.7 at every published wave for `jm-001`; dependence-block variants where `rho_outcome` is interpreted). Those are new registered sensitivity models, not a repair of the existing ones, and they are **deferred to a follow-up**: this batch makes the existing diagnosis visible beside the affected result rather than leaving it in a CSV nobody reads.

### D7 — Relabel the LOO-PIT target; do not manufacture a grouped one

The figure title now names the unit (`conditional leave-one-cell-out`), the value is recorded in `config.json`, and both the report and the model module state that neither the plot nor the child-level `elpd_loo` answers "how well would this predict a **new** child" — which would additionally require the omitted child's random effect to be redrawn from the population distribution. The report also refuses the child-level `elpd_loo` for model ranking while any Pareto-k exceeds 0.7, quoting the count and maximum. Implementing a genuine grouped calibration diagnostic (exact child re-LOO or child-level K-fold) is deferred with that stated consequence rather than approximated.

### D8 — The remaining smaller items

- **Wave minima.** A wave is fitted only with ≥ 10 usable children, ≥ 10 observations on **each** outcome and ≥ 10 children observing **both**. The union count bounded none of these, and the overlap is what identifies `rho_outcome` and the conditional slope.
- **Marginal predictive check.** Now a **required** artefact per wave, reported per outcome as well as pooled (the two denominators differ by an order of magnitude), with predeclared coverage floors — 0.35 at the 50% level, 0.75 at the 90% — that attach a qualification to the release decision rather than withholding it. Substantive misfit is a finding about the model, not evidence that sampling failed.
- **Wave-specific metadata.** `joint_mechanism_wave_eligibility.csv` and `wave_eligibility` in `config.json` record each wave's own counts and drops, separately from the four-timepoint `dropped_rows` ledger a wave subset inherits unchanged. `wave_exposure_logit_sd` records the per-wave exposure scale.
- **Source comments.** Both designs report `rho_outcome`; only the levels design adds the conditional slope. The pipeline docstring and the term-label comment said otherwise.
- **Joint-only comparison.** The 1A contrast is written whether or not the marginal forest can be.
- **Factory invariant.** A duplicate or incomplete contrast is rejected at the public factory boundary.
- **ANCOVA language.** `jm-002`'s estimand is each outcome's post-level given its own baseline, pooling between-child and within-child information — not "how much more a child's score moves". Rewritten in the run plan, the module, the report and `docs/models/README.md`.
- **Missing data.** The conditional missing-at-random assumption is stated in the recipe, both reports and the results partial, together with an explicit note that no MNAR or complete-case sensitivity is registered and that every published quantity is conditional on it.

## Tests added

`test_joint_mechanism_pipeline.py`: per-outcome and overlap wave minima, the wave-specific ledger, the ratio's suppressed mean and three probability regions, the unstable-denominator rule, asymmetric outcome missingness in the marginal coverage, and the per-wave exposure scale. `test_factories.py`: the duplicate/incomplete contrast rejection and a simulation-based recovery of a known slope difference. `test_release_decision.py`: the complete wave bundle publishes; a failed wave, a missing trace / predictive / power-scaling file, a missing provenance row and a slope table naming an unpublished wave each withhold; poor coverage qualifies instead. `test_key_findings.py`: no wave is selected after seeing it, a failed wave withholds, and the builder keeps its own filter underneath. `test_subfits.py`: `extra_var_names` extends and records the scan. `test_diagnostics.py`: the LOO-PIT figure names its leave-out unit. `test_compare_tier1_contract.py` (new): the row/scaler contract, its "not proven" default, and the joint-only 1A path.

## What still needs running

The code changes alter what a fit **publishes**, so the stored reporting-tier artefacts are stale until refitted:

- `lrp-rli-jm-001` and `lrp-rli-jm-002` need publication-tier refits before their reports are read again. A `rep-lite` run in the review worktree exercised the whole new lifecycle end to end.
- `lrp-rli-mech-096` and `lrp-rli-mech-101` need refits only to populate `exposure_logit_sd`; their posteriors are unchanged. Until then the Tier-1 comparison reports the scaler as _not recorded_, which is the correct fail-safe.
- The deferred items above (direct alternative-prior fits, grouped LOO calibration, an MNAR sensitivity) belong in follow-up issues, not in this batch.
