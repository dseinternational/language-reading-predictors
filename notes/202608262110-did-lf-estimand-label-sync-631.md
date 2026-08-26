# DiD and level-factors estimand-label synchronisation (#631 findings 12 and 13)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5).

**Date:** 2026-08-26. **Issue:** #631 (audit findings 12 and 13). **Scope:** descriptive relabelling only — no likelihood, prior, adjustment-set, release-gate or headline change anywhere in this change.

## What was decided

The #576 estimand review settled the waitlist-crossover family's reading in the resolver (`did.py`) and the reader-facing summaries (`reporting.py`): `tau_t2` is the randomised contrast of assignment to immediate treatment versus no treatment yet; `arm_gap_t3` is **also** identified by the original randomisation, but of a different exposure — assignment to the early-start versus delayed-start treatment schedule, both arms having been treated by t3 — so it is neither a treated-versus-untreated effect nor latent-ability-confounded, and what it cannot supply is the mechanism; `delta_crossover = tau_t2 - arm_gap_t3` is the change between two randomised regime contrasts, never an identified catch-up. The #631 audit found that many older surfaces still carried the pre-#576 labels ("post-crossover associations", "waitlist catch-up", a "treatment-presence" reading of `theta_treated`). This change synchronises every stale surface to the settled position, adopting the wording the DID-103/104/105/106 companions already carried — their SPEC string `causal_status="t2 randomised; t3 a randomised treatment-schedule contrast"` is taken verbatim as canonical for the whole arm-by-wave family.

Distinguishing the treatment effect targeted by a trial's randomised comparison from other quantities the same design identifies is the core of the ICH E9(R1) estimand framework ([E9(R1) Statistical Principles for Clinical Trials: Addendum: Estimands and Sensitivity Analysis in Clinical Trials](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/e9r1-statistical-principles-clinical-trials-addendum-estimands-and-sensitivity-analysis-clinical)): `tau_t2` and `arm_gap_t3` differ in their treatment attribute (immediate-versus-none-yet against early-start-versus-delayed-start), not in their identification, and the labels now say so.

## Finding 12 — DiD surfaces relabelled

- `factories.build_did_model` docstring: `arm_gap_t3` and `delta_crossover` described per the canonical wording; the dose paragraph now states that under the saturated arm-by-period cell coding `theta_treated` at the mean treated dose is the crossover **cell** contrast, not an isolated treatment-presence effect (the resolver at `did.py` already said both).
- Model-module docstrings LRPDID01/02/03/04/05/08/09/10/11/12/014/015 and the companion reading rules in LRPDID101/102; the dose docstrings LRPDID06/07/07base take the `theta_treated` cell-contrast wording.
- The 14 SPECs still carrying `causal_status="t2 randomised; post-crossover contrasts associational"` (001–005, 008–012, 014, 015, 101, 102) now carry the DID-103 string. `causal_status` is prose excluded from the blending pair gate's run-plan comparison (`_PLAN_PROSE_FIELDS`), so the stored LRPDID03+103 pair remains valid; no structural spec field changed.
- Report overviews `docs/models/lrp-rli-did-001/002/004/005/008/009/010/011/012/014/015/101/102` synchronised to the corrected wording (did-003 and did-103 were already correct and served as templates); `_results_did.qmd`'s predicted-scores caption and its `theta_treated` dose-table label; `docs/models/README.md`'s did-006 row; the refit runbook's `did` row; `METHODS.md`'s glossary entry for `delta_crossover` and the prior-tiering sentence (now "balance quantities or randomised schedule contrasts rather than additional treated-versus-untreated effects").
- `reporting.py`'s machine-readable `dose_interpretation` string now reads "beta_dose is an observational intensive-margin association; theta_treated is the crossover cell contrast at the mean treated dose, not an isolated treatment-presence effect"; the test fixture embedding the old string was updated.

## Finding 13 — level-factors t3/t4 relabelled

The LF t3/t4 arm-gap changes `d_grp_time[t3]`/`[t4]` are functions of the original randomised assignment (early-start versus delayed-start schedule), read available-case and model-adjusted; they are **not** latent-ability-confounded adjusted associations, **not** treated-versus-untreated effects, and carry no mechanistic reading (duration, carryover, maturation and ceilings are inseparable). They are now reported as **randomised schedule contrasts**:

- `level_factors.py`: module docstring, the `POST_PHASE_LABELS` comment, and the resolver's `estimand`/`causal_status` prose (the t3/t4 sentences are gated on the four-wave window so a two-wave comparator's plan never names coefficients its posterior lacks); `factor_summary_roles()` now assigns the post-t2 elements the DiD family's `regime` role.
- `prior_artifacts.py`: the `d_grp_time` prior-table row takes role `regime` with a rationale naming the t2 element the randomised treated-versus-untreated change and the t3/t4 elements randomised schedule contrasts of assignment, mechanism unidentified — the established DiD `arm_gap_t3` idiom, so the priors-table role glossary already covers it.
- `pipelines/level_factors.py` docstring and factor-summary comment; `reporting._kf_build_level_factors` (the key-findings causal sentence now says the later timepoints are still set by the original random assignment but compare an earlier with a later start of the same teaching, not treated versus untreated); `factor_summary`'s docstring documents the `regime` role.
- `_results_factors.qmd`: `d_grp_time[t3]`/`[t4]` move out of the "Adjusted associations — not causal" callout into their own "Randomised schedule contrasts" block, modelled on the levels-view block.
- Docs: `docs/models/lrp-rli-lf-001`–`011`, `-106` and `-201` report templates, `docs/models/README.md`, `docs/models/PRIORS.md`, `METHODS.md` level-factors bullet, and the refit runbook's `level_factors` row.
- Part 2 of the finding (the t1/t2 randomised-window comparator) had already been fixed and was left untouched.

**Unchanged by design:** `causal_terms` (only `d_grp_time[t2]` is flagged causal), the release gate and `release.gate_applies` boundary, the key-findings headline quantity, and the tau-prior tier (the t3/t4 elements keep their existing priors; the tier still applies to the t2 contrast only). This is a labelling correction, not a re-analysis.

## Stored fits

`config.json` in already-stored fit directories records the old `causal_status`, `estimand`, `dose_interpretation` and prior-table strings at fit time. Those stored strings are now stale relative to the source and remain so until each model is refit (or, where a regeneration script covers the artefact, regenerated); the pair gates are unaffected because prose fields are excluded from the run-plan comparison. Readers of stored `config.json` files should prefer the current resolver output for interpretation.

## Related

Finding 19 (the Pareto-k observation-unit prose in `_diagnostics.qmd`/`_footer.qmd`) was fixed in the same branch: the LOO unit is now derived from the resolved run plan's `loo_unit` with a corrected kind-list fallback, `dose_response` (#587) and `survival` (#631 finding 14) read as child-level, `block_exposure`/`pooled_levels` as row-level, and horseshoe split by its `gain` setting; the conditional-LOO random-intercept caveat no longer attaches to survival, which has no random intercept.
