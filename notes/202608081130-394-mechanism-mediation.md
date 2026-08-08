> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5).

# The mechanism and mediation families move: `mechanism`, `joint_mechanism`, `mediation` (#394, tranche 6)

**Date:** 2026-08-08. **Issue:** #394 (complete the statistical pipeline family split and artefact lifecycle refactor), implementation-sequence step 6, second group. **Change:** pure relocation — no estimand, likelihood, prior, analysis population, fitted equation, sampling preset, diagnostic threshold or artefact schema is touched.

## What this tranche does

The issue's second group leaves the monolith: `fit_mechanism`, `fit_joint_mechanism`, `fit_mediation`, `fit_mediation_period_stacked` and `fit_mediation_multi` now live in `statistical_models/pipelines/mechanism.py`, `pipelines/joint_mechanism.py` and `pipelines/mediation.py`, joining the eight families from [tranche 4](202608072330-394-pipelines-family-split.md) and [tranche 5](202608081000-394-family-group.md). `pipeline.py` drops from 5,612 lines to 3,485 — a third of the 10,182 it carried four tranches ago. Thirteen family entry points are now family-owned and thirteen remain in the monolith; `pipeline.py` re-exports all thirteen, so no model module changed its import path.

Dose-response was pulled forward into the previous tranche (recorded there), so the group here is the remainder of the issue's second grouping.

## The cut was already clean

The AST reachability scan over the remaining eighteen `fit_*` entry points found the mechanism and mediation families reach **22 functions that nothing else reaches** (1,974 lines) and exactly **one** shared with families that stay behind. That one is `_sample_model` (33 lines), the sub-fit sampler used by the adjusted, concurrent and RLM-adjusted families as well as the joint-mechanism levels design.

Better still, the three new modules do not reference each other at all: the scan found **zero** cross-module edges inside the group and **zero** references from anything staying in `pipeline.py` into anything moving. The three files could have been three separate pull requests; keeping them together only reflects the issue's own grouping.

Because the cut was clean, only one name crossed a module boundary and needed promoting. Every family-private helper — `_jm_marginal_ppc`, `_write_mechanism_curve`, `_fit_t3_sensitivity`, `_raw_covariate_confounders` and the rest — kept its name inside its new module, along with the five module-level constants (`_JM_TERM_LABELS`, `_JM_MIN_WAVE_ROWS`, `_JM_SLOPE_REQUIRED`, `_COVARIATE_EXPOSURE_LABELS`, `_T3_SENSITIVITY_TIME`), none of which had a reader outside its own family.

## Where the sub-fit sampler went, and why

`_sample_model` became `diagnostics.sample_subfit`. It samples a secondary model with nutpie without touching the headline `ctx.trace` / `trace.nc`, then runs the convergence check and returns the verdict alongside the trace, so the caller can persist a convergence flag onto the sub-fit's own published CSV — sub-fit traces bypass the primary `diagnostics_summary.json` gate, and the verdict used to be computed and discarded, leaving the bivariate, prior-sweep and SES-sensitivity tables reported with no convergence flag at all.

`diagnostics.py` is the honest home rather than a new module: it already owns `sample_posterior` (the primary sampler this one's docstring says it mirrors) and `subfit_convergence` (the verdict it computes). Design point 5 of the issue wants a typed `SubfitRunner` that also carries fitted-data identity, sampling settings, optional posterior prediction and structured failure; that is a behavioural change and belongs in its own tranche. Inventing a placeholder module for it now would have been guessing where that runner will live.

One consequence worth stating plainly: `diagnostics.py` is not a module the families may import _around_ — it is below them, so `test_pipeline_boundaries.py` now asserts it too may never import `pipeline.py`.

## Relocation, checked the same way

Same discipline as the previous two tranches, and again it passed on the first run: each of the eight cut regions declares its expected opening line, must end on a top-level statement boundary checked against the AST, must be followed by a blank line and must not overlap its neighbour; afterwards no moved function may still be defined in `pipeline.py`.

- **Byte-identity.** Re-applying the one-name rename map to the original line ranges reproduces the three new modules and the appended `sample_subfit` exactly — **5 of 5 regions byte-for-byte**. The single exception is deliberate and is itself asserted by the move script: inside `diagnostics.py` the `_diag.` alias does not resolve, so `conv = _diag.subfit_convergence(` became `conv = subfit_convergence(` — one line, changing which namespace the name is looked up in and nothing else. Monkeypatching `diagnostics.subfit_convergence` still intercepts it, because the lookup is still an attribute read on the module at call time.
- **No string was renamed.** Tokenising the pre-move `pipeline.py` and searching only `STRING` tokens for `_sample_model` returns **zero** hits, so no artefact filename, table key, console label or guard message could have been touched.

One comment did change: the `# Common helpers` banner headed `_raw_covariate_confounders` and nothing else, so once that moved the banner would have sat above the survival family alone. It is retitled `# Survival pipeline (LRP-RLI-SURV)`, with the move script asserting the old text occurred exactly once.

## Tests

`MIGRATED_FAMILIES` in `test_pipeline_boundaries.py` had to grow a dimension: every family so far has exactly one entry point, but `mediation` has three fit functions plus `prepare_mediation_data`, which `scripts/regenerate_mediation_calibration.py` imports by name. The map now holds a tuple per family and the re-export test is parametrised over the flattened pairs, so a family with several entry points cannot have one of them quietly dropped from the facade. The package-contents assertion is unchanged in spirit: `pipelines/` must match the guard exactly, so the next family move fails here until the guard is updated.

The direct-helper tests moved with their subjects. `test_joint_mechanism_pipeline.py` imports the six `_jm_*` helpers from `pipelines.joint_mechanism` and patches that module's `_diag`; `test_factories.py` and `test_pipeline_key_findings.py` reach into `pipelines.mechanism` for the curve and items writers. The `_diag` patches would in fact have kept working untouched — `pipeline._diag` and `joint_mechanism._diag` are the same module object — but a test that names the module it is exercising is worth more than one that happens to still pass.

## Verification

Thirteen dev fits spanning every moved branch — `mech-056` (linear slope), `mech-072` (linear slope with moderation), `mech-061` (HSGP shape, moderation and the readiness threshold), `mech-156` (HSGP shape alone), `jm-001` (the per-wave bivariate levels design with the LKJ residual correlation) and `jm-002` (the phase-stacked transition design), `med-059` (count mediator; the temporal-ordering sensitivity runs), `med-062` (Gaussian reading-route composite), `med-176` (`outcome_time` set, so the sensitivity is correctly skipped rather than double-lagged), `med-060` (`mediation_multi`), `med-092` (period-stacked), plus `adj-065` (a family that stays behind and calls the promoted `sample_subfit`) and `itt-001` as controls — were run from `main` in a detached worktree and again from this branch. Every CSV is byte-identical — **179/179** across the thirteen fits, with every manifest coherent and no untracked table CSV in any of them. Full suite, `ruff check src/ scripts/ tests/`, `npm run format:check` and `npm run spellcheck` pass.

## What is not in this tranche

Twelve families remain in `pipeline.py`, including the ones carrying the specialised algorithms design point 7 wants isolated (the longitudinal correlated-factor exact child-level log-likelihood and constrained-scale log-prior recovery). The `SubfitRunner` (design point 5), the release-decision boundary (point 3), typed settings for the remaining families (point 4) and the step-7/8 MyPy gate and dead-state clean-up are all still open. The next group on the issue's own list is adjusted/concurrent/horseshoe, which share `_beta_summary`, `_plot_associations` and `_natural_scale_contrasts` with the RLM-adjusted family — so unlike this one, that cut will need a decision about where those three live.
