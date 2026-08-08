> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5).

# The association families move: `adjusted`, `horseshoe`, `concurrent` (#394, tranche 7)

**Date:** 2026-08-08. **Issue:** #394 (complete the statistical pipeline family split and artefact lifecycle refactor), implementation-sequence step 6, third group. **Change:** pure relocation — no estimand, likelihood, prior, analysis population, fitted equation, sampling preset, diagnostic threshold or artefact schema is touched.

## What this tranche does

`fit_adjusted`, `fit_rlm_adjusted`, `fit_horseshoe`, `fit_rlm_horseshoe` and `fit_concurrent` now live in `statistical_models/pipelines/adjusted.py`, `pipelines/horseshoe.py` and `pipelines/concurrent.py`, joining the eleven families from [tranche 4](202608072330-394-pipelines-family-split.md), [tranche 5](202608081000-394-family-group.md) and [tranche 6](202608081130-394-mechanism-mediation.md). `pipeline.py` drops from 3,485 lines to 1,927 — under a fifth of the 10,182 it carried five tranches ago. Eighteen family entry points are now family-owned and eight remain in the monolith.

What is left is a coherent remainder rather than an arbitrary one: survival, the LRP67 LCSM, growth and its historical companion, and the Byrne measurement and joint-growth fits — including the correlated-factor models whose exact child-level log-likelihood and constrained-scale log-prior recovery design point 7 wants isolated.

## The Byrne ports came with their families, and the reason is not convenience

The issue's third group is "adjusted/concurrent/horseshoe". `fit_rlm_adjusted` and `fit_rlm_horseshoe` — the Byrne (RLM) cohort ports — are not named in any of the issue's four groups. They came along because `definitions.KINDS` settles the question: `lrp_rlm_adj_001` declares `kind="adjusted"` and `lrp_rlm_hs_001` declares `kind="horseshoe"`. They **are** those families, on a second dataset, publishing the same `predictor_associations.csv` and `predictor_ranking.csv` schemas so the shared report partials apply unchanged. Leaving them in the monolith would have split each family across two files — precisely what the issue's second acceptance criterion ("every family has an identifiable orchestration module") rules out.

It also happens to be the cleaner cut. Taking the issue's three families alone strands `_beta_summary`, shared with `fit_rlm_adjusted`; taking all five strands nothing at all.

## The one helper that needed a home below the split

`_beta_summary` became `reporting.beta_summary`: the posterior mean, equal-tailed `ci_prob`-coverage interval and P(>0) for a named variable. It is a pure function of a trace — no context, no output directory, no plotting — and `reporting.py` is where the family summary calculations already live (`band50`, `tau_summary_*`, `rope_summary`, the pushforward builders). That is also design point 6's stated direction: summaries as pure functions returning values or DataFrames, separate from artefact writing and console output.

The only other cross-module edge is `rlm_nuisance_names`, which both Byrne ports call to resolve the group-nuisance columns of the span frame. It stays in `adjusted.py` next to `rlm_natural_scale_contrasts`, and `horseshoe.py` imports it — the sibling-import pattern `joint`/`itt` and `did`/`dose_response` already established.

## Relocation, checked the same way

Ten cut regions, each declaring its expected opening line, ending on a top-level statement boundary checked against the AST, followed by a blank line and not overlapping its neighbour. The tripwires held on the first run and no moved function is still defined in `pipeline.py`.

- **Byte-identity.** Re-applying the two-name rename map to the original line ranges reproduces the three new modules and the appended `beta_summary` exactly — **8 of 8 regions byte-for-byte**, with `reporting.py`'s existing 6,028 lines untouched. Unlike the previous tranche there is no exception at all: `beta_summary` needed no alias fix, because it reaches nothing but `np`.
- **No string was renamed.** Tokenising the pre-move `pipeline.py` and searching only `STRING` tokens for `_beta_summary` and `_rlm_nuisance_names` returns **zero** hits, so no artefact filename, table key, console label or guard message could have been touched.
- **Nothing else changed in `pipeline.py`.** Of the 21 lines the diff adds, 18 are not present verbatim in the original: the rewritten docstring, the five re-export lines and one retitled banner.

One comment changed: the Byrne banner read "Phase B/D fits (#338): adjusted, horseshoe, corr_factor, joint" and two of those four are leaving, so it now reads "corr_factor, joint growth", with the move script asserting the old text occurred exactly once. The "Adjusted pipeline (LRP65)" banner was discarded into `pipelines/adjusted.py`'s module docstring.

## Tests

`MIGRATED_FAMILIES` needed no structural change this time — the tuple-per-family shape introduced for `mediation` last tranche absorbed `adjusted` and `horseshoe` carrying two entry points each. `reporting` joins the shared-module list that may not import the monolith. `test_concurrent_pipeline.py` imports its four `_ca_*` helpers from `pipelines.concurrent`.

## Verification

Ten dev fits spanning every moved branch — `adj-065` (the RLI adjusted headline with its bivariate, prior-sweep and SES complete-case sub-fits), `hs-001` and `hs-004` (horseshoe ranking, gain and level framings), `ca-001` and `ca-005` (concurrent, with their per-wave and bivariate sub-fits) and `ca-011` (the covariate-adjusted variant, decoding held fixed), `rlm-adj-001` and `rlm-hs-001` (the Byrne ports), plus `mech-056` (a family that stays put and calls into `reporting`) and `itt-001` as controls — were run from `main` in a detached worktree and again from this branch. Every CSV is byte-identical — **131/131** across the ten fits, with every manifest coherent and no untracked table CSV in any of them. Full suite, `ruff check src/ scripts/ tests/`, `npm run format:check` and `npm run spellcheck` pass.

## What is not in this tranche

Eight entry points remain in `pipeline.py`: survival, `lcsm`, growth, historical growth, the two Byrne measurement/joint-growth fits and the two correlated-factor fits. That is the issue's fourth and last group, and it is the one carrying design point 7's specialised algorithms — so it should be a relocation plus an isolation, not a relocation alone. The `SubfitRunner` (design point 5) now has three clear customers in one place: the adjusted bivariate and SES fits, the concurrent per-wave and bivariate fits, and the ITT floor branch. The release-decision boundary (point 3), typed settings for the remaining families (point 4) and the step-7/8 MyPy gate and dead-state clean-up are still open.
