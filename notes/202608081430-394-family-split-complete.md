> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5).

# The split is done: `pipeline.py` is a facade (#394, tranche 8)

**Date:** 2026-08-08. **Issue:** #394 (complete the statistical pipeline family split and artefact lifecycle refactor), implementation-sequence step 6, fourth and final group. **Change:** pure relocation — no estimand, likelihood, prior, analysis population, fitted equation, sampling preset, diagnostic threshold or artefact schema is touched.

## What this tranche does

The last eight entry points leave the monolith: `fit_survival`, `fit_lcsm`, `fit_growth`, `fit_historical_growth`, `fit_correlated_factor`, `fit_rlm_corr_factor`, `fit_rlm_joint_growth` and `fit_longitudinal_corr_factor` now live in seven new modules under `statistical_models/pipelines/`, joining the fourteen from tranches [4](202608072330-394-pipelines-family-split.md), [5](202608081000-394-family-group.md), [6](202608081130-394-mechanism-mediation.md) and [7](202608081300-394-adjusted-concurrent.md).

`pipeline.py` goes from 1,927 lines to **114** — from 10,182 six tranches ago — and, more to the point, from 117 top-level functions to **none at all**. It now contains a docstring and twenty-one import statements. That is the issue's first acceptance criterion, and it is checked rather than asserted: a new test parses the module and fails on any top-level node that is not an import.

## Two guards that only became possible now

The boundary test has grown with the split for four tranches; with the split finished it can pin two properties it could not before.

**The facade is structurally a facade.** `test_the_facade_holds_re_exports_and_nothing_else` parses `pipeline.py` and asserts every top-level node is an `Import` or `ImportFrom`. "No family-specific statistical calculations" is otherwise a matter of opinion; this makes the first re-entry of a function, class or constant fail a test rather than pass unnoticed, and anyone who genuinely needs one has to argue for it by editing the guard.

**Every family kind has a module.** `test_every_registered_family_kind_has_an_orchestration_module` walks `definitions.KINDS` — the authoritative family list — and requires a module of that name under `pipelines/`. The one deliberate exception is recorded in the test: `mediation_multi` is its own `kind` but shares the g-formula machinery with the single-mediator fits, so `pipelines/mediation.py` owns both. A new family kind now fails here until it has an orchestration module, which is acceptance criterion 2 turned into a check.

## Where things went

Seven modules, one per kind: `survival.py`, `lcsm.py`, `growth.py`, `historical_growth.py`, `corr_factor.py` (the RLI CFA and its Byrne measurement-only port, which `definitions.KINDS` keys as the same family), `historical_joint.py` and `long_corr_factor.py`.

One helper crossed a module boundary: `_coef_row` — posterior mean, equal-tailed central interval and P(coef > 0) for a labelled set of draws — used by both the LCSM and correlated-factor families. It became `reporting.coef_row`, next to `beta_summary` which moved there for the same reason last tranche, and next to `tau_summary_itt` whose interval convention its docstring already cross-referenced.

**Design point 7 needed nothing here, because it was already done.** The LCF exact child-level log-likelihood and constrained-scale log-prior recovery live in `lcf_inference.py`, and the descriptive comparison summaries in `lcf_summaries.py`; `pipeline.py` held only aliases under their historical private names. Those aliases moved verbatim with `fit_longitudinal_corr_factor`, since the fit calls them internally, and `pipelines/long_corr_factor.py`'s docstring now says plainly what is and is not in the file. The one genuinely LCF-specific piece of orchestration, `_lcf_stitch_loo`, moved with the family: it binds the recovered likelihood back onto the trace, which is orchestration rather than algorithm.

## Relocation, checked the same way

Fifteen cut regions with the usual tripwires — expected opening line, AST statement-boundary end, blank separator, no overlap. One fired: the two domain-map constants are dicts, and the region expectations said `(`. That is the tripwire doing its job on a hand-typed expectation rather than on a mis-cut, and it cost one re-run.

- **Byte-identity.** Re-applying the one-name rename map to the original line ranges reproduces all seven new modules and the appended `coef_row` exactly — **9 of 9 regions byte-for-byte**, with `reporting.py`'s existing body untouched.
- **No string was renamed.** Tokenising the pre-move `pipeline.py` and searching only `STRING` tokens for `_coef_row` returns **zero** hits.
- **The facade is complete.** A check compares the facade's re-exports against every `fit_*` / `prepare_*` function defined under `pipelines/`: 27 exported, 27 defined, no name missing and none extra. `fit_itt_floor_rule` is excluded by name as a branch of `fit_itt` rather than an entry point.

`pipelines/corr_factor.py` is the one module whose contents are not in original file order: the RLI family opens the file and the Byrne port follows, though the Byrne port came first in `pipeline.py`. The move script takes an explicit per-target ordering key for this, so the reordering is declared rather than incidental.

## Verification

Ten dev fits covering all eight moved entry points — `surv-009`, `lcsm-067` and `lcsm-091` (the lagged-change variant), `gc-069`, `rlm-hg-001`, `rlm-jc-001`, `mm-001` and `rlm-mm-001` (both corr-factor ports), `lcf-001` — plus `itt-001` as a control, were run from `main` in a detached worktree and again from this branch. Every CSV is byte-identical — **142/142** across the ten fits, with every manifest coherent and no untracked table CSV in any of them. Full suite, `ruff check src/ scripts/ tests/`, `npm run format:check` and `npm run spellcheck` pass.

## What #394 still wants

The family split (steps 5–6) and design points 6 and 7 are done; the artefact interface and manifest (step 3) and the primary-fit lifecycle (step 4) landed in earlier tranches. Open:

- **Design point 5, the `SubfitRunner`.** Now that the families sit side by side, its customers are easy to enumerate: the ITT floor branch, the adjusted bivariate / prior-sweep / SES fits, the concurrent per-wave and bivariate fits, the mechanism t3 sensitivity. All of them call `diagnostics.sample_subfit` and then hand-roll persistence and provenance. This is the obvious next behavioural tranche.
- **Design point 3's release-decision boundary**, building on `release.py`.
- **Design point 4**, typed settings for the families that still read `spec.extra`.
- **Steps 7 and 8**: the MyPy gate over the migrated modules, then migrating the 179 model modules off the facade and retiring it, plus the dead `model_vars` / `variables` clean-up.
