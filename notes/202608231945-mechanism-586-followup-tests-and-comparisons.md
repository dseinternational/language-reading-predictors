> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Closing the remaining #586 items: comparison tests, report-form tests and regenerated comparisons, 2026-08-23

## Purpose

PR #599 implemented the substance of #586. Four of the issue's checklist items were left partly open because they asked for a _test_ of a behaviour the PR changed, or for a regeneration the PR did not run. This note records closing them. Nothing here changes a fitted estimand.

## What was outstanding, and why

| item                                                                                                        | why it was still open                                                                                                         |
| ----------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| "Use posterior medians in the forest and **define/test** the nonlinear slope weighting"                     | The estimand was defined and implemented; the audit's "pin the choice with an irregular-grid test" was not written.           |
| "Copy mech-058/071 and mech-072/172 LOO comparison CSVs beside both model runs" (**and test the contract**) | The copy landed; nothing asserted it.                                                                                         |
| "Failed or missing gates cannot produce an ordinary mechanism forest"                                       | The gate landed; no test exercised a REVIEW or MISSING gate.                                                                  |
| "Linear, default-HSGP, tight-HSGP and covariate-HSGP **reports** show the correct form, prior and units"    | #599 tested the _prior inventory_ for all four forms, and the four reports were read by hand, but no test rendered the prose. |
| "Regenerate ... comparisons ... from stored traces"                                                         | Prior artefacts, key findings and reports were regenerated; the comparison artefacts were not.                                |

## The nonlinear slope estimand, pinned

`_mechanism_slope_distribution` is a fitted-row average derivative. The test uses a deliberately lopsided grid — a piecewise-linear curve with slope 1 below zero and 11 above, sampled at `x = [-2, -2, -2, -1, 0, 1]`:

- deduplicating and equal-weighting the unique grid gives `(1 + 1 + 6 + 11) / 4 = 4.75`;
- weighting by fitted rows gives `(1×3 + 1 + 6 + 11) / 6 = 3.5`.

The test asserts the second and explicitly rejects the first. This is the failure mode in miniature: on a bounded count measure many children share an exposure value, so deduplication moves the average toward the sparse tail of the range — exactly where an HSGP curve is least constrained.

## Regenerating the comparisons changed an interval, not a conclusion

Re-running `compare_statistical_models.py --config reporting` over the stored traces:

| row                    | before                        | after                         |
| ---------------------- | ----------------------------- | ----------------------------- |
| mech-056 (R→W, linear) | 0.0679 [−0.055, 0.191]        | 0.0679 [−0.055, 0.191]        |
| mech-057 (E→W, linear) | 0.1267 [−0.008, 0.263]        | 0.1262 [−0.008, 0.263]        |
| mech-058 (L→W, curve)  | 0.2318 [**0.077**, **0.402**] | 0.2323 [**0.105**, **0.360**] |

The linear rows barely move (posterior mean → median). The curve row's point estimate is unchanged to three decimals, but its 89% interval **narrows by about 30%**. That is the expected direction and worth stating plainly: the old estimand averaged derivatives over a grid that over-represented the sparsely-populated tails of the exposure range, where the curve is least identified, so it inherited their uncertainty. Weighting each fitted observation once concentrates the average where the data actually are.

The CSV now also carries `model_id`, `converged`, `gate_status` and a per-row `estimand` string, so a reader can tell a `beta_mech` coefficient from a curve's average slope without consulting the source.

## The refit guard fired, independently

Regenerating also exercised the comparison script's exact-refit repair, and it **refused to run** for the mech-063/163 pair:

> `lrp-rli-mech-063: rebuilt frame has 155 rows but the stored trace has 151; the construction path has drifted from the one that produced this fit — refusing to refit`

That is the right behaviour and a useful independent confirmation: a guard written long before #586, for a different purpose, reaches the same conclusion as `regenerate_mechanism_artefacts.py` about which fits are stale. It also has a consequence worth stating — **the joint-readiness L×N comparison is degraded until those two models are refitted.** Its `comparison_valid` is `False` and it now writes per-model `elpd_loo` (mech-063 −391.77, mech-163 −391.97) instead of an `elpd_diff`, because the Pareto-k values are unreliable (0.90 and 0.97) and the usual exact-refit repair is unavailable on a drifted frame. Nothing is published wrongly; the test of the L×N interaction is simply unavailable for now.

(The `lrp-rli-did-007`/`did-107` pair degrades the same way for an unrelated, pre-existing reason — exact refit is a mechanism-family facility only.)

## Report-form tests

`tests/statistical_models/test_mechanism_report_forms.py` renders the real shared partial against four synthetic fit directories — linear, default HSGP, tight HSGP and continuous-covariate HSGP — with a stand-in for `_setup.qmd`, so no `trace.nc` is needed and what is asserted is the prose a reader sees.

Verified as genuine regressions: all five fail against the pre-#599 partial (`f46a0522`) and pass against the current one. A first attempt at that check was void — PR #599 had merged during the session, so `main` already contained the fix and the comparison was the fix against itself.

## Verification

- Full test suite passes; 10 new tests (5 in `tests/test_compare_loo.py`, 5 in `tests/statistical_models/test_mechanism_report_forms.py`).
- `uv run ruff check src/`, `npm run format:check` and `npm run spellcheck` pass.
- The comparison directory was backed up to `comparison.pre-586-followup` before regeneration.

## Still open, by decision

The four refits — mech-063, mech-163, mech-158 and mech-191 — remain the only outstanding #586 work, along with the deferred Batch C sensitivities. See [`notes/202608231830-mechanism-audit-586-verification-and-fixes.md`](202608231830-mechanism-audit-586-verification-and-fixes.md).
