# Posterior-predictive check redesign: coverage statistic + labelled figures (#318)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

Date: 2026-07-15. Part of the undergraduate-readability workstream (#266), coordinating with the plot-thinning fixes (#270) and the report/code correspondence discipline (#271).

## Problem

The posterior-predictive figure was the stock ArviZ `plot_ppc_dist(trace)` overlay: unlabelled axes, a variable called `y_post`, every likelihood node pooled onto one axis, and no verdict a reader could extract — the caption asked the reader to judge band overlap by eye. There was no computed posterior-predictive coverage statistic anywhere in the codebase ("coverage" previously only ever meant the credible-interval `ci_prob`).

## What replaces it (per family, from the existing `posterior_predictive` group — no new sampling)

1. **Relabelled distribution overlay** (`posterior_predictive_check.png`) — observed count density (black) against the posterior-predictive band (blue: pointwise 5–95% of replicate-dataset densities + median), on a labelled items axis (test name + denominator from `measures`), with a real title. Emitted only for single-measure count families; skipped for the multi-outcome-flattened families (`joint`, `lcsm`/`growth` `y_obs`) whose one likelihood node pools measures with different denominators (6–170) — pooling those onto one count axis is the exact meaninglessness #271 item 2 flagged. Those families still get the coverage statistic and the calibration panel, which are per-observation and denominator-agnostic.
2. **Per-observation calibration panel** (`ppc_calibration.png`) — observed score (x) against the posterior-predictive median with a 90% interval (y), with a `y = x` diagonal. Deviations from the diagonal are directly readable. Valid for every count family (each point is compared to its own predictive draws, so mixed denominators do not distort it).
3. **Computed coverage statement** (`ppc_summary.csv`) — the share of observations whose observed score falls inside the model's 50% and 90% prediction intervals, rendered by `_diagnostics.qmd` as one sentence via `reporting.ppc_coverage_markdown`. A coverage number is decidable; band-gazing is not. The sentence (and its plain-language verdict) is derived from the CSV, never hand-stated.
4. **LOO-PIT stays** as the analyst-grade check, under the technical fold in `_diagnostics.qmd` (unchanged).

## Pre-specified design decisions (fixed across families)

- **Same-children (conditional), in-sample.** The prediction intervals are the standard in-sample PPC: how well the fitted model re-predicts the children it was fit on, _given_ their own fitted (random) effects. The rendered sentence says "for these children" and never implies new-child generalisation. This is the PPC analogue of the conditional-LOO caveat recorded in #270 item 4.
- **Discreteness / interval convention.** With bounded counts, exact interval coverage is affected by ties at the interval edges. We adopt one convention and keep it fixed across families: for observation `i` and level `p`, the interval is the **closed** interval `[q_lo, q_hi]` where `q_lo, q_hi` are the `(1-p)/2` and `(1+p)/2` empirical quantiles (numpy default linear interpolation) of that observation's posterior-predictive draws; the observed count is _inside_ iff `q_lo <= y_obs <= q_hi`. Closed comparison counts an observed value sitting exactly on a quantile edge as covered. Because the quantile endpoints are generally non-integer while `y_obs` is integer, edge ties are rare, but the convention makes the rule reproducible.
- **Floor-rule outcomes** (`likelihood="bernoulli_offfloor"`, node `y_offfloor`, and the survival `y_event` node) report coverage of the off-floor **rate** by group cell, not per-observation count intervals: per-observation interval coverage of a 0/1 indicator is degenerate (a Bernoulli's central interval is `[0, 1]` for any non-extreme `p`, so every observation is trivially "covered"). For each group cell (arm × wave where `prepared.G`/`prepared.phase` are available and aligned; a single overall cell otherwise) we compare the observed off-floor rate to the posterior-predictive distribution of that cell's rate, and count the cell as covered iff the observed rate falls in the closed central `p`-interval. This mirrors the existing `did_cell_ppc` and `proportion_at_zero_ppc` checks.

## Node routing

`_run_ppc` samples a per-family node list and passes the **last** node as the primary outcome (the outcome leg is last in every multi-node family: mediation `[mediator_post, y_post]`, two-mediator `[..., ..., y_post]`, corr_factor `[Z_obs, y_post]`). The PPC suite then routes:

- `y_offfloor` / `y_event` → off-floor rate coverage + a per-cell rate figure.
- `y_post` / `y_obs` / `score` → count interval coverage + calibration panel (+ overlay for single-measure families).
- anything else (measurement/latent nodes: `Z_obs` alone, the longitudinal-corr-factor `z_*` pattern nodes, standalone mediator legs) → skipped; the coverage CSV is simply absent and `_diagnostics.qmd` renders no coverage sentence (the same guarded-artefact behaviour as every other optional figure).

## Implementation

- `reporting.py`: `ppc_interval_coverage`, `ppc_calibration_table`, `ppc_offfloor_rate_coverage`, `ppc_offfloor_cell_table`, and the `ppc_coverage_markdown` renderer (all logic there so the prose cannot drift from the computation, per the #320 key-findings precedent).
- `pipeline.py`: `_save_ppc` is now an orchestrator (`_run_ppc` → sample → coverage CSV + figures); the figure builders `_ppc_overlay_figure` / `_ppc_calibration_figure` / `_ppc_offfloor_figure` are guarded so a plotting hiccup never aborts a fit.
- `_partials/_diagnostics.qmd`: the Posterior-predictive section renders `ppc_calibration.png`, the (optional) overlay, and the coverage sentence from `ppc_summary.csv`.
- Guard test (`tests/statistical_models/test_ppc_coverage.py`): synthetic traces with known behaviour (perfectly-calibrated → coverage ≈ nominal; degenerate point-mass predictive → near-zero; off-floor cell in/out).
