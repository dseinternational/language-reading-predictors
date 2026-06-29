# Statistical-model report templates: priors → prior-predictive → diagnostics → results (#125)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

**Date:** 2026-06-26. **Issue:** #125 (follow-on to #124/#127/#128; folds in #101, #130).

## Why

The Bayesian pipeline already did the expensive work — 1,000 prior-predictive
draws on every fit, full PSIS-LOO — and then discarded most of it; the reports
under-presented the Bayesian workflow (bare prior PDFs, no convergence gate, no
prior-predictive section, raw `tau_summary` dumps). The ~60 `index.qmd` files were
standalone near-duplicates, so every improvement was an N-file hand-edit. #125
closes the presentation gap and de-duplicates the templates. Scope this round:
**full**, including the Phase-3 items, with the floored-outcome (P/N)
risk-difference ROPE wired now using the **provisional** `ROPE_DELTA_PROB = 0.10`
(pending education-lead sign-off).

## What changed

**Foundation.**

- _0b — persist what's computed._ `diagnostics.save_trace` now grafts the `prior`
  and `prior_predictive` groups onto `trace.nc` (they were thrown away);
  `run_prior_predictive` samples _all_ free RVs + deterministics so the prior group
  carries the effect term and `eta` for pushforward/overlay.
  `compute_log_likelihood_and_loo` adds a `log_prior` group (for power-scaling
  sensitivity) and computes LOO `pointwise=True` (for Pareto-k). A new
  `write_diagnostics_summary` emits `diagnostics_summary.json` — the pass/fail
  convergence gate (divergences, BFMI per chain, R-hat ≤ 1.01, ESS ≥ 400). BFMI is
  computed directly from the sampler energy (`arviz.bfmi` is gone in the 1.x split).
- _0c — interval type._ `diagnostics.csv` now uses equal-tailed (`ci_kind="eti"`),
  matching the report cards and prose (finishing #101).
- _0a — Quarto partials._ New `docs/models/_partials/` (`_header`, `_setup`,
  `_convergence`, `_priors`, `_prior_predictive`, `_diagnostics`, `_footer`, plus
  per-archetype `_results_itt` / `_results_floored` / `_results_joint` /
  `_results_factors`). Each `index.qmd` is now title + model-specific Overview/Model
  prose + `{{< include _partials/… >}}`. `_copy_report_template` copies `_partials/`
  next to each report so includes resolve in the output dir at render time. Partials
  are driven by `config.json` + `measures` (no hard-coded outcome names) and guard
  every optional artifact, so one partial serves graded / floored / joint / factor
  models.

**Area 1 — priors.** `priors_table.csv` (parameter, distribution, role, rationale,
panel) is generated per model from the _registered_ RVs, so it captures inline
priors and never lists priors the model did not use; the role column makes the
"precision ≠ causal warrant" discipline structural. Dead prior panels are pruned
(only the used set is plotted). The prior is pushed through the items-scale AME
(`prior_pushforward.csv`) and overlaid against the posterior; power-scaling
sensitivity (`psense_summary.csv`) uses the new `log_prior` group.

**Area 2 — prior-predictive.** Surfaced for every family (was GF/LF only), with the
floored proportion-at-zero check available on the prior draws.

**Area 3 — diagnostics.** Convergence banner renders first; Pareto-k, rank,
ESS-evolution and LOO-PIT plots added; deterministics/HSGP weights summarised
separately for GP variants; `pair_plot`/`loo.txt`/energy demoted to a foldable
appendix.

**Area 4 — results.** Plain-language headline first; τ forest (the joint model
forests every outcome's τ in one panel); the #130 ROPE block rolled out across the
items-scale ITT outcomes, the gain family, and the level family's t2 contrast;
floored P/N get a risk-difference ROPE card (provisional δ); the joint model renders
a contrast heatmap; `evidence_label` is now attached to the plain
`tau_summary`/`factor_summary`/`did_summary` cards for uniform round-odds language;
adjusted associations are fenced in a clearly-labelled "not causal" block.

## Environment note

#131's conda/pip split shipped `h5netcdf` without its `h5py` backend, so saving
grouped `trace.nc` failed (`No module named 'h5py'`). `h5py` is now declared in
`environment.yml`. This blocked _all_ stat-model fits, not just #125.

## Follow-ups

- Set agreed δ for the floored outcomes P/N (replace the provisional 0.10) and for
  F/T (CELF concepts / TROG grammar, currently no ROPE card) with the education lead.
- The `arviz_plots.plot_psense_dist` call currently raises an internal
  "alpha already exists as coordinate" error and is skipped (the
  `psense_summary.csv` carries the numbers); revisit when arviz_plots fixes it.
- DID / mechanism / mediation keep their existing (non-thin) templates; they inherit
  the additive infra (prior-group persistence, pointwise LOO, eti diagnostics) but
  were intentionally out of the template-rollout scope this round.
