> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

# The psense figure never rendered for any model with an `alpha` parameter (#340)

**Date:** 2026-07-25

## What was wrong

`run_psense` (`src/language_reading_predictors/statistical_models/diagnostics.py`) writes two artefacts: the numeric `psense_summary.csv` and the `psense.png` / `psense.svg` figure. The figure call raised `ValueError: alpha already exists as coordinate or variable name` for every model whose posterior contains a variable called `alpha` — which, in this suite, means the intercept, and therefore nearly every family. `_save_pc` swallows the exception by design (a plotting hiccup must not abort a fit), so each affected run printed one yellow `psense.png skipped: …` line and carried on with the CSV intact.

The clash is inside `arviz_stats.power_scale_dataset`, which `arviz_plots.plot_psense_dist` calls before it applies `var_names`. It resamples the **whole** posterior group at three power-scaling factors and concatenates the three copies along a _new_ dimension it names `alpha`, then labels that dimension with `assign_coords(alpha=…)`. If the posterior already carries a variable of that name, the `assign_coords` fails. Two consequences follow, and both matter for how the bug reads:

- **Changing `var_names` cannot avoid it.** The resampling happens before the selection, so the collision is with the variable's mere presence in the trace, not with what the caller asked to plot. Every `run_psense` call site in `pipeline.py` was already passing a curated list; none of them helped.
- **`psense_summary.csv` was unaffected.** `arviz_stats.psense_summary` takes a different path and never builds the concatenated dataset, which is why the numbers were always there and only the picture was missing.

## Blast radius

Of the 179 fitted model directories in `output/statistical_models/models/`, 68 carry `psense_summary.csv` and exactly **one** carries `psense.png` — `lrp-rli-lcf-001`, the `corr_factor` model, whose posterior has no `alpha` (it is a pure measurement model) and whose psense selection is `factor_corr_pairs` / `trait_share`. Spot-checking `itt`, `gf`, `lf`, `did` and `bx` traces confirms an `alpha` variable in all of them. So the figure has been missing from 67 of 68 fits since the diagnostic was introduced in #125.

No published report links a broken image. `docs/models/_partials/_priors.qmd` emits the figure through `_img()`, which returns an empty string when the file is absent, so the psense callout has simply been rendering table-only. Exactly one rendered `index.html` (lcf-001's) references `psense.png`.

The gap is therefore **presentational, not evidential** — the power-scaling numbers that the floor-rule release gate reads (`tau_psense_status`, `_results_floored.qmd`) come from the CSV and were never affected.

## Not the same thing as the coverage gap in the prior review

`notes/202607211500-prior-critical-review.md` records that psense ran for only six of 22 families. That is a _different_ and now-stale observation: `run_psense` was extended to the estimand families in #381/#408 (commit `286adf1`, 2026-07-23), and the fits that lack `psense_summary.csv` all predate it (`al`, `mech`, `ca` directories are dated 2026-07-21). Running `arviz_stats.psense_summary` against those saved traces today succeeds. Those families will emit psense on their next refit; nothing further is needed in code.

## The fix

Two helpers in `diagnostics.py`, both applied only to the figure path:

1. `_psense_plot_view` renames any posterior variable that collides with a dimension `plot_psense_dist` introduces (`alpha`, and defensively `component_group` and `sample`), and maps `var_names` through the same rename so a requested parameter keeps its panel. The renamed label is `alpha (parameter)`, which reads unambiguously against the power-scaling α in the figure legend. `psense_summary.csv` is still computed from the untouched trace and keeps the original names.

   Renaming rather than dropping or subsetting, for two reasons: `alpha` is itself in the requested selection for several families (`horseshoe`, `adjusted`, `mediation`, `correlated_factor`), and `arviz_stats.extract` returns a bare `DataArray` — which `plot_psense_dist` cannot consume — when a posterior is cut down to a single variable.

2. `_psense_layout` sizes the figure by its row count. `plot_psense_dist` puts one row per parameter (times the levels of any non-sampling coordinate) against two fixed columns, and its auto-sized default barely grows with that count: a five-row selection got ~0.4 inches of plotting area per panel, flattening every density to a line. Rows now get ~2 inches each, capped at 36 inches total, and ArviZ's 40-panel guard is raised for selections that exceed it.

   **Only `figsize` is set, deliberately.** The house style (`set_matplotlib_default_style`, applied at the fit entry point) enables matplotlib's constrained layout, which handles inter-row spacing and makes room for the `suptitle` that `save_plotcollection` adds — and which _overrides_ `gridspec_kw`. An earlier attempt that passed `hspace` and `top` alongside `figsize` measured **worse** than the default (0.18 inch panels against 0.42), because constrained layout discarded the spacing hints while keeping the taller canvas. The mistake is easy to repeat; hence this paragraph.

## Verification

Reproduced on a synthetic three-variable trace and on saved `itt`/`gf`/`did`/`lf`/`bx` reporting traces (all raise before, all render after; `lcf-001`, which already worked, is left untouched — no rename is applied when there is no clash). End-to-end `--config dev` fits of `lrp-rli-itt-001` (one row), `lrp-rli-bx-001` (one row) and `lrp-rli-adj-065` (15 rows) all emit `psense.png` and `psense.svg` with no skip warning. Regression tests in `tests/statistical_models/test_diagnostics.py` assert the raise-before / render-after contract directly against `plot_psense_dist`, so a future arviz_plots release that fixes the underlying behaviour will surface as a failing test rather than silently dead code.

## Follow-up

Existing fits keep their missing figures — the artefacts are only regenerated by a refit. `psense.png` is a secondary diagnostic and the substantive numbers are already published, so this does not on its own justify a refit sweep; it will be picked up by the next full `--config reporting` run.
