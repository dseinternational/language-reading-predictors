> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5.5).

# Research utilities 0.16.0 and the Noto Sans house fonts

[Issue #693](https://github.com/dseinternational/language-reading-predictors/issues/693) moves the `dse-research-utils` pin from tag `v0.15.2` to `v0.16.0`, which resolves to commit `3c5b4d994248ec1df2a9d2f9d677f911f10412af`, the merge commit of [upstream PR #111](https://github.com/dseinternational/research/pull/111). The release changes the shared plot style and model-graph fonts to Noto Sans and Noto Sans Math, and raises three ArviZ minimums. The extras are unchanged. The upstream account is the [0.16.0 upgrade notes](https://github.com/dseinternational/research/blob/v0.16.0/docs/migrating-to-0.16.md).

The same change applies the fonts to everything this repository produces: figures, the model graph, the hand-drawn DAGs, the 327 model reports and the book report in HTML, PDF and Word. Text is set in Noto Sans and equations in Noto Sans Math. This note records where each output gets its fonts and the choices a later reader might question.

## What moved

`uv lock --upgrade-package dse-research-utils` changed four packages and added or removed none:

| Package              | Before              | After               |
| -------------------- | ------------------- | ------------------- |
| `dse-research-utils` | 0.15.2 (`3d804d76`) | 0.16.0 (`3c5b4d99`) |
| `arviz-base`         | 1.3.0               | 1.3.1               |
| `arviz-plots`        | 1.3.1               | 1.3.2               |
| `arviz-stats`        | 1.3.2               | 1.3.3               |

The compiled core is unchanged: NumPy 2.5.3, numba 0.67.0, PyTensor 3.3.2 and PyMC 6.3.2. No model-building code changed, so no model-identity sweep was run.

SQLAlchemy stays at 2.0.54. The upstream lock moved it to 2.1.1, but Git consumers resolve their own lock and a conservative `uv lock` does not move a package nothing forces. Optuna is the only user, and `scripts/tune_model.py` creates in-memory studies with no `storage=`.

`environment_lock_sha256` is now `698ed1eb4d0b018de1e67b32826728160f8165febb1802b3c3621f47ff4230e4`. Stored fits carry an earlier digest (`lrp-rli-al-001-reporting` has `7a0e2caa…`), so `--reuse-trace` is refused for them, as under [#667](https://github.com/dseinternational/language-reading-predictors/issues/667) and [#669](https://github.com/dseinternational/language-reading-predictors/issues/669).

## Figures

The shared style now sets `font.sans-serif` to Noto Sans first and a custom mathtext font set: Noto Sans Math for upright symbols, Noto Sans italic and bold for variables and `\mathbf`, and STIX Sans as the fallback. `\mathcal{N}` renders as an upright N under this set. No Python label here uses `\mathcal`, `\mathscr` or `\mathbb`, so nothing needed changing.

### Symbols that Noto Sans lacks

A dev fit of `lrp-rli-itt-001` showed a problem that the library change alone would have shipped. The predicted-score figure printed "average effect ▯ +1.4 items": an empty box where "≈" should be. Noto Sans has no arrows or mathematical relations. Fourteen non-ASCII characters in this repository's Python are missing from it, including → (28 files), ≈ (46 files), ≤, ≥, ↔ and ✓. Source Sans 3 had most of them, which is why this did not show before. matplotlib falls back glyph by glyph only across families named in `font.family`, and the shared style names only the generic `sans-serif`, which resolves to a single font.

This repository therefore sets `font.family` to Noto Sans, Noto Sans Math and DejaVu Sans, in that order. Noto Sans Math is designed to pair with Noto Sans and has all of those symbols except ⚠, which DejaVu Sans, bundled with matplotlib, supplies. Without the Noto fonts, DejaVu Sans draws everything, as before. Drawing those symbols under the shared style alone raised six missing-glyph warnings; under the house style it raises none, and a test now checks this. The fix belongs upstream too, so that other consumers of the shared style get it.

### Where each figure gets its fonts

`figure_io.use_house_style()` applies the shared style and then the family list above. Fits reach it through `init_plotting()` (statistical models) and `scripts/fit_model.py` (gradient boosting), which previously called the library's `setup.init_script()`. That function only applied the shared style, so nothing else changed. Two further groups of scripts did not use the style at all:

- **Scripts that redraw figures in fit directories.** `regenerate_itt_contrast_figures.py`, `regenerate_mechanism_artefacts.py`, `regenerate_psense.py` and `blocks_vocab_gb_diagnostic.py` drew in matplotlib's defaults, so a backfilled figure did not match the one its fit had drawn. They now apply the full house style. This changes more than the font, which is deliberate: a redrawn figure should match a fresh fit's.
- **Standalone scripts that lay out their own figures.** The descriptive and exploratory plots, `compare_statistical_models.py`, `design_analysis.py`, `predictability_readout.py`, `lrp_rli_gbl_012_weight_sensitivity.py`, `ability_vocab_association.py` and `learn_itt_model.py` size their figures and call `tight_layout`. The full style turns on constrained layout, which would conflict. They call the new `figure_io.use_house_fonts()`, which sets the family list and copies the mathtext settings from the shared style dictionary, so the two cannot drift. Sizes and layout are unchanged.

The archived probe script under `notes/assets/` was left alone because it reproduces a dated figure. Notebooks apply the full style through `init_workbook()`.

## Graphs

`model_to_graphviz` now labels model graphs in `Noto Sans,sans-serif`. The comment in `statistical_models/publication.py` that credited the shared helper with Helvetica was updated.

The hand-drawn DAGs in `dag/` moved from Helvetica to Noto Sans. Bold text needed a different mechanism. Graphviz maps its PostScript names, such as `Helvetica-Bold`, to a CSS family and weight in SVG output, but copies any other font name unchanged. A browser cannot match `font-family="Noto Sans Bold"` to a font, so bold titles, cluster labels and outcome nodes now use HTML-like `<B>` labels, which Graphviz writes as `font-weight="bold"`. `<BR/>` spaces lines by font size, which is tighter than Noto Sans's line height, so multi-line bold labels use a one-column table. Their spacing then matches plain labels to the point. One edge label in the Byrne DAG had fallen back to Graphviz's default Times and now uses Noto Sans.

The re-rendered SVGs contain exactly the same text as before and the same number of bold runs. They are 2–10% larger because Noto Sans is wider and taller than Helvetica. The PNGs keep their earlier resolutions: 96 dpi for the two lagged RLI graphs, 180 dpi for the Byrne graphs and 200 dpi for the report DAG. The report DAG is only rendered to PNG, where Pango resolves `Noto Sans Bold`, so it keeps a bold font name; a comment in the source records that SVG output would need `<B>` labels.

## Reports

**HTML model reports.** `docs/models/_partials/_fonts.scss` layers the fonts over the cosmo theme. It sets the Bootstrap sans-serif stack to Noto Sans, imports Noto Sans and Noto Sans Math from Google Fonts, and disables cosmo's own Source Sans Pro import. All 327 templates now declare `theme: [cosmo, _partials/_fonts.scss]`. The statistical report step already copies `_partials`; the gradient-boosting report step now copies the stylesheet too. The artefact manifest classifies `.scss` as report material.

**Equations in HTML.** MathJax draws equations in its own fonts and cannot use Noto Sans Math. The templates therefore set `html-math-method: mathml`, and the stylesheet gives `math` elements the Noto Sans Math family, which browsers use for MathML layout. The switch was checked before it was made:

- Pandoc converted every equation in all 376 report sources and partials to MathML without a warning. The equations that partials generate at render time use only the same commands.
- Google Fonts serves Noto Sans Math with its OpenType MATH table, which MathML layout needs. The served file had 4,060 glyphs and the MATH table.
- MathML does not break long lines. A long display equation widened the whole page in the test render, so display equations now scroll within their own box.

**The book report.** The PDF sets `mainfont` and `sansfont` to Noto Sans and `mathfont` to Noto Sans Math, which pandoc's template loads with `unicode-math`. The HTML uses the same stylesheet and MathML. `docs/template.docx` now uses Noto Sans for its theme fonts and font table, and Noto Sans Math as its math font.

Pandoc copies only some settings from a Word reference document, and the math font is not one of them, so a rendered report used Word's Cambria Math whatever the template said. `scripts/set_docx_math_font.py` runs as a Quarto `post-render` step and writes the math font into each rendered `.docx`. It uses only the standard library because Quarto may run it outside the project environment, and it edits the XML as text so that Word's namespace declarations survive.

## Checks

- `uv lock --check` passes, `uv sync --locked` installed `dse-research-utils` 0.16.0 from commit `3c5b4d99`, and matplotlib resolves Noto Sans (regular, italic, bold) and Noto Sans Math on this machine after its font cache was cleared.
- A stored fit (`lrp-rli-al-001-reporting`) was copied to scratch with the new template and rendered. The browser reported Noto Sans for body text and headings, Noto Sans Math for equations, and both web fonts loaded.
- The book rendered in all three formats. The PDF embeds Noto Sans, Noto Sans Bold, Noto Sans Italic and Noto Sans Math. Word exported the rendered `.docx` to PDF with NotoSansMath-Regular embedded for the equations and NotoSans-Regular for text.
- Dev fits of `lrp-rli-itt-001`, `lrp-rli-did-001` and `lrp-rli-gbg-001` into a scratch output root drew their figures, model graph and report in the new fonts with no missing-glyph warnings. No figure uses the smallest size presets, and none of the figures reviewed clipped a title, label or legend. Each run logs two one-off notices that Noto Sans Math and DejaVu Sans have no medium weight, which the shared style uses for axis titles and labels; matplotlib then uses their regular weight, which only affects fallback symbols.
- `uv run pytest`: 4,206 passed and 3 skipped. `uv run mypy` (536 source files), `ruff check src/`, `npm run format:check` and spelling on the tracked Markdown and Quarto files pass.
- New tests cover the house font settings and glyph fallback, `use_house_fonts()`, every model template's front matter, the gradient-boosting stylesheet copy, the book configuration and the Word math-font step.

## Not done here

Stored outputs were not regenerated. Existing figures and reports keep their old fonts until the models are refitted, and trace reuse is refused, so figures cannot be backfilled without sampling. A stored fit's report also renders from the `index.qmd` copied at fit time. **Whether and when to refit is a decision for the maintainer.** The [refit runbook](../docs/runbooks/full-statistical-model-refit.md) now checks that both fonts resolve before a run, because matplotlib falls back to DejaVu Sans without failing a fit.

CI does not install the fonts. Tests do not depend on them, but any figure CI draws is in DejaVu Sans.
