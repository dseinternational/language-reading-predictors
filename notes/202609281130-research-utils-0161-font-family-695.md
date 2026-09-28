> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5.5).

<!-- cspell:ignore RCPARAMS -->

# Research utilities 0.16.1 and the shared font fallback list

[Issue #695](https://github.com/dseinternational/language-reading-predictors/issues/695) moves the `dse-research-utils` pin from tag `v0.16.0` to `v0.16.1`, which resolves to commit `874a27358080579ff851abaa77f241036322b983`, the merge commit of [upstream PR #113](https://github.com/dseinternational/research/pull/113). That PR fixed [research#112](https://github.com/dseinternational/research/issues/112), which this repository reported under [#694](https://github.com/dseinternational/language-reading-predictors/pull/694): the 0.16.0 plot style drew symbols that Noto Sans lacks, such as → and ≈, as empty boxes. The upstream account is the [0.16.1 upgrade notes](https://github.com/dseinternational/research/blob/v0.16.1/docs/migrating-to-0.16.1.md). This note records why the house font list was removed and what was measured.

## What moved

`uv lock --upgrade-package dse-research-utils` changed one package, `dse-research-utils`, from 0.16.0 (`3c5b4d99`) to 0.16.1 (`874a2735`). Nothing was added or removed. The extras and the compiled core (NumPy, numba, PyTensor and PyMC) are unchanged, and no model-building code changed, so no model-identity sweep was run.

`environment_lock_sha256` is now `8f39c722cec4a18403546f7cdbc701736351ff169c9d24f7773f9336d0fd1a8d`. Stored fits carry earlier digests, so `--reuse-trace` is refused for them, as after every earlier upgrade.

## Why the house list went

Under 0.16.0, `figure_io` set `font.family` to `["Noto Sans", "Noto Sans Math", "DejaVu Sans"]` so that matplotlib would fall back glyph by glyph. The shared style now does the same with `["sans-serif", "Noto Sans Math", "DejaVu Sans"]`, and leaves out Noto Sans Math where it is not installed. The generic `sans-serif` still resolves to Noto Sans wherever it is installed, through `font.sans-serif`.

The house list had a cost that #694 did not measure. matplotlib logs `findfont: Font family '…' not found.` each time it lays out text in a named family it cannot find. It caches the failed lookup but logs it again on every use. A two-panel test figure, with titles, axis labels, legends and a figure title, logged the following warnings:

| Fonts available to matplotlib   | House list (0.16.0) | Shared list (0.16.1) |
| ------------------------------- | ------------------- | -------------------- |
| Noto Sans and Noto Sans Math    | 0                   | 0                    |
| Noto Sans only                  | 295                 | 0                    |
| Neither, as on the CI runner    | 590                 | 0                    |

The development machine used here has Noto Sans but not Noto Sans Math, so its own fits were affected; the note for #693 had checked both fonts on a different machine. The first row was measured by loading a downloaded copy of Noto Sans Math into matplotlib for the session only, and the last by hiding both Noto fonts from matplotlib's font list. None of the cases raised a missing-glyph warning.

## What changed here

- `use_house_style()` now only applies the shared style, which sets the list itself.
- `use_house_fonts()` still sets `font.family`, because the 33 standalone scripts that call it do not apply the full style. Two of them draw symbols that Noto Sans lacks: `ability_vocab_association.py` (→) and `compare_statistical_models.py` (≥). It takes the list from `default_font_families()` when it runs, because the list depends on which fonts are installed.
- `HOUSE_FONT_FAMILIES` is gone, and `HOUSE_FONT_RCPARAMS` no longer carries `font.family`. Nothing outside `figure_io.py` used either.
- The tests compare `font.family` with `default_font_families()` instead of a fixed list. The glyph test now covers both routes, and a new test hides the Noto fonts and checks that drawing logs no `not found` lines. Against the old `figure_io.py` that test fails on the flood, so it guards the CI case on any machine.
- The agent guidance, README and refit runbook no longer say that text falls back to DejaVu Sans.

## Rendering

Where both fonts are installed, nothing changes. A figure with fallback symbols and an equation, drawn under the old list and then the new one, was pixel-identical. Where Noto Sans is missing, ordinary text now falls back through `font.sans-serif` (Helvetica Neue LT Std, Helvetica, Arial, then DejaVu Sans) rather than going straight to DejaVu Sans. That gave Helvetica Neue LT Std on the development machine; a runner with none of them still gets DejaVu Sans.

Two kinds of log line remain. Both are one-off per process, and neither comes from this change:

- **Weight lines,** such as `Failed to find font weight medium for DejaVu Sans, now using 400.` Noto Sans Math and DejaVu Sans have no medium weight, which the style uses for titles and labels. These appeared under the house list too.
- **Equation fallbacks.** The shared mathtext settings name Noto Sans and Noto Sans Math directly. Without them, the first equation in a process logs `Font family ['Noto Sans Math'] not found. Falling back to DejaVu Sans.` and similar. Three figures of six labelled panels each logged four such lines in all, because matplotlib caches that lookup.

## Checks

- Upstream PR #113 is merged, tag `v0.16.1` is on its merge commit, and its checks passed on Linux and Windows.
- `uv lock --check` passes. `uv sync --locked` installed `dse-research-utils` 0.16.1, and its `direct_url.json` records tag `v0.16.1` at commit `874a2735`.
- The log and pixel comparisons above.
- `uv run pytest`: 4,208 passed and 4 skipped. The skips need stored fits, a backup that is not in this checkout, or POSIX permission bits. `uv run mypy` (536 source files), `ruff check src/`, `npm run format:check` and `npm run spellcheck` pass.

## Not done here

Stored outputs were not regenerated. Figures drawn with both Noto fonts are unchanged, and trace reuse is refused, so there is nothing to backfill. The 0.16.0 note ([#693](https://github.com/dseinternational/language-reading-predictors/issues/693)) describes the house list as it stood then and has been left unchanged.
