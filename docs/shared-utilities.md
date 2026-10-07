> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).
>
> Figure colour guidance added by a LLM-based AI tool (Claude Code/Opus 5.5).

# Shared research utilities

`dse-research-utils` supplies the scientific dependencies and shared sampling, diagnostic, figure and file helpers. [pyproject.toml](../pyproject.toml) declares the selected Git tag and extras. `uv.lock` records the resolved environment. Install that environment with `uv sync --locked`.

For local library development, use the sibling path source described in `pyproject.toml`. Restore the published source and regenerate the lock before proposing a release dependency change.

When upgrading the library, read its release and migration notes for the selected tag. The [0.17 migration guide](https://github.com/dseinternational/research/blob/v0.17.0/docs/migrating-to-0.17.md) covers public diagnostics and file-permission helpers used by this project. Check the installed distribution against the lock, run the affected tests and record any change that can affect saved-fit reuse.

Version 0.18.0 takes figure colours from the DSE design tokens ([dseinternational/research#121](https://github.com/dseinternational/research/pull/121)). Take colours from `dse_research_utils.plot.styles` and name them by role in each module, such as `_INTERVENTION_COLOR = CHART_COLOURS[0]`. This project draws the immediate intervention arm in `CHART_COLOURS[0]` (blue) and the wait-list control arm in `CHART_COLOURS[2]` (orange) in every figure. Model and posterior summaries are blue and observed values drawn over them are `CHART_COLOURS[1]` (green), with their own marker. Harm, negligible and benefit take `diverging_palette(3)`, low end first: orange, grey and blue. Signed matrices, such as Spearman correlations, use the diverging scale centred on zero (`plot_heatmap(..., centre=0.0)`); non-negative ones use the sequential scale. The design language allows six categorical series, so `categorical_palette(n)` raises an error for more than six unless a matplotlib `palette` is named. Draw text in `TEXT_COLOUR` or `MUTED_TEXT_COLOUR`, never in a chart colour. The `COLOUR_*` names are deprecated.

Keep historical fit manifests and recorded environments intact. An environment upgrade does not certify old fits under current methods or release rules. Follow the [refit runbook](runbooks/full-statistical-model-refit.md) when new sampling or revalidation is required.
