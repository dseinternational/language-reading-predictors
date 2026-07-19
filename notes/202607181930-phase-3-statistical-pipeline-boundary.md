> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# Phase 3 statistical-pipeline migration boundary

## Decision

Implement issue #361 Phase 3 through a compatibility-preserving migration rather than moving all 21 statistical families in one mechanical change. The first tranche extracts the invariant execution order into `stages.py`, replaces import-all discovery with lazy model entries, scopes command-level sampling overrides to one invocation, and makes `itt.py` the reference family module for specification resolution, preprocessing arguments, effective adjustment, factory arguments, diagnostic variables and family-specific audit outputs. Existing public names in `pipeline.py` and `factories.py` remain available while other families migrate.

## Why

The current registry exposes 176 runnable modules and the historical pipeline still contains family-specific logic for 21 model kinds. Moving every builder, loader, summary and report simultaneously would produce a large relocation diff in which an equation or output-contract change would be difficult to distinguish from file movement. A reference-family migration makes the new boundaries executable and testable first, while keeping each later family move small enough to compare against its existing deterministic builder and report tests.

## Preserved invariants

- A model module's existing `fit(config)` function and `ModelSpec` remain the source of its statistical definition.
- The order of prior generation, posterior sampling, PSIS-LOO, posterior-predictive drawing, metadata writing and report finalisation is unchanged.
- `--target-accept` retains the precedence command override > model-specific default > sampling preset, but no longer replaces a function in `dse_research_utils`.
- The ITT loader receives the same resolved keyword arguments, the model factory receives the same effective arguments, and existing compatibility wrappers remain covered by the prior pipeline and registry-wide construction tests.

## Follow-on migration

Move the remaining family equations and family-only preprocessing/report helpers behind the same boundary in coherent groups, retaining compatibility re-exports until downstream scripts and tests use the family modules directly. Delete each compatibility wrapper only after repository-wide import searches and full tests show that no caller depends on it.
