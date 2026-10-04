> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

# Shared utilities 0.17.0

The project now selects `dse-research-utils` from the published `v0.17.0` tag. The tag resolves to release commit `935bd38bdd09da05cd9895fb3ee73d38c27b3e7c`. The [library upgrade guide](https://github.com/dseinternational/research/blob/v0.17.0/docs/migrating-to-0.17.md) describes public sampling diagnostics, reductions of existing diagnostic tables, optional file-permission controls and the fix for nullable missing diagnostics.

The library's Python requirement, dependency minimums and extras are unchanged from `v0.16.2`. This project retains its existing extras, model specifications, sampling thresholds and publication rules. The lock refresh selects the new library tag while retaining unrelated package versions.

Install the updated environment with `uv sync --locked`. Both the installed distribution and `dse_research_utils.__version__` must report `0.17.0`. Use the project's existing fit and provenance checks before resuming or publishing stored results. Keep their historical manifests and recorded environments intact.

The atomic-file adapter now uses the shared mode probe and retains its 0644 fallback on a probe error. It still applies process-default permissions before the writer, so copied or explicitly chosen writer permissions remain final. Its existing private energy-diagnostic compatibility name now refers to the public helper.
