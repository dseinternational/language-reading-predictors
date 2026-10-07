> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

# Shared research utilities

`dse-research-utils` supplies the scientific dependencies and shared sampling, diagnostic, figure and file helpers. [pyproject.toml](../pyproject.toml) declares the selected Git tag and extras. `uv.lock` records the resolved environment. Install that environment with `uv sync --locked`.

For local library development, use the sibling path source described in `pyproject.toml`. Restore the published source and regenerate the lock before proposing a release dependency change.

When upgrading the library, read its release and migration notes for the selected tag. The [0.17 migration guide](https://github.com/dseinternational/research/blob/v0.17.0/docs/migrating-to-0.17.md) covers public diagnostics and file-permission helpers used by this project. Check the installed distribution against the lock, run the affected tests and record any change that can affect saved-fit reuse.

Keep historical fit manifests and recorded environments intact. An environment upgrade does not certify old fits under current methods or release rules. Follow the [refit runbook](runbooks/full-statistical-model-refit.md) when new sampling or revalidation is required.
