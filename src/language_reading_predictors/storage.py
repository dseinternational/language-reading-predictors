# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Upload model output directories to Azure Blob Storage."""

from __future__ import annotations

from rich.console import Console

DEFAULT_PROJECT = "language-reading-predictors"
_console = Console()


def upload_to_blob_storage(
    output_dir: str,
    model_label: str,
    *,
    project: str = DEFAULT_PROJECT,
    include_traces: bool = False,
    run_id: str | None = None,
) -> list[str]:
    """Upload ``output_dir`` to blob storage and return uploaded blob URLs.

    NetCDF trace files (``.nc``) are excluded by default because of their size;
    pass ``include_traces=True`` to include them. Authentication and container
    URL validation are handled by :mod:`dse_research_utils.storage.azure`, which
    authenticates with ``DefaultAzureCredential`` and writes to the container
    named in the ``DSERESEARCH_BLOB_CONTAINER_URL`` environment variable.
    """
    from dse_research_utils.storage.azure import upload_directory_to_blob_storage

    result = upload_directory_to_blob_storage(
        output_dir,
        model_label,
        project=project,
        include_traces=include_traces,
        run_id=run_id,
    )

    _console.print(
        f"[green]Uploaded[/green] [bold]{model_label}[/bold]: {result.uploaded_files} files "
        f"({result.bytes_uploaded / 1_000_000:.1f} MB, {result.skipped_files} .nc skipped, "
        f"{result.elapsed_seconds:.1f}s)"
    )
    _console.print(f"  prefix: {result.prefix_url}")
    if result.report_url:
        _console.print(f"  report: {result.report_url}")
    return result.urls
