# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Upload model output directories to Azure Blob Storage.

Mirrors the vocabulary-growth project's uploader. Authenticates with
``DefaultAzureCredential`` (so ``az login`` or a managed identity both work)
and writes to the container named in ``DSERESEARCH_BLOB_CONTAINER_URL``, under
the prefix ``projects/<project>/output/<run_id>/<model_label>/``. Returns the
list of uploaded blob URLs so callers can surface them for review.

NetCDF trace files (``.nc``) are excluded by default because of their size;
pass ``include_traces=True`` to include them.
"""

from __future__ import annotations

import mimetypes
import os
import time
import uuid
from urllib.parse import urlparse

from rich.console import Console

DEFAULT_PROJECT = "language-reading-predictors"
_ENV_VAR = "DSERESEARCH_BLOB_CONTAINER_URL"
_console = Console()


def _container_url() -> str:
    url = os.environ.get(_ENV_VAR)
    if not url:
        raise RuntimeError(
            f"{_ENV_VAR} environment variable is not set. Set it to your Azure "
            "Blob container URL, e.g. "
            "'https://<account>.blob.core.windows.net/<container>'."
        )
    return url


def upload_to_blob_storage(
    output_dir: str,
    model_label: str,
    *,
    project: str = DEFAULT_PROJECT,
    include_traces: bool = False,
    run_id: str | None = None,
) -> list[str]:
    """Upload ``output_dir`` to blob storage; return the uploaded blob URLs.

    Parameters
    ----------
    output_dir : str
        Local directory of model artifacts to upload.
    model_label : str
        Label used as the final path segment of the blob prefix (e.g. the
        model id, or ``"<id>-<config>"`` for statistical models).
    project : str
        Project segment of the blob prefix.
    include_traces : bool
        If True, include NetCDF trace files (``.nc``). Excluded by default.
    run_id : str | None
        Groups an upload batch under one prefix. Generated (uuid7) if omitted,
        so each call gets its own timestamped run.
    """
    # Lazy import so the package imports without the azure SDK installed.
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient, ContentSettings

    container_url = _container_url()
    parsed = urlparse(container_url.rstrip("/"))
    account_url = f"{parsed.scheme}://{parsed.netloc}"
    container_name = parsed.path.lstrip("/")
    base_url = container_url.rstrip("/")

    if run_id is None:
        run_id = str(uuid.uuid7())
    blob_prefix = f"projects/{project}/output/{run_id}/{model_label}"

    credential = DefaultAzureCredential()
    container_client = BlobServiceClient(
        account_url, credential=credential
    ).get_container_client(container_name)

    urls: list[str] = []
    uploaded = skipped = 0
    bytes_sent = 0
    started = time.perf_counter()
    for root, _dirs, files in os.walk(output_dir):
        for filename in files:
            if not include_traces and filename.endswith(".nc"):
                skipped += 1
                continue
            local_path = os.path.join(root, filename)
            relative_path = os.path.relpath(local_path, output_dir).replace("\\", "/")
            blob_name = f"{blob_prefix}/{relative_path}"
            content_type, _ = mimetypes.guess_type(filename)
            content_type = content_type or "application/octet-stream"
            bytes_sent += os.path.getsize(local_path)
            with open(local_path, "rb") as fh:
                container_client.upload_blob(
                    blob_name,
                    fh,
                    overwrite=True,
                    content_settings=ContentSettings(content_type=content_type),
                )
            urls.append(f"{base_url}/{blob_name}")
            uploaded += 1

    report_url = next((u for u in urls if u.endswith("/index.html")), None)
    elapsed = time.perf_counter() - started
    _console.print(
        f"[green]Uploaded[/green] [bold]{model_label}[/bold]: {uploaded} files "
        f"({bytes_sent / 1_000_000:.1f} MB, {skipped} .nc skipped, {elapsed:.1f}s)"
    )
    _console.print(f"  prefix: {base_url}/{blob_prefix}/")
    if report_url:
        _console.print(f"  report: {report_url}")
    return urls
