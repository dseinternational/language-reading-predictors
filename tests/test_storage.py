# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Exercise the consumer upload path against simulated Azure clients."""

import importlib.util
import io
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import pytest
from rich.console import Console

from language_reading_predictors.storage import upload_to_blob_storage


@pytest.fixture
def uploaded(tmp_path, monkeypatch):
    names = ["assets/index.html", "index.html", "a b.csv", "a+b.csv", "50%.svg", "café.csv", "trace.nc"]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
    blobs = {}

    def upload_blob(name, data, **kwargs):
        blobs[name] = data.read()

    client = SimpleNamespace(get_container_client=lambda name: SimpleNamespace(upload_blob=upload_blob))
    monkeypatch.setattr("azure.storage.blob.BlobServiceClient", lambda *a, **k: client)
    monkeypatch.setattr("azure.identity.DefaultAzureCredential", lambda: object())
    monkeypatch.setenv("DSERESEARCH_BLOB_CONTAINER_URL", "https://acct.blob.core.windows.net/reports")
    return tmp_path, blobs, names


def test_consumer_preserves_structured_result_and_raw_paths(uploaded):
    directory, blobs, names = uploaded
    result = upload_to_blob_storage(str(directory), "model + café", run_id="test")
    assert result.relative_paths == sorted(set(names) - {"trace.nc"})
    assert result.urls == [result.prefix_url + quote(name, safe="/") for name in result.relative_paths]
    assert result.report_url == result.prefix_url + "index.html"
    assert result.uploaded_files == len(blobs) == 6
    assert result.skipped_files == 1
    assert all(value.decode() in result.relative_paths for value in blobs.values())


def test_upload_script_selects_root_report_and_writes_encoded_urls(uploaded, monkeypatch):
    directory, _, _ = uploaded
    path = Path(__file__).parents[1] / "scripts" / "upload.py"
    spec = importlib.util.spec_from_file_location("upload_660", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = upload_to_blob_storage(str(directory), "model", run_id="test")
    # Nested index sorts before the root and must not become the report link.
    assert result.urls.index(result.prefix_url + "assets/index.html") < result.urls.index(result.report_url)
    monkeypatch.setattr(module, "resolve_targets", lambda _: [("model", directory)])
    monkeypatch.setattr(module, "upload_to_blob_storage", lambda *a, **k: result)
    monkeypatch.setattr(module._paths, "set_output_root", lambda _: None)
    output = io.StringIO()
    monkeypatch.setattr(module, "_console", Console(file=output, width=240, color_system=None))
    urls_file = directory / "urls.txt"
    monkeypatch.setattr(sys, "argv", [str(path), "model", "--urls-file", str(urls_file)])
    module.main()
    report_section = output.getvalue().split("Reports (index.html):")[1]
    assert result.report_url in report_section
    assert "assets/index.html" not in report_section
    assert urls_file.read_text().splitlines()[1:-1] == result.urls


def test_nested_index_alone_does_not_become_root_report(uploaded):
    directory, _, _ = uploaded
    (directory / "index.html").unlink()
    result = upload_to_blob_storage(str(directory), "model", run_id="test")
    assert result.report_url is None
