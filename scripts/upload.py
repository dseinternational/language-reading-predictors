# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Upload model output directories to Azure Blob Storage and list the URLs.

Targets:
    upload.py all              # every GB + statistical model output dir
    upload.py gb               # every output/models/<id> dir
    upload.py stat             # every output/statistical_models/models/<id>-<cfg> dir
    upload.py lrpgbl08            # a single GB model
    upload.py lrp-rli-itt-010         # a single statistical model (all its -<config> dirs)

Requires DSERESEARCH_BLOB_CONTAINER_URL and Azure auth (az login / managed
identity). Writes the full list of uploaded URLs to --urls-file.
"""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from rich.console import Console

from language_reading_predictors import paths as _paths
from language_reading_predictors.storage import upload_to_blob_storage

# Output dirs resolve via ``paths`` (#180), read at call time so ``--output-dir``
# and ``DSE_LRP_OUTPUT_DIR`` both apply.
_console = Console()


def _subdirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(d for d in root.iterdir() if d.is_dir())


def resolve_targets(target: str) -> list[tuple[str, Path]]:
    """Return (label, path) pairs for the requested target."""
    out: list[tuple[str, Path]] = []
    if target in ("all", "gb"):
        out += [(d.name, d) for d in _subdirs(_paths.gb_models_dir())]
    if target in ("all", "stat"):
        out += [(d.name, d) for d in _subdirs(_paths.stat_models_dir())]
    if target not in ("all", "gb", "stat"):
        gb = _paths.gb_models_dir() / target
        if gb.is_dir():
            out.append((gb.name, gb))
        # statistical dirs are named "<id>-<config>"
        out += [
            (d.name, d)
            for d in _subdirs(_paths.stat_models_dir())
            if d.name == target or d.name.startswith(f"{target}-")
        ]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="'all', 'gb', 'stat', or a model id.")
    parser.add_argument(
        "--include-traces",
        action="store_true",
        help="Include NetCDF trace files (.nc) in the upload (excluded by default).",
    )
    parser.add_argument(
        "--urls-file",
        default=None,
        help=(
            "Where to write the full list of uploaded blob URLs "
            "(default: <output-root>/uploaded_urls.txt)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Override the output root to upload from (highest precedence, above "
            "DSE_LRP_OUTPUT_DIR). Default: repo-local output/."
        ),
    )
    args = parser.parse_args()

    _paths.set_output_root(args.output_dir)
    _console.print(f"Output root: {_paths.describe_output_root()}")
    if args.urls_file is None:
        args.urls_file = str(_paths.output_root() / "uploaded_urls.txt")

    targets = resolve_targets(args.model.lower())
    if not targets:
        _console.print(f"[bold red]No model output directories found for: {args.model}[/bold red]")
        _console.print("Run fit_model.py / fit_statistical_model.py first.")
        raise SystemExit(1)

    run_id = str(uuid.uuid7())  # one run groups the whole batch
    _console.print(
        f"[bold]Uploading {len(targets)} model output dir(s)[/bold] "
        f"(run {run_id}); traces {'included' if args.include_traces else 'excluded'}"
    )

    all_urls: dict[str, list[str]] = {}
    reports: list[tuple[str, str]] = []
    for label, path in targets:
        urls = upload_to_blob_storage(
            str(path), label, include_traces=args.include_traces, run_id=run_id
        )
        all_urls[label] = urls
        report = next((u for u in urls if u.endswith("/index.html")), None)
        if report:
            reports.append((label, report))

    urls_path = Path(args.urls_file)
    urls_path.parent.mkdir(parents=True, exist_ok=True)
    with urls_path.open("w", encoding="utf-8") as fh:
        for label, urls in all_urls.items():
            fh.write(f"# {label}\n")
            for u in urls:
                fh.write(u + "\n")
            fh.write("\n")

    total = sum(len(u) for u in all_urls.values())
    _console.print(
        f"\n[bold green]Done[/bold green]: {total} files across {len(all_urls)} model(s). "
        f"Full URL list -> {urls_path}"
    )
    if reports:
        _console.print("\n[bold]Reports (index.html):[/bold]")
        for label, url in reports:
            _console.print(f"  {label:22s} {url}")


if __name__ == "__main__":
    main()
