# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Install the checksum-pinned UKDS RLI archive for a local sensitivity run.

The source is open access, but its ReShare item-level licence field is blank and
ReShare's two possible open-data licences are ShareAlike variants.  Consequently
the installed CSV is gitignored and is not redistributed under this repository's
CC BY 4.0 data licence.

Examples::

    python scripts/import_rli_randomised_archive.py --zip /path/to/DSE_Data.zip
    python scripts/import_rli_randomised_archive.py --csv /path/to/dse-rli-trial-data-archive.csv
    python scripts/import_rli_randomised_archive.py --download
"""

from __future__ import annotations

import argparse
import io
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from language_reading_predictors import paths as _paths
from language_reading_predictors.statistical_models.itt_missingness import (
    RLI_ARCHIVE_CSV_NAME,
    RLI_ARCHIVE_CSV_SHA256,
    RLI_ARCHIVE_ZIP_SHA256,
    RLI_ARCHIVE_ZIP_URL,
    load_randomised_w_archive,
)

_ZIP_MEMBER = f"DSE_Data/{RLI_ARCHIVE_CSV_NAME}"


def _sha256_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _archive_csv_from_zip(payload: bytes) -> bytes:
    observed = _sha256_bytes(payload)
    if observed != RLI_ARCHIVE_ZIP_SHA256:
        raise ValueError(
            "UKDS archive checksum mismatch: "
            f"expected {RLI_ARCHIVE_ZIP_SHA256}, observed {observed}"
        )
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        try:
            return archive.read(_ZIP_MEMBER)
        except KeyError as exc:
            raise ValueError(f"UKDS archive has no member {_ZIP_MEMBER!r}") from exc


def _install(payload: bytes, destination: Path) -> None:
    observed = _sha256_bytes(payload)
    if observed != RLI_ARCHIVE_CSV_SHA256:
        raise ValueError(
            "RLI CSV checksum mismatch: "
            f"expected {RLI_ARCHIVE_CSV_SHA256}, observed {observed}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}-",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
        # Validate the staged bytes before the atomic rename, so a structurally
        # invalid source can never replace a previously usable local copy.
        load_randomised_w_archive(Path(temporary_name))
        os.replace(temporary_name, destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--zip", type=Path, help="Existing DSE_Data.zip")
    source.add_argument("--csv", type=Path, help="Existing extracted archive CSV")
    source.add_argument(
        "--download",
        action="store_true",
        help="Explicitly download the pinned open-access ZIP from UK Data Service",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=_paths.DATA_DIR / "generated" / RLI_ARCHIVE_CSV_NAME,
        help="Gitignored local destination for the extracted CSV",
    )
    args = parser.parse_args()

    if args.zip is not None:
        payload = _archive_csv_from_zip(args.zip.read_bytes())
    elif args.csv is not None:
        payload = args.csv.read_bytes()
    else:
        with urllib.request.urlopen(RLI_ARCHIVE_ZIP_URL, timeout=60) as response:
            payload = _archive_csv_from_zip(response.read())

    destination = args.destination.resolve()
    _install(payload, destination)
    print(f"Installed checksum-verified local archive: {destination}")
    print(
        "Use it with: python scripts/fit_statistical_model.py lrp-rli-itt-010 "
        f"--rli-randomised-archive {destination}"
    )


if __name__ == "__main__":
    main()
