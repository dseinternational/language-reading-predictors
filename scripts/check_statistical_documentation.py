# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Check or regenerate registry-derived statistical documentation counts."""

from __future__ import annotations

import argparse
from pathlib import Path

from language_reading_predictors.statistical_models.documentation import (
    assert_registry_count_snapshot,
    write_registry_count_snapshot,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = REPOSITORY_ROOT / "docs" / "models" / "registry-counts.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="Regenerate the checked registry-count snapshot.",
    )
    args = parser.parse_args()
    if args.write:
        write_registry_count_snapshot(SNAPSHOT_PATH)
        print(f"Updated {SNAPSHOT_PATH}")
        return
    assert_registry_count_snapshot(SNAPSHOT_PATH)
    print(f"Registry documentation counts are current: {SNAPSHOT_PATH}")


if __name__ == "__main__":
    main()
