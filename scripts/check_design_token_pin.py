# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify vendored Quarto theme and font bytes against their pin, offline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def check_pin(root: Path) -> list[str]:
    """Return missing, modified or unrecorded shared assets."""
    pin = json.loads((root / "docs/theme/design-tokens.pin.json").read_text(encoding="utf-8"))
    errors = []
    if pin["repository"] != "dsegroup/apps-common" or not pin["tag"].startswith("tokens/"):
        errors.append("The pin must name a DSE design-token release")
    recorded = set()
    for entry in pin["files"]:
        relative = entry["path"]
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            errors.append(f"Asset path leaves the repository: {relative}")
            continue
        if relative in recorded:
            errors.append(f"Duplicate pin entry: {relative}")
        recorded.add(relative)
        if not path.is_file():
            errors.append(f"Missing asset: {relative}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            errors.append(f"Asset differs from {pin['tag']}: {relative}")
    assets = {"docs/theme/dse.scss"}
    assets.update(path.relative_to(root).as_posix() for path in (root / "docs/fonts").rglob("*") if path.is_file())
    for relative in sorted(assets - recorded):
        errors.append(f"Asset missing from pin: {relative}")
    for relative in sorted(recorded - assets):
        errors.append(f"Pin entry outside the shared theme/fonts: {relative}")
    return errors


def main() -> int:
    errors = check_pin(Path(__file__).resolve().parents[1])
    if errors:
        print("\n".join(errors))
        return 1
    print("Shared Quarto theme, fonts and licences match the design-token pin")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
