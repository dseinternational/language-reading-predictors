# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Registry-derived contracts for statistical-model documentation."""

from __future__ import annotations

import json
from pathlib import Path

from language_reading_predictors.statistical_models.definitions import (
    KINDS,
    MODEL_REGISTRY,
)
from language_reading_predictors.statistical_models.registry import discover_models


def registry_counts() -> dict[str, int]:
    """Return the authoritative family, RLI, and all-study runnable totals."""
    runnable = discover_models()
    return {
        "model_kinds": len(KINDS),
        "rli_models": len(MODEL_REGISTRY),
        "runnable_models": len(runnable),
        "historical_rlm_models": sum(
            model_id.startswith("lrp-rlm-") for model_id in runnable
        ),
    }


def write_registry_count_snapshot(path: str | Path) -> Path:
    """Write the deterministic documentation snapshot used by CI."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_from": [
            "definitions.KINDS",
            "definitions.MODEL_REGISTRY",
            "registry.discover_models",
        ],
        **registry_counts(),
    }
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def assert_registry_count_snapshot(path: str | Path) -> None:
    """Fail with a regeneration command when documentation totals have drifted."""
    snapshot_path = Path(path)
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AssertionError(
            f"statistical registry snapshot is missing: {snapshot_path}"
        ) from exc
    expected = registry_counts()
    observed = {key: snapshot.get(key) for key in expected}
    if observed != expected:
        raise AssertionError(
            "statistical registry documentation counts have drifted: "
            f"expected {expected}, found {observed}. Run "
            "`python scripts/check_statistical_documentation.py --write`."
        )
