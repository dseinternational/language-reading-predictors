# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Small release-status checks for the statistical-model CLI."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def _script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "fit_statistical_model.py"
    spec = importlib.util.spec_from_file_location("_fit_statistical_model_status", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_distinguishes_withheld_development_and_publication_runs(tmp_path):
    module = _script_module()
    ctx = SimpleNamespace(output_dir=str(tmp_path))
    decision = tmp_path / "release_decision.json"

    decision.write_text(json.dumps({"status": "artifacts_incomplete", "publishable": False}))
    assert module._completed_run_status(ctx) == "WITHHELD: artifacts_incomplete"

    decision.write_text(
        json.dumps(
            {
                "status": "ok",
                "publishable": True,
                "scientific_publication_eligible": False,
            }
        )
    )
    assert module._completed_run_status(ctx) == "ok (development-only)"

    decision.write_text(
        json.dumps(
            {
                "status": "ok",
                "publishable": True,
                "scientific_publication_eligible": True,
            }
        )
    )
    assert module._completed_run_status(ctx) == "ok"
