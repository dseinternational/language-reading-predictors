# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A reporting-only calculation must verify the saved likelihood and rows."""

import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_path = Path(__file__).resolve().parents[2] / "scripts" / "regenerate_mediation_sensitivity.py"
_spec = importlib.util.spec_from_file_location("review_mediation_regeneration", _path)
assert _spec is not None and _spec.loader is not None
regeneration = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(regeneration)


@pytest.mark.parametrize(
    "field",
    [
        "data_sha256",
        "n_obs",
        "n_children",
        "resolved_run_plan",
        "fitted_subject_identity",
        "fitted_data_identity",
        "model_design_identity",
    ],
)
def test_sensitivity_regenerator_refuses_every_changed_fit_binding(monkeypatch, field):
    prepared = SimpleNamespace(data_sha256="abc", n_obs=2, n_children=2)
    built = SimpleNamespace(prepared=prepared, model=object())
    plan = SimpleNamespace(as_dict=lambda: {"score_mean_link": "three_choice_guessing_floor"})
    monkeypatch.setattr(regeneration, "fitted_subject_identity", lambda p: {"n_rows": 2})
    monkeypatch.setattr(regeneration, "_fitted_data_identity", lambda c: {"observed_digest": "xyz"})
    monkeypatch.setattr(regeneration, "_model_design_identity", lambda c: {"graph_digest": "def"})
    contract = {
        "data_sha256": "abc",
        "n_obs": 2,
        "n_children": 2,
        "resolved_run_plan": plan.as_dict(),
        "fitted_subject_identity": {"n_rows": 2},
        "fitted_data_identity": {"observed_digest": "xyz"},
        "model_design_identity": {"graph_digest": "def"},
    }
    regeneration._require_matching_fit({"reuse_contract": contract}, plan, built)
    changed = {**contract, field: None}
    with pytest.raises(ValueError, match=field):
        regeneration._require_matching_fit({"reuse_contract": changed}, plan, built)


def test_manifest_refresh_preserves_existing_write_provenance(tmp_path):
    entry = {
        "name": "primary",
        "filename": "primary.csv",
        "kind": "table",
        "required": True,
        "status": "written",
        "n_rows": 2,
        "columns": ["x"],
        "error": None,
        "error_type": None,
    }
    (tmp_path / "artifact_manifest.json").write_text(json.dumps({"artifacts": [entry]}))
    log = regeneration._existing_log(tmp_path)
    assert log.records["primary.csv"].to_json_dict() == entry
