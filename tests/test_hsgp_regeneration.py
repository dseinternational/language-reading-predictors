# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Report regeneration cannot make an old HSGP fit current without resampling."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("name", ["regenerate_mechanism_artefacts", "regenerate_mechanism_moderation_items"])
def test_legacy_hsgp_is_refused_before_reconstruction(name, tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = {"model_id": "lrp-rli-mech-058", "kind": "mechanism", "model_settings": {"linear_mechanism": False}}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    before = config_path.read_bytes()

    def unexpected(*args, **kwargs):
        pytest.fail("legacy HSGP must be refused before model reconstruction")

    if name == "regenerate_mechanism_artefacts":
        monkeypatch.setattr(module, "_spec_for", unexpected)
        status, reason = module._regenerate(tmp_path, dry_run=False)
        assert status == "needs refit" and "#660" in reason
    else:
        monkeypatch.setattr(module, "discover_models", unexpected)
        with pytest.raises(SystemExit, match="HSGP refit required"):
            module.regenerate(tmp_path)
    assert config_path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [config_path]
