# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for ``scripts/tune_models_batch.py``'s completeness check (#631).

``_is_complete`` decides whether a model's stored ``best_params.json`` was tuned
under the requested policy and can be skipped on resume. Finding 20b: it used to
compare only scoring + objective, so a study tuned under a different seed or a
smaller trial budget masqueraded as complete. It must now also compare the seed
exactly and require the recorded trial count to reach the requested budget,
tolerating a shortfall only when the study recorded a ``timeout``. Scripts
aren't on the import path, so the module is loaded by file path.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from language_reading_predictors import paths as _paths

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "tune_models_batch.py"

_MODEL_ID = "lrp-rli-gbg-012"


@pytest.fixture(scope="module")
def batch():
    spec = importlib.util.spec_from_file_location("tune_models_batch", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def tuning_root(tmp_path):
    """Redirect the output root to a temp dir; restore afterwards."""
    _paths.set_output_root(tmp_path)
    try:
        yield _paths.gb_tuning_dir()
    finally:
        _paths.set_output_root(None)


def _write_best_params(tuning_root: Path, **overrides) -> None:
    data = {
        "model_id": _MODEL_ID,
        "scoring": "mae",
        "n_trials": 150,
        "timeout": None,
        "seed": 47,
        "params": {"objective": "mae"},
    }
    data.update(overrides)
    out = tuning_root / _MODEL_ID
    out.mkdir(parents=True, exist_ok=True)
    (out / "best_params.json").write_text(json.dumps(data))


def test_missing_file_is_incomplete(batch, tuning_root):
    assert batch._is_complete(_MODEL_ID, "mae", "mae", 47, 150) is False


def test_exact_policy_match_is_complete(batch, tuning_root):
    _write_best_params(tuning_root)
    assert batch._is_complete(_MODEL_ID, "mae", "mae", 47, 150) is True


def test_scoring_or_objective_mismatch_is_incomplete(batch, tuning_root):
    _write_best_params(tuning_root)
    assert batch._is_complete(_MODEL_ID, "rmse", "mae", 47, 150) is False
    assert batch._is_complete(_MODEL_ID, "mae", "regression", 47, 150) is False


def test_seed_mismatch_is_incomplete(batch, tuning_root):
    _write_best_params(tuning_root, seed=42)
    assert batch._is_complete(_MODEL_ID, "mae", "mae", 47, 150) is False


def test_fewer_trials_without_timeout_is_incomplete(batch, tuning_root):
    # A study with fewer recorded trials and NO recorded timeout was tuned
    # under a smaller trial budget — it must re-tune.
    _write_best_params(tuning_root, n_trials=50, timeout=None)
    assert batch._is_complete(_MODEL_ID, "mae", "mae", 47, 150) is False


def test_fewer_trials_with_recorded_timeout_is_tolerated(batch, tuning_root):
    # A deliberately time-capped study can legitimately record fewer trials.
    _write_best_params(tuning_root, n_trials=50, timeout=1800.0)
    assert batch._is_complete(_MODEL_ID, "mae", "mae", 47, 150) is True


def test_missing_n_trials_is_incomplete(batch, tuning_root):
    # Pre-#631 files always recorded n_trials, but fail closed if absent.
    _write_best_params(tuning_root)
    bp = tuning_root / _MODEL_ID / "best_params.json"
    data = json.loads(bp.read_text())
    del data["n_trials"]
    bp.write_text(json.dumps(data))
    assert batch._is_complete(_MODEL_ID, "mae", "mae", 47, 150) is False


def test_more_trials_than_requested_is_complete(batch, tuning_root):
    _write_best_params(tuning_root, n_trials=200)
    assert batch._is_complete(_MODEL_ID, "mae", "mae", 47, 150) is True
