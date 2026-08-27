# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The sweep driver's end-of-run check for decisions stale by fit order.

A release decision is written once, at fit time, from whatever its companions
looked like *then*. A sweep that fits a parent before its registered companion
therefore leaves the parent qualified against a companion that now exists, and
nothing revisits it.

That is not hypothetical: it caught ``lrp-rlm-jc-002`` twice. The 2026-08-26
full-registry batch fitted it 2m20s before ``lrp-rlm-jc-102``, and the 2026-08-27
tail repeated the pattern at 24/25 and 25/25 — both times publishing "its own
release decision withholds publication" about a companion that was fitted,
converged and publishable.

Scripts are not on the import path in this repo, so the module is loaded by file
path, matching ``test_regenerate_key_findings``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before executing: this module defines a dataclass, and
    # ``dataclasses`` resolves field types through ``sys.modules[cls.__module__]``,
    # which is absent for a module loaded purely by file path.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


@pytest.fixture(scope="module")
def sweep():
    return _load("run_refit_sweep")


def _fit(root: Path, model_id: str, qualification: str = "") -> Path:
    directory = root / f"{model_id}-reporting"
    directory.mkdir(parents=True)
    record: dict[str, object] = {"status": "ok", "publishable": True}
    if qualification:
        record["publication_qualification"] = qualification
    (directory / "release_decision.json").write_text(json.dumps(record))
    return directory


@pytest.fixture
def models_root(tmp_path, monkeypatch, sweep):
    root = tmp_path / "statistical_models" / "models"
    root.mkdir(parents=True)
    monkeypatch.setattr(sweep.paths, "stat_models_dir", lambda: root)
    return root


def test_a_parent_qualified_against_a_later_model_in_the_same_sweep_is_flagged(
    sweep, models_root
):
    _fit(
        models_root,
        "lrp-rlm-jc-002",
        "the registered within-scale prior sensitivity (lrp-rlm-jc-102) is not "
        "release-ready beside this fit",
    )
    _fit(models_root, "lrp-rlm-jc-102")

    lines = sweep._stale_by_ordering(
        "statistical", ["lrp-rlm-jc-002", "lrp-rlm-jc-102"], "reporting"
    )

    assert any("lrp-rlm-jc-002 qualified against lrp-rlm-jc-102" in x for x in lines)
    # The remedy is named, because the capability already exists.
    assert any("regenerate_key_findings.py" in x for x in lines)


def test_a_cleared_qualification_is_silent(sweep, models_root):
    _fit(models_root, "lrp-rlm-jc-002")
    _fit(models_root, "lrp-rlm-jc-102")

    assert (
        sweep._stale_by_ordering(
            "statistical", ["lrp-rlm-jc-002", "lrp-rlm-jc-102"], "reporting"
        )
        == []
    )


def test_a_qualification_naming_a_model_outside_the_sweep_is_silent(
    sweep, models_root
):
    """Only *ordering* is in scope here.

    A fit qualified against a model this sweep never touched is qualified for some
    other reason — a genuinely unfitted companion, say — and re-deriving its
    decision would change nothing. Flagging it would train the reader to ignore
    the warning.
    """
    _fit(
        models_root,
        "lrp-rlm-jc-002",
        "the registered within-scale prior sensitivity (lrp-rlm-jc-102) is not "
        "release-ready beside this fit",
    )

    assert sweep._stale_by_ordering("statistical", ["lrp-rlm-jc-002"], "reporting") == []


def test_the_gradient_boosting_layer_is_out_of_scope(sweep, models_root):
    """GB fits carry no release decision, so there is nothing to go stale."""
    _fit(
        models_root,
        "lrp-rlm-jc-002",
        "... (lrp-rlm-jc-102) is not release-ready beside this fit",
    )
    _fit(models_root, "lrp-rlm-jc-102")

    assert (
        sweep._stale_by_ordering(
            "gb", ["lrp-rlm-jc-002", "lrp-rlm-jc-102"], "reporting"
        )
        == []
    )


def test_an_unreadable_decision_does_not_break_the_sweep_summary(sweep, models_root):
    """The warning runs after a completed sweep; it must never mask the result."""
    directory = models_root / "lrp-rlm-jc-002-reporting"
    directory.mkdir(parents=True)
    (directory / "release_decision.json").write_text("{ not json")
    _fit(models_root, "lrp-rlm-jc-102")

    assert (
        sweep._stale_by_ordering(
            "statistical", ["lrp-rlm-jc-002", "lrp-rlm-jc-102"], "reporting"
        )
        == []
    )
