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


@pytest.fixture
def reusable_fit(sweep, tmp_path):
    directory = tmp_path / "lrp-rli-itt-001-reporting"
    directory.mkdir()
    data_path = tmp_path / "data.csv"
    data_path.write_text("score\n1\n")
    identity = sweep.SweepIdentity("abc123", False, "environment-digest")
    config = {
        "model_id": "lrp-rli-itt-001",
        "config_name": "reporting",
        "provenance": {"source": {"commit": identity.commit, "dirty": False}},
        "environment_lock_sha256": identity.environment_sha256,
        "data_path": str(data_path),
        "data_sha256": sweep._sha256_file(data_path),
        "sampling": {
            "draws": 6000, "tune": 6000, "chains": 6,
            "target_accept": 0.95, "random_seed": 47,
        },
    }
    (directory / "release_decision.json").write_text("{}")
    (directory / "config.json").write_text(json.dumps(config))
    return directory, identity, config


def test_matching_statistical_fit_can_be_reused(sweep, reusable_fit):
    directory, identity, _ = reusable_fit
    assert sweep._reuse_reason("statistical", directory, "reporting", identity, require_render=False) is None


@pytest.mark.parametrize("target_accept", [0.95, 0.99])
def test_matching_sampler_override_can_be_reused(sweep, reusable_fit, target_accept):
    directory, identity, config = reusable_fit
    config["sampling"]["target_accept"] = target_accept
    (directory / "config.json").write_text(json.dumps(config))
    assert sweep._reuse_reason(
        "statistical", directory, "reporting", identity,
        require_render=False, target_accept=target_accept,
    ) is None


@pytest.mark.parametrize("field,value", [("draws", 100), ("tune", 100), ("chains", 2), ("random_seed", 1)])
def test_changed_numeric_sampling_settings_require_a_refit(sweep, reusable_fit, field, value):
    directory, identity, config = reusable_fit
    config["sampling"][field] = value
    (directory / "config.json").write_text(json.dumps(config))
    reason = sweep._reuse_reason("statistical", directory, "reporting", identity, require_render=False)
    assert reason is not None and field in reason


def test_archive_request_is_not_silently_skipped(sweep, reusable_fit):
    directory, identity, _ = reusable_fit
    reason = sweep._reuse_reason(
        "statistical", directory, "reporting", identity,
        require_render=False, rli_randomised_archive="archive.csv",
    )
    assert reason is not None and "archive" in reason


@pytest.mark.parametrize("field", ["data_path", "data_sha256", "provenance", "environment_lock_sha256"])
def test_missing_identity_evidence_requires_a_refit(sweep, reusable_fit, field):
    directory, identity, config = reusable_fit
    config.pop(field)
    (directory / "config.json").write_text(json.dumps(config))
    assert sweep._reuse_reason("statistical", directory, "reporting", identity, require_render=False) is not None


@pytest.mark.parametrize("dirty", [True, None, "false", 0])
def test_dirty_or_unknown_source_cannot_certify_reuse(sweep, reusable_fit, dirty):
    directory, identity, config = reusable_fit
    config["provenance"]["source"]["dirty"] = dirty
    (directory / "config.json").write_text(json.dumps(config))
    current = sweep.SweepIdentity(identity.commit, dirty, identity.environment_sha256)
    assert sweep._reuse_reason("statistical", directory, "reporting", current, require_render=False) is not None


@pytest.mark.parametrize("stored", [None, [], "bad"])
def test_malformed_config_requires_a_refit_without_crashing(sweep, reusable_fit, stored):
    directory, identity, _ = reusable_fit
    (directory / "config.json").write_text(json.dumps(stored))
    assert sweep._reuse_reason("statistical", directory, "reporting", identity, require_render=False) is not None


@pytest.mark.parametrize("provenance", [[], {"source": []}, {"source": None}])
def test_malformed_source_evidence_requires_a_refit(sweep, reusable_fit, provenance):
    directory, identity, config = reusable_fit
    config["provenance"] = provenance
    (directory / "config.json").write_text(json.dumps(config))
    assert sweep._reuse_reason("statistical", directory, "reporting", identity, require_render=False) is not None


@pytest.mark.parametrize("change", ["missing_commit", "missing_environment", "changed_data", "missing_data", "wrong_model"])
def test_unverifiable_identity_is_never_a_match(sweep, reusable_fit, change):
    directory, identity, config = reusable_fit
    if change == "missing_commit":
        config["provenance"]["source"]["commit"] = None
        identity = sweep.SweepIdentity(None, False, identity.environment_sha256)
    elif change == "missing_environment":
        config["environment_lock_sha256"] = None
        identity = sweep.SweepIdentity(identity.commit, False, None)
    elif change == "changed_data":
        Path(config["data_path"]).write_text("score\n2\n")
    elif change == "missing_data":
        Path(config["data_path"]).unlink()
    else:
        config["model_id"] = "lrp-rli-itt-002"
    (directory / "config.json").write_text(json.dumps(config))
    assert sweep._reuse_reason("statistical", directory, "reporting", identity, require_render=False) is not None


@pytest.fixture
def reusable_gb_fit(reusable_fit):
    stat_directory, identity, config = reusable_fit
    model_id = "lrp-rli-gbg-001"
    directory = stat_directory.with_name(model_id)
    directory.mkdir()
    config.update(model_id=model_id, run_config="dev")
    config.pop("config_name")
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "metrics.json").write_text(json.dumps({"model_id": model_id, "fit_complete": True}))
    return directory, identity, config


@pytest.mark.parametrize("complete", [None, False, True, "true"])
def test_boosting_resume_requires_explicit_completion(sweep, reusable_gb_fit, complete):
    directory, identity, config = reusable_gb_fit
    metrics = {"model_id": config["model_id"]}
    if complete is not None:
        metrics["fit_complete"] = complete
    (directory / "metrics.json").write_text(json.dumps(metrics))
    reason = sweep._reuse_reason(
        "gb", directory, "dev", identity, require_render=False,
    )
    assert (reason is None) == (complete is True)


@pytest.mark.parametrize("field", ["data_path", "data_sha256", "provenance", "environment_lock_sha256"])
def test_boosting_resume_also_requires_complete_provenance(sweep, reusable_gb_fit, field):
    directory, identity, config = reusable_gb_fit
    config.pop(field)
    (directory / "config.json").write_text(json.dumps(config))
    assert sweep._reuse_reason("gb", directory, "dev", identity, require_render=False) is not None


def test_boosting_resume_rejects_changed_data(sweep, reusable_gb_fit):
    directory, identity, config = reusable_gb_fit
    Path(config["data_path"]).write_text("score\n2\n")
    assert sweep._reuse_reason("gb", directory, "dev", identity, require_render=False) == "data digest changed"


@pytest.mark.parametrize("target_accept", ["0", "1", "nan"])
def test_invalid_sampler_override_fails_before_output_creation(sweep, tmp_path, target_accept):
    output = tmp_path / "untouched"
    with pytest.raises(SystemExit) as exc:
        sweep.main(["statistical", "--target-accept", target_accept, "--output-dir", str(output)])
    assert exc.value.code == 2
    assert not output.exists()


def test_sweep_honours_sampler_override_before_deciding_to_skip(sweep, reusable_fit, monkeypatch):
    directory, identity, _ = reusable_fit
    monkeypatch.setattr(sweep.SweepIdentity, "current", lambda: identity)
    monkeypatch.setattr(sweep, "_fit_dir", lambda *args: directory)
    ran = []
    monkeypatch.setattr(sweep, "_run_one", lambda *args, **kwargs: (ran.append(args[1]) or 0, 0.1))
    monkeypatch.setattr(sweep, "_stale_by_ordering", lambda *args: [])
    try:
        result = sweep.main([
            "statistical", "--models", "lrp-rli-itt-001", "--target-accept", "0.99",
            "--output-dir", str(directory.parent),
        ])
    finally:
        sweep.paths.set_output_root(None)
    assert result == 0
    assert ran == ["lrp-rli-itt-001"]


def test_resume_without_override_does_not_reuse_an_overridden_fit(sweep, reusable_fit):
    directory, identity, config = reusable_fit
    config["sampling"]["target_accept"] = 0.99
    (directory / "config.json").write_text(json.dumps(config))
    assert sweep._reuse_reason("statistical", directory, "reporting", identity, require_render=False) is not None


def test_stop_on_failure_counts_only_attempted_models(sweep, tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "_reuse_reason", lambda *args, **kwargs: "no stored fit")
    monkeypatch.setattr(sweep, "_run_one", lambda *args, **kwargs: (1, 0.1))
    messages = []
    monkeypatch.setattr(sweep, "_print", messages.append)
    examined = []
    monkeypatch.setattr(sweep, "_stale_by_ordering", lambda kind, ran, config: examined.extend(ran) or [])
    try:
        result = sweep.main([
            "gb", "--models", "first,second,third", "--stop-on-failure",
            "--output-dir", str(tmp_path),
        ])
    finally:
        sweep.paths.set_output_root(None)
    assert result == 1
    assert any("0 ok, 1 failed, 2 not run" in line for line in messages)
    assert examined == ["first"]
