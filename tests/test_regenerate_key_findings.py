# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Target resolution for the scripts that walk the statistical-model output root.

The regenerate backfills re-run a fit-time generator over already-published output
directories, and ``upload.py`` publishes them, so the one thing each must get right
before touching a file is *which* directories are published.
``StatisticalFitContext.reset_output_dir`` stages every run in a hidden sibling and
promotes it only after the last stage succeeds, so the output root routinely holds
dotted directories that are either in flight or abandoned. Walking into one writes
artefacts that are about to be discarded, races a live fit, or — for the upload
path — publishes the half-written output of a run that was never accepted. Hence
the shared exclusion pinned here for every script that walks that root.

Scripts aren't on the import path in this repo, so the modules are loaded by file
path. ``regenerate_psense.py`` carries the same rule and pins it in its own tests.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"

# The published directory each script would accept, paired with the staging sibling
# it must not: the mediation backfill narrows to the model ids it can calibrate, so
# a generic name would be filtered out for reasons unrelated to what is under test.
_WALKERS = [
    ("regenerate_key_findings", "lrp-rli-itt-010-reporting"),
    ("regenerate_itt_contrast_figures", "lrp-rli-itt-010-reporting"),
    ("regenerate_mediation_calibration", "lrp-rli-med-059-reporting"),
    ("upload", "lrp-rli-itt-010-reporting"),
]

# ``upload.py`` resolves to (label, path) pairs across both output roots, so it is
# covered by its own end-to-end test below rather than this shared signature.
_BACKFILLS = [entry for entry in _WALKERS if entry[0] != "upload"]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def regen():
    return _load("regenerate_key_findings")


def test_targets_exclude_in_flight_output_transactions(regen, tmp_path, monkeypatch):
    """Fits stage into a hidden ``.<id>-<config>.staging-XXXX`` sibling and are
    promoted only on success. Regenerating into one writes artefacts that are about
    to be discarded, or races a live fit."""
    from language_reading_predictors import paths as _paths

    root = tmp_path / "models"
    (root / "lrp-rli-itt-010-reporting").mkdir(parents=True)
    (root / ".lrp-rli-itt-010-reporting.staging-9hbfh_9x").mkdir(parents=True)
    monkeypatch.setattr(_paths, "stat_models_dir", lambda: root)

    names = [d.name for d in regen.resolve_targets("all")]
    assert names == ["lrp-rli-itt-010-reporting"]


def test_a_model_id_target_excludes_its_own_staging_directory(
    regen, tmp_path, monkeypatch
):
    """The single-model form matches on the ``<id>-`` prefix, so it must not be
    satisfied by a staging directory carrying that id — the run whose artefacts it
    holds has not been published, and may still be writing them."""
    from language_reading_predictors import paths as _paths

    root = tmp_path / "models"
    (root / "lrp-rli-itt-010-dev").mkdir(parents=True)
    (root / "lrp-rli-itt-010-reporting").mkdir(parents=True)
    (root / ".lrp-rli-itt-010-reporting.staging-9hbfh_9x").mkdir(parents=True)
    (root / "lrp-rli-itt-011-reporting").mkdir(parents=True)
    monkeypatch.setattr(_paths, "stat_models_dir", lambda: root)

    names = [d.name for d in regen.resolve_targets("lrp-rli-itt-010")]
    assert names == ["lrp-rli-itt-010-dev", "lrp-rli-itt-010-reporting"]


def test_backup_directories_are_excluded_too(regen, tmp_path, monkeypatch):
    """Publication keeps the superseded fit as a hidden ``.<name>.backup-XXXX`` until
    it ages out. It is a *previous* fit, not the current one, so regenerating into it
    would refresh artefacts no report reads."""
    from language_reading_predictors import paths as _paths

    root = tmp_path / "models"
    (root / "lrp-rli-itt-010-reporting").mkdir(parents=True)
    (root / ".lrp-rli-itt-010-reporting.backup-3kd0waz1").mkdir(parents=True)
    monkeypatch.setattr(_paths, "stat_models_dir", lambda: root)

    names = [d.name for d in regen.resolve_targets("all")]
    assert names == ["lrp-rli-itt-010-reporting"]


def test_regeneration_refreshes_release_decision_and_findings(
    regen, tmp_path, monkeypatch
):
    """A legacy non-RLI fit must not retain a stale publishable decision."""
    from language_reading_predictors import paths as _paths

    root = tmp_path / "models"
    fit = root / "lrp-rlm-hg-001-reporting"
    fit.mkdir(parents=True)
    (fit / "diagnostics_summary.json").write_text(
        json.dumps(
            {
                "passed": True,
                "checks": {
                    "rhat": True,
                    "ess": True,
                    "divergences": True,
                    "bfmi": True,
                },
                "divergences": 0,
                "max_rhat": 1.001,
                "min_ess": 1000.0,
                "bfmi_per_chain": [0.8, 0.9],
            }
        )
    )
    (fit / "config.json").write_text(
        json.dumps(
            {
                "model_id": "lrp-rlm-hg-001",
                "kind": "historical_growth",
                "study_id": "rlm",
                "config_name": "reporting",
            }
        )
    )
    (fit / "release_decision.json").write_text(
        json.dumps({"status": "ok", "publishable": True})
    )
    monkeypatch.setattr(_paths, "stat_models_dir", lambda: root)
    monkeypatch.setattr(sys, "argv", ["regenerate_key_findings.py", fit.name])

    regen.main()

    decision = json.loads((fit / "release_decision.json").read_text())
    findings = json.loads((fit / "key_findings.json").read_text())
    assert decision["status"] == "inputs_unresolved"
    assert decision["publishable"] is False
    assert findings["status"] == "inputs_unresolved"


@pytest.mark.parametrize(("module_name", "published"), _WALKERS)
def test_every_output_root_walker_skips_hidden_transactions(
    module_name, published, tmp_path
):
    """The same output root is walked by several scripts, and the exclusion has to
    hold in each — one that keeps the raw ``iterdir`` reintroduces the hazard on its
    own. ``_subdirs`` is asserted directly because a script whose own id filter
    happens to exclude dotted names would otherwise pass without carrying the rule."""
    module = _load(module_name)
    root = tmp_path / "models"
    (root / published).mkdir(parents=True)
    (root / f".{published}.staging-9hbfh_9x").mkdir(parents=True)

    assert [d.name for d in module._subdirs(root)] == [published]


@pytest.mark.parametrize(("module_name", "published"), _BACKFILLS)
def test_backfill_targets_resolve_to_published_dirs_only(
    module_name, published, tmp_path, monkeypatch
):
    from language_reading_predictors import paths as _paths

    module = _load(module_name)
    root = tmp_path / "models"
    (root / published).mkdir(parents=True)
    (root / f".{published}.staging-9hbfh_9x").mkdir(parents=True)
    monkeypatch.setattr(_paths, "stat_models_dir", lambda: root)

    assert [d.name for d in module.resolve_targets("all")] == [published]


def test_upload_targets_exclude_in_flight_output_transactions(tmp_path, monkeypatch):
    """``upload.py`` pushes each resolved directory to blob storage under its own
    name, so an unfiltered walk publishes the half-written artefacts of a run that
    was never promoted — under a label no report or comparison refers to."""
    from language_reading_predictors import paths as _paths

    upload = _load("upload")
    stat_root = tmp_path / "statistical_models" / "models"
    gb_root = tmp_path / "models"
    (stat_root / "lrp-rli-itt-010-reporting").mkdir(parents=True)
    (stat_root / ".lrp-rli-itt-010-reporting.staging-9hbfh_9x").mkdir(parents=True)
    gb_root.mkdir(parents=True)
    monkeypatch.setattr(_paths, "stat_models_dir", lambda: stat_root)
    monkeypatch.setattr(_paths, "gb_models_dir", lambda: gb_root)

    assert [label for label, _ in upload.resolve_targets("all")] == [
        "lrp-rli-itt-010-reporting"
    ]
    assert [label for label, _ in upload.resolve_targets("lrp-rli-itt-010")] == [
        "lrp-rli-itt-010-reporting"
    ]


@pytest.mark.parametrize(("module_name", "published"), _WALKERS)
def test_missing_output_root_resolves_to_no_targets(module_name, published, tmp_path):
    """A run against an output root that was never created (a fresh checkout, or a
    scratch disk that has been torn down) reports no targets rather than raising."""
    module = _load(module_name)
    assert module._subdirs(tmp_path / "absent") == []


def test_itt_contrast_backfill_preserves_the_registered_score_mean_link(
    tmp_path, monkeypatch
):
    """The 108 backfill must not overwrite guessing-floor artefacts as logit."""

    module = _load("regenerate_itt_contrast_figures")
    fit_dir = tmp_path / "lrp-rli-itt-108-reporting"
    fit_dir.mkdir()
    (fit_dir / "config.json").write_text(
        json.dumps(
            {
                "kind": "itt",
                "ci_prob": 0.89,
                "sampling": {"random_seed": 47},
                "resolved_run_plan": {
                    "outcome_symbol": "B",
                    "floor_rule": False,
                    "headline_likelihood": "beta_binomial",
                    "score_mean_link": "three_choice_guessing_floor",
                    "tau_moderator_symbol": None,
                },
            }
        )
    )
    (fit_dir / "trace.nc").write_text("placeholder")
    trace = SimpleNamespace(
        constant_data={"G": SimpleNamespace(values=np.array([0.0, 1.0]))}
    )
    monkeypatch.setattr(module.az, "from_netcdf", lambda _path: trace)
    monkeypatch.setattr(module, "_PARTIALS_SRC", tmp_path / "no-partials")
    calls: dict[str, str] = {}

    def predicted(*_args, **kwargs):
        calls["predicted"] = kwargs["score_mean_link"]

    def overlap(*_args, **kwargs):
        calls["overlap"] = kwargs["score_mean_link"]
        return {}

    def ame(*_args, **kwargs):
        calls["ame"] = kwargs["score_mean_link"]
        return np.zeros(2), np.zeros(2)

    monkeypatch.setattr(module, "write_predicted_scores_artifacts", predicted)
    monkeypatch.setattr(module, "write_arm_overlap_artifacts", overlap)
    monkeypatch.setattr(module._report, "_itt_ame_draws", ame)
    monkeypatch.setattr(module, "write_rope_figures", lambda *_args, **_kwargs: None)

    assert module._regenerate_one(fit_dir).startswith("ok")
    assert calls == {
        "predicted": "three_choice_guessing_floor",
        "overlap": "three_choice_guessing_floor",
        "ame": "three_choice_guessing_floor",
    }
