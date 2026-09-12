# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Rewriting a stored fit's prior table without resampling it.

``lrp-rli-jm-001``'s levels design carries ``beta_mech`` on
``predictor_slope_prior`` — ``Normal(0, 0.3)`` — while the pre-#637 name map keyed
on the *name* ``beta_mech`` and published "Linear-mechanism slope beta_mech ~
Normal(0, 1)" with a panel plotting that wider density, beside a ``distribution``
column that correctly read ``Normal(0, 0.3)``.

The script repairs a stored fit in place. These tests pin the three properties
that make that safe: it writes exactly what it previewed, it removes only the
panels the corrected table orphans, and it prunes the manifest without turning a
fit-time inventory into a directory listing.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pymc as pm
import pytest

from language_reading_predictors.statistical_models import priors
from types import SimpleNamespace

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "regenerate_priors_table.py"
_spec = importlib.util.spec_from_file_location("_regenerate_priors_table", _SCRIPT)
regenerate_priors_table = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(regenerate_priors_table)


def _table(panels: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "parameter": [f"p{i}" for i in range(len(panels))],
            "distribution": ["Normal(0, 1)"] * len(panels),
            "role": ["nuisance"] * len(panels),
            "rationale": ["..."] * len(panels),
            "panel": panels,
        }
    )


def test_only_the_orphaned_panels_are_removed(tmp_path):
    """A panel the corrected table still points at must survive.

    ``emit_priors`` would redraw every panel, and a panel redrawn today is laid
    out by today's matplotlib — a different canvas for an identical density. The
    orphan must go, because it plots a prior the model does not use.
    """
    for key in ("alpha", "beta_mech", "predictor_slope", "gamma_cross"):
        for ext in ("png", "svg"):
            (tmp_path / f"prior_{key}.{ext}").write_text("figure")
    (tmp_path / "prior_posterior.png").write_text("not a named-prior panel")

    removed = regenerate_priors_table._drop_orphaned_panels(tmp_path, _table(["alpha", "predictor_slope", ""]))

    assert removed == {
        "prior_beta_mech.png",
        "prior_beta_mech.svg",
        "prior_gamma_cross.png",
        "prior_gamma_cross.svg",
    }
    assert (tmp_path / "prior_alpha.png").exists()
    assert (tmp_path / "prior_predictor_slope.svg").exists()
    assert not (tmp_path / "prior_beta_mech.png").exists()
    # Overlays and predictive figures are not named-prior panels and are untouched.
    assert (tmp_path / "prior_posterior.png").exists()


def test_orphan_cleanup_handles_parameter_specific_panels(tmp_path):
    retained = "gamma_cross__" + "a" * 16
    orphan = "gamma_cross__" + "b" * 16
    for name in (retained, orphan, "predictive", "posterior"):
        (tmp_path / f"prior_{name}.png").write_text("figure")
    removed = regenerate_priors_table._drop_orphaned_panels(tmp_path, _table([retained]))
    assert removed == {f"prior_{orphan}.png"}
    assert (tmp_path / f"prior_{retained}.png").exists()
    assert (tmp_path / "prior_predictive.png").exists()
    assert (tmp_path / "prior_posterior.png").exists()


def test_the_manifest_is_pruned_not_rescanned(tmp_path):
    """A rescan would absorb whatever has appeared since the fit.

    A rendered ``index.html`` and its Quarto asset tree live beside a published
    fit. Rescanning folds them into the manifest, turning the fit's own record of
    what it wrote into a listing of what happens to be in the directory.
    """
    manifest = {
        "model_id": "lrp-rli-jm-001",
        "n_written": 2,
        "n_skipped": 0,
        "n_untracked": 3,
        "artifacts": [
            {"filename": "priors_table.csv", "status": "written", "required": True},
            {"filename": "trace.nc", "status": "written", "required": True},
            {"filename": "prior_beta_mech.png", "status": "untracked"},
            {"filename": "prior_beta_mech.svg", "status": "untracked"},
            {"filename": "prior_alpha.png", "status": "untracked"},
        ],
    }
    path = tmp_path / "artifact_manifest.json"
    path.write_text(json.dumps(manifest))
    (tmp_path / "index.html").write_text("<html>rendered after the fit</html>")

    regenerate_priors_table._prune_manifest(tmp_path, {"prior_beta_mech.png", "prior_beta_mech.svg"})

    updated = json.loads(path.read_text())
    names = [entry["filename"] for entry in updated["artifacts"]]
    assert names == ["priors_table.csv", "trace.nc", "prior_alpha.png"]
    assert updated["n_untracked"] == 1
    assert updated["n_written"] == 2
    assert "index.html" not in names


def test_pruning_nothing_leaves_the_manifest_untouched(tmp_path):
    path = tmp_path / "artifact_manifest.json"
    original = '{"artifacts": [], "n_untracked": 0}'
    path.write_text(original)
    regenerate_priors_table._prune_manifest(tmp_path, set())
    assert path.read_text() == original


@pytest.mark.parametrize(
    ("stored", "fresh", "equal"),
    [
        ("nan", "", True),
        ("None", "", True),
        ("alpha", "alpha", True),
        ("beta_mech", "predictor_slope", False),
    ],
)
def test_an_absent_panel_reads_the_same_either_side_of_a_csv_round_trip(stored, fresh, equal):
    """A blank panel is NaN once read back; that must not read as a change."""
    normalise = regenerate_priors_table._normalise
    assert (normalise(stored) == normalise(fresh)) is equal


def test_a_family_without_a_checked_rebuild_is_skipped(tmp_path):
    """Incomplete on purpose: a family is added once its rebuild is checked.

    Guessing at a rebuild would write a table describing a model other than the
    one that was fitted, which is the defect being repaired.
    """
    (tmp_path / "priors_table.csv").write_text("parameter\nalpha\n")
    (tmp_path / "config.json").write_text(json.dumps({"model_id": "lrp-rli-itt-010", "kind": "itt"}))
    status, detail = regenerate_priors_table.regenerate(tmp_path, dry_run=True)
    assert status == "skipped"
    assert "no checked rebuild for kind 'itt'" in detail


def test_a_directory_with_no_stored_table_is_skipped(tmp_path):
    status, detail = regenerate_priors_table.regenerate(tmp_path, dry_run=True)
    assert status == "skipped"
    assert "priors_table.csv" in detail


def test_rebuild_refuses_changed_prior_distributions_before_writing(monkeypatch, tmp_path):
    with pm.Model() as stored_model:
        priors.gamma_cross_prior().to_pymc("slope")
    with pm.Model() as changed_model:
        priors.gamma_cross_prior(sigma=1).to_pymc("slope")
    table = tmp_path / "priors_table.csv"
    priors.priors_table(stored_model).to_csv(table, index=False)
    original = table.read_bytes()
    (tmp_path / "config.json").write_text(json.dumps({"model_id": "test", "kind": "test"}))
    monkeypatch.setattr(regenerate_priors_table, "_spec_for", lambda _: object())
    monkeypatch.setattr(
        regenerate_priors_table, "BUILDERS", {"test": lambda *_: (SimpleNamespace(model=changed_model), None)}
    )
    status, detail = regenerate_priors_table.regenerate(tmp_path, dry_run=False)
    assert status == "needs refit"
    assert "prior distributions differ" in detail
    assert table.read_bytes() == original
    assert list(tmp_path.glob("prior_*.png")) == []


def test_missing_panels_are_previewed_and_repaired_with_matching_table(monkeypatch, tmp_path):
    with pm.Model() as model:
        priors.gamma_cross_prior(sigma=1).to_pymc("slope")
    priors.priors_table(model).to_csv(tmp_path / "priors_table.csv", index=False)
    (tmp_path / "config.json").write_text(json.dumps({"model_id": "test", "kind": "test"}))
    (tmp_path / "artifact_manifest.json").write_text(json.dumps({"artifacts": [], "n_untracked": 0}))
    monkeypatch.setattr(regenerate_priors_table, "_spec_for", lambda _: object())
    monkeypatch.setattr(regenerate_priors_table, "BUILDERS", {"test": lambda *_: (SimpleNamespace(model=model), None)})

    def save(density, output_dir, name, *, title):
        assert density.params_dict == {"mu": 0, "sigma": 1}
        assert "slope" in title
        for ext in ("png", "svg"):
            (Path(output_dir) / f"{name}.{ext}").write_text("rendered density")

    monkeypatch.setattr(priors, "plot_and_save", save)
    status, detail = regenerate_priors_table.regenerate(tmp_path, dry_run=True)
    assert status == "would rewrite"
    assert "missing density panels" in detail
    assert list(tmp_path.glob("prior_*.png")) == []
    assert regenerate_priors_table.regenerate(tmp_path, dry_run=False)[0] == "rewritten"
    manifest = json.loads((tmp_path / "artifact_manifest.json").read_text())
    assert manifest["n_untracked"] == 2
    assert {row["filename"] for row in manifest["artifacts"]} == {path.name for path in tmp_path.glob("prior_*.*")}
    assert regenerate_priors_table.regenerate(tmp_path, dry_run=False)[0] == "unchanged"
