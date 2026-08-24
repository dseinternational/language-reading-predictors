# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The Tier-1 1A comparison contract in ``compare_statistical_models.py``.

The 2026-08-23 joint-mechanism follow-up review (#591, finding 2) found that the
"identified" joint contrast and the "product-of-marginals" sensitivity it is set
beside are not fitted on the same rows or the same exposure unit: ``jm-002``
requires both outcome baselines and standardises the letter-sound logit once over
that joint union, while ``mech-096`` and ``mech-101`` filter to their own outcome's
rows and re-standardise there. The gap between the two rows was nevertheless
described as the cost of the working-independence assumption.

These tests pin the contract that says so: fitted rows and exposure scaler are read
from each fit's own ``config.json``, a field a fit never recorded counts as *not
proven* rather than as agreement, and the verdict travels with both contrast rows.
Scripts are not on the import path, so the module is loaded by file path.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "scripts" / "compare_statistical_models.py"
)


@pytest.fixture(scope="module")
def cmp_mod():
    spec = importlib.util.spec_from_file_location("compare_statistical_models", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fits(cmp_mod, tmp_path, monkeypatch):
    """Redirect the script's run-directory resolution into a temp output root."""
    def _run_dir(model_id: str, config: str) -> str:
        d = tmp_path / f"{model_id}-{config}"
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    monkeypatch.setattr(cmp_mod, "_run_dir", _run_dir)
    return _run_dir


def _write_config(run_dir, model_id, config, **extra_fields):
    path = Path(run_dir(model_id, config)) / "config.json"
    payload = {"model_id": model_id, "n_obs": extra_fields.pop("n_obs", None),
               "n_children": extra_fields.pop("n_children", 53),
               "extra": extra_fields}
    path.write_text(json.dumps(payload))


def test_matching_rows_and_scalers_are_reported_comparable(cmp_mod, fits):
    for model_id in ("lrp-rli-mech-096", "lrp-rli-mech-101", "lrp-rli-jm-002"):
        _write_config(fits, model_id, "reporting", n_obs=150, exposure_logit_sd=1.4)
    rows = [
        cmp_mod._tier1_sample_contract(m, "reporting")
        for m in ("lrp-rli-mech-096", "lrp-rli-mech-101", "lrp-rli-jm-002")
    ]
    comparable, note = cmp_mod._tier1_comparability(rows)
    assert comparable is True
    assert "agree" in note


def test_different_fitted_rows_are_not_comparable(cmp_mod, fits):
    """The real situation: three different fitted samples, so the difference between
    the joint and marginal contrasts is not a dependence effect alone."""
    for model_id, n_obs in (
        ("lrp-rli-mech-096", 152),
        ("lrp-rli-mech-101", 156),
        ("lrp-rli-jm-002", 153),
    ):
        _write_config(fits, model_id, "reporting", n_obs=n_obs, exposure_logit_sd=1.4)
    rows = [
        cmp_mod._tier1_sample_contract(m, "reporting")
        for m in ("lrp-rli-mech-096", "lrp-rli-mech-101", "lrp-rli-jm-002")
    ]
    comparable, note = cmp_mod._tier1_comparability(rows)
    assert comparable is False
    assert "n_rows differs" in note and "152" in note and "156" in note


def test_a_different_exposure_scaler_alone_breaks_comparability(cmp_mod, fits):
    """Same rows, different unit: 'per SD' then means a different raw increment in
    each fit, so the two contrasts are not on one scale."""
    for model_id, sd in (
        ("lrp-rli-mech-096", 1.385682),
        ("lrp-rli-mech-101", 1.433543),
        ("lrp-rli-jm-002", 1.411770),
    ):
        _write_config(fits, model_id, "reporting", n_obs=153, exposure_logit_sd=sd)
    rows = [
        cmp_mod._tier1_sample_contract(m, "reporting")
        for m in ("lrp-rli-mech-096", "lrp-rli-mech-101", "lrp-rli-jm-002")
    ]
    comparable, note = cmp_mod._tier1_comparability(rows)
    assert comparable is False
    assert "exposure_logit_sd differs" in note


def test_an_unrecorded_field_is_not_proven_rather_than_assumed_equal(cmp_mod, fits):
    """A fit predating the scaler metadata must not be waved through as matching."""
    for model_id in ("lrp-rli-mech-096", "lrp-rli-mech-101", "lrp-rli-jm-002"):
        _write_config(fits, model_id, "reporting", n_obs=153)
    rows = [
        cmp_mod._tier1_sample_contract(m, "reporting")
        for m in ("lrp-rli-mech-096", "lrp-rli-mech-101", "lrp-rli-jm-002")
    ]
    comparable, note = cmp_mod._tier1_comparability(rows)
    assert comparable is False
    assert "exposure_logit_sd not recorded" in note


def test_the_1a_contrast_is_still_written_without_the_marginal_fits(
    cmp_mod, fits, tmp_path
):
    """A joint-only run must still emit the identified contrast: the early return on
    an empty marginal-forest row set discarded it (2026-08-23 review, gap 5)."""
    import numpy as np
    import arviz as az

    for model_id in ("lrp-rli-mech-096", "lrp-rli-mech-101", "lrp-rli-jm-002"):
        _write_config(fits, model_id, "reporting", n_obs=153, exposure_logit_sd=1.4)
    joint_dir = Path(fits("lrp-rli-jm-002", "reporting"))
    rng = np.random.default_rng(5)
    az.from_dict(
        {"posterior": {"delta_ls_decoding": rng.normal(0.8, 0.2, size=(2, 200))}}
    ).to_netcdf(joint_dir / "trace.nc")
    (joint_dir / "diagnostics_summary.json").write_text(json.dumps({"passed": True}))

    out = tmp_path / "comparison"
    out.mkdir()
    assert cmp_mod.tier1_decoding_specificity("reporting", str(out)) is True

    import pandas as pd

    contrast = pd.read_csv(out / "tier1_1a_contrast.csv")
    # Only the identified row — there is no marginal pair to convolve.
    assert list(contrast["identified"]) == [True]
    assert list(contrast["comparable"]) == [True]
    # And the reconciliation is written whether or not the forest could be.
    contract = pd.read_csv(out / "tier1_1a_comparison_contract.csv")
    assert set(contract["model"]) == {
        "lrp-rli-mech-096",
        "lrp-rli-mech-101",
        "lrp-rli-jm-002",
    }
    assert not (out / "tier1_negative_control_forest.csv").exists()
