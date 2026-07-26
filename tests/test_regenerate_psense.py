# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for ``scripts/regenerate_psense.py`` (#381 backfill).

The backfill exists because power-scaling sensitivity was wired into the family
pipelines only in #408 / #416: every earlier fit shows no psense flags because none
were **measured**, which a reader cannot distinguish from measured-clean. What makes
the repair possible without resampling is that power-scaling is importance
reweighting over the stored draws, so these tests pin the two decisions that keep it
faithful — the reported-parameter set comes from the fit's own record, and a trace
that cannot support the measurement is reported rather than silently skipped.

Scripts aren't on the import path in this repo, so the module is loaded by file path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "regenerate_psense.py"


@pytest.fixture(scope="module")
def regen():
    spec = importlib.util.spec_from_file_location("regenerate_psense", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_diagnostics(fit_dir: Path, labels: list[str]) -> None:
    fit_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"mean": [0.0] * len(labels)}, index=labels).to_csv(
        fit_dir / "diagnostics.csv"
    )


def test_reported_var_names_collapses_elements_and_keeps_order(regen, tmp_path):
    """``diagnostics.csv`` carries one row per element; ``psense_summary`` takes the
    base name and expands it itself. Order is preserved because it drives the psense
    figure's row layout, so a set would silently reshuffle the published plot."""
    fit = tmp_path / "lrp-rli-hs-001-reporting"
    _write_diagnostics(
        fit, ["alpha", "gamma_own", "beta[L]", "beta[R]", "beta[E]", "kappa"]
    )
    assert regen.reported_var_names(fit) == ["alpha", "gamma_own", "beta", "kappa"]


def test_reported_var_names_is_empty_without_the_fits_own_record(regen, tmp_path):
    """No ``diagnostics.csv`` means no record of what the fit reported. Guessing a
    parameter set here would measure something the report never showed."""
    fit = tmp_path / "lrp-rli-hs-002-reporting"
    fit.mkdir(parents=True)
    assert regen.reported_var_names(fit) == []


def test_backfill_reports_a_trace_that_predates_the_log_prior_wiring(
    regen, tmp_path, monkeypatch
):
    """Power-scaling needs ``log_prior`` and ``log_likelihood``. A fit stored before
    that wiring cannot be repaired from its draws, and saying so is the point — a
    silent skip would leave the estimand unmeasured while the sweep looked clean."""
    fit = tmp_path / "lrp-rli-med-062-reporting"
    _write_diagnostics(fit, ["beta_code"])
    (fit / "trace.nc").write_bytes(b"")

    monkeypatch.setattr(
        regen, "_trace_groups", lambda _idata: {"posterior", "sample_stats"}
    )
    import arviz as az

    monkeypatch.setattr(az, "from_netcdf", lambda _p: SimpleNamespace(groups=[]))

    status, detail = regen.backfill(fit, force=False, dry_run=False)
    assert status == "needs refit"
    assert "log_prior" in detail


def test_backfill_measures_only_parameters_the_posterior_carries(
    regen, tmp_path, monkeypatch
):
    """A spec edit since the fit can leave a name in ``diagnostics.csv`` that this
    trace never sampled. Measuring the intersection keeps the rest of the model's
    parameters covered instead of failing the whole fit over one absent term."""
    fit = tmp_path / "lrp-rli-lcsm-082-reporting"
    _write_diagnostics(fit, ["a_change", "g_W_L", "kappa"])
    (fit / "trace.nc").write_bytes(b"")

    import arviz as az

    monkeypatch.setattr(az, "from_netcdf", lambda _p: SimpleNamespace(groups=[]))
    monkeypatch.setattr(
        regen, "_trace_groups", lambda _idata: {"log_prior", "log_likelihood"}
    )
    monkeypatch.setattr(
        az,
        "extract",
        lambda *_a, **_k: SimpleNamespace(data_vars={"a_change": 0, "kappa": 0}),
    )

    measured: dict[str, list[str]] = {}

    def _fake_artifacts(_trace, _out, var_names):
        measured["names"] = list(var_names)
        return pd.DataFrame({"diagnosis": ["✓", "potential prior-data conflict"]})

    monkeypatch.setattr(regen, "psense_artifacts", _fake_artifacts)

    status, detail = regen.backfill(fit, force=False, dry_run=False)
    assert status == "written"
    assert measured["names"] == ["a_change", "kappa"]
    assert "g_W_L" in detail  # the dropped name is reported, not hidden
    assert "1 flagged" in detail


def test_targets_exclude_in_flight_output_transactions(regen, tmp_path, monkeypatch):
    """Fits stage into a hidden ``.<id>-<config>.staging-XXXX`` sibling and are
    promoted only on success. Backfilling into one writes artefacts that are about to
    be discarded, or races a live fit."""
    from language_reading_predictors import paths as _paths

    root = tmp_path / "models"
    (root / "lrp-rli-al-001-reporting").mkdir(parents=True)
    (root / ".lrp-rli-ca-008-reporting.staging-9hbfh_9x").mkdir(parents=True)
    monkeypatch.setattr(_paths, "stat_models_dir", lambda: root)

    names = [d.name for d in regen.resolve_targets("all")]
    assert names == ["lrp-rli-al-001-reporting"]


def test_backfill_leaves_an_existing_summary_alone_unless_forced(regen, tmp_path):
    fit = tmp_path / "lrp-rli-itt-010-reporting"
    _write_diagnostics(fit, ["tau"])
    (fit / "psense_summary.csv").write_text("tau,0.1,0.1,✓\n", encoding="utf-8")

    status, _ = regen.backfill(fit, force=False, dry_run=False)
    assert status == "present"


def test_dry_run_writes_nothing(regen, tmp_path, monkeypatch):
    fit = tmp_path / "lrp-rli-al-001-reporting"
    _write_diagnostics(fit, ["alpha", "beta_cohort"])
    (fit / "trace.nc").write_bytes(b"")

    import arviz as az

    monkeypatch.setattr(az, "from_netcdf", lambda _p: SimpleNamespace(groups=[]))
    monkeypatch.setattr(
        regen, "_trace_groups", lambda _idata: {"log_prior", "log_likelihood"}
    )
    monkeypatch.setattr(
        az,
        "extract",
        lambda *_a, **_k: SimpleNamespace(data_vars={"alpha": 0, "beta_cohort": 0}),
    )

    def _must_not_run(*_a, **_k):  # pragma: no cover - asserts absence
        raise AssertionError("dry run must not compute or write psense")

    monkeypatch.setattr(regen, "psense_artifacts", _must_not_run)

    status, detail = regen.backfill(fit, force=False, dry_run=True)
    assert status == "would write"
    assert "2 parameters" in detail
    assert not (fit / "psense_summary.csv").exists()
