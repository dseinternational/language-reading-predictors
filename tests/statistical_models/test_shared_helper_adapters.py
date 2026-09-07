# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Contracts the 0.14.0 shared helpers are adopted under (#662).

Each adapter delegates one mechanism to ``dse_research_utils`` and keeps a
project decision the library deliberately does not make. These tests pin the
kept decision, not the delegated mechanism — the library has its own tests for
that — and pin the numerical agreement wherever a published artefact is at stake.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from language_reading_predictors.atomic_files import (
    process_default_file_mode,
    write_atomic,
)


# --- Atomic single-file replacement ------------------------------------------


def test_write_atomic_replaces_only_after_the_writer_returns(tmp_path):
    destination = tmp_path / "estimates.csv"
    destination.write_text("old\n", encoding="utf-8")

    def _write(temporary: Path) -> None:
        temporary.write_text("new\n", encoding="utf-8")
        # A reader at this instant must still see the previous publication.
        assert destination.read_text(encoding="utf-8") == "old\n"

    write_atomic(destination, _write)
    assert destination.read_text(encoding="utf-8") == "new\n"
    assert [p.name for p in tmp_path.iterdir()] == ["estimates.csv"]


def test_write_atomic_preserves_the_destination_when_the_writer_fails(tmp_path):
    destination = tmp_path / "estimates.csv"
    destination.write_text("old\n", encoding="utf-8")

    def _fail(temporary: Path) -> None:
        temporary.write_text("half", encoding="utf-8")
        raise RuntimeError("writer failed")

    with pytest.raises(RuntimeError, match="writer failed"):
        write_atomic(destination, _fail)

    assert destination.read_text(encoding="utf-8") == "old\n"
    assert [p.name for p in tmp_path.iterdir()] == ["estimates.csv"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_the_mode_policy_decides_what_the_published_file_is_readable_by(tmp_path):
    """``mkstemp``'s 0600 would silently narrow files that were world-readable.

    The bundle indexes ``influence`` and ``historical_growth_influence`` publish
    were written through a plain ``open`` and therefore carried the process's
    umask mode; the report render and ``scripts/upload.py`` read them back. The
    adapter restores that mode explicitly rather than inheriting the library's
    private temporary file.
    """
    control = tmp_path / "control.csv"
    control.touch()
    expected = stat.S_IMODE(control.stat().st_mode)
    assert process_default_file_mode(tmp_path) == expected

    private = tmp_path / "private.csv"
    write_atomic(private, lambda temporary: temporary.write_text("x", encoding="utf-8"))
    assert stat.S_IMODE(private.stat().st_mode) == 0o600

    shared = tmp_path / "shared.csv"
    write_atomic(
        shared,
        lambda temporary: temporary.write_text("x", encoding="utf-8"),
        mode="process_default",
    )
    assert stat.S_IMODE(shared.stat().st_mode) == expected


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_the_bundle_index_writers_keep_publishing_a_readable_file(tmp_path):
    """The two ``_atomic_write_csv`` adapters that used a plain ``open``."""
    from language_reading_predictors.statistical_models import influence

    control = tmp_path / "control.csv"
    control.touch()
    expected = stat.S_IMODE(control.stat().st_mode)

    destination = tmp_path / "nested" / "influence_sensitivity.csv"
    influence._atomic_write_csv(pd.DataFrame({"model_id": ["a"]}), destination)
    assert destination.read_text(encoding="utf-8").startswith("model_id")
    assert stat.S_IMODE(destination.stat().st_mode) == expected


def test_the_archive_copy_verifies_the_digest_before_it_replaces_anything(tmp_path):
    """A source that changed under us must never reach the archive path."""
    from language_reading_predictors.statistical_models import blending_sensitivity

    source = tmp_path / "trace.nc"
    source.write_bytes(b"contents")
    destination = tmp_path / "archive" / "trace.nc"
    destination.parent.mkdir()
    destination.write_bytes(b"previous")

    with pytest.raises(ValueError, match="SHA-256"):
        blending_sensitivity._install_content_addressed_copy(
            source, destination, expected_sha256="0" * 64
        )
    assert destination.read_bytes() == b"previous"
    assert sorted(p.name for p in destination.parent.iterdir()) == ["trace.nc"]


# --- Frozen HSGP geometry -----------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_the_mechanism_basis_geometry_is_the_arithmetic_it_replaced(seed):
    """Half the input range about the input midpoint, expanded by ``c``."""
    from dse_research_utils.statistics.models.hsgp_design import HSGPDesign

    rng = np.random.default_rng(seed)
    values = rng.normal(scale=10.0 ** rng.uniform(-2, 2), size=int(rng.integers(2, 40)))
    if values.min() == values.max():  # pragma: no cover - degenerate draw
        pytest.skip("a degenerate domain has no basis")

    design = HSGPDesign.from_domain(values, m=10, c=1.5)
    assert design.center == float((values.min() + values.max()) / 2)
    assert design.L == float((values.max() - values.min()) / 2 * 1.5)
    assert design.m == 10


def test_a_legacy_or_invalid_saved_design_still_demands_a_fresh_full_fit():
    """The #660 contract survives the shared validation (#662)."""
    from language_reading_predictors.statistical_models.fitted_payloads import (
        MechanismDesign,
    )
    from language_reading_predictors.statistical_models.preprocessing import (
        Standardiser,
    )

    scaler = Standardiser(mean=0.0, sd=1.0)
    for incomplete in (
        MechanismDesign(mech_scaler=scaler, hsgp_L=None),
        MechanismDesign(mech_scaler=scaler, hsgp_L=2.0),
        MechanismDesign(mech_scaler=scaler, hsgp_L=2.0, hsgp_m=10),
    ):
        with pytest.raises(ValueError, match="fresh full fit"):
            incomplete.hsgp_design()

    for invalid in (
        MechanismDesign(mech_scaler=scaler, hsgp_L=0.0, hsgp_m=10, hsgp_center=0.0),
        MechanismDesign(mech_scaler=scaler, hsgp_L=2.0, hsgp_m=0, hsgp_center=0.0),
        MechanismDesign(
            mech_scaler=scaler, hsgp_L=2.0, hsgp_m=10, hsgp_center=float("nan")
        ),
    ):
        with pytest.raises(ValueError, match="fresh full fit"):
            invalid.hsgp_design()

    complete = MechanismDesign(
        mech_scaler=scaler, hsgp_L=2.0, hsgp_m=10, hsgp_center=-0.25
    )
    assert complete.hsgp_kwargs() == {"m": 10, "L": 2.0, "center": -0.25}


# --- Per-observation predictive summaries -------------------------------------


@pytest.mark.parametrize("prob", [0.5, 0.8, 0.89, 0.9, 0.94, 0.95, 0.99])
def test_the_shared_upper_quantile_matches_the_one_it_replaced(prob):
    """``(1 + p) / 2`` and ``1 - (1 - p) / 2`` are equal in float64 for p >= 0.25.

    The shared helper uses the complement form. The two are algebraically the
    same and agree exactly at every interval this project publishes, but they can
    differ by an ulp for very narrow intervals, which can flip an inclusion flag
    at an endpoint. A future narrower reporting width would fail here first.
    """
    assert (1.0 + prob) / 2.0 == 1.0 - (1.0 - prob) / 2.0


def _labelled(dataset: xr.Dataset) -> xr.Dataset:
    return dataset.assign_coords(
        {name: np.arange(size) for name, size in dataset.sizes.items()}
    )


def _predictive_trace(rep: np.ndarray, obs: np.ndarray) -> xr.DataTree:
    return xr.DataTree.from_dict(
        {
            "posterior_predictive": _labelled(
                xr.Dataset({"y_post": (("chain", "draw", "obs_id"), rep)})
            ),
            "observed_data": _labelled(
                xr.Dataset({"y_post": (("obs_id",), obs)})
            ),
        }
    )


def test_a_fit_with_no_finite_observation_still_writes_its_coverage_row():
    """The shared helper rejects an empty observation axis; the report needs a row."""
    from language_reading_predictors.statistical_models import predictive_checks as pc

    trace = _predictive_trace(
        np.zeros((2, 8, 3), dtype=float), np.full(3, np.nan)
    )
    coverage = pc.ppc_interval_coverage(trace)
    assert list(coverage["n_total"]) == [0, 0]
    assert list(coverage["n_inside"]) == [0, 0]
    assert coverage["coverage"].isna().all()

    calibration = pc.ppc_calibration_table(trace)
    assert calibration.empty
    assert list(calibration.columns) == [
        "observed",
        "pp_median",
        "pp_lo",
        "pp_hi",
        "inside",
    ]


def test_a_floor_rule_cell_with_no_observations_still_gets_its_row():
    """The shared helper refuses a non-finite observation; the table needs a row.

    A floor-rule node whose single ``"all"`` cell has no finite observed row
    produced a NaN observed rate, NaN bounds and ``inside=False`` under the
    quantile arithmetic this replaced. Dropping the row instead would make an
    unusable check look like a check that was never requested.
    """
    from language_reading_predictors.statistical_models import predictive_checks as pc

    trace = _predictive_trace(
        np.zeros((2, 8, 3), dtype=float), np.full(3, np.nan)
    )
    cells = pc.ppc_offfloor_cell_table(trace, node="y_post")
    assert list(cells["cell"]) == ["all"]
    assert list(cells["n"]) == [0]
    assert cells["observed_rate"].isna().all()
    assert cells["pp_rate_median"].isna().all()
    assert cells["pp_rate_lo"].isna().all()
    assert cells["pp_rate_hi"].isna().all()
    assert list(cells["inside"]) == [False]
    assert cells["inside"].dtype == bool

    coverage = pc.ppc_offfloor_rate_coverage(trace, node="y_post")
    assert list(coverage["n_total"]) == [1, 1]
    assert list(coverage["n_inside"]) == [0, 0]


def test_observed_rows_must_identify_the_replicated_rows():
    """Equal row counts are not evidence of alignment (#662)."""
    from language_reading_predictors.statistical_models import predictive_checks as pc

    rep = np.zeros((2, 4, 3), dtype=float)
    trace = xr.DataTree.from_dict(
        {
            "posterior_predictive": _labelled(
                xr.Dataset({"y_post": (("chain", "draw", "obs_id"), rep)})
            ),
            # Same length, different labels: the retired reshape accepted this.
            "observed_data": xr.Dataset(
                {"y_post": (("obs_id",), np.zeros(3))},
                coords={"obs_id": [7, 8, 9]},
            ),
        }
    )
    with pytest.raises(ValueError, match="do not identify"):
        pc._ppc_node_arrays(trace, "y_post")


# --- Child-level likelihood aggregation ---------------------------------------


def test_a_child_with_no_likelihood_row_is_refused_not_scored_as_zero():
    """A phantom zero-likelihood unit would silently flatter the PSIS-LOO."""
    from language_reading_predictors.statistical_models import diagnostics as diag

    values = np.array([[[1.0, 2.0, 4.0], [10.0, 20.0, 40.0]]])
    trace = xr.DataTree.from_dict(
        {
            "log_likelihood": xr.Dataset(
                {"y_post": (("chain", "draw", "cell"), values)},
                coords={"chain": [0], "draw": [0, 1]},
            ),
            "constant_data": xr.Dataset(
                {
                    # Four children declared by G, but no row maps to child 1.
                    "G": ("obs_id", np.array([1.0, 0.0, 1.0, 0.0])),
                    "y_post_cell_row": ("cell", np.array([0, 2, 3])),
                }
            ),
        }
    )
    with pytest.raises(ValueError, match="no 'y_post' likelihood row"):
        diag._joint_log_likelihood_by_child(trace)


def test_the_child_aggregate_is_summed_in_float64():
    """A float32 likelihood is widened before the row sum, as it always was."""
    from language_reading_predictors.statistical_models import diagnostics as diag

    values = np.array([[[-1e8, -1.0, -1.0]]], dtype=np.float32)
    trace = xr.DataTree.from_dict(
        {
            "log_likelihood": xr.Dataset(
                {"y_post": (("chain", "draw", "obs_id"), values)},
                coords={"chain": [0], "draw": [0]},
            ),
            "constant_data": xr.Dataset(
                {"loo_child_idx": ("obs_id", np.array([0, 0, 0]))}
            ),
        }
    )
    aggregated = diag._joint_log_likelihood_by_child(trace)
    assert aggregated.dtype == np.float64
    assert float(aggregated.values.item()) == -100000002.0
