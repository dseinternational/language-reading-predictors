# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Independent unit tests for the isolated LCF inference algorithms (#394 pillar 7).

These exercise ``lcf_inference.child_log_likelihood`` on hand-built inputs — no
factory, PyMC model, output directory or plotting — to confirm the numerical
algorithm is testable in isolation. The end-to-end factory-backed comparison
(against SciPy, plus the LOO stitch and log-prior recovery) lives in
``test_factories.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from scipy.stats import multivariate_normal

from language_reading_predictors.statistical_models.lcf_inference import (
    child_log_likelihood,
)


def _synthetic_trace_and_built():
    """A minimal two-child, single-pattern, two-cell LCF posterior + built stub."""
    cell_names = ["dom0_w0", "dom1_w0"]
    means = np.array([[[0.0, 0.5], [0.2, -0.1]]])  # (chain=1, draw=2, cell=2)
    covs = np.array(
        [
            [
                [[1.0, 0.3], [0.3, 1.0]],
                [[1.2, 0.2], [0.2, 0.8]],
            ]
        ]
    )  # (chain=1, draw=2, cell=2, cell_b=2)
    posterior = xr.Dataset(
        {
            "mean_z": (("chain", "draw", "cell"), means),
            "Sigma_z": (("chain", "draw", "cell", "cell_b"), covs),
        },
        coords={
            "chain": [0],
            "draw": np.arange(2),
            "cell": cell_names,
            "cell_b": cell_names,
        },
    )
    trace = SimpleNamespace(posterior=posterior, children={})
    observed = np.array([[0.1, 0.4], [-0.3, 0.2]])  # (child=2, cell=2)
    built = SimpleNamespace(
        extras={
            "cell_names": cell_names,
            "z_nodes": ["z_obs"],
            "child_of_node": {"z_obs": [0, 1]},
            "cell_indices_of_node": {"z_obs": [0, 1]},
            "observed_z_of_node": {"z_obs": observed},
            "n_used_children": 2,
        }
    )
    return trace, built, observed


def test_child_log_likelihood_matches_scipy_mvnormal_without_a_factory():
    trace, built, observed = _synthetic_trace_and_built()

    actual = child_log_likelihood(trace, built, chunk_size=1)

    assert actual.dims == ("chain", "draw", "child_lcf")
    assert actual.shape == (1, 2, 2)
    assert list(actual.coords["child_lcf"].values) == [0, 1]

    means = trace.posterior["mean_z"].values
    covs = trace.posterior["Sigma_z"].values
    for draw in range(2):
        for child in range(2):
            expected = multivariate_normal.logpdf(
                observed[child], mean=means[0, draw], cov=covs[0, draw]
            )
            np.testing.assert_allclose(
                actual.isel(chain=0, draw=draw).sel(child_lcf=child),
                expected,
                rtol=1e-10,
                atol=1e-10,
            )


def test_child_log_likelihood_is_chunk_size_invariant():
    trace, built, _ = _synthetic_trace_and_built()
    one = child_log_likelihood(trace, built, chunk_size=1)
    both = child_log_likelihood(trace, built, chunk_size=8)
    np.testing.assert_allclose(one.values, both.values, rtol=1e-10, atol=1e-10)


def test_child_log_likelihood_rejects_coordinate_drift():
    """Positional drift between the rebuilt model's cells and the posterior must be
    refused rather than silently producing a mismatched likelihood."""
    trace, built, _ = _synthetic_trace_and_built()
    built.extras["cell_names"] = ["dom0_w0", "WRONG"]
    with pytest.raises(ValueError, match="coordinates do not match"):
        child_log_likelihood(trace, built)


def test_child_log_likelihood_rejects_nonpositive_chunk_size():
    trace, built, _ = _synthetic_trace_and_built()
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        child_log_likelihood(trace, built, chunk_size=0)
