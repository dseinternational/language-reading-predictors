# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Independent unit tests for the isolated LCF summary computations (#394 pillar 6).

Exercises the public ``lcf_summaries`` API without a factory, PyMC model, output
directory or plotting session — confirming the family's summary calculations are
testable in isolation: the conditional measurement translation, the observed
same-wave comparator, and the items-scale translation behind the family's
key-findings headline (the last two were untested before the 2026-08-21 review,
finding 10, while this docstring claimed ``test_factories.py`` covered them — it
covers the stitched LOO and the directed #312 concurrent comparison instead).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import expit

from language_reading_predictors.statistical_models.fitted_payloads import (
    LongCorrFactorPayload,
)
from language_reading_predictors.statistical_models.lcf_summaries import (
    items_scale,
    observed_conditional_slope,
    observed_domain_corr,
)


class _StubBuilt:
    """Duck-typed BuiltModel carrying just what the summaries read."""

    def __init__(self, prepared, payload):
        self.prepared = prepared
        self.payload = payload

    def require_payload(self, payload_type, *, family):
        assert isinstance(self.payload, payload_type), family
        return self.payload


def _payload(*, waves, standardisers):
    return LongCorrFactorPayload(
        z_nodes=(),
        child_of_node={},
        cell_indices_of_node={},
        observed_z_of_node={},
        domains={"d1": ("A",), "d2": ("B",)},
        domain_of={"A": "d1", "B": "d2"},
        indicators=("A", "B"),
        cell_names=(),
        standardisers=standardisers,
        waves=waves,
        n_children=5,
        n_used_children=5,
        invariance="pooled",
    )


def test_observed_conditional_slope_matches_the_delta_method_by_hand():
    """The conditional measurement translation is the delta-method slope of the
    target indicator's item count per +1 item of the predictor indicator, at the
    pooled-mean operating point, conditioning the two domains on the third. With C
    the third domain, ``Cov(a, b | C) = 0.5 - 0.2 * 0.3 = 0.44`` and
    ``Var(b | C) = 1 - 0.3**2 = 0.91``."""
    corr = np.array([[[[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]]]])
    loadings = np.array([[2.0, 0.8]])
    residual_sds = np.array([[0.4, 0.6]])

    slope = observed_conditional_slope(
        corr,
        loadings,
        residual_sds,
        target_domain_idx=0,
        predictor_domain_idx=1,
        target_indicator_idx=0,
        predictor_indicator_idx=1,
    )

    expected = 2.0 * 0.8 * 0.44 / (0.8**2 * 0.91 + 0.6**2)
    np.testing.assert_allclose(slope, expected, rtol=1e-12, atol=1e-12)


def test_observed_domain_corr_means_pairs_and_leaves_sparse_waves_nan():
    """Wave 1 has five complete pairs (the Pearson correlation of the two
    standardised columns); wave 2 has only two, below the >= 3 floor, so its
    comparator is NaN with a zero pair count — the value the cross-check must
    treat as no-comparator, never as a reversal."""
    a = np.array([[0.0, 0.5], [1.0, np.nan], [2.0, np.nan], [3.0, 1.5], [4.0, np.nan]])
    b = np.array([[0.1, 0.4], [1.1, np.nan], [1.9, np.nan], [3.2, 1.4], [3.9, np.nan]])
    prepared = SimpleNamespace(logit={"A": a, "B": b})
    built = _StubBuilt(
        prepared,
        _payload(waves=(1, 2), standardisers={"A": (0.0, 1.0), "B": (0.0, 1.0)}),
    )

    out = observed_domain_corr(built)

    assert list(out["wave"]) == [1, 2]
    expected_w1 = float(np.corrcoef(a[:, 0], b[:, 0])[0, 1])
    np.testing.assert_allclose(out.loc[0, "observed_corr"], expected_w1, rtol=1e-12)
    assert int(out.loc[0, "n_indicator_pairs"]) == 1
    assert np.isnan(out.loc[1, "observed_corr"])
    assert int(out.loc[1, "n_indicator_pairs"]) == 0


def test_items_scale_matches_the_hand_derived_operating_point_translation():
    """One deterministic draw, one wave, one indicator per domain: the items-scale
    slope must equal the hand-derived
    ``slope_z * (sd_m / sd_k) * (info_m / info_k)`` with the Haldane
    ``(N + 1) p (1 - p)`` operating-point information, and the median/50% band
    columns (2026-08-21 review, finding 10) must be present and consistent."""
    lam = xr.DataArray(
        [[[0.8, 0.6]]],
        dims=("chain", "draw", "indicator"),
        coords={"indicator": ["A", "B"]},
    )
    sig = xr.DataArray(
        [[[0.6, 0.8]]],
        dims=("chain", "draw", "indicator"),
        coords={"indicator": ["A", "B"]},
    )
    rho = 0.5
    corr = xr.DataArray(
        [[[[[1.0, rho], [rho, 1.0]]]]],
        dims=("chain", "draw", "wave", "domain", "domain_b"),
        coords={"wave": [1], "domain": ["d1", "d2"], "domain_b": ["d1", "d2"]},
    )
    posterior = xr.Dataset(
        {"lambda_load": lam, "sigma_indicator": sig, "factor_corr": corr}
    )
    payload = _payload(waves=(1,), standardisers={"A": (0.0, 2.0), "B": (1.0, 0.5)})
    ctx = SimpleNamespace(
        trace=SimpleNamespace(posterior=posterior),
        prepared=SimpleNamespace(n_trials={"A": 9, "B": 19}),
        reporting=SimpleNamespace(ci_prob=0.89),
    )

    out = items_scale(ctx, _StubBuilt(SimpleNamespace(), payload))

    assert len(out) == 1
    row = out.iloc[0]
    assert row["predictor_indicator"] == "B"
    assert row["target_indicator"] == "A"
    # Target A: standardiser mean 0 -> p = 0.5, N = 9 -> info = 10 * 0.25.
    # Predictor B: mean 1 -> p = expit(1), N = 19 -> info = 20 * p(1-p).
    slope_z = 0.8 * 0.6 * rho / (0.6**2 + 0.8**2)
    info_m = (9 + 1) * 0.5 * 0.5
    p_k = float(expit(1.0))
    info_k = (19 + 1) * p_k * (1 - p_k)
    expected = slope_z * (2.0 / 0.5) * (info_m / info_k)
    np.testing.assert_allclose(row["items_per_item_mean"], expected, rtol=1e-12)
    # A single deterministic draw collapses every location statistic and band.
    for col in (
        "items_per_item_median",
        "items_per_item_lo50",
        "items_per_item_hi50",
        "items_per_item_lo",
        "items_per_item_hi",
    ):
        np.testing.assert_allclose(row[col], expected, rtol=1e-12)
    assert row["prob_pos"] == 1.0
    assert isinstance(out, pd.DataFrame)
