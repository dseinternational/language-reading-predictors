# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Interval mechanics and release parsing preserve their distinct policies."""

import numpy as np
import pytest

from language_reading_predictors.statistical_models.posteriors import band50, coef_row
from language_reading_predictors.statistical_models.release.base import _read_json


def test_posterior_summary_preserves_mean_median_and_two_equal_tail_bands():
    draws = np.array([-3.0, 0.0, 1.0, 9.0])
    summary = coef_row("term", draws, hdi_prob=0.89)
    expected = np.quantile(draws, [0.055, 0.945, 0.25, 0.75])
    np.testing.assert_allclose([summary[key] for key in ("lo", "hi", "lo50", "hi50")], expected)
    assert summary["mean"] == draws.mean()
    assert summary["median"] == np.median(draws)
    assert summary["prob_pos"] == 0.5


def test_inner_interval_propagates_nan():
    assert np.isnan(band50(np.array([0, 1, np.nan]))).all()


def test_empty_posterior_samples_still_stop_summary_generation():
    with pytest.raises(ValueError, match="empty posterior"):
        band50(np.array([]))
    with pytest.raises(ValueError, match="empty posterior"):
        coef_row("empty", np.array([]), hdi_prob=0.89)


@pytest.mark.parametrize("contents", ["", "{broken", '{"x": NaN}', '{"x": 1e999}'])
def test_invalid_release_json_stays_unreadable(tmp_path, contents):
    source = tmp_path / "config.json"
    source.write_text(contents)
    assert _read_json(source) == (None, "unreadable")


def test_release_json_distinguishes_missing_from_present_null(tmp_path):
    source = tmp_path / "config.json"
    assert _read_json(source) == (None, "missing")
    source.write_text("null")
    assert _read_json(source) == (None, None)
