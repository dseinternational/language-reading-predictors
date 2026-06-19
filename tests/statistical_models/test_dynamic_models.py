# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Smoke tests for the longitudinal dynamic models (LRP67 LCSM, LRP68 RI-CLPM).

These check the wave-panel loader and that each factory *builds* and can draw a
small prior predictive sample. Full posterior-sampling correctness is validated
by the end-to-end fits in ``scripts/fit_statistical_model.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm
import pytest

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.factories import (
    build_lcsm_model,
    build_riclpm_model,
    riclpm_structure_mask,
)
from language_reading_predictors.statistical_models.preprocessing import load_wave_panel


def _write_panel_csv(tmp_path, n_children: int = 20, seed: int = 3):
    """Four-wave long CSV with the three dynamic-model measures + dose.

    Drops two cells (one ewrswr, one yarclet) so the observation mask is
    exercised.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_children):
        sid = f"S{i:03d}"
        age0 = int(rng.integers(60, 110))
        g = int(rng.integers(1, 3))
        for t in (1, 2, 3, 4):
            rows.append(
                {
                    V.SUBJECT_ID: sid,
                    V.TIME: t,
                    V.GROUP: g,
                    V.AGE: age0 + 6 * (t - 1),
                    V.EWRSWR: int(rng.integers(0, 91)),
                    V.YARCLET: int(rng.integers(0, 33)),
                    V.EOWPVT: int(rng.integers(0, 171)),
                    V.ATTEND: int(rng.integers(0, 10)),
                }
            )
    df = pd.DataFrame(rows)
    # Inject missing cells: S000 reading at t2, S001 letter-sounds at t3.
    df.loc[(df[V.SUBJECT_ID] == "S000") & (df[V.TIME] == 2), V.EWRSWR] = np.nan
    df.loc[(df[V.SUBJECT_ID] == "S001") & (df[V.TIME] == 3), V.YARCLET] = np.nan
    p = tmp_path / "rli.csv"
    df.to_csv(p, index=False)
    return p


def _n_observed(panel) -> int:
    return int(sum(int(panel.obs_mask[s].sum()) for s in panel.outcomes))


def test_load_wave_panel_shapes_mask_and_denominators(tmp_path):
    p = _write_panel_csv(tmp_path, n_children=20)
    panel = load_wave_panel(path=p)

    assert panel.n_children == 20
    assert panel.n_waves == 4
    assert panel.outcomes == ("W", "L", "E")
    for s, denom in (("W", 90), ("L", 32), ("E", 170)):
        assert panel.counts[s].shape == (20, 4)
        assert panel.n_trials[s] == denom
        # logit is NaN exactly where the count is missing.
        assert np.array_equal(np.isnan(panel.logit[s]), ~panel.obs_mask[s])

    # Injected missing cells are masked out (S000/S001 sort to indices 0/1).
    assert not panel.obs_mask["W"][0, 1]  # S000, wave t2
    assert not panel.obs_mask["L"][1, 2]  # S001, wave t3
    assert _n_observed(panel) == 20 * 4 * 3 - 2

    # Age (interpolated) and dose (NaN->0) are fully populated after prep.
    assert not np.isnan(panel.age_std).any()
    assert not np.isnan(panel.dose_std).any()
    assert panel.age_std.shape == (20, 4)


def test_lcsm_factory_builds_and_samples_prior(tmp_path):
    p = _write_panel_csv(tmp_path, n_children=20)
    panel = load_wave_panel(path=p)
    built = build_lcsm_model(panel)
    names = {v.name for v in built.model.free_RVs}
    assert {"g_L", "g_E", "b_self", "kappa", "sigma_proc"}.issubset(names)
    assert any(v.name == "x_latent" for v in built.model.deterministics)
    with built.model:
        pp = pm.sample_prior_predictive(draws=5, random_seed=1)
    assert pp.prior_predictive["y_obs"].shape[-1] == _n_observed(panel)


def test_lcsm_factory_process_noise_toggle(tmp_path):
    p = _write_panel_csv(tmp_path, n_children=15)
    panel = load_wave_panel(path=p)
    built = build_lcsm_model(panel, use_process_noise=False)
    names = {v.name for v in built.model.free_RVs}
    assert "sigma_proc" not in names
    assert not any(n.startswith("zproc_") for n in names)


def test_riclpm_structure_mask_entries():
    out = ("W", "L", "E")
    assert int((riclpm_structure_mask("ar", out) - np.eye(3)).sum()) == 0
    l2r = riclpm_structure_mask("l_to_r", out)
    assert l2r[0, 1] == 1.0  # A[W<-L] free
    assert l2r[1, 0] == 0.0  # A[L<-W] not free
    assert int((l2r - np.eye(3)).sum()) == 1
    rdr = riclpm_structure_mask("r_driven", out)
    assert rdr[1, 0] == 1.0 and rdr[2, 0] == 1.0  # reading drives L and E
    assert int((rdr - np.eye(3)).sum()) == 2
    assert int((riclpm_structure_mask("reciprocal", out) - np.eye(3)).sum()) == 6
    with pytest.raises(ValueError):
        riclpm_structure_mask("nope", out)


@pytest.mark.parametrize("structure", ["ar", "l_to_r", "r_driven", "reciprocal"])
def test_riclpm_factory_builds_each_structure(tmp_path, structure):
    p = _write_panel_csv(tmp_path, n_children=18)
    panel = load_wave_panel(path=p)
    built = build_riclpm_model(panel, structure=structure)
    free = {v.name for v in built.model.free_RVs}
    dets = {v.name for v in built.model.deterministics}
    assert {"A_raw", "zu", "zw1", "zinn", "kappa"}.issubset(free)
    assert {"A", "u_child", "m_logit"}.issubset(dets)
    with built.model:
        pp = pm.sample_prior_predictive(draws=5, random_seed=2)
    assert pp.prior_predictive["y_obs"].shape[-1] == _n_observed(panel)
