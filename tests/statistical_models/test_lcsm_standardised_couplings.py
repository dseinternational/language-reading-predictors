# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The SD-standardised level -> change couplings (2026-08-19).

Raw LCSM couplings are per unit of the *source's* latent logit, so sources on
different latent scales are not comparable in size; the table standardises each
by the model's own latent scales and contrasts sources of the same target.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from language_reading_predictors.statistical_models.pipelines.lcsm import (
    standardised_coupling_rows,
    write_standardised_couplings,
)


def _posterior(seed: int = 0) -> xr.Dataset:
    """A small posterior with two sources (L, E) into one target (W), where the
    latent L levels spread three times as widely as the latent E levels."""
    rng = np.random.default_rng(seed)
    chain, draw, child, wave = 2, 50, 12, 4
    x = np.zeros((chain, draw, child, wave, 3))
    x[..., 0] = rng.normal(0.0, 0.5, size=(chain, draw, child, wave))  # W
    x[..., 1] = rng.normal(0.0, 1.5, size=(chain, draw, child, wave))  # L
    x[..., 2] = rng.normal(0.0, 0.5, size=(chain, draw, child, wave))  # E
    coords = {
        "chain": np.arange(chain),
        "draw": np.arange(draw),
        "child": np.arange(child),
        "wave": np.arange(wave),
        "outcome": ["W", "L", "E"],
    }
    return xr.Dataset(
        {
            "x_latent": (("chain", "draw", "child", "wave", "outcome"), x),
            "g_L": (("chain", "draw"), np.full((chain, draw), 0.14)),
            "g_E": (("chain", "draw"), np.full((chain, draw), 0.28)),
        },
        coords=coords,
    )


def test_standardised_rows_apply_the_dominance_formula_and_contrast_sources():
    post = _posterior()
    names = {("L", "W"): "g_L", ("E", "W"): "g_E"}
    rows = standardised_coupling_rows(post, names, 0.89)
    by = {r["coefficient"]: r for r in rows}
    assert set(by) == {
        "std g (L -> W change)",
        "std g (E -> W change)",
        "std g L->W - std g E->W (contrast)",
        "|std g L->W| - |std g E->W| (dominance)",
    }
    # Hand computation of g* = g * sd(prior source levels) / sd(target changes),
    # per draw, then the median over draws.
    x = post["x_latent"]
    sd_dt = x.sel(outcome="W").diff("wave").std(dim=("child", "wave"))
    sd_L = x.isel(wave=slice(0, -1)).sel(outcome="L").std(dim=("child", "wave"))
    sd_E = x.isel(wave=slice(0, -1)).sel(outcome="E").std(dim=("child", "wave"))
    g_L = (post["g_L"] * sd_L / sd_dt).values.ravel()
    g_E = (post["g_E"] * sd_E / sd_dt).values.ravel()
    assert by["std g (L -> W change)"]["median"] == float(np.median(g_L))
    assert by["std g (E -> W change)"]["median"] == float(np.median(g_E))
    # Raw E is twice raw L, but L's levels spread three times wider: standardised,
    # L is the larger and the signed contrast is positive with certainty here.
    assert by["std g (L -> W change)"]["median"] > by["std g (E -> W change)"]["median"]
    assert by["std g L->W - std g E->W (contrast)"]["prob_pos"] == 1.0
    assert by["std g (L -> W change)"]["kind"] == "standardised_coupling"
    assert by["std g (L -> W change)"]["source"] == "L"
    assert by["std g (L -> W change)"]["target"] == "W"
    assert by["std g L->W - std g E->W (contrast)"]["kind"] == "contrast"
    assert by["|std g L->W| - |std g E->W| (dominance)"]["kind"] == "dominance"


def test_contrasts_only_between_sources_of_the_same_target():
    post = _posterior()
    # L -> W and E -> L: different targets, so no contrast row.
    post = post.assign(g_E_L=(("chain", "draw"), np.full((2, 50), 0.1)))
    names = {("L", "W"): "g_L", ("E", "L"): "g_E_L"}
    rows = standardised_coupling_rows(post, names, 0.89)
    kinds = [r["kind"] for r in rows]
    assert kinds == ["standardised_coupling", "standardised_coupling"]


def test_writer_works_over_a_stored_fit_with_a_lightweight_context(tmp_path):
    post = _posterior()
    ctx = SimpleNamespace(output_dir=str(tmp_path), reporting=SimpleNamespace(ci_prob=0.89))
    df = write_standardised_couplings(ctx, post, {("L", "W"): "g_L", ("E", "W"): "g_E"})
    assert df is not None and len(df) == 4
    on_disk = pd.read_csv(tmp_path / "standardised_couplings.csv")
    assert list(on_disk["coefficient"]) == list(df["coefficient"])
    assert write_standardised_couplings(ctx, post, {}) is None
