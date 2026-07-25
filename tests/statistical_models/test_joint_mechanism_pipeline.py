# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Output-contract guards for the joint bivariate mechanism family (#421 Tier 3 (1)).

The #427 review found four house artefacts silently absent from a fit that reported
success: the per-outcome LOO-PIT plots, ``ppc_summary.csv``, power-scaling prior
sensitivity, and the inner 50% intervals. Three of those are guarded here (the
LOO-PIT one in ``test_diagnostics.py``, where the helper lives), so a future refactor
cannot drop them without a red test.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from language_reading_predictors.statistical_models.lrp_rli_jm_001 import SPEC as JM001
from language_reading_predictors.statistical_models.lrp_rli_jm_002 import SPEC as JM002
from language_reading_predictors.statistical_models.pipeline import (
    _JM_SLOPE_REQUIRED,
    _jm_diag_vars,
    _jm_slope_rows,
    _jm_write_slopes,
)

_OUTCOMES = ("W", "N")
_CONTRAST = ("N", "W")


def _trace(*, levels: bool) -> xr.DataTree:
    """A posterior with the variables the family reports, in the given design."""
    rng = np.random.default_rng(7)
    chains, draws = 2, 200
    data = {
        "beta_mech": (
            ("chain", "draw", "outcome"),
            np.stack(
                [
                    rng.normal(0.25, 0.08, size=(chains, draws)),
                    rng.normal(1.00, 0.20, size=(chains, draws)),
                ],
                axis=-1,
            ),
        ),
        "delta_ls_decoding": (
            ("chain", "draw"),
            rng.normal(0.75, 0.20, size=(chains, draws)),
        ),
        "rho_outcome": (
            ("chain", "draw"),
            rng.normal(0.40, 0.15, size=(chains, draws)),
        ),
    }
    if levels:
        data["beta_held_on_focal"] = (
            ("chain", "draw"),
            rng.normal(0.30, 0.09, size=(chains, draws)),
        )
        data["beta_mech_focal_given_held"] = (
            ("chain", "draw"),
            rng.normal(0.18, 0.06, size=(chains, draws)),
        )
        data["share_retained"] = (
            ("chain", "draw"),
            rng.normal(0.72, 0.12, size=(chains, draws)),
        )
    posterior = xr.Dataset(
        data,
        coords={
            "chain": range(chains),
            "draw": range(draws),
            "outcome": list(_OUTCOMES),
        },
    )
    return xr.DataTree.from_dict({"posterior": posterior})


def _ctx(tmp_path) -> SimpleNamespace:
    return SimpleNamespace(
        output_dir=str(tmp_path),
        tables={},
        reporting=SimpleNamespace(ci_prob=0.89),
    )


def test_slope_rows_carry_the_house_interval_convention():
    """Median + inner 50% + outer 89% + P(>0) — #421's acceptance criterion. The
    first cut recorded only the outer interval (#427 review P2)."""
    rows = _jm_slope_rows(
        _trace(levels=True),
        outcome_symbols=_OUTCOMES,
        contrast=_CONTRAST,
        ci_prob=0.89,
        wave="t3",
        converged=True,
    )
    frame = pd.DataFrame(rows)
    assert _JM_SLOPE_REQUIRED <= set(frame.columns)
    for _, row in frame.iterrows():
        # The inner interval must be strictly inside the outer one, not a copy.
        assert row["lo"] < row["lo50"] < row["median"] < row["hi50"] < row["hi"]
    assert set(frame["term"]) == {
        "beta_mech[W]",
        "beta_mech[N]",
        "delta_ls_decoding",
        "rho_outcome",
        "beta_held_on_focal",
        "beta_mech_focal_given_held",
        "share_retained",
    }


def test_transition_rows_omit_the_conditional_slope_terms():
    """A between-child covariance does not answer "holding this child's decoding
    fixed at this wave", so the transition design must not publish a share retained
    just because the column exists in the schema."""
    rows = _jm_slope_rows(
        _trace(levels=False),
        outcome_symbols=_OUTCOMES,
        contrast=_CONTRAST,
        ci_prob=0.89,
        wave="stacked",
        converged=True,
    )
    terms = {row["term"] for row in rows}
    assert "delta_ls_decoding" in terms and "rho_outcome" in terms
    assert "share_retained" not in terms
    assert "beta_mech_focal_given_held" not in terms


def test_write_slopes_rejects_a_table_without_the_contrast(tmp_path):
    """The identified contrast is the deliverable; a table missing it must fail loudly
    rather than publish per-outcome slopes under a contrast heading."""
    ctx = _ctx(tmp_path)
    rows = _jm_slope_rows(
        _trace(levels=True),
        outcome_symbols=_OUTCOMES,
        contrast=_CONTRAST,
        ci_prob=0.89,
        wave="t3",
        converged=True,
    )
    written = _jm_write_slopes(ctx, rows, contrast=_CONTRAST)
    assert (tmp_path / "joint_mechanism_slopes.csv").exists()
    assert ctx.tables["joint_mechanism_slopes"] is written

    with pytest.raises(ValueError, match="delta_ls_decoding"):
        _jm_write_slopes(
            _ctx(tmp_path),
            [r for r in rows if r["term"] != "delta_ls_decoding"],
            contrast=_CONTRAST,
        )


def test_diag_vars_include_the_dependence_block_per_design():
    """The gate must cover the covariance parameters — they are what the family's
    identification claim rests on, so an unconverged correlation cannot pass
    unnoticed."""
    available = {
        "alpha", "beta_mech", "delta_ls_decoding", "beta_group_nuisance", "beta_G",
        "gamma_A", "gamma_hs", "gamma_own", "alpha_phase", "kappa",
        "sigma_u_resid", "sigma_u_child", "rho_outcome",
        "beta_mech_focal_given_held", "share_retained",
    }
    levels = _jm_diag_vars(
        available,
        design="levels",
        adjust_for=("hs",),
        confounder_symbols=("G", "A"),
        include_group=True,
    )
    assert {"sigma_u_resid", "rho_outcome", "share_retained"} <= set(levels)
    assert "beta_group_nuisance" in levels and "beta_G" not in levels

    transition = _jm_diag_vars(
        available,
        design="transition",
        adjust_for=("hs",),
        confounder_symbols=("G", "A"),
        include_group=True,
    )
    assert {"sigma_u_child", "rho_outcome", "gamma_own", "kappa"} <= set(transition)
    assert "beta_G" in transition and "beta_group_nuisance" not in transition


def test_registered_specs_declare_their_designs_and_comparators():
    """jm-001 must match ca-010 / ca-011 and jm-002 must match mech-096 / mech-101,
    or the identified quantities are not like-for-like replacements for the
    paired-draws ones they exist to replace."""
    assert JM001.kind == JM002.kind == "joint_mechanism"
    assert JM001.estimand_type == JM002.estimand_type == "association"
    assert JM001.causal_status == JM002.causal_status == "none"

    assert JM001.extra["design"] == "levels"
    # ca-010 / ca-011 adjust for block design and hearing at a Normal(0, 0.3) slope.
    assert tuple(JM001.extra["covariates"]) == ("blocks", "hs")
    assert JM001.extra["predictor_slope_sigma"] == 0.3

    assert JM002.extra["design"] == "transition"
    # mech-096 / mech-101 share {G, A, HS, IS, SP} + own baseline.
    assert tuple(JM002.extra["adjust_for"]) == (
        "hs",
        "hs_missing",
        "attend",
        "deapp_c",
        "deapp_c_missing",
    )
    assert JM001.extra["contrast"] == JM002.extra["contrast"] == ("N", "W")
