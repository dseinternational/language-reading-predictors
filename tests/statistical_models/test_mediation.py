# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for the g-formula mediation decomposition (``mediation.decompose``).

These build the joint mediator + outcome model on synthetic data to obtain a
``MediationData`` aligned to the model, then run :func:`decompose` against a
small *synthetic* posterior (no MCMC sampling) — fast and deterministic. The
focus is the issue #85 guarantee: the g-formula adjusts for exactly the
confounder set the model was fitted with, sourced from ``MediationData`` rather
than a hardcoded ``{E, R}``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import xarray as xr

from language_reading_predictors.statistical_models.factories import (
    build_mediation_model,
    build_period_stacked_mediation_model,
    build_two_mediator_model,
)
from language_reading_predictors.statistical_models.mediation import (
    _proportion_row,
    calibrate_session_confounding,
    decompose,
    decompose_period_stacked,
    decompose_two_mediator,
    sensitivity_sweep,
    sensitivity_sweep_two_mediator,
)
from language_reading_predictors.statistical_models.preprocessing import (
    load_and_prepare,
)

from .test_factories import _write_synthetic

_QUANTITIES = {"total", "NDE", "NIE", "proportion_mediated"}
_INTERVENTIONAL_QUANTITIES = {"total", "IDE", "IIE", "proportion_mediated"}
# Outcome- and mediator-leg coefficients the decomposition reads, minus the
# per-confounder ``b_<s>`` / ``a_<s>`` terms (added per test to match the
# fitted confounder set).
_OUTCOME_DRAWS = ["b0", "b_G", "b_M", "b_GM", "b_W", "b_A"]
_BB_MEDIATOR_DRAWS = ["a0", "a_G", "a_L", "a_A"]
_GAUSSIAN_MEDIATOR_DRAWS = ["a0", "a_G", "a_comp", "a_A"]


def _prepare(tmp_path, n_children: int = 15):
    p = _write_synthetic(tmp_path, n_children=n_children)
    return load_and_prepare(path=p, phase_mode="itt")


def _fake_trace(
    names, *, positive=(), values=None, chains: int = 2, draws: int = 20, seed: int = 0
):
    """A minimal posterior stand-in: an xarray ``Dataset`` of (chain, draw)
    coefficient draws exposed as ``.posterior`` — exactly what ``decompose``
    consumes (``trace.posterior[name].stack(...)``).

    ``values`` pins a coefficient to a near-constant value (tiny jitter for a
    non-degenerate posterior), so a deterministic scenario can be constructed.
    """
    rng = np.random.default_rng(seed)
    values = values or {}
    data = {}
    for name in names:
        if name in values:
            arr = float(values[name]) + rng.normal(0.0, 1e-3, size=(chains, draws))
        else:
            arr = rng.normal(0.0, 0.2, size=(chains, draws))
            if name in positive:
                arr = np.abs(arr) + 5.0  # kappa / sigma must be positive
        data[name] = (("chain", "draw"), arr)
    return SimpleNamespace(posterior=xr.Dataset(data))


def test_proportion_row_handles_all_nonfinite_ratios():
    """Degenerate Total == 0 on every draw: the NIE/Total ratio is non-finite for
    all draws, so the proportion interval must be NaN, not a np.quantile crash
    (reviewer #333)."""
    total = np.zeros(50)
    nie = np.full(50, 0.1)
    row = _proportion_row(nie, total, 0.025, 0.975)
    assert row["quantity"] == "proportion_mediated"
    for col in ("prob_median", "prob_lo", "prob_hi", "prob_lo90", "prob_hi90"):
        assert np.isnan(row[col])
    # P(Total > 0) is still well-defined on the full array (0 here).
    assert row["prob_pos"] == 0.0


def test_decompose_beta_binomial(tmp_path):
    """LRP59: single Beta-Binomial mediator, confounders {E, R}."""
    prep = _prepare(tmp_path)
    _, med = build_mediation_model(prep, confounder_symbols=("E", "R"))
    names = _OUTCOME_DRAWS + ["b_E", "b_R"] + _BB_MEDIATOR_DRAWS + ["a_E", "a_R", "kappa_M"]
    df = decompose(_fake_trace(names, positive=["kappa_M"]), med, n_replicates=4)
    assert set(df["quantity"]) == _QUANTITIES
    effects = df[df["quantity"].isin(["total", "NDE", "NIE"])]
    assert effects[["prob_mean", "words_mean"]].notna().all().all()


def test_decompose_offfloor_outcome(tmp_path):
    """#228 item 12 (LRP86, nonword N): an off-floor (Bernoulli) OUTCOME. The outcome
    leg drops the own-baseline b_W, so decompose must NOT read it, and it reports
    NIE/NDE on the off-floor risk-difference scale (n_trials_W = 1, so words_* ==
    prob_*)."""
    prep = _prepare(tmp_path)
    _, med = build_mediation_model(
        prep, confounder_symbols=("E", "R"), outcome_kind="bernoulli_offfloor"
    )
    assert med.off_floor is True
    assert med.n_trials_W == 1
    # Fake trace WITHOUT b_W (never created on the off-floor leg) and WITHOUT kappa_Y
    # (the Bernoulli has no dispersion) — so if decompose read b_W it would KeyError.
    names = (
        ["b0", "b_G", "b_M", "b_GM", "b_A", "b_E", "b_R"]
        + _BB_MEDIATOR_DRAWS
        + ["a_E", "a_R", "kappa_M"]
    )
    df = decompose(_fake_trace(names, positive=["kappa_M"]), med, n_replicates=4)
    assert set(df["quantity"]) == _QUANTITIES
    eff = df[df["quantity"].isin(["total", "NDE", "NIE"])]
    assert eff[["prob_mean", "words_mean"]].notna().all().all()
    # n_trials_W = 1 => the items ("words") columns collapse onto the risk difference.
    assert np.allclose(eff["prob_mean"].to_numpy(), eff["words_mean"].to_numpy())
    # Every row carries the off_floor flag so the report labels the scale.
    assert bool(df["off_floor"].iloc[0]) is True


def test_decompose_interventional_offfloor_outcome(tmp_path):
    """#323 (MED-186): the interventional flag and Bernoulli off-floor outcome
    compose cleanly. The rows use IDE/IIE labels, stay on the risk-difference
    scale and do not require the graded outcome's absent ``b_W`` coefficient."""
    prep = _prepare(tmp_path)
    _, med = build_mediation_model(
        prep, confounder_symbols=("E", "R"), outcome_kind="bernoulli_offfloor"
    )
    names = (
        ["b0", "b_G", "b_M", "b_GM", "b_A", "b_E", "b_R"]
        + _BB_MEDIATOR_DRAWS
        + ["a_E", "a_R", "kappa_M"]
    )
    trace = _fake_trace(names, positive=["kappa_M"])
    natural = decompose(trace, med, n_replicates=4)
    df = decompose(
        trace,
        med,
        n_replicates=4,
        interventional=True,
    )
    assert set(df["quantity"]) == _INTERVENTIONAL_QUANTITIES
    eff = df[df["quantity"].isin(["total", "IDE", "IIE"])]
    assert np.allclose(eff["prob_median"], eff["words_median"])
    assert bool(df["off_floor"].iloc[0]) is True
    # The current parametric implementation changes estimand labels/interpretation,
    # not the numerical functional. With the same trace and random seed, the
    # parent NIE/NDE and companion IIE/IDE must be exactly equal.
    natural = natural.replace({"quantity": {"NDE": "IDE", "NIE": "IIE"}})
    numeric = [c for c in df.columns if c not in ("quantity", "off_floor")]
    assert np.allclose(
        natural.set_index("quantity").loc[df["quantity"], numeric],
        df.set_index("quantity").loc[df["quantity"], numeric],
        equal_nan=True,
    )


def test_code_route_interventional_specs_mirror_their_parents():
    """The #323 companions may change only the estimand interpretation and
    identifying metadata; adjustment sets and model-building options must remain
    exactly aligned with MED-086/087."""
    from language_reading_predictors.statistical_models.lrp_rli_med_086 import (
        SPEC as med086,
    )
    from language_reading_predictors.statistical_models.lrp_rli_med_087 import (
        SPEC as med087,
    )
    from language_reading_predictors.statistical_models.lrp_rli_med_186 import (
        SPEC as med186,
    )
    from language_reading_predictors.statistical_models.lrp_rli_med_187 import (
        SPEC as med187,
    )

    for parent, companion in ((med086, med186), (med087, med187)):
        assert companion.kind == parent.kind == "mediation"
        assert companion.outcome_symbol == parent.outcome_symbol
        assert companion.mechanism_symbol == parent.mechanism_symbol
        assert companion.adjustment == parent.adjustment
        assert companion.adjustment is not parent.adjustment
        assert companion.extra == {
            **parent.extra,
            "estimand": "interventional",
            "companion_of": parent.model_id,
        }


def test_decompose_gaussian_composite(tmp_path):
    """LRP62: Gaussian route-composite mediator, confounders {E, R}."""
    prep = _prepare(tmp_path)
    _, med = build_mediation_model(
        prep,
        mediator_kind="gaussian_composite",
        route_symbols=("L", "B"),
        confounder_symbols=("E", "R"),
    )
    names = (
        _OUTCOME_DRAWS + ["b_E", "b_R"] + _GAUSSIAN_MEDIATOR_DRAWS + ["a_E", "a_R", "sigma_M"]
    )
    df = decompose(_fake_trace(names, positive=["sigma_M"]), med, n_replicates=4)
    assert set(df["quantity"]) == _QUANTITIES


def test_sensitivity_sweep(tmp_path):
    """#230: the NIE sensitivity sweep returns a delta-indexed NIE curve plus a
    summary whose flags are mutually exclusive (already-null / robust / tipping),
    reusing the g-formula via the ``b_m_shift`` lever."""
    prep = _prepare(tmp_path)
    _, med = build_mediation_model(prep, confounder_symbols=("E", "R"))
    names = _OUTCOME_DRAWS + ["b_E", "b_R"] + _BB_MEDIATOR_DRAWS + ["a_E", "a_R", "kappa_M"]
    trace = _fake_trace(names, positive=["kappa_M"], seed=1)
    sweep, summary = sensitivity_sweep(trace, med, n_deltas=6, n_replicates=4)
    assert sweep["delta"].iloc[0] == 0.0
    assert {"nie_median", "nie_lo", "nie_hi", "delta_frac_of_bM"} <= set(sweep.columns)
    assert {
        "tipping_delta", "tipping_frac_of_bM", "already_null_at_zero",
        "robust_over_full_sweep", "b_M_effective_mean",
    } <= set(summary)
    # Exactly one state holds: already-null at 0, robust over the whole sweep, or a
    # finite tipping point in between.
    finite_tip = not summary["already_null_at_zero"] and not summary[
        "robust_over_full_sweep"
    ]
    assert (
        int(summary["already_null_at_zero"])
        + int(summary["robust_over_full_sweep"])
        + int(finite_tip)
    ) == 1
    # Tie the flags to the value they summarise: a finite tipping_delta iff finite_tip.
    assert np.isfinite(summary["tipping_delta"]) == finite_tip


def test_sensitivity_sweep_attenuates_toward_null_for_both_signs(tmp_path):
    """#289 review: the sweep must shrink the effective slope toward 0 regardless of its
    sign. A fixed positive subtraction pushed a *negative* fitted slope further from 0 and
    silently reported ``robust_over_full_sweep=True``. Here treatment raises the mediator
    (a_G>0) and |b_M| is large, so the NIE is clearly non-null at delta=0 and must reach a
    finite tipping point (not "robust") for b_M>0 AND b_M<0."""
    prep = _prepare(tmp_path)
    _, med = build_mediation_model(prep, confounder_symbols=("E", "R"))
    names = _OUTCOME_DRAWS + ["b_E", "b_R"] + _BB_MEDIATOR_DRAWS + ["a_E", "a_R", "kappa_M"]
    for b_m in (3.0, -3.0):
        vals = {
            "b0": 0.0, "b_G": 0.0, "b_M": b_m, "b_GM": 0.0, "b_W": 0.0, "b_A": 0.0,
            "b_E": 0.0, "b_R": 0.0,
            "a0": 0.0, "a_G": 1.5, "a_L": 0.0, "a_A": 0.0, "a_E": 0.0, "a_R": 0.0,
        }
        trace = _fake_trace(names, positive=["kappa_M"], values=vals, seed=3)
        _, summary = sensitivity_sweep(trace, med, n_deltas=21, n_replicates=8)
        assert not summary["already_null_at_zero"], f"b_M={b_m}: NIE should be non-null at 0"
        assert not summary["robust_over_full_sweep"], (
            f"b_M={b_m}: sweep must attenuate toward 0 and find a tipping point, not 'robust'"
        )
        assert np.isfinite(summary["tipping_delta"])
        assert 0.0 < summary["tipping_delta"]
        # Fraction of the fitted slope is a positive magnitude whatever the slope's sign.
        assert summary["tipping_frac_of_bM"] > 0.0
        assert np.sign(summary["b_M_effective_mean"]) == np.sign(b_m)


def test_decompose_follows_fitted_confounder_set(tmp_path):
    """The core issue #85 guarantee: ``decompose`` adjusts for exactly the
    confounders carried on ``MediationData``. Fit with a single, *non-default*
    confounder (R only) and supply a posterior that contains only the R
    coefficients — under the old hardcoded ``{E, R}`` this would ``KeyError`` on
    the missing E coefficients / ``conf_logit['E']``."""
    prep = _prepare(tmp_path)
    _, med = build_mediation_model(prep, confounder_symbols=("R",))
    assert med.confounder_symbols == ("R",)
    assert set(med.conf_logit) == {"R"}
    names = _OUTCOME_DRAWS + ["b_R"] + _BB_MEDIATOR_DRAWS + ["a_R", "kappa_M"]
    df = decompose(_fake_trace(names, positive=["kappa_M"]), med, n_replicates=4)
    assert set(df["quantity"]) == _QUANTITIES


# --- Two-mediator per-leg sensitivity + IS calibration (#335) ---------------

_TWO_MEDIATOR_DRAWS = [
    "b0", "b_G", "b_L", "b_E", "b_GL", "b_GE", "b_W", "b_A", "b_R",
    "aL0", "aL_G", "aL_L", "aL_A", "aL_R", "kappa_L",
    "aE0", "aE_G", "aE_E", "aE_A", "aE_R", "kappa_E",
]


def _prepare_two_mediator(tmp_path, n_children: int = 30):
    p = _write_synthetic(tmp_path, n_children=n_children)
    prep = load_and_prepare(path=p, phase_mode="itt", covariates=("attend",))
    built, med = build_two_mediator_model(
        prep,
        outcome_symbol="W",
        mediator_symbols=("L", "E"),
        confounder_symbols=("R",),
    )
    return built.prepared, med


def _two_mediator_trace(seed: int = 17):
    values = {name: 0.0 for name in _TWO_MEDIATOR_DRAWS if not name.startswith("kappa")}
    values.update({"aL_G": 1.5, "aE_G": 1.0, "b_L": 2.5, "b_E": 1.5})
    return _fake_trace(
        _TWO_MEDIATOR_DRAWS,
        positive=("kappa_L", "kappa_E"),
        values=values,
        seed=seed,
    )


def test_decompose_two_mediator_per_leg_shift_attenuates_target(tmp_path):
    """A shift on the L outcome leg must attenuate NIE_L and the joint NIE."""
    _, med = _prepare_two_mediator(tmp_path)
    trace = _two_mediator_trace()
    base = decompose_two_mediator(trace, med, n_replicates=8).set_index("quantity")
    shifted = decompose_two_mediator(
        trace,
        med,
        n_replicates=8,
        b_m_shifts={"L": 2.0},
    ).set_index("quantity")
    assert abs(shifted.loc["NIE_L", "prob_median"]) < abs(
        base.loc["NIE_L", "prob_median"]
    )
    assert abs(shifted.loc["NIE_joint", "prob_median"]) < abs(
        base.loc["NIE_joint", "prob_median"]
    )


def test_sensitivity_sweep_two_mediator_reports_each_leg_and_joint(tmp_path):
    """Each one-dimensional sweep carries its path NIE and NIE_joint robustness."""
    _, med = _prepare_two_mediator(tmp_path)
    sweep, summary = sensitivity_sweep_two_mediator(
        _two_mediator_trace(),
        med,
        n_deltas=6,
        n_replicates=6,
    )
    assert set(sweep["mediator"]) == {"L", "E"}
    assert (sweep.groupby("mediator")["delta"].first() == 0.0).all()
    assert {
        "nie_median", "nie_lo", "nie_hi",
        "nie_joint_median", "nie_joint_lo", "nie_joint_hi",
        "delta_frac_of_effective_slope",
    } <= set(sweep.columns)
    assert set(summary["mediator"]) == {"L", "E"}
    assert {
        "tipping_delta", "already_null_at_zero", "robust_over_full_sweep",
        "joint_tipping_delta", "joint_already_null_at_zero",
        "joint_robust_over_full_sweep",
    } <= set(summary.columns)


def test_two_mediator_sweep_generalises_to_blending_and_chain(tmp_path):
    """MED-066 and chained MED-075 use B-labelled variables, not the E defaults."""
    p = _write_synthetic(tmp_path, n_children=30)
    prep = load_and_prepare(path=p, phase_mode="itt")
    names = [
        "b0", "b_G", "b_L", "b_B", "b_GL", "b_GB", "b_W", "b_A", "b_R",
        "aL0", "aL_G", "aL_L", "aL_A", "aL_R", "kappa_L",
        "aB0", "aB_G", "aB_B", "aB_A", "aB_R", "kappa_B",
    ]
    for chain in (False, True):
        _, med = build_two_mediator_model(
            prep,
            outcome_symbol="W",
            mediator_symbols=("L", "B"),
            confounder_symbols=("R",),
            chain=chain,
        )
        chain_names = [*names]
        if chain:
            chain_names.append("aB_L")
        values = {
            name: 0.0 for name in chain_names if not name.startswith("kappa")
        }
        values.update({"aL_G": 1.2, "aB_G": 0.8, "b_L": 2.0, "b_B": 1.0})
        trace = _fake_trace(
            chain_names,
            positive=("kappa_L", "kappa_B"),
            values=values,
            seed=23,
        )
        sweep, summary = sensitivity_sweep_two_mediator(
            trace,
            med,
            n_deltas=3,
            n_replicates=3,
            order=("L", "B"),
        )
        assert set(sweep["mediator"]) == {"L", "B"}
        assert set(summary["quantity"]) == {"NIE_L", "NIE_B"}


def test_session_calibration_returns_one_computed_row_per_leg(tmp_path):
    """The named IS benchmark is treated-arm only and produces both leg readouts."""
    prepared, med = _prepare_two_mediator(tmp_path, n_children=40)
    _, summary = sensitivity_sweep_two_mediator(
        _two_mediator_trace(seed=19),
        med,
        n_deltas=5,
        n_replicates=4,
    )
    calibration = calibrate_session_confounding(
        prepared,
        med,
        summary,
        n_bootstrap=80,
        seed=4,
    )
    assert set(calibration["mediator"]) == {"L", "E"}
    assert (calibration["n_treated"] >= 8).all()
    assert {
        "delta_is", "delta_is_lo", "delta_is_hi", "tipping_delta",
        "mediator_is_slope", "outcome_is_slope", "is_band_reaches_tipping",
        "conclusion",
    } <= set(calibration.columns)
    assert calibration["conclusion"].str.contains("NIE_").all()


# --- MED-092: period-stacked g-formula on the gain-factor scaffold (#229) ----

_PS_SCALARS = [
    "a0", "a_trt", "a_L", "a_A", "a_E", "a_R", "kappa_M",
    "b0", "b_trt", "b_M", "b_trtM", "b_W", "b_A", "b_E", "b_R", "kappa_Y",
]


def _prepare_stacked(tmp_path, n_children: int = 15):
    p = _write_synthetic(tmp_path, n_children=n_children)
    return load_and_prepare(path=p, phase_mode="all")


def _fake_trace_stacked(
    med, *, chains: int = 2, draws: int = 20, seed: int = 0, values=None
):
    """A synthetic posterior for :func:`decompose_period_stacked`: the scalar
    coefficients plus the vector parameters the stacked design adds (per-phase
    intercepts and the per-leg child random-intercept deterministics)."""
    rng = np.random.default_rng(seed)
    values = values or {}
    data = {}
    for name in _PS_SCALARS:
        if name in values:
            arr = float(values[name]) + rng.normal(0.0, 1e-3, size=(chains, draws))
        elif name.startswith("kappa"):
            arr = np.abs(rng.normal(0.0, 0.2, size=(chains, draws))) + 5.0
        else:
            arr = rng.normal(0.0, 0.2, size=(chains, draws))
        data[name] = (("chain", "draw"), arr)
    for name, dim, size in (
        ("a_phase", "phase", med.n_phases),
        ("b_phase", "phase", med.n_phases),
        ("u_child_M", "child", med.n_children),
        ("u_child_Y", "child", med.n_children),
    ):
        data[name] = (
            ("chain", "draw", dim),
            rng.normal(0.0, 0.1, size=(chains, draws, size)),
        )
    return SimpleNamespace(posterior=xr.Dataset(data))


def test_period_stacked_factory_builds(tmp_path):
    """MED-092: both legs get the gain-factor machinery (phase intercepts,
    per-leg child random intercepts) and the exposure is the on-intervention
    indicator, not the randomised group."""
    prep = _prepare_stacked(tmp_path)
    built, med = build_period_stacked_mediation_model(
        prep, confounder_symbols=("E", "R")
    )
    names = {v.name for v in built.model.free_RVs}
    assert {
        "a_trt", "a_L", "kappa_M", "sigma_child_M",
        "b_trt", "b_M", "b_trtM", "b_W", "kappa_Y", "sigma_child_Y",
        "a_phase", "b_phase",
    }.issubset(names)
    det = {v.name for v in built.model.deterministics}
    assert {"u_child_M", "u_child_Y", "mu_M", "eta"}.issubset(det)
    # Stacked rows: several phases, and the exposure varies (waitlist period 1 off).
    assert med.n_phases > 1
    assert set(np.unique(med.trt)) == {0.0, 1.0}
    assert med.trt.shape == med.phase_idx.shape == med.child_idx.shape
    import pymc as pm

    with built.model:
        pp = pm.sample_prior_predictive(draws=5, random_seed=1)
    assert pp.prior_predictive["y_post"].shape[-1] == built.prepared.n_obs


def test_period_stacked_factory_requires_all_phase_mode(tmp_path):
    p = _write_synthetic(tmp_path, n_children=15)
    prep_itt = load_and_prepare(path=p, phase_mode="itt")
    import pytest

    with pytest.raises(ValueError, match="phase_mode='all'"):
        build_period_stacked_mediation_model(prep_itt)


def test_period_stacked_factory_rejects_ungraded_denominator(tmp_path):
    """Graded-only contract: an outcome with a unit denominator would make the
    g-formula's "words" scale a risk difference while off_floor stays False, so
    the factory must fail fast (reviewer #333)."""
    import dataclasses

    import pytest

    prep = _prepare_stacked(tmp_path)
    # Nonword N has n_trials = 6; force it to 1 to trip the guard without needing
    # a floored measure with a real unit denominator.
    prep_bad = dataclasses.replace(prep, n_trials={**prep.n_trials, "W": 1})
    with pytest.raises(ValueError, match="graded-only"):
        build_period_stacked_mediation_model(prep_bad, confounder_symbols=("E", "R"))


def test_decompose_period_stacked_all_rows_and_p1_mask(tmp_path):
    """The stacked g-formula returns the standard quantity set, and the period-1
    row restriction (the LRP59-comparable readout) runs off the same posterior."""
    prep = _prepare_stacked(tmp_path)
    _, med = build_period_stacked_mediation_model(prep, confounder_symbols=("E", "R"))
    trace = _fake_trace_stacked(med)
    df = decompose_period_stacked(trace, med, n_replicates=4)
    assert set(df["quantity"]) == _QUANTITIES
    effects = df[df["quantity"].isin(["total", "NDE", "NIE"])]
    assert effects[["prob_mean", "words_mean"]].notna().all().all()
    df_p1 = decompose_period_stacked(
        trace, med, n_replicates=4, row_mask=med.phase_idx == 0
    )
    assert set(df_p1["quantity"]) == _QUANTITIES


def test_decompose_period_stacked_positive_exposure_effect(tmp_path):
    """A pinned-posterior sanity direction check: exposure raises the mediator
    (a_trt >> 0) and the mediator raises the outcome (b_M >> 0), everything else
    0 — so the NIE and total must come out positive."""
    prep = _prepare_stacked(tmp_path)
    _, med = build_period_stacked_mediation_model(prep, confounder_symbols=("E", "R"))
    vals = {n: 0.0 for n in _PS_SCALARS if not n.startswith("kappa")}
    vals.update({"a_trt": 2.0, "b_M": 2.0})
    trace = _fake_trace_stacked(med, values=vals, seed=5)
    df = decompose_period_stacked(trace, med, n_replicates=8).set_index("quantity")
    assert df.loc["NIE", "prob_mean"] > 0
    assert df.loc["total", "prob_mean"] > 0


def test_sensitivity_sweep_period_stacked(tmp_path):
    """The #230 sweep generalises to the stacked design via ``decompose_fn`` +
    ``interaction_name`` (its exposure x mediator coefficient is b_trtM)."""
    prep = _prepare_stacked(tmp_path)
    _, med = build_period_stacked_mediation_model(prep, confounder_symbols=("E", "R"))
    trace = _fake_trace_stacked(med, seed=2)
    sweep, summary = sensitivity_sweep(
        trace,
        med,
        n_deltas=4,
        n_replicates=4,
        decompose_fn=decompose_period_stacked,
        interaction_name="b_trtM",
    )
    assert sweep["delta"].iloc[0] == 0.0
    assert {"tipping_delta", "already_null_at_zero", "robust_over_full_sweep"} <= set(
        summary
    )
