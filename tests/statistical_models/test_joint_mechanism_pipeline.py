# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Output-contract guards for the joint bivariate mechanism family (#421 Tier 3 (1)).

The #427 review found four house artefacts silently absent from a fit that reported
success: the per-outcome LOO-PIT plots, ``ppc_summary.csv``, power-scaling prior
sensitivity, and the inner 50% intervals. All four are guarded here — the LOO-PIT one
by asserting the pipeline asks for it per outcome by name, with the artefact-level
counterpart in ``test_diagnostics.py`` where the helper lives — so a future refactor
cannot drop any of them without a red test.

The marginal-coverage guard is the subtle one. In the levels design the residual is
saturated (one bivariate latent per child over exactly two cells), so the conditional
coverage is 1.00 by construction and an implementation that quietly reuses the fitted
residuals would report that same vacuous figure under a "new-child" label. That is not
hypothetical: it is what ``pm.sample_posterior_predictive(var_names=[...])`` did.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from language_reading_predictors.statistical_models import (
    joint_mechanism as _joint_mechanism,
)
from language_reading_predictors.statistical_models.new_child_kfold import KFoldPlan
from language_reading_predictors.statistical_models.new_child_predictive import (
    NewChildPlan,
)
from language_reading_predictors.statistical_models.registry import discover_models
from language_reading_predictors.statistical_models.lrp_rli_jm_001 import SPEC as JM001
from language_reading_predictors.statistical_models.lrp_rli_jm_002 import SPEC as JM002
from language_reading_predictors.statistical_models.joint_mechanism import (
    JointMechanismModelSettings,
    resolve_joint_mechanism_run_plan,
)
from language_reading_predictors.statistical_models.pipelines import (
    joint_mechanism as _jm_pipeline,
)
from language_reading_predictors.statistical_models.pipelines.joint_mechanism import (
    _JM_SLOPE_REQUIRED,
    _JM_TERM_LABELS,
    _jm_cell_outcome_labels,
    _jm_exposure_logit_sd,
    _jm_marginal_ppc,
    _jm_primary_fit_plan,
    _jm_ratio_governance_row,
    _jm_ratio_stability,
    _jm_slope_rows,
    _jm_wave_eligibility,
    _jm_write_slopes,
)

_OUTCOMES = ("W", "N")
#: The refit estimator's plan and the run plan the fold rebuild reads (#626).
_KFOLD_PLAN = KFoldPlan(n_folds=2)
_RUN_PLAN = _joint_mechanism.resolve_joint_mechanism_run_plan(
    discover_models()["lrp-rli-jm-002"].load().SPEC
)

#: The family's declared out-of-sample target (#626), as the levels design resolves it.
_NEW_CHILD_PLAN = NewChildPlan(
    child_dims=("obs_id",),
    latent_vars=("u_resid_z",),
    observed_nodes=("y_post",),
)
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
        "abs_slope_reduction",
    }


# ---------------------------------------------------------------------------
# 2026-08-23 joint audit, findings 4 and 10: the reported labels describe what the
# model identifies, and the ratio is published only where it is stable.
# ---------------------------------------------------------------------------


def _levels_trace_with(*, denominator_scale: float, held_scale: float) -> xr.DataTree:
    """A levels-shaped posterior with tunable ratio-instability routes."""
    rng = np.random.default_rng(11)
    chains, draws = 2, 400
    data = {
        "beta_mech": (
            ("chain", "draw", "outcome"),
            np.stack(
                [
                    rng.normal(denominator_scale, 0.08, size=(chains, draws)),
                    rng.normal(1.00, 0.20, size=(chains, draws)),
                ],
                axis=-1,
            ),
        ),
        "delta_ls_decoding": (
            ("chain", "draw"),
            rng.normal(0.75, 0.20, size=(chains, draws)),
        ),
        "rho_outcome": (("chain", "draw"), rng.normal(0.40, 0.15, size=(chains, draws))),
        "beta_held_on_focal": (
            ("chain", "draw"),
            rng.normal(0.30, 0.09, size=(chains, draws)),
        ),
        "beta_mech_focal_given_held": (
            ("chain", "draw"),
            rng.normal(0.18, 0.06, size=(chains, draws)),
        ),
        "share_retained": (
            ("chain", "draw"),
            rng.normal(0.72, 0.12, size=(chains, draws)),
        ),
        "sigma_u_resid": (
            ("chain", "draw", "outcome"),
            np.stack(
                [
                    np.full((chains, draws), 0.80),
                    np.abs(rng.normal(held_scale, 0.01, size=(chains, draws))),
                ],
                axis=-1,
            ),
        ),
    }
    posterior = xr.Dataset(
        data,
        coords={
            "chain": range(chains),
            "draw": range(draws),
            "outcome": list(_OUTCOMES),
        },
    )
    return xr.DataTree.from_dict({"posterior": posterior})


def test_the_reported_labels_do_not_claim_construct_level_decoding_specificity():
    """The contrast is between two adjusted test-score associations. The tests
    differ in item count, distribution, discrimination, reliability and floor
    behaviour, and nothing calibrates them to a common latent outcome scale, so a
    shared ability loading differently on them produces a non-zero contrast alone."""
    delta = _JM_TERM_LABELS["delta_ls_decoding"]
    assert "operational test-score slope contrast" in delta
    assert "not a construct-level" in delta
    share = _JM_TERM_LABELS["share_retained"]
    assert "ratio of adjusted associations" in share
    assert "share" not in share.lower().split("ratio")[0]


def test_a_stable_ratio_is_published_but_never_with_a_posterior_mean():
    trace = _levels_trace_with(denominator_scale=1.0, held_scale=0.8)
    rows = {
        r["term"]: r
        for r in _jm_slope_rows(
            trace,
            outcome_symbols=_OUTCOMES,
            contrast=_CONTRAST,
            ci_prob=0.89,
            wave="t3",
            converged=True,
        )
    }
    share = rows["share_retained"]
    assert share["share_retained_stable"] is True
    assert np.isfinite(share["median"])
    # A ratio's posterior mean is a property of the draws, not of the quantity.
    assert np.isnan(share["mean"])
    assert np.isfinite(rows["abs_slope_reduction"]["median"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"denominator_scale": 0.0, "held_scale": 0.8},  # denominator at zero
        {"denominator_scale": 1.0, "held_scale": 0.001},  # held-fixed scale at zero
    ],
)
def test_an_unstable_ratio_is_withheld_and_the_difference_survives(kwargs):
    """Either instability route makes the ratio heavy-tailed, so a finite Monte
    Carlo summary describes the draws rather than the quantity. The denominator-free
    absolute reduction has no denominator to blow up and is still published."""
    trace = _levels_trace_with(**kwargs)
    stability = _jm_ratio_stability(trace, contrast=_CONTRAST)
    assert stability["share_retained_stable"] is False
    rows = {
        r["term"]: r
        for r in _jm_slope_rows(
            trace,
            outcome_symbols=_OUTCOMES,
            contrast=_CONTRAST,
            ci_prob=0.89,
            wave="t3",
            converged=True,
        )
    }
    share = rows["share_retained"]
    assert all(
        np.isnan(share[key])
        for key in ("median", "mean", "lo50", "hi50", "lo", "hi", "prob_pos")
    )
    assert np.isfinite(rows["abs_slope_reduction"]["median"])


def test_the_transition_design_has_no_ratio_to_stabilise():
    assert _jm_ratio_stability(_trace(levels=False), contrast=_CONTRAST) is None


def test_per_outcome_coverage_labels_come_from_the_saved_cell_map():
    """Never re-derived: a reconstructed ordering could misalign a measure with
    another's counts."""
    ctx = SimpleNamespace(
        trace=SimpleNamespace(
            constant_data={
                "y_post_cell_outcome": SimpleNamespace(
                    values=np.array([0, 1, 0, 1, 1])
                )
            }
        )
    )
    assert _jm_cell_outcome_labels(ctx, _OUTCOMES) == ["W", "N", "W", "N", "N"]
    empty = SimpleNamespace(trace=SimpleNamespace(constant_data={}))
    assert _jm_cell_outcome_labels(empty, _OUTCOMES) is None


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
    levels = resolve_joint_mechanism_run_plan(JM001).diagnostic_vars(available)
    assert {"sigma_u_resid", "rho_outcome", "share_retained"} <= set(levels)
    assert "beta_group_nuisance" in levels and "beta_G" not in levels

    transition = resolve_joint_mechanism_run_plan(JM002).diagnostic_vars(available)
    assert {"sigma_u_child", "rho_outcome", "gamma_own", "kappa"} <= set(transition)
    assert "beta_G" in transition and "beta_group_nuisance" not in transition


def test_registered_specs_declare_their_designs_and_comparators():
    """jm-001 must be constructed against ca-010 / ca-011 and jm-002 against
    mech-096 / mech-101, or the identified quantities are not even comparable with
    the paired-draws ones they sit beside. (Comparable in construction is not
    nested: each plan also carries an explicit ``comparator_equivalence`` statement
    saying what still differs — 2026-08-23 follow-up review, finding 2.)"""
    assert JM001.kind == JM002.kind == "joint_mechanism"
    assert JM001.estimand_type == JM002.estimand_type == "association"
    assert JM001.causal_status == JM002.causal_status == "none"

    assert isinstance(JM001.model_settings, JointMechanismModelSettings)
    assert isinstance(JM002.model_settings, JointMechanismModelSettings)
    assert JM001.extra == JM002.extra == {}
    levels = resolve_joint_mechanism_run_plan(JM001)
    transition = resolve_joint_mechanism_run_plan(JM002)
    assert levels.design == "levels"
    # ca-010 / ca-011 adjust for block design and hearing — including the
    # hs_missing indicator the missing-indicator policy pairs with the filled hs
    # (2026-08-21 joint-mechanism review, finding 1) — at a Normal(0, 0.3) slope.
    assert levels.declared_adjustment == ("blocks", "hs", "hs_missing")
    assert levels.predictor_slope_sigma == 0.3

    assert transition.design == "transition"
    # mech-096 / mech-101 share {G, A, HS, IS, SP} + own baseline.
    assert transition.declared_adjustment == (
        "hs",
        "hs_missing",
        "attend",
        "deapp_c",
        "deapp_c_missing",
    )
    assert levels.contrast == transition.contrast == ("N", "W")

    # Neither is claimed as a nested replacement, and the reason travels with the
    # plan into config.json, the recipe and the report.
    for plan in (levels, transition):
        assert plan.comparator_equivalence
        assert "NOT" in plan.comparator_equivalence
    assert "latent" in levels.comparator_equivalence
    assert "exposure scale" in transition.comparator_equivalence


# --- artefact contract: the four outputs the #427 review found silently absent ----


def _artefact_trace(*, n_obs: int = 12, exact: bool) -> xr.DataTree:
    """A trace complete enough for the coverage helpers.

    ``exact=True`` gives a *conditional* predictive that reproduces every observation
    — the saturated-residual situation the levels design is actually in — so a
    marginal helper that quietly reuses the conditional draws is detectable.
    """
    rng = np.random.default_rng(11)
    chains, draws, n_outcomes = 2, 60, 2
    n_trials = np.array([79, 6])
    # Two cells per row: (row, outcome) in the flattened order the factory writes.
    rows = np.repeat(np.arange(n_obs), n_outcomes)
    cols = np.tile(np.arange(n_outcomes), n_obs)
    observed = rng.integers(0, n_trials[cols] + 1)

    if exact:
        y_rep = np.broadcast_to(observed, (chains, draws, observed.size)).copy()
    else:
        y_rep = rng.integers(0, n_trials[cols] + 1, size=(chains, draws, observed.size))

    eta = rng.normal(0.0, 0.5, size=(chains, draws, n_obs, n_outcomes))
    u_resid = rng.normal(0.0, 0.3, size=(chains, draws, n_obs, n_outcomes))
    return xr.DataTree.from_dict(
        {
            "posterior": xr.Dataset(
                {
                    "eta": (("chain", "draw", "obs_id", "outcome"), eta),
                    "u_resid": (("chain", "draw", "obs_id", "outcome"), u_resid),
                    "sigma_u_resid": (
                        ("chain", "draw", "outcome"),
                        rng.uniform(0.5, 1.5, size=(chains, draws, n_outcomes)),
                    ),
                    "rho_outcome": (
                        ("chain", "draw"),
                        rng.uniform(0.1, 0.6, size=(chains, draws)),
                    ),
                    "beta_mech": (
                        ("chain", "draw", "outcome"),
                        rng.normal(0.5, 0.2, size=(chains, draws, n_outcomes)),
                    ),
                },
                coords={
                    "chain": range(chains),
                    "draw": range(draws),
                    "obs_id": range(n_obs),
                    "outcome": list(_OUTCOMES),
                },
            ),
            "observed_data": xr.Dataset({"y_post": ("cell", observed)}),
            "posterior_predictive": xr.Dataset(
                {"y_post": (("chain", "draw", "cell"), y_rep)}
            ),
            "constant_data": xr.Dataset(
                {
                    "y_post_cell_row": ("cell", rows),
                    "y_post_cell_outcome": ("cell", cols),
                }
            ),
        }
    )


def _artefact_ctx(tmp_path, trace) -> SimpleNamespace:
    return SimpleNamespace(
        output_dir=str(tmp_path),
        tables={},
        trace=trace,
        model=None,
        prepared=SimpleNamespace(n_trials={"W": 79, "N": 6}),
        sampling=SimpleNamespace(random_seed=5),
        reporting=SimpleNamespace(ci_prob=0.89),
    )


@pytest.fixture
def _silent_diagnostics(monkeypatch):
    """Stub the plotting/sampling stages so the artefact contract can be checked
    without a fit, recording the calls the review asked to be reinstated."""
    calls: dict[str, list] = {"loo_pit": [], "new_child": []}
    for name in (
        "summary_diagnostics",
        "sample_posterior_predictive",
        "save_joint_posterior_predictive_plot",
        "run_extended_diagnostics",
        "save_trace",
        "save_prior_posterior_plot",
    ):
        monkeypatch.setattr(_jm_pipeline._diag, name, lambda *a, **k: None)
    monkeypatch.setattr(
        _jm_pipeline._diag,
        "save_joint_loo_pit_plot",
        lambda ctx, symbol, **k: calls["loo_pit"].append((symbol, k.get("posterior_var"))),
    )
    # The new-child validation samples the real model (#626); these tests check the
    # artefact contract against a stub trace, so record the call instead of running it.
    monkeypatch.setattr(
        _jm_pipeline,
        "write_new_child_validation",
        lambda ctx, plan: calls["new_child"].append(plan),
    )
    return calls


def test_primary_plan_declares_psense_and_writes_custom_diagnostics(
    tmp_path, _silent_diagnostics
):
    """Coverage, reported-coefficient psense and named LOO-PIT stay declared."""
    ctx = _artefact_ctx(tmp_path, _artefact_trace(exact=False))

    plan = _jm_primary_fit_plan(
        outcome_symbols=_OUTCOMES,
        diag_vars=["beta_mech"],
        psense_vars=["beta_mech", "delta_ls_decoding"],
        new_child_plan=_NEW_CHILD_PLAN,
        kfold_plan=_KFOLD_PLAN,
        run_plan=_RUN_PLAN,
        exposure_scale=(0.0, 1.0),
    )
    assert plan.custom_posterior_predictive is not None
    assert plan.post_extended_audit is not None
    plan.custom_posterior_predictive(ctx)
    plan.post_extended_audit(ctx)

    assert plan.diagnostic_vars == ("beta_mech",)
    assert plan.psense_vars == ("beta_mech", "delta_ls_decoding")
    assert plan.extended_term == "delta_ls_decoding"
    assert plan.include_loo_pit is False
    summary = pd.read_csv(tmp_path / "ppc_summary.csv")
    assert set(summary["level_pct"]) == {50, 90}
    assert {"coverage", "n_total", "n_inside"} <= set(summary.columns)
    assert ctx.tables["ppc_summary"] is not None
    # One LOO-PIT per outcome, each naming this family's own coefficient — the
    # hard-coded `tau` was what made these silently no-op.
    assert _silent_diagnostics["loo_pit"] == [("W", "beta_mech"), ("N", "beta_mech")]
    # No marginal companion unless the design asks for one.
    assert not (tmp_path / "ppc_summary_marginal.csv").exists()


def test_no_loo_plan_skips_psis_artefacts_but_keeps_psense_groups():
    """The saturated levels design computes no PSIS-LOO: the plan must skip both
    PSIS-based artefacts (LOO and the per-outcome LOO-PIT built on the same
    weights) while still attaching the log-density groups power scaling needs —
    the mediation families' no-LOO route (2026-08-21 review, finding 2)."""
    plan = _jm_primary_fit_plan(
        outcome_symbols=_OUTCOMES,
        diag_vars=["beta_mech"],
        psense_vars=["beta_mech"],
        new_child_plan=_NEW_CHILD_PLAN,
        kfold_plan=_KFOLD_PLAN,
        run_plan=_RUN_PLAN,
        exposure_scale=(0.0, 1.0),
        compute_loo=False,
    )
    assert plan.compute_loo is False
    assert plan.post_extended_audit is None
    assert plan.post_sampling_audit is not None

    calls: list = []
    validated: list = []
    kfolded: list = []

    class _Diag:
        @staticmethod
        def compute_log_likelihood_and_prior(ctx, *, strict):
            calls.append(strict)

    real = _jm_pipeline._diag
    real_validate = _jm_pipeline.write_new_child_validation
    real_kfold = _jm_pipeline.write_child_kfold
    try:
        _jm_pipeline._diag = _Diag
        # The new-child validation samples the real model (#626), so record the call.
        # It runs in **both** designs, unlike the PSIS artefacts above: the saturated
        # per-child residual that rules out conditional LOO here is exactly what the
        # validation integrates away, so gating it on ``compute_loo`` would withhold
        # it from the one design that most needs it.
        _jm_pipeline.write_new_child_validation = lambda ctx, p: validated.append(p)
        _jm_pipeline.write_child_kfold = lambda ctx, p, k, rebuild: kfolded.append(k)
        plan.post_sampling_audit(SimpleNamespace())
    finally:
        _jm_pipeline._diag = real
        _jm_pipeline.write_new_child_validation = real_validate
        _jm_pipeline.write_child_kfold = real_kfold
    assert calls == [False]
    assert validated == [_NEW_CHILD_PLAN]
    # The stub returns ``None`` — the expected-absence case — so the refit route must
    # NOT fire: it would ask for the same inputs that were just found missing.
    assert kfolded == []

    with_loo = _jm_primary_fit_plan(
        outcome_symbols=_OUTCOMES,
        diag_vars=["beta_mech"],
        psense_vars=["beta_mech"],
        new_child_plan=_NEW_CHILD_PLAN,
        kfold_plan=_KFOLD_PLAN,
        run_plan=_RUN_PLAN,
        exposure_scale=(0.0, 1.0),
        compute_loo=True,
    )
    assert with_loo.compute_loo is True
    assert with_loo.post_extended_audit is not None
    # Not ``None`` any more: the new-child validation runs here too, and the density
    # groups it would otherwise attach are already attached by the LOO step.
    assert with_loo.post_sampling_audit is not None


def test_marginal_ppc_is_not_the_conditional_predictive(tmp_path, _silent_diagnostics):
    """The levels design is saturated in its residual, so the conditional coverage is
    1.00 by construction. The marginal companion must redraw the residual — a version
    that quietly reuses the fitted values would report the same vacuous 1.00 under a
    'new-child' label, which is exactly what `pm.sample_posterior_predictive` did."""
    ctx = _artefact_ctx(tmp_path, _artefact_trace(exact=True))

    plan = _jm_primary_fit_plan(
        outcome_symbols=_OUTCOMES,
        diag_vars=["beta_mech"],
        psense_vars=["beta_mech"],
        new_child_plan=_NEW_CHILD_PLAN,
        kfold_plan=_KFOLD_PLAN,
        run_plan=_RUN_PLAN,
        exposure_scale=(0.0, 1.0),
        marginal_ppc=True,
    )
    assert plan.custom_posterior_predictive is not None
    plan.custom_posterior_predictive(ctx)

    conditional = pd.read_csv(tmp_path / "ppc_summary.csv")
    marginal = pd.read_csv(tmp_path / "ppc_summary_marginal.csv")
    # The conditional predictive reproduces every observation exactly.
    assert (conditional["coverage"] == 1.0).all()
    # The marginal one is a genuinely different, falsifiable statistic.
    assert set(marginal["mode"]) == {"count_interval_marginal"}
    assert (marginal["coverage"] < 1.0).any()
    # Pooled *and* per-outcome, at both levels: the two denominators differ by an
    # order of magnitude, so a pooled figure alone can hide one badly calibrated leg
    # (2026-08-23 follow-up review, robustness gap 2).
    assert set(marginal["level_pct"]) == {50, 90}
    # The pooled row leaves ``outcome`` null and the per-outcome rows name it — the
    # convention ``ppc_interval_coverage_by_group`` uses for the conditional table,
    # so one filter reads both files.
    assert set(marginal["outcome"].dropna()) == {"W", "N"}
    for level in (50, 90):
        at_level = marginal[marginal["level_pct"] == level]
        pooled = at_level[at_level["outcome"].isna()].iloc[0]
        legs = at_level[at_level["outcome"].notna()]
        assert int(pooled["n_total"]) == int(legs["n_total"].sum())
        assert int(pooled["n_inside"]) == int(legs["n_inside"].sum())


def test_marginal_ppc_degrades_when_the_covariance_block_is_absent(tmp_path):
    """A design with no residual covariance (the transition companion) must simply not
    write the file, rather than fabricating a marginal from missing variables."""
    full = _artefact_trace(exact=True)
    # Rebuild without the correlation: DataTree.ds hands back a copy, so deleting
    # from it would leave the real tree untouched and silently pass.
    trace = xr.DataTree.from_dict(
        {
            "posterior": full["posterior"].ds.drop_vars("rho_outcome"),
            "observed_data": full["observed_data"].ds,
            "posterior_predictive": full["posterior_predictive"].ds,
            "constant_data": full["constant_data"].ds,
        }
    )
    assert "rho_outcome" not in trace["posterior"].ds
    ctx = _artefact_ctx(tmp_path, trace)

    _jm_marginal_ppc(ctx, outcome_symbols=_OUTCOMES)

    assert not (tmp_path / "ppc_summary_marginal.csv").exists()


# --- 2026-08-23 follow-up review (#591): lifecycle, governance and semantics -----


def _prepared_wave(
    *,
    n: int,
    observed_w: int | None = None,
    observed_n: int | None = None,
    mechanism_missing: int = 0,
):
    """A minimal one-wave prepared frame for the eligibility rule.

    Only the fields the rule reads: the exposure and each outcome's post counts.
    ``observed_*`` truncates how many rows carry that outcome, which is how an
    asymmetric-missingness wave is built.
    """
    def _column(observed: int | None) -> np.ndarray:
        values = np.full(n, 5.0)
        if observed is not None:
            values[observed:] = np.nan
        return values

    mechanism = np.full(n, 3.0)
    if mechanism_missing:
        mechanism[:mechanism_missing] = np.nan
    return SimpleNamespace(
        n_obs=n,
        n_trials={"L": 26, "W": 79, "N": 6},
        post_counts={
            "L": mechanism,
            "W": _column(observed_w),
            "N": _column(observed_n),
        },
    )


def test_a_wave_needs_rows_on_each_outcome_and_on_jointly_observed_pairs():
    """The union count bounds neither leg nor the overlap, and it is the overlap that
    identifies the residual correlation and the conditional slope (2026-08-23 review,
    robustness gap 1)."""
    plan = resolve_joint_mechanism_run_plan(JM001)
    assert (
        plan.min_wave_rows,
        plan.min_wave_outcome_rows,
        plan.min_wave_overlap_rows,
    ) == (10, 10, 10)

    healthy = _jm_wave_eligibility(
        _prepared_wave(n=40),
        plan=plan,
        outcome_symbols=_OUTCOMES,
        timepoint=1,
    )
    assert healthy["fitted"] is True
    assert healthy["cells_W"] == healthy["cells_N"] == 40
    assert healthy["jointly_observed_rows"] == 40
    assert healthy["skipped_because"] == ""

    # 40 usable rows, but only 4 of them observe N: the union floor passes and the
    # per-outcome floor is what stops a prior-dominated rho being published.
    thin_leg = _jm_wave_eligibility(
        _prepared_wave(n=40, observed_n=4),
        plan=plan,
        outcome_symbols=_OUTCOMES,
        timepoint=2,
    )
    assert thin_leg["fitted"] is False
    assert "4 N cells < 10" in thin_leg["skipped_because"]
    assert thin_leg["usable_rows"] == 40

    # Both legs are well observed, but on disjoint rows: no jointly observed pair.
    disjoint = _prepared_wave(n=40)
    disjoint.post_counts["W"][20:] = np.nan
    disjoint.post_counts["N"][:20] = np.nan
    no_overlap = _jm_wave_eligibility(
        disjoint, plan=plan, outcome_symbols=_OUTCOMES, timepoint=3
    )
    assert no_overlap["fitted"] is False
    assert "0 jointly observed rows < 10" in no_overlap["skipped_because"]


def test_the_wave_ledger_separates_wave_eligibility_from_the_panel_drop_count():
    """A wave subset inherits ``dropped_rows`` from the four-timepoint panel, so the
    wave-specific counts have to be recorded separately (2026-08-23 review, gap 3)."""
    plan = resolve_joint_mechanism_run_plan(JM001)
    record = _jm_wave_eligibility(
        _prepared_wave(n=40, mechanism_missing=3),
        plan=plan,
        outcome_symbols=_OUTCOMES,
        timepoint=2,
    )
    assert record["panel_rows_at_wave"] == 40
    assert record["usable_rows"] == 37
    assert record["wave_eligibility_dropped"] == 3
    assert record["wave"] == "t2" and record["timepoint"] == 2


def _ratio_trace(ratio_draws: np.ndarray, denominator_draws: np.ndarray):
    chains, draws = 1, ratio_draws.size
    return xr.DataTree.from_dict(
        {
            "posterior": xr.Dataset(
                {
                    "share_retained": (
                        ("chain", "draw"),
                        ratio_draws.reshape(chains, draws),
                    ),
                    "beta_mech": (
                        ("chain", "draw", "outcome"),
                        np.stack(
                            [
                                denominator_draws.reshape(chains, draws),
                                np.full((chains, draws), 1.0),
                            ],
                            axis=-1,
                        ),
                    ),
                },
                coords={
                    "chain": range(chains),
                    "draw": range(draws),
                    # beta_mech's first column is W, the focal outcome.
                    "outcome": ["W", "N"],
                },
            )
        }
    )


def test_the_slope_ratio_publishes_no_mean_and_reports_its_three_regions():
    """A ratio's mean is dominated by small-denominator draws, and classifying its
    median against 0.5 reads a negative ratio as 'most of it runs through decoding'.
    Both are retired (2026-08-23 review, finding 5)."""
    rng = np.random.default_rng(3)
    # A suppression-and-amplification mixture: mass below zero and above one.
    ratio = np.concatenate(
        [
            rng.normal(-0.4, 0.05, 200),
            rng.normal(0.6, 0.05, 600),
            rng.normal(1.4, 0.05, 200),
        ]
    )
    trace = _ratio_trace(ratio, rng.normal(0.8, 0.05, ratio.size))

    rows = _jm_slope_rows(
        trace,
        outcome_symbols=_OUTCOMES,
        contrast=_CONTRAST,
        ci_prob=0.89,
        wave="t1",
        converged=True,
    )
    ratio_row = next(r for r in rows if r["term"] == "share_retained")
    assert np.isnan(ratio_row["mean"])
    assert np.isfinite(ratio_row["median"])
    # Every other reported term keeps its mean.
    assert all(
        np.isfinite(r["mean"]) for r in rows if r["term"] != "share_retained"
    )

    governance = _jm_ratio_governance_row(
        trace, ci_prob=0.89, wave="t1", contrast=_CONTRAST, converged=True
    )
    assert governance is not None
    # The governance table reproduces the pipeline's ONE stability rule rather than
    # applying a second, competing one (#591 follow-up review, finding 5).
    assert governance["share_retained_stable"] is True
    assert (
        governance["share_retained_stable"]
        is _jm_ratio_stability(trace, contrast=_CONTRAST)["share_retained_stable"]
    )
    assert governance["prob_lt_0"] == pytest.approx(0.2, abs=0.02)
    assert governance["prob_gt_1"] == pytest.approx(0.2, abs=0.02)
    assert governance["prob_in_unit"] == pytest.approx(0.6, abs=0.02)
    total = (
        governance["prob_lt_0"]
        + governance["prob_in_unit"]
        + governance["prob_gt_1"]
    )
    assert total == pytest.approx(1.0)
    assert "not a mediated share" in governance["label"]


def test_the_slope_ratio_is_marked_unstable_when_its_denominator_straddles_zero():
    """The identity divides by the unconditional slope, so the ratio means nothing
    once that slope is compatible with zero — and nothing in the pipeline said so."""
    rng = np.random.default_rng(4)
    denominator = rng.normal(0.0, 0.4, 2000)
    ratio = rng.normal(0.7, 0.3, 2000)
    governance = _jm_ratio_governance_row(
        _ratio_trace(ratio, denominator),
        ci_prob=0.89,
        wave="t2",
        contrast=_CONTRAST,
        converged=True,
    )
    assert governance is not None
    assert governance["share_retained_stable"] is False
    assert governance["denominator_lo"] < 0 < governance["denominator_hi"]
    assert governance["prob_denominator_above_minimum"] < 0.95


def test_the_transition_design_registers_no_ratio_to_govern():
    """No conditional slope, no ratio row — rather than a fabricated one."""
    assert (
        _jm_ratio_governance_row(
            _trace(levels=False),
            ci_prob=0.89,
            wave="stacked",
            contrast=_CONTRAST,
            converged=True,
        )
        is None
    )


def test_marginal_coverage_survives_asymmetric_outcome_missingness(tmp_path):
    """One outcome missing on some rows must reduce that leg's denominator, not
    silently pair an observation with another cell's replicate."""
    trace = _artefact_trace(exact=False, n_obs=12)
    observed = np.asarray(trace["observed_data"].ds["y_post"].values, dtype=float)
    cols = np.asarray(trace["constant_data"].ds["y_post_cell_outcome"].values)
    # Blank three nonword cells: the flattened likelihood carries NaN there.
    nonword_cells = np.flatnonzero(cols == 1)[:3]
    observed[nonword_cells] = np.nan
    trace = xr.DataTree.from_dict(
        {
            "posterior": trace["posterior"].ds,
            "observed_data": xr.Dataset({"y_post": ("cell", observed)}),
            "posterior_predictive": trace["posterior_predictive"].ds,
            "constant_data": trace["constant_data"].ds,
        }
    )
    ctx = _artefact_ctx(tmp_path, trace)

    _jm_marginal_ppc(ctx, outcome_symbols=_OUTCOMES)

    frame = pd.read_csv(tmp_path / "ppc_summary_marginal.csv")
    at_50 = frame[frame["level_pct"] == 50]
    by_outcome = at_50[at_50["outcome"].notna()].set_index("outcome")
    assert int(by_outcome.loc["W", "n_total"]) == 12
    assert int(by_outcome.loc["N", "n_total"]) == 9
    assert int(at_50[at_50["outcome"].isna()].iloc[0]["n_total"]) == 21


def test_the_exposure_scale_is_recorded_per_wave(tmp_path):
    """One SD is a different raw increment at every wave, because the exposure is
    re-standardised within each (2026-08-23 review, robustness gap 8)."""
    values = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    built = SimpleNamespace(
        prepared=SimpleNamespace(
            n_trials={"L": 26},
            post_counts={"L": values},
        )
    )
    from language_reading_predictors.statistical_models.preprocessing import logit_safe

    expected = float(np.std(logit_safe(values, 26), ddof=1))
    assert _jm_exposure_logit_sd(built, "L") == pytest.approx(expected)
    # A measure the frame does not carry is reported as unavailable, not as zero.
    assert _jm_exposure_logit_sd(built, "W") is None
