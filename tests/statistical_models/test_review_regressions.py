# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Independent numerical and persisted-artifact checks for the September review."""

from types import SimpleNamespace

import numpy as np
import pymc as pm
import pytest
import xarray as xr
from scipy.integrate import quad
from scipy.special import expit
from scipy.stats import betabinom, norm

from language_reading_predictors.statistical_models import diagnostics, run_metadata
from language_reading_predictors.statistical_models.factories.mediation import MediationData
from language_reading_predictors.statistical_models.mediation import decompose
from language_reading_predictors.statistical_models.mediation_integration import normal_cells
from language_reading_predictors.statistical_models.pipelines.dose_response import dose_marginal_draws
from language_reading_predictors.statistical_models.preprocessing import logit_safe
from language_reading_predictors.statistical_models.subfits import run_subfit

from .test_diagnostics import _primary_reuse_context
from .test_subfits import _built, _ctx, _persisted_toy_trace, _write_reuse_contract


@pytest.mark.parametrize("link,scale", [("logit", 1.0), ("three_choice_guessing_floor", 2 / 3)])
def test_dose_endpoints_use_quartiles_even_when_observed_doses_differ(link, scale):
    # Two rows have different attendance and different nuisance predictors.
    observed = np.array([0.0, 4.0])
    nuisance = np.array([-1.0, 0.3])
    slope = 0.8
    eta = nuisance + slope * observed
    group = xr.Dataset(
        {
            "eta": (("chain", "draw", "obs_id"), eta[None, None, :]),
            "beta_dose": (("chain", "draw"), [[slope]]),
        }
    )
    common = dict(
        phase_idx=np.zeros(2, dtype=int),
        delta_std=np.ones(2) * 2,
        n_trials=10,
        period_varying=False,
        score_mean_link=link,
    )
    result = dose_marginal_draws(group, start_offset_std=1 - observed, **common)
    expected = scale * 10 * np.mean(expit(nuisance + slope * 3) - expit(nuisance + slope))
    np.testing.assert_allclose(result, [expected])
    # The shared DiD caller retains its explicit shift from observed attendance.
    shifted = dose_marginal_draws(group, **common)
    np.testing.assert_allclose(shifted, [scale * 10 * np.mean(expit(eta + 2 * slope) - expit(eta))])
    assert not np.isclose(result[0], shifted[0])


@pytest.mark.parametrize("name", ["beta", "measure_corr_chol"])
def test_an_unconstrained_matrix_cannot_pass_by_imitating_a_cholesky(name):
    rng = np.random.default_rng(12)
    values = rng.normal(size=(4, 1000, 2, 2))
    values[:, :, 0, 0] = 1
    values[:, :, 0, 1] = 0
    trace = xr.DataTree.from_dict(
        {
            "posterior": xr.Dataset({name: (("chain", "draw", "i", "j"), values)}),
            "sample_stats": xr.Dataset(
                {
                    "energy": (("chain", "draw"), rng.normal(size=(4, 1000))),
                    "diverging": (("chain", "draw"), np.zeros((4, 1000), bool)),
                }
            ),
        }
    )
    from language_reading_predictors.statistical_models.structural_constants import record_structural_constraints

    with pm.Model() as model:
        pm.Normal(name, shape=(2, 2))
    record_structural_constraints(trace.posterior, model)
    verdict = diagnostics.subfit_convergence(trace, label="ordinary matrix")
    assert verdict["converged"] is False
    assert f"{name}[0, 1]" in verdict["unassessable_parameters"]
    assert verdict["structurally_constant_parameters"] == ""


def test_primary_reuse_rejects_a_sign_reversal_with_unchanged_names_and_data(tmp_path):
    context, source = _primary_reuse_context(tmp_path)

    def build(sign):
        with pm.Model() as model:
            x = pm.Data("x", [0.0, 1.0, 2.0])
            beta = pm.Normal("beta")
            eta = pm.Deterministic("eta", sign * beta * x)
            pm.Bernoulli("y", logit_p=eta, observed=[0, 1, 1])
        return model

    context.model = build(1)
    run_metadata.write_run_metadata(SimpleNamespace(**{**vars(context), "output_dir": str(source)}))
    context.model = build(-1)
    with pytest.raises(ValueError, match="model_design_identity"):
        run_metadata.require_reuse_compatibility(context, source)


def test_secondary_reuse_binds_its_own_prior_even_when_primary_is_unchanged(tmp_path, monkeypatch):
    source, staging = tmp_path / "published", tmp_path / "staging"
    source.mkdir()
    staging.mkdir()
    ctx = _ctx(staging, draws=50, tune=50, chains=2)
    ctx.final_output_dir = str(source)
    _persisted_toy_trace(source / "trace_toy_subfit.nc")
    _write_reuse_contract(ctx, _built(prior=2), source)
    monkeypatch.setenv("DSE_LRP_REUSE_TRACE", "1")
    monkeypatch.setattr(pm, "sample", lambda **kw: pytest.fail("reuse must not sample"))
    # The context keeps the original primary model; only the secondary changes.
    with pytest.raises(ValueError, match="model_identity"):
        run_subfit(
            ctx, _built(prior=20), label="reused toy sub-fit", role="sensitivity", trace_filename="trace_toy_subfit.nc"
        )


def test_small_indirect_effect_intervals_match_exact_count_integration():
    rng = np.random.default_rng(903)
    coefficients = {
        "b0": 0.0,
        "b_G": 0.3,
        "b_M": 2.0,
        "b_GM": 0.0,
        "b_W": 0.0,
        "b_A": 0.0,
        "a0": 0.0,
        "a_G": 0.001,
        "a_L": 0.0,
        "a_A": 0.0,
        "kappa_M": 5.0,
    }
    post = xr.Dataset(
        {name: (("chain", "draw"), value + rng.normal(0, 1e-4, (4, 1000))) for name, value in coefficients.items()}
    )
    med = MediationData(
        G=np.tile([0, 1], 27),
        W1_logit=np.zeros(54),
        A_std=np.zeros(54),
        conf_logit={},
        n_trials_W=30,
        L1_logit=np.zeros(54),
        n_trials_L=26,
    )
    result = decompose(SimpleNamespace(posterior=post), med, ci_prob=0.89).set_index("quantity")
    values = {key: post[key].values.reshape(-1, 1) for key in coefficients}
    support = np.arange(27)[None, :]
    z = logit_safe(support, 26)
    p0, p1 = expit(values["a0"]), expit(values["a0"] + values["a_G"])
    mass0 = betabinom.pmf(support, 26, p0 * values["kappa_M"], (1 - p0) * values["kappa_M"])
    mass1 = betabinom.pmf(support, 26, p1 * values["kappa_M"], (1 - p1) * values["kappa_M"])
    outcome = expit(values["b0"] + values["b_G"] + (values["b_M"] + values["b_GM"]) * z)
    exact = np.sum((mass1 - mass0) * outcome, axis=1)
    np.testing.assert_allclose(
        result.loc["NIE", ["prob_lo", "prob_hi"]].to_numpy(float), np.quantile(exact, [0.055, 0.945]), atol=1e-13
    )
    assert result.loc["NIE", "prob_pos"] == 1
    assert result.loc["NIE", "prob_lo"] > 0


def test_normal_integration_matches_independent_adaptive_integral():
    def outcome(g, z):
        return expit(-0.4 + 0.2 * g + 1.3 * z)

    cells = normal_cells(outcome, np.array([[0.7]]), np.array([[-0.2]]), np.array([1.1]))
    expected = [
        quad(lambda z: outcome(g, z) * norm.pdf(z, mu, 1.1), -np.inf, np.inf, epsabs=1e-12)[0]
        for g, mu in [(1.0, 0.7), (0.0, -0.2), (1.0, -0.2)]
    ]
    np.testing.assert_allclose(cells[:, 0], expected, atol=1e-10)


def test_normal_integration_failure_stops_the_decomposition(monkeypatch):
    from language_reading_predictors.statistical_models import mediation_integration as integration

    monkeypatch.setattr(integration, "NORMAL_INTEGRATION_ORDERS", (2, 4))
    with pytest.raises(ValueError, match="integration did not converge"):
        normal_cells(lambda g, z: expit(z + 0.3 * g), np.array([[1.0]]), np.array([[0.0]]), np.array([10.0]))


@pytest.mark.parametrize("chain", [False, True])
@pytest.mark.parametrize("off_floor", [False, True])
def test_two_mediator_cells_match_full_cross_world_enumeration(chain, off_floor):
    from itertools import product
    from language_reading_predictors.statistical_models.factories.mediation import TwoMediatorData
    from language_reading_predictors.statistical_models.mediation import decompose_two_mediator

    values = dict(
        b0=-0.3,
        b_G=0.2,
        b_L=0.7,
        b_E=0.4,
        b_GL=0.15,
        b_GE=-0.1,
        b_W=0.0,
        b_A=0.0,
        aL0=-0.2,
        aL_G=0.4,
        aL_L=0.0,
        aL_A=0.0,
        kappa_L=5.0,
        aE0=0.1,
        aE_G=0.3,
        aE_E=0.0,
        aE_A=0.0,
        kappa_E=4.0,
        aE_L=0.6 if chain else 0.0,
        aE_own_offfloor=0.0,
    )
    rng = np.random.default_rng(8)
    post = xr.Dataset(
        {name: (("chain", "draw"), value + rng.normal(0, 1e-4, (2, 8))) for name, value in values.items()}
    )
    med = TwoMediatorData(
        G=np.array([0.0, 1.0]),
        A_std=np.zeros(2),
        W1_logit=np.zeros(2),
        conf1_logit={},
        n_trials_W=10,
        L1_logit=np.zeros(2),
        n_trials_L=2,
        zL_mean=0.0,
        zL_sd=1.0,
        E1_logit=np.zeros(2),
        n_trials_E=2,
        zE_mean=0.0,
        zE_sd=1.0,
        confounder_symbols=(),
        chain=chain,
        second_mediator_offfloor=off_floor,
        second_mediator_offfloor_pre=np.zeros(2),
    )
    result = decompose_two_mediator(SimpleNamespace(posterior=post), med, hdi_prob=0.89).set_index("quantity")
    draws = {k: post[k].values.ravel() for k in values}
    cells = np.zeros((5, 16))
    second_support = range(2 if off_floor else 3)
    for lt, lc, et, ec in product(range(3), range(3), second_support, second_support):
        zlt, zlc = logit_safe(np.array([lt, lc]), 2)
        zet, zec = (et, ec) if off_floor else logit_safe(np.array([et, ec]), 2)

        def mass_l(k, g):
            p = expit(draws["aL0"] + g * draws["aL_G"])
            return betabinom.pmf(k, 2, p * draws["kappa_L"], (1 - p) * draws["kappa_L"])

        def mass_e(k, g, zl):
            p = expit(draws["aE0"] + g * draws["aE_G"] + (draws["aE_L"] * zl if chain else 0))
            return (
                (p if k else 1 - p)
                if off_floor
                else betabinom.pmf(k, 2, p * draws["kappa_E"], (1 - p) * draws["kappa_E"])
            )

        def outcome(g, zl, ze):
            return expit(
                draws["b0"]
                + g * draws["b_G"]
                + (draws["b_L"] + g * draws["b_GL"]) * zl
                + (draws["b_E"] + g * draws["b_GE"]) * ze
            )

        weight = mass_l(lt, 1) * mass_l(lc, 0) * mass_e(et, 1, zlt) * mass_e(ec, 0, zlc)
        cells += weight * np.array(
            [
                outcome(1, zlt, zet),
                outcome(1, zlc, zec),
                outcome(0, zlc, zec),
                outcome(1, zlt, zec),
                outcome(1, zlc, zet),
            ]
        )
    expected = {
        "total": cells[0] - cells[2],
        "NDE": cells[1] - cells[2],
        "NIE_joint": cells[0] - cells[1],
        "NIE_L": cells[3] - cells[1],
        "NIE_E": cells[0] - cells[3],
    }
    for name, draws in expected.items():
        np.testing.assert_allclose(
            result.loc[name, ["prob_lo", "prob_median", "prob_hi"]].to_numpy(float),
            np.quantile(draws, [0.055, 0.5, 0.945]),
            atol=1e-13,
        )


@pytest.mark.parametrize("likelihood", ["beta_binomial", "bernoulli_offfloor"])
def test_gain_factory_loo_removes_all_of_each_childs_transitions(tmp_path, likelihood):
    from language_reading_predictors.statistical_models.factories import build_gain_factors_model
    from language_reading_predictors.statistical_models.preprocessing import load_and_prepare
    from .test_factories import _write_synthetic

    prepared = load_and_prepare(path=_write_synthetic(tmp_path, n_children=12), phase_mode="all")
    built = build_gain_factors_model(prepared, outcome_symbol="W", likelihood=likelihood)
    rows = built.model["loo_child_idx"].get_value()
    np.testing.assert_array_equal(rows, built.prepared.child_idx)
    assert len(rows) > built.prepared.n_children
    values = -np.arange(1, len(rows) + 1, dtype=float)
    node = "y_offfloor" if likelihood == "bernoulli_offfloor" else "y_post"
    trace = xr.DataTree.from_dict(
        {
            "constant_data": xr.Dataset({"loo_child_idx": ("obs_id", rows)}),
            "log_likelihood": xr.Dataset({node: (("chain", "draw", "obs_id"), values[None, None, :])}),
        }
    )
    grouped = diagnostics._joint_log_likelihood_by_child(trace)
    assert grouped.sizes["loo_child"] == built.prepared.n_children
    for child in range(built.prepared.n_children):
        assert float(grouped.isel(loo_child=child).values.item()) == values[rows == child].sum()


def test_kfold_saves_reuses_and_binds_each_training_partition(tmp_path, monkeypatch):
    import pandas as pd
    from language_reading_predictors.statistical_models import new_child_kfold as kfold
    from language_reading_predictors.statistical_models.context import ModelSpec
    from language_reading_predictors.statistical_models.new_child_predictive import NewChildPlan

    source, staging = tmp_path / "published", tmp_path / "staging"
    source.mkdir()
    staging.mkdir()
    built = _built()
    built.prepared.child_idx = np.arange(4)
    ctx = _ctx(source, draws=1000, tune=50, chains=4)
    ctx.model, ctx.prepared = built.model, built.prepared
    ctx.reporting = SimpleNamespace(config_name="reporting", ci_prob=0.89)
    ctx.resolved_plan = None
    ctx.spec = ModelSpec(model_id="lrp-rli-hg-999", kind="historical_growth", title="K-fold reuse")
    rng = np.random.default_rng(44)
    trace = xr.DataTree.from_dict(
        {
            "posterior": xr.Dataset({"p": (("chain", "draw"), rng.beta(3, 4, (4, 1000)))}),
            "sample_stats": xr.Dataset(
                {
                    "energy": (("chain", "draw"), rng.normal(size=(4, 1000))),
                    "diverging": (("chain", "draw"), np.zeros((4, 1000), bool)),
                }
            ),
            "observed_data": xr.Dataset({"y": ("obs_id", [3, 5, 2, 7])}),
        }
    )
    ctx.trace = trace
    trace.to_netcdf(source / "trace.nc")
    run_metadata.write_run_metadata(ctx)
    # Sampling and held-out scoring are already tested separately. Here real
    # persistence, primary compatibility, sub-fit compatibility and fold dispatch
    # run together; no sampler is permitted on the reuse pass.
    monkeypatch.setattr(pm, "sample", lambda **kw: trace.copy(deep=True))
    monkeypatch.setattr(kfold, "_score_held_out", lambda *args, **kw: (-np.arange(1.0, 5.0), None))
    monkeypatch.setattr(kfold, "_fold_pit", lambda *args, **kw: pd.DataFrame())

    def rebuild(training, held_out):
        return _built(counts=np.asarray([3, 5, 2, 7])[training], n_children=len(training), subject_ids=training)

    plan = NewChildPlan(child_dims=("obs_id",), latent_vars=())
    split = kfold.KFoldPlan(n_folds=2, random_seed=47)
    first = kfold.run_child_kfold(ctx, plan, split, rebuild)
    assert first.complete
    provenance = pd.read_csv(source / "subfit_provenance.csv")
    assert len(provenance) == 2
    assert provenance["trace_sha256"].str.len().eq(64).all()
    assert all((source / name).is_file() for name in provenance["trace_file"])
    new = _ctx(staging, draws=1000, tune=50, chains=4)
    for name in ("model", "prepared", "reporting", "resolved_plan", "spec", "trace"):
        setattr(new, name, getattr(ctx, name))
    new.final_output_dir = str(source)
    monkeypatch.setenv("DSE_LRP_REUSE_TRACE", "1")
    monkeypatch.setattr(pm, "sample", lambda **kw: pytest.fail("reuse must not sample"))
    reused = kfold.run_child_kfold(new, plan, split, rebuild)
    assert reused.complete and reused.elpd == first.elpd
    with pytest.raises(ValueError, match="model_identity|data_digest"):
        kfold.run_child_kfold(new, plan, kfold.KFoldPlan(n_folds=2, random_seed=91), rebuild)
