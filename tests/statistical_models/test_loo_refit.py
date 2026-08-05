# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Guards for the exact-refit (``reloo``) repair of unreliable PSIS-LOO points (#438).

The value of a spliced exact elpd rests entirely on the refit being *the same model*
as the original fit. These tests cover the mechanisms that make that checkable: the
family-owned plan is the single construction path, the wrapper refuses to proceed
when it cannot prove alignment, and a held-out row that would change the parameter
vector is rejected rather than quietly mis-indexed.
"""

from __future__ import annotations

import numpy as np
import pytest

from language_reading_predictors.statistical_models import mechanism as M
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.loo_refit import RefitPlan


def _spec(**extra) -> ModelSpec:
    base = {
        "outcomes": ("W", "L"),
        "adjust_baseline_symbol": "W",
        "adjust_for": ("hs", "hs_missing"),
    }
    base.update(extra)
    return ModelSpec(
        model_id="test-mech-plan",
        kind="mechanism",
        title="test",
        outcome_symbol="W",
        mechanism_symbol="L",
        adjustment=["G", "A", "W_pre"],
        extra=base,
    )


class _FakePrepared:
    """Minimal stand-in carrying only what the holdout guards read."""

    def __init__(self, child_idx):
        self.child_idx = np.asarray(child_idx)
        self.n_obs = len(self.child_idx)


def test_holdout_rejects_a_childs_only_observation():
    """Dropping a child entirely re-indexes ``u_child_raw``, so the refit posterior
    would no longer align with the full model used to evaluate the held-out point.
    ``_subset`` re-indexes silently, so this must be refused explicitly rather than
    left to produce a plausible-but-misaligned number."""
    prepared = _FakePrepared([0, 0, 1, 2, 2])
    safe, why = M.holdout_is_safe(prepared, 2)  # the lone row for child 1
    assert safe is False
    assert "only observation" in why

    safe, why = M.holdout_is_safe(prepared, 0)  # child 0 keeps another row
    assert safe is True
    assert why == ""


def test_holdout_mask_drops_exactly_one_row():
    prepared = _FakePrepared([0, 0, 1, 1])
    mask = M.holdout_mask(prepared, 1)
    assert mask.tolist() == [True, False, True, True]


def test_refit_plan_reuses_the_original_sampler_settings():
    """A refit at different sampler settings yields a held-out density from a
    differently-converged posterior, which does not belong beside the PSIS values it
    is spliced among. The plan therefore reads the fit's own recorded settings."""
    cfg = {
        "sampling": {
            "draws": 6000,
            "tune": 6000,
            "chains": 6,
            "target_accept": 0.999,
            "random_seed": 47,
        }
    }
    plan = RefitPlan.from_config(cfg)
    assert (plan.draws, plan.tune, plan.chains) == (6000, 6000, 6)
    assert plan.target_accept == 0.999
    assert plan.random_seed == 47


def test_refit_plan_refuses_an_incomplete_sampling_block():
    with pytest.raises(ValueError, match="missing"):
        RefitPlan.from_config({"sampling": {"draws": 100}})


def test_plan_factory_kwargs_are_shared_by_reference_with_the_fit():
    """The refit must not be able to differ in likelihood, priors or adjustment set.
    Building on a row subset reuses the very same keyword mapping, so any drift would
    have to be a change to the plan itself rather than to one of two parallel paths."""
    spec = _spec(moderator_symbol="N", include_interaction=False)
    # Resolution touches the loader, so assert on the keyword mapping via a stub plan
    # rather than loading data here (covered live in test_mechanism_plan).
    kwargs = {
        "mechanism_symbol": "L",
        "outcome_symbol": "W",
        "include_interaction": False,
        "moderator_symbol": "N",
    }
    plan = M.MechanismPlan(
        spec=spec,
        prepared=None,  # type: ignore[arg-type]
        factory_kwargs=kwargs,
        confounders=("G", "A"),
        adjust_for=("hs",),
    )
    assert plan.factory_kwargs is kwargs
    assert plan.factory_kwargs["include_interaction"] is False


def test_mechanism_diagnostic_vars_track_the_moderator_and_interaction():
    spec = _spec(moderator_symbol="N", include_interaction=True)
    plan = M.MechanismPlan(
        spec=spec,
        prepared=None,  # type: ignore[arg-type]
        factory_kwargs={},
        confounders=("G", "A"),
        adjust_for=("hs",),
    )
    names = M.mechanism_diagnostic_vars(plan)
    assert "gamma_mod" in names and "gamma_int" in names
    assert "gamma_hs" in names
    assert "gamma_A" in names

    spec_no_int = _spec(moderator_symbol="N", include_interaction=False)
    plan_no_int = M.MechanismPlan(
        spec=spec_no_int,
        prepared=None,  # type: ignore[arg-type]
        factory_kwargs={},
        confounders=("G", "A"),
        adjust_for=(),
    )
    assert "gamma_int" not in M.mechanism_diagnostic_vars(plan_no_int)


def test_frozen_design_reproduces_the_boundary_on_a_subset():
    """A refit must interpret its basis weights against the *fit's* design. The HSGP
    boundary is `max(|X|) * c`, so a row subset moves it unless `c` is adjusted to
    compensate — which is what `hsgp_c_for` computes."""
    from language_reading_predictors.statistical_models.factories import MechanismDesign
    from language_reading_predictors.statistical_models.preprocessing import Standardiser

    design = MechanismDesign(
        mech_scaler=Standardiser(mean=1.0535, sd=1.4335), hsgp_L=3.719753
    )
    subset_x = np.array([-2.4712, 0.0, 1.9])  # support shrunk by dropping a row
    c = design.hsgp_c_for(subset_x)
    realised_L = float(max(abs(subset_x.min()), abs(subset_x.max())) * c)
    assert realised_L == pytest.approx(design.hsgp_L, rel=0, abs=1e-12)


def test_frozen_design_refuses_when_it_cannot_reproduce():
    from language_reading_predictors.statistical_models.factories import MechanismDesign
    from language_reading_predictors.statistical_models.preprocessing import Standardiser

    no_boundary = MechanismDesign(
        mech_scaler=Standardiser(mean=0.0, sd=1.0), hsgp_L=None
    )
    with pytest.raises(ValueError, match="no HSGP boundary"):
        no_boundary.hsgp_c_for(np.array([-1.0, 1.0]))

    with pytest.raises(ValueError, match="degenerate support"):
        MechanismDesign(
            mech_scaler=Standardiser(mean=0.0, sd=1.0), hsgp_L=2.0
        ).hsgp_c_for(np.zeros(3))

    # A model with a moderator needs a moderator scaler; a design captured from a
    # model without one must fail loudly rather than silently re-standardising.
    with pytest.raises(ValueError, match="no moderator scaler"):
        MechanismDesign(
            mech_scaler=Standardiser(mean=0.0, sd=1.0), hsgp_L=1.0
        ).require_moderator_scaler()


class _FakeRV:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeModel:
    """Stand-in exposing only ``observed_RVs``, which is all the guard reads."""

    def __init__(self, *names: str) -> None:
        self.observed_RVs = [_FakeRV(n) for n in names]


def _fake_idata(*names: str):
    import xarray as xr

    from types import SimpleNamespace

    ds = xr.Dataset(
        {n: (("chain", "draw", "obs_id"), np.zeros((2, 3, 4))) for n in names}
    )
    return SimpleNamespace(log_likelihood=ds)


def test_observed_variable_name_follows_the_likelihood_rather_than_assuming_y_post():
    """The observed node's name is a property of the likelihood, not a constant.

    Beta-Binomial mechanism models register ``y_post``; the floor-rule likelihood
    registers ``y_offfloor``. Hard-coding the former made the alignment guards raise
    ``KeyError`` at *comparison* time — after the fits — the moment an off-floor
    mechanism model was registered (#433)."""
    from language_reading_predictors.statistical_models.loo_refit import (
        _observed_variable_name,
    )

    assert (
        _observed_variable_name(_FakeModel("y_post"), _fake_idata("y_post"), "m")
        == "y_post"
    )
    assert (
        _observed_variable_name(
            _FakeModel("y_offfloor"), _fake_idata("y_offfloor"), "m"
        )
        == "y_offfloor"
    )


def test_observed_variable_name_refuses_a_multi_outcome_model():
    """A joint model has one ``obs_id`` axis per outcome, so there is no single row
    for ``reloo`` to hold out. Refuse rather than splice against the first one."""
    from language_reading_predictors.statistical_models.loo_refit import (
        _observed_variable_name,
    )

    with pytest.raises(ValueError, match="multi-outcome"):
        _observed_variable_name(
            _FakeModel("y_W", "y_N"), _fake_idata("y_W", "y_N"), "m"
        )


def test_observed_variable_name_refuses_when_the_trace_names_something_else():
    from language_reading_predictors.statistical_models.loo_refit import (
        _observed_variable_name,
    )

    with pytest.raises(ValueError, match="drifted"):
        _observed_variable_name(_FakeModel("y_offfloor"), _fake_idata("y_post"), "m")

    with pytest.raises(ValueError, match="which variable"):
        _observed_variable_name(
            _FakeModel("y_post"), _fake_idata("y_post", "y_aux"), "m"
        )


def test_held_out_density_is_read_from_the_derived_observed_name(monkeypatch):
    """Closes the loop: the wrapper must *use* the derived name, not just record it."""
    import contextlib

    import xarray as xr

    from language_reading_predictors.statistical_models import loo_refit as _loo_refit
    from language_reading_predictors.statistical_models.loo_refit import (
        MechanismSamplingWrapper,
    )

    log_lik = xr.Dataset(
        {
            "y_offfloor": (("chain", "draw", "obs_id"), np.arange(24.0).reshape(2, 3, 4)),
        }
    )
    # Patch the name bound in ``loo_refit`` — it imports from ``pymc.stats`` directly,
    # so patching the pymc root namespace would succeed but patch nothing it calls.
    monkeypatch.setattr(
        _loo_refit, "compute_log_likelihood", lambda *a, **k: log_lik, raising=True
    )

    wrapper = object.__new__(MechanismSamplingWrapper)
    wrapper.obs_var = "y_offfloor"
    wrapper.full_model = contextlib.nullcontext()

    got = wrapper.log_likelihood__i(2, idata__i=None)
    assert got.name == "y_offfloor"
    np.testing.assert_allclose(got.values, log_lik["y_offfloor"].isel(obs_id=2).values)


def test_as_dataset_unwraps_a_datatree_group():
    """ArviZ 1.x groups are ``DataTree``s whose ``.items()`` yields child *nodes*, not
    variables — iterating one directly and reading ``.values`` raises instead of
    comparing anything. This regression cost a whole comparison run: the log-prior
    guard failed closed on every model, so no repair ran at all."""
    import xarray as xr

    from language_reading_predictors.statistical_models.loo_refit import _as_dataset

    ds = xr.Dataset({"alpha": (("chain", "draw"), np.zeros((2, 3)))})
    assert _as_dataset(ds) is ds  # a plain Dataset passes through

    tree = xr.DataTree(dataset=ds)
    unwrapped = _as_dataset(tree)
    assert "alpha" in unwrapped.data_vars
    assert np.asarray(unwrapped["alpha"].values).shape == (2, 3)
