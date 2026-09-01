# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""New-child prediction target and its two estimators (#626).

The joint families declared ``loo_unit="child"`` long before anything checked that a
child-aggregated PSIS-LOO actually answered a new-child question. Where a model carries
a child-level latent it does not: the importance weights reweight a posterior in which
the held-out child's own random effect is still fitted to its own data.

These tests hold the three things that turns into: every joint family declares a target;
the latent declaration fails closed rather than silently reverting to the conditional
answer; and an estimate whose Pareto-k or whose own integration error is unacceptable is
withheld rather than published.
"""

from __future__ import annotations

import math
import types

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

from language_reading_predictors.statistical_models.family_registry import (
    descriptor_for,
)
from language_reading_predictors.statistical_models.new_child_kfold import (
    KFoldPlan,
    _fold_assignment,
    _transplant,
)
from language_reading_predictors.statistical_models.new_child_predictive import (
    PREDICTION_TARGET_NEW_CHILD,
    NewChildEvidenceUnavailable,
    NewChildPlan,
    NewChildValidation,
    child_row_maps,
    verify_child_latents,
)
from language_reading_predictors.statistical_models.registry import discover_models

#: Families whose fits carry a child unit and therefore must declare a target.
JOINT_KINDS = ("joint", "joint_mechanism", "historical_joint")


def _plan_for(model_id: str):
    spec = discover_models()[model_id].load().SPEC
    return descriptor_for(spec.kind).resolver()(spec)


def _registered(kind: str) -> list[str]:
    out = []
    for model_id, lazy in discover_models().items():
        spec = getattr(lazy.load(), "SPEC", None)
        if spec is not None and spec.kind == kind:
            out.append(model_id)
    return out


# --------------------------------------------------------------------------------
# The declaration
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", JOINT_KINDS)
def test_every_joint_family_model_declares_a_prediction_target(kind):
    """No registered joint fit may leave its out-of-sample target implicit.

    ``historical_joint`` used to record the target as
    ``undeclared_prediction_target_not_implemented``; that is the state this forbids
    coming back, for any of the three families.
    """
    model_ids = _registered(kind)
    assert model_ids, f"no registered {kind} models to check"
    for model_id in model_ids:
        plan = _plan_for(model_id)
        assert plan.prediction_target == PREDICTION_TARGET_NEW_CHILD, model_id
        declared = plan.new_child_plan()
        assert declared.child_dims, f"{model_id} declares no child dimension"
        assert declared.observed_nodes, f"{model_id} declares no likelihood node"


def test_an_unimplemented_target_is_refused_rather_than_relabelled():
    """Declaring the other target must fail, not quietly reuse the new-child engine."""
    with pytest.raises(ValueError, match="not implemented"):
        NewChildPlan(
            prediction_target="new_occasion_known_child", child_dims=("obs_id",)
        )


def test_an_unknown_target_is_refused():
    with pytest.raises(ValueError, match="prediction_target must be one of"):
        NewChildPlan(prediction_target="whatever", child_dims=("obs_id",))


def test_a_plan_must_name_a_child_dimension():
    with pytest.raises(ValueError, match="child_dims"):
        NewChildPlan()


def test_the_recipe_records_the_target():
    """``model_recipe.md`` is where a reader looks; the target has to be in it."""
    plan = _plan_for("lrp-rli-itt-012")
    recipe = plan.recipe_markdown(title="x")
    assert "prediction target: new_child" in recipe.lower()


# --------------------------------------------------------------------------------
# Failing closed on the latent declaration
# --------------------------------------------------------------------------------


def _toy_model(with_child_latent: bool) -> pm.Model:
    coords = {"obs_id": np.arange(4), "outcome": ["A"]}
    with pm.Model(coords=coords) as model:
        alpha = pm.Normal("alpha", 0.0, 1.0, dims="outcome")
        eta = alpha[None, :] * np.ones((4, 1))
        if with_child_latent:
            u = pm.Normal("u_z", 0.0, 1.0, dims=("obs_id", "outcome"))
            eta = eta + u
        pm.Binomial(
            "y_post",
            n=10,
            p=pm.math.sigmoid(eta).ravel(),
            observed=np.array([3, 4, 5, 6]),
            dims="obs_id",
        )
    return model


def test_an_undeclared_child_latent_fails_the_run():
    """The defect #626 is about: a child effect left at its fitted value.

    Silence here is what turns a new-child label into a conditional number, so the
    check refuses rather than warning.
    """
    model = _toy_model(with_child_latent=True)
    plan = NewChildPlan(child_dims=("obs_id",), latent_vars=())
    with pytest.raises(ValueError, match="not declared child latents"):
        verify_child_latents(model, plan)


def test_a_declared_latent_that_is_not_a_free_variable_fails():
    model = _toy_model(with_child_latent=False)
    plan = NewChildPlan(child_dims=("obs_id",), latent_vars=("u_z",))
    with pytest.raises(ValueError, match="not free random variables"):
        verify_child_latents(model, plan)


def test_a_model_with_no_child_latent_declares_none_and_passes():
    """The identity case: no child latent means conditional and marginal coincide."""
    model = _toy_model(with_child_latent=False)
    plan = NewChildPlan(child_dims=("obs_id",), latent_vars=())
    assert verify_child_latents(model, plan) == ()


def test_the_joint_family_declares_its_residual_block_exactly_when_it_is_on():
    """``u_z`` exists only with the LKJ block; the declaration has to track it."""
    off = _plan_for("lrp-rli-itt-012").new_child_plan()
    on = _plan_for("lrp-rli-itt-215").new_child_plan()
    assert off.latent_vars == ()
    assert on.latent_vars == ("u_z",)


def test_the_historical_joint_family_declares_both_latent_levels():
    """The within-child block adds a second latent; a design carrying it redraws both."""
    stable = _plan_for("lrp-rlm-jc-001").new_child_plan()
    within = _plan_for("lrp-rlm-jc-002").new_child_plan()
    assert stable.latent_vars == ("z_subject",)
    assert within.latent_vars == ("z_subject", "z_within")


# --------------------------------------------------------------------------------
# Withholding an estimate that cannot be published
# --------------------------------------------------------------------------------


def _validation(*, pareto_k, mc_error, elpd_se=10.0, n_children=50) -> NewChildValidation:
    return NewChildValidation(
        plan=NewChildPlan(child_dims=("obs_id",), latent_vars=("u_z",)),
        n_children=n_children,
        posterior_draws_used=1000,
        elpd=-100.0,
        elpd_se=elpd_se,
        p_loo=5.0,
        pointwise_elpd=np.zeros(n_children),
        pareto_k=np.asarray(pareto_k, dtype=float),
        good_k=0.7,
        latents_redrawn=("u_z",),
        observed_nodes=("y_post",),
        latent_mc_error=mc_error,
    )


def test_a_clean_estimate_is_publishable():
    assert _validation(pareto_k=[0.1, 0.5, 0.69], mc_error=0.001).reliable


def test_an_unacceptable_pareto_k_withholds_the_estimate():
    """#626: naive PSIS is not published where Pareto-k is unacceptable."""
    result = _validation(pareto_k=[0.1, 0.9], mc_error=0.001)
    assert not result.reliable
    assert result.n_unreliable == 1


def test_a_rough_latent_integral_withholds_the_estimate_even_with_clean_k():
    """The second failure mode, which Pareto-k cannot see.

    The historical joint-growth probe moved its ELPD by hundreds of nats between 64 and
    256 latent draws while every k value stayed finite; an estimate whose numerical
    error rivals its own standard error is not measuring the model.
    """
    result = _validation(pareto_k=[0.1, 0.2], mc_error=1.0, elpd_se=5.0, n_children=50)
    assert result.n_unreliable == 0
    assert not result.integration_reliable
    assert not result.reliable


def test_no_latent_means_no_integration_error_to_gate_on():
    result = NewChildValidation(
        plan=NewChildPlan(child_dims=("obs_id",), latent_vars=()),
        n_children=10,
        posterior_draws_used=1000,
        elpd=-10.0,
        elpd_se=2.0,
        p_loo=1.0,
        pointwise_elpd=np.zeros(10),
        pareto_k=np.full(10, 0.2),
        good_k=0.7,
        latents_redrawn=(),
        observed_nodes=("y_post",),
        latent_mc_error=0.0,
    )
    assert result.integration_reliable and result.reliable


def test_a_non_finite_integration_error_withholds_the_estimate():
    assert not _validation(pareto_k=[0.1], mc_error=math.nan).reliable


def test_the_summary_row_publishes_both_verdicts():
    row = _validation(pareto_k=[0.1, 0.9], mc_error=0.001).summary_row()
    assert row["prediction_target"] == PREDICTION_TARGET_NEW_CHILD
    assert row["holdout_unit"] == "child"
    assert row["reliable"] is False
    assert row["integration_reliable"] is True


# --------------------------------------------------------------------------------
# The child map
# --------------------------------------------------------------------------------


def _trace_with(constant: dict | None, observed: dict) -> xr.DataTree:
    tree = xr.DataTree()
    tree["observed_data"] = xr.DataTree(
        xr.Dataset({k: ("row", np.asarray(v)) for k, v in observed.items()})
    )
    if constant is not None:
        tree["constant_data"] = xr.DataTree(
            xr.Dataset({k: ("cell", np.asarray(v)) for k, v in constant.items()})
        )
    return tree


def test_the_child_map_reuses_the_persisted_joint_cell_map():
    """The new-child unit must be the unit the stored PSIS-LOO already uses."""
    ctx = types.SimpleNamespace(
        trace=_trace_with({"y_post_cell_row": [0, 0, 1, 1]}, {"y_post": [1, 2, 3, 4]}),
        prepared=None,
    )
    maps, n_children = child_row_maps(ctx, ("y_post",))
    assert n_children == 2
    np.testing.assert_array_equal(maps["y_post"], [0, 0, 1, 1])


def test_a_missing_child_map_is_expected_absence_not_a_crash():
    ctx = types.SimpleNamespace(
        trace=_trace_with(None, {"y_post": [1, 2]}), prepared=None
    )
    with pytest.raises(NewChildEvidenceUnavailable):
        child_row_maps(ctx, ("y_post",))


def test_a_map_that_aligns_with_no_node_is_refused():
    ctx = types.SimpleNamespace(
        trace=_trace_with({"loo_child_idx": [0, 1]}, {"y_post": [1, 2, 3, 4]}),
        prepared=None,
    )
    with pytest.raises(NewChildEvidenceUnavailable, match="no child map aligns"):
        child_row_maps(ctx, ("y_post",))


# --------------------------------------------------------------------------------
# K-fold
# --------------------------------------------------------------------------------


def test_folds_are_balanced_within_group():
    """A fold that held out most of one cohort would not be comparable to the others."""
    groups = np.array([0] * 30 + [1] * 20 + [2] * 10)
    folds = _fold_assignment(groups.size, groups, KFoldPlan(n_folds=5))
    for value in np.unique(groups):
        counts = np.bincount(folds[groups == value], minlength=5)
        assert counts.max() - counts.min() <= 1


def test_fold_assignment_is_deterministic():
    plan = KFoldPlan(n_folds=4)
    first = _fold_assignment(40, None, plan)
    second = _fold_assignment(40, None, plan)
    np.testing.assert_array_equal(first, second)


def test_every_child_lands_in_exactly_one_fold():
    folds = _fold_assignment(37, None, KFoldPlan(n_folds=5))
    assert folds.shape == (37,)
    assert set(folds.tolist()) == {0, 1, 2, 3, 4}


def test_the_transplant_carries_free_variables_and_leaves_deterministics_alone():
    """A Deterministic is the training cohort's shape, not a global to import.

    Transplanting ``subject_offset`` — one row per *training* child — into a model with
    a row per child is a shape error dressed as data, and it is what refused every fold
    of the first K-fold run.
    """
    model = _toy_model(with_child_latent=True)
    full = xr.Dataset(
        {
            "alpha": (("chain", "draw", "outcome"), np.zeros((1, 2, 1))),
            "u_z": (("chain", "draw", "obs_id", "outcome"), np.zeros((1, 2, 4, 1))),
            "subject_offset": (("chain", "draw", "obs_id"), np.zeros((1, 2, 4))),
        }
    )
    fold = xr.Dataset(
        {
            "alpha": (("chain", "draw", "outcome"), np.ones((1, 2, 1))),
            "u_z": (("chain", "draw", "obs_id", "outcome"), np.ones((1, 2, 3, 1))),
            "subject_offset": (("chain", "draw", "obs_id"), np.ones((1, 2, 3))),
        }
    )
    out = _transplant(model, full, fold, ("u_z",))
    assert set(out.data_vars) == {"alpha"}


def test_the_transplant_refuses_a_reshaped_global():
    model = _toy_model(with_child_latent=False)
    full = xr.Dataset({"alpha": (("chain", "draw", "outcome"), np.zeros((1, 2, 1)))})
    fold = xr.Dataset({"alpha": (("chain", "draw", "outcome"), np.zeros((1, 2, 3)))})
    with pytest.raises(ValueError, match="reshaped"):
        _transplant(model, full, fold, ())


def test_the_transplant_refuses_a_missing_global():
    model = _toy_model(with_child_latent=False)
    full = xr.Dataset({"alpha": (("chain", "draw", "outcome"), np.zeros((1, 2, 1)))})
    with pytest.raises(ValueError, match="missing the free variable"):
        _transplant(model, full, xr.Dataset(), ())


def test_a_partial_kfold_is_not_reported_as_the_declared_estimate():
    """Which refits happened to work is a selection, not a smaller sample."""
    from language_reading_predictors.statistical_models.new_child_kfold import (
        KFoldValidation,
    )

    result = KFoldValidation(
        plan=NewChildPlan(child_dims=("subject",), latent_vars=("z_subject",)),
        kfold=KFoldPlan(n_folds=5),
        n_children=71,
        n_scored=57,
        elpd=-100.0,
        elpd_se=10.0,
        pointwise_elpd=np.zeros(71),
        fold_of_child=np.zeros(71, dtype=int),
        fold_converged={0: True, 1: True, 2: True, 3: True},
        latents_redrawn=("z_subject",),
        observed_nodes=("score_basread",),
        refused_folds={4: "rebuild failed"},
    )
    assert not result.complete
    assert result.summary_row()["n_folds_refused"] == 1


def test_a_complete_kfold_reports_as_complete():
    from language_reading_predictors.statistical_models.new_child_kfold import (
        KFoldValidation,
    )

    result = KFoldValidation(
        plan=NewChildPlan(child_dims=("subject",), latent_vars=("z_subject",)),
        kfold=KFoldPlan(n_folds=2),
        n_children=4,
        n_scored=4,
        elpd=-10.0,
        elpd_se=1.0,
        pointwise_elpd=np.zeros(4),
        fold_of_child=np.array([0, 0, 1, 1]),
        fold_converged={0: True, 1: True},
        latents_redrawn=("z_subject",),
        observed_nodes=("score_basread",),
    )
    assert result.complete


# --------------------------------------------------------------------------------
# Panel subsetting
# --------------------------------------------------------------------------------


def test_subsetting_a_panel_keeps_child_order_and_narrows_every_array():
    """Fold indices are positions in the full model's ordering; order has to survive."""
    from language_reading_predictors.statistical_models.new_child_kfold import (
        subset_panel_children,
    )

    dataset = types.SimpleNamespace(subject_col="sid", wave_col="w", group_col="g")

    class _Panel:
        def __init__(self):
            self.dataset = dataset
            self.subject_ids = ["a", "b", "c", "d"]
            self.n_subjects = 4
            self.long = pd.DataFrame(
                {"sid": ["a", "a", "b", "c", "d"], "w": [1, 2, 1, 1, 1]}
            )
            self.counts = {"m": np.arange(8).reshape(4, 2)}
            self.obs_mask = {"m": np.ones((4, 2), dtype=bool)}

        def __eq__(self, other):  # pragma: no cover - dataclasses.replace needs none
            return self is other

    panel = _Panel()
    # ``dataclasses.replace`` needs a real dataclass; use the family's own container.
    from language_reading_predictors.statistical_models.preprocessing import (
        LongitudinalPanel,
    )

    real = LongitudinalPanel(
        dataset=dataset,
        measures=("m",),
        long=panel.long,
        subject_ids=panel.subject_ids,
        group_codes=[1],
        group_labels=["g"],
        waves=(1, 2),
        counts=panel.counts,
        obs_mask=panel.obs_mask,
        n_trials={"m": 10},
        n_subjects=4,
        n_waves=2,
        dropped_subjects=0,
        group_label_col="g_label",
    )
    narrowed = subset_panel_children(real, [0, 2])
    assert narrowed.subject_ids == ["a", "c"]
    assert narrowed.n_subjects == 2
    assert set(narrowed.long["sid"]) == {"a", "c"}
    np.testing.assert_array_equal(narrowed.counts["m"], real.counts["m"][[0, 2]])
    assert narrowed.n_trials == real.n_trials


def test_subsetting_refuses_an_index_outside_the_panel():
    from language_reading_predictors.statistical_models.new_child_kfold import (
        subset_panel_children,
    )
    from language_reading_predictors.statistical_models.preprocessing import (
        LongitudinalPanel,
    )

    dataset = types.SimpleNamespace(subject_col="sid", wave_col="w", group_col="g")
    real = LongitudinalPanel(
        dataset=dataset,
        measures=("m",),
        long=pd.DataFrame({"sid": ["a"], "w": [1]}),
        subject_ids=["a"],
        group_codes=[1],
        group_labels=["g"],
        waves=(1,),
        counts={"m": np.zeros((1, 1))},
        obs_mask={"m": np.ones((1, 1), dtype=bool)},
        n_trials={"m": 10},
        n_subjects=1,
        n_waves=1,
        dropped_subjects=0,
        group_label_col="g_label",
    )
    with pytest.raises(ValueError, match="outside the panel"):
        subset_panel_children(real, [0, 5])


def test_every_declaration_survives_verification_against_its_built_model():
    """The declarations are checked against real models, not only against themselves.

    Building each registered joint fit and running the guard is the only thing that
    catches a declaration that has drifted from the factory — a renamed latent, or a
    child-indexed variable added on one branch of a design switch. It is the check that
    would have caught the defect #626 is about, so it runs over every registered fit
    rather than over a representative one.
    """
    from language_reading_predictors.statistical_models import (
        datasets as _datasets,
        factories as _factories,
    )
    from language_reading_predictors.statistical_models.preprocessing import (
        load_and_prepare,
        load_longitudinal_panel,
    )

    checked = 0
    for kind in JOINT_KINDS:
        for model_id in _registered(kind):
            plan = _plan_for(model_id)
            if kind == "historical_joint":
                dataset, measures = _datasets.resolve_dataset(plan.study_id)
                prepared = load_longitudinal_panel(
                    dataset,
                    [measures[m] for m in plan.measures],
                    **plan.prepare_kwargs(),
                )
                built = _factories.build_rlm_joint_growth_model(
                    prepared, **plan.factory_kwargs()
                )
            elif kind == "joint":
                prepared = load_and_prepare(**plan.prepare_kwargs())
                built = _factories.build_joint_model(prepared, **plan.factory_kwargs())
            else:
                # The joint-mechanism levels design needs a single-wave subset its
                # pipeline assembles; verifying the transition design is enough to
                # hold the declaration against a real model here.
                if plan.design == "levels":
                    continue
                prepared = load_and_prepare(**plan.prepare_kwargs())
                built = _factories.build_joint_mechanism_model(
                    prepared, **plan.factory_kwargs()
                )
            declared = plan.new_child_plan()
            assert verify_child_latents(built.model, declared) == declared.latent_vars
            checked += 1
    assert checked, "no joint model was actually built and verified"


def test_calibration_groups_never_pool_measures_with_different_denominators():
    """One PIT group per measure, whatever the family calls its outcome tuple.

    ``joint`` names it ``outcomes`` and ``joint_mechanism`` ``outcome_symbols``.
    Reading only the first produced a single group for the joint-mechanism node, which
    would have summed a 79-item word-reading count with a 6-item nonword one into one
    test quantity — the incompatible-denominator pooling every other predictive check
    in this repo refuses.
    """
    from language_reading_predictors.statistical_models.new_child_predictive import (
        _pit_groups,
    )

    trace = xr.DataTree()
    trace["constant_data"] = xr.DataTree(
        xr.Dataset(
            {
                "y_post_cell_row": ("cell", np.array([0, 0, 1, 1])),
                "y_post_cell_outcome": ("cell", np.array([0, 1, 0, 1])),
            }
        )
    )
    maps = {"y_post": np.array([0, 0, 1, 1])}

    for attribute in ("outcomes", "outcome_symbols"):
        ctx = types.SimpleNamespace(
            trace=trace,
            resolved_plan=types.SimpleNamespace(**{attribute: ("W", "N")}),
        )
        groups = _pit_groups(ctx, ("y_post",), maps)
        assert [label for label, _node, _mask in groups] == ["W", "N"], attribute


def test_a_node_with_no_outcome_map_stays_one_group():
    """One likelihood node per measure needs no split — its name is the label."""
    from language_reading_predictors.statistical_models.new_child_predictive import (
        _pit_groups,
    )

    trace = xr.DataTree()
    ctx = types.SimpleNamespace(trace=trace, resolved_plan=None)
    groups = _pit_groups(ctx, ("score_basread",), {"score_basread": np.array([0, 1])})
    assert [label for label, _node, _mask in groups] == ["basread"]


def test_the_half_split_error_normalises_each_half_by_its_own_count():
    """An odd number of re-draws splits unevenly; each half divides by what it holds.

    Deriving the two counts from the total got them the wrong way round for an odd
    ``n_latent_draws`` — the even-indexed half holds the extra draw, not the odd one —
    which biased the diagnostic that decides whether an ELPD may be published.
    """
    from language_reading_predictors.statistical_models.new_child_predictive import (
        _half_split_error,
    )

    # Two halves holding the same *mean* per-draw likelihood must disagree by zero,
    # whatever their sizes: log(3 * exp(x)) - log(3) == log(2 * exp(x)) - log(2).
    value = -1.5
    first = np.full((1, 1, 4), value + math.log(3))  # three re-draws
    second = np.full((1, 1, 4), value + math.log(2))  # two re-draws
    assert _half_split_error([first, second], [3, 2], ("z",)) == pytest.approx(0.0)

    # Swapping the counts is the defect, and it must show up as a non-zero error.
    assert _half_split_error([first, second], [2, 3], ("z",)) > 0.3


def test_the_half_split_error_is_zero_without_a_latent_to_integrate():
    from language_reading_predictors.statistical_models.new_child_predictive import (
        _half_split_error,
    )

    assert _half_split_error([None, None], [0, 0], ()) == 0.0


def test_the_half_split_error_is_not_a_number_when_a_half_is_empty():
    from language_reading_predictors.statistical_models.new_child_predictive import (
        _half_split_error,
    )

    value = _half_split_error([np.zeros((1, 1, 2)), None], [1, 0], ("z",))
    assert math.isnan(value)
