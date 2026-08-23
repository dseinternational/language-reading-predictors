# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Fitted-versus-declared contract tests for the mediation families (#585).

The pre-#585 models split the declared adjustment set between the two legs: the
mediator law never received the outcome baseline and the outcome law never
received the mediator baseline, while
``tests/test_lagged_dag_adjustment_sets.py`` certified only their union. Nothing
compared the *declared* set with the two *design matrices*, so the mismatch was
invisible. These tests close that gap: every model's built graph is inspected
directly, and the complete-case rule is checked against the terms the legs
actually use.
"""

from __future__ import annotations

import importlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from language_reading_predictors.statistical_models import factories as F
from language_reading_predictors.statistical_models.mediation_settings import (
    MediationModelSettings,
    resolve_mediation_multi_run_plan,
    resolve_mediation_run_plan,
)
from language_reading_predictors.statistical_models.pipelines.mediation import (
    _prepare_mediation_data,
)
from language_reading_predictors.statistical_models.preprocessing import (
    load_and_prepare,
)

_MODEL_ROOT = (
    Path(__file__).parents[2] / "src/language_reading_predictors/statistical_models"
)


def _registered():
    for path in sorted(_MODEL_ROOT.glob("lrp_rli_med_*.py")):
        module = importlib.import_module(
            f"language_reading_predictors.statistical_models.{path.stem}"
        )
        yield module.SPEC


def _resolve(spec):
    if spec.kind == "mediation_multi":
        return resolve_mediation_multi_run_plan(spec)
    return resolve_mediation_run_plan(spec)


def _build(spec):
    """Resolve, prepare and build one registered mediation model."""
    plan = _resolve(spec)
    if spec.kind == "mediation_multi":
        prepared = load_and_prepare(**plan.prepare_kwargs())
    elif plan.entrypoint == "period_stacked":
        prepared = load_and_prepare(**plan.period_prepare_kwargs())
    else:
        prepared, _ = _prepare_mediation_data(plan)
    active = tuple(
        symbol
        for symbol in plan.declared_confounders
        if symbol in prepared.covariates or symbol in prepared.pre_logit
    )
    plan = plan.with_effective_confounders(active)
    if spec.kind == "mediation_multi":
        built, _ = F.build_two_mediator_model(prepared, **plan.factory_kwargs())
    elif plan.entrypoint == "period_stacked":
        built, _ = F.build_period_stacked_mediation_model(
            prepared, **plan.period_factory_kwargs()
        )
    else:
        built, _ = F.build_mediation_model(prepared, **plan.factory_kwargs())
    return plan, prepared, built


def _coefficients(built) -> set[str]:
    return {rv.name for rv in built.model.free_RVs if rv.ndim == 0}


ALL_SPECS = list(_registered())
SPEC_IDS = [spec.model_id for spec in ALL_SPECS]


@pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
def test_every_declared_term_is_fitted(spec):
    """No declared term may be silently absent from the built graph.

    MED-060 declared expressive and receptive vocabulary but requested a load set
    that excluded them, so the pipeline's post-preparation filter removed both
    before model construction and ``dropped_confounders`` — which compared only
    raw covariates — never recorded the loss (#585 finding 3).
    """
    plan, prepared, built = _build(spec)
    coefficients = _coefficients(built)
    # A constant missing-indicator gets no coefficient, so the loader drops it —
    # the one explicitly permitted removal. Everything else must reach the graph.
    permitted = set(getattr(prepared, "dropped_covariates", ()) or ())
    missing = [
        symbol
        for symbol in plan.declared_confounders
        if symbol not in prepared.covariates
        and symbol not in prepared.pre_logit
        and symbol not in permitted
    ]
    assert not missing, f"declared confounder(s) never loaded: {missing}"
    for term in plan.outcome_cross_baselines:
        assert term.coefficient in coefficients
    cross = plan.mediator_cross_baselines
    legs = cross.values() if isinstance(cross, dict) else [cross]
    for terms in legs:
        for term in terms:
            assert term.coefficient in coefficients


@pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
def test_common_baseline_vector_reaches_both_legs(spec):
    """Each leg conditions on every member of the common pre-exposure vector.

    A member is present either as that leg's own-baseline term, as a legacy
    confounder coefficient, or as one of the ``*_base_*`` terms #585 added. This
    is the check that would have failed before the fix: the mediator law lacked
    the outcome baseline and the outcome law lacked the mediator baseline.
    """
    plan, _, built = _build(spec)
    coefficients = _coefficients(built)

    def covered(symbol: str, prefix: str, own: tuple[str, ...]) -> bool:
        if symbol in own:
            return True  # own-baseline term, legacy name
        return any(
            name in coefficients
            for name in (
                f"{prefix}_{symbol}",
                f"{prefix}_base_{symbol}",
                f"{prefix}_base_{symbol}_offfloor",
                f"{prefix}_conf_{symbol}",
            )
        )

    if spec.kind == "mediation_multi":
        for mediator in plan.mediators:
            for symbol in plan.common_baselines:
                assert covered(symbol, f"a{mediator}", (mediator,)), (
                    f"{symbol} missing from the {mediator} leg"
                )
        for symbol in plan.common_baselines:
            assert covered(symbol, "b", (plan.outcome_symbol,)), (
                f"{symbol} missing from the outcome leg"
            )
        return

    mediator_own = (
        plan.route_symbols
        if plan.mediator_kind == "gaussian_composite"
        else (plan.mediator_symbol,)
    )
    for symbol in plan.common_baselines:
        assert covered(symbol, "a", mediator_own), (
            f"{symbol} missing from the mediator leg"
        )
    # A composite mediator enters the outcome leg as one composite baseline term
    # rather than as its route symbols separately.
    if plan.mediator_kind == "gaussian_composite":
        assert "b_base_M" in _coefficients(built)
    else:
        for symbol in plan.common_baselines:
            assert covered(symbol, "b", (plan.outcome_symbol,)), (
                f"{symbol} missing from the outcome leg"
            )


@pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
def test_complete_case_rule_matches_the_modelled_baselines(spec):
    """Only a baseline some leg models may restrict the fitted sample (#585 f.4).

    The loader's default requires the baseline of every *loaded* outcome, so
    MED-060/086/186 excluded three children for a nonword baseline their
    likelihood never used. The rule is now resolved from the legs' own terms.
    """
    plan, _, _ = _build(spec)
    assert set(plan.pre_required) == set(plan.common_baselines)


def test_unused_loaded_baseline_cannot_change_row_membership():
    """Blanking an unmodelled measure's baseline must not drop a child.

    MED-059 loads the default ITT outcome set but models only W, L, E and R at
    baseline. Wiping another measure's t1 column used to remove those children
    from the fit through the loader's default ``pre_required``.
    """
    import pandas as pd

    from language_reading_predictors.statistical_models.measures import MEASURES
    from language_reading_predictors.statistical_models.preprocessing import (
        _default_data_path,
    )

    spec = next(s for s in ALL_SPECS if s.model_id == "lrp-rli-med-059")
    plan = _resolve(spec)
    unused = next(
        symbol
        for symbol in ("B", "TR")
        if symbol in MEASURES and symbol not in plan.pre_required
    )
    baseline = load_and_prepare(**plan.prepare_kwargs())

    frame = pd.read_csv(_default_data_path())
    frame.loc[:, MEASURES[unused].column] = np.nan
    perturbed_path = Path(
        pytest.importorskip("tempfile").mkdtemp()
    ) / "rli_data_long.csv"
    frame.to_csv(perturbed_path, index=False)
    perturbed = load_and_prepare(path=perturbed_path, **plan.prepare_kwargs())

    assert perturbed.n_obs == baseline.n_obs


def test_med_060_fits_its_declared_vocabulary_confounders():
    """Regression for the #585 finding-3 witness itself."""
    spec = next(s for s in ALL_SPECS if s.model_id == "lrp-rli-med-060")
    _, _, built = _build(spec)
    coefficients = _coefficients(built)
    for leg in ("aL", "aN", "b"):
        assert f"{leg}_E" in coefficients
        assert f"{leg}_R" in coefficients


def test_unloaded_declared_measure_fails_before_any_io():
    """The resolver refuses a load set that omits a declared measure confounder."""
    spec = next(s for s in ALL_SPECS if s.model_id == "lrp-rli-med-060")
    broken = replace(
        spec,
        model_settings=replace(spec.model_settings, outcomes=("W", "L", "N")),
    )
    with pytest.raises(ValueError, match="declared but not loaded"):
        resolve_mediation_multi_run_plan(broken)


def test_composite_mediator_rejects_an_offfloor_outcome():
    """The composite factory has no off-floor leg, so the pair must not resolve.

    It used to resolve, silently fit a graded outcome, and then ask the PPC
    writer for a ``y_offfloor`` node that was never built (#585).
    """
    spec = next(s for s in ALL_SPECS if s.model_id == "lrp-rli-med-062")
    broken = replace(
        spec,
        model_settings=replace(
            spec.model_settings, outcome_kind="bernoulli_offfloor"
        ),
    )
    with pytest.raises(ValueError, match="no off-floor outcome leg"):
        resolve_mediation_run_plan(broken)


def test_offfloor_legs_carry_a_binary_baseline_contrast():
    """An off-floor leg models its baseline rather than dropping it (#585 f.4)."""
    single = next(s for s in ALL_SPECS if s.model_id == "lrp-rli-med-086")
    _, _, built = _build(single)
    coefficients = _coefficients(built)
    assert "b_own_offfloor" in coefficients  # binary contrast
    assert "b_W" not in coefficients  # the graded logit stays out

    multi = next(s for s in ALL_SPECS if s.model_id == "lrp-rli-med-060")
    _, _, built_multi = _build(multi)
    assert "aN_own_offfloor" in _coefficients(built_multi)


def test_period_stacked_primary_is_the_supported_window():
    """MED-092's exposure has no untreated rows after the crossover (#585 f.5)."""
    spec = next(s for s in ALL_SPECS if s.model_id == "lrp-rli-med-092")
    plan = resolve_mediation_run_plan(spec)
    prepared = load_and_prepare(**plan.period_prepare_kwargs())
    trt = ((prepared.G == 1) | (prepared.phase >= 1)).astype(int)
    by_period = {
        int(ph): (int(trt[prepared.phase == ph].sum()),
                  int((1 - trt)[prepared.phase == ph].sum()))
        for ph in sorted(set(prepared.phase.tolist()))
    }
    supported = [ph for ph, (t, u) in by_period.items() if t and u]
    assert supported == [0], (
        "only the first period has both arms, so it is the only window whose "
        f"untreated counterfactual is supported: {by_period}"
    )


def test_named_confounder_calibration_recovers_a_known_linear_bias():
    """Simulation validation of the ``delta = |b(U->M) * b(U->Y)|`` form (#585 f.6).

    The product is the omitted-variable bias only because both slopes are put on
    one-standard-deviation scales, which makes ``Var(M) = 1``. Generate data with
    a known unmeasured common cause and confirm the inflation of the fitted
    mediator coefficient matches the product to within Monte-Carlo error.
    """
    rng = np.random.default_rng(585)
    n = 200_000
    beta_u_m, beta_u_y, true_b = 0.6, 0.8, 0.5

    u = rng.normal(size=n)
    m_raw = beta_u_m * u + rng.normal(size=n)
    m = (m_raw - m_raw.mean()) / m_raw.std()
    y = true_b * m + beta_u_y * u + rng.normal(size=n) * 0.1

    fitted_b = float(np.polyfit(m, y, 1)[0])
    actual_bias = fitted_b - true_b
    slope_u_to_m = float(np.polyfit(u, m, 1)[0])

    # With the PARTIAL U -> Y slope (U's coefficient given M) the bare product is
    # the exact omitted-variable bias, because standardising M makes Var(M) = 1.
    design = np.column_stack([np.ones(n), m, u])
    partial_u_to_y = float(np.linalg.lstsq(design, y, rcond=None)[0][2])
    assert abs(slope_u_to_m * partial_u_to_y) == pytest.approx(actual_bias, rel=0.01)

    # The module deliberately supplies the MARGINAL U -> Y slope instead, which
    # also absorbs the genuine U -> M -> Y path. That is conservative by
    # construction: it can only overstate the bias, never understate it.
    marginal_u_to_y = float(np.polyfit(u, y, 1)[0])
    assert abs(slope_u_to_m * marginal_u_to_y) > actual_bias
