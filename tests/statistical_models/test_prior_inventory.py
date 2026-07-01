# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Completeness guard for ``priors.priors_table`` (issue #141).

Every actively-used prior across *every* model family must be documented with a
real distribution and a concrete role — never dropped to ``(model prior)`` /
``other`` and never mislabelled by a name prefix. These tests build one
representative model per family on synthetic data and assert that invariant, so a
future factory that introduces a new inline prior (HSGP, LKJ, latent-change,
measurement, mediator-leg, ...) fails here until it is documented in ``priors.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models import priors
from language_reading_predictors.statistical_models.factories import (
    build_aligned_model,
    build_correlated_factor_model,
    build_did_model,
    build_dose_response_model,
    build_gain_factors_model,
    build_itt_model,
    build_joint_model,
    build_lcsm_model,
    build_level_factors_model,
    build_mechanism_model,
    build_mediation_model,
    build_two_mediator_model,
)
from language_reading_predictors.statistical_models.measures import (
    ITT_OUTCOMES,
    MEASURES,
    TAUGHT_BLOCK1_OUTCOMES,
)
from language_reading_predictors.statistical_models.preprocessing import (
    load_and_prepare,
    load_and_prepare_aligned,
    load_wave_panel,
)


def _write_synthetic(tmp_path, n_children: int = 24, seed: int = 7):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_children):
        sid = f"S{i:03d}"
        age_base = int(rng.integers(60, 110))
        g = int(rng.integers(1, 3))
        blocks = int(rng.integers(5, 40))
        for t in (1, 2, 3, 4):
            row = {
                V.SUBJECT_ID: sid, V.TIME: t, V.GROUP: g,
                V.AGE: age_base + 6 * (t - 1),
                V.MUMEDUPOST16: int(rng.integers(0, 8)),
                V.DADEDUPOST16: int(rng.integers(0, 8)),
                V.AGEBOOKS: int(rng.integers(0, 48)),
            }
            if t == 1:
                row[V.BLOCKS] = blocks
            if t in (1, 2, 3):
                row[V.BEHAV] = float(rng.integers(1, 6))
            for s in (*ITT_OUTCOMES, *TAUGHT_BLOCK1_OUTCOMES):
                row[MEASURES[s].column] = int(rng.integers(0, MEASURES[s].n_trials + 1))
            row[V.NONWORD] = int(rng.integers(0, 7))
            row[V.ATTEND] = int(rng.integers(0, 90))
            row[V.ATTEND_CUMUL] = int(rng.integers(0, 200))
            rows.append(row)
    p = tmp_path / "rli.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def _representative_models(tmp_path) -> dict[str, object]:
    """One built model per family (+ the variants that exercise the fallback)."""
    p = _write_synthetic(tmp_path)
    itt = load_and_prepare(path=p, phase_mode="itt")
    allp = load_and_prepare(path=p, phase_mode="all")
    models: dict[str, object] = {}

    models["itt"] = build_itt_model(
        itt, outcome_symbol="R", cross_symbols=(), use_age_linear=True
    ).model
    nprep = load_and_prepare(path=p, phase_mode="itt", outcomes=("N",))
    models["itt_offfloor"] = build_itt_model(
        nprep, outcome_symbol="N", likelihood="bernoulli_offfloor",
        cross_symbols=(), use_age_linear=True, use_own_baseline=False,
    ).model
    # ITT with the age HSGP on — exercises the eta_main / ell suffix mapping.
    models["itt_age_gp"] = build_itt_model(
        itt, outcome_symbol="R", cross_symbols=(), use_age_gp=True
    ).model

    models["joint"] = build_joint_model(
        itt, use_cross_baselines=False, use_age_linear=True
    ).model
    taught = load_and_prepare(path=p, phase_mode="itt", outcomes=("TE", "UE"))
    models["joint_lkj"] = build_joint_model(
        taught, outcomes=("TE", "UE"), use_residual_correlation=True
    ).model

    models["mechanism"] = build_mechanism_model(
        allp, mechanism_symbol="L", outcome_symbol="W", confounder_symbols=("G", "A")
    ).model
    models["mechanism_linear"] = build_mechanism_model(
        allp, mechanism_symbol="L", outcome_symbol="B",
        confounder_symbols=("G", "A"), linear_mechanism=True,
    ).model

    dosep = load_and_prepare(
        path=p, phase_mode="all", outcomes=("W",), covariates=("attend", "attend_cumul")
    )
    models["dose_response"] = build_dose_response_model(
        dosep, outcome_symbol="W", period_varying_dose=True
    ).model

    models["did"] = build_did_model(allp, outcome_symbol="W").model

    models["mediation"] = build_mediation_model(
        itt, mediator_symbol="L", outcome_symbol="W", confounder_symbols=("E", "R")
    )[0].model
    models["mediation_gaussian"] = build_mediation_model(
        itt, outcome_symbol="W", confounder_symbols=("E", "R"),
        mediator_kind="gaussian_composite", route_symbols=("L", "B"),
    )[0].model
    models["mediation_multi"] = build_two_mediator_model(
        itt, outcome_symbol="W", mediator_symbols=("L", "E"), confounder_symbols=("R",)
    )[0].model

    models["gain_factors"] = build_gain_factors_model(allp, outcome_symbol="W").model

    levels = load_and_prepare(path=p, phase_mode="levels")
    levels.covariates["blocks"] = np.linspace(-1, 1, levels.n_obs)
    models["level_factors"] = build_level_factors_model(
        levels, outcome_symbol="W", ability_covariate="blocks"
    ).model

    aligned = load_and_prepare_aligned(path=p)
    models["aligned"] = build_aligned_model(aligned, outcome_symbol="W").model

    cf = load_and_prepare(path=p, phase_mode="itt")
    cf.covariates["blocks"] = np.linspace(-1, 1, cf.n_obs)
    models["corr_factor"] = build_correlated_factor_model(
        cf, outcome_symbol="W",
        domains={"vocabulary": ("R", "E"), "code": ("L", "B"), "grammar": ("F", "T")},
        structural_covariates=("blocks",),
    ).model

    panel = load_wave_panel(path=p, outcomes=("W", "L", "E"))
    models["lcsm"] = build_lcsm_model(panel).model
    return models


@pytest.fixture(scope="module")
def built_models(tmp_path_factory) -> dict[str, object]:
    return _representative_models(tmp_path_factory.mktemp("prior_inv"))


def test_every_family_prior_is_documented(built_models):
    """No free RV, in any family, is left as ``(model prior)`` / ``other`` or
    given a panel that is not a real shared-prior key."""
    problems: list[str] = []
    for fam, model in built_models.items():
        table = priors.priors_table(model)
        for _, row in table.iterrows():
            if row["distribution"] in ("", "(model prior)"):
                problems.append(f"{fam}:{row['parameter']} distribution={row['distribution']!r}")
            if row["role"] == "other":
                problems.append(f"{fam}:{row['parameter']} role='other'")
            if row["panel"] and row["panel"] not in priors.ALL_PRIORS:
                problems.append(f"{fam}:{row['parameter']} panel={row['panel']!r} not in ALL_PRIORS")
    assert not problems, "Undocumented priors:\n" + "\n".join(problems)


def test_used_panels_exist_for_every_family(built_models):
    """``used_prior_keys`` returns only real panel keys (feeds panel pruning)."""
    for fam, model in built_models.items():
        keys = priors.used_prior_keys(model)
        assert all(k in priors.ALL_PRIORS for k in keys), fam
        assert len(keys) == len(set(keys)), f"{fam}: duplicate panel keys"


def test_hsgp_amplitude_lengthscale_are_panelled(built_models):
    """The age-HSGP ITT model surfaces the GP amplitude + lengthscale priors
    (previously dropped to ``(model prior)``)."""
    table = priors.priors_table(built_models["itt_age_gp"])
    by_param = {r["parameter"]: r for _, r in table.iterrows()}
    eta_rows = [r for n, r in by_param.items() if n.endswith("__eta")]
    ell_rows = [r for n, r in by_param.items() if n.endswith("__ell")]
    assert eta_rows and all(r["panel"] == "eta_main" and r["role"] == "gp" for r in eta_rows)
    assert ell_rows and all(r["panel"] == "ell" and r["role"] == "gp" for r in ell_rows)
    assert all(r["distribution"] == "HalfNormal(0.3)" for r in eta_rows)
    assert all(r["distribution"] == "InverseGamma(3, 1)" for r in ell_rows)
    assert "eta_main" in priors.used_prior_keys(built_models["itt_age_gp"])


def test_lcsm_inline_priors_not_mislabelled(built_models):
    """The LCSM ``a_change`` / ``b_self`` carry their real Normal(0, 1.5) /
    Normal(0, 0.5), not gamma_cross's Normal(0, 0.3) (the prefix-fallback bug)."""
    table = priors.priors_table(built_models["lcsm"])
    by_param = {r["parameter"]: r for _, r in table.iterrows()}
    assert by_param["a_change"]["distribution"] == "Normal(0, 1.5)"
    assert by_param["b_self"]["distribution"] == "Normal(0, 0.5)"
    # And none of them are the wrongly-inherited cross-coupling scale.
    assert by_param["a_change"]["distribution"] != "Normal(0, 0.3)"


def test_joint_lkj_residual_documented(built_models):
    table = priors.priors_table(built_models["joint_lkj"])
    by_param = {r["parameter"]: r for _, r in table.iterrows()}
    assert "u_chol" in by_param
    assert by_param["u_chol"]["distribution"] not in ("", "(model prior)")
    assert by_param["u_chol"]["role"] != "other"


def test_dist_from_rv_normalisation():
    import pymc as pm

    with pm.Model():
        rv_tau = pm.Normal("tau", 0.0, 0.5)
        rv_kappa = pm.HalfNormal("kappa", 50.0)
        rv_ig = pm.InverseGamma("ell", alpha=3.0, beta=1.0)
    assert priors._dist_from_rv(rv_tau) == "Normal(0, 0.5)"
    # HalfNormal's printed loc is stripped to match the docstring house style.
    assert priors._dist_from_rv(rv_kappa) == "HalfNormal(50)"
    assert priors._dist_from_rv(rv_ig) == "InverseGamma(3, 1)"
    # A plain name (no built RV) is unreadable -> None, so the caller falls back.
    assert priors._dist_from_rv(None) is None


def test_classify_fallback_routes_cross_coupling_to_gamma_panel():
    # A non-age Normal(0, 0.3) coupling is routed to the gamma_cross panel by its
    # scale (not by a name prefix).
    assert priors._classify_fallback("b_R", "Normal(0, 0.3)") == (
        "association", "gamma_cross"
    )
    # An age coupling (…_A) is a precision covariate, not an association.
    assert priors._classify_fallback("b_A", "Normal(0, 0.3)")[0] == "precision"
    # An LCSM change intercept is a nuisance, not a mis-scaled coupling.
    assert priors._classify_fallback("a_change", "Normal(0, 1.5)")[0] == "nuisance"
    # Non-centred offsets are nuisances with no panel.
    assert priors._classify_fallback("u_child_raw", "Normal(0, 1)") == ("nuisance", "")
    assert priors._classify_fallback("zproc_W", "Normal(0, 1)") == ("nuisance", "")
