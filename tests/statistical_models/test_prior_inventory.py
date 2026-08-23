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

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models import priors
from language_reading_predictors.statistical_models.datasets import RLM_MEASURES
from language_reading_predictors.statistical_models.factories import (
    build_adjusted_model,
    build_aligned_model,
    build_correlated_factor_model,
    build_did_model,
    build_dose_response_model,
    build_gain_factors_model,
    build_growth_model,
    build_historical_growth_model,
    build_itt_model,
    build_joint_mechanism_model,
    build_joint_model,
    build_lcsm_model,
    build_level_factors_model,
    build_longitudinal_corr_factor_model,
    build_mechanism_model,
    build_mediation_model,
    build_rlm_adjusted_model,
    build_rlm_concurrent_model,
    build_rlm_horseshoe_model,
    build_rlm_joint_growth_model,
    build_rlm_transition_adjusted_model,
    build_two_mediator_model,
)
from language_reading_predictors.statistical_models.prior_artifacts import (
    _prior_table_overrides,
)
from language_reading_predictors.statistical_models.measures import (
    ITT_OUTCOMES,
    MEASURES,
    TAUGHT_BLOCK1_OUTCOMES,
)
from language_reading_predictors.statistical_models.preprocessing import (
    _subset_prepared,
    load_and_prepare,
    load_and_prepare_aligned,
    load_longitudinal_panel,
    load_rlm_concurrent_frames,
    load_rlm_span_frame,
    load_rlm_transition_frame,
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
    # The tight-lengthscale (InverseGamma(8, 8)) opt-in taken by 16 of the 20
    # registered HSGP mechanism fits, and the continuous-covariate exposure form.
    # Neither was in the fixture, so nothing caught the lengthscale panel/rationale
    # drift (#586 finding 3).
    models["mechanism_tight"] = build_mechanism_model(
        allp, mechanism_symbol="L", outcome_symbol="W",
        confounder_symbols=("G", "A"), mech_hsgp_m=6,
        mech_lengthscale_prior=priors.ell_prior_mech_tight(),
    ).model
    models["mechanism_covariate"] = build_mechanism_model(
        load_and_prepare(
            path=p, phase_mode="all", outcomes=("W",), covariates=("attend",)
        ),
        mechanism_symbol="attend", outcome_symbol="W",
        confounder_symbols=("G", "A"), mechanism_is_covariate=True,
    ).model

    dosep = load_and_prepare(
        path=p, phase_mode="all", outcomes=("W",), covariates=("attend", "attend_cumul")
    )
    models["dose_response"] = build_dose_response_model(
        dosep, outcome_symbol="W", period_varying_dose=True
    ).model

    levels = load_and_prepare(path=p, phase_mode="levels")
    models["did"] = build_did_model(levels, outcome_symbol="W").model
    # Heterogeneity variant (#294): guards the sigma_delta prior + v_delta_raw offset,
    # which the plain did model above does not exercise.
    models["did_varying_delta"] = build_did_model(
        levels, outcome_symbol="W", use_varying_delta=True
    ).model

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

    lcf_panel = load_wave_panel(
        path=p, outcomes=("R", "E", "TR", "TE", "L", "B", "F", "T")
    )
    models["long_corr_factor"] = build_longitudinal_corr_factor_model(
        lcf_panel,
        domains={
            "vocabulary": ("R", "E", "TR", "TE"),
            "code": ("L", "B"),
            "grammar": ("F", "T"),
        },
    ).model

    # #384: kinds whose prior-table role/rationale label fixes need a built model to
    # exercise the override path (and to guard against role='other' regressions).
    itt_adj = load_and_prepare(path=p, phase_mode="itt")
    itt_adj.covariates["blocks"] = np.linspace(-1, 1, itt_adj.n_obs)
    models["itt_adjusted"] = build_itt_model(
        itt_adj, outcome_symbol="R", cross_symbols=(), adjust_for=("blocks",)
    ).model
    cf_grp = load_and_prepare(path=p, phase_mode="itt")
    cf_grp.covariates["blocks"] = np.linspace(-1, 1, cf_grp.n_obs)
    models["corr_factor_group"] = build_correlated_factor_model(
        cf_grp,
        outcome_symbol="W",
        domains={"vocabulary": ("R", "E"), "code": ("L", "B"), "grammar": ("F", "T")},
        structural_covariates=("blocks",),
        use_group=True,
    ).model
    # Two mediators (L, B) with a non-mediator confounder (E): exercises the
    # second-mediator a-path aB_G (previously role 'other') and the confounder
    # b_E reroute (E is a confounder, not a mediator, here).
    models["mediation_multi_conf"] = build_two_mediator_model(
        itt, outcome_symbol="W", mediator_symbols=("L", "B"), confounder_symbols=("E",)
    )[0].model
    # Latent growth-curve family: the mean slope (beta), non-centred RE offsets
    # (z_intercept/z_slope) and the shared-tempo factor scores (G_tempo) previously
    # dropped to role='other' (a pre-existing #141 gap). Now classified, so the
    # family joins the no-'other' completeness guard.
    gpanel = load_wave_panel(
        path=p, outcomes=("W", "L", "E"), baseline_covariates=("blocks",)
    )
    models["growth"] = build_growth_model(
        gpanel, use_shared_factor=True, age_ability_interaction=True
    ).model
    # Joint-mechanism family (#421 Tier 3), both designs: the LKJ dependence-block
    # RVs (u_resid_chol/z, u_child_chol/z) and the per-design group terms were
    # invisible to the no-'other' guard while the family had no entry here
    # (2026-08-21 joint-mechanism review, finding 5).
    jm_levels_all = load_and_prepare(
        path=p, phase_mode="levels", outcomes=("W", "N", "L")
    )
    models["joint_mechanism_levels"] = build_joint_mechanism_model(
        _subset_prepared(jm_levels_all, jm_levels_all.phase == 2), design="levels"
    ).model
    jm_all = load_and_prepare(path=p, phase_mode="all", outcomes=("W", "N", "L"))
    models["joint_mechanism_transition"] = build_joint_mechanism_model(
        jm_all, design="transition"
    ).model

    # The two Byrne (RLM) historical families. Neither had ever been under this
    # guard — the fixture built only RLI models — which is how the within-child
    # joint model shipped a reporting-tier fit whose HEADLINE prior
    # (``within_corr_chol``) was published as role='other' with an empty
    # rationale, alongside ``z_within`` and an unexplained ``sigma_within``
    # (2026-08-21 historical-families review, finding 7).
    rlm_path = _write_rlm_synthetic(tmp_path)
    rlm_measures = ("basread", "bpvs", "basdig")
    rlm_panel = load_longitudinal_panel(
        _rlm_dataset(rlm_path),
        [RLM_MEASURES[m] for m in rlm_measures],
        waves=(1, 2, 3),
    )
    models["historical_growth"] = build_historical_growth_model(
        rlm_panel, measure="basread"
    ).model
    models["historical_joint"] = build_rlm_joint_growth_model(
        rlm_panel, measures=rlm_measures
    ).model
    models["historical_joint_within"] = build_rlm_joint_growth_model(
        rlm_panel, measures=rlm_measures, within_correlation=True
    ).model
    # Adjusted family, both ports (2026-08-22 review, finding 8): none of the three
    # factories had ever been under this guard, which is how the stacked Byrne
    # transition model's ``alpha_transition`` reached a published priors table as
    # role='other'.
    span = load_and_prepare(
        path=p,
        phase_mode="span",
        outcomes=("W", "L", "B", "R", "E", "F"),
        covariates=(V.BLOCKS, V.BEHAV),
    )
    models["adjusted"] = build_adjusted_model(
        span, outcome_symbol="W",
        predictors=["L", "lang", "B", "age", V.BLOCKS, V.BEHAV],
    ).model
    rlm_span = load_rlm_span_frame(
        path=rlm_path, predictor_measures=("bpvs", "basdig"), include_age=False
    )
    models["rlm_adjusted"] = build_rlm_adjusted_model(rlm_span).model
    rlm_transition = load_rlm_transition_frame(
        path=rlm_path,
        predictor_measures=("bpvs", "basdig"),
        include_age=False,
        transition_waves=(1, 2, 3),
    )
    models["rlm_transition_adjusted"] = build_rlm_transition_adjusted_model(
        rlm_transition
    ).model
    models["rlm_transition_adjusted_varying"] = build_rlm_transition_adjusted_model(
        rlm_transition, varying_slopes=True
    ).model
    # The Byrne horseshoe and concurrent factories share the dispersion-scale
    # concentration prior since 2026-08-22; they join the guard with the family
    # whose frames they partner.
    models["rlm_horseshoe"] = build_rlm_horseshoe_model(rlm_span).model
    rlm_wave = load_rlm_concurrent_frames(
        path=rlm_path,
        outcome="basread",
        predictor_measures=("bpvs", "basdig"),
        waves=(1,),
        include_age=False,
    )[1]
    models["rlm_concurrent"] = build_rlm_concurrent_model(
        rlm_wave, predictor_symbols=("bpvs", "basdig"), include_age=False
    ).model
    return models


def _write_rlm_synthetic(tmp_path, n_per_group: int = 6, seed: int = 11):
    """A Byrne-shaped long CSV: three reading groups x three waves, three measures."""
    rng = np.random.default_rng(seed)
    rows = []
    for group in (1, 2, 3):
        for k in range(n_per_group):
            for wave in (1, 2, 3):
                row = {
                    "subject_id": f"S{group}{k}",
                    "time": wave,
                    "readgrp": group,
                }
                for symbol in ("basread", "bpvs", "basdig"):
                    ceiling = RLM_MEASURES[symbol].n_trials
                    row[symbol] = int(
                        rng.integers(0, max(2, ceiling // 2)) + 2 * (wave - 1)
                    )
                rows.append(row)
    path = tmp_path / "rlm_prior_inventory_long.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _rlm_dataset(path):
    from language_reading_predictors.statistical_models.datasets import DatasetSpec

    return DatasetSpec(
        study_id="rlm_test",
        label="synthetic",
        path=path,
        group_labels={1: "Down syndrome", 2: "Average readers", 3: "Reading-matched"},
    )


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


def test_did_varying_delta_guards_and_rvs(tmp_path):
    """Varying catch-up has guarded inputs and explicit waitlist-child nodes."""
    p = _write_synthetic(tmp_path)
    levels = load_and_prepare(path=p, phase_mode="levels")
    dosep = load_and_prepare(
        path=p, phase_mode="all", outcomes=("W",), covariates=("attend",)
    )
    # The exploratory waitlist catch-up deviation needs the child intercept.
    with pytest.raises(ValueError):
        build_did_model(
            levels,
            outcome_symbol="W",
            use_varying_delta=True,
            use_child_re=False,
        )
    # Catch-up heterogeneity is not an intensive session-dose slope.
    with pytest.raises(ValueError):
        build_did_model(dosep, outcome_symbol="W", use_varying_delta=True, dose=True)
    built = build_did_model(levels, outcome_symbol="W", use_varying_delta=True)
    free = {rv.name for rv in built.model.free_RVs}
    dets = {d.name for d in built.model.deterministics}
    assert "sigma_delta" in free and "v_delta_raw" in free
    assert "delta_crossover_i" in dets


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
    Normal(-0.3, 0.2), not gamma_cross's Normal(0, 0.3) (the prefix-fallback bug).

    ``b_self`` was recentred to a mean-reverting Normal(-0.3, 0.2) — the level AR(1)
    coefficient is phi = 1 + b_self, so the old Normal(0, 0.5) centred phi on a unit
    root with ~50% explosive mass (review finding A3, 2026-07-13)."""
    table = priors.priors_table(built_models["lcsm"])
    by_param = {r["parameter"]: r for _, r in table.iterrows()}
    assert by_param["a_change"]["distribution"] == "Normal(0, 1.5)"
    assert by_param["b_self"]["distribution"] == "Normal(-0.3, 0.2)"
    # And none of them are the wrongly-inherited cross-coupling scale.
    assert by_param["a_change"]["distribution"] != "Normal(0, 0.3)"
    # b_self is the own-measure AR(1) self-feedback (phi = 1 + b_self), the LCSM
    # analogue of the own-baseline gamma_own — a precision own-dynamics term, not one
    # of the reported cross-couplings (#384 review, Frank).
    assert by_param["b_self"]["role"] == "precision"
    assert priors._classify_fallback("b_self", "Normal(-0.3, 0.2)")[0] == "precision"


def test_joint_lkj_residual_documented(built_models):
    table = priors.priors_table(built_models["joint_lkj"])
    by_param = {r["parameter"]: r for _, r in table.iterrows()}
    assert "u_chol" in by_param
    assert by_param["u_chol"]["distribution"] not in ("", "(model prior)")
    assert by_param["u_chol"]["role"] != "other"


def test_long_corr_factor_priors_have_roles_and_rationales(built_models):
    """The bespoke LCF measurement/dependence priors must be fully inventoried."""
    table = priors.priors_table(built_models["long_corr_factor"])
    assert set(table["role"]) <= {"association", "nuisance"}
    assert table["rationale"].str.strip().ne("").all()
    # Under the pooled-budget communality parameterisation (#383 follow-up) the
    # free measurement RV is the communality; lambda_load / sigma_indicator are
    # derived Deterministics and so no longer appear in the free-RV priors table.
    assert {
        "communality",
        "trait_share",
        "trait_corr_chol",
        "factor_mean",
    } <= set(table["parameter"])
    assert not {"lambda_load", "sigma_indicator"} & set(table["parameter"])
    by_param = {r["parameter"]: r for _, r in table.iterrows()}
    assert by_param["communality"]["distribution"] == "Beta(2, 2)"
    state = table[table["parameter"].str.startswith("state_corr_chol_w")]
    assert len(state) == 4
    assert state["role"].eq("association").all()


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


# ---------------------------------------------------------------------------
# Prior-table role/rationale label corrections (issue #384). The distribution
# column is read off the built RV and is always correct; these guard the derived
# ``role`` / ``rationale`` labels, which reused-constructor RVs otherwise inherit
# from the wrong constructor's docstring.
# ---------------------------------------------------------------------------


def _labelled(model, *, kind, extra=None, outcome_symbol="W"):
    """priors_table with the per-kind ``_prior_table_overrides`` applied, keyed by
    parameter name (mirrors what the pipeline writes to ``priors_table.csv``)."""
    if kind == "itt":
        from language_reading_predictors.statistical_models.context import ModelSpec
        from language_reading_predictors.statistical_models.itt import (
            resolve_itt_run_plan,
        )

        # ``adjustment`` mirrors the legacy ``extra["adjust_for"]``: resolution
        # requires the two to agree (2026-08-22 ITT audit, finding 9).
        spec = ModelSpec(
            model_id="lrp-test-itt-prior",
            kind="itt",
            title="t",
            outcome_symbol=outcome_symbol,
            adjustment=list((extra or {}).get("adjust_for", ()) or ()),
            extra=extra or {},
        )
        ctx = SimpleNamespace(
            spec=spec,
            resolved_plan=resolve_itt_run_plan(spec),
            model=model,
        )
    elif kind == "mechanism":
        # The mechanism branch reads its lengthscale constructor off the resolved
        # run plan rather than the RV-name suffix (#586 finding 3), so the fixture
        # must hand it a real spec + plan the way the pipeline does.
        from language_reading_predictors.statistical_models.context import ModelSpec
        from language_reading_predictors.statistical_models.mechanism import (
            resolve_mechanism_run_plan,
        )

        _extra = dict(extra or {})
        baseline = _extra.get("adjust_baseline_symbol", "W")
        spec = ModelSpec(
            model_id="lrp-test-mech-prior",
            kind="mechanism",
            title="t",
            outcome_symbol=outcome_symbol,
            mechanism_symbol=_extra.pop("_mechanism_symbol", "L"),
            adjustment=["G", "A", f"{baseline}_pre"],
            extra=_extra,
        )
        ctx = SimpleNamespace(
            spec=spec,
            resolved_plan=resolve_mechanism_run_plan(spec),
            model=model,
        )
    else:
        spec = SimpleNamespace(kind=kind, extra=extra or {}, outcome_symbol=outcome_symbol)
        ctx = SimpleNamespace(spec=spec, model=model)
    ctor, role, rationale = _prior_table_overrides(ctx)
    table = priors.priors_table(
        model,
        ctor_overrides=ctor,
        role_overrides=role,
        rationale_overrides=rationale,
    )
    return {r["parameter"]: r for _, r in table.iterrows()}


def test_corr_factor_beta_G_is_association_not_causal(built_models):
    """mm-002's arm covariate beta_G shipped as role=causal + 'Treatment effect
    tau' — it is a mech-058 backdoor covariate, an adjusted association."""
    by = _labelled(built_models["corr_factor_group"], kind="corr_factor")
    assert by["beta_G"]["role"] == "association"
    assert by["beta_G"]["panel"] == "predictor_slope"
    assert "mech-058" in by["beta_G"]["rationale"]
    assert "Treatment effect tau" not in by["beta_G"]["rationale"]
    # factor_corr_chol is an association: its off-diagonals are the reported
    # factor-correlation matrix (the ``factor_corr_pairs`` deterministic), consistent
    # with this branch's ``*_corr_chol`` carve-out (#384 review, Frank).
    if "factor_corr_chol" in by:
        assert by["factor_corr_chol"]["role"] == "association"
        assert by["factor_corr_chol"]["rationale"].strip()


def test_joint_mechanism_group_terms_and_dependence_blocks(built_models):
    """jm-002's ``beta_G`` shipped role='causal' with the tau rationale, jm-001's
    ``beta_group_nuisance`` row published the inline record's Normal(0, 1) even
    when the fitted prior differed, and the LKJ blocks dropped to role='other' —
    none catchable while the family was absent from this fixture (2026-08-21
    joint-mechanism review, finding 5)."""
    trans = _labelled(
        built_models["joint_mechanism_transition"], kind="joint_mechanism"
    )
    assert trans["beta_G"]["role"] == "association"
    assert "Treatment effect tau" not in trans["beta_G"]["rationale"]
    assert "mech-096" in trans["beta_G"]["rationale"]
    for name in ("u_child_chol", "u_child_z"):
        assert trans[name]["role"] == "nuisance"
        assert trans[name]["rationale"].strip()

    levels = _labelled(built_models["joint_mechanism_levels"], kind="joint_mechanism")
    row = levels["beta_group_nuisance"]
    assert row["role"] == "nuisance"
    # The distribution is read off the built RV, never the inline record's
    # hard-coded default (which silently misreported a differing fitted prior).
    assert row["distribution"] == "Normal(0, 1)"
    assert "ca-010" in row["rationale"]
    for name in ("u_resid_chol", "u_resid_z"):
        assert levels[name]["role"] == "nuisance"
        assert levels[name]["rationale"].strip()


def test_mechanism_beta_G_and_ell_rationales(built_models):
    by = _labelled(built_models["mechanism"], kind="mechanism")
    assert by["beta_G"]["role"] == "association"
    assert "backdoor" in by["beta_G"]["rationale"]
    assert "Treatment effect tau" not in by["beta_G"]["rationale"]
    ell = [r for n, r in by.items() if n.endswith("__ell")]
    assert ell, "mechanism HSGP lengthscale row missing"
    for r in ell:
        assert r["distribution"] == "InverseGamma(5, 5)"
        assert "InverseGamma(5, 5)" in r["rationale"]  # not the IG(3, 1) default doc


@pytest.mark.parametrize(
    ("fixture", "extra", "outcome", "expected"),
    [
        ("mechanism", {}, "W", "InverseGamma(5, 5)"),
        (
            "mechanism_tight",
            {"mech_hsgp_m": 6, "mech_lengthscale_tight": True},
            "W",
            "InverseGamma(8, 8)",
        ),
        (
            "mechanism_covariate",
            {"outcomes": ("W",), "mechanism_is_covariate": True,
             "_mechanism_symbol": "attend"},
            "W",
            "InverseGamma(5, 5)",
        ),
    ],
)
def test_mechanism_lengthscale_panel_matches_the_fitted_rv(
    built_models, fixture, extra, outcome, expected
):
    """The PyMC RV, distribution column, rationale and density panel must agree.

    Before #586 finding 3 the ``__ell`` name suffix routed every mechanism
    lengthscale to the shared ``ell`` constructor: the panel plotted
    ``InverseGamma(3, 1)``, the rationale hard-coded ``InverseGamma(5, 5)`` and only
    the distribution column (read off the RV) was right — so all 20 HSGP mechanism
    reports displayed a density their model never fitted, and the 16 tight fits
    contradicted their own row.
    """
    by = _labelled(built_models[fixture], kind="mechanism", extra=extra,
                   outcome_symbol=outcome)
    rows = [r for n, r in by.items() if n.endswith("__ell")]
    assert rows, f"{fixture}: mechanism HSGP lengthscale row missing"
    for row in rows:
        assert row["distribution"] == expected
        assert expected in row["rationale"]
        # ``panel`` is the constructor key ``save_shared_prior_panel`` plots, so
        # reconciling it with the RV closes the last leg of the loop.
        panel_ctor = priors.ALL_PRIORS[row["panel"]]
        assert priors._dist_from_doc(panel_ctor) == expected


def test_linear_mechanism_registers_no_lengthscale(built_models):
    """A linear-mechanism fit builds no GP, so it must panel no lengthscale at all.

    21 of the 41 registered mechanism specifications are linear; the shared report
    nonetheless described every one of them as an HSGP curve (#586 finding 3).
    """
    by = _labelled(
        built_models["mechanism_linear"],
        kind="mechanism",
        extra={"linear_mechanism": True, "outcomes": ("W", "L", "B")},
        outcome_symbol="B",
    )
    assert not [n for n in by if n.endswith("__ell") or n.endswith("__eta")]
    assert "beta_mech" in by


def test_aligned_beta_cohort_is_not_a_treatment_effect(built_models):
    by = _labelled(built_models["aligned"], kind="aligned")
    assert by["beta_cohort"]["role"] == "association"
    assert "per-protocol" in by["beta_cohort"]["rationale"].lower()
    assert "Treatment effect tau" not in by["beta_cohort"]["rationale"]


def test_dose_beta_G_and_mu_dose_rationales(built_models):
    by = _labelled(built_models["dose_response"], kind="dose_response")
    assert by["beta_G"]["role"] == "association"
    assert "Treatment effect tau" not in by["beta_G"]["rationale"]
    assert "backdoor" in by["beta_G"]["rationale"]
    assert "dose-response slope" in by["mu_dose"]["rationale"]


def test_growth_interaction_and_tempo_loading(built_models):
    by = _labelled(built_models["growth"], kind="growth")
    assert by["gamma_int"]["role"] == "association"
    assert "age x ability interaction" in by["gamma_int"]["rationale"]
    assert "age main effect" in by["gamma_age"]["rationale"]
    # ``loading`` must not inherit the CFA test->domain measurement-loading text.
    assert "growth-tempo factor" in by["loading"]["rationale"]
    assert "domain factor" not in by["loading"]["rationale"]


def test_growth_bespoke_terms_classified(built_models):
    """The growth mean-slope + random-effect/tempo terms previously dropped to
    role='other' (a #141 completeness gap). beta is labelled an association (a
    descriptive maturational trend — a reviewable call, see PR note); the offsets
    and factor scores are nuisances."""
    by = _labelled(built_models["growth"], kind="growth")
    # beta: per-measure population mean growth rate -> association (not 'other').
    assert by["beta"]["role"] == "association"
    assert "growth rate" in by["beta"]["rationale"]
    # Non-centred RE offsets + shared-tempo factor scores -> nuisance with a real
    # rationale (were role='other', empty rationale).
    for name in ("z_intercept", "z_slope", "G_tempo"):
        assert by[name]["role"] == "nuisance", name
        assert by[name]["rationale"].strip(), name
    # The concurrent-family focal ``beta`` (Normal(0, 0.3)) is unaffected by the
    # growth-scale beta rationale (kept out by the scale guard).
    assert priors._fallback_rationale("beta", "Normal(0, 0.3)") == ""
    # A future inline ``beta`` at any other scale must still fall through to
    # role='other' so the completeness guard catches it (Copilot #397): the role
    # rule is keyed to the growth Normal(0, 0.5) signature, not the bare name.
    assert priors._classify_fallback("beta", "Normal(0, 1)") == ("other", "")


def test_itt_adjust_covariate_and_ses_are_precision(built_models):
    by = _labelled(
        built_models["itt_adjusted"],
        kind="itt",
        extra={"adjust_for": ("blocks",)},
        outcome_symbol="R",
    )
    assert by["gamma_blocks"]["role"] == "precision"
    assert "cannot confound" in by["gamma_blocks"]["rationale"]
    # SES covariates are precision too (#384 review, Frank): pre-randomisation and
    # balanced across arms in expectation, so they cannot confound tau and only
    # sharpen it — the identical causal status to blocks/area, documented in the
    # LRPITT13/113 module docstrings so the role is quoted, not inferred.
    from language_reading_predictors.statistical_models.context import ModelSpec
    from language_reading_predictors.statistical_models.itt import resolve_itt_run_plan

    spec = ModelSpec(
        model_id="lrp-test-itt-ses-prior",
        kind="itt",
        title="t",
        # Mirrors ``adjust_for``, which resolution requires (2026-08-22 ITT
        # audit, finding 9).
        adjustment=["mumedupost16"],
        extra={"adjust_for": ("mumedupost16",)},
        outcome_symbol="R",
    )
    _, role, rationale = _prior_table_overrides(
        SimpleNamespace(
            spec=spec,
            resolved_plan=resolve_itt_run_plan(spec),
            model=SimpleNamespace(free_RVs=[]),
        )
    )
    assert role["gamma_mumedupost16"] == "precision"
    assert "cannot confound" in rationale["gamma_mumedupost16"]


def test_missing_indicator_coefficients_are_nuisance():
    """beta/gamma missing indicators are subgroup mean-offsets under the
    missing-indicator method, confounded with the fill value and uninterpretable as
    an effect, so they are nuisance (not predictor-slope associations) in every
    family that carries them (currently adjusted LRP65 and correlated-factor mm-002;
    #384 review, Frank)."""
    cases = (
        ("adjusted", "beta_hs_missing"),
        ("corr_factor", "beta_hs_missing"),
        ("concurrent", "gamma_hs_missing"),
    )
    for kind, name in cases:
        spec = SimpleNamespace(kind=kind, extra={}, outcome_symbol="W")
        _, role, rationale = _prior_table_overrides(
            SimpleNamespace(
                spec=spec,
                model=SimpleNamespace(free_RVs=[SimpleNamespace(name=name)]),
            )
        )
        assert role[name] == "nuisance"
        assert "missing-indicator" in rationale[name].lower()


def test_mediation_paths_and_confounders(built_models):
    by = _labelled(built_models["mediation"], kind="mediation")
    assert by["a_G"]["role"] == "association" and "a-path" in by["a_G"]["rationale"]
    assert "direct-path" in by["b_G"]["rationale"]
    assert "Treatment effect tau" not in by["a_G"]["rationale"]
    # The own-baseline autoregression a_L (Normal(1, 0.25)) stays a precision term.
    assert by["a_L"]["role"] == "precision"
    # A confounder b-path (Normal(0, 0.3)) is rerouted to gamma_cross.
    assert by["b_E"]["panel"] == "gamma_cross"
    assert "confounder" in by["b_E"]["rationale"]
    # The reported mediator b-path (Normal(0, 1)) is left alone.
    assert by["b_M"]["panel"] == "b_path"


def test_two_mediator_second_a_path_documented(built_models):
    by = _labelled(built_models["mediation_multi_conf"], kind="mediation_multi")
    # aB_G was role 'other' with an empty rationale before the #384 fix.
    assert by["aB_G"]["role"] == "association"
    assert "a-path" in by["aB_G"]["rationale"]
    # E is a confounder in this (L, B) two-mediator model, so b_E reroutes.
    assert by["b_E"]["panel"] == "gamma_cross"


def test_block_exposure_and_survival_terms_are_not_causal():
    """The new block_exposure/survival branches demote the reused causal
    constructor to an association (the modules declare no randomised effect)."""
    for kind, name in (("block_exposure", "delta"), ("survival", "tau")):
        spec = SimpleNamespace(kind=kind, extra={}, outcome_symbol="W")
        _, role, rationale = _prior_table_overrides(
            SimpleNamespace(spec=spec, model=SimpleNamespace(free_RVs=[]))
        )
        assert role[name] == "association"
        assert rationale[name].strip()


def test_base_fallback_role_and_rationale_additions():
    # Previously role 'other' (guard failures for the rlm/two-mediator families).
    assert priors._classify_fallback("z_subject", "Normal(0, 1)")[0] == "nuisance"
    assert priors._classify_fallback("eta_cell", "Normal(0, 1.5)")[0] == "nuisance"
    assert (
        priors._classify_fallback("measure_corr_chol", "LKJCholeskyCov(3, ...)")[0]
        == "association"
    )
    assert priors._classify_fallback("aB_G", "Normal(0, 0.5)")[0] == "association"
    # Previously empty rationales.
    for nm, dist in (
        ("b_self", "Normal(-0.3, 0.2)"),
        ("z_subject", "Normal(0, 1)"),
        ("eta_cell", "Normal(0, 1.5)"),
        ("sigma_subject", "HalfNormal(0.5)"),
        ("measure_corr_chol", "LKJCholeskyCov(3, ...)"),
    ):
        assert priors._fallback_rationale(nm, dist).strip(), nm


def test_beta_group_nuisance_slug_is_nuisance():
    """The slug-suffixed RLM cohort dummies must resolve to the inline nuisance
    entry, not the beta_ prefix -> predictor_slope (association) path."""
    import pymc as pm

    with pm.Model():
        rv = pm.Normal("beta_group_nuisance_down_syndrome", 0.0, 1.0)
    info = priors.prior_info_for_rv("beta_group_nuisance_down_syndrome", rv=rv)
    assert info["role"] == "nuisance"
    assert info["distribution"] == "Normal(0, 1)"
    assert "nuisance dummy" in info["rationale"]
