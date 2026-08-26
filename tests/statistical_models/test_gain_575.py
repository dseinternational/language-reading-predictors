# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The #575 gain-factor audit's contracts, pinned.

Each test names the finding it closes. The settings-validation tests are pure
unit tests; the factory tests build real PyMC graphs on the repository data
(no sampling); the release-policy tests drive the decision function directly;
the report-contract test walks every registered gain page against its module's
declaration so stale prose fails in CI rather than in review.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from language_reading_predictors.statistical_models.gain_factors import (
    GainFactorsModelSettings,
    resolve_gain_factors_run_plan,
)

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Finding 11 — typed adjustment roles and interaction hygiene
# ---------------------------------------------------------------------------


def test_settings_reject_the_attend_collider():
    with pytest.raises(ValueError, match="attend"):
        GainFactorsModelSettings(adjust_for=("attend",))


def test_settings_reject_an_arbitrary_adjuster_name():
    with pytest.raises(ValueError, match="confounder vocabulary"):
        GainFactorsModelSettings(adjust_for=("ses",))


def test_settings_accept_the_registered_confounder_vocabulary():
    GainFactorsModelSettings(
        adjust_for=(
            "hs", "hs_missing", "deapp_c", "deapp_c_missing", "erbto",
            "erbto_missing",
        )
    )


def test_settings_reject_duplicate_adjusters():
    with pytest.raises(ValueError, match="repeats"):
        GainFactorsModelSettings(adjust_for=("hs", "hs_missing", "hs"))


def test_settings_reject_an_unpaired_missingness_indicator():
    with pytest.raises(ValueError, match="without the covariate"):
        GainFactorsModelSettings(adjust_for=("hs_missing",))


def test_settings_reject_interaction_self_pairs():
    with pytest.raises(ValueError, match="self-pair"):
        GainFactorsModelSettings(interactions=(("own", "own"),))


def test_settings_reject_duplicate_interaction_pairs():
    with pytest.raises(ValueError, match="duplicates an earlier pair"):
        GainFactorsModelSettings(
            ability_covariate="blocks",
            interactions=(("age", "ability"), ("age", "ability")),
        )


def test_settings_reject_reversed_duplicate_interaction_pairs():
    with pytest.raises(ValueError, match="duplicates an earlier pair"):
        GainFactorsModelSettings(
            ability_covariate="blocks",
            interactions=(("age", "ability"), ("ability", "age")),
        )


def test_settings_reject_descriptive_skills_outside_skill_symbols():
    with pytest.raises(ValueError, match="skill_symbols"):
        GainFactorsModelSettings(skill_symbols=("L",), descriptive_skills=("R",))


def test_settings_validate_the_new_prior_axes():
    with pytest.raises(ValueError, match="kappa_prior_family"):
        GainFactorsModelSettings(kappa_prior_family="lognormal")
    with pytest.raises(ValueError, match="gamma_own_prior_sigma"):
        GainFactorsModelSettings(gamma_own_prior_sigma=0.0)


# ---------------------------------------------------------------------------
# Findings 1, 5, 9, 10 — the resolved plan and the built model
# ---------------------------------------------------------------------------


def _plan_for(model_id: str):
    from language_reading_predictors.statistical_models.registry import (
        discover_models,
    )

    return resolve_gain_factors_run_plan(discover_models()[model_id].load().SPEC)


def test_plan_threads_the_new_prior_axes_and_flags():
    plan = _plan_for("lrp-rli-gf-001")
    assert plan.kappa_prior_family == "halfnormal_concentration"
    assert plan.gamma_own_prior_sigma == 0.25
    assert plan.period1_sensitivity_required is True
    kwargs = plan.factory_kwargs()
    assert kwargs["kappa_prior_family"] == "halfnormal_concentration"
    assert kwargs["gamma_own_prior_sigma"] == 0.25
    recorded = plan.as_dict()
    for field in (
        "kappa_prior_family",
        "gamma_own_prior_sigma",
        "descriptive_skills",
        "period1_sensitivity_required",
    ):
        assert field in recorded, field


def test_period1_sensitivity_binds_to_the_model_of_record_only():
    assert _plan_for("lrp-rli-gf-001").period1_sensitivity_required is True
    assert _plan_for("lrp-rli-gf-306").period1_sensitivity_required is True
    assert _plan_for("lrp-rli-gf-101").period1_sensitivity_required is False
    assert _plan_for("lrp-rli-gf-201").period1_sensitivity_required is False


def test_gf_012_and_013_declare_their_descriptive_skills():
    assert _plan_for("lrp-rli-gf-012").descriptive_skills == ("R", "E")
    plan_13 = _plan_for("lrp-rli-gf-013")
    assert plan_13.descriptive_skills == ("R", "E")
    assert "TR" in plan_13.skill_symbols  # upstream adjuster retained


@pytest.fixture(scope="module")
def _built_gf_005():
    from language_reading_predictors.statistical_models import factories
    from language_reading_predictors.statistical_models.preprocessing import (
        load_and_prepare,
    )

    plan = _plan_for("lrp-rli-gf-005")
    prepared = load_and_prepare(**plan.prepare_kwargs())
    adjust = tuple(c for c in plan.adjust_for if c in prepared.covariates)
    return factories.build_gain_factors_model(
        prepared, **plan.factory_kwargs(effective_adjustment=adjust)
    )


def test_final_mask_refilter_drops_the_erbto_missing_alias(_built_gf_005):
    # Finding 1's concrete case: after the focal-outcome mask, erbto_missing is
    # constant on gf-005's 159 fitted rows — it must be dropped, recorded, and
    # absent from the free RVs.
    payload = _built_gf_005.payload
    assert "erbto_missing" in payload.post_mask_dropped_adjusters
    assert "erbto_missing" not in payload.effective_adjust_for
    names = [rv.name for rv in _built_gf_005.model.free_RVs]
    assert not any("erbto_missing" in n for n in names)
    assert "erbto_missing" in _built_gf_005.prepared.dropped_covariates


def test_period_arm_support_is_recorded_per_cell(_built_gf_005):
    support = dict()
    for period, arm, n_rows, n_children in _built_gf_005.payload.period_arm_support:
        support[(period, arm)] = (n_rows, n_children)
    # Both randomised arms present in period 1 (the audit's realised counts).
    assert support[(0, "immediate")][0] > 0
    assert support[(0, "waitlist")][0] > 0


def test_causal_fit_requires_both_arms_in_period_1():
    from language_reading_predictors.statistical_models import factories
    from language_reading_predictors.statistical_models.preprocessing import (
        _subset_prepared,
        load_and_prepare,
    )

    plan = _plan_for("lrp-rli-gf-001")
    prepared = load_and_prepare(**plan.prepare_kwargs())
    one_arm = _subset_prepared(
        prepared, ~((np.asarray(prepared.G) == 0) & (np.asarray(prepared.phase) == 0))
    )
    with pytest.raises(ValueError, match="both randomised arms"):
        factories.build_gain_factors_model(
            one_arm, **plan.factory_kwargs(effective_adjustment=())
        )


def test_gain_factory_accepts_the_dispersion_and_own_prior_axes():
    from language_reading_predictors.statistical_models import factories
    from language_reading_predictors.statistical_models.preprocessing import (
        load_and_prepare,
    )

    plan = _plan_for("lrp-rli-gf-004")
    prepared = load_and_prepare(**plan.prepare_kwargs())
    adjust = tuple(c for c in plan.adjust_for if c in prepared.covariates)
    built = factories.build_gain_factors_model(
        prepared,
        **{
            **plan.factory_kwargs(effective_adjustment=adjust),
            "kappa_prior_family": "halfnormal_inverse_sqrt",
            "gamma_own_prior_sigma": 0.5,
        },
    )
    # The inverse-sqrt parameterisation keeps kappa as a deterministic transform
    # of the free dispersion-scale variable rather than a free kappa RV.
    free = [rv.name for rv in built.model.free_RVs]
    assert "kappa" not in free
    assert "kappa" in [d.name for d in built.model.deterministics]


# ---------------------------------------------------------------------------
# Finding 10d — the off-floor release policy mirrors the ITT floor rule
# ---------------------------------------------------------------------------


def _off_floor_config() -> dict:
    return {
        "model_id": "lrp-rli-gf-005",
        "kind": "gain_factors",
        "outcome_symbol": "P",
        "resolved_run_plan": {"off_floor": True},
    }


def test_off_floor_conflict_without_sweep_withholds(tmp_path):
    from language_reading_predictors.statistical_models.release import (
        _gain_offfloor_decision,
    )

    decision = _gain_offfloor_decision(
        tmp_path,
        _off_floor_config(),
        tier="primary",
        tau_class="prior_data_conflict",
        prior=0.3,
        likelihood=0.2,
        diagnosis="prior-data conflict",
        causal_term="beta_trt",
    )
    assert decision.status == "withhold"
    assert "no treatment-prior sweep" in (decision.reason or "")


def test_off_floor_clear_releases_without_a_grid(tmp_path):
    from language_reading_predictors.statistical_models.release import (
        _gain_offfloor_decision,
    )

    decision = _gain_offfloor_decision(
        tmp_path,
        _off_floor_config(),
        tier="primary",
        tau_class="clear",
        prior=0.05,
        likelihood=0.4,
        diagnosis="-",
        causal_term="beta_trt",
    )
    assert decision.status == "release"


def test_off_floor_prior_dominant_without_sweep_withholds(tmp_path):
    from language_reading_predictors.statistical_models.release import (
        _gain_offfloor_decision,
    )

    decision = _gain_offfloor_decision(
        tmp_path,
        _off_floor_config(),
        tier="primary",
        tau_class="prior_dominant",
        prior=0.4,
        likelihood=0.01,
        diagnosis="prior dominant",
        causal_term="beta_trt",
    )
    assert decision.status == "withhold"


def test_evaluate_itt_release_routes_gain_off_floor_fits(tmp_path):
    # The dispatch itself: an off-floor gain config with no psense at all must
    # land in the off-floor branch (floor_rule recorded on the decision) rather
    # than the graded route.
    from language_reading_predictors.statistical_models.release import (
        evaluate_itt_release,
    )

    (tmp_path / "config.json").write_text(json.dumps(_off_floor_config()))
    decision = evaluate_itt_release(
        tmp_path, _off_floor_config(), causal_term="beta_trt"
    )
    assert decision.floor_rule is True
    assert decision.status == "withhold"


# ---------------------------------------------------------------------------
# Finding 10c — MC precision of the derived AME
# ---------------------------------------------------------------------------


def test_treatment_marginal_effect_reports_mc_diagnostics():
    import xarray as xr

    from language_reading_predictors.statistical_models.reporting import (
        treatment_marginal_effect,
    )

    rng = np.random.default_rng(5)
    n_obs, n_draws = 12, 200
    eta = rng.normal(0.0, 1.0, (1, n_draws, n_obs))
    beta = rng.normal(0.4, 0.1, (1, n_draws))
    trace = xr.DataTree.from_dict(
        {
            "posterior": xr.Dataset(
                {
                    "eta": (("chain", "draw", "obs_id"), eta),
                    "beta_trt": (("chain", "draw"), beta),
                }
            )
        }
    )
    trt = (rng.random(n_obs) > 0.5).astype(float)
    out = treatment_marginal_effect(trace, trt=trt, n_trials=20, ci_prob=0.89)
    for key in ("trt_prob_ess_bulk", "trt_prob_ess_tail", "trt_prob_mcse_median"):
        assert key in out and np.isfinite(out[key]), key


# ---------------------------------------------------------------------------
# Finding 8 — the registry-driven report contract
# ---------------------------------------------------------------------------


def _gain_model_ids() -> list[str]:
    from language_reading_predictors.statistical_models.registry import (
        discover_models,
    )

    return sorted(m for m in discover_models() if "-gf-" in m)


def test_gain_pages_do_not_claim_unfitted_treatment_interactions():
    """No non-moderation gain page may describe group interactions as fitted.

    The fitted primaries and treated-only companions carry only the
    age x ability precision interaction since #391; sixteen pages still claimed
    the retired treatment moderation when the audit ran.
    """
    for model_id in _gain_model_ids():
        page = REPO / "docs" / "models" / model_id / "index.qmd"
        if not page.is_file():
            continue
        plan = _plan_for(model_id)
        text = page.read_text(encoding="utf-8")
        declares_trt_interaction = any("trt" in pair for pair in plan.interactions)
        if not declares_trt_interaction:
            for phrase in (
                r"group $\times$ ability",
                r"group $\times$ own",
                r"group \times ability",
                r"group \times own",
            ):
                assert phrase not in text, f"{model_id}: stale interaction claim"


def test_off_floor_pages_show_the_indicator_equation():
    for model_id in _gain_model_ids():
        plan = _plan_for(model_id)
        if not plan.off_floor:
            continue
        page = REPO / "docs" / "models" / model_id / "index.qmd"
        if not page.is_file():
            continue
        text = page.read_text(encoding="utf-8")
        assert r"\gamma_{\text{own}}\operatorname{logit}" not in text, (
            f"{model_id}: page shows the graded own-baseline term, but the "
            "factory fits the binary off-floor-at-pre indicator"
        )
        assert "mathbb{1}" in text, (
            f"{model_id}: off-floor page should display the indicator equation"
        )


def test_gf_005_delta_is_not_called_provisional():
    text = (REPO / "docs" / "models" / "lrp-rli-gf-005" / "index.qmd").read_text(
        encoding="utf-8"
    )
    assert "provisional δ" not in text and "provisional delta" not in text.lower()


def test_old_gain_findings_note_is_marked_superseded():
    text = (REPO / "notes" / "202607161800-findings-gain_factors.md").read_text(
        encoding="utf-8"
    )
    assert "Superseded" in text.split("\n\n")[0] or "Superseded (2026-08-26)" in text


# ---------------------------------------------------------------------------
# Finding 7 — the shared partials fail closed
# ---------------------------------------------------------------------------


def test_setup_partial_requires_a_release_decision():
    text = (REPO / "docs" / "models" / "_partials" / "_setup.qmd").read_text(
        encoding="utf-8"
    )
    assert "release_decision.json" in text
    assert "_release_blocked_structurally" in text


def test_key_findings_partial_fails_closed_on_missing_or_stale_artefacts():
    text = (REPO / "docs" / "models" / "_partials" / "_key_findings.qmd").read_text(
        encoding="utf-8"
    )
    # A missing key_findings.json and an unrecognised status both suppress.
    assert text.count("_scientific_results_released = False") >= 8
    # The stored "ok" is re-decided against the directory's current evidence.
    assert "evaluate_publication" in text


def test_results_factors_partial_guards_and_labels():
    text = (
        REPO / "docs" / "models" / "_partials" / "_results_factors.qmd"
    ).read_text(encoding="utf-8")
    assert 'if _has("tau_forest.png")' in text
    assert 'if _has("rope_summary.png")' in text
    assert "treated-only companion" in text
    assert "period1_sensitivity.csv" in text
    assert "off-floor risk difference (percentage points)" in text


# --- the hearing adjuster's category contrast (#631 finding 4) -----------------


@pytest.fixture(scope="module")
def _built_gf_004_with_hearing():
    """A fitted GF-004 build whose adjustment set includes hearing (`hs`)."""
    from language_reading_predictors.statistical_models import factories
    from language_reading_predictors.statistical_models.preprocessing import (
        load_and_prepare,
    )

    plan = _plan_for("lrp-rli-gf-004")
    prepared = load_and_prepare(**plan.prepare_kwargs())
    adjust = tuple(c for c in plan.adjust_for if c in prepared.covariates)
    assert "hs" in adjust
    built = factories.build_gain_factors_model(
        prepared, **plan.factory_kwargs(effective_adjustment=adjust)
    )
    return plan, built, adjust


def test_standardised_hearing_never_has_literal_binary_support(
    _built_gf_004_with_hearing,
):
    """The regression pin behind #631 finding 4: the design column is standardised,
    so testing it for literal {0, 1} could never detect the binary adjuster."""
    _, built, _ = _built_gf_004_with_hearing
    values = np.asarray(built.prepared.covariates["hs"], dtype=float)
    assert not np.isin(values, (0.0, 1.0)).all()
    # ... while the raw support recovered through the carried scaler is binary.
    raw = np.asarray(built.prepared.covariate_scalers["hs"].inverse(values), dtype=float)
    assert np.all(np.isclose(raw, 0.0) | np.isclose(raw, 1.0))
    assert set(np.unique(np.round(raw, 6))) == {0.0, 1.0}


def test_hearing_association_term_is_a_category_contrast_not_a_sd_shift(
    _built_gf_004_with_hearing,
):
    """`hs` must use the net-out-and-toggle idiom on the raw 0/1 indicator, with
    the eta shift of a raw 0 -> 1 switch (gamma / sd), not a +1 standardised unit."""
    from language_reading_predictors.statistical_models.pipelines.gain_factors import (
        _gf_association_terms,
    )

    plan, built, adjust = _built_gf_004_with_hearing
    terms = _gf_association_terms(
        plan, built, adjust_for=adjust, off_floor=False
    )
    hs_term = next(t for t in terms if t.label == "hs")

    scaler = built.prepared.covariate_scalers["hs"]
    values = np.asarray(built.prepared.covariates["hs"], dtype=float)
    raw = np.asarray(scaler.inverse(values), dtype=float)

    assert hs_term.coef == "gamma_hs"
    assert hs_term.toggle_vector is not None
    assert np.allclose(hs_term.toggle_vector, np.isclose(raw, 1.0).astype(float))
    assert hs_term.main_scale == pytest.approx(1.0 / float(scaler.sd))
    # The defect published a "+1 SD" forward shift of the standardised column,
    # whose scale is ~1.0 — materially smaller than the true category contrast.
    assert hs_term.main_scale > 1.5
    assert "toggled 0 to 1" in (hs_term.perturbation_label or "")


def test_continuous_adjusters_keep_the_sd_shift(_built_gf_004_with_hearing):
    """The fix must not convert genuinely continuous adjusters to toggles."""
    from language_reading_predictors.statistical_models.pipelines.gain_factors import (
        _gf_association_terms,
    )

    plan, built, adjust = _built_gf_004_with_hearing
    terms = _gf_association_terms(plan, built, adjust_for=adjust, off_floor=False)
    labels = {t.label: t for t in terms}
    continuous = [
        name
        for name in adjust
        if name in labels and name != "hs" and not name.endswith("_missing")
    ]
    assert continuous, "expected at least one continuous adjuster in gf-004"
    for name in continuous:
        assert labels[name].toggle_vector is None
        assert labels[name].main_scale == pytest.approx(1.0, abs=0.05)
    # `_missing` companions stay skipped: a +1 SD shift on them means nothing.
    assert not any(t.label.endswith("_missing") for t in terms)
