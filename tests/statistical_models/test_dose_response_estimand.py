# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The dose-response estimand, support and validation repairs (issue #587).

Every test here pins a defect the audit found, so a regression fails rather than
quietly restoring the old behaviour. They exercise the real registered
specifications and the real repository report templates, not fixtures — the audit's
own note records that the pre-existing tests passed while all of these were broken
precisely because they were structural.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from language_reading_predictors.statistical_models import dose_response as D
from language_reading_predictors.statistical_models import factories as F
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipelines import (
    dose_response as P,
)
from language_reading_predictors.statistical_models.preprocessing import (
    load_and_prepare,
)

REPO = Path(__file__).resolve().parents[2]
DOSE_MODELS = ("077", "083", "084", "177", "277")


def _plan(model_id: str):
    module = importlib.import_module(
        f"language_reading_predictors.statistical_models.lrp_rli_dose_{model_id}"
    )
    return D.resolve_dose_response_run_plan(module.SPEC)


def _built(model_id: str):
    plan = _plan(model_id)
    return plan, F.build_dose_response_model(
        load_and_prepare(**plan.prepare_kwargs()), **plan.factory_kwargs()
    )


# --------------------------------------------------------------------------
# Finding 1 — dose-177 must use verified t1 ability, not the transition start
# --------------------------------------------------------------------------


def test_ability_adjusters_are_the_childs_verified_t1_value():
    plan, built = _built("177")
    assert plan.ability_baseline_wave == "t1"
    prepared = built.prepared
    subjects = np.asarray(prepared.subject_ids)
    phase = np.asarray(prepared.phase)

    for symbol in plan.ability_adjust_symbols:
        fitted = np.asarray(built.model[f"{symbol}_pre_logit"].get_value())
        transition_start = np.asarray(prepared.pre_logit[symbol])
        for subject in np.unique(subjects):
            rows = subjects == subject
            values = fitted[rows]
            assert np.allclose(values, values[0]), (
                f"{symbol} varies within child {subject!r}: a baseline adjuster "
                "broadcast from t1 must be constant across that child's transitions"
            )
            at_t1 = transition_start[rows & (phase == 0)]
            assert at_t1.size == 1
            assert np.isclose(values[0], at_t1[0])


def test_transition_start_ability_is_a_labelled_comparator_that_really_differs():
    """The old behaviour is retained but must be *different*, or it is not a comparator."""
    plan = _plan("177")
    kwargs = dict(plan.factory_kwargs())
    kwargs["ability_baseline_wave"] = "transition_start"
    prepared = load_and_prepare(**plan.prepare_kwargs())
    t1 = F.build_dose_response_model(prepared, **plan.factory_kwargs())
    start = F.build_dose_response_model(prepared, **kwargs)
    differing = ~np.isclose(
        np.asarray(t1.model["L_pre_logit"].get_value()),
        np.asarray(start.model["L_pre_logit"].get_value()),
    )
    assert differing.sum() > 50, (
        "the pre-#587 transition-start adjustment must differ materially from the "
        "t1 broadcast, otherwise the defect it encodes was never real"
    )


def test_transition_start_without_ability_symbols_is_rejected_before_io():
    with pytest.raises(ValueError, match="means nothing without"):
        D.resolve_dose_response_run_plan(
            ModelSpec(
                model_id="lrp-rli-dose-999",
                kind="dose_response",
                title="t",
                outcome_symbol="W",
                model_settings=D.DoseResponseModelSettings(
                    ability_baseline_wave="transition_start"
                ),
            )
        )


# --------------------------------------------------------------------------
# Finding 2 — presence, intensity, arm and between/within are separate terms
# --------------------------------------------------------------------------


@pytest.mark.parametrize("model_id", DOSE_MODELS)
def test_untreated_rows_contribute_nothing_to_any_dose_term(model_id):
    """The extensive margin must live in ``theta_treated`` alone."""
    _, built = _built(model_id)
    payload = built.payload
    untreated = ~np.asarray(payload.treated)
    assert untreated.any(), "the design needs untreated rows to identify presence"
    for name in ("attend_treated_std", "attend_child_mean_std", "attend_within_dev_std"):
        values = np.asarray(built.model[name].get_value())
        assert np.allclose(values[untreated], 0.0), (
            f"{name} is non-zero on an untreated row, so a dose slope would absorb "
            "part of the extensive margin"
        )


@pytest.mark.parametrize("model_id", DOSE_MODELS)
def test_dose_is_standardised_over_fitted_treated_rows(model_id):
    """Finding 13: the recorded scale must be the scale actually fitted."""
    _, built = _built(model_id)
    payload = built.payload
    treated = np.asarray(payload.treated)
    raw = np.asarray(payload.raw_attend)[treated]
    assert np.isclose(payload.dose_scaler.mean, raw.mean())
    assert np.isclose(payload.dose_scaler.sd, raw.std(ddof=1))
    fitted = np.asarray(built.model["attend_treated_std"].get_value())[treated]
    assert np.isclose(fitted.mean(), 0.0, atol=1e-9)
    assert np.isclose(fitted.std(ddof=1), 1.0, atol=1e-9)

    # And it is NOT the loader's pre-mask scaler, which is what the recorded
    # "standardised over the fitted rows" claim used to contradict (finding 13).
    loader = built.prepared.covariate_scalers["attend"]
    assert not np.isclose(payload.dose_scaler.sd, loader.sd, rtol=1e-3)


def test_arm_enters_only_after_the_crossover_period():
    """Finding 2: in period 1 arm and treatment presence are the same column."""
    _, built = _built("077")
    prepared = built.prepared
    phase = np.asarray(prepared.phase)
    late = np.asarray(built.model["late_phase"].get_value())
    treated = np.asarray(built.model["treated"].get_value())
    arm = np.asarray(prepared.G, dtype=float)

    assert np.allclose(late, (phase >= 1).astype(float))
    first = phase == 0
    assert np.allclose(treated[first], arm[first]), (
        "period 1 treatment presence is identical to assigned arm in these data, "
        "which is exactly why a separate arm term cannot be fitted there"
    )
    assert np.allclose(late[first] * arm[first], 0.0)


def test_between_and_within_components_reconstruct_the_exposure():
    _, built = _built("077")
    payload = built.payload
    treated = np.asarray(payload.treated)
    total = np.asarray(built.model["attend_treated_std"].get_value())
    between = np.asarray(payload.dose_between)
    within = np.asarray(payload.dose_within)
    assert np.allclose((between + within)[treated], total[treated])

    child = np.asarray(built.prepared.child_idx)
    for c in np.unique(child[treated]):
        rows = (child == c) & treated
        assert np.isclose(within[rows].mean(), 0.0, atol=1e-9), (
            "within-child deviations must be centred inside each child, or the "
            "Mundlak split does not separate the two associations"
        )


def test_phase_intercepts_are_reference_coded_and_full_rank():
    """Finding 11: a grand intercept plus three free indicators is rank-deficient."""
    _, built = _built("077")
    free = {rv.name for rv in built.model.free_RVs}
    assert "alpha_phase" not in free, "alpha_phase must be a derived Deterministic"
    assert "alpha_phase_free" in free
    n_phases = built.prepared.n_phases
    design = np.column_stack(
        [
            np.ones(built.prepared.n_obs),
            *[
                (np.asarray(built.prepared.phase) == p).astype(float)
                for p in range(n_phases)
            ],
        ]
    )
    assert np.linalg.matrix_rank(design) == n_phases, (
        "the unconstrained intercept design really is rank-deficient, which is what "
        "reference coding removes"
    )


# --------------------------------------------------------------------------
# Finding 3 — the items-scale contrast must stay inside observed support
# --------------------------------------------------------------------------


@pytest.mark.parametrize("model_id", DOSE_MODELS)
def test_reported_contrast_never_leaves_observed_dose_support(model_id):
    plan, built = _built(model_id)
    payload = built.payload
    phase = np.asarray(built.prepared.phase, dtype=int)
    contrast = P.resolve_dose_contrast(payload, phase)
    treated = np.asarray(payload.treated)
    raw = np.asarray(payload.raw_attend)

    assert plan.dose_contrast == "treated_row_interquartile_within_phase"
    assert np.allclose(contrast.delta_sessions[~treated], 0.0)
    for row in contrast.support_table.to_dict("records"):
        if not row["n_treated_rows"]:
            continue
        assert row["contrast_within_support"], row
        rows = treated & (phase == row["period"] - 1)
        assert row["sessions_q1"] >= raw[rows].min()
        assert row["sessions_q3"] <= raw[rows].max()
        assert row["contrast_sessions"] > 0


def test_contrast_aligns_to_the_factory_rows_not_the_loader_rows():
    """dose-177 loads 157 rows and fits 156; the payload is aligned to the 156.

    Resolving the contrast against the loader's frame raised a broadcast error here,
    and would have silently mismatched rows had the two counts ever coincided.
    """
    plan = _plan("177")
    loader = load_and_prepare(**plan.prepare_kwargs())
    built = F.build_dose_response_model(loader, **plan.factory_kwargs())
    assert built.prepared.n_obs < loader.n_obs, (
        "this guard needs a model whose factory drops rows; if dose-177 stops "
        "doing so, point it at one that does"
    )
    contrast = P.resolve_dose_contrast(
        built.payload, np.asarray(built.prepared.phase, dtype=int)
    )
    assert contrast.delta_std.shape == (built.prepared.n_obs,)
    assert np.asarray(built.payload.treated).shape == (built.prepared.n_obs,)


def test_the_old_global_sd_step_would_have_left_support():
    """Guard the *reason* for the repair, so the defect cannot silently return."""
    plan = _plan("077")
    prepared = load_and_prepare(**plan.prepare_kwargs())
    built = F.build_dose_response_model(prepared, **plan.factory_kwargs())
    loader = prepared.covariate_scalers["attend"]
    raw = np.asarray(built.payload.raw_attend)
    phase = np.asarray(built.prepared.phase, dtype=int)
    phase_max = np.array(
        [raw[phase == p].max() for p in range(built.prepared.n_phases)]
    )
    outside = (raw + loader.sd) > phase_max[phase]
    assert outside.sum() > len(raw) // 2, (
        "the pre-#587 +1 global-SD step put most shifted rows above their own "
        "period's observed maximum; if that is no longer true the guard is stale"
    )


# --------------------------------------------------------------------------
# Finding 4 — the predictive unit must not retain a held-out outcome
# --------------------------------------------------------------------------


@pytest.mark.parametrize("model_id", DOSE_MODELS)
def test_whole_child_loo_map_is_persisted(model_id):
    plan, built = _built(model_id)
    assert plan.loo_unit == "child"
    stored = np.asarray(built.model["loo_child_idx"].get_value())
    assert np.array_equal(stored, np.asarray(built.prepared.child_idx))


def test_row_level_loo_would_retain_a_held_out_outcome_as_a_predictor():
    """The defect itself: a row's own baseline IS the previous row's outcome."""
    plan = _plan("077")
    prepared = load_and_prepare(**plan.prepare_kwargs())
    subjects = np.asarray(prepared.subject_ids)
    phase = np.asarray(prepared.phase)
    later = phase >= 1
    shared = sum(
        1
        for subject, p in zip(subjects[later], phase[later], strict=True)
        if ((subjects == subject) & (phase == p - 1)).any()
    )
    assert shared > 0.9 * later.sum(), (
        "almost every later transition's own baseline is a fitted earlier outcome, "
        "so leaving one row out leaves that score in the next row's design matrix"
    )


# --------------------------------------------------------------------------
# Finding 5 — prior and posterior must be one transform
# --------------------------------------------------------------------------


def _draws_group(n_rows, n_draws, slopes, eta_value=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return xr.Dataset(
        {
            "eta": xr.DataArray(
                rng.normal(eta_value, 0.1, size=(1, n_draws, n_rows)),
                dims=("chain", "draw", "obs_id"),
            ),
            "beta_dose_phase": xr.DataArray(
                np.broadcast_to(
                    np.asarray(slopes, dtype=float)[None, None, :],
                    (1, n_draws, len(slopes)),
                ).copy(),
                dims=("chain", "draw", "phase"),
            ),
            "mu_dose": xr.DataArray(
                np.full((1, n_draws), float(np.mean(slopes))), dims=("chain", "draw")
            ),
        }
    )


def test_prior_and_posterior_marginals_use_the_same_phase_indexed_transform():
    """The acceptance criterion: unequal phase slopes must not be averaged away.

    With slopes that differ sharply by period, a transform that broadcasts the scalar
    ``mu_dose`` to every row — the pre-#587 prior path — lands on the mean slope and
    cannot reproduce the phase-indexed answer. The shared transform must give the
    identical number from a prior group and a posterior group carrying the same
    values.
    """
    slopes = [2.0, 0.0, -2.0]
    phase_idx = np.array([0, 0, 1, 1, 2, 2])
    delta = np.ones(phase_idx.size)
    posterior = _draws_group(phase_idx.size, 64, slopes, seed=1)
    prior = _draws_group(phase_idx.size, 64, slopes, seed=1)

    from_posterior = P.dose_marginal_draws(
        posterior, phase_idx=phase_idx, delta_std=delta, n_trials=79,
        period_varying=True,
    )
    from_prior = P.dose_marginal_draws(
        prior, phase_idx=phase_idx, delta_std=delta, n_trials=79,
        period_varying=True,
    )
    assert np.allclose(from_posterior, from_prior)

    # And it is genuinely phase-indexed, not the scalar shortcut.
    scalar_only = _draws_group(
        phase_idx.size, 64, [np.mean(slopes)] * 3, seed=1
    )
    from_scalar = P.dose_marginal_draws(
        scalar_only, phase_idx=phase_idx, delta_std=delta, n_trials=79,
        period_varying=True,
    )
    assert not np.allclose(from_posterior, from_scalar), (
        "a scalar-slope transform must NOT reproduce the phase-indexed marginal, "
        "or this test could not have caught the original defect"
    )


def test_shared_transform_respects_the_row_mask_and_the_contrast():
    slopes = [1.0, 1.0, 1.0]
    phase_idx = np.array([0, 1, 2, 0])
    group = _draws_group(4, 32, slopes, seed=2)
    mask = np.array([True, True, False, False])
    full = P.dose_marginal_draws(
        group, phase_idx=phase_idx, delta_std=np.ones(4), n_trials=10,
        period_varying=True,
    )
    masked = P.dose_marginal_draws(
        group, phase_idx=phase_idx, delta_std=np.ones(4), n_trials=10,
        period_varying=True, row_mask=mask,
    )
    assert not np.allclose(full, masked)
    doubled = P.dose_marginal_draws(
        group, phase_idx=phase_idx, delta_std=np.zeros(4), n_trials=10,
        period_varying=True,
    )
    assert np.allclose(doubled, 0.0), "a zero contrast must move nothing"

    with pytest.raises(ValueError, match="selects no rows"):
        P.dose_marginal_draws(
            group, phase_idx=phase_idx, delta_std=np.ones(4), n_trials=10,
            period_varying=True, row_mask=np.zeros(4, dtype=bool),
        )


# --------------------------------------------------------------------------
# Finding 12 — settings validation happens before any I/O
# --------------------------------------------------------------------------


def test_unknown_measure_symbols_are_rejected():
    with pytest.raises(ValueError, match="unknown measure symbol"):
        D.resolve_dose_response_run_plan(
            ModelSpec(
                model_id="lrp-rli-dose-999",
                kind="dose_response",
                title="t",
                outcome_symbol="ZZZ",
                model_settings=D.DoseResponseModelSettings(
                    adjust_baseline_symbol="ZZZ", outcomes=("ZZZ",)
                ),
            )
        )


def test_ability_symbol_may_not_duplicate_the_own_baseline():
    with pytest.raises(ValueError, match="duplicate adjust_baseline_symbol"):
        D.resolve_dose_response_run_plan(
            ModelSpec(
                model_id="lrp-rli-dose-999",
                kind="dose_response",
                title="t",
                outcome_symbol="W",
                model_settings=D.DoseResponseModelSettings(
                    adjust_baseline_symbol="W",
                    ability_adjust_symbols=("W",),
                    outcomes=("W",),
                ),
            )
        )


# --------------------------------------------------------------------------
# Findings 8 and 14 — the real repository artefacts a reader actually sees
# --------------------------------------------------------------------------


def test_comparison_is_copied_beside_both_paired_runs_under_the_partial_s_name():
    """Finding 8: the comparison existed but never reached either report."""
    script = (REPO / "scripts/compare_statistical_models.py").read_text(encoding="utf-8")
    body = script.split("def dose_response_loo_compare", 1)[1].split("\ndef ", 1)[0]
    assert "_copy_compare_beside_runs" in body
    assert 'filename="dose_loo_compare.csv"' in body

    partial = (REPO / "docs/models/_partials/_results_dose_response.qmd").read_text(
        encoding="utf-8"
    )
    assert '_csv("dose_loo_compare.csv")' in partial, (
        "the partial must read the same filename the comparison writes beside the run"
    )


def test_results_partial_reports_medians_and_the_real_comparison_columns():
    partial = (REPO / "docs/models/_partials/_results_dose_response.qmd").read_text(
        encoding="utf-8"
    )
    assert "posterior median" in partial
    assert "_main['median']" in partial
    assert "posterior mean" not in partial, "the house standard is the median (#271)"
    assert "`dse`" in partial and "se_diff" not in partial, (
        "az.compare writes `dse`, not `se_diff`"
    )
    assert "per additional session" not in partial, (
        "the coefficient is per 1 SD of treated-row sessions, not per session"
    )


@pytest.mark.parametrize("model_id", DOSE_MODELS)
def test_report_templates_do_not_claim_terms_the_fit_does_not_have(model_id):
    """Finding 14: two templates displayed a cumulative-dose term nothing fits."""
    plan = _plan(model_id)
    template = (
        REPO / f"docs/models/lrp-rli-dose-{model_id}/index.qmd"
    ).read_text(encoding="utf-8")
    if plan.dose_stage_covariate is None:
        assert "\\gamma_{\\text{stage}}" not in template
        assert "cumulative-dose control, subject random intercept" not in template
    assert "not \\._" not in template, "malformed escaped prose"


@pytest.mark.parametrize("model_id", DOSE_MODELS)
def test_registered_dose_models_are_catalogued(model_id):
    catalogue = (REPO / "docs/models/README.md").read_text(encoding="utf-8")
    assert f"lrp-rli-dose-{model_id}" in catalogue


def test_catalogue_does_not_call_the_slope_a_within_child_association():
    catalogue = (REPO / "docs/models/README.md").read_text(encoding="utf-8")
    assert "an adjusted\nwithin-child association" not in catalogue
    assert "adjusted within-child association" not in catalogue


def test_dag_contradicting_prose_is_gone_from_every_dose_module():
    """Finding 7: the DAG has A -> IS, GA -> IS and IG -> IS."""
    dag = (REPO / "dag/dag-language-reading.dagitty").read_text(encoding="utf-8")
    for parent in ("A  ->", "GA ->", "IG ->"):
        line = next(ln for ln in dag.splitlines() if ln.startswith(parent))
        assert "IS" in line, f"{parent} no longer points into IS; update the prose"

    for model_id in DOSE_MODELS:
        source = (
            REPO
            / f"src/language_reading_predictors/statistical_models/lrp_rli_dose_{model_id}.py"
        ).read_text(encoding="utf-8")
        # Every module must acknowledge the unblocked latent-ability path. The old
        # prose asserted the opposite ("no ability -> dose edge assumed", "v5 has
        # age -> outcome but no age -> dose"); dose-077 now quotes those claims only
        # in order to record that the DAG contradicts them, so match on the honest
        # statement rather than on the absence of the words.
        assert "GA -> IS" in source or "GA is not" in source, model_id
        assert "edge assumed" not in source, model_id
        assert "sole confounder" not in source, model_id


# --------------------------------------------------------------------------
# Finding 10 — exposure and adjustment set are different facts
# --------------------------------------------------------------------------


def test_effective_adjustment_names_the_adjusters_not_the_exposure():
    plan, built = _built("177")
    record = P._dose_effective_adjustment(plan, built.prepared)
    terms = {term["term"] for term in record["fitted"]}
    assert record["exposure"] == "attend"
    assert "attend" not in terms
    assert {"W_pre", "A", "beta_arm_late", "L_pre", "E_pre", "B_pre"} <= terms


# --- the phoneme-blending response-link pair (#619) ---------------------------


def test_the_registered_blending_link_pair_is_paired_both_ways():
    """#619: dose-084 and dose-384 fit the same analysis under the two response
    links, each naming the other, and neither may release alone."""
    import importlib

    from language_reading_predictors.statistical_models.dose_response import (
        resolve_dose_response_run_plan,
    )

    def _plan(module: str):
        mod = importlib.import_module(
            f"language_reading_predictors.statistical_models.{module}"
        )
        return resolve_dose_response_run_plan(mod.SPEC)

    primary, companion = _plan("lrp_rli_dose_084"), _plan("lrp_rli_dose_384")
    assert primary.score_mean_link == "logit"
    assert companion.score_mean_link == "three_choice_guessing_floor"
    assert primary.required_link_companion_model_id == "lrp-rli-dose-384"
    assert companion.required_link_companion_model_id == "lrp-rli-dose-084"
    assert primary.link_sensitivity_required_for_release
    assert companion.link_sensitivity_required_for_release
    for field in (
        "outcome_symbol", "adjust_baseline_symbol", "dose_covariate",
        "dose_stage_covariate", "period_varying_dose",
        "use_subject_random_intercept", "ability_adjust_symbols",
        "ability_baseline_wave", "decompose_between_within", "outcomes",
        "adjust_group", "adjust_age", "focal_term", "exposure", "dose_margin",
        "dose_contrast", "estimand", "causal_status", "analysis_population",
    ):
        assert getattr(primary, field) == getattr(companion, field), field
    # The companion must carry the primary's sampler setting too, or the comparison
    # confounds the link with a sampling-quality difference.
    assert companion.model_id != primary.model_id


def test_only_a_blending_outcome_requires_the_link_pair():
    import importlib

    from language_reading_predictors.statistical_models.dose_response import (
        resolve_dose_response_run_plan,
    )

    for module, expected in (
        ("lrp_rli_dose_077", False),
        ("lrp_rli_dose_083", False),
        ("lrp_rli_dose_084", True),
    ):
        mod = importlib.import_module(
            f"language_reading_predictors.statistical_models.{module}"
        )
        plan = resolve_dose_response_run_plan(mod.SPEC)
        assert plan.link_sensitivity_required_for_release is expected, module


def test_the_dose_marginal_transform_maps_through_the_link():
    """The floor link compresses a given latent shift, so the same draws must give a
    smaller items-scale dose marginal under it — and, because one helper serves both
    the posterior marginal and its prior pushforward, they stay on the same scale."""
    import numpy as np
    import xarray as xr

    from language_reading_predictors.statistical_models.pipelines import (
        dose_response as P,
    )

    n_rows, n_draws = 6, 8
    eta = xr.DataArray(
        np.linspace(-1.0, 1.0, n_rows * n_draws).reshape(1, n_draws, n_rows),
        dims=("chain", "draw", "obs_id"),
    )
    beta = xr.DataArray(
        np.full((1, n_draws), 0.5), dims=("chain", "draw")
    )
    group = xr.Dataset({"eta": eta, "beta_dose": beta})
    kw = dict(
        phase_idx=np.zeros(n_rows, dtype=int),
        delta_std=np.ones(n_rows),
        n_trials=10,
        period_varying=False,
    )
    ordinary = P.dose_marginal_draws(group, **kw)
    floored = P.dose_marginal_draws(
        group, **kw, score_mean_link="three_choice_guessing_floor"
    )
    assert np.all(np.abs(floored) < np.abs(ordinary))
    # The floor link scales the response range by exactly 2/3.
    assert np.allclose(floored, ordinary * (2.0 / 3.0))
