# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The DiD family's #576 estimand, release-gate and marginalisation contracts.

Three things this file pins that nothing else did:

* the release gate's sign-stability clause follows the fit's **published** estimand,
  not the coefficient the sweep happens to vary (finding 1);
* a graded phoneme-blending DiD fit cannot release without its guessing-floor twin
  (finding 2);
* a run plan that changes the fitted *equation* invalidates attached sweep evidence
  even though identity, data hash and row counts are unchanged (finding 6).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from language_reading_predictors.statistical_models.blending_sensitivity import (
    evaluate_did_blending_link_pair,
)
from language_reading_predictors.statistical_models.did import (
    DID_BLENDING_COMPANION_MODEL_ID,
    DID_BLENDING_PRIMARY_MODEL_ID,
    DiDModelSettings,
    did_run_plan_digest,
    resolve_did_run_plan,
)
from language_reading_predictors.statistical_models.release import (
    _standard_sweep_evidence,
    sweep_sign_column,
)
from language_reading_predictors.statistical_models.sensitivity import (
    STANDARD_SENSITIVITY_FILENAME,
    _STANDARD_REQUIRED_COLUMNS,
)


# --- finding 1: the gate follows the published estimand -----------------------


def _dose_plan(**overrides) -> dict:
    """A did-007-style resolved plan as it is persisted to ``config.json``."""
    from language_reading_predictors.statistical_models.context import ModelSpec

    spec = ModelSpec(
        model_id="lrp-rli-did-007",
        kind="did",
        title="t",
        outcome_symbol="L",
        family="did",
        design="d",
        estimand_type="association",
        causal_status="none",
        model_settings=DiDModelSettings(
            outcomes=("L",), periods=(0, 1), dose=True, period_varying_dose=True
        ),
    )
    plan = resolve_did_run_plan(spec).as_dict()
    plan.update(overrides)
    return plan


def _sweep_fit_dir(
    tmp_path: Path,
    *,
    plan: dict,
    tau_logit_means: tuple[float, ...],
    items_means: tuple[float, ...],
    run_plan_digest_in_rows: str | None = None,
) -> Path:
    """A fit directory with a self-consistent, trace-backed treatment-prior sweep.

    Only the two sign columns and the run-plan binding vary between tests; every
    other clause of ``_standard_sweep_evidence`` (columns, sigma count, convergence,
    config/trace digests, installed cell traces) is satisfied so that a refusal can
    only come from the clause under test.
    """
    from language_reading_predictors.statistical_models.sensitivity import sha256_file

    directory = tmp_path / "lrp-rli-did-007-reporting"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(
        json.dumps(
            {
                "model_id": "lrp-rli-did-007",
                "kind": "did",
                "outcome_symbol": "L",
                "config_name": "reporting",
                "resolved_run_plan": plan,
            }
        ),
        encoding="utf-8",
    )
    (directory / "trace.nc").write_bytes(b"not a real trace, only hashed here")
    config_digest = sha256_file(directory / "config.json")
    trace_digest = sha256_file(directory / "trace.nc")

    rows = []
    for index, (sigma, tau_mean, items_mean) in enumerate(
        zip((0.5, 1.0, 1.5), tau_logit_means, items_means, strict=True)
    ):
        cell = directory / f"cell_{index}.nc"
        cell.write_bytes(f"cell {index}".encode())
        row = dict.fromkeys(_STANDARD_REQUIRED_COLUMNS, "")
        row.update(
            config="reporting",
            outcome="L",
            tau_sigma=sigma,
            converged=True,
            tau_logit_mean=tau_mean,
            items_mean=items_mean,
            primary_model_id="lrp-rli-did-007",
            primary_config_sha256=config_digest,
            primary_trace_sha256=trace_digest,
            trace_file=cell.name,
            trace_sha256=sha256_file(cell),
        )
        if run_plan_digest_in_rows is not None:
            row["primary_run_plan_sha256"] = run_plan_digest_in_rows
        rows.append(row)
    pd.DataFrame(rows).to_csv(directory / STANDARD_SENSITIVITY_FILENAME, index=False)
    return directory


def test_sweep_sign_column_switches_on_the_declared_estimand_scale():
    natural = {"resolved_run_plan": _dose_plan()}
    assert sweep_sign_column(natural)[0] == "items_mean"
    # A stored fit written before the field existed keeps the coefficient column,
    # so its release decision is reproducible without a refit.
    legacy = {"resolved_run_plan": {"dose": True, "period_varying": True}}
    assert sweep_sign_column(legacy)[0] == "tau_logit_mean"
    assert sweep_sign_column(None)[0] == "tau_logit_mean"


def test_gate_follows_the_published_marginal_not_mu_dose(tmp_path):
    """#576 finding 1 / acceptance criterion 2, in the case where the two differ.

    ``beta_dose_phase[p] = mu_dose + sigma_dose * z[p]``: the hierarchical centre and
    the weighted realised-slope marginal are different quantities, and the items
    transform is nonlinear over unequal per-period row counts, so their signs can
    genuinely diverge. The gate must track the number the report publishes.
    """
    plan = _dose_plan()
    digest = plan["run_plan_digest"]

    # mu_dose flips sign across the grid; the published marginal does not.
    stable_marginal = _sweep_fit_dir(
        tmp_path / "a",
        plan=plan,
        tau_logit_means=(-0.10, 0.05, 0.20),
        items_means=(0.31, 0.44, 0.52),
        run_plan_digest_in_rows=digest,
    )
    ready, reason = _standard_sweep_evidence(stable_marginal, "L")
    assert ready, reason

    # The mirror image: mu_dose is sign-stable, the published marginal is not.
    unstable_marginal = _sweep_fit_dir(
        tmp_path / "b",
        plan=plan,
        tau_logit_means=(0.10, 0.15, 0.20),
        items_means=(-0.22, 0.05, 0.41),
        run_plan_digest_in_rows=digest,
    )
    ready, reason = _standard_sweep_evidence(unstable_marginal, "L")
    assert not ready
    assert "treated_row_dose_marginal" in reason
    assert "changes sign" in reason


def test_legacy_dose_plan_still_gates_on_the_coefficient(tmp_path):
    """A stored pre-#576 fit re-decides exactly as it did, without a refit."""
    legacy_plan = {"dose": True, "period_varying": True}
    directory = _sweep_fit_dir(
        tmp_path,
        plan=legacy_plan,
        tau_logit_means=(0.10, 0.15, 0.20),
        items_means=(-0.22, 0.05, 0.41),
    )
    ready, _ = _standard_sweep_evidence(directory, "L")
    assert ready


# --- finding 6: run-plan binding ----------------------------------------------


def test_sweep_bound_to_a_different_run_plan_fails_closed(tmp_path):
    plan = _dose_plan()
    directory = _sweep_fit_dir(
        tmp_path,
        plan=plan,
        tau_logit_means=(0.10, 0.15, 0.20),
        items_means=(0.31, 0.44, 0.52),
        run_plan_digest_in_rows="0" * 64,
    )
    ready, reason = _standard_sweep_evidence(directory, "L")
    assert not ready
    assert "different resolved run plan" in reason


def test_sweep_without_run_plan_binding_fails_closed_for_a_new_plan(tmp_path):
    plan = _dose_plan()
    directory = _sweep_fit_dir(
        tmp_path,
        plan=plan,
        tau_logit_means=(0.10, 0.15, 0.20),
        items_means=(0.31, 0.44, 0.52),
        run_plan_digest_in_rows=None,
    )
    ready, reason = _standard_sweep_evidence(directory, "L")
    assert not ready
    assert "predates run-plan binding" in reason


def test_run_plan_digest_ignores_prose_and_honours_defaults():
    """The digest must survive a wording revision and a newly added field."""
    plan = _dose_plan()
    reworded = dict(plan)
    reworded["estimand"] = "completely different prose"
    reworded["causal_status"] = "reworded too"
    reworded["settings_source"] = "typed"
    assert did_run_plan_digest(reworded) == did_run_plan_digest(plan)

    # A stored plan predating a field digests as one that takes its default...
    older = {k: v for k, v in plan.items() if k != "kappa_prior_family"}
    assert did_run_plan_digest(older) == did_run_plan_digest(plan)
    # ...but a real modelling change does not.
    changed = dict(plan, kappa_prior_family="halfnormal_inverse_sqrt")
    assert did_run_plan_digest(changed) != did_run_plan_digest(plan)


@pytest.mark.parametrize(
    "field,value",
    [
        ("likelihood", "bernoulli_offfloor"),
        ("use_intercept_anchor", False),
        ("use_age", False),
        ("use_child_re", False),
        ("tau_t2_prior_sigma", 1.0),
        ("score_mean_link", "three_choice_guessing_floor"),
        ("arm_gap_t1_prior_sigma", 1.0),
    ],
)
def test_every_non_swept_equation_field_changes_the_digest(field, value):
    """The fields the review named as invisible to the old bindings (#576 finding 6).

    None of these move the model id, the outcome, the data hash, the row count or the
    arm totals — the entire pre-#576 binding — so each is a way a sweep generated
    under a newer plan could have lifted an older primary's gate.
    """
    from language_reading_predictors.statistical_models.context import ModelSpec

    spec = ModelSpec(
        model_id="lrp-rli-did-001",
        kind="did",
        title="t",
        outcome_symbol="W",
        family="did",
        design="d",
        estimand_type="mixed",
        causal_status="c",
        model_settings=DiDModelSettings(outcomes=("W",)),
    )
    base = resolve_did_run_plan(spec).as_dict()
    assert did_run_plan_digest(dict(base, **{field: value})) != did_run_plan_digest(base)


# --- finding 2: the phoneme-blending link pair --------------------------------


def _blending_fit_dir(
    root: Path,
    model_id: str,
    *,
    link: str,
    gate_passed: bool = True,
    data_sha256: str = "d" * 64,
    rows_digest: str = "r" * 64,
    n_obs: int = 162,
) -> Path:
    directory = root / f"{model_id}-reporting"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "kind": "did",
                "outcome_symbol": "B",
                "config_name": "reporting",
                "n_obs": n_obs,
                "data_sha256": data_sha256,
                "fitted_data_identity": {"digest": rows_digest},
                "resolved_run_plan": {
                    "score_mean_link": link,
                    "link_sensitivity_required_for_release": True,
                    "run_plan_digest": "p" * 64,
                },
            }
        ),
        encoding="utf-8",
    )
    (directory / "diagnostics_summary.json").write_text(
        json.dumps(
            {
                "passed": gate_passed,
                "checks": {
                    "rhat": gate_passed,
                    "ess": gate_passed,
                    "divergences": gate_passed,
                    "bfmi": gate_passed,
                },
                # The gate re-derives every check from the raw numbers, so a fixture
                # cannot simply declare a pass.
                "divergences": 0 if gate_passed else 4,
                "max_rhat": 1.001 if gate_passed else 1.05,
                "min_ess": 4000.0 if gate_passed else 90.0,
                "bfmi_per_chain": [0.8, 0.9] if gate_passed else [0.2, 0.9],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "tau_t2_items_median": 0.9 if link == "logit" else 0.5,
                "tau_t2_items_lo": 0.1,
                "tau_t2_items_hi": 1.7,
                "prob_tau_t2_pos": 0.95,
            }
        ]
    ).to_csv(directory / "did_summary.csv", index=False)
    return directory


def test_blending_pair_is_ready_when_both_sides_are_present_and_gated(tmp_path):
    primary = _blending_fit_dir(tmp_path, DID_BLENDING_PRIMARY_MODEL_ID, link="logit")
    _blending_fit_dir(
        tmp_path,
        DID_BLENDING_COMPANION_MODEL_ID,
        link="three_choice_guessing_floor",
    )
    status = evaluate_did_blending_link_pair(primary)
    assert status["required"] and status["ready"], status
    assert set(status["cards"]) == {
        DID_BLENDING_PRIMARY_MODEL_ID,
        DID_BLENDING_COMPANION_MODEL_ID,
    }


def test_blending_primary_cannot_release_without_its_twin(tmp_path):
    primary = _blending_fit_dir(tmp_path, DID_BLENDING_PRIMARY_MODEL_ID, link="logit")
    status = evaluate_did_blending_link_pair(primary)
    assert status["required"] and not status["ready"]
    assert DID_BLENDING_COMPANION_MODEL_ID in status["reason"]


@pytest.mark.parametrize(
    "companion_kwargs,fragment",
    [
        ({"gate_passed": False}, "convergence gate"),
        ({"data_sha256": "e" * 64}, "dataset"),
        ({"rows_digest": "s" * 64}, "fitted rows"),
        ({"n_obs": 150}, "row count"),
        ({"link": "logit"}, "opposite score-mean links"),
    ],
    ids=["ungated", "other-data", "other-rows", "other-n", "same-link"],
)
def test_blending_pair_fails_closed_on_a_mismatched_twin(
    tmp_path, companion_kwargs, fragment
):
    primary = _blending_fit_dir(tmp_path, DID_BLENDING_PRIMARY_MODEL_ID, link="logit")
    companion_kwargs.setdefault("link", "three_choice_guessing_floor")
    _blending_fit_dir(
        tmp_path, DID_BLENDING_COMPANION_MODEL_ID, **companion_kwargs
    )
    status = evaluate_did_blending_link_pair(primary)
    assert status["required"] and not status["ready"]
    assert fragment in status["reason"], status["reason"]


def test_release_withholds_a_blending_did_fit_with_no_twin(tmp_path):
    """The gate itself, not only the evaluator, refuses the unpaired fit."""
    from language_reading_predictors.statistical_models.release import (
        _blending_pair_release_failures,
    )

    primary = _blending_fit_dir(tmp_path, DID_BLENDING_PRIMARY_MODEL_ID, link="logit")
    config = json.loads((primary / "config.json").read_text())
    failures = _blending_pair_release_failures(primary, config)
    assert failures and "phoneme-blending link pair" in failures[0]


def test_an_unregistered_graded_b_did_fit_fails_closed(tmp_path):
    """A future graded B arm-by-wave fit outside the pair must not publish unpaired."""
    directory = _blending_fit_dir(tmp_path, "lrp-rli-did-903", link="logit")
    status = evaluate_did_blending_link_pair(directory)
    assert status["required"] and not status["ready"]
    assert "not one of the registered DiD blending fits" in status["reason"]


def test_registered_blending_pair_resolves_from_the_model_modules():
    """Both sides declare the pairing and name each other, not a hard-coded string."""
    import importlib

    for model_id, expected_link, expected_twin in (
        (DID_BLENDING_PRIMARY_MODEL_ID, "logit", DID_BLENDING_COMPANION_MODEL_ID),
        (
            DID_BLENDING_COMPANION_MODEL_ID,
            "three_choice_guessing_floor",
            DID_BLENDING_PRIMARY_MODEL_ID,
        ),
    ):
        module = importlib.import_module(
            "language_reading_predictors.statistical_models."
            + model_id.replace("-", "_")
        )
        plan = resolve_did_run_plan(module.SPEC)
        assert plan.score_mean_link == expected_link
        assert plan.link_sensitivity_required_for_release
        assert plan.required_link_companion_model_id == expected_twin


# --- finding 5: the LRPDID13 trajectory is fully marginalised ------------------


def test_group_trajectory_integrates_a_second_random_effect():
    """#576 finding 5: ``v_delta`` must be removed and integrated, not left in eta.

    Constructed so the answer is known: the waitlist t3 rows carry a large positive
    fitted deviation. Leaving it in ``eta`` reports the *fitted children's* level;
    integrating over ``Normal(0, sigma_delta)`` reports the population one, which for
    a concave-then-convex inverse logit at these values is materially lower.
    """
    from language_reading_predictors.statistical_models.trajectory_plots import (
        marginal_cell_probabilities,
    )

    n_rows, n_draws = 8, 64
    eta = np.zeros((n_rows, n_draws))
    arm = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    wave = np.asarray([0, 1, 2, 2, 0, 1, 2, 2])
    extra_mask = (arm == 0) & (wave == 2)
    extra_rows = np.zeros((n_rows, n_draws))
    extra_rows[extra_mask] = 1.5
    eta = eta + extra_rows * extra_mask[:, None]
    sigma_delta = np.full(n_draws, 0.8)

    conditional = marginal_cell_probabilities(eta, arm=arm, wave=wave)
    integrated = marginal_cell_probabilities(
        eta,
        arm=arm,
        wave=wave,
        extra_effect_rows=extra_rows,
        extra_effect_sd=sigma_delta,
        extra_effect_mask=extra_mask,
    )
    # The waitlist t3 cell is the only one that moves, and it moves towards 0.5.
    assert conditional[(0, 2)].mean() > 0.80
    assert integrated[(0, 2)].mean() == pytest.approx(0.5, abs=0.02)
    for cell in ((0, 0), (0, 1), (1, 0), (1, 1), (1, 2)):
        assert integrated[cell].mean() == pytest.approx(
            conditional[cell].mean(), abs=1e-9
        )


def test_group_trajectory_applies_the_score_mean_link():
    from language_reading_predictors.statistical_models.trajectory_plots import (
        marginal_cell_probabilities,
    )

    eta = np.zeros((2, 4))
    arm = np.asarray([0, 1])
    wave = np.asarray([0, 0])
    plain = marginal_cell_probabilities(eta, arm=arm, wave=wave)
    floored = marginal_cell_probabilities(
        eta, arm=arm, wave=wave, score_mean_link="three_choice_guessing_floor"
    )
    assert plain[(0, 0)].mean() == pytest.approx(0.5)
    assert floored[(0, 0)].mean() == pytest.approx(1 / 3 + (2 / 3) * 0.5)
