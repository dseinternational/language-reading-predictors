# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Waitlist-crossover / difference-in-differences orchestration (``kind="did"``).

``fit_did`` fits the arm-by-wave family: bounded t1/t2/t3 levels with separate
immediate-minus-waitlist gaps. ``tau_t2`` is the randomised immediate-versus-not-yet
assignment contrast; ``arm_gap_t3`` is a *different* randomised contrast — assignment
to the early-start rather than the delayed-start treatment schedule — and
``delta_crossover`` is the change between the two, never an identified catch-up
mechanism (#576 finding 3). The dose companions add treated-centred session intensity
and report their slopes as observational associations — through the same summary
writer the dose-response family uses, imported from :mod:`.dose_response`, so both
dose designs publish the same named marginal on the same treated-row population.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from language_reading_predictors.models._reporting import (
    metrics_table,
    print_table,
    section_header,
)
from language_reading_predictors.statistical_models import (
    diagnostics as _diag,
    factories as _factories,
    priors as _priors,
    reporting as _report,
)
from language_reading_predictors.statistical_models.artifacts import (
    guard_optional,
    save_table,
)
from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
    make_context,
)
from language_reading_predictors.statistical_models.did import (
    DiDRunPlan,
    resolve_did_run_plan,
)
from language_reading_predictors.statistical_models.figure_artifacts import (
    save_did_cell_ppc_plot,
    save_forest_plot,
    write_child_fit,
    write_group_trajectory,
    write_predicted_scores,
)
from language_reading_predictors.statistical_models.fitted_payloads import (
    DidArmWavePayload,
    DidDosePayload,
    FittedPayload,
)
from language_reading_predictors.statistical_models.pipelines.dose_response import (
    write_dose_slope_summary,
)
from language_reading_predictors.statistical_models.preprocessing import (
    PreparedData,
    load_and_prepare,
)
from language_reading_predictors.statistical_models.publication import (
    print_header,
    render_model_graph,
)
from language_reading_predictors.statistical_models.runtime import (
    attach_built,
    finalize_report,
    require_spec,
    shared_stages,
    write_run_metadata,
)
from language_reading_predictors.statistical_models.stages import PrimaryFitPlan


# Negligible-heterogeneity threshold on the logit scale for the "does the
# between-child waitlist catch-up SD concentrate near zero?" diagnostic (#230
# §4a): an order of magnitude below the delta / tau prior scale (Normal(0, 0.5)).
_SIGMA_DELTA_ROPE = 0.1


def _did_heterogeneity_summary(trace: Any, *, ci_prob: float) -> dict[str, float]:
    """Between-waitlist-child SD of post-crossover catch-up near zero.

    Reports ``sigma_delta`` (median + equal-tailed CI on the logit scale), the ROPE-style
    ``P(sigma_delta < delta_het)`` "concentrates near zero" probability, and the prior mass
    below the same threshold under the HalfNormal(0.5) prior — so the reader can see the
    data moved it (#230 §2/§4a). A near-zero posterior is the clean "no reliable
    between-child variation" result. This is exploratory variation in the waitlist
    arm's t3 catch-up association, not treatment-effect heterogeneity.
    """
    sd = np.asarray(trace.posterior["sigma_delta"].values).reshape(-1)
    lo, hi = (1 - ci_prob) / 2, 1 - (1 - ci_prob) / 2
    # Prior mass below the threshold read straight from the sigma_delta prior constructor
    # (not a re-typed scale), so prior_P can't silently drift if the prior changes (#294).
    prior_below = float(_priors.sigma_delta_prior().cdf(_SIGMA_DELTA_ROPE))
    key = f"P(sigma_delta<{_SIGMA_DELTA_ROPE})"
    return {
        "sigma_delta_median": float(np.median(sd)),
        "sigma_delta_ci_low": float(np.quantile(sd, lo)),
        "sigma_delta_ci_high": float(np.quantile(sd, hi)),
        key: float(np.mean(sd < _SIGMA_DELTA_ROPE)),
        f"prior_{key}": float(prior_below),
    }


def _waitlist_crossover_index(
    built: _factories.BuiltModel[FittedPayload],
) -> np.ndarray:
    """Row -> ``waitlist_child`` position, reproducing the factory's ``safe_idx``.

    ``build_did_model`` indexes ``v_delta`` by a dense position over the waitlist
    children who have a t3 row, with every other row pointed at position 0 and
    multiplied by a zero mask. The trajectory helper has to remove exactly those
    fitted deviations again, so the mapping is rebuilt here from the fitted rows
    rather than re-derived by a second, subtly different rule (#576 finding 5).
    """
    prepared = built.prepared
    waitlist_subjects = np.unique(
        prepared.subject_ids[(prepared.G == 0) & (prepared.phase == 2)]
    )
    lookup = {subject: position for position, subject in enumerate(waitlist_subjects)}
    return np.asarray(
        [lookup.get(subject, 0) for subject in prepared.subject_ids], dtype=int
    )


def _did_analysis_contract(
    ctx: StatisticalFitContext,
    plan: DiDRunPlan,
    built: _factories.BuiltModel[FittedPayload],
    *,
    dose: bool,
    loaded_prepared: PreparedData,
) -> dict[str, Any]:
    """Persist the exact fitted rows and return auditable DiD design metadata."""
    prepared = built.prepared
    payload = built.require_payload(
        DidDosePayload if dose else DidArmWavePayload,
        family="did",
    )
    row_ids = np.asarray(payload.analysis_row_ids, dtype=str)
    phase_name = "period" if dose else "wave"
    labels = (
        np.asarray([f"P{int(p) + 1}" for p in prepared.phase])
        if dose
        else np.asarray([f"t{int(p) + 1}" for p in prepared.phase])
    )
    manifest = pd.DataFrame(
        {
            "row_id": row_ids,
            "subject_id": prepared.subject_ids.astype(str),
            "child_idx": prepared.child_idx.astype(int),
            phase_name: labels,
            "phase_code": prepared.phase.astype(int),
            "arm": np.where(prepared.G == 1, "immediate", "waitlist"),
            "G": prepared.G.astype(int),
        }
    )
    if dose:
        dose_payload = built.require_payload(DidDosePayload, family="did dose")
        manifest["treated"] = np.asarray(dose_payload.treated, dtype=int)
        manifest["sessions_raw"] = np.asarray(dose_payload.raw_attend, dtype=float)
        manifest["dose_treated_std"] = np.asarray(
            dose_payload.dose_treated_std, dtype=float
        )
    save_table(ctx, "analysis_rows", manifest, register=False)

    counts = (
        manifest.groupby([phase_name, "arm"], observed=True)
        .size()
        .rename("n")
        .reset_index()
        .to_dict("records")
    )
    design_codes = (0, 1) if dose else (0, 1, 2)
    design_eligible = int(np.isin(loaded_prepared.phase, design_codes).sum())
    contract: dict = {
        "design": payload.design,
        # The one named quantity this fit publishes, so a reader of config.json
        # never has to infer it from which CSV happens to exist (#576 finding 1).
        "focal_estimand": {
            "name": plan.focal_estimand,
            "scale": plan.focal_estimand_scale,
            "artifact": plan.focal_estimand_artifact,
            "swept_coefficient": plan.effect_term,
        },
        "run_plan_digest": plan.run_plan_digest,
        "analysis_row_manifest": "analysis_rows.csv",
        "analysis_row_sha256": hashlib.sha256(
            "\n".join(row_ids).encode("utf-8")
        ).hexdigest(),
        "analysis_row_count": int(len(row_ids)),
        "loaded_row_count": int(loaded_prepared.n_obs),
        "loader_dropped_rows": int(loaded_prepared.dropped_rows),
        "design_excluded_rows": int(loaded_prepared.n_obs - design_eligible),
        "factory_missing_excluded_rows": int(design_eligible - len(row_ids)),
        "fitted_n_phases": int(prepared.n_phases),
        "cell_counts": counts,
        "arm_coding": "G=1 immediate; G=0 waitlist",
        "use_age": plan.use_age,
        "use_child_re": plan.use_child_re,
        "use_varying_delta": plan.use_varying_delta,
        "likelihood": plan.likelihood,
    }
    if dose:
        dose_payload = built.require_payload(DidDosePayload, family="did dose")
        scaler = dose_payload.dose_scaler
        contract.update(
            {
                "analysis_periods": ["P1", "P2"],
                "baseline_policy": (
                    "shared pre-randomisation t1 outcome and t1 age; never the "
                    "treatment-affected P2 period-start score"
                ),
                "dose_standardization": {
                    "scope": "raw sessions among treated P1/P2 rows",
                    "mean": float(scaler.mean),
                    "sd": float(scaler.sd),
                    "untreated_value": 0.0,
                },
                # The parameters this fit's posterior actually contains, with
                # labels the four-cell design supports (#576 finding 9 and
                # lower-severity 2). The pre-#576 block hard-coded ``beta_dose``,
                # which a period-varying fit does not have, and called
                # ``theta_treated`` a "current-treatment presence" effect: with
                # ``treated = (G == 1) OR (period == P2)`` the arm-by-period fixed
                # effects are saturated, so at the mean treated dose that
                # coefficient is the crossover *cell* contrast, not a separately
                # identified presence effect.
                "dose_terms": {
                    **(
                        {
                            "mu_dose": (
                                "hierarchical centre of the per-period session "
                                "slopes; not the published marginal"
                            ),
                            "sigma_dose": "between-period SD of the session slopes",
                            "beta_dose_phase": (
                                "partial-pooled per-period intensive session-dose "
                                "associations"
                            ),
                        }
                        if plan.period_varying
                        else {
                            "beta_dose": "intensive session-dose association",
                        }
                    ),
                    "theta_treated": (
                        "crossover cell contrast at the mean treated dose: "
                        "(waitlist P2 - waitlist P1) - (immediate P2 - immediate P1); "
                        "a time-by-arm cell/treatment-timing association, not an "
                        "isolated current-treatment-presence effect"
                    ),
                    "beta_group": "randomised-arm/treatment-history adjustment",
                    "period_slope_scope": (
                        "the P2 slope relates P2 sessions to the t3 period-end level "
                        "conditional on t1; it is not a P2 gain slope, because the "
                        "treatment-affected t2 period-start score and prior P1 dose "
                        "are deliberately omitted"
                    ),
                },
            }
        )
    else:
        arm_payload = built.require_payload(
            DidArmWavePayload, family="did arm-by-wave"
        )
        contract.update(
            {
                "analysis_waves": ["t1", "t2", "t3"],
                "baseline_policy": (
                    "t1 is modelled as an outcome level; no period-start outcome "
                    "is conditioned on"
                ),
                # None for the LRPDID101 independent-prior companion: its free
                # alpha has no outcome-informed location to record.
                "alpha_anchor_logit": (
                    float(arm_payload.alpha_anchor)
                    if arm_payload.alpha_anchor is not None
                    else None
                ),
                "score_mean_link": arm_payload.score_mean_link,
                "arm_gap_orientation": "immediate minus waitlist",
                # #576 finding 3: t3 is a *different randomised exposure contrast*,
                # not an observational one. Assignment stays randomised after
                # crossover, so latent ability does not confound it; what is missing
                # is the mechanism, because duration, carryover, maturation, ceiling
                # effects and different taught blocks are inseparable there.
                "contrast_status": {
                    "arm_gap_t1": "pre-randomisation balance association",
                    "tau_t2": (
                        "randomised assignment contrast: immediate treatment versus "
                        "no treatment yet, at t2"
                    ),
                    "arm_gap_t3": (
                        "randomised assignment contrast between treatment schedules: "
                        "early-start (about 40 weeks) versus delayed-start (about 20 "
                        "weeks) treatment history at t3; not a treated-versus-"
                        "untreated effect and not mechanism-identified"
                    ),
                    "delta_crossover": (
                        "change between two randomised regime contrasts (t2 gap minus "
                        "t3 gap); not an identified catch-up mechanism"
                    ),
                },
                "marginal_standardization": (
                    "wave-specific fitted-row standardised arm means and gaps"
                ),
            }
        )
    return contract


def fit_did(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    require_spec(spec, "did", outcome=True)

    # Resolve and validate the family contract before the context resets an output
    # directory or the loader reads any data (#394 pillar 4). One plan then drives
    # preparation, factory arguments, the teaching recipe and config.json. Binary
    # models use the t1-t3 levels frame so the randomised t2 arm gap and the
    # post-crossover t3 gap are estimated separately; dose models retain the
    # transition frame because sessions are interval exposures (resolved in the plan).
    plan = resolve_did_run_plan(spec)
    ctx = make_context(spec, config)
    ctx.resolved_plan = plan
    _report.write_model_recipe(ctx)

    sym = spec.outcome_symbol
    dose = plan.dose
    period_varying = plan.period_varying
    off_floor = plan.off_floor

    section_header("Prepare data")
    prepared = load_and_prepare(**plan.prepare_kwargs())
    ctx.prepared = prepared

    print_header(ctx)

    section_header("Build model")
    built = _factories.build_did_model(prepared, **plan.factory_kwargs())
    attach_built(ctx, built)
    print_header(ctx)
    did_contract = _did_analysis_contract(
        ctx,
        plan,
        built,
        dose=dose,
        loaded_prepared=prepared,
    )

    render_model_graph(ctx)

    def _write_did_cell_ppc(c: StatisticalFitContext) -> None:
        node = "y_offfloor" if off_floor else "y_post"
        cell_ppc = _report.did_cell_ppc(
            c.trace,
            phase=c.prepared.phase,
            G=c.prepared.G,
            dose=dose,
            node=node,
            ci_prob=c.reporting.ci_prob,
        )
        save_table(c, "did_cell_ppc", cell_ppc)
        save_did_cell_ppc_plot(c, cell_ppc)
        if not dose:
            # The repeated-measures covariance check (#576 material qualification
            # 3). The cell PPC above compares marginal cell means and zero rates,
            # which a model with badly wrong within-child dependence can still
            # reproduce; this one compares within-child changes and wave-to-wave
            # correlations, which it cannot. Guarded: an expensive fit must not be
            # lost to a diagnostic.
            with guard_optional(
                c, "DiD within-child PPC",
                filename="did_within_child_ppc.csv", kind="table", verb="skipped",
            ):
                save_table(
                    c,
                    "did_within_child_ppc",
                    _report.did_within_child_ppc(
                        c.trace,
                        phase=c.prepared.phase,
                        subject_ids=c.prepared.subject_ids,
                        G=c.prepared.G,
                        node=node,
                        ci_prob=c.reporting.ci_prob,
                    ),
                    required=False,
                )

    _did_diag = plan.diagnostic_vars()
    _did_effect = plan.effect_term
    shared_stages().run_primary_fit(
        ctx,
        PrimaryFitPlan(
            diagnostic_vars=tuple(_did_diag),
            ppc_var_names=("y_offfloor",) if off_floor else ("y_post",),
            plot_prior_predictive=lambda c: _diag.save_prior_predictive_plot(
                c, spec.outcome_symbol or "W"
            ),
            post_ppc_audit=_write_did_cell_ppc,
            psense_timing="family_tail",
            extended_term=_did_effect,
        ),
    )
    did_cell_ppc = ctx.tables["did_cell_ppc"]
    _within_child_ppc = ctx.tables.get("did_within_child_ppc")
    # Preserve the existing post-trace diagnostic tail.
    _diag.save_prior_posterior_plot(ctx, var_names=_did_diag)
    _diag.run_psense(ctx, var_names=list(plan.psense_terms))
    if not dose:
        with guard_optional(
            ctx, "DiD prior pushforward",
            filename="prior_pushforward.csv", kind="table", verb="skipped",
        ):
            from language_reading_predictors.statistical_models.measures import MEASURES

            prior_pushforward = _report.prior_pushforward(
                ctx.prior_samples,
                G=ctx.prepared.G,
                n_trials=1 if off_floor else MEASURES[sym].n_trials,
                term="tau_t2",
                varying_term="",
                eta_name="eta",
                ci_prob=ctx.reporting.ci_prob,
                row_mask=ctx.prepared.phase == 1,
                score_mean_link=plan.score_mean_link,
            )
            prior_pushforward_df = pd.DataFrame([prior_pushforward])
            save_table(
                ctx, "prior_pushforward", prior_pushforward_df, required=False
            )
        save_forest_plot(
            ctx,
            ["tau_t2", "arm_gap_t3", "delta_crossover"],
            name="did_contrasts_forest.png",
            title="Randomised t2 and t3 treatment-schedule contrasts",
        )

    from language_reading_predictors.statistical_models.measures import MEASURES

    section_header(
        "Dose-model association summary"
        if dose
        else "Arm-by-wave crossover contrasts"
    )
    did_s = _report.did_summary(
        ctx.trace,
        ci_prob=ctx.reporting.ci_prob,
        n_trials=1 if off_floor else MEASURES[sym].n_trials,
        dose=dose,
        off_floor=off_floor,
        wave=None if dose else ctx.prepared.phase,
        score_mean_link=plan.score_mean_link,
        subject_ids=None if dose else ctx.prepared.subject_ids,
    )
    did_df = pd.DataFrame([did_s])
    save_table(ctx, "did_summary", did_df)
    print_table(
        metrics_table(
            [{"metric": k, "value": v} for k, v in did_s.items()],
            title=(
                f"{'dose-model associations' if dose else 'arm-by-wave contrasts'} "
                f"({sym}{', off-floor probability' if off_floor else ''}) - "
                f"{int(ctx.reporting.ci_prob * 100)}% CI (equal-tailed)"
            ),
            columns=["metric", "value"],
        )
    )

    if not dose:
        # Predicted-scores contrast panel + icon array (#316) for the one clean
        # randomised quantity, tau_t2, at the t2 rows' covariate distribution.
        # The dose companions carry no randomised on/off contrast and are skipped.
        from language_reading_predictors.statistical_models.measures import (
            ROPE_DELTA,
            ROPE_DELTA_PROB,
        )

        write_predicted_scores(
            ctx,
            outcome_symbol=sym,
            G=built.prepared.G,
            n_trials=1 if off_floor else int(MEASURES[sym].n_trials),
            term="tau_t2",
            varying_term="",
            row_mask=built.prepared.phase == 1,
            likelihood="bernoulli" if off_floor else "beta_binomial",
            child_re=plan.use_child_re,
            child_idx=built.prepared.child_idx,
            delta=ROPE_DELTA_PROB.get(sym) if off_floor else ROPE_DELTA.get(sym),
            population=(
                "covariate profiles drawn from the fitted t2 rows"
            ),
            contrast_status=(
                "randomised t2 assignment contrast — immediate treatment versus no "
                "treatment yet — within a within-child longitudinal "
                "(waitlist-crossover) model"
            ),
            event_label="off the floor at t2 (prevalence)",
            split=True,
            score_mean_link=plan.score_mean_link,
        )

        # Data-space figures (#317): the crossover trajectory (headline picture) and
        # per-child fitted-vs-observed panels. Only the binary t1--t3 levels model
        # carries a per-wave level; the dose companions are transition-frame and skip.
        _obs_node = "y_offfloor" if off_floor else "y_post"
        write_group_trajectory(
            ctx,
            outcome_symbol=sym,
            arm=built.prepared.G,
            wave=built.prepared.phase,
            child_idx=built.prepared.child_idx,
            off_floor=off_floor,
            obs_node=_obs_node,
            score_mean_link=plan.score_mean_link,
            # LRPDID13 adds a waitlist-child t3 deviation. It is a random effect
            # like the child intercept, so a *population* trajectory has to
            # integrate it too; leaving the fitted values in eta made the "curve"
            # a same-children conditional display wearing a population label
            # (#576 finding 5).
            extra_effect_name="v_delta" if plan.use_varying_delta else None,
            extra_effect_sd_name="sigma_delta" if plan.use_varying_delta else None,
            extra_effect_rows=(
                ((built.prepared.G == 0) & (built.prepared.phase == 2))
                if plan.use_varying_delta
                else None
            ),
            extra_effect_idx=(
                _waitlist_crossover_index(built)
                if plan.use_varying_delta
                else None
            ),
        )
        write_child_fit(
            ctx,
            outcome_symbol=sym,
            wave=built.prepared.phase,
            child_idx=built.prepared.child_idx,
            off_floor=off_floor,
            obs_node=_obs_node,
        )

    if dose:
        # The dose readout (#135), written by the shared dose-slope summary for
        # BOTH dose designs (#576 finding 1). Before this the pooled companions
        # emitted only an inline prior pushforward and no ``dose_marginal_summary``
        # at all, so "the published dose estimand" meant one thing for did-007 and
        # something else for did-006/107. For the period-varying fit the headline
        # question — does the L dose-gain slope vary by period? — is answered by
        # the nested PSIS-LOO against the pooled comparator (lrp-rli-did-107) in
        # compare_statistical_models.py, not by this single-fit table.
        section_header(
            "Period-resolved dose-slope summary"
            if period_varying
            else "Pooled dose-slope summary"
        )
        # The DiD dose factory standardises sessions among treated P1/P2 rows
        # only, so the persisted per-session calibration must come from the
        # fitted payload's scaler — not the loader's all-rows scaler, whose SD
        # (diluted by the untreated zero-session cell and the P3 rows) would
        # misstate the slope's scale and contradict config.json's
        # ``dose_standardization`` record. The natural-scale marginal (and its
        # prior pushforward) averages over the treated rows for the same reason:
        # a +1 SD dose step on an untreated waitlist-P1 row is not a supported
        # counterfactual of the treated-centred intensive-margin design.
        _dose_payload = built.require_payload(DidDosePayload, family="did dose")
        write_dose_slope_summary(
            ctx,
            period_varying=period_varying,
            dose_scaler=_dose_payload.dose_scaler,
            marginal_row_mask=np.asarray(_dose_payload.treated) == 1,
        )
    het = None
    if plan.use_varying_delta:
        section_header("Exploratory waitlist catch-up heterogeneity")
        het = _did_heterogeneity_summary(ctx.trace, ci_prob=ctx.reporting.ci_prob)
        save_table(ctx, "heterogeneity_summary", pd.DataFrame([het]))
        print_table(
            metrics_table(
                [{"metric": k, "value": v} for k, v in het.items()],
                title=(
                    f"waitlist catch-up heterogeneity ({sym}): between-child SD of "
                    "the exploratory t3 catch-up association (logit)"
                ),
                columns=["metric", "value"],
            )
        )

    write_run_metadata(
        ctx,
        extra={
            "loo_elpd": float(ctx.loo.elpd),
            "did_summary": did_s,
            "dose": dose,
            "period_varying_dose": period_varying,
            "did_cell_ppc": {
                "file": "did_cell_ppc.csv",
                "n_cells": int(len(did_cell_ppc)),
                "mean_tail_flags": int(did_cell_ppc["mean_tail_flag"].sum()),
                "zero_tail_flags": int(did_cell_ppc["zero_tail_flag"].sum()),
            },
            **(
                {
                    "did_within_child_ppc": {
                        "file": "did_within_child_ppc.csv",
                        "n_statistics": int(len(_within_child_ppc)),
                        "tail_flags": int(_within_child_ppc["tail_flag"].sum()),
                    }
                }
                if _within_child_ppc is not None and not _within_child_ppc.empty
                else {}
            ),
            **did_contract,
            **(
                {
                    "dose_slope_summary": ctx.tables[
                        "dose_slope_summary"
                    ].to_dict("records"),
                    "dose_marginal_summary": ctx.tables[
                        "dose_marginal_summary"
                    ].to_dict("records"),
                }
                if dose
                else {}
            ),
            **({"heterogeneity_summary": het} if het is not None else {}),
        },
    )

    return finalize_report(ctx)
