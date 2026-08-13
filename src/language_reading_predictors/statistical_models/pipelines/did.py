# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Waitlist-crossover / difference-in-differences orchestration (``kind="did"``).

``fit_did`` fits the arm-by-wave family: bounded t1/t2/t3 levels with separate
immediate-minus-waitlist gaps, where ``tau_t2`` is the clean randomised contrast
and ``arm_gap_t3`` / ``delta_crossover`` are post-crossover associations. The dose
companions add treated-centred session intensity and report their slopes as
observational associations — through the same summary writer the dose-response
family uses, imported from :mod:`.dose_response`.
"""

from __future__ import annotations

import hashlib

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
    resolve_did_run_plan,
)
from language_reading_predictors.statistical_models.figure_artifacts import (
    save_did_cell_ppc_plot,
    save_forest_plot,
    write_child_fit,
    write_group_trajectory,
    write_predicted_scores,
)
from language_reading_predictors.statistical_models.pipelines.dose_response import (
    write_dose_slope_summary,
)
from language_reading_predictors.statistical_models.preprocessing import (
    load_and_prepare,
)
from language_reading_predictors.statistical_models.prior_artifacts import (
    marginal_pushforward_rows,
    pushforward_outcome_label,
    write_prior_pushforward,
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


def _did_diag_vars(spec: ModelSpec) -> list[str]:
    """Coefficients to summarise for a crossover/DiD fit, given the spec."""
    dose = bool(spec.extra.get("dose", False))
    period_varying = dose and bool(spec.extra.get("period_varying_dose", False))
    off_floor = spec.extra.get("likelihood") == "bernoulli_offfloor"
    if not dose:
        v = [
            # LRPDID101 (use_intercept_anchor=False) fits a free alpha with no
            # anchored offset; every other arm-by-wave fit summarises the offset.
            "alpha_offset" if spec.extra.get("use_intercept_anchor", True) else "alpha",
            "beta_period",
            "arm_gap_t1",
            "tau_t2",
            "arm_gap_t3",
        ]
    else:
        dose_vars = (
            ["mu_dose", "sigma_dose", "beta_dose_phase"]
            if period_varying
            else ["beta_dose"]
        )
        v = [
            "alpha",
            "beta_period",
            "beta_group",
            "theta_treated",
            "gamma_t1",
            *dose_vars,
        ]
    if not off_floor:
        v += ["kappa"]
    if spec.extra.get("use_age", True):
        v.append("gamma_A")
    if spec.extra.get("use_child_re", True):
        v.append("sigma_child")
    if spec.extra.get("use_varying_delta", False):
        v.append("sigma_delta")
    return v


# Negligible-heterogeneity threshold on the logit scale for the "does the
# between-child waitlist catch-up SD concentrate near zero?" diagnostic (#230
# §4a): an order of magnitude below the delta / tau prior scale (Normal(0, 0.5)).
_SIGMA_DELTA_ROPE = 0.1


def _did_heterogeneity_summary(trace, *, ci_prob: float) -> dict[str, float]:
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


def _did_analysis_contract(
    ctx: StatisticalFitContext,
    built,
    *,
    dose: bool,
    loaded_prepared,
) -> dict:
    """Persist the exact fitted rows and return auditable DiD design metadata."""
    prepared = built.prepared
    row_ids = np.asarray(built.extras["analysis_row_ids"], dtype=str)
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
        manifest["treated"] = np.asarray(built.extras["treated"], dtype=int)
        manifest["sessions_raw"] = np.asarray(
            built.extras["raw_attend"], dtype=float
        )
        manifest["dose_treated_std"] = np.asarray(
            built.extras["dose_treated_std"], dtype=float
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
        "design": built.extras["design"],
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
        "use_age": bool(ctx.spec.extra.get("use_age", True)),
        "use_child_re": bool(ctx.spec.extra.get("use_child_re", True)),
        "use_varying_delta": bool(
            ctx.spec.extra.get("use_varying_delta", False)
        ),
        "likelihood": ctx.spec.extra.get("likelihood", "beta_binomial"),
    }
    if dose:
        scaler = built.extras["dose_scaler"]
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
                "dose_terms": {
                    "theta_treated": "current-treatment presence association",
                    "beta_dose": "intensive session-dose association",
                    "beta_group": "randomised-arm/history adjustment",
                },
            }
        )
    else:
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
                    float(built.extras["alpha_anchor"])
                    if built.extras["alpha_anchor"] is not None
                    else None
                ),
                "arm_gap_orientation": "immediate minus waitlist",
                "contrast_status": {
                    "arm_gap_t1": "pre-randomisation balance association",
                    "tau_t2": "randomised t2 causal contrast",
                    "arm_gap_t3": "post-crossover 40-week-vs-20-week association",
                    "delta_crossover": "t2 gap minus t3 gap; catch-up association",
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
        built,
        dose=dose,
        loaded_prepared=prepared,
    )

    render_model_graph(ctx)

    def _write_did_cell_ppc(c: StatisticalFitContext) -> None:
        cell_ppc = _report.did_cell_ppc(
            c.trace,
            phase=c.prepared.phase,
            G=c.prepared.G,
            dose=dose,
            node="y_offfloor" if off_floor else "y_post",
            ci_prob=c.reporting.ci_prob,
        )
        save_table(c, "did_cell_ppc", cell_ppc)
        save_did_cell_ppc_plot(c, cell_ppc)

    _did_diag = _did_diag_vars(spec)
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
            )
            prior_pushforward_df = pd.DataFrame([prior_pushforward])
            save_table(
                ctx, "prior_pushforward", prior_pushforward_df, required=False
            )
        save_forest_plot(
            ctx,
            ["tau_t2", "arm_gap_t3", "delta_crossover"],
            name="did_contrasts_forest.png",
            title="Randomised t2 and post-crossover contrasts",
        )
    elif not period_varying:
        # Pooled-dose companions (#381). The period-varying ones route through
        # ``write_dose_slope_summary`` below, which emits the same check from the
        # shared writer; these do not reach it, so they emit it here.
        from language_reading_predictors.statistical_models.measures import MEASURES

        write_prior_pushforward(
            ctx,
            marginal_pushforward_rows(
                ctx,
                [
                    (
                        "beta_dose",
                        "the association of a +1 SD session-dose step with "
                        f"{pushforward_outcome_label(ctx, sym)}",
                    )
                ],
                n_trials=1 if off_floor else MEASURES[sym].n_trials,
                convention="forward",
            ),
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
                "randomised t2 arm contrast within a within-child longitudinal "
                "(waitlist-crossover) model"
            ),
            event_label="off the floor at t2 (prevalence)",
            split=True,
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
        )
        write_child_fit(
            ctx,
            outcome_symbol=sym,
            wave=built.prepared.phase,
            child_idx=built.prepared.child_idx,
            off_floor=off_floor,
            obs_node=_obs_node,
        )

    if period_varying:
        # Period-resolved dose readout (#135): partial-pooled per-period dose
        # slopes + a between-period SD, written by the shared dose-slope summary.
        # The headline question — does the L dose-gain slope vary by period? — is
        # answered by the nested PSIS-LOO vs the pooled comparator (lrp-rli-did-107)
        # in compare_statistical_models.py, not by this single-fit table.
        section_header("Period-resolved dose-slope summary")
        write_dose_slope_summary(ctx, period_varying=True)
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
            **did_contract,
            **(
                {
                    "dose_slope_summary": ctx.tables[
                        "dose_slope_summary"
                    ].to_dict("records")
                }
                if period_varying
                else {}
            ),
            **({"heterogeneity_summary": het} if het is not None else {}),
        },
    )

    return finalize_report(ctx)
