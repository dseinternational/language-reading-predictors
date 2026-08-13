# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Concurrent conditional-associations orchestration (``kind="concurrent"``, #312).

``fit_concurrent`` fits, at each timepoint, a between-child Beta-Binomial
regression of the focal outcome's *level* on the standardised same-wave logits of a
predictor skill set, plus age and a group nuisance term. Four separate
cross-sectional fits are reported side by side: the diagnostic-anchor wave (most
rows, ties to the latest) carries the standard trace, convergence gate and PPC
artefacts, while the other waves and every single-skill comparator are sub-fits
with their own recorded convergence diagnostics. The comparator retains the same
trait covariates while omitting age, group and the other skills.

Predictors and outcome are measured at the same visit, so no temporal ordering is
available and every coefficient is an adjusted association by construction —
conditioning on contemporaneous, post-treatment skill levels is intentional here
precisely because the family makes no causal claim.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dse_research_utils.plot.styles import COLOUR_BLUE
from rich import print as rprint

from language_reading_predictors.models._reporting import (
    print_table,
    ranked_dataframe_table,
    section_header,
)
from language_reading_predictors.statistical_models import (
    diagnostics as _diag,
    factories as _factories,
    reporting as _report,
)
from language_reading_predictors.statistical_models.artifacts import save_table
from language_reading_predictors.statistical_models.adjustment import (
    effective_adjustment,
)
from language_reading_predictors.statistical_models.concurrent import (
    resolve_concurrent_run_plan,
)
from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
    make_context,
)
from language_reading_predictors.statistical_models.factories import default_of
from language_reading_predictors.statistical_models.plotting import save_styled_figure
from language_reading_predictors.statistical_models.preprocessing import (
    _subset_prepared,
    filter_informative_covariates,
    load_and_prepare,
    logit_safe,
    standardise,
)
from language_reading_predictors.statistical_models.prior_artifacts import (
    marginal_pushforward_rows,
    write_prior_pushforward,
)
from language_reading_predictors.statistical_models.publication import (
    print_header,
    render_model_graph,
)
from language_reading_predictors.statistical_models.reporting import beta_summary
from language_reading_predictors.statistical_models.runtime import (
    attach_built,
    finalize_report,
    require_spec,
    shared_stages,
    write_run_metadata,
)
from language_reading_predictors.statistical_models.stages import PrimaryFitPlan
from language_reading_predictors.statistical_models.subfits import run_subfit


_CA_LABELS = {
    "W": "Word reading",
    "L": "Letter sounds",
    "B": "Blending",
    "TR": "Taught receptive vocab",
    "TE": "Taught expressive vocab",
    "R": "Receptive vocab",
    "E": "Expressive vocab",
    "age": "Age",
}


def _ca_label(sym: str) -> str:
    return _CA_LABELS.get(sym, sym)


def _ca_wave_predictors(
    wave_prepared, predictor_symbols: list[str]
) -> tuple[list[str], list[str]]:
    """Split ``predictor_symbols`` into those usable at this wave and those dropped.

    A predictor is usable only if its same-wave logit has positive, finite variance on
    the wave's rows — otherwise the factory's ``standardise`` would raise (an all-missing
    or constant predictor at a wave carries no association and cannot be standardised).
    Returns ``(available, dropped)`` preserving input order.
    """
    from language_reading_predictors.statistical_models.measures import MEASURES

    available, dropped = [], []
    for sym in predictor_symbols:
        vals = np.asarray(wave_prepared.post_counts.get(sym), dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size < 2:
            dropped.append(sym)
            continue
        sd = float(np.nanstd(logit_safe(vals, MEASURES[sym].n_trials), ddof=1))
        (available if np.isfinite(sd) and sd > 0 else dropped).append(sym)
    return available, dropped


def _ca_concurrent_terms(wave_prepared, predictor_symbols: list[str]) -> list:
    """``ConcurrentTerm`` list for a wave's items-scale marginals (#312).

    Recomputes, per predictor, the same-wave logit SD (matching the factory's
    ``standardise``), the mean item count (the ``+k items`` operating point) and a
    per-measure items increment ``k = max(1, round(N / 10))`` — so a fixed ``+5`` does
    not span 3 %-50 % of predictor scales that differ tenfold (the #310/#325 caveat,
    applied here from the outset).
    """
    from language_reading_predictors.statistical_models.measures import MEASURES

    terms = []
    for sym in predictor_symbols:
        m = MEASURES[sym]
        vals = np.asarray(wave_prepared.post_counts[sym], dtype=float)
        _z, scaler = standardise(logit_safe(vals, m.n_trials))
        mean_items = float(np.nanmean(vals))
        k = max(1, round(m.n_trials / 10))
        terms.append(
            _report.ConcurrentTerm(
                label=sym,
                coef=f"beta_{sym}",
                sd_logit=float(scaler.sd),
                n_items=m.n_trials,
                mean_items=mean_items,
                k_items=k,
            )
        )
    return terms


def _ca_margin_fields(prefix: str, row: pd.Series) -> dict[str, float]:
    """Wide probability/items fields for one ``+1 SD`` concurrent marginal row."""
    return {
        f"{prefix}_ame_{scale}_{stat}": float(row[f"{scale}_{stat}"])
        for scale in ("prob", "items")
        for stat in ("median", "lo", "hi", "lo50", "hi50")
    }


def _ca_sd_margin(df: pd.DataFrame, predictor: str) -> pd.Series:
    """Return the unique ``+1 SD`` marginal row for ``predictor``."""
    rows = df[(df["term"] == predictor) & (df["scale"] == "+1 SD")]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one +1 SD marginal for {predictor!r}; found {len(rows)}"
        )
    return rows.iloc[0]


_CA_MARGIN_STATS = ("median", "lo", "hi", "lo50", "hi50")
_CA_ASSOCIATION_REQUIRED = {
    "timepoint",
    "predictor",
    "label",
    "n",
    "predictor_n",
    "predictor_imputed_n",
    "ame_contrast",
    "adj_median",
    "adj_mean",
    "adj_lo",
    "adj_hi",
    "adj_lo50",
    "adj_hi50",
    "adj_prob_pos",
    "biv_median",
    "biv_mean",
    "biv_lo",
    "biv_hi",
    "biv_lo50",
    "biv_hi50",
    "biv_prob_pos",
    "adj_converged",
    "biv_converged",
} | {
    f"{prefix}_ame_{scale}_{stat}"
    for prefix in ("adj", "biv")
    for scale in ("prob", "items")
    for stat in _CA_MARGIN_STATS
}
_CA_MARGINAL_REQUIRED = {
    "timepoint",
    "adjustment",
    "term",
    "role",
    "scale",
    "prob_median",
    "prob_lo",
    "prob_hi",
    "prob_lo50",
    "prob_hi50",
    "items_median",
    "items_lo",
    "items_hi",
    "items_lo50",
    "items_hi50",
    "prob_pos",
    "label",
    "converged",
}
_CA_DIAGNOSTIC_REQUIRED = {
    "timepoint",
    "fit_kind",
    "predictor",
    "n",
    "n_predictors",
    "n_covariates",
    "effective_covariates",
    "dropped_covariates",
    "converged",
    "max_rhat",
    "min_ess",
    "min_bfmi",
    "n_divergences",
}


def _write_concurrent_outputs(
    ctx: StatisticalFitContext,
    *,
    association_rows: list[dict],
    marginal_frames: list[pd.DataFrame],
    diagnostic_rows: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate and write the three concurrent-family output tables.

    The explicit cross-table checks make the issue #312 contract executable: every
    wave-predictor association must have adjusted and legacy-labelled ``bivariate``
    (single-skill, trait-adjusted) ``+1 SD`` natural-scale rows and a matching fit-diagnostics row, while every wave has one adjusted-fit
    diagnostics row. This prevents a future refactor from silently publishing only one
    side of the requested adjusted/single-skill comparison.
    """
    association_df = pd.DataFrame(association_rows)
    marginal_df = pd.concat(marginal_frames, ignore_index=True)
    diagnostic_df = pd.DataFrame(diagnostic_rows)

    for name, frame, required in (
        ("concurrent_associations", association_df, _CA_ASSOCIATION_REQUIRED),
        ("concurrent_marginals", marginal_df, _CA_MARGINAL_REQUIRED),
        ("concurrent_fit_diagnostics", diagnostic_df, _CA_DIAGNOSTIC_REQUIRED),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {sorted(missing)}")

    association_pairs = {
        (int(row.timepoint), str(row.predictor))
        for row in association_df[["timepoint", "predictor"]].itertuples(index=False)
    }
    expected_marginals = {
        (timepoint, predictor, adjustment)
        for timepoint, predictor in association_pairs
        for adjustment in ("adjusted", "bivariate")
    }
    sd_marginals = marginal_df[marginal_df["scale"] == "+1 SD"]
    actual_marginals = {
        (int(row.timepoint), str(row.term), str(row.adjustment))
        for row in sd_marginals[
            ["timepoint", "term", "adjustment"]
        ].itertuples(index=False)
    }
    if actual_marginals != expected_marginals:
        missing = sorted(expected_marginals - actual_marginals)
        extra = sorted(actual_marginals - expected_marginals)
        raise ValueError(
            "concurrent_marginals +1 SD cross-product mismatch: "
            f"missing={missing}, extra={extra}"
        )

    expected_adjusted = {timepoint for timepoint, _ in association_pairs}
    adjusted_diagnostics = diagnostic_df[
        diagnostic_df["fit_kind"] == "adjusted"
    ]
    actual_adjusted = {
        int(row.timepoint)
        for row in adjusted_diagnostics[["timepoint"]].itertuples(index=False)
    }
    bivariate_diagnostics = diagnostic_df[
        diagnostic_df["fit_kind"] == "bivariate"
    ]
    actual_bivariate = {
        (int(row.timepoint), str(row.predictor))
        for row in bivariate_diagnostics[
            ["timepoint", "predictor"]
        ].itertuples(index=False)
    }
    if actual_adjusted != expected_adjusted or actual_bivariate != association_pairs:
        raise ValueError(
            "concurrent_fit_diagnostics does not cover every published fit: "
            f"adjusted={sorted(actual_adjusted)}, "
            f"bivariate={sorted(actual_bivariate)}"
        )

    for name, frame in (
        ("concurrent_associations", association_df),
        ("concurrent_marginals", marginal_df),
        ("concurrent_fit_diagnostics", diagnostic_df),
    ):
        save_table(ctx, name, frame)

    return association_df, marginal_df, diagnostic_df


def fit_concurrent(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Per-wave concurrent conditional associations (LRP-CA, #312).

    Fits, at each timepoint, a between-child Beta-Binomial regression of the focal
    outcome's *level* on the standardised same-wave logits of a predictor skill set
    (plus age and a group nuisance term) — "at wave t, among children alike on age and
    the other skills, +n of predictor X is associated with +m of the outcome". Every
    coefficient is an **adjusted association**; the family makes no causal claim, so
    conditioning on contemporaneous (post-treatment) skill levels is intentional.

    Design (issue #312): four separate cross-sectional fits, reported side by side. The
    diagnostic-anchor wave (most rows; ties → latest) is the fit that carries the
    standard trace / convergence-gate / PPC artefacts; the other waves and every
    single-skill (trait-adjusted, but without age/group/other skills) fit is a
    sub-fit. Every published fit has
    R-hat, ESS, BFMI and divergence diagnostics recorded in
    ``concurrent_fit_diagnostics.csv``. ``concurrent_associations.csv`` carries the
    adjusted and single-skill-comparator logit coefficients plus matched +1-SD probability/items
    marginals (wave × predictor); ``concurrent_marginals.csv`` carries both fit kinds'
    detailed probability/items marginals (wave × predictor × {+1 SD, +k items}).
    """
    require_spec(spec, "concurrent", outcome=True)
    # Resolve and validate the family contract before the context resets an output
    # directory or the loader reads any data (#394 pillar 4). One plan drives
    # preparation, the teaching recipe and config.json.
    plan = resolve_concurrent_run_plan(spec)
    outcome = plan.outcome_symbol
    predictor_symbols = list(plan.predictor_symbols)
    # Trait covariates (non-verbal ability, hearing, speech, phonological memory),
    # aligned with the gains panel. They are t1-measured, so they enter as
    # baseline covariates broadcast across the waves (there is no per-wave value).
    covariates = list(plan.covariates)
    include_age = plan.include_age
    include_group = plan.include_group
    # ``predictor_slope_sigma`` is None on the plan when a spec does not set it, so the
    # build_concurrent_model default is filled via default_of here — the anti-drift
    # single source #394 retains until typed family defaults replace it.
    sigma0 = (
        float(plan.predictor_slope_sigma)
        if plan.predictor_slope_sigma is not None
        else float(
            default_of(_factories.build_concurrent_model, "predictor_slope_sigma")
        )
    )

    from language_reading_predictors.statistical_models.measures import MEASURES

    ctx = make_context(spec, config)
    ctx.resolved_plan = plan
    _report.write_model_recipe(ctx)
    hdi = ctx.reporting.ci_prob
    N_focal = MEASURES[outcome].n_trials

    section_header("Prepare data")
    prepared_all = load_and_prepare(**plan.prepare_kwargs())

    # Timepoints present; each wave's row count and its usable predictor set (a
    # predictor whose same-wave logit has positive variance on the wave's rows —
    # anything constant/all-missing at that wave is dropped, and a wave with no usable
    # predictor is skipped below).
    wave_indices = sorted({int(p) for p in np.unique(prepared_all.phase)})
    wave_subsets: dict[int, object] = {}
    wave_n: dict[int, int] = {}
    wave_preds: dict[int, list[str]] = {}
    dropped_by_wave: dict[int, list[str]] = {}
    wave_covariates: dict[int, list[str]] = {}
    dropped_covariates_by_wave: dict[int, list[str]] = {}
    for w in wave_indices:
        sub = _subset_prepared(prepared_all, prepared_all.phase == w)
        keep = ~np.isnan(sub.post_counts[outcome])
        sub = _subset_prepared(sub, keep)
        sub, effective_covariates, dropped_covariates = (
            filter_informative_covariates(sub, covariates)
        )
        wave_subsets[w] = sub
        wave_n[w] = sub.n_obs
        wave_preds[w], dropped_by_wave[w] = _ca_wave_predictors(sub, predictor_symbols)
        wave_covariates[w] = list(effective_covariates)
        dropped_covariates_by_wave[w] = list(dropped_covariates)
    # Diagnostic anchor = most complete-outcome rows; tie → latest timepoint. This is
    # an operational artefact-selection rule, not a claim that the wave is best-powered
    # or substantively primary. Choose it ONLY among waves that actually have a usable
    # predictor: a wave whose predictors are all constant/all-missing is skipped in the
    # fit loop, so making it the anchor would leave ``wave_fits[primary_wave]`` unset
    # and crash the fit.
    fittable_waves = [w for w in wave_indices if wave_preds[w]]
    if not fittable_waves:
        raise ValueError(
            f"{spec.model_id}: no wave has a usable predictor (all "
            f"{predictor_symbols} are constant/all-missing at every timepoint); "
            "cannot fit the concurrent model."
        )
    primary_wave = max(fittable_waves, key=lambda w: (wave_n[w], w))

    # Provisional; replaced with the primary-wave subset once known so the report's
    # header / n_obs describe the gated fit.
    ctx.prepared = wave_subsets[primary_wave]
    print_header(ctx)

    def _build(sub, preds, covs, *, age, group):
        return _factories.build_concurrent_model(
            sub,
            outcome_symbol=outcome,
            predictor_symbols=preds,
            covariates=covs,
            include_age=age,
            include_group=group,
            predictor_slope_sigma=sigma0,
        )

    # ---- Fit each wave's mutually-adjusted model --------------------------------
    wave_fits: dict[int, dict] = {}

    def _fit_non_primary_wave(w: int) -> None:
        sub = wave_subsets[w]
        preds = wave_preds[w]
        covs = wave_covariates[w]
        tp = w + 1  # 1-based timepoint for reports
        if not preds:
            rprint(f"[yellow]Concurrent: wave t{tp} has no usable predictors; skipped.[/yellow]")
            return
        built = _build(sub, preds, covs, age=include_age, group=include_group)
        res = run_subfit(
            ctx, built, label=f"{spec.model_id} wave t{tp}", role="wave"
        )
        wave_fits[w] = {
            "trace": res.trace,
            "prepared": built.prepared,
            "preds": preds,
            "covariates": covs,
            "dropped_covariates": dropped_covariates_by_wave[w],
            "convergence": res.convergence,
        }

    # Preserve the established chronological fit order: waves before the anchor
    # remain before its build/sampling, while later waves run immediately after the
    # anchor sample and before its summary diagnostics.
    for w in wave_indices:
        if w == primary_wave:
            break
        _fit_non_primary_wave(w)

    primary_sub = wave_subsets[primary_wave]
    primary_preds = wave_preds[primary_wave]
    primary_covs = wave_covariates[primary_wave]
    section_header(f"Build model (primary wave t{primary_wave + 1})")
    primary_built = _build(
        primary_sub,
        primary_preds,
        primary_covs,
        age=include_age,
        group=include_group,
    )
    attach_built(ctx, primary_built)
    render_model_graph(ctx)
    wave_fits[primary_wave] = {
        "trace": None,
        "prepared": primary_built.prepared,
        "preds": primary_preds,
        "covariates": primary_covs,
        "dropped_covariates": dropped_covariates_by_wave[primary_wave],
        "convergence": None,
    }

    prim = wave_fits[primary_wave]
    beta_names = [f"beta_{s}" for s in prim["preds"]]
    diag_vars = ["alpha", "kappa", *beta_names]
    if include_age:
        diag_vars.append("beta_age")
    if include_group:
        diag_vars.append("beta_group_nuisance")
    diag_vars.extend(f"gamma_{name}" for name in prim["covariates"])

    def _finish_wave_fits(c: StatisticalFitContext) -> None:
        wave_fits[primary_wave]["trace"] = c.trace
        after_primary = False
        for w in wave_indices:
            if w == primary_wave:
                after_primary = True
                continue
            if after_primary:
                _fit_non_primary_wave(w)

    def _record_primary_convergence(
        c: StatisticalFitContext, gate: dict
    ) -> None:
        primary_conv = _diag.subfit_convergence(
            c.trace,
            label=f"{spec.model_id} primary wave t{primary_wave + 1}",
            var_names=[rv.name for rv in c.model.free_RVs],
        )
        primary_conv["converged"] = bool(
            _report.convergence_gate_clean_passed(gate)
            and primary_conv.get("converged")
        )
        wave_fits[primary_wave]["convergence"] = primary_conv

    shared_stages().run_primary_fit(
        ctx,
        PrimaryFitPlan(
            diagnostic_vars=tuple(diag_vars),
            summary_header="Summary diagnostics (primary wave)",
            extended_header="Extended diagnostics (primary wave)",
            plot_prior_predictive=lambda c: _diag.save_prior_predictive_plot(
                c, outcome
            ),
            post_sampling_audit=_finish_wave_fits,
            post_gate_audit=_record_primary_convergence,
        ),
    )
    _diag.save_prior_posterior_plot(ctx, var_names=diag_vars)
    # Estimand-scale prior check on the primary wave's adjusted associations
    # (#381) — one row per predictor, on the same ``+1 SD`` forward-shift scale
    # ``concurrent_marginals`` reports below. Only the primary wave persists a
    # prior group; the other waves are refits sampled without one.
    write_prior_pushforward(
        ctx,
        marginal_pushforward_rows(
            ctx,
            [
                (
                    f"beta_{s}",
                    f"the adjusted association of +1 SD {_ca_label(s)} with "
                    f"{MEASURES[outcome].label} at t{primary_wave + 1}",
                )
                for s in prim["preds"]
            ],
            n_trials=N_focal,
            convention="forward",
        ),
    )

    # ---- Adjusted vs single-skill coefficients + natural-scale marginals --------
    section_header("Concurrent associations (adjusted vs single-skill)")
    assoc_rows: list[dict] = []
    marg_frames: list[pd.DataFrame] = []
    fit_diagnostic_rows: list[dict] = []
    for w in wave_indices:
        if w not in wave_fits:
            continue
        tp = w + 1
        fit = wave_fits[w]
        sub, preds, trace = fit["prepared"], fit["preds"], fit["trace"]
        covs = fit["covariates"]
        dropped_covs = fit["dropped_covariates"]
        covariate_text = "|".join(covs)
        dropped_covariate_text = "|".join(dropped_covs)
        adj_conv = fit["convergence"]
        fit_diagnostic_rows.append(
            {
                "timepoint": tp,
                "fit_kind": "adjusted",
                "predictor": "all",
                "n": sub.n_obs,
                "n_predictors": len(preds),
                "n_covariates": len(covs),
                "effective_covariates": covariate_text,
                "dropped_covariates": dropped_covariate_text,
                **adj_conv,
            }
        )

        # Natural-scale marginals for the mutually-adjusted associations at this wave.
        terms = _ca_concurrent_terms(sub, preds)
        terms_by_symbol = {term.label: term for term in terms}
        adj_mdf = _report.concurrent_marginals(
            trace, terms=terms, n_trials=N_focal, ci_prob=hdi
        )
        adj_mdf.insert(0, "timepoint", tp)
        adj_mdf.insert(1, "adjustment", "adjusted")
        adj_mdf["label"] = adj_mdf["term"].map(_ca_label)
        adj_mdf["converged"] = adj_conv["converged"]
        marg_frames.append(adj_mdf)

        # Per-predictor: adjusted beta plus the legacy-labelled ``bivariate``
        # single-skill beta (same trait adjustment, but no age/group/other skills).
        for sym in preds:
            adj = beta_summary(trace, f"beta_{sym}", hdi)
            b = _build(sub, [sym], covs, age=False, group=False)
            bres = run_subfit(
                ctx,
                b,
                label=f"{spec.model_id} t{tp} single-skill {sym}",
                role="bivariate",
            )
            bt, bconv = bres.trace, bres.convergence
            biv = beta_summary(bt, f"beta_{sym}", hdi)
            biv_mdf = _report.concurrent_marginals(
                bt,
                terms=[terms_by_symbol[sym]],
                n_trials=N_focal,
                ci_prob=hdi,
            )
            biv_mdf.insert(0, "timepoint", tp)
            biv_mdf.insert(1, "adjustment", "bivariate")
            biv_mdf["label"] = biv_mdf["term"].map(_ca_label)
            biv_mdf["converged"] = bconv["converged"]
            marg_frames.append(biv_mdf)

            adj_sd = _ca_sd_margin(adj_mdf, sym)
            biv_sd = _ca_sd_margin(biv_mdf, sym)
            predictor_n = int(np.isfinite(sub.post_counts[sym]).sum())
            assoc_rows.append(
                {
                    "timepoint": tp,
                    "predictor": sym,
                    "label": _ca_label(sym),
                    "n": sub.n_obs,
                    "predictor_n": predictor_n,
                    "predictor_imputed_n": sub.n_obs - predictor_n,
                    "ame_contrast": "+1 SD",
                    "adj_median": adj["median"],
                    "adj_mean": adj["mean"],
                    "adj_lo": adj["lo"],
                    "adj_hi": adj["hi"],
                    "adj_lo50": adj["lo50"],
                    "adj_hi50": adj["hi50"],
                    "adj_prob_pos": adj["prob_pos"],
                    **_ca_margin_fields("adj", adj_sd),
                    "biv_median": biv["median"],
                    "biv_mean": biv["mean"],
                    "biv_lo": biv["lo"],
                    "biv_hi": biv["hi"],
                    "biv_lo50": biv["lo50"],
                    "biv_hi50": biv["hi50"],
                    "biv_prob_pos": biv["prob_pos"],
                    **_ca_margin_fields("biv", biv_sd),
                    "adj_converged": adj_conv["converged"],
                    "biv_converged": bconv["converged"],
                }
            )
            fit_diagnostic_rows.append(
                {
                    "timepoint": tp,
                    "fit_kind": "bivariate",
                    "predictor": sym,
                    "n": sub.n_obs,
                    "n_predictors": 1,
                    "n_covariates": len(covs),
                    "effective_covariates": covariate_text,
                    "dropped_covariates": dropped_covariate_text,
                    **bconv,
                }
            )

    assoc_df, marg_df, fit_diagnostics_df = _write_concurrent_outputs(
        ctx,
        association_rows=assoc_rows,
        marginal_frames=marg_frames,
        diagnostic_rows=fit_diagnostic_rows,
    )
    print_table(
        ranked_dataframe_table(
            assoc_df,
            title=f"Concurrent associations (per-SD, logit; {int(hdi * 100)}% interval)",
            columns=[
                "timepoint", "label", "adj_mean", "adj_lo", "adj_hi", "adj_prob_pos",
                "biv_mean", "biv_lo", "biv_hi",
            ],
            rank_column=False,
            precision=3,
        )
    )
    _plot_concurrent(ctx, assoc_df, hdi, primary_tp=primary_wave + 1)

    all_fits_converged = bool(
        not fit_diagnostics_df.empty
        and fit_diagnostics_df["converged"].eq(True).all()
    )
    meta_extra = {
        "loo_elpd": float(ctx.loo.elpd) if ctx.loo is not None else None,
        "estimand": "concurrent conditional associations (per wave)",
        "predictors": prim["preds"],
        "predictors_requested": predictor_symbols,
        "dropped_by_wave": {f"t{w + 1}": dropped_by_wave[w] for w in wave_indices},
        "primary_timepoint": primary_wave + 1,
        "diagnostic_anchor_timepoint": primary_wave + 1,
        "timepoints": [w + 1 for w in wave_indices],
        "wave_n": {f"t{w + 1}": wave_n[w] for w in wave_indices},
        "include_age": include_age,
        "include_group_nuisance": include_group,
        "bivariate_adjustment": (
            "single-skill comparator; same effective trait covariates retained; "
            "age, group and other skills omitted"
        ),
        "covariates_requested": covariates,
        "effective_covariates_by_wave": {
            f"t{w + 1}": wave_covariates[w] for w in wave_indices
        },
        "dropped_covariates_by_wave": {
            f"t{w + 1}": dropped_covariates_by_wave[w] for w in wave_indices
        },
        "effective_adjustment_by_timepoint": {
            f"t{w + 1}": {
                **effective_adjustment(
                    spec,
                    wave_subsets[w],
                    adjust_for=tuple(wave_covariates[w]),
                ),
                "requested": covariates,
                "dropped_constant": dropped_covariates_by_wave[w],
            }
            for w in wave_indices
        },
        "averaging_population": "all fitted rows at the wave (descriptive)",
        "predictor_slope_sigma": sigma0,
        "standardisation": (
            "same-wave Haldane-corrected logit, standardised within each wave"
        ),
        "n_published_fits": int(len(fit_diagnostics_df)),
        "all_published_fits_converged": all_fits_converged,
        "n_failed_or_unchecked_fits": int(
            (~fit_diagnostics_df["converged"].eq(True)).sum()
        ),
        "output_contract": (
            "concurrent_associations.csv contains mutually adjusted and single-skill "
            "comparator logit, "
            "probability and items summaries for +1 SD; concurrent_marginals.csv "
            "contains both fit kinds for +1 SD and +k items"
        ),
    }
    write_run_metadata(ctx, extra=meta_extra)

    return finalize_report(ctx)


def _plot_concurrent(
    ctx: StatisticalFitContext, df: pd.DataFrame, hdi: float, *, primary_tp: int
) -> None:
    """Forest of adjusted vs single-skill coefficients for the primary wave (#312)."""
    if df.empty:
        return
    d = df[df["timepoint"] == primary_tp].reset_index(drop=True)
    if d.empty:
        return
    y = np.arange(len(d))[::-1]
    plt.figure(figsize=(7.0, 0.6 * len(d) + 1.6))
    plt.errorbar(
        d["adj_mean"], y + 0.12,
        xerr=[d["adj_mean"] - d["adj_lo"], d["adj_hi"] - d["adj_mean"]],
        fmt="o", color=COLOUR_BLUE, capsize=3, label="adjusted (mutual)",
    )
    plt.errorbar(
        d["biv_mean"], y - 0.12,
        xerr=[d["biv_mean"] - d["biv_lo"], d["biv_hi"] - d["biv_mean"]],
        fmt="s", color="#999999", capsize=3, label="single-skill (trait-adjusted)",
    )
    plt.axvline(0.0, color="grey", ls=":", lw=1)
    plt.yticks(y, d["label"])
    plt.xlabel(
        f"Standardised coefficient (per-SD, logit scale); {int(hdi * 100)}% interval"
    )
    plt.title(f"Concurrent associations at t{primary_tp} (between-child)")
    plt.legend(fontsize=8, loc="best")
    # NB: distinct stem from ``concurrent_associations.csv`` (the full wave×predictor
    # table) — save_styled_figure(data=...) writes a sidecar ``{stem}.csv`` of just the
    # plotted (primary-wave) rows, which would otherwise clobber the full table.
    save_styled_figure(ctx.output_dir, "concurrent_associations_forest", data=d)
