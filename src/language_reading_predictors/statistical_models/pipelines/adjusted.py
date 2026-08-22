# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Between-child adjusted-associations orchestration (``kind="adjusted"``, LRP65).

``fit_adjusted`` is the mutually-adjusted between-child regression of full-study
gain on the T1 predictor set — one row per child — with three sensitivity layers
beside it: the bivariate baseline-only association for each predictor, a sweep over
the predictor-slope prior scale, and a complete-case SES fit. ``fit_rlm_adjusted``
is the same family on the Byrne (RLM) span frame, pooled across three groups with
non-interpretable group-nuisance dummies; ``definitions.KINDS`` keys both as
``adjusted``, and they publish the same ``predictor_associations.csv`` schema, so
they live together here. RLM plans may target all three observational groups or
a pre-specified subset; nuisance dummies are fitted only when the selected frame
contains more than one group.

Nothing in either cohort is randomised. Every predictor slope is an adjusted
association, and the natural-scale contrasts translate a +1 SD difference into
outcome items for readability — not into an effect of changing the predictor.
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
    adjusted as _adjusted,
    diagnostics as _diag,
    factories as _factories,
    reporting as _report,
)
from language_reading_predictors.statistical_models.artifacts import save_table
from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
    make_context,
)
from language_reading_predictors.statistical_models.plotting import save_styled_figure
from language_reading_predictors.statistical_models.preprocessing import (
    load_and_prepare,
)
from language_reading_predictors.statistical_models.prior_artifacts import (
    at_mean_pushforward_rows,
    marginal_pushforward_rows,
    pushforward_n_trials,
    pushforward_outcome_label,
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


# Human-readable labels for the LRP65 predictor keys (for tables / forest plot).
_ADJ_LABELS = {
    "L": "Letter sounds (T1)",
    "lang": "Language composite (T1)",
    "B": "Blending (T1)",
    "age": "Age (T1)",
    "blocks": "Non-verbal MA (T1)",
    "behav": "Behaviour (T1)",
    # Revised-DAG upstream traits, entered as tested covariates (#247). Hearing
    # status is a 0/1 flag entered standardised like every other predictor, so its
    # "+1 SD" rows are roughly half the clear-versus-impaired contrast — say so in
    # the label, since the tables never otherwise distinguish a binary covariate
    # (2026-08-22 review, finding 10).
    "hs": "Hearing status (T1, 0/1 flag; per SD)",
    "hs_missing": "Hearing missing (indicator)",
    "deapp_c": "Speech production (T1)",
    "deapp_c_missing": "Speech missing (indicator)",
    "erbto": "Phonological memory (T1)",
    "erbto_missing": "Phon. memory missing (indicator)",
    "mumedupost16": "SES: mother post-16 educ.",
    "dadedupost16": "SES: father post-16 educ.",
}


def _adj_label(key: str) -> str:
    return _ADJ_LABELS.get(key, key)


#: Column order of ``predictor_associations.csv`` — identical for the RLI and Byrne
#: ports (2026-08-22 review, finding 9). ``adj_*`` / ``biv_*`` are the prefixes the
#: concurrent family also uses for its adjusted-versus-single-skill table.
PREDICTOR_ASSOCIATION_COLUMNS: tuple[str, ...] = (
    "predictor", "label",
    "adj_median", "adj_mean", "adj_lo", "adj_hi", "adj_lo50", "adj_hi50",
    "adj_prob_pos",
    "biv_median", "biv_mean", "biv_lo", "biv_hi", "biv_lo50", "biv_hi50",
    "biv_prob_pos",
    "adj_converged", "biv_converged",
)

#: Column order of ``prior_sensitivity.csv`` — identical for every port. The fitted
#: model is the first block (its own slope / own-baseline prior SDs), then one block
#: per swept slope SD and one per swept own-baseline SD; ``subfit_converged`` is
#: the primary gate verdict for the fitted block and the sub-fit verdict otherwise.
PRIOR_SENSITIVITY_COLUMNS: tuple[str, ...] = (
    "predictor_slope_sigma", "gamma_own_sigma", "predictor", "label",
    "median", "mean", "lo", "hi", "lo50", "hi50", "prob_pos",
    "subfit_converged",
)


def reported_predictors(predictors: list[str]) -> list[str]:
    """The predictors the family *reports*: every key except the missing-data indicators.

    ``{cov}_missing`` coefficients are subgroup mean-offsets under the
    missing-indicator method — nuisance terms (Greenland & Finkle 1995), labelled so
    in the priors table (#384) and kept out of ``predictor_associations.csv``. The
    natural-scale table, the prior pushforward and the key-findings ranking have to
    use the same list, or an indicator can be headlined as "the clearest adjusted
    predictor" (2026-08-22 review, finding 3).
    """
    return [key for key in predictors if not key.endswith("_missing")]


def _association_row(
    key: str,
    label: str,
    adjusted: dict,
    bivariate: dict,
    *,
    adj_converged,
    biv_converged,
) -> dict:
    return {
        "predictor": key,
        "label": label,
        "adj_median": adjusted["median"],
        "adj_mean": adjusted["mean"],
        "adj_lo": adjusted["lo"],
        "adj_hi": adjusted["hi"],
        "adj_lo50": adjusted["lo50"],
        "adj_hi50": adjusted["hi50"],
        "adj_prob_pos": adjusted["prob_pos"],
        "biv_median": bivariate["median"],
        "biv_mean": bivariate["mean"],
        "biv_lo": bivariate["lo"],
        "biv_hi": bivariate["hi"],
        "biv_lo50": bivariate["lo50"],
        "biv_hi50": bivariate["hi50"],
        "biv_prob_pos": bivariate["prob_pos"],
        # Convergence flags: the adjusted column is the primary (gated) fit; the
        # bivariate column is a sub-fit that bypasses the primary gate (B1).
        "adj_converged": adj_converged,
        "biv_converged": biv_converged,
    }


def _sensitivity_rows(
    trace,
    predictors: list[str],
    labels,
    *,
    predictor_slope_sigma: float,
    gamma_own_sigma: float,
    converged,
    ci_prob: float,
) -> list[dict]:
    """One ``prior_sensitivity.csv`` row per reported predictor for one prior setting."""
    rows = []
    for key in predictors:
        summary = beta_summary(trace, f"beta_{key}", ci_prob)
        rows.append(
            {
                "predictor_slope_sigma": float(predictor_slope_sigma),
                "gamma_own_sigma": float(gamma_own_sigma),
                "predictor": key,
                "label": labels(key),
                **{k: summary[k] for k in ("median", "mean", "lo", "hi", "lo50", "hi50", "prob_pos")},
                "subfit_converged": converged,
            }
        )
    return rows


def _prior_sweep_table(
    ctx: StatisticalFitContext,
    *,
    plan: _adjusted.AdjustedRunPlan,
    build,
    predictors: list[str],
    labels,
    primary_converged,
    ci_prob: float,
) -> pd.DataFrame:
    """The slope-prior and own-baseline-prior sweep, shared by all three ports.

    ``build(predictor_slope_sigma=..., gamma_own_sigma=...)`` returns the built
    model for one prior setting. The fitted model contributes the first block from
    ``ctx.trace`` (no refit); every other block is a ``run_subfit`` with its own
    provenance row. The own-baseline sweep is the ``gamma_own_prior`` docstring's
    "required 0.25-vs-0.5 sensitivity", which this family did not run before the
    2026-08-22 review (finding 5).
    """
    rows: list[dict] = []
    rows.extend(
        _sensitivity_rows(
            ctx.trace,
            predictors,
            labels,
            predictor_slope_sigma=plan.predictor_slope_sigma,
            gamma_own_sigma=plan.gamma_own_sigma,
            converged=primary_converged,
            ci_prob=ci_prob,
        )
    )
    settings = [
        *((sigma, plan.gamma_own_sigma, f"sigma={sigma}") for sigma in plan.prior_sensitivity_sigmas),
        *(
            (plan.predictor_slope_sigma, sigma, f"gamma_own_sigma={sigma}")
            for sigma in plan.gamma_own_sensitivity_sigmas
        ),
    ]
    for slope_sigma, own_sigma, tag in settings:
        candidate = build(
            predictor_slope_sigma=float(slope_sigma), gamma_own_sigma=float(own_sigma)
        )
        result = run_subfit(
            ctx,
            candidate,
            label=f"{ctx.spec.model_id} prior-sweep {tag}",
            role="prior_sweep",
        )
        rows.extend(
            _sensitivity_rows(
                result.trace,
                predictors,
                labels,
                predictor_slope_sigma=slope_sigma,
                gamma_own_sigma=own_sigma,
                converged=result.converged,
                ci_prob=ci_prob,
            )
        )
    return pd.DataFrame(rows, columns=list(PRIOR_SENSITIVITY_COLUMNS))


def _write_influence(ctx: StatisticalFitContext) -> tuple:
    """Persist the per-child PSIS-LOO Pareto-k table the results partial reads.

    Shared by all three ports: the Byrne fits previously wrote only the shared
    ``pareto_k.csv`` and the report's Influence section printed its how-to-read
    prose under "No per-child influence table." (2026-08-22 review, finding 7).
    """
    section_header("Influence (PSIS-LOO Pareto-k)")
    infl_df, k_thr, n_flagged = _diag.influence_diagnostics(ctx)
    if infl_df is not None:
        save_table(ctx, "influence", infl_df)
        rprint(
            f"  max Pareto-k = {infl_df['pareto_k'].max():.2f}; "
            f"{n_flagged} of {len(infl_df)} LOO units exceed k = {k_thr:.2f}"
        )
    else:
        rprint("[yellow]Pareto-k unavailable from LOO; influence check skipped[/yellow]")
    return infl_df, k_thr, n_flagged


def _plot_associations(ctx: StatisticalFitContext, df: pd.DataFrame, hdi: float) -> None:
    y = np.arange(len(df))[::-1]
    plt.figure(figsize=(7.0, 0.6 * len(df) + 1.6))
    plt.errorbar(
        df["adj_mean"], y + 0.12,
        xerr=[df["adj_mean"] - df["adj_lo"], df["adj_hi"] - df["adj_mean"]],
        fmt="o", color=COLOUR_BLUE, capsize=3, label="adjusted (mutual)",
    )
    plt.errorbar(
        df["biv_mean"], y - 0.12,
        xerr=[df["biv_mean"] - df["biv_lo"], df["biv_hi"] - df["biv_mean"]],
        fmt="s", color="#999999", capsize=3, label="bivariate (baseline-only)",
    )
    plt.axvline(0.0, color="grey", ls=":", lw=1)
    plt.yticks(y, df["label"])
    plt.xlabel(
        f"Standardised coefficient (per-SD, logit scale); {int(hdi * 100)}% interval"
    )
    plt.title(ctx.spec.title)
    plt.legend(fontsize=8, loc="best")
    save_styled_figure(
        ctx.output_dir, "predictor_associations", data=df
    )


def _natural_scale_contrasts(
    ctx: StatisticalFitContext, prepared, headline: list, outcome: str, hdi: float
) -> pd.DataFrame:
    """Predicted +1 SD contrast for each predictor on the natural (words) scale.

    For two children with the *same* baseline word reading (held at the sample
    mean) who differ by one standard deviation on a single predictor (others at
    their mean), the model-implied difference in word-reading count at the final
    wave — i.e. the differential gain, in words out of ``N``. Computed per
    posterior draw then summarised, so the interval carries the full uncertainty.
    This turns the per-SD logit coefficients into something a teacher can read.

    This is a single *at-the-mean* operating point (``alpha + gamma_own · mean
    pre-logit``, every standardised predictor at zero), not the row-averaged
    contrast of the stacked transition design; on a logit model the two differ by
    a few tenths of an item here. The prior pushforward uses the same functional
    (``prior_artifacts.at_mean_pushforward_rows``), and the recipe, the results
    partial and ``config.json`` state the operating point (2026-08-22 review,
    finding 6). ``headline`` should be the *reported* predictors — the missing-data
    indicators are nuisance subgroup offsets and belong in no published table.
    """
    from scipy.special import expit

    post = ctx.trace.posterior
    N = prepared.n_trials[outcome]
    mean_pre_logit = float(np.mean(prepared.pre_logit[outcome]))

    def draws(name: str) -> np.ndarray:
        return post[name].stack(sample=("chain", "draw")).values

    # All standardised predictors at their mean (z = 0); baseline at sample mean.
    base_eta = draws("alpha") + draws("gamma_own") * mean_pre_logit
    base_words = N * expit(base_eta)

    lo_q, hi_q = (1 - hdi) / 2, 1 - (1 - hdi) / 2
    rows = []
    for k in headline:
        delta = N * expit(base_eta + draws(f"beta_{k}")) - base_words
        rows.append(
            {
                "predictor": k,
                "label": _adj_label(k),
                "delta_words_median": float(np.median(delta)),
                "delta_words_mean": float(np.mean(delta)),
                "delta_words_lo": float(np.quantile(delta, lo_q)),
                "delta_words_hi": float(np.quantile(delta, hi_q)),
                "delta_words_lo50": float(np.quantile(delta, 0.25)),
                "delta_words_hi50": float(np.quantile(delta, 0.75)),
                "prob_pos": float(np.mean(delta > 0)),
            }
        )
    return pd.DataFrame(rows)


def fit_adjusted(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Between-child adjusted fit (LRP65): independent T1 predictors of gain.

    Headline = the mutually-adjusted between-child regression (one row per child,
    T1 baselines, full-study gain ``W_last | W_T1``). Also fits, per the brief:
    the bivariate (baseline-only-adjusted) association for each predictor; a
    prior-sensitivity sweep over the predictor-slope sigma; and a complete-case
    SES sensitivity fit. Writes ``predictor_associations.csv`` (+ forest plot),
    ``prior_sensitivity.csv`` and ``ses_sensitivity.csv`` alongside the standard
    trace / diagnostics / LOO / PPC artefacts.
    """
    require_spec(spec, "adjusted")
    plan = _adjusted.resolve_adjusted_run_plan(spec)
    if plan.port != "rli":
        raise ValueError(f"{spec.model_id}: RLM settings require fit_rlm_adjusted")
    outcome = plan.outcome_symbol
    post_time = plan.post_time
    assert post_time is not None
    lang_symbols = plan.language_composite_symbols
    ses_covs = list(plan.ses_covariates)
    sigma0 = plan.predictor_slope_sigma
    prior_sens = list(plan.prior_sensitivity_sigmas)

    # House-standard 89% equal-tailed intervals (METHODS.md; see
    # notes/202607172359-credible-interval-standard.md).
    ctx = make_context(spec, config, ci_prob=0.89)
    ctx.resolved_plan = plan
    _report.write_model_recipe(ctx)
    hdi = ctx.reporting.ci_prob

    section_header("Prepare data")
    prepared = load_and_prepare(**plan.rli_prepare_kwargs())
    ctx.prepared = prepared
    # Drop any covariate the loader removed as constant on the fitted rows (e.g. a
    # `_missing` indicator that is all-zero once the complete cases are kept) so the
    # model never requests a coefficient for a term that was never estimated (#247).
    covariates = tuple(
        symbol
        for symbol in plan.declared_covariates
        if symbol in prepared.covariates
    )
    if covariates != plan.active_covariates:
        plan = plan.with_active_covariates(covariates)
        ctx.resolved_plan = plan
        _report.write_model_recipe(ctx)
    # Headline predictor key order: skills, language composite, age, tested covariates.
    headline = list(plan.headline_predictors())
    print_header(ctx)

    section_header("Build model")
    built = _factories.build_adjusted_model(
        prepared,
        **plan.rli_factory_kwargs(),
    )
    attach_built(ctx, built)

    render_model_graph(ctx)

    beta_names = [f"beta_{k}" for k in headline]
    _adjusted_diag_vars = ["alpha", "gamma_own", "kappa", *beta_names]
    # Capture the primary gate verdict so the sub-fit tables can label their
    # primary-derived rows (the adjusted/mutual associations and the headline-sigma
    # prior-sweep rows come from ``ctx.trace``, which this gate covers) consistently
    # with the sub-fits' own ``subfit_convergence`` flags (this review's finding B1).
    _primary_gate = shared_stages().run_primary_fit(
        ctx,
        PrimaryFitPlan(
            diagnostic_vars=tuple(_adjusted_diag_vars),
            plot_prior_predictive=lambda c: _diag.save_prior_predictive_plot(c, outcome),
            # This RLI fit has historically written PPC before power scaling.
            psense_timing="after_ppc",
            compute_loo=plan.compute_loo,
        ),
    )
    _primary_converged = _report.convergence_gate_clean_passed(_primary_gate)
    _diag.save_prior_posterior_plot(ctx, var_names=_adjusted_diag_vars)

    # --- Adjusted vs bivariate associations --------------------------------
    section_header("Predictor associations (adjusted vs bivariate)")
    # ``reported`` is the one list the associations table, the bivariate
    # sub-fits, the natural-scale table, the prior pushforward and the prior sweep
    # all share: every headline key except the missing-data indicators, which are
    # nuisance subgroup offsets and belong in no published table (finding 3).
    reported = reported_predictors(headline)
    adjusted = {k: beta_summary(ctx.trace, f"beta_{k}", hdi) for k in reported}
    bivariate: dict[str, dict] = {}
    biv_converged: dict[str, object] = {}
    for k in reported:
        b = _factories.build_adjusted_model(
            prepared,
            **plan.rli_factory_kwargs(predictors=(k,)),
        )
        res = run_subfit(
            ctx, b, label=f"{spec.model_id} bivariate {k}", role="bivariate"
        )
        bivariate[k] = beta_summary(res.trace, f"beta_{k}", hdi)
        biv_converged[k] = res.converged

    rows = [
        _association_row(
            k,
            _adj_label(k),
            adjusted[k],
            bivariate[k],
            adj_converged=_primary_converged,
            biv_converged=biv_converged[k],
        )
        for k in reported
    ]
    # Missing-data-indicator coefficients are subgroup mean-offsets under the
    # missing-indicator method, not interpretable predictor associations — the same
    # basis on which the prior table now labels them nuisance (the missing-indicator
    # sweep in _prior_table_overrides; #384 review, Frank). Keep them out of the
    # reported associations table + forest so it does not contradict that nuisance
    # label; they remain in the fitted model (as adjusters) and in the full
    # diagnostics summary above. ``reported`` is the one list the associations
    # table, the natural-scale table, the prior pushforward and the prior sweep all
    # share (2026-08-22 review, finding 3).
    assoc_df = pd.DataFrame(
        [row for row in rows if row["predictor"] in reported],
        columns=list(PREDICTOR_ASSOCIATION_COLUMNS),
    )
    save_table(ctx, "predictor_associations", assoc_df)
    # Estimand-scale prior check on the headline adjusted associations (#381),
    # pushed through the same at-the-mean functional as the posterior contrast
    # below (2026-08-22 review, finding 6).
    _pf_n = pushforward_n_trials(ctx, outcome)
    _pf_outcome = pushforward_outcome_label(ctx, outcome)
    write_prior_pushforward(
        ctx,
        at_mean_pushforward_rows(
            ctx,
            [
                (
                    f"beta_{r.predictor}",
                    f"the adjusted association of +1 SD {r.label} with {_pf_outcome}",
                )
                for r in assoc_df.itertuples()
            ],
            n_trials=_pf_n,
            own_pre_logit_mean=float(np.mean(ctx.prepared.pre_logit[outcome])),
        ),
    )
    print_table(
        ranked_dataframe_table(
            assoc_df,
            title=(
                f"Predictor associations (per-SD, logit; {int(hdi * 100)}% interval)"
            ),
            columns=[
                "label", "adj_mean", "adj_lo", "adj_hi", "adj_prob_pos",
                "biv_mean", "biv_lo", "biv_hi",
            ],
            rank_column=False,
            precision=3,
        )
    )
    _plot_associations(ctx, assoc_df, hdi)

    # --- Prior sensitivity (does the clear-zero conclusion move?) ----------
    section_header("Prior sensitivity (slope and own-baseline prior SDs)")
    ps_df = _prior_sweep_table(
        ctx,
        plan=plan,
        build=lambda **kw: _factories.build_adjusted_model(
            prepared, **{**plan.rli_factory_kwargs(), **kw}
        ),
        predictors=reported,
        labels=_adj_label,
        primary_converged=_primary_converged,
        ci_prob=hdi,
    )
    save_table(ctx, "prior_sensitivity", ps_df)

    # --- SES complete-case sensitivity -------------------------------------
    section_header("SES sensitivity (complete cases)")
    ses_df = None
    ses_n = None
    ses_error = None
    try:
        prepared_ses = load_and_prepare(
            **plan.rli_prepare_kwargs(include_ses=True),
        )
        # Re-filter against the SES-complete subset: a `_missing` indicator can go
        # constant on this smaller subset even if it survived the headline fit, and the
        # loader then drops it — so rebuild the predictor list here too, or
        # ``build_adjusted_model`` would KeyError on the dropped term (#287 review). The
        # non-covariate predictors (skills / lang / age) are always kept.
        ses_headline = [
            k
            for k in headline
            if k not in plan.active_covariates or k in prepared_ses.covariates
        ]
        ses_covs_fit = [c for c in ses_covs if c in prepared_ses.covariates]
        ses_predictors = ses_headline + ses_covs_fit
        b = _factories.build_adjusted_model(
            prepared_ses,
            **plan.rli_factory_kwargs(predictors=ses_predictors),
        )
        res = run_subfit(
            ctx, b, label=f"{spec.model_id} SES complete-case", role="sensitivity"
        )
        ses_n = int(b.prepared.n_children)
        ses_rows = [
            {
                "predictor": k,
                "label": _adj_label(k),
                "n_children": ses_n,
                **beta_summary(res.trace, f"beta_{k}", hdi),
                "converged": res.converged,
            }
            for k in ses_predictors
        ]
        ses_df = pd.DataFrame(ses_rows)
        save_table(ctx, "ses_sensitivity", ses_df)
        rprint(f"  SES sensitivity fit on {ses_n} complete-case children")
    except Exception as exc:  # pragma: no cover
        # Record the failure (type + message + traceback) rather than swallowing
        # it to a one-line warning: a genuine bug (missing column, factory error)
        # should not silently produce a "successful" reporting run with no
        # ses_sensitivity.csv. The error is surfaced in the run metadata.
        import traceback

        ses_error = f"{type(exc).__name__}: {exc}"
        rprint(f"[red]SES sensitivity fit failed: {ses_error}[/red]")
        rprint(f"[yellow]{traceback.format_exc()}[/yellow]")

    # --- Natural-scale interpretation (predicted gain, in words) -----------
    section_header("Predicted gain on the natural (words) scale")
    words_df = _natural_scale_contrasts(ctx, ctx.prepared, reported, outcome, hdi)
    save_table(ctx, "predicted_gain_words", words_df)
    print_table(
        ranked_dataframe_table(
            words_df,
            title=(
                f"Predicted differential gain per +1 SD (words out of "
                f"{ctx.prepared.n_trials[outcome]}; {int(hdi * 100)}% interval)"
            ),
            columns=[
                "label", "delta_words_mean", "delta_words_lo",
                "delta_words_hi", "prob_pos",
            ],
            rank_column=False,
            precision=2,
        )
    )

    # --- Influence (does the fit rest on a few children?) ------------------
    infl_df, k_thr, n_flagged = _write_influence(ctx)

    write_run_metadata(
        ctx,
        extra={
            "loo_elpd": float(ctx.loo.elpd) if ctx.loo is not None else None,
            "design": "between_child",
            "post_time": post_time,
            "predictors": headline,
            "reported_predictors": reported,
            "predictor_slope_sigma": sigma0,
            "prior_sensitivity_sigmas": prior_sens,
            "gamma_own_sigma": plan.gamma_own_sigma,
            "gamma_own_sensitivity_sigmas": list(plan.gamma_own_sensitivity_sigmas),
            "contrast_operating_point": "at_mean",
            "language_composite_symbols": list(lang_symbols),
            "n_children": int(ctx.prepared.n_children),
            "ses_n_children": ses_n,
            "ses_error": ses_error,
            "associations": rows,
            "predicted_gain_words": words_df.to_dict("records"),
            "max_pareto_k": (
                float(infl_df["pareto_k"].max()) if infl_df is not None else None
            ),
            "n_pareto_k_flagged": n_flagged,
        },
    )

    return finalize_report(ctx)


def rlm_nuisance_names(frame) -> list[str]:
    """The group-nuisance coefficient names the RLM factories create."""
    codes = sorted(set(np.asarray(frame.group_code, dtype=int)))
    counts = {c: int((frame.group_code == c).sum()) for c in codes}
    reference = max(counts, key=lambda c: (counts[c], -c))
    return [
        "beta_group_nuisance_"
        + frame.group_labels[c].lower().replace(" ", "_").replace("-", "_")
        for c in codes
        if c != reference
    ]


def _rlm_natural_scale_contrasts(
    ctx: StatisticalFitContext, frame, headline: list, hdi: float
) -> pd.DataFrame:
    """Predicted +1 SD contrast per predictor on the items scale (RLM span frame).

    The Byrne analogue of :func:`_natural_scale_contrasts`: for two children with
    the same pre-wave outcome score (held at the sample mean) who differ by one
    SD on a single predictor, the model-implied difference in outcome items at
    the later wave, per posterior draw.
    """
    from scipy.special import expit

    post = ctx.trace.posterior
    outcome = frame.outcome
    N = frame.n_trials[outcome]
    mean_pre_logit = float(np.mean(frame.pre_logit[outcome]))

    def draws(name: str) -> np.ndarray:
        return post[name].stack(sample=("chain", "draw")).values

    base_eta = draws("alpha") + draws("gamma_own") * mean_pre_logit
    base_items = N * expit(base_eta)
    lo_q, hi_q = (1 - hdi) / 2, 1 - (1 - hdi) / 2
    rows = []
    for k in headline:
        delta = N * expit(base_eta + draws(f"beta_{k}")) - base_items
        rows.append(
            {
                "predictor": k,
                "label": frame.predictor_labels.get(k, k),
                "delta_words_median": float(np.median(delta)),
                "delta_words_mean": float(np.mean(delta)),
                "delta_words_lo": float(np.quantile(delta, lo_q)),
                "delta_words_hi": float(np.quantile(delta, hi_q)),
                "delta_words_lo50": float(np.quantile(delta, 0.25)),
                "delta_words_hi50": float(np.quantile(delta, 0.75)),
                "prob_pos": float(np.mean(delta > 0)),
            }
        )
    return pd.DataFrame(rows)


def _rlm_transition_natural_scale_contrasts(
    ctx: StatisticalFitContext, frame, headline: list[str], hdi: float
) -> pd.DataFrame:
    """Average +1 within-transition SD contrasts on the outcome-item scale.

    Operating point: the contrast ``N · [expit(eta_fixed + β) − expit(eta_fixed)]``
    is taken at every fitted row's own covariates with the child random intercept
    at zero (``eta_fixed`` excludes ``u_child``) and averaged over rows — a
    row-averaged, median-child contrast, unlike the span ports' single
    at-the-mean operating point. Stated in the recipe, the results partial and
    ``config.json`` (``contrast_operating_point``; 2026-08-22 review, finding 6).
    """
    from scipy.special import expit

    posterior = ctx.trace.posterior
    n_trials = frame.n_trials[frame.outcome]
    eta_fixed = (
        posterior["eta_fixed"]
        .stack(sample=("chain", "draw"))
        .transpose("obs_id", "sample")
        .values
    )
    base_items = n_trials * expit(eta_fixed)
    lo_q, hi_q = (1 - hdi) / 2, 1 - (1 - hdi) / 2
    rows = []
    for key in headline:
        beta = posterior[f"beta_{key}"].stack(sample=("chain", "draw")).values
        delta = (
            n_trials * expit(eta_fixed + beta[np.newaxis, :]) - base_items
        ).mean(axis=0)
        rows.append(
            {
                "predictor": key,
                "label": frame.predictor_labels.get(key, key),
                "delta_words_median": float(np.median(delta)),
                "delta_words_mean": float(np.mean(delta)),
                "delta_words_lo": float(np.quantile(delta, lo_q)),
                "delta_words_hi": float(np.quantile(delta, hi_q)),
                "delta_words_lo50": float(np.quantile(delta, 0.25)),
                "delta_words_hi50": float(np.quantile(delta, 0.75)),
                "prob_pos": float(np.mean(delta > 0)),
            }
        )
    return pd.DataFrame(rows)


def _draw_summary(draws: np.ndarray, ci_prob: float) -> dict[str, float]:
    values = np.asarray(draws, dtype=float).reshape(-1)
    lo_q, hi_q = (1 - ci_prob) / 2, 1 - (1 - ci_prob) / 2
    return {
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "lo": float(np.quantile(values, lo_q)),
        "hi": float(np.quantile(values, hi_q)),
        "lo50": float(np.quantile(values, 0.25)),
        "hi50": float(np.quantile(values, 0.75)),
        "prob_pos": float(np.mean(values > 0)),
    }


def _rlm_transition_analysis_set(frame) -> pd.DataFrame:
    rows = []
    for phase, label in enumerate(frame.transition_labels):
        for code, group_label in frame.group_labels.items():
            rows.append(
                {
                    "transition": label,
                    "pre_wave": frame.transition_waves[phase],
                    "post_wave": frame.transition_waves[phase + 1],
                    "group_code": code,
                    "group_label": group_label,
                    "n_rows": frame.transition_group_counts[label].get(code, 0),
                    "transition_total": frame.transition_n_obs[label],
                    "eligible_children": frame.eligible_n_children,
                    "missing_required_transition_rows": (
                        frame.eligible_n_children - frame.transition_n_obs[label]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _fit_rlm_transition_adjusted(
    spec: ModelSpec,
    plan: _adjusted.AdjustedRunPlan,
    config: str,
) -> StatisticalFitContext:
    """Fit the stacked annual-transition branch of the Byrne adjusted family."""
    from language_reading_predictors.statistical_models.preprocessing import (
        load_rlm_transition_frame,
    )

    outcome = plan.outcome_symbol
    waves = plan.transition_waves
    assert waves is not None
    sigma0 = plan.predictor_slope_sigma
    prior_sens = list(plan.prior_sensitivity_sigmas)
    ctx = make_context(spec, config, ci_prob=0.89)
    ctx.resolved_plan = plan
    _report.write_model_recipe(ctx)
    hdi = ctx.reporting.ci_prob

    section_header("Prepare stacked transition data")
    frame = load_rlm_transition_frame(**plan.rlm_prepare_kwargs())
    ctx.prepared = frame
    headline = list(frame.predictors)
    print_header(ctx)
    save_table(ctx, "analysis_set_by_transition", _rlm_transition_analysis_set(frame))

    section_header("Build pooled transition model")
    built = _factories.build_rlm_transition_adjusted_model(
        frame, **plan.rlm_factory_kwargs(headline)
    )
    attach_built(ctx, built)
    render_model_graph(ctx)
    nuisance = rlm_nuisance_names(frame)
    diag_vars = [
        "alpha_transition",
        "gamma_own",
        "sigma_child",
        "kappa",
        *(f"beta_{key}" for key in headline),
        *nuisance,
    ]
    primary_gate = shared_stages().run_primary_fit(
        ctx,
        PrimaryFitPlan(
            diagnostic_vars=tuple(diag_vars),
            plot_prior_predictive=lambda c: _diag.save_prior_predictive_plot(
                c, outcome, node="y_post"
            ),
            compute_loo=plan.compute_loo,
        ),
    )
    primary_converged = _report.convergence_gate_clean_passed(primary_gate)
    _diag.save_prior_posterior_plot(ctx, var_names=diag_vars)

    section_header("Predictor associations (adjusted vs bivariate)")
    adjusted = {key: beta_summary(ctx.trace, f"beta_{key}", hdi) for key in headline}
    bivariate: dict[str, dict] = {}
    bivariate_converged: dict[str, object] = {}
    for key in headline:
        candidate = _factories.build_rlm_transition_adjusted_model(
            frame, **plan.rlm_factory_kwargs((key,))
        )
        result = run_subfit(
            ctx,
            candidate,
            label=f"{spec.model_id} bivariate {key}",
            role="bivariate",
        )
        bivariate[key] = beta_summary(result.trace, f"beta_{key}", hdi)
        bivariate_converged[key] = result.converged
    reported = reported_predictors(headline)
    associations = pd.DataFrame(
        [
            _association_row(
                key,
                frame.predictor_labels.get(key, key),
                adjusted[key],
                bivariate[key],
                adj_converged=primary_converged,
                biv_converged=bivariate_converged[key],
            )
            for key in reported
        ],
        columns=list(PREDICTOR_ASSOCIATION_COLUMNS),
    )
    save_table(ctx, "predictor_associations", associations)
    _plot_associations(ctx, associations, hdi)
    write_prior_pushforward(
        ctx,
        marginal_pushforward_rows(
            ctx,
            [
                (
                    f"beta_{row.predictor}",
                    f"the pooled annual association of +1 within-transition SD "
                    f"{row.label} with {pushforward_outcome_label(ctx, outcome)}",
                )
                for row in associations.itertuples()
            ],
            n_trials=pushforward_n_trials(ctx, outcome),
            convention="forward",
        ),
    )

    section_header("Items-scale +1 SD contrasts")
    gain_words = _rlm_transition_natural_scale_contrasts(
        ctx, frame, reported, hdi
    )
    save_table(ctx, "predicted_gain_words", gain_words)

    section_header("Prior sensitivity (slope and own-baseline prior SDs)")
    save_table(
        ctx,
        "prior_sensitivity",
        _prior_sweep_table(
            ctx,
            plan=plan,
            build=lambda **kw: _factories.build_rlm_transition_adjusted_model(
                frame, **{**plan.rlm_factory_kwargs(headline), **kw}
            ),
            predictors=reported,
            labels=lambda k: frame.predictor_labels.get(k, k),
            primary_converged=primary_converged,
            ci_prob=hdi,
        ),
    )

    if plan.common_horizon_last_wave is not None:
        section_header("Common-horizon sensitivity")
        cutoff = waves.index(plan.common_horizon_last_wave)
        common_waves = waves[: cutoff + 1]
        common_frame = load_rlm_transition_frame(
            **{
                **plan.rlm_prepare_kwargs(),
                "transition_waves": common_waves,
            }
        )
        candidate = _factories.build_rlm_transition_adjusted_model(
            common_frame, **plan.rlm_factory_kwargs(headline)
        )
        result = run_subfit(
            ctx,
            candidate,
            label=(
                f"{spec.model_id} common horizon through "
                f"wave {plan.common_horizon_last_wave}"
            ),
            role="sensitivity",
        )
        common_rows = []
        for key in headline:
            for analysis, trace, converged, n_obs, n_children in (
                (
                    "all_declared_transitions",
                    ctx.trace,
                    primary_converged,
                    frame.n_obs,
                    frame.n_children,
                ),
                (
                    f"common_horizon_through_w{plan.common_horizon_last_wave}",
                    result.trace,
                    result.converged,
                    common_frame.n_obs,
                    common_frame.n_children,
                ),
            ):
                summary = beta_summary(trace, f"beta_{key}", hdi)
                common_rows.append(
                    {
                        "analysis": analysis,
                        "predictor": key,
                        "label": frame.predictor_labels.get(key, key),
                        **summary,
                        "n_obs": n_obs,
                        "n_children": n_children,
                        "subfit_converged": converged,
                    }
                )
        save_table(
            ctx, "common_horizon_sensitivity", pd.DataFrame(common_rows)
        )

    if plan.per_transition_sensitivity:
        section_header("Transition-specific slope sensitivity")
        candidate = _factories.build_rlm_transition_adjusted_model(
            frame,
            **plan.rlm_factory_kwargs(headline),
            varying_slopes=True,
        )
        result = run_subfit(
            ctx,
            candidate,
            label=f"{spec.model_id} transition-specific slopes",
            role="sensitivity",
        )
        beta = result.trace.posterior["beta_transition"]
        transition_rows = []
        for label in frame.transition_labels:
            for key in headline:
                summary = _draw_summary(
                    beta.sel(transition=label, predictor=key).values, hdi
                )
                transition_rows.append(
                    {
                        "transition": label,
                        "predictor": key,
                        "label": frame.predictor_labels.get(key, key),
                        **summary,
                        "n_obs": frame.transition_n_obs[label],
                        "group_counts": "; ".join(
                            f"{frame.group_labels[code]}={count}"
                            for code, count in frame.transition_group_counts[
                                label
                            ].items()
                        ),
                        "subfit_converged": result.converged,
                    }
                )
        save_table(
            ctx, "transition_slope_sensitivity", pd.DataFrame(transition_rows)
        )

    infl_df, _k_thr, n_flagged = _write_influence(ctx)

    write_run_metadata(
        ctx,
        extra={
            "study_id": "rlm",
            "outcome": outcome,
            "design": plan.design,
            "transition_waves": list(waves),
            "transition_n_obs": frame.transition_n_obs,
            "transition_group_counts": frame.transition_group_counts,
            "predictors": headline,
            "reported_predictors": reported,
            "predictors_standardised_within_transition": True,
            "prior_sensitivity_sigmas": prior_sens,
            "gamma_own_sigma": plan.gamma_own_sigma,
            "gamma_own_sensitivity_sigmas": list(plan.gamma_own_sensitivity_sigmas),
            "contrast_operating_point": "row_averaged_median_child",
            "max_pareto_k": (
                float(infl_df["pareto_k"].max()) if infl_df is not None else None
            ),
            "n_pareto_k_flagged": n_flagged,
            "group_nuisance_terms": nuisance,
            "source_n_children": frame.source_n_children,
            "eligible_n_children": frame.eligible_n_children,
            "n_children": frame.n_children,
            "n_obs": frame.n_obs,
            "loo_unit": "child",
            "final_transition_single_group": (
                len(frame.transition_group_counts[frame.transition_labels[-1]]) == 1
            ),
            "common_horizon_last_wave": plan.common_horizon_last_wave,
            "predictor_slope_sigma": sigma0,
        },
    )
    return finalize_report(ctx)


def fit_rlm_adjusted(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Byrne between-child adjusted fit (#338 Phase D, ``lrp-rlm-adj-001``).

    The RLI ``fit_adjusted`` shape on the Byrne span frame: the mutually-adjusted
    wave-1-predictors -> later-wave outcome regression (all declared groups or a
    pre-specified subset, with non-interpretable nuisance dummies when needed), the
    per-predictor bivariate comparison fits, a slope-prior sensitivity sweep and
    the items-scale +1 SD contrasts. Writes ``predictor_associations.csv``,
    ``predicted_gain_words.csv`` and ``prior_sensitivity.csv`` so the shared
    ``adjusted`` report partial and key-findings builder apply unchanged. Every
    coefficient is an adjusted association - nothing in this cohort is causal.
    """
    from language_reading_predictors.statistical_models.preprocessing import (
        load_rlm_span_frame,
    )

    require_spec(spec, "adjusted")
    plan = _adjusted.resolve_adjusted_run_plan(spec)
    if plan.port != "rlm":
        raise ValueError(f"{spec.model_id}: RLI settings require fit_adjusted")
    if plan.transition_waves is not None:
        return _fit_rlm_transition_adjusted(spec, plan, config)
    outcome = plan.outcome_symbol
    pre_wave = plan.pre_wave
    post_wave = plan.post_wave
    assert pre_wave is not None and post_wave is not None
    sigma0 = plan.predictor_slope_sigma
    prior_sens = list(plan.prior_sensitivity_sigmas)

    # House-standard 89% equal-tailed intervals, as in the RLI port.
    ctx = make_context(spec, config, ci_prob=0.89)
    ctx.resolved_plan = plan
    _report.write_model_recipe(ctx)
    hdi = ctx.reporting.ci_prob

    section_header("Prepare data")
    frame = load_rlm_span_frame(**plan.rlm_prepare_kwargs())
    ctx.prepared = frame
    headline = list(frame.predictors)
    print_header(ctx)

    section_header("Build model")
    built = _factories.build_rlm_adjusted_model(
        frame, **plan.rlm_factory_kwargs(headline)
    )
    attach_built(ctx, built)
    render_model_graph(ctx)

    nuisance = rlm_nuisance_names(frame)
    diag_vars = plan.diagnostic_vars(headline, nuisance)
    _primary_gate = shared_stages().run_primary_fit(
        ctx,
        PrimaryFitPlan(
            diagnostic_vars=tuple(diag_vars),
            plot_prior_predictive=lambda c: _diag.save_prior_predictive_plot(
                c, outcome, node="y_post"
            ),
            compute_loo=plan.compute_loo,
        ),
    )
    _primary_converged = _report.convergence_gate_clean_passed(_primary_gate)
    _diag.save_prior_posterior_plot(ctx, var_names=diag_vars)

    # --- Adjusted vs bivariate associations --------------------------------
    section_header("Predictor associations (adjusted vs bivariate)")
    adjusted = {k: beta_summary(ctx.trace, f"beta_{k}", hdi) for k in headline}
    bivariate: dict[str, dict] = {}
    biv_converged: dict[str, object] = {}
    for k in headline:
        b = _factories.build_rlm_adjusted_model(
            frame, **plan.rlm_factory_kwargs((k,))
        )
        res = run_subfit(
            ctx, b, label=f"{spec.model_id} bivariate {k}", role="bivariate"
        )
        bivariate[k] = beta_summary(res.trace, f"beta_{k}", hdi)
        biv_converged[k] = res.converged
    reported = reported_predictors(headline)
    assoc = pd.DataFrame(
        [
            _association_row(
                k,
                frame.predictor_labels.get(k, k),
                adjusted[k],
                bivariate[k],
                adj_converged=_primary_converged,
                biv_converged=biv_converged[k],
            )
            for k in reported
        ],
        columns=list(PREDICTOR_ASSOCIATION_COLUMNS),
    )
    save_table(ctx, "predictor_associations", assoc)
    _plot_associations(ctx, assoc, hdi)
    # Estimand-scale prior check on the headline adjusted associations (#381),
    # pushed through the same at-the-mean functional as the posterior contrast
    # (reference group, sample-mean own baseline; 2026-08-22 review, finding 6).
    _pf_n = pushforward_n_trials(ctx, outcome)
    _pf_outcome = pushforward_outcome_label(ctx, outcome)
    write_prior_pushforward(
        ctx,
        at_mean_pushforward_rows(
            ctx,
            [
                (
                    f"beta_{r.predictor}",
                    f"the adjusted association of +1 SD {r.label} with {_pf_outcome}",
                )
                for r in assoc.itertuples()
            ],
            n_trials=_pf_n,
            own_pre_logit_mean=float(np.mean(frame.pre_logit[outcome])),
        ),
    )
    print_table(
        ranked_dataframe_table(
            assoc,
            title=f"Wave-{pre_wave} predictors of {outcome} at wave {post_wave} "
            f"(adjusted vs bivariate) - {int(hdi * 100)}% CI",
            columns=[
                "label", "adj_mean", "adj_lo", "adj_hi", "adj_prob_pos",
                "biv_mean", "biv_prob_pos",
            ],
            rank_column=False,
            precision=3,
        )
    )

    # --- Items-scale contrasts (the key-findings headline) ------------------
    section_header("Items-scale +1 SD contrasts")
    gain_words = _rlm_natural_scale_contrasts(ctx, frame, reported, hdi)
    save_table(ctx, "predicted_gain_words", gain_words)

    # --- Prior-sensitivity sweep (slope and own-baseline prior SDs) ----------
    section_header("Prior sensitivity (slope and own-baseline prior SDs)")
    sens = _prior_sweep_table(
        ctx,
        plan=plan,
        build=lambda **kw: _factories.build_rlm_adjusted_model(
            frame, **{**plan.rlm_factory_kwargs(headline), **kw}
        ),
        predictors=reported,
        labels=lambda k: frame.predictor_labels.get(k, k),
        primary_converged=_primary_converged,
        ci_prob=hdi,
    )
    save_table(ctx, "prior_sensitivity", sens)

    infl_df, _k_thr, n_flagged = _write_influence(ctx)

    write_run_metadata(
        ctx,
        extra={
            "study_id": "rlm",
            "outcome": outcome,
            "pre_wave": pre_wave,
            "post_wave": post_wave,
            "predictors": headline,
            "reported_predictors": reported,
            "gamma_own_sigma": plan.gamma_own_sigma,
            "gamma_own_sensitivity_sigmas": list(plan.gamma_own_sensitivity_sigmas),
            "prior_sensitivity_sigmas": prior_sens,
            "contrast_operating_point": "at_mean_reference_group",
            "max_pareto_k": (
                float(infl_df["pareto_k"].max()) if infl_df is not None else None
            ),
            "n_pareto_k_flagged": n_flagged,
            "group_nuisance_terms": nuisance,
            "group_codes": sorted(set(frame.group_code.astype(int))),
            "group_labels": {
                str(code): frame.group_labels[code]
                for code in sorted(set(frame.group_code.astype(int)))
            },
            "source_n_children": frame.source_n_children,
            "eligible_n_children": frame.eligible_n_children,
            "n_children": frame.n_children,
            "predictor_slope_sigma": sigma0,
        },
    )
    return finalize_report(ctx)
