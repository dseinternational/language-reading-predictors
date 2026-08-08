# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Joint-mechanism orchestration (``kind="joint_mechanism"``, #421 Tier 3).

``fit_joint_mechanism`` fits letter-sound knowledge jointly against word reading
and nonword decoding in one of two designs: a *levels* fit carrying a bivariate
residual dependence block on the observation row, and a *transition* fit over
period transitions. Only the levels design yields ``rho_outcome`` and the
conditional slopes, so those terms are emitted conditionally, and a wave with too
few usable rows is not fitted at all — a residual correlation from a handful of
children would be prior-dominated. Nothing here is randomised: every slope is an
adjusted association.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dse_research_utils.plot.styles import COLOUR_BLUE
from rich import print as rprint

from language_reading_predictors.models._reporting import (
    metrics_table,
    print_table,
    section_header,
)
from language_reading_predictors.statistical_models import (
    diagnostics as _diag,
    factories as _factories,
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
from language_reading_predictors.statistical_models.diagnostics import sample_subfit
from language_reading_predictors.statistical_models.plotting import save_styled_figure
from language_reading_predictors.statistical_models.preprocessing import (
    _subset_prepared,
    load_and_prepare,
    split_covariates_by_wave,
)
from language_reading_predictors.statistical_models.publication import (
    print_header,
    render_model_graph,
)
from language_reading_predictors.statistical_models.runtime import (
    attach_built,
    finalize_report,
    require_spec,
    run_sampling_and_loo,
    write_run_metadata,
)


#: Terms reported for both designs, in report order. ``rho_outcome`` and the two
#: conditional-slope terms exist only where the dependence block sits on the
#: observation row (the ``levels`` design), so they are emitted conditionally.
_JM_TERM_LABELS: dict[str, str] = {
    "delta_ls_decoding": "Delta = beta(LS->N) - beta(LS->W) (decoding specificity)",
    "rho_outcome": "residual correlation between the two outcomes",
    "beta_held_on_focal": "implied coefficient of the held-fixed outcome",
    "beta_mech_focal_given_held": "beta(LS->W) holding nonword decoding fixed",
    "share_retained": "share of beta(LS->W) retained when decoding is held fixed",
}

#: Fewest usable rows a wave needs before the levels design will fit it. A bivariate
#: residual covariance on a handful of children is prior-dominated, and reporting a
#: correlation from it would misrepresent what the joint fit adds.
_JM_MIN_WAVE_ROWS = 10

#: Columns every ``joint_mechanism_slopes.csv`` row carries. The house standard is
#: median + inner 50% + outer 89% + tail probability (METHODS.md; #421 acceptance
#: criterion), so the inner interval is part of the contract, not an extra.
_JM_SLOPE_REQUIRED: frozenset[str] = frozenset(
    {
        "wave",
        "term",
        "label",
        "median",
        "mean",
        "lo50",
        "hi50",
        "lo",
        "hi",
        "prob_pos",
        "converged",
    }
)


def _jm_term_summary(
    trace,
    name: str,
    ci_prob: float,
    *,
    outcome: str | None = None,
) -> dict | None:
    """Median / mean / inner-50% / outer-``ci_prob`` / P(>0) for one posterior term.

    Returns ``None`` when the variable is absent, so a design that does not register
    a term (the transition design has no conditional slope) simply omits its rows
    instead of failing.
    """
    if name not in trace.posterior:
        return None
    var = trace.posterior[name]
    if outcome is not None:
        var = var.sel(outcome=outcome)
    draws = np.asarray(var.values).ravel()
    lo_q, hi_q = (1 - ci_prob) / 2, 1 - (1 - ci_prob) / 2
    return {
        "median": float(np.median(draws)),
        "mean": float(np.mean(draws)),
        "lo50": float(np.quantile(draws, 0.25)),
        "hi50": float(np.quantile(draws, 0.75)),
        "lo": float(np.quantile(draws, lo_q)),
        "hi": float(np.quantile(draws, hi_q)),
        "prob_pos": float(np.mean(draws > 0)),
    }


def _jm_slope_rows(
    trace,
    *,
    outcome_symbols: tuple[str, ...],
    contrast: tuple[str, str],
    ci_prob: float,
    wave: str,
    converged: bool | None,
) -> list[dict]:
    """One fit's reported terms as ``joint_mechanism_slopes.csv`` rows."""
    rows: list[dict] = []

    def _add(term: str, label: str, summary: dict | None) -> None:
        if summary is None:
            return
        rows.append(
            {
                "wave": wave,
                "term": term,
                "label": label,
                **summary,
                "converged": converged,
            }
        )

    for sym in outcome_symbols:
        _add(
            f"beta_mech[{sym}]",
            f"letter-sound slope on {sym} (logit per SD)",
            _jm_term_summary(trace, "beta_mech", ci_prob, outcome=sym),
        )
    delta_label = _JM_TERM_LABELS["delta_ls_decoding"].replace(
        "beta(LS->N) - beta(LS->W)", f"beta(LS->{contrast[0]}) - beta(LS->{contrast[1]})"
    )
    _add("delta_ls_decoding", delta_label, _jm_term_summary(trace, "delta_ls_decoding", ci_prob))
    for term in (
        "rho_outcome",
        "beta_held_on_focal",
        "beta_mech_focal_given_held",
        "share_retained",
    ):
        _add(term, _JM_TERM_LABELS[term], _jm_term_summary(trace, term, ci_prob))
    return rows


def _jm_diag_vars(
    trace_or_model_names: set[str],
    *,
    design: str,
    adjust_for: tuple[str, ...],
    confounder_symbols: tuple[str, ...],
    include_group: bool,
) -> list[str]:
    """Reported parameters for the convergence gate and the prior-vs-posterior panel."""
    names = ["alpha", "beta_mech", "delta_ls_decoding"]
    if include_group or "G" in confounder_symbols:
        names.append("beta_group_nuisance" if design == "levels" else "beta_G")
    if "A" in confounder_symbols:
        names.append("gamma_A")
    names += [f"gamma_{c}" for c in adjust_for]
    if design == "levels":
        names += [
            "sigma_u_resid",
            "rho_outcome",
            "beta_mech_focal_given_held",
            "share_retained",
        ]
    else:
        names += ["gamma_own", "alpha_phase", "kappa", "sigma_u_child", "rho_outcome"]
    return [n for n in names if n in trace_or_model_names]


def _jm_marginal_ppc(
    ctx: StatisticalFitContext,
    *,
    outcome_symbols: tuple[str, ...],
    ci_levels: tuple[float, ...] = (0.5, 0.9),
) -> None:
    """New-child predictive coverage, written to ``ppc_summary_marginal.csv``.

    Why this exists. The levels design carries one bivariate latent residual **per
    child**, and each child contributes exactly two likelihood cells — so the residual
    block is saturated and the ordinary posterior predictive, which conditions on the
    fitted residuals, reproduces every observation. The dev fit's conditional coverage
    is 1.00 at both the 50% and 90% levels with intervals only ~8 items wide on a
    79-item test: not a good fit, an arithmetic consequence of the design. Publishing
    that number alone would read as a passed calibration check.

    This redraws each child's residual from its **estimated** distribution
    (Sigma = diag(sigma) Corr diag(sigma), Cholesky-factorised in closed form for the
    2x2 case) instead of reusing the fitted value, giving the predictive for a *new*
    child with the same covariates — the check that can actually fail. Both files are
    written: ``ppc_summary.csv`` keeps the house cross-family schema, and this is the
    informative companion the results partial reads alongside it.

    Done in NumPy rather than via ``pm.sample_posterior_predictive(var_names=[...])``
    deliberately: naming the residual there still conditioned on its posterior values,
    silently reproducing the conditional numbers under a marginal label.
    """
    with guard_optional(
        ctx, "ppc_summary_marginal.csv",
        filename="ppc_summary_marginal.csv", kind="table", verb="skipped",
    ):
        post = ctx.trace.posterior
        required = {"eta", "u_resid", "sigma_u_resid", "rho_outcome"}
        if not required <= set(post.data_vars):
            return
        # Linear predictor with the fitted residual removed.
        eta = np.asarray(post["eta"].values)
        core = eta - np.asarray(post["u_resid"].values)
        n_obs, n_outcomes = core.shape[-2], core.shape[-1]
        core = core.reshape(-1, n_obs, n_outcomes)
        sigma = np.asarray(post["sigma_u_resid"].values).reshape(-1, n_outcomes)
        rho = np.asarray(post["rho_outcome"].values).reshape(-1)
        if n_outcomes != 2:
            return

        rng = np.random.default_rng(ctx.sampling.random_seed)
        z = rng.standard_normal((core.shape[0], n_obs, 2))
        # L = [[s0, 0], [s1 rho, s1 sqrt(1 - rho^2)]]; u = L z.
        tail = np.sqrt(np.clip(1.0 - rho**2, 0.0, None))
        u_new = np.stack(
            [
                sigma[:, None, 0] * z[..., 0],
                sigma[:, None, 1]
                * (rho[:, None] * z[..., 0] + tail[:, None] * z[..., 1]),
            ],
            axis=-1,
        )
        p = 1.0 / (1.0 + np.exp(-(core + u_new)))

        row = np.asarray(ctx.trace.constant_data["y_post_cell_row"].values).astype(int)
        col = np.asarray(
            ctx.trace.constant_data["y_post_cell_outcome"].values
        ).astype(int)
        n_trials = np.array(
            [ctx.prepared.n_trials[s] for s in outcome_symbols], dtype=int
        )[col]
        y_rep = rng.binomial(n_trials[None, :], p[:, row, col])  # (draw, cell)

        y_obs = np.asarray(ctx.trace.observed_data["y_post"].values, dtype=float)
        finite = np.isfinite(y_obs)
        y_rep, y_obs = y_rep[:, finite], y_obs[finite]
        n = int(y_obs.shape[0])
        rows = []
        for level in ci_levels:
            lo = np.quantile(y_rep, (1.0 - level) / 2.0, axis=0)
            hi = np.quantile(y_rep, (1.0 + level) / 2.0, axis=0)
            n_in = int(np.count_nonzero((y_obs >= lo) & (y_obs <= hi)))
            rows.append(
                {
                    "mode": "count_interval_marginal",
                    "node": "y_post",
                    "unit": "observations",
                    "quantity": "observed score (new-child predictive)",
                    "level": float(level),
                    "level_pct": int(round(level * 100)),
                    "n_total": n,
                    "n_inside": n_in,
                    "coverage": float(n_in / n) if n else float("nan"),
                }
            )
        frame = pd.DataFrame(rows)
        save_table(ctx, "ppc_summary_marginal", frame, required=False)


def _jm_standard_artefacts(
    ctx: StatisticalFitContext,
    *,
    outcome_symbols: tuple[str, ...],
    diag_vars: list[str],
    psense_vars: list[str],
    marginal_ppc: bool = False,
) -> dict | None:
    """The shared post-sampling artefact set for a joint-mechanism fit.

    Everything the house technical report expects of a published fit, in the order
    ``stages.py`` uses elsewhere: summary diagnostics, **power-scaling prior
    sensitivity** (#381 — the mechanism family already audits its reported slopes and
    this family must not be the one exemption), per-outcome posterior-predictive
    overlays, the **``ppc_summary.csv`` coverage statistic** (#318), the convergence
    gate, per-outcome LOO-PIT, the trace and the prior-vs-posterior panel.

    The last three were silently absent before the #427 review: the per-outcome
    LOO-PIT calls raised inside :func:`diagnostics.save_joint_loo_pit_plot` because
    the helper hard-required ``posterior['tau']``, which this family does not have.
    ``posterior_var="beta_mech"`` now names the family's own reported coefficient.
    """
    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=diag_vars)
    # Power-scaling prior sensitivity on the reported coefficients (#381). The
    # share_retained ratio is deliberately excluded: psense summarises a location
    # shift, which is not a stable description of a ratio whose denominator has
    # posterior mass on both sides of small values.
    _diag.run_psense(ctx, var_names=psense_vars)

    section_header("Posterior predictive")
    _diag.sample_posterior_predictive(ctx, var_names=["y_post"])
    for index, symbol in enumerate(outcome_symbols):
        stem = (
            "posterior_predictive_check"
            if index == 0
            else f"posterior_predictive_check_{symbol.lower()}"
        )
        _diag.save_joint_posterior_predictive_plot(ctx, symbol, filename_stem=stem)
    # Coverage statistic (#318). Per-observation interval coverage is
    # denominator-agnostic — each flattened child x outcome cell is scored against
    # its own predictive draws — so it is well defined on the flattened ``y_post``
    # even though the distribution overlays must be split per outcome above.
    with guard_optional(ctx, "ppc_summary.csv", filename="ppc_summary.csv", kind="table"):
        coverage = _report.ppc_interval_coverage(ctx.trace, node="y_post")
        save_table(ctx, "ppc_summary", coverage, required=False)
    if marginal_ppc:
        _jm_marginal_ppc(ctx, outcome_symbols=outcome_symbols)

    section_header("Extended diagnostics")
    gate = _diag.write_diagnostics_summary(ctx, var_names=diag_vars)
    # The generic LOO-PIT would pool flattened cells from tests with different
    # denominators (79 items vs 6); save one calibrated plot per outcome instead.
    # ``causal_term`` only selects which parameter the rank and ESS-evolution plots
    # focus on — it carries no causal claim, and nothing in this family is causal.
    # Naming the headline contrast matters: left unset, the ESS-evolution plot tries
    # to draw every variable (563 subplots for the transition design) and is skipped
    # entirely, which is how a required diagnostic goes quietly missing.
    _diag.run_extended_diagnostics(
        ctx, causal_term="delta_ls_decoding", include_loo_pit=False
    )
    for index, symbol in enumerate(outcome_symbols):
        stem = "loo_pit" if index == 0 else f"loo_pit_{symbol.lower()}"
        _diag.save_joint_loo_pit_plot(
            ctx, symbol, filename_stem=stem, posterior_var="beta_mech"
        )
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=diag_vars)
    return gate


def _jm_write_slopes(
    ctx: StatisticalFitContext,
    rows: list[dict],
    *,
    contrast: tuple[str, str],
) -> pd.DataFrame:
    """Validate and write ``joint_mechanism_slopes.csv``."""
    df = pd.DataFrame(rows)
    missing = _JM_SLOPE_REQUIRED.difference(df.columns)
    if missing:
        raise ValueError(
            f"joint_mechanism_slopes is missing required columns: {sorted(missing)}"
        )
    if not (df["term"] == "delta_ls_decoding").any():
        raise ValueError("joint_mechanism_slopes has no delta_ls_decoding row")
    save_table(ctx, "joint_mechanism_slopes", df)
    print_table(
        metrics_table(
            df.to_dict("records"),
            title=(
                f"Letter-sound slopes + identified {contrast[0]}-{contrast[1]} contrast "
                f"(median, inner 50%, outer {int(ctx.reporting.ci_prob * 100)}%, "
                "equal-tailed)"
            ),
            columns=["wave", "term", "median", "lo50", "hi50", "lo", "hi", "prob_pos"],
        )
    )
    return df


def fit_joint_mechanism(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Fit the bivariate joint mechanism (#421 Tier 3 (1)) in one of two designs.

    ``design="levels"`` is the per-wave levels/concurrent ``{W, N}`` model the issue
    specifies: one fit per timepoint, each with an LKJ residual correlation between
    the two outcomes, reporting both the identified decoding-specificity contrast
    ``delta_ls_decoding`` **and** the identified ``share_retained`` — the within-model
    replacement for the paired-draws ratio the ``ca-010`` / ``ca-011`` pair can only
    approximate. ``design="transition"`` is the phase-stacked ANCOVA companion matched
    to ``mech-096`` / ``mech-101``, which re-reports the Tier-1 Delta on the same
    parameterisation it was originally computed on, with a bivariate child random
    intercept supplying the cross-outcome covariance.
    """
    require_spec(spec, "joint_mechanism")
    e = spec.extra
    design = str(e.get("design", "levels"))
    if design not in {"levels", "transition"}:
        raise ValueError(
            f"{spec.model_id}: joint-mechanism design must be 'levels' or "
            f"'transition'; got {design!r}"
        )
    if design == "levels":
        return _fit_joint_mechanism_levels(spec, config)
    return _fit_joint_mechanism_transition(spec, config)


def _jm_common_settings(spec: ModelSpec) -> dict:
    e = spec.extra
    return {
        "mechanism_symbol": spec.mechanism_symbol or "L",
        "outcome_symbols": tuple(e.get("outcome_symbols", ("W", "N"))),
        "contrast": tuple(e.get("contrast", ("N", "W"))),
        "confounder_symbols": tuple(e.get("confounder_symbols", ("G", "A"))),
        "include_group": bool(e.get("include_group", True)),
    }


def _fit_joint_mechanism_levels(
    spec: ModelSpec, config: str
) -> StatisticalFitContext:
    """Per-wave levels/concurrent bivariate fit (#421 Tier 3 (1); ``jm-001``).

    One cross-sectional fit per timepoint, mirroring the ``concurrent`` family's
    published shape: the diagnostic-anchor wave (most rows; ties -> latest) carries
    the trace / gate / PPC artefacts, every wave's convergence is recorded in
    ``joint_mechanism_fit_diagnostics.csv``, and no wave is silently dropped.
    """
    e = spec.extra
    s = _jm_common_settings(spec)
    outcome_symbols = s["outcome_symbols"]
    contrast = s["contrast"]
    # Trait covariates are t1-measured, so they broadcast from baseline across the
    # four timepoint rows — exactly as ca-010 / ca-011 enter them, which is what
    # makes the identified share_retained a like-for-like replacement.
    covariates = tuple(e.get("covariates", ()))
    predictor_slope_sigma = float(e.get("predictor_slope_sigma", 0.3))

    ctx = make_context(spec, config)
    ci = ctx.reporting.ci_prob

    section_header("Prepare data")
    prepared_all = load_and_prepare(
        phase_mode="levels",
        outcomes=(*outcome_symbols, s["mechanism_symbol"]),
        baseline_covariates=covariates,
    )
    covariates = tuple(c for c in covariates if c in prepared_all.covariates)

    def _build(sub):
        return _factories.build_joint_mechanism_model(
            sub,
            design="levels",
            mechanism_symbol=s["mechanism_symbol"],
            outcome_symbols=outcome_symbols,
            contrast=contrast,
            adjust_for=covariates,
            confounder_symbols=s["confounder_symbols"],
            include_group=s["include_group"],
            predictor_slope_sigma=predictor_slope_sigma,
        )

    # Usable rows at a wave: the exposure observed (it is never imputed — imputing
    # the focal exposure would bias both slopes toward zero) and at least one outcome
    # observed. A wave below the floor is skipped *and named*: a silently dropped
    # timepoint would read as "that wave was not estimable" when it was never tried.
    wave_indices = sorted({int(p) for p in np.unique(prepared_all.phase)})
    wave_subsets: dict[int, object] = {}
    skipped: list[str] = []
    for w in wave_indices:
        sub = _subset_prepared(prepared_all, prepared_all.phase == w)
        usable = ~np.isnan(sub.post_counts[s["mechanism_symbol"]])
        any_outcome = np.zeros(sub.n_obs, dtype=bool)
        for symbol in outcome_symbols:
            any_outcome |= ~np.isnan(sub.post_counts[symbol])
        n_usable = int(np.count_nonzero(usable & any_outcome))
        if n_usable < _JM_MIN_WAVE_ROWS:
            skipped.append(f"t{w + 1} ({n_usable} usable rows)")
            continue
        wave_subsets[w] = sub
    if skipped:
        rprint(
            "[yellow]Joint mechanism: skipped "
            f"{', '.join(skipped)} — fewer than {_JM_MIN_WAVE_ROWS} rows with the "
            "exposure and at least one outcome observed.[/yellow]"
        )
    if not wave_subsets:
        raise ValueError(
            f"{spec.model_id}: no timepoint has at least {_JM_MIN_WAVE_ROWS} rows "
            f"with the exposure {s['mechanism_symbol']!r} and an outcome observed."
        )
    # Build after the data filter, so a specification error (a bad contrast, a missing
    # covariate) raises rather than being swallowed as "this wave is not fittable".
    wave_built = {w: _build(sub) for w, sub in wave_subsets.items()}
    # Diagnostic anchor = most rows; ties -> latest. An operational artefact-selection
    # rule, not a claim that the wave is best-powered or substantively primary.
    primary_wave = max(wave_built, key=lambda w: (wave_built[w].prepared.n_obs, w))

    ctx.prepared = wave_built[primary_wave].prepared
    print_header(ctx)

    slope_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    gate = None
    for w in sorted(wave_built):
        built = wave_built[w]
        tp = w + 1
        if w == primary_wave:
            section_header(f"Build model (anchor wave t{tp})")
            attach_built(ctx, built)
            render_model_graph(ctx)
            section_header("Prior predictive")
            _diag.run_prior_predictive(ctx, draws=1000)
            for index, symbol in enumerate(outcome_symbols):
                stem = (
                    "prior_predictive_check"
                    if index == 0
                    else f"prior_predictive_check_{symbol.lower()}"
                )
                _diag.save_prior_predictive_plot(ctx, symbol, filename_stem=stem)
            run_sampling_and_loo(ctx)
            trace = ctx.trace
            model_names = {rv.name for rv in ctx.model.free_RVs} | set(
                ctx.trace.posterior.data_vars
            )
            diag_vars = _jm_diag_vars(
                model_names,
                design="levels",
                adjust_for=covariates,
                confounder_symbols=s["confounder_symbols"],
                include_group=s["include_group"],
            )
            psense_vars = [
                v
                for v in ("beta_mech", "delta_ls_decoding", "rho_outcome",
                          "beta_mech_focal_given_held")
                if v in ctx.trace.posterior
            ]
            gate = _jm_standard_artefacts(
                ctx,
                outcome_symbols=outcome_symbols,
                diag_vars=diag_vars,
                psense_vars=psense_vars,
                # One latent residual per child over two cells: the conditional
                # coverage is structurally 1.00, so publish the new-child version too.
                marginal_ppc=True,
            )
            convergence = _diag.subfit_convergence(
                ctx.trace,
                label=f"{spec.model_id} anchor wave t{tp}",
                var_names=[rv.name for rv in ctx.model.free_RVs],
            )
            convergence["converged"] = bool(
                _report.convergence_gate_clean_passed(gate)
                and convergence.get("converged")
            )
        else:
            trace, convergence = sample_subfit(
                built.model, ctx.sampling, label=f"{spec.model_id} wave t{tp}"
            )
        slope_rows += _jm_slope_rows(
            trace,
            outcome_symbols=outcome_symbols,
            contrast=contrast,
            ci_prob=ci,
            wave=f"t{tp}",
            converged=bool(convergence.get("converged")),
        )
        diagnostic_rows.append(
            {
                "wave": f"t{tp}",
                "timepoint": tp,
                "role": "anchor" if w == primary_wave else "sub-fit",
                "n": built.prepared.n_obs,
                **convergence,
            }
        )

    section_header("Decoding-specificity contrast and share retained")
    slopes_df = _jm_write_slopes(ctx, slope_rows, contrast=contrast)
    diagnostics_df = pd.DataFrame(diagnostic_rows)
    save_table(ctx, "joint_mechanism_fit_diagnostics", diagnostics_df)
    _plot_joint_mechanism_by_wave(ctx, slopes_df, ci)

    write_run_metadata(
        ctx,
        extra={
            "design": "levels",
            "loo_elpd": float(ctx.loo.elpd) if ctx.loo is not None else None,
            "estimand": (
                "per-wave bivariate letter-sound slopes on {W, N} with an LKJ "
                "within-wave residual correlation; identified decoding-specificity "
                "contrast and share-retained"
            ),
            "joint_dependence": "lkj_residual_within_wave",
            "likelihood": "binomial",
            "mechanism_symbol": s["mechanism_symbol"],
            "outcome_symbols": list(outcome_symbols),
            "contrast": list(contrast),
            "covariates": list(covariates),
            "include_group_nuisance": s["include_group"],
            "predictor_slope_sigma": predictor_slope_sigma,
            "diagnostic_anchor_timepoint": primary_wave + 1,
            "timepoints": [w + 1 for w in sorted(wave_built)],
            "n_published_fits": int(len(diagnostics_df)),
            "all_published_fits_converged": bool(
                not diagnostics_df.empty
                and diagnostics_df["converged"].eq(True).all()
            ),
            "matched_comparators": ["lrp-rli-ca-010", "lrp-rli-ca-011"],
            "output_contract": (
                "joint_mechanism_slopes.csv carries median + inner 50% + outer "
                "reporting interval + P(>0) per wave for both slopes, their "
                "difference, the residual correlation and the conditional slope / "
                "share retained; joint_mechanism_fit_diagnostics.csv records every "
                "published wave's convergence"
            ),
            "joint_mechanism_slopes": slopes_df.to_dict("records"),
        },
    )
    return finalize_report(ctx)


def _plot_joint_mechanism_by_wave(
    ctx: StatisticalFitContext, df: pd.DataFrame, ci_prob: float
) -> None:
    """Per-wave forest of the two letter-sound slopes and their identified difference.

    One figure per file (PNG + SVG + CSV via ``save_styled_figure``), not a panel, so
    it can be reused on its own.
    """
    keep = df[df["term"].str.startswith(("beta_mech[", "delta_ls_decoding"))]
    if keep.empty:
        return
    keep = keep.reset_index(drop=True)
    y = np.arange(len(keep))[::-1]
    plt.figure(figsize=(7.2, 0.42 * len(keep) + 1.6))
    colours = [
        COLOUR_BLUE if str(t).startswith("beta_mech[") else "#B45309"
        for t in keep["term"]
    ]
    for i, (_, row) in enumerate(keep.iterrows()):
        plt.errorbar(
            row["median"],
            y[i],
            xerr=[[row["median"] - row["lo"]], [row["hi"] - row["median"]]],
            fmt="o",
            color=colours[i],
            capsize=3,
        )
        plt.plot(
            [row["lo50"], row["hi50"]], [y[i], y[i]], color=colours[i], lw=3.0, alpha=0.7
        )
    plt.axvline(0.0, color="grey", ls=":", lw=1)
    plt.yticks(y, [f"{r['wave']}  {r['term']}" for _, r in keep.iterrows()], fontsize=8)
    plt.xlabel(
        f"logit per SD of letter sounds (median, inner 50%, outer "
        f"{int(ci_prob * 100)}%)"
    )
    plt.title("Letter-sound slopes and their identified difference, by wave")
    save_styled_figure(ctx.output_dir, "joint_mechanism_by_wave", data=keep)


def _fit_joint_mechanism_transition(
    spec: ModelSpec, config: str
) -> StatisticalFitContext:
    """Phase-stacked ANCOVA companion (#421 Tier 3 (1); ``jm-002``).

    Matched term-for-term to ``mech-096`` / ``mech-101`` so the Tier-1 Delta is
    re-reported on its original parameterisation, with a bivariate child random
    intercept (LKJ) carrying the cross-outcome covariance the separate fits cannot.
    """
    e = spec.extra
    s = _jm_common_settings(spec)
    outcome_symbols = s["outcome_symbols"]
    contrast = s["contrast"]
    adjust_for = tuple(e.get("adjust_for", ()))

    ctx = make_context(spec, config)
    ci = ctx.reporting.ci_prob

    section_header("Prepare data")
    pre_adj, post_adj = split_covariates_by_wave(adjust_for)
    prepared = load_and_prepare(
        phase_mode="all",
        outcomes=(*outcome_symbols, s["mechanism_symbol"]),
        covariates=pre_adj,
        post_covariates=post_adj,
    )
    ctx.prepared = prepared
    # Drop any adjuster the loader dropped as constant on the fitted rows.
    adjust_for = tuple(c for c in adjust_for if c in prepared.covariates)

    print_header(ctx)

    section_header("Build model")
    built = _factories.build_joint_mechanism_model(
        prepared,
        design="transition",
        mechanism_symbol=s["mechanism_symbol"],
        outcome_symbols=outcome_symbols,
        contrast=contrast,
        adjust_for=adjust_for,
        confounder_symbols=s["confounder_symbols"],
        include_group=s["include_group"],
    )
    attach_built(ctx, built)
    render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    for index, symbol in enumerate(outcome_symbols):
        stem = (
            "prior_predictive_check"
            if index == 0
            else f"prior_predictive_check_{symbol.lower()}"
        )
        _diag.save_prior_predictive_plot(ctx, symbol, filename_stem=stem)

    run_sampling_and_loo(ctx)

    model_names = {rv.name for rv in ctx.model.free_RVs} | set(
        ctx.trace.posterior.data_vars
    )
    diag_vars = _jm_diag_vars(
        model_names,
        design="transition",
        adjust_for=adjust_for,
        confounder_symbols=s["confounder_symbols"],
        include_group=s["include_group"],
    )
    psense_vars = [
        v
        for v in ("beta_mech", "delta_ls_decoding", "rho_outcome")
        if v in ctx.trace.posterior
    ]
    gate = _jm_standard_artefacts(
        ctx,
        outcome_symbols=outcome_symbols,
        diag_vars=diag_vars,
        psense_vars=psense_vars,
    )

    section_header("Decoding-specificity contrast")
    slopes_df = _jm_write_slopes(
        ctx,
        _jm_slope_rows(
            ctx.trace,
            outcome_symbols=outcome_symbols,
            contrast=contrast,
            ci_prob=ci,
            wave="stacked",
            converged=_report.convergence_gate_clean_passed(gate),
        ),
        contrast=contrast,
    )

    write_run_metadata(
        ctx,
        extra={
            "design": "transition",
            "loo_elpd": float(ctx.loo.elpd) if ctx.loo is not None else None,
            "estimand": (
                "phase-stacked bivariate letter-sound slopes on {W, N} with a "
                "bivariate (LKJ) child random intercept; identified "
                "decoding-specificity contrast"
            ),
            "joint_dependence": "lkj_child_intercept",
            "likelihood": "beta_binomial",
            "mechanism_symbol": s["mechanism_symbol"],
            "outcome_symbols": list(outcome_symbols),
            "contrast": list(contrast),
            "adjust_for": list(adjust_for),
            "matched_comparators": ["lrp-rli-mech-096", "lrp-rli-mech-101"],
            "output_contract": (
                "joint_mechanism_slopes.csv carries median + inner 50% + outer "
                "reporting interval + P(>0) for both slopes, their identified "
                "difference and the child-level residual correlation"
            ),
            "joint_mechanism_slopes": slopes_df.to_dict("records"),
        },
    )
    return finalize_report(ctx)
