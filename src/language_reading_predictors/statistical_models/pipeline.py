# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
End-to-end fit pipeline for the statistical models.

``fit_itt(spec, config)`` is the entry point for the ITT models.
``fit_joint(spec, config)`` is the entry point for the joint models (LRPITT12, LRPITT15/15b).
``fit_did(spec, config)`` is the entry point for the waitlist-crossover / DiD models.
``fit_mechanism(spec, config)`` is the entry point for LRP56/57/58.

Each pipeline:

1. Loads data via :func:`preprocessing.load_and_prepare`.
2. Builds the PyMC model via the appropriate factory.
3. Writes prior-panel plots.
4. Runs prior predictive, posterior sampling (nutpie), LOO, posterior
   predictive.
5. Saves ``trace.nc`` (with the prior / prior_predictive / log_prior groups
   attached — issue #125 step 0b), ``config.json``, ``diagnostics_summary.json``
   (the pass/fail convergence gate), ``priors_table.csv`` and the standard
   diagnostic plots to ``output/statistical_models/models/{model_id}-{config}/``.
6. Copies ``docs/models/{model_id}/index.qmd`` and the shared
   ``docs/models/_partials/`` alongside the artefacts so the Quarto report can be
   rendered in-place.
"""

from __future__ import annotations

import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich import print as rprint

from language_reading_predictors.models._reporting import (
    metrics_table,
    print_panel,
    print_table,
    ranked_dataframe_table,
    run_summary_panel,
    section_header,
    stat_model_header_panel,
)
from language_reading_predictors.statistical_models import (
    diagnostics as _diag,
    factories as _factories,
    priors as _priors,
    reporting as _report,
)
from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
    make_context,
)
from language_reading_predictors.statistical_models.environment import DOCS_DIR
from language_reading_predictors.statistical_models.preprocessing import (
    load_and_prepare,
    load_and_prepare_aligned,
    load_and_prepare_lagged_outcome,
)


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


def _print_header(ctx: StatisticalFitContext) -> None:
    """Print the start-of-fit banner panel."""
    spec = ctx.spec
    prepared = ctx.prepared
    rprint()
    print_panel(
        stat_model_header_panel(
            model_id=spec.model_id,
            title=spec.title,
            kind=spec.kind,
            config_name=ctx.reporting.config_name,
            outcome_symbol=spec.outcome_symbol,
            mechanism_symbol=spec.mechanism_symbol,
            adjustment=spec.adjustment or None,
            n_obs=prepared.n_obs if prepared else None,
            n_children=prepared.n_children if prepared else None,
            n_phases=prepared.n_phases if prepared else None,
        )
    )


def _print_footer(ctx: StatisticalFitContext) -> None:
    """Print the end-of-fit banner panel."""
    rprint()
    print_panel(run_summary_panel(output_dir=ctx.output_dir))


def _print_loo_row(ctx: StatisticalFitContext) -> None:
    """Render the LOO ELPD / p / se summary as a small table.

    arviz 1.x ``ELPDData`` exposes ``elpd`` / ``se`` / ``p`` (the 0.x
    ``elpd_loo`` / ``p_loo`` / ``looic`` attributes were removed).
    """
    if ctx.loo is None:
        return
    rows = [
        {"metric": "elpd_loo", "value": float(ctx.loo.elpd)},
        {"metric": "se", "value": float(ctx.loo.se)},
        {"metric": "p_loo", "value": float(ctx.loo.p)},
    ]
    print_table(
        metrics_table(
            rows,
            title="LOO-PSIS",
            columns=["metric", "value"],
        )
    )


def _copy_report_template(context: StatisticalFitContext) -> None:
    src = os.path.join(DOCS_DIR, "models", context.spec.model_id, "index.qmd")
    dst = os.path.join(context.output_dir, "index.qmd")
    if os.path.exists(src):
        shutil.copy(src, dst)
        rprint(f"  Report template copied to {dst}")
    else:
        rprint(f"  [yellow]No report template found at {src}[/yellow]")

    # Copy the shared Quarto partials alongside the report so ``{{< include
    # _partials/... >}}`` resolves at render time in the output dir (issue #125
    # step 0a). Quarto resolves includes relative to the rendered file.
    partials_src = os.path.join(DOCS_DIR, "models", "_partials")
    partials_dst = os.path.join(context.output_dir, "_partials")
    if os.path.isdir(partials_src):
        shutil.copytree(partials_src, partials_dst, dirs_exist_ok=True)


def _emit_priors(context: StatisticalFitContext) -> None:
    """Write the pruned prior panel + ``priors_table.csv`` (issue #125 Area 1).

    Only the priors the model actually registered are panelled (no more 4–6 dead
    panels per model), and ``priors_table.csv`` documents every parameter's
    distribution, role (causal / precision / association / nuisance / GP) and
    rationale, driven by the built model so it cannot drift from the source.
    """
    model = context.model
    # Clear stale prior-PDF panels from a previous run so only the used set
    # remains (one file per named prior; not the prior-predictive / overlay PNGs).
    for key in _priors.ALL_PRIORS:
        for ext in ("png", "svg"):
            stale = os.path.join(context.output_dir, f"prior_{key}.{ext}")
            try:
                os.remove(stale)
            except OSError:
                pass
    _priors.save_shared_prior_panel(
        context.output_dir, used=_priors.used_prior_keys(model)
    )
    table = _priors.priors_table(model)
    table.to_csv(os.path.join(context.output_dir, "priors_table.csv"), index=False)
    context.tables["priors_table"] = table


def _render_model_graph(context: StatisticalFitContext) -> None:
    try:
        g = _graphviz(context.model)
        g.render(
            filename=os.path.join(context.output_dir, "model_graph"),
            format="svg",
            cleanup=True,
        )
    except Exception as exc:  # pragma: no cover
        rprint(f"[yellow]Graphviz render failed: {exc}[/yellow]")


def _graphviz(model):
    import pymc as pm

    g = pm.model_to_graphviz(model)
    g.graph_attr["fontname"] = "Helvetica"
    g.node_attr["fontname"] = "Helvetica"
    g.edge_attr["fontname"] = "Helvetica"
    return g


def _save_ppc(context: StatisticalFitContext) -> None:
    # arviz 1.x removed az.plot_ppc; the equivalent is arviz_plots.plot_ppc_dist
    # (returns a PlotCollection with .savefig). Guarded — a PPC plot failure must
    # not abort the fit.
    try:
        import arviz_plots as azp

        pc = azp.plot_ppc_dist(context.trace)
        pc.savefig(
            os.path.join(context.output_dir, "posterior_predictive_check.png"),
            dpi=300,
        )
        plt.close("all")
    except Exception as exc:  # pragma: no cover
        rprint(f"[yellow]PPC plot failed: {exc}[/yellow]")


def _save_proportion_at_zero_plot(
    ctx: StatisticalFitContext, symbol: str, ppc0: dict
) -> None:
    """Plot the proportion-at-zero PPC: replicated distribution vs observed."""
    try:
        rep = ppc0["rep"]
        obs = ppc0["obs_prop_at_zero"]
        plt.figure(figsize=(6, 4))
        plt.hist(rep, bins=30, color="#1f77b4", alpha=0.6, density=True)
        plt.axvline(obs, color="#d62728", lw=2, label=f"observed = {obs:.2f}")
        plt.xlabel(f"proportion of {symbol} post-scores at zero")
        plt.ylabel("posterior-predictive density")
        plt.title(
            f"Proportion-at-zero PPC ({symbol}); p = {ppc0['ppc_p_value']:.2f}"
        )
        plt.legend()
        plt.savefig(
            os.path.join(ctx.output_dir, "proportion_at_zero_ppc.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    except Exception as exc:  # pragma: no cover
        rprint(f"[yellow]Proportion-at-zero PPC plot failed: {exc}[/yellow]")


def _save_rope_plot(
    ctx: StatisticalFitContext,
    symbol: str,
    G: np.ndarray | None,
    n_trials: int,
    delta: float,
    *,
    term: str = "tau",
    varying_term: str = "tau_i",
    items: np.ndarray | None = None,
) -> None:
    """ROPE-anchored figure for a randomised effect: the items-scale posterior with
    the region of practical equivalence, and ``P(effect > delta)`` as the
    minimally-important difference rises. Single-outcome version of the note figure
    (notes/202606261304-evidence-strength-and-rope-reporting.md).

    The ITT/gain path recomputes the items draws from ``_itt_ame_draws`` (``term`` /
    ``varying_term`` / ``G`` select the effect); the level family passes its t2
    contrast items draws directly via ``items`` (its AME nets out a group×ability
    interaction the generic core cannot reconstruct).
    """
    try:
        from scipy.stats import gaussian_kde

        if items is None:
            _, ame_prob = _report._itt_ame_draws(
                ctx.trace, G=G, term=term, varying_term=varying_term
            )
            items = ame_prob * float(n_trials)
        med = float(np.median(items))
        xmax = float(np.quantile(items, 0.995)) + 0.5
        xmin = min(-delta - 0.5, float(np.quantile(items, 0.005)))
        xs = np.linspace(xmin, xmax, 300)
        kde = gaussian_kde(items)

        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.2))
        ax_l.axvspan(
            -delta, delta, color="#bdbdbd", alpha=0.30,
            label=f"ROPE (|effect| < {delta:g})",
        )
        ax_l.axvline(0, color="#444444", lw=1.0, ls=":")
        ax_l.plot(xs, kde(xs), color="#1b7837", lw=2.2)
        ax_l.fill_between(xs, kde(xs), color="#1b7837", alpha=0.12)
        ax_l.axvline(med, color="#1b7837", lw=1.2, label=f"median {med:+.1f}")
        ax_l.set_xlabel("treatment effect (extra items correct)")
        ax_l.set_ylabel("posterior density")
        ax_l.set_title(f"{symbol}: effect on the items scale, with ROPE")
        ax_l.legend(fontsize=8, frameon=False)

        dgrid = np.linspace(0.0, max(xmax, delta + 0.5), 200)
        pex = np.array([float((items > d).mean()) for d in dgrid])
        ax_r.plot(dgrid, pex, color="#2166ac", lw=2.2)
        ax_r.axvline(delta, color="#888888", lw=1.0, ls="--", label=f"delta = {delta:g}")
        ax_r.axhline(0.975, color="#cccccc", lw=1.0, ls=":")
        ax_r.set_ylim(0, 1.02)
        ax_r.set_xlabel("minimally-important difference, delta (items)")
        ax_r.set_ylabel("P(effect > delta)")
        ax_r.set_title("Probability of a meaningful benefit")
        ax_r.legend(fontsize=8, frameon=False)

        for ax in (ax_l, ax_r):
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        fig.tight_layout()
        fig.savefig(
            os.path.join(ctx.output_dir, "rope_summary.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        rprint(f"[yellow]ROPE plot failed: {exc}[/yellow]")


def _save_contrast_heatmap(ctx: StatisticalFitContext, contrast) -> None:
    """Heatmap of the joint pairwise contrast matrix P(tau_k > tau_j) (#125 Area 4)."""
    try:
        import numpy as _np

        labels = list(contrast.index)
        M = contrast.to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(1.1 + 0.6 * len(labels), 1.0 + 0.6 * len(labels)))
        im = ax.imshow(M, cmap="RdBu_r", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        for i in range(len(labels)):
            for j in range(len(labels)):
                if _np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title("P(row tau > column tau)", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(
            os.path.join(ctx.output_dir, "contrast_heatmap.png"), dpi=300, bbox_inches="tight"
        )
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        rprint(f"[yellow]Contrast heatmap failed: {exc}[/yellow]")


def _emit_itt_extras(
    ctx: StatisticalFitContext,
    built,
    *,
    n_trials: int,
    overlay_vars: list[str],
    term: str = "tau",
    varying_term: str = "tau_i",
) -> None:
    """Area 1/4 extras for an ITT-style fit (issue #125).

    Writes ``prior_pushforward.csv`` (the estimand-scale prior check), the causal
    forest, the prior-vs-posterior overlay, and power-scaling sensitivity. Reads
    the persisted ``prior`` group (on ``ctx.prior_samples``) and the full trace,
    so call after ``save_trace``. ``n_trials=1`` gives the risk-difference scale
    for the binary off-floor model.
    """
    try:
        pf = _report.prior_pushforward(
            ctx.prior_samples,
            G=built.prepared.G,
            n_trials=n_trials,
            term=term,
            varying_term=varying_term,
            ci_prob=ctx.reporting.hdi,
        )
        pd.DataFrame([pf]).to_csv(
            os.path.join(ctx.output_dir, "prior_pushforward.csv"), index=False
        )
        ctx.tables["prior_pushforward"] = pd.DataFrame([pf])
    except Exception as exc:  # pragma: no cover
        rprint(f"[yellow]prior pushforward skipped: {exc}[/yellow]")
    _save_forest_plot(ctx, [term])
    _diag.save_prior_posterior_plot(ctx, var_names=overlay_vars)
    _diag.run_psense(ctx, var_names=[term])


def _save_forest_plot(
    ctx: StatisticalFitContext,
    var_names: list[str],
    *,
    name: str = "tau_forest.png",
) -> None:
    """Forest plot of the causal term(s) with a reference line at 0 (#125 Area 4).

    For a single-outcome model ``var_names=["tau"]`` shows the one effect; for the
    joint model the vector ``tau`` forests every outcome's effect in one panel —
    the single most communicative artifact for the suite. Guarded.
    """
    try:
        import arviz_plots as azp

        tr = _diag.thin_for_plots(ctx.trace)
        pc = azp.plot_forest(tr, var_names=var_names, combined=True)
        try:
            azp.add_lines(pc, values=0)
        except Exception:
            pass  # the forest itself is the substantive output
        pc.savefig(os.path.join(ctx.output_dir, name), dpi=300)
        plt.close("all")
    except Exception as exc:  # pragma: no cover
        rprint(f"[yellow]Forest plot ({name}) failed: {exc}[/yellow]")


def _save_association_forest(
    ctx: StatisticalFitContext,
    coef_names: list[str],
    causal_terms: tuple[str, ...],
) -> None:
    """Forest of a factor model's adjusted-association coefficients (#125 Area 4).

    Companion to the single causal-term forest: shows every *non-randomised*
    predictor's posterior coefficient (the adjusted associations) so the cross-skill
    predictor->outcome relationships are visible, not only tabulated. Excludes any RV
    that carries a causal element — e.g. the level model's ``b_grp_time`` vector, whose
    t2 entry is the one randomised contrast — so the causal/association split stays
    clean. Guarded via :func:`_save_forest_plot`.
    """
    assoc = [
        c
        for c in coef_names
        if c in ctx.trace.posterior
        and not any(ct == c or ct.startswith(c + "[") for ct in causal_terms)
    ]
    if assoc:
        _save_forest_plot(ctx, assoc, name="association_forest.png")


# ---------------------------------------------------------------------------
# ITT pipeline (LRPITT suite + SES companions)
# ---------------------------------------------------------------------------


def _itt_diag_vars(
    spec: ModelSpec,
    adjust_for: tuple[str, ...],
    *,
    likelihood: str = "beta_binomial",
) -> list[str]:
    """Scalar coefficients to summarise for an ITT fit, conditional on the spec.

    The own-baseline (``gamma_own``) is only present when ``use_own_baseline``;
    the linear age term (``gamma_A``) only when ``use_age_linear``; the
    Beta-Binomial concentration (``kappa``) only for the graded likelihood (the
    binary off-floor model has none); plus any adjuster / tau-moderator
    coefficients. Listing a missing RV would crash ``summary_diagnostics``.
    """
    extra = spec.extra
    dvars = ["alpha", "tau"]
    if extra.get("use_own_baseline", True):
        dvars.append("gamma_own")
    if extra.get("use_age_linear", False):
        dvars.append("gamma_A")
    if likelihood == "beta_binomial":
        dvars.append("kappa")
    dvars.extend(f"gamma_{c}" for c in adjust_for)
    if extra.get("tau_moderator_symbol") is not None:
        dvars.append("gamma_tau_mod")
        if extra.get("tau_moderator_interaction", True):
            dvars.append("gamma_tau_int")
    return dvars


def fit_itt(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    assert spec.kind == "itt"
    assert spec.outcome_symbol is not None

    ctx = make_context(spec, config)

    section_header("Prepare data")
    adjust_for = tuple(spec.extra.get("adjust_for", ()))
    # Optionally restrict to the complete-case subset of some columns *without*
    # adjusting for them (matched comparator, e.g. LRPITT14 vs LRPITT13).
    restrict_complete = tuple(spec.extra.get("restrict_complete", ()))
    # An ITT model whose outcome is outside ``ITT_OUTCOMES`` (e.g. the taught-
    # vocabulary block measures, the taught LRPITT models) overrides the prepared outcome set
    # so only the outcome and its chosen cross-baselines are loaded - this keeps
    # the complete-case mask from dropping rows for the eight standardised
    # outcomes the model never uses. ``cross_symbols`` selects the cross-baseline
    # set (default = every other ITT outcome).
    extra_outcomes = spec.extra.get("outcomes")
    cross_symbols = spec.extra.get("cross_symbols")
    # Post-only / age-only outcomes (e.g. nonword N) exempt their baseline from
    # the complete-case mask via ``pre_required`` so missing pre-scores on a
    # baseline the model never uses do not silently drop rows (#119).
    pre_required = spec.extra.get("pre_required")
    if pre_required is not None:
        pre_required = tuple(pre_required)
    drop_missing_pre = bool(spec.extra.get("drop_missing_pre", True))
    if extra_outcomes is not None:
        extra_outcomes = tuple(extra_outcomes)
        if spec.outcome_symbol not in extra_outcomes:
            raise ValueError(
                f"outcome_symbol {spec.outcome_symbol!r} must be included in "
                f"outcomes={extra_outcomes!r}"
            )
        prepared = load_and_prepare(
            phase_mode="itt",
            outcomes=extra_outcomes,
            covariates=adjust_for,
            restrict_complete=restrict_complete,
            drop_missing_pre=drop_missing_pre,
            pre_required=pre_required,
        )
    else:
        prepared = load_and_prepare(
            phase_mode="itt",
            covariates=adjust_for,
            restrict_complete=restrict_complete,
            drop_missing_pre=drop_missing_pre,
            pre_required=pre_required,
        )
    ctx.prepared = prepared

    _print_header(ctx)

    section_header("Build model")

    # Heavily-floored outcomes (P, N) take the pre-specified floor-rule branch:
    # a binary off-floor estimand as PRIMARY plus a graded SECONDARY (#119).
    if spec.extra.get("floor_rule", False):
        return _fit_itt_floor_rule(ctx, spec, prepared, adjust_for)

    built = _factories.build_itt_model(
        prepared,
        outcome_symbol=spec.outcome_symbol,
        use_age_gp=spec.extra.get("use_age_gp", False),
        use_own_baseline_gp=spec.extra.get("use_own_baseline_gp", False),
        use_varying_tau=spec.extra.get("use_varying_tau", False),
        adjust_for=adjust_for,
        cross_symbols=cross_symbols,
        use_age_linear=spec.extra.get("use_age_linear", False),
        use_own_baseline=spec.extra.get("use_own_baseline", True),
        tau_moderator_symbol=spec.extra.get("tau_moderator_symbol"),
        tau_moderator_is_covariate=spec.extra.get("tau_moderator_is_covariate", False),
        tau_moderator_interaction=spec.extra.get("tau_moderator_interaction", True),
    )
    ctx.model = built.model
    ctx.model_vars = built.variables
    ctx.prepared = built.prepared

    _emit_priors(ctx)
    _render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)

    section_header("Sampling posterior (nutpie)")
    _diag.sample_posterior(ctx)

    section_header("LOO-PSIS")
    _diag.compute_log_likelihood_and_loo(ctx)
    _report.write_loo_summary(ctx)
    _print_loo_row(ctx)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=_itt_diag_vars(spec, adjust_for))

    section_header("Posterior predictive")
    _diag.sample_posterior_predictive(ctx, var_names=["y_post"])
    _save_ppc(ctx)

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=_itt_diag_vars(spec, adjust_for))
    _diag.save_prior_predictive_plot(ctx, spec.outcome_symbol)
    _diag.run_extended_diagnostics(ctx, causal_term="tau")

    _diag.save_trace(ctx)

    # Area 1/4 extras that read the attached prior group or the full trace:
    # the prior pushforward to the items scale (estimand-scale prior check), the
    # tau forest, the prior-vs-posterior overlay, and power-scaling sensitivity.
    n_trials_own = int(built.prepared.n_trials[spec.outcome_symbol])
    _emit_itt_extras(
        ctx, built, n_trials=n_trials_own,
        overlay_vars=_itt_diag_vars(spec, adjust_for),
    )

    # Treatment-effect summary on both scales.
    section_header("Treatment-effect summary")
    tau_s = _report.tau_summary_itt(
        ctx.trace,
        ci_prob=ctx.reporting.hdi,
        # built.prepared is the (possibly row-subset) frame the model was fit
        # on, so G aligns with eta's obs_id axis (finding #2 in issue #78).
        G=built.prepared.G,
    )
    tau_df = pd.DataFrame([tau_s])
    tau_df.to_csv(os.path.join(ctx.output_dir, "tau_summary.csv"), index=False)
    ctx.tables["tau_summary"] = tau_df
    print_table(
        metrics_table(
            [{"metric": k, "value": v} for k, v in tau_s.items()],
            title=f"tau ({spec.outcome_symbol}) - {int(ctx.reporting.hdi * 100)}% CI (equal-tailed)",
            columns=["metric", "value"],
        )
    )

    # ROPE-anchored continuous summary on the items scale
    # (notes/202606261304-evidence-strength-and-rope-reporting.md). Emitted for
    # graded outcomes with an agreed minimally-important difference (delta);
    # floored outcomes (P/N) take the floor-rule path and a probability-scale
    # delta, which is not yet wired.
    from language_reading_predictors.statistical_models.measures import ROPE_DELTA

    delta_items = ROPE_DELTA.get(spec.outcome_symbol)
    if delta_items is not None:
        rope_s = _report.rope_summary(
            ctx.trace,
            G=built.prepared.G,
            n_trials=int(built.prepared.n_trials[spec.outcome_symbol]),
            delta=delta_items,
            ci_prob=ctx.reporting.hdi,
        )
        rope_df = pd.DataFrame([rope_s])
        rope_df.to_csv(os.path.join(ctx.output_dir, "rope_summary.csv"), index=False)
        ctx.tables["rope_summary"] = rope_df
        print_table(
            metrics_table(
                [{"metric": k, "value": v} for k, v in rope_s.items()],
                title=f"ROPE summary ({spec.outcome_symbol}, delta={delta_items:g} items)",
                columns=["metric", "value"],
            )
        )
        _save_rope_plot(
            ctx,
            spec.outcome_symbol,
            built.prepared.G,
            int(built.prepared.n_trials[spec.outcome_symbol]),
            delta_items,
        )

    _report.write_run_metadata(
        ctx,
        extra={
            "loo_elpd": float(ctx.loo.elpd),
            "tau_summary": tau_s,
            "adjust_for": list(adjust_for),
        },
    )

    section_header("Report")
    _copy_report_template(ctx)
    _print_footer(ctx)
    return ctx


def _fit_itt_floor_rule(
    ctx: StatisticalFitContext,
    spec: ModelSpec,
    prepared,
    adjust_for: tuple[str, ...],
) -> StatisticalFitContext:
    """Floor-rule fit for heavily-floored outcomes P / N (#119).

    Fits two age-only models on the same prepared data: the PRIMARY binary
    off-floor estimand (Bernoulli on ``post > 0``) and a flagged,
    detection-limited SECONDARY graded Beta-Binomial. Writes ``tau_summary.csv``
    (off-floor, primary), the per-arm mover table, the proportion-at-zero PPC,
    and ``tau_summary_graded.csv``. The floor rule is pre-specified and arm-blind.
    """
    import pymc as pm

    from language_reading_predictors.statistical_models import floor as _floor

    own = spec.outcome_symbol

    # Pre-specification gate: the floor rule is fixed before fitting and applied
    # arm-blind, so the outcome must actually qualify (>= 40% at zero at t2).
    p0 = _floor.proportion_at_zero(prepared, own)
    if not _floor.is_floored(prepared, own):
        raise ValueError(
            f"floor_rule set for {own!r}, but only {p0:.0%} of its post-scores "
            f"are at zero at t2 (threshold {_floor.FLOOR_THRESHOLD:.0%}); the "
            "floor rule is pre-specified and arm-blind - remove floor_rule or "
            "check the data."
        )
    rprint(
        f"  Floor rule: {own} is {p0:.0%} floored at t2 "
        f"(>= {_floor.FLOOR_THRESHOLD:.0%}); binary off-floor estimand is PRIMARY, "
        "graded Beta-Binomial is SECONDARY (detection-limited)."
    )

    common = dict(
        outcome_symbol=own,
        use_age_gp=spec.extra.get("use_age_gp", False),
        use_own_baseline_gp=spec.extra.get("use_own_baseline_gp", False),
        use_age_linear=spec.extra.get("use_age_linear", True),
        use_own_baseline=spec.extra.get("use_own_baseline", False),
        cross_symbols=spec.extra.get("cross_symbols", ()),
        adjust_for=adjust_for,
    )

    # ----- PRIMARY: binary off-floor (Bernoulli on post > 0) -----
    section_header("Build model (PRIMARY: binary off-floor)")
    built = _factories.build_itt_model(
        prepared, likelihood="bernoulli_offfloor", **common
    )
    ctx.model = built.model
    ctx.model_vars = built.variables
    ctx.prepared = built.prepared
    _emit_priors(ctx)
    _render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    section_header("Sampling posterior (nutpie)")
    _diag.sample_posterior(ctx)
    section_header("LOO-PSIS")
    _diag.compute_log_likelihood_and_loo(ctx)
    _report.write_loo_summary(ctx)
    _print_loo_row(ctx)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(
        ctx,
        var_names=_itt_diag_vars(spec, adjust_for, likelihood="bernoulli_offfloor"),
    )

    section_header("Posterior predictive")
    _diag.sample_posterior_predictive(ctx, var_names=["y_offfloor"])
    _save_ppc(ctx)

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(
        ctx,
        var_names=_itt_diag_vars(spec, adjust_for, likelihood="bernoulli_offfloor"),
    )
    _diag.save_prior_predictive_plot(ctx, own)
    _diag.run_extended_diagnostics(ctx, causal_term="tau")
    _diag.save_trace(ctx)

    # Off-floor estimand is a risk difference (Pr off-floor), so the items scale is
    # n_trials = 1; no age-varying term in the floor-rule model.
    _emit_itt_extras(
        ctx, built, n_trials=1, varying_term="",
        overlay_vars=_itt_diag_vars(spec, adjust_for, likelihood="bernoulli_offfloor"),
    )

    section_header("Off-floor treatment-effect summary (PRIMARY)")
    off = _report.tau_summary_offfloor(
        ctx.trace, ci_prob=ctx.reporting.hdi, G=built.prepared.G
    )
    pd.DataFrame([off]).to_csv(
        os.path.join(ctx.output_dir, "tau_summary.csv"), index=False
    )
    ctx.tables["tau_summary"] = pd.DataFrame([off])
    print_table(
        metrics_table(
            [{"metric": k, "value": v} for k, v in off.items()],
            title=(
                f"off-floor tau ({own}) - {int(ctx.reporting.hdi * 100)}% CI "
                "(equal-tailed); positive = intervention raises Pr(off-floor)"
            ),
            columns=["metric", "value"],
        )
    )

    movers = _report.offfloor_mover_table(built.prepared, own)
    movers.to_csv(os.path.join(ctx.output_dir, "offfloor_movers.csv"), index=False)
    ctx.tables["offfloor_movers"] = movers
    print_table(
        ranked_dataframe_table(
            movers,
            title=f"Off-floor movers by arm ({own})",
            columns=["arm", "n", "off_floor", "at_floor", "prop_off_floor"],
            rank_column=False,
            precision=3,
        )
    )

    # ROPE-anchored card on the off-floor RISK-DIFFERENCE scale (issue #125 Area 4;
    # #130 follow-up). delta is a probability (risk difference), n_trials = 1; the
    # value is the provisional ROPE_DELTA_PROB pending education-lead sign-off.
    from language_reading_predictors.statistical_models.measures import ROPE_DELTA_PROB

    delta_prob = ROPE_DELTA_PROB.get(own)
    if delta_prob is not None:
        rope_s = _report.rope_summary(
            ctx.trace,
            G=built.prepared.G,
            n_trials=1,
            delta=delta_prob,
            ci_prob=ctx.reporting.hdi,
            varying_term="",
        )
        rope_s["provisional_delta"] = True
        rope_s["delta_scale"] = "risk_difference"
        pd.DataFrame([rope_s]).to_csv(
            os.path.join(ctx.output_dir, "rope_summary.csv"), index=False
        )
        ctx.tables["rope_summary"] = pd.DataFrame([rope_s])
        _save_rope_plot(
            ctx, own, built.prepared.G, 1, delta_prob, varying_term=""
        )

    # ----- SECONDARY: graded Beta-Binomial (detection-limited) -----
    section_header("Build model (SECONDARY: graded Beta-Binomial, detection-limited)")
    built_g = _factories.build_itt_model(prepared, likelihood="beta_binomial", **common)
    s = ctx.sampling
    with built_g.model:
        trace_g = pm.sample(
            draws=s.draws,
            tune=s.tune,
            chains=s.chains,
            cores=s.cores,
            target_accept=s.target_accept,
            nuts_sampler="nutpie",
            return_inferencedata=True,
            random_seed=s.random_seed,
            progressbar=False,
        )
        trace_g = pm.sample_posterior_predictive(
            trace_g,
            var_names=["y_post"],
            extend_inferencedata=True,
            random_seed=s.random_seed,
            progressbar=False,
        )

    graded = _report.tau_summary_itt(
        trace_g, ci_prob=ctx.reporting.hdi, G=built_g.prepared.G
    )
    pd.DataFrame([graded]).to_csv(
        os.path.join(ctx.output_dir, "tau_summary_graded.csv"), index=False
    )
    ctx.tables["tau_summary_graded"] = pd.DataFrame([graded])

    # Proportion-at-zero PPC on the graded model: does the graded Beta-Binomial
    # reproduce the observed floor? (Usually not - the motivation for the binary
    # primary estimand.)
    ppc0 = _report.proportion_at_zero_ppc(built_g.prepared, own, trace_g)
    _save_proportion_at_zero_plot(ctx, own, ppc0)
    pd.DataFrame([{k: v for k, v in ppc0.items() if k != "rep"}]).to_csv(
        os.path.join(ctx.output_dir, "proportion_at_zero_ppc.csv"), index=False
    )

    _report.write_run_metadata(
        ctx,
        extra={
            "loo_elpd": float(ctx.loo.elpd),
            "floor_rule": {
                "outcome": own,
                "proportion_at_zero": p0,
                "threshold": _floor.FLOOR_THRESHOLD,
            },
            "tau_offfloor_primary": off,
            "tau_graded_secondary": graded,
            "proportion_at_zero_ppc": {k: v for k, v in ppc0.items() if k != "rep"},
            "adjust_for": list(adjust_for),
        },
    )

    section_header("Report")
    _copy_report_template(ctx)
    _print_footer(ctx)
    return ctx


# ---------------------------------------------------------------------------
# Joint pipeline (LRPITT12 joint; LRPITT15/15b contrasts)
# ---------------------------------------------------------------------------


def fit_joint(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    assert spec.kind == "joint"

    ctx = make_context(spec, config)

    section_header("Prepare data")
    # A joint model may target an explicit outcome set (e.g. the taught-vs-not-
    # taught contrast in LRPITT15/15b); load exactly those so the complete-case mask is
    # not driven by the eight standardised outcomes. Defaults to ITT_OUTCOMES.
    joint_outcomes = spec.extra.get("outcomes")
    if joint_outcomes is not None:
        prepared = load_and_prepare(phase_mode="itt", outcomes=tuple(joint_outcomes))
    else:
        prepared = load_and_prepare(phase_mode="itt")
    ctx.prepared = prepared

    _print_header(ctx)

    section_header("Build model")

    built = _factories.build_joint_model(
        prepared,
        outcomes=spec.extra.get("outcomes"),
        use_age_gp=spec.extra.get("use_age_gp", False),
        partial_pool_age_gp=spec.extra.get("partial_pool_age_gp", True),
        use_residual_correlation=spec.extra.get("use_residual_correlation", False),
        use_cross_baselines=spec.extra.get("use_cross_baselines", True),
        use_age_linear=spec.extra.get("use_age_linear", False),
    )
    ctx.model = built.model
    ctx.model_vars = built.variables
    ctx.prepared = built.prepared

    _emit_priors(ctx)
    _render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)

    section_header("Sampling posterior (nutpie)")
    _diag.sample_posterior(ctx)

    section_header("LOO-PSIS")
    _diag.compute_log_likelihood_and_loo(ctx)
    _report.write_loo_summary(ctx)
    _print_loo_row(ctx)

    section_header("Summary diagnostics")
    _joint_vars = ["alpha", "tau", "gamma_own", "kappa"]
    if spec.extra.get("use_age_linear", False):
        _joint_vars.append("gamma_A")
    if spec.extra.get("use_residual_correlation", False):
        _joint_vars.append("sigma_outcome")
    _diag.summary_diagnostics(ctx, var_names=_joint_vars)

    section_header("Posterior predictive")
    _diag.sample_posterior_predictive(ctx, var_names=["y_post"])
    _save_ppc(ctx)

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=_joint_vars)
    _diag.run_extended_diagnostics(ctx, causal_term="tau")
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=_joint_vars)
    # One forest of every outcome's tau — the single most communicative joint artifact.
    _save_forest_plot(ctx, ["tau"])

    section_header("Treatment-effect summary")
    outcomes = list(ctx.trace.posterior["outcome"].values)
    tau_df = _report.tau_summary_joint(ctx.trace, outcomes, ci_prob=ctx.reporting.hdi)
    tau_df.to_csv(os.path.join(ctx.output_dir, "tau_summary.csv"), index=False)
    ctx.tables["tau_summary"] = tau_df
    print_table(
        ranked_dataframe_table(
            tau_df,
            title=f"tau by outcome - {int(ctx.reporting.hdi * 100)}% CI (equal-tailed)",
            columns=["outcome", "tau_mean", "tau_lo", "tau_hi", "prob_pos"],
            rank_column=False,
        )
    )

    contrast = _report.tau_contrast_matrix(ctx.trace, outcomes)
    contrast.to_csv(os.path.join(ctx.output_dir, "tau_contrast_matrix.csv"))
    ctx.tables["tau_contrast_matrix"] = contrast
    _save_contrast_heatmap(ctx, contrast)

    meta_extra: dict = {"loo_elpd": float(ctx.loo.elpd)}

    # Headline difference parameter for a two-outcome contrast (LRPITT15/15b: taught vs
    # not-taught). ``difference = (a, b)`` reports the posterior of tau[a]-tau[b].
    difference = spec.extra.get("difference")
    if difference is not None:
        pair = tuple(difference)
        section_header("Treatment-effect difference")
        diff_s = _report.tau_difference_summary(
            ctx.trace, outcomes, pair, ci_prob=ctx.reporting.hdi
        )
        diff_df = pd.DataFrame([diff_s])
        diff_df.to_csv(os.path.join(ctx.output_dir, "tau_difference.csv"), index=False)
        ctx.tables["tau_difference"] = diff_df
        print_table(
            metrics_table(
                [{"metric": k, "value": v} for k, v in diff_s.items()],
                title=(
                    f"tau[{pair[0]}] - tau[{pair[1]}] "
                    f"- {int(ctx.reporting.hdi * 100)}% CI (equal-tailed)"
                ),
                columns=["metric", "value"],
            )
        )
        meta_extra["tau_difference"] = diff_s

    _report.write_run_metadata(ctx, extra=meta_extra)

    section_header("Report")
    _copy_report_template(ctx)
    _print_footer(ctx)
    return ctx


# ---------------------------------------------------------------------------
# Waitlist-crossover / difference-in-differences pipeline (kind="did")
# ---------------------------------------------------------------------------


def _did_diag_vars(spec: ModelSpec) -> list[str]:
    """Scalar coefficients to summarise for a crossover/DiD fit, given the spec."""
    dose = bool(spec.extra.get("dose", False))
    v = ["alpha", "beta_period", "beta_dose" if dose else "delta", "gamma_own", "kappa"]
    if spec.extra.get("use_age", True):
        v.append("gamma_A")
    if spec.extra.get("use_child_re", True):
        v.append("sigma_child")
    return v


def fit_did(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    assert spec.kind == "did"
    assert spec.outcome_symbol is not None

    ctx = make_context(spec, config)

    section_header("Prepare data")
    sym = spec.outcome_symbol
    dose = bool(spec.extra.get("dose", False))
    # Phase-stacked frame; load only this outcome so the complete-case mask does
    # not drop rows for measures the model never uses. The dose variant also needs
    # the per-period intervention-session count.
    outcomes = tuple(spec.extra.get("outcomes", (sym,)))
    covariates = ("attend",) if dose else ()
    prepared = load_and_prepare(
        phase_mode="all", outcomes=outcomes, covariates=covariates
    )
    ctx.prepared = prepared

    _print_header(ctx)

    section_header("Build model")
    _priors.save_shared_prior_panel(ctx.output_dir)
    built = _factories.build_did_model(
        prepared,
        outcome_symbol=sym,
        periods=tuple(spec.extra.get("periods", (0, 1))),
        use_child_re=spec.extra.get("use_child_re", True),
        use_age=spec.extra.get("use_age", True),
        dose=dose,
    )
    ctx.model = built.model
    ctx.model_vars = built.variables
    ctx.prepared = built.prepared

    _render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000, var_names=["y_post", "eta"])

    section_header("Sampling posterior (nutpie)")
    _diag.sample_posterior(ctx)

    section_header("LOO-PSIS")
    _diag.compute_log_likelihood_and_loo(ctx)
    _report.write_loo_summary(ctx)
    _print_loo_row(ctx)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=_did_diag_vars(spec))

    section_header("Posterior predictive")
    _diag.sample_posterior_predictive(ctx, var_names=["y_post"])
    _save_ppc(ctx)
    _diag.save_trace(ctx)

    section_header("Crossover / DiD treatment-effect summary")
    from language_reading_predictors.statistical_models.measures import MEASURES

    did_s = _report.did_summary(
        ctx.trace,
        ci_prob=ctx.reporting.hdi,
        n_trials=MEASURES[sym].n_trials,
        dose=dose,
    )
    did_df = pd.DataFrame([did_s])
    did_df.to_csv(os.path.join(ctx.output_dir, "did_summary.csv"), index=False)
    ctx.tables["did_summary"] = did_df
    print_table(
        metrics_table(
            [{"metric": k, "value": v} for k, v in did_s.items()],
            title=(
                f"crossover/DiD effect ({sym}) - {int(ctx.reporting.hdi * 100)}% CI "
                "(equal-tailed); positive = intervention helps"
            ),
            columns=["metric", "value"],
        )
    )

    _report.write_run_metadata(
        ctx,
        extra={"loo_elpd": float(ctx.loo.elpd), "did_summary": did_s, "dose": dose},
    )

    section_header("Report")
    _copy_report_template(ctx)
    _print_footer(ctx)
    return ctx


# ---------------------------------------------------------------------------
# Mechanism pipeline (LRP56 / LRP57 / LRP58)
# ---------------------------------------------------------------------------


def fit_mechanism(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    assert spec.kind == "mechanism"
    assert spec.mechanism_symbol is not None

    ctx = make_context(spec, config)

    section_header("Prepare data")
    # A model may restrict the prepared outcomes (e.g. LRP72 uses only L/B/N) so
    # ``drop_missing_pre`` does not discard rows for measures the model ignores.
    extra_outcomes = spec.extra.get("outcomes")
    if extra_outcomes is not None:
        prepared = load_and_prepare(phase_mode="all", outcomes=tuple(extra_outcomes))
    else:
        prepared = load_and_prepare(phase_mode="all")
    ctx.prepared = prepared

    _print_header(ctx)

    section_header("Build model")
    _priors.save_shared_prior_panel(ctx.output_dir)

    moderator_symbol = spec.extra.get("moderator_symbol")
    # Drop the autoregressive baseline (any ``*_pre`` token, e.g. W_pre / N_pre)
    # from the confounder list — it enters via ``adjust_baseline_symbol``.
    confounders = [s for s in spec.adjustment if not s.endswith("_pre")]
    if moderator_symbol is not None:
        # The moderator is carried by its standardised main effect + interaction
        # in the factory, so drop it from the plain confounder loop to avoid a
        # collinear duplicate main effect. The standardised term still adjusts
        # for M, preserving any DAG-required conditioning (e.g. E in LRP71).
        confounders = [s for s in confounders if s != moderator_symbol]

    built = _factories.build_mechanism_model(
        prepared,
        mechanism_symbol=spec.mechanism_symbol,
        outcome_symbol=spec.outcome_symbol or "W",
        adjust_baseline_symbol=spec.extra.get("adjust_baseline_symbol", "W"),
        confounder_symbols=tuple(s for s in confounders if s in ("G", "A") or len(s) == 1),
        use_age_gp=spec.extra.get("use_age_gp", False),
        phase_specific_mechanism=spec.extra.get("phase_specific_mechanism", False),
        use_subject_random_intercept=spec.extra.get(
            "use_subject_random_intercept", True
        ),
        moderator_symbol=moderator_symbol,
        moderator_is_covariate=spec.extra.get("moderator_is_covariate", False),
        include_interaction=spec.extra.get("include_interaction", True),
        linear_mechanism=spec.extra.get("linear_mechanism", False),
    )
    ctx.model = built.model
    ctx.model_vars = built.variables
    ctx.prepared = built.prepared

    _render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000, var_names=["y_post", "eta"])

    section_header("Sampling posterior (nutpie)")
    _diag.sample_posterior(ctx)

    section_header("LOO-PSIS")
    _diag.compute_log_likelihood_and_loo(ctx)
    _report.write_loo_summary(ctx)
    _print_loo_row(ctx)

    section_header("Summary diagnostics")
    _mech_vars = ["alpha", "beta_G", "gamma_own", "kappa"]
    if spec.extra.get("use_subject_random_intercept", True):
        _mech_vars.append("sigma_child")
    if spec.extra.get("linear_mechanism", False):
        _mech_vars.append("beta_mech")
    if moderator_symbol is not None:
        _mech_vars.append("gamma_mod")
        if spec.extra.get("include_interaction", True):
            _mech_vars.append("gamma_int")
    _diag.summary_diagnostics(ctx, var_names=_mech_vars)

    section_header("Posterior predictive")
    _diag.sample_posterior_predictive(ctx, var_names=["y_post"])
    _save_ppc(ctx)

    # Mechanism curve: f_mech vs mech_post_logit grid (logit-contribution scale only).
    section_header("Mechanism curve")
    _write_mechanism_curve(ctx)

    meta_extra = {"loo_elpd": float(ctx.loo.elpd), "adjustment": spec.adjustment}

    # Linear-moderation summary (gamma_int / gamma_mod), when a moderator is set.
    if moderator_symbol is not None:
        section_header("Interaction summary")
        gi = _report.gamma_interaction_summary(ctx.trace, ci_prob=ctx.reporting.hdi)
        gi_df = pd.DataFrame([gi])
        gi_df.to_csv(
            os.path.join(ctx.output_dir, "interaction_summary.csv"), index=False
        )
        ctx.tables["interaction_summary"] = gi_df
        print_table(
            metrics_table(
                [{"metric": k, "value": v} for k, v in gi.items()],
                title=(
                    f"Linear moderation by {moderator_symbol} "
                    f"- {int(ctx.reporting.hdi * 100)}% CI (equal-tailed)"
                ),
                columns=["metric", "value"],
            )
        )
        meta_extra["moderator_symbol"] = moderator_symbol
        meta_extra["interaction_summary"] = gi

    _diag.save_trace(ctx)
    _report.write_run_metadata(ctx, extra=meta_extra)

    section_header("Report")
    _copy_report_template(ctx)
    _print_footer(ctx)
    return ctx


def _write_mechanism_curve(ctx: StatisticalFitContext) -> None:
    """Posterior adjusted dose-response of the mechanism predictor on the outcome.

    With the HSGP ``f_mech`` on (the default) this is the non-parametric curve. When
    the model uses the linear slope instead (``linear_mechanism=True``, so no
    ``f_mech`` variable exists) it falls back to the straight
    ``beta_mech * z(logit(predictor))`` band — the predictor's linear logit
    contribution (at the mean of any moderator) — so the adjusted predictor->outcome
    relationship is still shown rather than left implicit in a coefficient. Both
    branches hold the adjustment set fixed and write the identical CSV/PNG schema.
    Guarded by the caller.
    """
    post = ctx.trace.posterior

    from language_reading_predictors.statistical_models.measures import MEASURES
    from language_reading_predictors.statistical_models.preprocessing import (
        logit_safe,
        standardise,
    )

    sym = ctx.spec.mechanism_symbol
    N = MEASURES[sym].n_trials
    mech_logit = logit_safe(ctx.prepared.post_counts[sym], N)

    if "f_mech" in post:
        f = post["f_mech"].stack(sample=("chain", "draw")).values  # (n_obs, n_sample)
        kind = "GP"
    elif "beta_mech" in post:
        # Linear mechanism: the predictor enters as beta_mech * z(logit), with z the
        # same standardisation the factory applied. Build the per-observation logit
        # contribution so the band mirrors the GP branch (an exact straight line).
        z_L, _ = standardise(mech_logit)
        b = post["beta_mech"].stack(sample=("chain", "draw")).values  # (n_sample,)
        f = z_L[:, None] * b[None, :]  # (n_obs, n_sample)
        kind = "linear"
    else:
        return

    order = np.argsort(mech_logit)
    x = mech_logit[order]
    f_ord = f[order]
    mean = f_ord.mean(axis=1)
    lo = np.quantile(f_ord, 0.025, axis=1)
    hi = np.quantile(f_ord, 0.975, axis=1)
    pd.DataFrame(
        {"mech_logit": x, "f_mean": mean, "f_lo": lo, "f_hi": hi}
    ).to_csv(os.path.join(ctx.output_dir, "mechanism_curve.csv"), index=False)
    outcome = ctx.spec.outcome_symbol or "W"
    plt.figure(figsize=(6, 4))
    plt.plot(x, mean, color="#1f77b4", lw=2)
    plt.fill_between(x, lo, hi, color="#1f77b4", alpha=0.2)
    plt.xlabel(f"logit({sym}_post)")
    plt.ylabel("predictor logit contribution")
    plt.title(f"Mechanism curve ({kind}): {sym} -> {outcome}")
    plt.savefig(
        os.path.join(ctx.output_dir, "mechanism_curve.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# ---------------------------------------------------------------------------
# Mediation pipeline (LRP59)
# ---------------------------------------------------------------------------

_MED_COEF_VARS = [
    "a0", "a_G", "a_L", "a_A", "a_E", "a_R", "kappa_M",
    "b0", "b_G", "b_M", "b_GM", "b_W", "b_A", "b_E", "b_R", "kappa_Y",
]

# LRP62 (gaussian_composite): the mediator leg is Normal (a_comp / sigma_M) and
# the observed mediator node is "M_post" rather than the Beta-Binomial "L_post".
_MED_COEF_VARS_GAUSSIAN = [
    "a0", "a_G", "a_comp", "a_A", "a_E", "a_R", "sigma_M",
    "b0", "b_G", "b_M", "b_GM", "b_W", "b_A", "b_E", "b_R", "kappa_Y",
]


_T3_SENSITIVITY_TIME = 3  # post-RCT wave used for the temporal-ordering check


def _fit_t3_sensitivity(
    ctx: StatisticalFitContext,
    spec: ModelSpec,
    *,
    confounders: tuple[str, ...],
    mediator_kind: str,
    route_symbols: tuple[str, ...],
):
    """Temporal-ordering sensitivity fit for the mediation models (issue #84).

    Refits the *identical* mediation model but with the outcome measured at a
    later wave (t3) while the mediator stays at t2, so the mediator precedes the
    outcome in time. The t2 -> t3 increment is **not randomised** (both arms are
    treated after t2), so this is a triangulation point for the contemporaneous
    measurement caveat, not a cleaner causal estimate. Returns the g-formula
    decomposition DataFrame for the t3-outcome variant.
    """
    import pymc as pm

    from language_reading_predictors.statistical_models import mediation as _med

    outcome_symbol = spec.outcome_symbol or "W"
    prepared_t3 = load_and_prepare_lagged_outcome(
        outcome_symbol, outcome_time=_T3_SENSITIVITY_TIME
    )
    built_t3, med_t3 = _factories.build_mediation_model(
        prepared_t3,
        mediator_symbol=spec.mechanism_symbol or "L",
        outcome_symbol=outcome_symbol,
        confounder_symbols=confounders,
        mediator_kind=mediator_kind,
        route_symbols=route_symbols,
    )
    s = ctx.sampling
    with built_t3.model:
        trace_t3 = pm.sample(
            draws=s.draws,
            tune=s.tune,
            chains=s.chains,
            cores=s.cores,
            target_accept=s.target_accept,
            nuts_sampler="nutpie",
            return_inferencedata=True,
            random_seed=s.random_seed,
            progressbar=False,
        )
    return _med.decompose(
        trace_t3,
        med_t3,
        ci_prob=ctx.reporting.hdi,
    )
def fit_mediation(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """ITT-phase mediation decomposition (LRP59): how much of G -> W flows via L."""
    assert spec.kind == "mediation"
    from language_reading_predictors.statistical_models import mediation as _med

    ctx = make_context(spec, config)

    section_header("Prepare data")
    # Phase 0 only (t1 -> t2): the single randomised contrast. One row per child.
    prepared = load_and_prepare(phase_mode="itt")
    ctx.prepared = prepared

    _print_header(ctx)

    section_header("Build model")
    _priors.save_shared_prior_panel(ctx.output_dir)

    confounders = tuple(
        s for s in spec.adjustment if s not in ("G", "A", "L_t1", "W_pre")
    )
    mediator_kind = spec.extra.get("mediator_kind", "beta_binomial")
    route_symbols = tuple(spec.extra.get("route_symbols", ()))
    built, med_data = _factories.build_mediation_model(
        prepared,
        mediator_symbol=spec.mechanism_symbol or "L",
        outcome_symbol=spec.outcome_symbol or "W",
        confounder_symbols=confounders,
        mediator_kind=mediator_kind,
        route_symbols=route_symbols,
    )
    ctx.model = built.model
    ctx.model_vars = built.variables
    ctx.prepared = built.prepared

    # The mediator observed node differs by kind: Beta-Binomial "L_post" vs the
    # Gaussian composite "M_post".
    is_gaussian = mediator_kind == "gaussian_composite"
    mediator_node = "M_post" if is_gaussian else "L_post"
    # Diagnose every scalar coefficient the model actually built (deterministics
    # and the observed mediator/outcome nodes are not free RVs), so the list
    # tracks the fitted confounder set instead of a hand-maintained constant.
    coef_vars = sorted(rv.name for rv in built.model.free_RVs if rv.ndim == 0)

    _render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000, var_names=[mediator_node, "y_post"])

    section_header("Sampling posterior (nutpie)")
    _diag.sample_posterior(ctx)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=coef_vars)

    section_header("Posterior predictive")
    _diag.sample_posterior_predictive(ctx, var_names=[mediator_node, "y_post"])
    _save_ppc(ctx)
    _diag.save_trace(ctx)

    section_header("Mediation decomposition (g-formula)")
    med_df = _med.decompose(
        ctx.trace,
        med_data,
        ci_prob=ctx.reporting.hdi,
    )
    med_df.to_csv(os.path.join(ctx.output_dir, "mediation_summary.csv"), index=False)
    ctx.tables["mediation_summary"] = med_df
    print_table(
        ranked_dataframe_table(
            med_df,
            title=f"Mediation (intervention-helps; words out of {med_data.n_trials_W})",
            columns=["quantity", "words_mean", "words_lo", "words_hi", "prob_pos"],
            rank_column=False,
            precision=3,
        )
    )

    # --- Temporal-ordering sensitivity: outcome at t3, mediator still at t2 ---
    # Triangulation for the contemporaneous-measurement caveat (issue #84): the
    # mediator now precedes the outcome in time. NB the t2 -> t3 increment is not
    # randomised (both arms treated after t2), so read this as triangulation only.
    section_header("Temporal-ordering sensitivity (outcome at t3)")
    med_df_t3 = _fit_t3_sensitivity(
        ctx,
        spec,
        confounders=confounders,
        mediator_kind=mediator_kind,
        route_symbols=route_symbols,
    )
    med_df_t3.to_csv(
        os.path.join(ctx.output_dir, "mediation_summary_t3.csv"), index=False
    )
    ctx.tables["mediation_summary_t3"] = med_df_t3
    print_table(
        ranked_dataframe_table(
            med_df_t3,
            title="Temporal-ordering sensitivity (outcome W at t3; NOT randomised)",
            columns=["quantity", "words_mean", "words_lo", "words_hi", "prob_pos"],
            rank_column=False,
            precision=3,
        )
    )

    _summary = {r["quantity"]: r for r in med_df.to_dict("records")}
    _summary_t3 = {r["quantity"]: r for r in med_df_t3.to_dict("records")}
    _report.write_run_metadata(
        ctx,
        extra={
            "adjustment": spec.adjustment,
            "n_obs": prepared.n_obs,
            "mediation": _summary,
            "mediation_t3_sensitivity": _summary_t3,
        },
    )

    section_header("Report")
    _copy_report_template(ctx)
    _print_footer(ctx)
    return ctx


# ---------------------------------------------------------------------------
# Gain-factors / level-factors pipelines (LRPGF / LRPLF, #127)
# ---------------------------------------------------------------------------


def _gf_coef_names(spec: ModelSpec) -> list[str]:
    """Factor coefficients to report in the LRPGF factor table (interpretable
    terms only; nuisance alpha/alpha_phase/kappa/sigma_child are excluded)."""
    extra = spec.extra
    treated_only = bool(extra.get("treated_only", False))
    names: list[str] = []
    if not treated_only:
        names.append("beta_trt")
    names += ["gamma_own", "gamma_A"]
    if extra.get("ability_covariate"):
        names.append("gamma_ability")
    names += [f"gamma_{s}" for s in extra.get("skill_symbols", ())]
    for pair in extra.get("interactions", ()):
        a, b = tuple(pair)
        if treated_only and "trt" in (a, b):
            continue
        names.append(f"gamma_int_{a}_{b}")
    return names


def _gf_diag_vars(spec: ModelSpec) -> list[str]:
    # No kappa under the off-floor Bernoulli likelihood.
    tail = (
        ["sigma_child"]
        if spec.extra.get("likelihood") == "bernoulli_offfloor"
        else ["kappa", "sigma_child"]
    )
    return ["alpha", *_gf_coef_names(spec), *tail]


def fit_gain_factors(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    assert spec.kind == "gain_factors"
    assert spec.outcome_symbol is not None
    ctx = make_context(spec, config)
    extra = spec.extra

    section_header("Prepare data")
    skill_symbols = tuple(extra.get("skill_symbols", ()))
    ability_covariate = extra.get("ability_covariate")
    interactions = tuple(tuple(p) for p in extra.get("interactions", ()))
    treated_only = bool(extra.get("treated_only", False))
    likelihood = extra.get("likelihood", "beta_binomial")
    off_floor = likelihood == "bernoulli_offfloor"
    obs_node = "y_offfloor" if off_floor else "y_post"
    baseline_covariates = (ability_covariate,) if ability_covariate else ()
    prepared = load_and_prepare(
        phase_mode="all",
        outcomes=(spec.outcome_symbol, *skill_symbols),
        baseline_covariates=baseline_covariates,
    )
    ctx.prepared = prepared
    _print_header(ctx)

    section_header("Build model")
    built = _factories.build_gain_factors_model(
        prepared,
        outcome_symbol=spec.outcome_symbol,
        skill_symbols=skill_symbols,
        ability_covariate=ability_covariate,
        interactions=interactions,
        treated_only=treated_only,
        likelihood=likelihood,
    )
    ctx.model = built.model
    ctx.model_vars = built.variables
    ctx.prepared = built.prepared

    _emit_priors(ctx)
    _render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    _diag.save_prior_predictive_plot(ctx, spec.outcome_symbol, node=obs_node)

    section_header("Sampling posterior (nutpie)")
    _diag.sample_posterior(ctx)

    section_header("LOO-PSIS")
    _diag.compute_log_likelihood_and_loo(ctx)
    _report.write_loo_summary(ctx)
    _print_loo_row(ctx)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=_gf_diag_vars(spec))

    section_header("Posterior predictive")
    _diag.sample_posterior_predictive(ctx, var_names=[obs_node])
    _save_ppc(ctx)

    section_header("Extended diagnostics")
    _causal_gf = None if treated_only else "beta_trt"
    _diag.write_diagnostics_summary(ctx, var_names=_gf_diag_vars(spec))
    _diag.run_extended_diagnostics(ctx, causal_term=_causal_gf)
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=_gf_diag_vars(spec))
    if _causal_gf is not None:
        _save_forest_plot(ctx, [_causal_gf])
        _diag.run_psense(ctx, var_names=[_causal_gf])

    section_header("Factor summary")
    fs = _report.factor_summary(
        ctx.trace, _gf_coef_names(spec), ci_prob=ctx.reporting.hdi,
        causal_terms=("beta_trt",),
    )
    fs.to_csv(os.path.join(ctx.output_dir, "factor_summary.csv"), index=False)
    ctx.tables["factor_summary"] = fs
    _save_association_forest(ctx, _gf_coef_names(spec), ("beta_trt",))
    print_table(
        ranked_dataframe_table(
            fs,
            title=f"Factor summary ({spec.outcome_symbol}) - {int(ctx.reporting.hdi * 100)}% CrI",
            columns=["term", "role", "mean", "lo", "hi", "prob_positive"],
            rank_column=False,
            precision=3,
        )
    )

    meta_extra = {"loo_elpd": float(ctx.loo.elpd), "treated_only": treated_only}
    # Items-scale marginal effect of the treatment term. Skipped when
    # treated_only (the on-intervention indicator is then constant and beta_trt
    # is absent).
    if not treated_only:
        trt = ((built.prepared.G == 1) | (built.prepared.phase >= 1)).astype(float)
        # Off-floor models are Bernoulli on Pr(post > 0); the "items" scale then
        # collapses to the off-floor risk difference (n_trials = 1).
        n_marg = 1 if off_floor else built.prepared.n_trials[spec.outcome_symbol]
        tme = _report.treatment_marginal_effect(
            ctx.trace,
            trt=trt,
            n_trials=n_marg,
            ci_prob=ctx.reporting.hdi,
        )
        pd.DataFrame([tme]).to_csv(
            os.path.join(ctx.output_dir, "treatment_marginal.csv"), index=False
        )
        ctx.tables["treatment_marginal"] = pd.DataFrame([tme])
        meta_extra["treatment_marginal"] = tme
        print_table(
            metrics_table(
                [{"metric": k, "value": v} for k, v in tme.items()],
                title="Treatment items-scale marginal effect",
                columns=["metric", "value"],
            )
        )

        # Prior pushforward on the same scale (estimand-scale prior check, #125).
        try:
            pf = _report.prior_pushforward(
                ctx.prior_samples, G=trt, n_trials=n_marg,
                term="beta_trt", varying_term="", ci_prob=ctx.reporting.hdi,
            )
            pd.DataFrame([pf]).to_csv(
                os.path.join(ctx.output_dir, "prior_pushforward.csv"), index=False
            )
            ctx.tables["prior_pushforward"] = pd.DataFrame([pf])
        except Exception as exc:  # pragma: no cover
            rprint(f"[yellow]prior pushforward skipped: {exc}[/yellow]")

        # ROPE-anchored continuous report for the one causal term (beta_trt),
        # mirroring fit_itt (notes/202606261304-evidence-strength-and-rope-
        # reporting.md): separates direction (pd) from a *meaningful* benefit
        # (P(items >= delta)). Graded outcomes with an agreed items-scale delta
        # (ROPE_DELTA -> W/R/E/L/B) use the items scale; the floored outcome P (off-
        # floor) uses the provisional risk-difference delta (ROPE_DELTA_PROB, #130
        # follow-up); F/T have no agreed delta and are skipped.
        from language_reading_predictors.statistical_models.measures import (
            ROPE_DELTA,
            ROPE_DELTA_PROB,
        )

        delta_items = ROPE_DELTA.get(spec.outcome_symbol)
        delta_prob = ROPE_DELTA_PROB.get(spec.outcome_symbol)
        if delta_items is not None and not off_floor:
            rope_s = _report.rope_summary(
                ctx.trace,
                G=trt,
                n_trials=n_marg,
                delta=delta_items,
                ci_prob=ctx.reporting.hdi,
                term="beta_trt",
                varying_term="",
            )
            rope_df = pd.DataFrame([rope_s])
            rope_df.to_csv(os.path.join(ctx.output_dir, "rope_summary.csv"), index=False)
            ctx.tables["rope_summary"] = rope_df
            meta_extra["rope_summary"] = rope_s
            print_table(
                metrics_table(
                    [{"metric": k, "value": v} for k, v in rope_s.items()],
                    title=f"ROPE summary ({spec.outcome_symbol}, delta={delta_items:g} items)",
                    columns=["metric", "value"],
                )
            )
            _save_rope_plot(
                ctx, spec.outcome_symbol, trt, n_marg, delta_items,
                term="beta_trt", varying_term="",
            )
        elif off_floor and delta_prob is not None:
            # Off-floor risk-difference ROPE (provisional delta), matching the
            # floored ITT path (#125 Area 4; #130 follow-up).
            rope_s = _report.rope_summary(
                ctx.trace, G=trt, n_trials=1, delta=delta_prob,
                ci_prob=ctx.reporting.hdi, term="beta_trt", varying_term="",
            )
            rope_s["provisional_delta"] = True
            rope_s["delta_scale"] = "risk_difference"
            pd.DataFrame([rope_s]).to_csv(
                os.path.join(ctx.output_dir, "rope_summary.csv"), index=False
            )
            ctx.tables["rope_summary"] = pd.DataFrame([rope_s])
            meta_extra["rope_summary"] = rope_s
            _save_rope_plot(
                ctx, spec.outcome_symbol, trt, 1, delta_prob,
                term="beta_trt", varying_term="",
            )

    _report.write_run_metadata(ctx, extra=meta_extra)
    section_header("Report")
    _copy_report_template(ctx)
    _print_footer(ctx)
    return ctx


def _lf_coef_names(spec: ModelSpec) -> list[str]:
    extra = spec.extra
    names = ["b_grp_time" if extra.get("group_by_time", True) else "beta_grp", "gamma_A"]
    if extra.get("ability_covariate"):
        names.append(
            "gamma_ability_time" if extra.get("ability_by_time", True) else "gamma_ability"
        )
        if extra.get("group_ability", True):
            names.append("gamma_grp_ability")
    return names


def _lf_diag_vars(spec: ModelSpec) -> list[str]:
    tail = (
        ["sigma_child"]
        if spec.extra.get("likelihood") == "bernoulli_offfloor"
        else ["kappa", "sigma_child"]
    )
    return ["alpha", "alpha_time", *_lf_coef_names(spec), *tail]


def fit_level_factors(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    assert spec.kind == "level_factors"
    assert spec.outcome_symbol is not None
    ctx = make_context(spec, config)
    extra = spec.extra

    section_header("Prepare data")
    ability_covariate = extra.get("ability_covariate")
    likelihood = extra.get("likelihood", "beta_binomial")
    off_floor = likelihood == "bernoulli_offfloor"
    obs_node = "y_offfloor" if off_floor else "y_post"
    baseline_covariates = (ability_covariate,) if ability_covariate else ()
    prepared = load_and_prepare(
        phase_mode="levels",
        outcomes=(spec.outcome_symbol,),
        baseline_covariates=baseline_covariates,
    )
    ctx.prepared = prepared
    _print_header(ctx)

    section_header("Build model")
    built = _factories.build_level_factors_model(
        prepared,
        outcome_symbol=spec.outcome_symbol,
        ability_covariate=ability_covariate,
        group_by_time=bool(extra.get("group_by_time", True)),
        ability_by_time=bool(extra.get("ability_by_time", True)),
        group_ability=bool(extra.get("group_ability", True)),
        likelihood=likelihood,
    )
    ctx.model = built.model
    ctx.model_vars = built.variables
    ctx.prepared = built.prepared

    _emit_priors(ctx)
    _render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    _diag.save_prior_predictive_plot(ctx, spec.outcome_symbol, node=obs_node)

    section_header("Sampling posterior (nutpie)")
    _diag.sample_posterior(ctx)

    section_header("LOO-PSIS")
    _diag.compute_log_likelihood_and_loo(ctx)
    _report.write_loo_summary(ctx)
    _print_loo_row(ctx)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=_lf_diag_vars(spec))

    section_header("Posterior predictive")
    _diag.sample_posterior_predictive(ctx, var_names=[obs_node])
    _save_ppc(ctx)

    section_header("Extended diagnostics")
    _causal_lf = "b_grp_time" if extra.get("group_by_time", True) else "beta_grp"
    _diag.write_diagnostics_summary(ctx, var_names=_lf_diag_vars(spec))
    _diag.run_extended_diagnostics(ctx, causal_term=_causal_lf)
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=_lf_diag_vars(spec))
    _save_forest_plot(ctx, [_causal_lf])
    _diag.run_psense(ctx, var_names=[_causal_lf])

    section_header("Factor summary")
    # Only the t2 group contrast (b_grp_time[1]) is the clean randomised effect;
    # the other timepoints are post-crossover (see the level-model caveat).
    causal = ("b_grp_time[1]",) if extra.get("group_by_time", True) else ()
    fs = _report.factor_summary(
        ctx.trace, _lf_coef_names(spec), ci_prob=ctx.reporting.hdi, causal_terms=causal
    )
    fs.to_csv(os.path.join(ctx.output_dir, "factor_summary.csv"), index=False)
    ctx.tables["factor_summary"] = fs
    _save_association_forest(ctx, _lf_coef_names(spec), causal)
    print_table(
        ranked_dataframe_table(
            fs,
            title=f"Factor summary ({spec.outcome_symbol}) - {int(ctx.reporting.hdi * 100)}% CrI",
            columns=["term", "role", "mean", "lo", "hi", "prob_positive"],
            rank_column=False,
            precision=3,
        )
    )

    meta_extra = {"loo_elpd": float(ctx.loo.elpd)}
    # ROPE-anchored continuous report for the one causal term — the t2 randomised
    # contrast b_grp_time[1] (notes/202606261304-...). The level model enters group
    # as a per-timepoint vector and also carries a group×ability interaction, so the
    # t2 items-scale AME nets both group terms out at the t2 rows
    # (reporting.level_t2_marginal_effect) rather than reusing the gain core. Emitted
    # only for graded outcomes with an agreed delta (ROPE_DELTA -> W/R/E/L/B) and when
    # the t2 contrast exists (group_by_time); P (off-floor) and F/T are skipped, as in
    # the ITT path.
    from language_reading_predictors.statistical_models.measures import ROPE_DELTA

    delta_items = ROPE_DELTA.get(spec.outcome_symbol)
    if delta_items is not None and not off_floor and extra.get("group_by_time", True):
        ability = (
            built.prepared.covariates[ability_covariate]
            if ability_covariate is not None
            else None
        )
        contrast_draws, ame_prob = _report.level_t2_marginal_effect(
            ctx.trace,
            phase=built.prepared.phase,
            G=built.prepared.G,
            ability=ability,
        )
        n_marg = int(built.prepared.n_trials[spec.outcome_symbol])
        items = ame_prob * n_marg
        rope_s = _report._rope_card(
            contrast_draws, items, delta=delta_items, ci_prob=ctx.reporting.hdi
        )
        rope_df = pd.DataFrame([rope_s])
        rope_df.to_csv(os.path.join(ctx.output_dir, "rope_summary.csv"), index=False)
        ctx.tables["rope_summary"] = rope_df
        meta_extra["rope_summary"] = rope_s
        print_table(
            metrics_table(
                [{"metric": k, "value": v} for k, v in rope_s.items()],
                title=f"ROPE summary (t2 contrast, {spec.outcome_symbol}, delta={delta_items:g} items)",
                columns=["metric", "value"],
            )
        )
        _save_rope_plot(ctx, spec.outcome_symbol, None, n_marg, delta_items, items=items)

    _report.write_run_metadata(ctx, extra=meta_extra)
    section_header("Report")
    _copy_report_template(ctx)
    _print_footer(ctx)
    return ctx


def _al_coef_names(spec: ModelSpec) -> list[str]:
    """Interpretable LRPAL coefficients (alpha/kappa excluded)."""
    extra = spec.extra
    names: list[str] = []
    if extra.get("use_cohort", True):
        names.append("beta_cohort")
    names += ["gamma_own", "gamma_A"]
    if extra.get("ability_covariate"):
        names.append("gamma_ability")
    if extra.get("use_dose", False):
        names.append("gamma_dose")
    return names


def _al_diag_vars(spec: ModelSpec) -> list[str]:
    # No child random intercept (one row per child); no kappa off-floor.
    tail = [] if spec.extra.get("likelihood") == "bernoulli_offfloor" else ["kappa"]
    return ["alpha", *_al_coef_names(spec), *tail]


def fit_aligned(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    assert spec.kind == "aligned"
    assert spec.outcome_symbol is not None
    ctx = make_context(spec, config)
    extra = spec.extra

    section_header("Prepare data")
    ability_covariate = extra.get("ability_covariate")
    use_cohort = bool(extra.get("use_cohort", True))
    use_dose = bool(extra.get("use_dose", False))
    likelihood = extra.get("likelihood", "beta_binomial")
    off_floor = likelihood == "bernoulli_offfloor"
    obs_node = "y_offfloor" if off_floor else "y_post"
    prepared = load_and_prepare_aligned(
        outcomes=(spec.outcome_symbol,),
        ability_covariate=ability_covariate,
        include_dose=use_dose,
    )
    ctx.prepared = prepared
    _print_header(ctx)

    section_header("Build model")
    _priors.save_shared_prior_panel(ctx.output_dir)
    built = _factories.build_aligned_model(
        prepared,
        outcome_symbol=spec.outcome_symbol,
        ability_covariate=ability_covariate,
        use_cohort=use_cohort,
        use_dose=use_dose,
        likelihood=likelihood,
    )
    ctx.model = built.model
    ctx.model_vars = built.variables
    ctx.prepared = built.prepared

    _render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000, var_names=[obs_node, "eta"])
    _diag.save_prior_predictive_plot(ctx, spec.outcome_symbol, node=obs_node)

    section_header("Sampling posterior (nutpie)")
    _diag.sample_posterior(ctx)

    section_header("LOO-PSIS")
    _diag.compute_log_likelihood_and_loo(ctx)
    _report.write_loo_summary(ctx)
    _print_loo_row(ctx)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=_al_diag_vars(spec))

    section_header("Posterior predictive")
    _diag.sample_posterior_predictive(ctx, var_names=[obs_node])
    _save_ppc(ctx)
    _diag.save_trace(ctx)

    section_header("Factor summary")
    # Per-protocol design: NOTHING is a clean randomised effect, so no term is
    # flagged causal -- every coefficient (cohort included) is an association.
    fs = _report.factor_summary(
        ctx.trace, _al_coef_names(spec), ci_prob=ctx.reporting.hdi, causal_terms=()
    )
    fs.to_csv(os.path.join(ctx.output_dir, "factor_summary.csv"), index=False)
    ctx.tables["factor_summary"] = fs
    # Per-protocol: every term is an association, so the forest shows them all.
    _save_association_forest(ctx, _al_coef_names(spec), ())
    print_table(
        ranked_dataframe_table(
            fs,
            title=f"Factor summary ({spec.outcome_symbol}) - {int(ctx.reporting.hdi * 100)}% CrI",
            columns=["term", "role", "mean", "lo", "hi", "prob_positive"],
            rank_column=False,
            precision=3,
        )
    )

    meta_extra = {"loo_elpd": float(ctx.loo.elpd)}
    # Items-scale cohort contrast (immediate vs wait-list at aligned endpoints).
    # This is a PER-PROTOCOL association, NOT a randomised treatment effect --
    # confounded by age-at-onset and cohort/timing (see the LRPAL design note).
    if use_cohort:
        cohort = built.prepared.G.astype(float)
        n_marg = 1 if off_floor else built.prepared.n_trials[spec.outcome_symbol]
        cme = _report.treatment_marginal_effect(
            ctx.trace, trt=cohort, n_trials=n_marg, term="beta_cohort",
            ci_prob=ctx.reporting.hdi,
        )
        pd.DataFrame([cme]).to_csv(
            os.path.join(ctx.output_dir, "cohort_marginal.csv"), index=False
        )
        ctx.tables["cohort_marginal"] = pd.DataFrame([cme])
        meta_extra["cohort_marginal"] = cme
        print_table(
            metrics_table(
                [{"metric": k, "value": v} for k, v in cme.items()],
                title="Per-protocol cohort marginal (NOT randomised)",
                columns=["metric", "value"],
            )
        )

    _report.write_run_metadata(ctx, extra=meta_extra)
    section_header("Report")
    _copy_report_template(ctx)
    _print_footer(ctx)
    return ctx
