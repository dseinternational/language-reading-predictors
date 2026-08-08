# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Fit-context wrappers over the report figure modules.

The drawing lives in the focused modules — :mod:`effect_plots`,
:mod:`predicted_scores`, :mod:`arm_overlap`, :mod:`trajectory_plots`. This module
holds the fit-context side: pull the draws, labels and denominators off a
:class:`StatisticalFitContext`, hand them to the drawing module, register the
tables, and guard the whole thing so a plotting failure never costs an expensive
fit. Split out of ``pipeline.py`` for #394.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dse_research_utils.plot.styles import (
    COLOUR_BLUE,
    COLOUR_RED,
    FIGSIZE_LG,
)

from language_reading_predictors.statistical_models import (
    diagnostics as _diag,
    reporting as _report,
)
from language_reading_predictors.statistical_models.artifacts import (
    guard_optional,
    record_artifact,
)
from language_reading_predictors.statistical_models.context import (
    StatisticalFitContext,
)
from language_reading_predictors.statistical_models.plotting import (
    save_plotcollection,
    save_styled_figure,
)


def _draw_did_cell_panel(
    ax: "plt.Axes", cell_ppc: pd.DataFrame, *, stem: str, ylabel: str, title: str
) -> None:
    """One DiD cell-PPC panel: replicated median/interval vs observed, by cell."""
    x = np.arange(len(cell_ppc))
    labels = cell_ppc["cell"].str.replace("_", "\n").tolist()
    centre = cell_ppc[f"replicated_{stem}_median"].to_numpy(float)
    lo = cell_ppc[f"replicated_{stem}_lo"].to_numpy(float)
    hi = cell_ppc[f"replicated_{stem}_hi"].to_numpy(float)
    observed = cell_ppc[f"observed_{stem}"].to_numpy(float)
    ax.errorbar(
        x, centre, yerr=np.vstack((centre - lo, hi - centre)), fmt="o", capsize=4,
        color=COLOUR_BLUE, label="posterior predictive median and 95% interval",
    )
    ax.scatter(x, observed, marker="x", s=55, linewidth=2, color=COLOUR_RED,
               label="observed")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.2)
    ax.set_xticks(x, labels)
    ax.set_xlabel("fitted arm-by-time cell")
    ax.set_title(title)
    ax.legend(loc="best")


def save_did_cell_ppc_plot(ctx: StatisticalFitContext, cell_ppc: pd.DataFrame) -> None:
    """Cell-stratified DiD posterior-predictive checks as two individual figures:
    ``did_cell_ppc_mean`` (cell mean) and ``did_cell_ppc_zero_rate`` (proportion
    at zero)."""
    with guard_optional(
        ctx, "DiD cell PPC plot",
        filename="did_cell_ppc_mean.png", kind="figure", verb="failed",
    ):
        for stem, ylabel, name in (
            ("mean", "cell mean", "did_cell_ppc_mean"),
            ("zero_rate", "proportion at zero", "did_cell_ppc_zero_rate"),
        ):
            fig, ax = plt.subplots(figsize=FIGSIZE_LG)
            _draw_did_cell_panel(
                ax, cell_ppc, stem=stem, ylabel=ylabel,
                title=f"Cell-stratified PPC: {ylabel}",
            )
            fig.tight_layout()
            save_styled_figure(ctx.output_dir, name, fig=fig)


def save_proportion_at_zero_plot(
    ctx: StatisticalFitContext, symbol: str, ppc0: dict
) -> None:
    """Plot the proportion-at-zero PPC: replicated distribution vs observed."""
    with guard_optional(
        ctx, "Proportion-at-zero PPC plot",
        filename="proportion_at_zero_ppc.png", kind="figure", verb="failed",
    ):
        rep = ppc0["rep"]
        obs = ppc0["obs_prop_at_zero"]
        plt.figure(figsize=FIGSIZE_LG)
        plt.hist(rep, bins=30, color=COLOUR_BLUE, alpha=0.6, density=True)
        plt.axvline(obs, color=COLOUR_RED, lw=2, label=f"observed = {obs:.2f}")
        plt.xlabel(f"proportion of {symbol} post-scores at zero")
        plt.ylabel("posterior-predictive density")
        plt.title(
            f"Proportion-at-zero PPC ({symbol}); two-sided tail = "
            f"{ppc0['ppc_two_sided_tail']:.2f}"
        )
        plt.legend()
        # Scalar PPC summary (rep excluded) is already written to CSV by the
        # graded/floor path, so no data= here — just the styled PNG + SVG.
        save_styled_figure(ctx.output_dir, "proportion_at_zero_ppc")


def save_rope_plot(
    ctx: StatisticalFitContext,
    symbol: str,
    G: np.ndarray | None,
    n_trials: int,
    delta: float,
    *,
    term: str = "tau",
    varying_term: str = "tau_i",
    moderators: Sequence[tuple[str, np.ndarray]] | None = None,
    items: np.ndarray | None = None,
    row_mask: np.ndarray | None = None,
    split: bool = False,
    score_mean_link: str = "logit",
) -> None:
    """ROPE-anchored figure for a randomised effect: the items-scale posterior with
    the region of practical equivalence, and ``P(effect > delta)`` as the
    minimally-important difference rises. Single-outcome version of the note figure
    (notes/202606261304-evidence-strength-and-rope-reporting.md).

    The ITT/gain path recomputes the items draws from ``_itt_ame_draws`` (``term`` /
    ``varying_term`` / ``moderators`` / ``G`` select the effect, including any
    treatment interactions); the level family passes its t2 contrast items draws
    directly via ``items`` (its AME nets out a group×ability interaction the generic
    core cannot reconstruct). With ``split=True`` (the ITT reports) the two panels
    are written as individual files (``rope_summary`` + ``rope_benefit_curve``)
    rather than one combined figure.
    """
    with guard_optional(
        ctx, "ROPE plot",
        filename="rope_summary.png", kind="figure", verb="failed",
    ):
        from language_reading_predictors.statistical_models.effect_plots import (
            write_rope_figures,
        )

        if items is None:
            _, ame_prob = _report._itt_ame_draws(
                ctx.trace, G=G, term=term, varying_term=varying_term,
                moderators=moderators, row_mask=row_mask,
                score_mean_link=score_mean_link,
            )
            items = ame_prob * float(n_trials)
        write_rope_figures(
            ctx.output_dir, items, symbol=symbol, delta=delta,
            n_trials=n_trials, split=split,
        )


def write_predicted_scores(
    ctx: StatisticalFitContext,
    *,
    outcome_symbol: str,
    G: np.ndarray,
    n_trials: int,
    term: str,
    varying_term: str = "tau_i",
    moderators: Sequence[tuple[str, np.ndarray]] | None = None,
    row_mask: np.ndarray | None = None,
    likelihood: str = "beta_binomial",
    child_re: bool = False,
    child_idx: np.ndarray | None = None,
    delta: float | None = None,
    population: str,
    contrast_status: str,
    event_label: str = "off the floor at follow-up",
    split: bool = False,
    score_mean_link: str = "logit",
) -> None:
    """Predicted-scores contrast panel, ROPE-triple density and icon array (#316).

    Guarded like the other optional figure emitters: a plotting failure warns
    rather than killing an expensive fit. The plotted AME draws reuse the exact
    ``_itt_ame_draws`` arithmetic (guard-tested), so ``predicted_scores.csv``'s
    ``average_marginal_effect`` row matches ``treatment_marginal.csv`` /
    ``rope_summary.csv``.
    """
    from language_reading_predictors.statistical_models.measures import MEASURES
    from language_reading_predictors.statistical_models.predicted_scores import (
        write_predicted_scores_artifacts,
    )

    with guard_optional(
        ctx, "Predicted-scores figures",
        filename="predicted_scores.png", kind="figure", verb="failed",
    ):
        summary = write_predicted_scores_artifacts(
            ctx.output_dir,
            ctx.trace,
            outcome_symbol=outcome_symbol,
            item_label=MEASURES[outcome_symbol].label,
            G=np.asarray(G, dtype=float),
            n_trials=int(n_trials),
            term=term,
            varying_term=varying_term,
            moderators=moderators,
            row_mask=row_mask,
            likelihood=likelihood,
            child_effect_name="u_child" if child_re else None,
            child_sd_name="sigma_child" if child_re else None,
            child_idx=child_idx,
            delta=delta,
            ci_prob=ctx.reporting.ci_prob,
            population=population,
            contrast_status=contrast_status,
            event_label=event_label,
            random_seed=ctx.sampling.random_seed,
            split=split,
            score_mean_link=score_mean_link,
        )
        ctx.tables["predicted_scores"] = summary
        # ``predicted_scores.csv`` is written inside the helper (which takes an
        # output directory, not a context); record it so the manifest lists it.
        record_artifact(ctx, "predicted_scores", df=summary, required=False)


def write_arm_overlap(
    ctx: StatisticalFitContext,
    *,
    outcome_symbol: str,
    G: np.ndarray,
    n_trials: int,
    term: str,
    varying_term: str = "tau_i",
    moderators: Sequence[tuple[str, np.ndarray]] | None = None,
    row_mask: np.ndarray | None = None,
    likelihood: str = "beta_binomial",
    child_re: bool = False,
    child_idx: np.ndarray | None = None,
    population: str,
    contrast_status: str,
    event_label: str = "off the floor at follow-up",
    score_mean_link: str = "logit",
) -> None:
    """Intervention vs no-intervention posterior-overlap figures (two individual
    files: ``arm_overlap_mean`` and, for graded outcomes, ``arm_overlap_predictive``).

    Guarded like the other optional figure emitters. The contrast reuses the
    exact ``counterfactual_predictive_contrast`` machinery behind
    ``predicted_scores``, so the annotated average marginal effect matches
    ``rope_summary.csv`` and the predictive curves are drawn from the same
    simulated new-child scores.
    """
    from language_reading_predictors.statistical_models.arm_overlap import (
        write_arm_overlap_artifacts,
    )
    from language_reading_predictors.statistical_models.measures import MEASURES

    with guard_optional(
        ctx, "Arm-overlap figures",
        filename="arm_overlap_mean.png", kind="figure", verb="failed",
    ):
        tables = write_arm_overlap_artifacts(
            ctx.output_dir,
            ctx.trace,
            outcome_symbol=outcome_symbol,
            item_label=MEASURES[outcome_symbol].label,
            G=np.asarray(G, dtype=float),
            n_trials=int(n_trials),
            term=term,
            varying_term=varying_term,
            moderators=moderators,
            row_mask=row_mask,
            likelihood=likelihood,
            child_effect_name="u_child" if child_re else None,
            child_sd_name="sigma_child" if child_re else None,
            child_idx=child_idx,
            ci_prob=ctx.reporting.ci_prob,
            population=population,
            contrast_status=contrast_status,
            event_label=event_label,
            random_seed=ctx.sampling.random_seed,
            score_mean_link=score_mean_link,
        )
        for name, table in tables.items():
            ctx.tables[name] = table


def _ctx_pareto_k(ctx: StatisticalFitContext) -> np.ndarray | None:
    """Per-observation Pareto-k vector from ``ctx.loo`` (``None`` when unavailable)."""
    loo = getattr(ctx, "loo", None)
    pk = getattr(loo, "pareto_k", None) if loo is not None else None
    if pk is None:
        return None
    return np.asarray(getattr(pk, "values", pk), dtype=float)


def write_group_trajectory(
    ctx: StatisticalFitContext,
    *,
    outcome_symbol: str,
    arm: np.ndarray,
    wave: np.ndarray,
    child_idx: np.ndarray,
    off_floor: bool,
    obs_node: str = "y_post",
    crossover_wave: int = 1,
) -> None:
    """Population per-arm score-trajectory figure (#317 fig 1). Guarded like the PPC."""
    from language_reading_predictors.statistical_models import trajectory_plots as _tp
    from language_reading_predictors.statistical_models.measures import MEASURES

    with guard_optional(
        ctx, "Group-trajectory figure",
        filename="group_trajectory.png", kind="figure", verb="failed",
    ):
        m = MEASURES[outcome_symbol]
        summary = _tp.write_group_arm_trajectory(
            ctx.output_dir,
            ctx.trace,
            arm=np.asarray(arm, dtype=int),
            wave=np.asarray(wave, dtype=int),
            child_idx=np.asarray(child_idx, dtype=int),
            n_trials=int(m.n_trials),
            outcome_symbol=outcome_symbol,
            item_label=m.label,
            off_floor=off_floor,
            ci_prob=ctx.reporting.ci_prob,
            crossover_wave=crossover_wave,
            obs_node=obs_node,
        )
        ctx.tables["group_trajectory"] = summary


def write_child_fit(
    ctx: StatisticalFitContext,
    *,
    outcome_symbol: str,
    wave: np.ndarray,
    child_idx: np.ndarray,
    off_floor: bool,
    obs_node: str = "y_post",
    x_label: str = "assessment wave",
) -> None:
    """Per-child fitted-vs-observed small multiples for an obs_id family (#317 fig 2)."""
    from language_reading_predictors.statistical_models import trajectory_plots as _tp
    from language_reading_predictors.statistical_models.measures import MEASURES

    with guard_optional(
        ctx, "Per-child fit figure",
        filename="child_fit_panels.png", kind="figure", verb="failed",
    ):
        m = MEASURES[outcome_symbol]
        summary = _tp.write_child_fit_obsid(
            ctx.output_dir,
            ctx.trace,
            wave=np.asarray(wave, dtype=int),
            child_idx=np.asarray(child_idx, dtype=int),
            n_trials=int(m.n_trials),
            outcome_symbol=outcome_symbol,
            item_label=m.label,
            off_floor=off_floor,
            obs_node=obs_node,
            pareto_k=_ctx_pareto_k(ctx),
            seed=ctx.sampling.random_seed,
            ci_prob=ctx.reporting.ci_prob,
            x_label=x_label,
        )
        ctx.tables["child_fit_panels"] = summary


def write_panel_trajectory(ctx: StatisticalFitContext, *, latent_name: str) -> None:
    """Per-measure cohort growth-trajectory figure for a masked panel family (#317)."""
    from language_reading_predictors.statistical_models import trajectory_plots as _tp

    with guard_optional(
        ctx, "Cohort-trajectory figure",
        filename="group_trajectory.png", kind="figure", verb="failed",
    ):
        summary = _tp.write_outcome_trajectory(
            ctx.output_dir,
            ctx.trace,
            ctx.prepared,
            latent_name=latent_name,
            ci_prob=ctx.reporting.ci_prob,
        )
        ctx.tables["group_trajectory"] = summary


def write_panel_child_fit(
    ctx: StatisticalFitContext,
    *,
    latent_name: str,
    focal_symbol: str,
    kappa_name: str = "kappa",
) -> None:
    """Per-child small multiples (one focal outcome) for a masked panel family (#317)."""
    from language_reading_predictors.statistical_models import trajectory_plots as _tp

    with guard_optional(
        ctx, "Per-child fit figure",
        filename="child_fit_panels.png", kind="figure", verb="failed",
    ):
        summary = _tp.write_child_fit_panel(
            ctx.output_dir,
            ctx.trace,
            ctx.prepared,
            latent_name=latent_name,
            focal_symbol=focal_symbol,
            kappa_name=kappa_name,
            pareto_k=_ctx_pareto_k(ctx),
            seed=ctx.sampling.random_seed,
            ci_prob=ctx.reporting.ci_prob,
        )
        ctx.tables["child_fit_panels"] = summary


def save_contrast_heatmap(ctx: StatisticalFitContext, contrast) -> None:
    """Heatmap of joint pairwise probability-scale AME ordering (#125 Area 4)."""
    with guard_optional(
        ctx, "Contrast heatmap",
        filename="contrast_heatmap.png", kind="figure", verb="failed",
    ):
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
        ax.set_title("P(row AME > column AME)", fontsize=9)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("P(row AME > column AME)", fontsize=8)
        # ``save_styled_figure`` owns the layout engine. Switching engines after
        # a colorbar has been created raises on recent Matplotlib versions.
        save_styled_figure(ctx.output_dir, "contrast_heatmap", fig=fig)


def save_forest_plot(
    ctx: StatisticalFitContext,
    var_names: list[str],
    *,
    name: str = "tau_forest.png",
    title: str | None = None,
) -> None:
    """Forest plot of the causal term(s) with a reference line at 0 (#125 Area 4).

    For a single-outcome model ``var_names=["tau"]`` shows the one effect; for the
    joint model the vector ``tau`` forests every outcome's effect in one panel —
    the single most communicative artifact for the suite. Guarded.
    """
    with guard_optional(
        ctx, f"Forest plot ({name})", filename=name, kind="figure", verb="failed"
    ):
        import arviz_plots as azp

        tr = _diag.thin_for_plots(ctx.trace)
        # Equal-tailed nested bands (#177): inner central 50% + outer equal-tailed
        # headline (89% house standard, from ctx.reporting.ci_prob), matching the
        # reported interval convention rather than the arviz default (which can be
        # an HDI, inconsistent with the prose).
        pc = azp.plot_forest(
            tr,
            var_names=var_names,
            combined=True,
            ci_kind=ctx.reporting.interval_kind,
            ci_probs=(0.5, ctx.reporting.ci_prob),
        )
        try:
            azp.add_lines(pc, values=0)
        except Exception:
            pass  # the forest itself is the substantive output
        if title is None:
            title = (
                "Adjusted-association coefficients (forest)"
                if "association" in name
                else "Effect posterior (forest, reference line at 0)"
            )
        save_plotcollection(pc, ctx.output_dir, name, suptitle=title)


def save_association_forest(
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
    clean. Guarded via :func:`save_forest_plot`.
    """
    assoc = [
        c
        for c in coef_names
        if c in ctx.trace.posterior
        and not any(ct == c or ct.startswith(c + "[") for ct in causal_terms)
    ]
    if assoc:
        save_forest_plot(ctx, assoc, name="association_forest.png")
