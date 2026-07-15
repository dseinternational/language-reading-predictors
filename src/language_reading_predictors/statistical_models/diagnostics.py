# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared sampling / diagnostics helpers for the statistical models.

Each LRP model runs the same diagnostic suite:

1. Prior predictive check (1000 draws). The prior + prior-predictive groups are
   persisted onto ``trace.nc`` (issue #125 step 0b) so the report can show
   prior-predictive checks and prior-vs-posterior overlays without recomputation.
2. Sampling via NUTS (nutpie backend).
3. Summary diagnostics (R-hat, ESS over the scalar parameters; a separate
   summary for deterministics / HSGP basis weights), trace / energy / posterior
   plots, and a ``diagnostics_summary.json`` pass/fail convergence verdict
   (divergences, BFMI, R-hat, ESS) that the report renders *first*.
4. LOO-PSIS via ArviZ (pointwise, so Pareto-k bands are available) and a
   ``log_prior`` group for power-scaling prior sensitivity.
5. Posterior predictive draws, plus the extended diagnostics (Pareto-k, rank,
   ESS-evolution, LOO-PIT).

Everything is written to ``context.output_dir`` and the trace persisted as
``trace.nc`` (NetCDF, an ``xarray`` DataTree).

ArviZ note: this is the ArviZ 1.x split stack (``arviz`` / ``arviz_plots`` /
``arviz_stats``). Legacy ``az.plot_ppc`` / ``plot_posterior`` do not exist; plots
go through ``arviz_plots`` and return a ``PlotCollection`` saved via ``.savefig``.
Every plot / extra-diagnostic call is guarded so a backend or API hiccup degrades
to a warning rather than aborting the fit — the numeric summaries are the
substantive output.
"""

from __future__ import annotations

import os
import re
import tempfile

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import xarray as xr
from rich import print as rprint

from dse_research_utils.statistics.diagnostics import (
    BFMI_THRESHOLD,
    ESS_THRESHOLD,
    RHAT_MAX,
    _bfmi_per_chain,
)
from dse_research_utils.statistics.diagnostics import (
    write_diagnostics_summary as _shared_write_diagnostics_summary,
)

from language_reading_predictors.models._reporting import (
    print_table,
    ranked_dataframe_table,
)
from language_reading_predictors.statistical_models.context import (
    StatisticalFitContext,
)
from language_reading_predictors.statistical_models.plotting import (
    save_plotcollection,
    save_styled_figure,
)

# Convergence-gate thresholds (issue #125 Area 3; Vehtari et al. 2021 for R-hat)
# and the per-chain BFMI helper are now owned by the shared package and
# re-exported here so existing call sites and tests keep their import paths.
__all__ = ["RHAT_MAX", "ESS_THRESHOLD", "BFMI_THRESHOLD", "_bfmi_per_chain"]


def run_prior_predictive(
    context: StatisticalFitContext,
    draws: int = 1000,
    var_names: list[str] | None = None,
) -> None:
    """Draw from the prior + prior predictive into ``context.prior_samples``.

    When ``var_names`` is ``None`` (the default for every family now), the prior
    is sampled over *all* free RVs, deterministics, and observed nodes of the
    model. This makes the persisted ``prior`` group rich enough for prior-vs-
    posterior overlays and the prior pushforward (it carries the effect term and
    ``eta``), and the ``prior_predictive`` group carries the outcome node — at no
    extra cost beyond the draws already taken. Falls back to a minimal
    observed + ``eta`` set if the full draw fails.
    """
    model = context.model
    if var_names is None:
        names: list[str] = []
        names += [rv.name for rv in model.free_RVs]
        names += [d.name for d in model.deterministics]
        names += [rv.name for rv in model.observed_RVs]
        var_names = list(dict.fromkeys(names))  # de-dupe, preserve order
    try:
        with model:
            prior = pm.sample_prior_predictive(
                draws=draws,
                var_names=var_names,
                random_seed=context.sampling.random_seed,
            )
    except Exception as exc:  # pragma: no cover - defensive fallback
        rprint(f"[yellow]Full prior-predictive draw failed ({exc}); retrying minimal set[/yellow]")
        fallback = [rv.name for rv in model.observed_RVs]
        if any(d.name == "eta" for d in model.deterministics):
            fallback.append("eta")
        with model:
            prior = pm.sample_prior_predictive(
                draws=draws,
                var_names=fallback,
                random_seed=context.sampling.random_seed,
            )
    context.prior_samples = prior


def sample_posterior(context: StatisticalFitContext) -> None:
    s = context.sampling
    with context.model:
        trace = pm.sample(
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
    context.trace = trace


def _joint_log_likelihood_by_child(trace: xr.DataTree) -> xr.DataArray | None:
    """Aggregate a flattened multi-outcome log likelihood to the child unit.

    The joint ITT likelihood stores one cell per observed child-outcome pair.
    Leaving cells out would condition prediction of one outcome on the same
    child's remaining outcomes and would not answer leave-one-child-out
    generalisation. New joint traces therefore persist ``y_post_cell_row`` and
    this helper sums each draw's cell log likelihood within child.
    """
    constant = getattr(trace, "constant_data", None)
    log_likelihood = getattr(trace, "log_likelihood", None)
    if (
        constant is None
        or log_likelihood is None
        or "y_post_cell_row" not in constant
        or "y_post" not in log_likelihood
    ):
        return None
    ll = log_likelihood["y_post"]
    cell_dims = [d for d in ll.dims if d not in {"chain", "draw"}]
    if len(cell_dims) != 1:
        raise ValueError(
            "flattened joint y_post log likelihood must have one cell dimension"
        )
    cell_dim = cell_dims[0]
    rows = np.asarray(constant["y_post_cell_row"].values, dtype=int).ravel()
    if rows.size != ll.sizes[cell_dim]:
        raise ValueError("joint cell-row map does not align with y_post log likelihood")
    n_children = (
        int(constant["G"].size)
        if "G" in constant
        else (int(rows.max()) + 1 if rows.size else 0)
    )
    if rows.size and (rows.min() < 0 or rows.max() >= n_children):
        raise ValueError("joint cell-row map contains an out-of-range child index")
    ordered = ll.transpose("chain", "draw", cell_dim)
    aggregated = np.zeros(
        (ordered.sizes["chain"], ordered.sizes["draw"], n_children), dtype=float
    )
    values = np.asarray(ordered.values, dtype=float)
    for child in range(n_children):
        aggregated[..., child] = values[..., rows == child].sum(axis=-1)
    return xr.DataArray(
        aggregated,
        dims=("chain", "draw", "obs_id"),
        coords={
            "chain": ordered.coords["chain"],
            "draw": ordered.coords["draw"],
            "obs_id": np.arange(n_children),
        },
        name="y_post_child",
        attrs={"loo_unit": "child", "aggregation": "sum over observed outcomes"},
    )


def compute_log_likelihood_and_loo(context: StatisticalFitContext) -> None:
    """Add log-likelihood + log-prior groups and compute pointwise LOO.

    The LOO is computed ``pointwise=True`` so the per-observation Pareto-k
    diagnostics survive on ``context.loo`` for the report's Pareto-k bands
    (load-bearing at n ≈ 33–54, where one influential child can drive elpd). The
    ``log_prior`` group is added (guarded) so power-scaling prior sensitivity
    (``arviz_stats.psense``) is reachable from the persisted trace.
    """
    with context.model:
        context.trace = pm.compute_log_likelihood(context.trace)
        try:
            from pymc.stats import compute_log_prior

            context.trace = compute_log_prior(context.trace)
        except Exception as exc:  # pragma: no cover - psense is secondary
            rprint(f"[yellow]log_prior group skipped: {exc}[/yellow]")
    child_ll = _joint_log_likelihood_by_child(context.trace)
    if child_ll is not None:
        context.trace.log_likelihood["y_post_child"] = child_ll
        context.loo = az.loo(
            context.trace, var_name="y_post_child", pointwise=True
        )
    else:
        context.loo = az.loo(context.trace, pointwise=True)


def _interval_cols(columns) -> list[str]:
    """Return the credible-interval column names in an ``az.summary`` frame.

    ArviZ 1.x names equal-tailed columns ``eti95_lb`` / ``eti95_ub`` (and HDI
    ``hdi_3%`` style), so match the ``eti``/``hdi`` prefix with an optional
    coverage number before the separator.
    """
    return [c for c in columns if re.match(r"^(eti|hdi)\d*_", str(c))]


def summary_diagnostics(
    context: StatisticalFitContext,
    var_names: list[str] | None = None,
    max_vars_for_pairs: int = 8,
) -> None:
    out = context.output_dir
    os.makedirs(out, exist_ok=True)

    # Narrow to scalar RVs by default so the summary table is readable.
    if var_names is None:
        scalar_vars = []
        for rv in context.model.unobserved_RVs:
            try:
                if int(np.prod(rv.shape.eval())) <= 2:
                    scalar_vars.append(rv.name)
            except Exception:
                continue
        var_names = scalar_vars

    if var_names:
        # Central interval driven by the shared reporting config (issue #125 0c /
        # #101): the report cards and prose use equal-tailed quantiles
        # (``interval_kind="eti"``), so the diagnostics table follows the same
        # convention via ``context.reporting.interval_kind`` rather than hard-coding
        # it — keeping the table, plots and config in step.
        summary = az.summary(
            context.trace,
            var_names=var_names,
            round_to=3,
            ci_prob=context.reporting.ci_prob,
            ci_kind=context.reporting.interval_kind,
        )
        summary.to_csv(os.path.join(out, "diagnostics.csv"))
        context.tables["diagnostics"] = summary

        ci_pct = int(round(context.reporting.ci_prob * 100))
        interval_cols = _interval_cols(summary.columns)
        wanted = [
            c
            for c in ["mean", "sd", *interval_cols, "ess_bulk", "ess_tail", "r_hat"]
            if c in summary.columns
        ]
        display_df = summary.reset_index().rename(columns={"index": "variable"})
        print_table(
            ranked_dataframe_table(
                display_df,
                title=f"Posterior diagnostics (equal-tailed {ci_pct}%)",
                columns=["variable", *wanted],
                rank_column=False,
                precision=3,
            )
        )

    # Deterministics / HSGP basis-weight summary (issue #125 Area 3): the
    # ``var_names is None`` autodetect above keeps only scalars, so GP-bearing
    # variants never get their basis weights summarised. Emit those to a separate
    # CSV so the scalar table stays readable. Guarded and best-effort.
    _summarise_deterministics(context, scalar_var_names=var_names)

    # Diagnostic plots. arviz 1.x routes plotting through arviz_plots, whose
    # functions return a PlotCollection saved via ``.savefig`` (not the old
    # matplotlib ``plt.savefig``) and drop several 0.x kwargs (``combined``,
    # ``kind``, ``divergences``). Each plot is guarded so a backend/API hiccup
    # degrades to a warning rather than failing the fit — the numeric summary
    # above is the substantive output.
    import arviz_plots as azp

    # Plots use a draw-thinned view (visually identical, but the full reporting
    # trace can be very slow / hang for these routines); diagnostics.csv above
    # used the full trace.
    tr = thin_for_plots(context.trace)

    _save_pc(
        out,
        lambda: azp.plot_trace(tr, var_names=var_names or None),
        "trace_plot.png",
        title="MCMC trace by chain",
    )
    _save_pc(
        out,
        lambda: azp.plot_energy(tr),
        "energy_plot.png",
        title="Energy transitions (NUTS diagnostic)",
    )

    if var_names:
        _save_pc(
            out,
            lambda: azp.plot_dist(
                tr,
                var_names=var_names,
                group="posterior",
                ci_prob=context.reporting.ci_prob,
            ),
            "posterior_plot.png",
            title="Marginal posterior distributions",
        )
        if len(var_names) <= max_vars_for_pairs:
            _save_pc(
                out,
                lambda: azp.plot_pair(tr, var_names=var_names),
                "pair_plot.png",
                title="Posterior pairwise joint distributions",
            )


def _save_pc(out: str, make, name: str, title: str | None = None) -> None:
    """Build a PlotCollection and save it, degrading to a warning on failure.

    Routes through :func:`save_plotcollection` so every ArviZ figure gets the
    house style, a figure-level title (they render untitled otherwise), and an
    SVG sibling alongside the referenced PNG (issue #208).
    """
    try:
        pc = make()
        save_plotcollection(pc, out, name, suptitle=title)
    except Exception as exc:  # pragma: no cover
        rprint(f"[yellow]{name} skipped: {exc}[/yellow]")


def thin_for_plots(trace, max_draws: int = 1000):
    """Return a draw-thinned view of the trace for plotting at scale.

    Several ``arviz_plots`` routines — ``plot_rank`` in particular — are
    pathologically slow (effectively hang) on a reporting-config trace
    (6000 draws × 6 chains = 36k draws), while running fine at dev scale (~1k).
    The numeric summaries always use the full trace; the diagnostic *plots* are
    visually identical on a thinned view, so thin the draw dimension so that
    chain × draw ≲ ``max_draws`` before plotting. Guarded — returns the original
    trace if anything about the structure is unexpected.
    """
    try:
        post = trace.posterior
        total = int(post.sizes.get("chain", 1)) * int(post.sizes.get("draw", 1))
        if total <= max_draws:
            return trace
        k = int(np.ceil(total / max_draws))
        return trace.isel(draw=slice(None, None, k))
    except Exception:  # pragma: no cover
        return trace


def thin_posterior_only(trace, max_draws: int = 1000):
    """Thin *only* the posterior group; leave the other groups (notably the small
    1-chain prior) at full resolution.

    :func:`thin_for_plots` applies its stride to the whole DataTree, so at
    reporting scale (posterior 6×6000 → stride 36) it also decimates the 1×1000
    ``prior`` group to ~28 jagged draws — which then misrepresents the
    prior-vs-posterior overlay ("how far the data moved each parameter from its
    prior"). Only the posterior is large enough to need thinning for plotting, so
    thin that alone and keep every other group intact (issue #270 item 1).
    """
    try:
        post = trace.posterior.to_dataset()
        total = int(post.sizes.get("chain", 1)) * int(post.sizes.get("draw", 1))
        if total <= max_draws:
            return trace
        k = int(np.ceil(total / max_draws))
        groups = {}
        for name in trace.children:
            ds = trace[name].to_dataset()
            if name == "posterior":
                ds = ds.isel(draw=slice(None, None, k))
            groups[name] = ds
        return type(trace).from_dict(groups)
    except Exception:  # pragma: no cover - defensive
        # Return the trace UNCHANGED rather than falling back to thin_for_plots:
        # that would thin the whole tree (including the small prior group) and
        # reintroduce the exact bug this helper exists to avoid (issue #270 review).
        # An un-thinned prior-overlay is slower but correct.
        return trace


def _summarise_deterministics(
    context: StatisticalFitContext, scalar_var_names: list[str] | None
) -> None:
    """Write ``diagnostics_deterministics.csv`` for vector / GP-weight nodes."""
    try:
        scalar = set(scalar_var_names or [])
        det_names = [d.name for d in context.model.deterministics if d.name not in scalar]
        # Only summarise nodes actually present in the posterior, and skip the
        # per-observation ``eta`` (n_obs rows) which would dominate the table.
        present = [
            n
            for n in det_names
            if n in context.trace.posterior and n not in ("eta",)
        ]
        if not present:
            return
        summary = az.summary(
            context.trace, var_names=present, round_to=3, ci_kind=context.reporting.interval_kind
        )
        summary.to_csv(os.path.join(context.output_dir, "diagnostics_deterministics.csv"))
        context.tables["diagnostics_deterministics"] = summary
    except Exception as exc:  # pragma: no cover
        rprint(f"[yellow]deterministics summary skipped: {exc}[/yellow]")


def _gate_var_names(
    context: StatisticalFitContext, curated: list[str] | None
) -> list[str] | None:
    """Full-coverage variable set for the convergence gate (issue #274 item 2).

    The per-family ``var_names`` lists passed by the pipeline are hand-curated
    *headline scalars* — right for the human-readable ``diagnostics.csv`` and the
    prior-overlay, but they silently omit the parameters where hierarchical models
    at n ~ tens actually fail: the non-centred per-child intercept vector
    (``u_child_raw``), the HSGP amplitude / lengthscale / basis-weight RVs, and the
    joint model's LKJ block. So the gate scanned only the scalars it already
    trusted.

    Gate R-hat / ESS over the model's **free RVs** instead — which include exactly
    those, and *exclude* the per-observation deterministics (``eta`` / ``theta`` /
    ``f_mech``) that ``var_names=None`` would drag in and that would bloat and slow
    the scan — unioned with the curated headline terms so the causal
    *deterministics* (``tau``, ``delta``, the AMEs) stay covered as well. Names are
    filtered to those actually present in the posterior so a headline term a given
    fit does not instantiate cannot make ``az.summary`` raise. Falls back to the
    curated list if the model is unavailable.
    """
    if context.model is None:
        return curated
    try:
        free = [rv.name for rv in context.model.free_RVs]
    except Exception:  # pragma: no cover - defensive
        return curated
    names = list(dict.fromkeys([*free, *(curated or [])]))  # de-dup, preserve order
    try:
        post = context.trace.posterior
        names = [n for n in names if n in post]
    except Exception:  # pragma: no cover - defensive
        pass
    return names or None


def write_diagnostics_summary(
    context: StatisticalFitContext,
    *,
    var_names: list[str] | None = None,
) -> dict:
    """Emit ``diagnostics_summary.json`` — the report's pass/fail convergence gate.

    Thin wrapper over :func:`dse_research_utils.statistics.diagnostics.write_diagnostics_summary`
    so the convergence gate (and its JSON schema) is defined once across DSE
    projects. The shared implementation (>= v0.7.0) evaluates the gate on
    *unrounded* R-hat / ESS — ``round_to="none"``, the string; ``round_to=None``
    would fall through to ``rcParams["stats.round_to"]`` (2 sig figs) so a
    borderline 1.01004 would round to 1.0100 and slip through the ``<= 1.01`` gate
    (dseinternational/research#65) — and treats a non-finite per-chain BFMI as a
    failure rather than letting it pass order-dependently. Written unconditionally
    for every family (incl. mediation, which has no LOO) so the report's banner
    always renders.

    The R-hat / ESS scan runs over :func:`_gate_var_names` (the model's free RVs +
    the curated headline terms), **not** the ``var_names`` alone, so the per-child
    random-intercept vector and the GP / LKJ hyperparameters are gated (issue
    #274 item 2). The curated ``var_names`` still drive the human-readable
    ``diagnostics.csv`` (via :func:`summary_diagnostics`) and the prior-overlay.
    """
    return _shared_write_diagnostics_summary(
        context.trace,
        context.output_dir,
        var_names=_gate_var_names(context, var_names),
        tables=context.tables,
    )


def subfit_convergence(trace, *, label: str, var_names: list[str] | None = None) -> dict:
    """Lightweight convergence check for a *sub-fit* trace (issue: ungated sub-fits).

    The headline gate (:func:`write_diagnostics_summary` → ``diagnostics_summary.json``)
    only covers the primary trace. Secondary / sensitivity / bivariate sub-fits (the
    floor-rule graded secondary, the t3 temporal-ordering sensitivity, the adjusted
    family's bivariate + prior-sweep + SES refits) publish CSVs from their own
    standalone traces with no gate — a silently non-converged sub-fit would be
    reported without any flag. This computes the same signals as the main gate
    (unrounded max R-hat, min bulk/tail ESS, total divergences and minimum per-chain
    BFMI) and returns a small dict whose ``converged`` value is ``True`` when the gate
    passes, ``False`` when it fails, and ``None`` when the diagnostic calculation
    itself cannot be completed. It is a *flag*, not a hard stop: sensitivity sub-fits
    should still be reported, but failed or unchecked fits must be marked.
    """
    result = {
        "converged": None,
        "max_rhat": None,
        "min_ess": None,
        "min_bfmi": None,
        "n_divergences": None,
    }
    try:
        # ``round_to="none"`` (the string) genuinely disables rounding; ``round_to=None``
        # falls through to ``rcParams["stats.round_to"]`` (2 sig figs) and would silently
        # gate on rounded R-hat/ESS — the dseinternational/research#65 bug this check must
        # not reproduce (it advertises "unrounded" signals above).
        summ = az.summary(
            trace, var_names=var_names, round_to="none", kind="diagnostics"
        )
        max_rhat = float(summ["r_hat"].max())
        min_ess = float(min(summ["ess_bulk"].min(), summ["ess_tail"].min()))
        n_div = int(np.asarray(trace.sample_stats["diverging"].values).sum())
        bfmi = _bfmi_per_chain(trace)
        min_bfmi = (
            float(np.min(bfmi))
            if bfmi is not None and np.all(np.isfinite(bfmi))
            else None
        )
        result.update(
            max_rhat=max_rhat,
            min_ess=min_ess,
            min_bfmi=min_bfmi,
            n_divergences=n_div,
        )
        result["converged"] = bool(
            max_rhat <= RHAT_MAX
            and min_ess >= ESS_THRESHOLD
            and min_bfmi is not None
            and min_bfmi >= BFMI_THRESHOLD
            and n_div == 0
        )
    except Exception as exc:  # pragma: no cover
        rprint(f"[yellow]sub-fit convergence check failed for {label}: {exc}[/yellow]")
        result["converged"] = None
        return result
    if result["converged"] is False:
        rprint(
            f"[red]Sub-fit '{label}' did not meet the convergence gate "
            f"(max R-hat={result['max_rhat']:.4f}, min ESS={result['min_ess']:.0f}, "
            f"min BFMI={result['min_bfmi'] if result['min_bfmi'] is not None else 'missing'}, "
            f"divergences={result['n_divergences']}); its published estimates are "
            "flagged not-converged.[/red]"
        )
    return result


def run_extended_diagnostics(
    context: StatisticalFitContext,
    *,
    causal_term: str | None = None,
    include_loo_pit: bool = True,
) -> None:
    """Pareto-k, rank, ESS-evolution and LOO-PIT plots (issue #125 Area 3).

    Called after posterior-predictive sampling so all groups are present. Pareto-k
    reuses ``context.loo`` (computed ``pointwise=True``); rank focuses on the
    causal term; LOO-PIT needs the posterior-predictive group. All guarded.
    """
    out = context.output_dir
    import arviz_plots as azp

    # Pareto-k reads context.loo (per-observation, not draws) — full trace, fast.
    if context.loo is not None:
        _save_pc(
            out,
            lambda: azp.plot_khat(context.loo),
            "pareto_k.png",
            title="Pareto-k (LOO influence; flag k > 0.7)",
        )

    # Draw-based plots use a thinned view (full trace hangs plot_rank at reporting
    # scale; thinning is visually identical and reproduces the fast dev path).
    tr = thin_for_plots(context.trace)

    if causal_term is not None and causal_term in context.trace.posterior:
        _save_pc(
            out,
            lambda: azp.plot_rank(tr, var_names=[causal_term]),
            "rank_plot.png",
            title="Rank plot (chain mixing)",
        )

    # ESS evolution must use the *full* trace: thinning caps the plotted ESS near
    # min(n_thinned_draws, true ESS), so a 36k-draw fit would show ESS pinned well
    # under the 400 reference line and contradict the "ESS climbs above 400"
    # guidance (issue #270 item 1). Only the pathologically-slow plot_rank needs
    # the thinned view.
    _save_pc(
        out,
        lambda: azp.plot_ess_evolution(
            context.trace,
            var_names=[causal_term] if causal_term else None,
            min_ess=ESS_THRESHOLD,
        ),
        "ess_evolution.png",
        title="Effective sample size evolution",
    )

    try:
        if include_loo_pit and "posterior_predictive" in context.trace.children:
            _save_pc(
                out,
                lambda: azp.plot_loo_pit(tr),
                "loo_pit.png",
                title="LOO-PIT calibration",
            )
    except Exception as exc:  # pragma: no cover
        rprint(f"[yellow]loo_pit skipped: {exc}[/yellow]")


def save_prior_posterior_plot(
    context: StatisticalFitContext,
    *,
    var_names: list[str] | None = None,
) -> None:
    """Prior-vs-posterior overlay for the key parameters (issue #125 Area 1).

    Needs the ``prior`` group on the trace (attached by :func:`save_trace`), so
    call this *after* ``save_trace``. Shows how far the data moved each parameter
    from its prior. Guarded.
    """
    out = context.output_dir
    import arviz_plots as azp

    # Thin only the posterior: thinning the whole tree would decimate the small
    # 1-chain prior group and misrepresent the overlay (issue #270 item 1).
    tr = thin_posterior_only(context.trace)

    # ``plot_prior_posterior`` expands every non-sampling coordinate into a
    # separate panel.  ArviZ's default 40-panel guard therefore rejects the
    # full joint ITT overlay (five explicitly curated parameter arrays x ten
    # outcomes = 50 panels) before a figure is created.  Raise the guard only
    # as far as this caller's explicit selection requires, and only for the
    # duration of this plot; an unconstrained ``var_names=None`` call retains
    # the configured safety limit.
    panel_count = 0
    if var_names is not None:
        try:
            posterior = tr.posterior
            for name in var_names:
                if name not in posterior:
                    continue
                non_sample_sizes = [
                    size
                    for dim, size in posterior[name].sizes.items()
                    if dim not in {"chain", "draw"}
                ]
                panel_count += int(np.prod(non_sample_sizes, dtype=int))
        except Exception:  # pragma: no cover - plotting remains guarded below
            panel_count = 0

    configured_limit = az.rcParams["plot.max_subplots"]
    if (
        panel_count
        and configured_limit is not None
        and panel_count > configured_limit
    ):
        rc = {"plot.max_subplots": panel_count}
    else:
        rc = {}

    # More than 40 marginal panels need explicit geometry as well as a raised
    # safety limit. The default four-column auto-layout is too short for a
    # 50-panel joint overlay: row titles collide with the preceding row's tick
    # labels. Five columns with roughly 4.4 x 3.4 inches per panel preserves
    # readable labels while retaining one complete, lightbox-enabled figure.
    plot_kwargs: dict[str, object] = {}
    if panel_count > 40:
        col_wrap = 5
        n_rows = (panel_count + col_wrap - 1) // col_wrap
        plot_kwargs = {
            "col_wrap": col_wrap,
            "figure_kwargs": {
                "figsize": (4.4 * col_wrap, 3.4 * n_rows),
                "gridspec_kw": {"hspace": 0.85, "wspace": 0.25},
            },
        }
    with az.rc_context(rc):
        _save_pc(
            out,
            lambda: azp.plot_prior_posterior(
                tr, var_names=var_names, **plot_kwargs
            ),
            "prior_posterior.png",
            title="Prior vs posterior overlay",
        )


def run_psense(
    context: StatisticalFitContext,
    *,
    var_names: list[str],
) -> None:
    """Power-scaling prior/likelihood sensitivity (issue #125 Area 1, secondary).

    Writes ``psense_summary.csv`` and ``psense.png`` for the named parameters
    (usually the causal term). Requires the ``log_prior`` and ``log_likelihood``
    groups added by :func:`compute_log_likelihood_and_loo`. Guarded — a missing
    group or an API mismatch degrades to a warning (psense is recommended-but-
    secondary at this n). Kallioinen et al. 2024.
    """
    out = context.output_dir
    summary_path = os.path.join(out, "psense_summary.csv")
    context.tables.pop("psense_summary", None)
    try:
        os.unlink(summary_path)
    except FileNotFoundError:
        pass
    temporary_path: str | None = None
    try:
        import arviz_stats as azs

        # The published psense_summary.csv is a numeric diagnostic — compute it on
        # the FULL trace (thin_for_plots' own contract: "numeric summaries always
        # use the full trace"); the thinned view is only for the figure below
        # (issue #270 item 2).
        s = azs.psense_summary(context.trace, var_names=var_names)
        if hasattr(s, "to_dataframe"):
            df = s.to_dataframe()
        else:
            import pandas as pd

            df = pd.DataFrame(s)
        descriptor, temporary_path = tempfile.mkstemp(
            dir=out,
            prefix=".psense_summary-",
            suffix=".tmp",
        )
        os.close(descriptor)
        df.to_csv(temporary_path)
        os.replace(temporary_path, summary_path)
        temporary_path = None
        context.tables["psense_summary"] = df
    except Exception as exc:  # pragma: no cover
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
        rprint(f"[yellow]psense_summary skipped: {exc}[/yellow]")

    import arviz_plots as azp

    tr = thin_for_plots(context.trace)
    _save_pc(
        out,
        lambda: azp.plot_psense_dist(tr, var_names=var_names),
        "psense.png",
        title="Prior/likelihood power-scaling sensitivity",
    )


def sample_posterior_predictive(
    context: StatisticalFitContext,
    var_names: list[str] | None = None,
) -> None:
    with context.model:
        context.trace = pm.sample_posterior_predictive(
            context.trace,
            var_names=var_names,
            extend_inferencedata=True,
            random_seed=context.sampling.random_seed,
            progressbar=False,
        )


def _attach_prior_groups(context: StatisticalFitContext) -> None:
    """Graft the prior + prior_predictive groups onto the trace before saving.

    ``run_prior_predictive`` stores 1000 prior draws on ``context.prior_samples``;
    previously they were discarded (issue #125 step 0b). Copy the ``prior`` and
    ``prior_predictive`` subtrees onto ``context.trace`` (an ``xarray`` DataTree)
    so ``trace.nc`` carries them for prior-predictive checks and prior-vs-
    posterior overlays. Guarded — a merge failure must not lose the trace.
    """
    if context.prior_samples is None or context.trace is None:
        return
    for group in ("prior", "prior_predictive"):
        try:
            if group in context.prior_samples.children and group not in context.trace.children:
                context.trace[group] = context.prior_samples[group]
        except Exception as exc:  # pragma: no cover
            rprint(f"[yellow]Could not attach {group} group to trace: {exc}[/yellow]")


def save_trace(context: StatisticalFitContext, filename: str = "trace.nc") -> str:
    _attach_prior_groups(context)
    path = os.path.join(context.output_dir, filename)
    context.trace.to_netcdf(path)
    return path


def _joint_cell_outcome_index(
    context: StatisticalFitContext, outcome_symbol: str
) -> tuple[np.ndarray | None, str]:
    """Per-cell outcome index for the joint model's flattened ``y_post``, or None.

    Returns ``(index, outcome_symbol)``. ``index`` is the constant-data
    ``y_post_cell_outcome`` array (each flattened cell's outcome position in the
    outcome order); ``None`` marks a non-joint family. If a joint map is present,
    failure to resolve the requested outcome raises rather than silently pooling
    counts with incompatible denominators.
    """
    samples = context.prior_samples if context.prior_samples is not None else context.trace
    cd = getattr(samples, "constant_data", None)
    if cd is None or "y_post_cell_outcome" not in cd:
        return None, outcome_symbol
    idx = np.asarray(cd["y_post_cell_outcome"].values).ravel().astype(int)
    outcomes: list[str] = []
    extra = getattr(getattr(context, "spec", None), "extra", {}) or {}
    outcomes = [str(o) for o in extra.get("outcomes", ())]
    if not outcomes:
        for source, group in (
            (context.prior_samples, "prior"),
            (context.trace, "posterior"),
        ):
            dataset = getattr(source, group, None) if source is not None else None
            if dataset is not None and "outcome" in dataset.coords:
                outcomes = [str(o) for o in dataset.coords["outcome"].values]
                break
    if not outcomes:
        model_coords = getattr(getattr(context, "model", None), "coords", {}) or {}
        outcomes = [str(o) for o in model_coords.get("outcome", ())]
    if not outcomes:
        raise ValueError("joint predictive cells exist but outcome labels are unavailable")
    if outcome_symbol not in outcomes:
        raise KeyError(
            f"outcome {outcome_symbol!r} is not in the joint outcome set {outcomes}"
        )
    if idx.size and (idx.min() < 0 or idx.max() >= len(outcomes)):
        raise ValueError("joint predictive cell map contains an invalid outcome index")
    return idx, outcome_symbol


def _predictive_values_for_outcome(
    context: StatisticalFitContext,
    samples: xr.DataTree,
    *,
    group: str,
    node: str,
    outcome_symbol: str,
) -> tuple[np.ndarray, str]:
    """Select one outcome's predictive cells without pooling denominators."""
    predictive = getattr(samples, group)[node]
    if "outcome" in predictive.dims:
        labels = [str(o) for o in predictive.coords["outcome"].values]
        if outcome_symbol not in labels:
            raise KeyError(
                f"outcome {outcome_symbol!r} is not in predictive outcomes {labels}"
            )
        predictive = predictive.sel(outcome=outcome_symbol)
        return np.asarray(predictive.values, dtype=float), outcome_symbol
    values = np.asarray(predictive.values, dtype=float)
    cell_idx, outcome_symbol = _joint_cell_outcome_index(context, outcome_symbol)
    if cell_idx is None:
        return values, outcome_symbol
    if cell_idx.size != values.shape[-1]:
        raise ValueError("joint predictive cell map does not align with predictive draws")
    extra = getattr(context.spec, "extra", {}) or {}
    labels = [str(o) for o in extra.get("outcomes", ())]
    if not labels:
        for source, source_group in (
            (context.prior_samples, "prior"),
            (context.trace, "posterior"),
        ):
            dataset = (
                getattr(source, source_group, None) if source is not None else None
            )
            if dataset is not None and "outcome" in dataset.coords:
                labels = [str(o) for o in dataset.coords["outcome"].values]
                break
    target = labels.index(outcome_symbol)
    return values[..., cell_idx == target], outcome_symbol


def _shared_count_histogram_edges(
    replicated: np.ndarray, observed: np.ndarray
) -> np.ndarray:
    """Return unit-width bin edges shared by observed and replicated counts."""

    values = np.concatenate(
        [np.asarray(replicated, dtype=float).ravel(), np.asarray(observed, dtype=float).ravel()]
    )
    values = values[np.isfinite(values)]
    if not values.size:
        raise ValueError("predictive histogram has no finite values")
    lower = int(np.floor(values.min()))
    upper = int(np.ceil(values.max()))
    return np.arange(lower - 0.5, upper + 1.5, 1.0)


def _overlay_count_histograms(
    replicated: np.ndarray,
    observed: np.ndarray,
    *,
    predictive_label: str,
) -> None:
    """Overlay comparable count histograms using exactly the same support."""

    bins = _shared_count_histogram_edges(replicated, observed)
    plt.hist(
        replicated,
        bins=bins,
        density=True,
        color="#1f77b4",
        alpha=0.55,
        label=predictive_label,
    )
    plt.hist(
        observed,
        bins=bins,
        density=True,
        color="#d62728",
        alpha=0.55,
        label="observed",
    )


def save_prior_predictive_plot(
    context: StatisticalFitContext,
    outcome_symbol: str,
    *,
    node: str | None = None,
    filename_stem: str = "prior_predictive_check",
) -> None:
    """Surface the prior-predictive check in the report (#127 / #125 Area 2).

    Overlays the prior-predictive distribution of the outcome count against the
    observed counts and writes ``prior_predictive_check.png``. ``node`` selects the
    likelihood node to plot; pass it explicitly for models whose *first* observed
    RV is not the outcome (e.g. the mediation families register the mediator
    likelihood before the outcome ``y_post`` — defaulting to ``observed_RVs[0]``
    would overlay mediator draws on the outcome's observed counts). It defaults to
    the model's first observed node otherwise. For a multi-outcome likelihood (the
    joint model's ``(obs, outcome)`` ``y_post``) the column for ``outcome_symbol``
    is selected so counts with different denominators are not pooled into one
    histogram. A rootogram is added when the count outcome makes one meaningful.
    Guarded — a plotting failure must not abort the fit.
    """
    if context.prior_samples is None or context.prepared is None:
        rprint("[yellow]No prior samples to plot[/yellow]")
        return
    if node is None:
        try:
            node = context.model.observed_RVs[0].name
        except Exception:
            node = "y_post"
    try:
        rep, outcome_symbol = _predictive_values_for_outcome(
            context,
            context.prior_samples,
            group="prior_predictive",
            node=node,
            outcome_symbol=outcome_symbol,
        )
        rep = rep.ravel()
        obs = np.asarray(context.prepared.post_counts[outcome_symbol], dtype=float)
        obs = obs[np.isfinite(obs)]
        if node == "y_offfloor":
            obs = (obs > 0).astype(float)  # off-floor indicator, to match the node
        import pandas as pd

        plt.figure(figsize=(6, 4))
        _overlay_count_histograms(rep, obs, predictive_label="prior predictive")
        plt.xlabel(f"{outcome_symbol} count")
        plt.ylabel("density")
        plt.title(f"Prior-predictive check ({outcome_symbol})")
        plt.legend()
        # Data behind the plot (issue #208): compact distributional summary of the
        # prior-predictive replicates vs the observed counts (the raw replicate
        # array is large and already recoverable from trace.nc's prior group).
        summary = (
            pd.DataFrame(
                {
                    "prior_predictive": pd.Series(rep).describe(),
                    "observed": pd.Series(obs).describe(),
                }
            )
            .reset_index()
            .rename(columns={"index": "statistic"})
        )
        save_styled_figure(
            context.output_dir, filename_stem, data=summary
        )
    except Exception as exc:  # pragma: no cover
        rprint(f"[yellow]Prior-predictive plot failed: {exc}[/yellow]")


def save_joint_posterior_predictive_plot(
    context: StatisticalFitContext,
    outcome_symbol: str,
    *,
    node: str = "y_post",
    filename_stem: str = "posterior_predictive_check",
) -> None:
    """Plot one joint outcome's posterior predictive distribution.

    The joint likelihood is flattened across child-outcome cells. Selecting by
    the persisted cell map is mandatory: pooling raw counts from tests with
    different maxima has no interpretable predictive distribution. A mapping
    error therefore skips the plot with a warning rather than producing a pooled
    fallback.
    """
    if context.trace is None or context.prepared is None:
        rprint("[yellow]No posterior-predictive samples to plot[/yellow]")
        return
    try:
        rep, outcome_symbol = _predictive_values_for_outcome(
            context,
            context.trace,
            group="posterior_predictive",
            node=node,
            outcome_symbol=outcome_symbol,
        )
        rep = rep.ravel()
        obs = np.asarray(context.prepared.post_counts[outcome_symbol], dtype=float)
        obs = obs[np.isfinite(obs)]
        import pandas as pd

        plt.figure(figsize=(6, 4))
        _overlay_count_histograms(rep, obs, predictive_label="posterior predictive")
        plt.xlabel(f"{outcome_symbol} count")
        plt.ylabel("density")
        plt.title(f"Posterior-predictive check ({outcome_symbol})")
        plt.legend()
        summary = (
            pd.DataFrame(
                {
                    "posterior_predictive": pd.Series(rep).describe(),
                    "observed": pd.Series(obs).describe(),
                }
            )
            .reset_index()
            .rename(columns={"index": "statistic"})
        )
        save_styled_figure(
            context.output_dir, filename_stem, data=summary
        )
    except Exception as exc:  # pragma: no cover
        rprint(f"[yellow]Joint posterior-predictive plot failed: {exc}[/yellow]")


def _joint_outcome_predictive_tree(
    context: StatisticalFitContext,
    outcome_symbol: str,
    *,
    samples: xr.DataTree | None = None,
) -> xr.DataTree:
    """Return one outcome's observed, replicated and log-likelihood cells.

    LOO-PIT must compare like with like. The fitted multi-outcome likelihood is
    flattened over a named ``cell`` axis, so this helper uses its outcome map to
    construct an outcome-specific ArviZ tree. Keeping the matching pointwise log
    likelihood means PSIS weights are recomputed for that outcome rather than
    borrowed from pooled tests with different denominators.
    """
    samples = context.trace if samples is None else samples
    if samples is None:
        raise ValueError("joint LOO-PIT requires a posterior trace")
    cell_idx, outcome_symbol = _joint_cell_outcome_index(context, outcome_symbol)
    if cell_idx is None:
        raise ValueError("joint LOO-PIT requires y_post_cell_outcome constant data")
    outcomes = [str(o) for o in (context.spec.extra.get("outcomes", ()) or ())]
    if outcome_symbol not in outcomes:
        raise KeyError(f"outcome {outcome_symbol!r} is not in {outcomes}")
    keep = cell_idx == outcomes.index(outcome_symbol)
    if not np.any(keep):
        raise ValueError(f"joint outcome {outcome_symbol!r} has no predictive cells")

    def _subset(group: str) -> xr.Dataset:
        tree = getattr(samples, group, None)
        if tree is None or "y_post" not in tree:
            raise KeyError(f"joint LOO-PIT requires {group}['y_post']")
        data = tree.ds[["y_post"]]
        cell_dims = [
            dim for dim in data["y_post"].dims if dim not in {"chain", "draw"}
        ]
        if len(cell_dims) != 1:
            raise ValueError(f"{group}['y_post'] must have one cell dimension")
        cell_dim = cell_dims[0]
        if data.sizes[cell_dim] != cell_idx.size:
            raise ValueError(
                f"{group}['y_post'] does not align with the joint cell map"
            )
        return data.isel({cell_dim: keep})

    posterior = samples.posterior.ds
    if "tau" not in posterior:
        raise KeyError("joint LOO-PIT requires posterior['tau'] for relative ESS")
    return xr.DataTree.from_dict(
        {
            "/posterior": posterior[["tau"]],
            "/observed_data": _subset("observed_data"),
            "/posterior_predictive": _subset("posterior_predictive"),
            "/log_likelihood": _subset("log_likelihood"),
        }
    )


def save_joint_loo_pit_plot(
    context: StatisticalFitContext,
    outcome_symbol: str,
    *,
    filename_stem: str = "loo_pit",
) -> None:
    """Save an outcome-specific LOO-PIT calibration plot for a joint fit."""
    try:
        samples = thin_for_plots(context.trace)
        outcome_tree = _joint_outcome_predictive_tree(
            context, outcome_symbol, samples=samples
        )
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        rprint(f"[yellow]Joint LOO-PIT preparation failed: {exc}[/yellow]")
        return

    import arviz_plots as azp

    _save_pc(
        context.output_dir,
        lambda: azp.plot_loo_pit(outcome_tree, var_names=["y_post"]),
        f"{filename_stem}.png",
        title=f"LOO-PIT calibration ({outcome_symbol})",
    )
