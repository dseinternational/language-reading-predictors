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
   (divergences, BFMI, R-hat, ESS) that drives the findings-first badge and the
   full banner inside the collapsed Technical checks section.
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

import json
import os
import re
import tempfile
from collections.abc import Sequence
from typing import Any

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from pymc.stats import compute_log_likelihood, compute_log_prior
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

from language_reading_predictors.statistical_models.artifacts import (
    guard_optional,
    record_artifact,
    save_table,
)
from language_reading_predictors.models._reporting import (
    print_table,
    ranked_dataframe_table,
)
from language_reading_predictors.statistical_models.context import (
    StatisticalFitContext,
)
from language_reading_predictors.statistical_models.sampling_quality import (
    sampling_quality as _sampling_quality,
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


def _reuse_existing_trace(context: StatisticalFitContext) -> bool:
    """Load a saved posterior instead of sampling, when re-emitting artefacts.

    Enabled by ``DSE_LRP_REUSE_TRACE``: ``"1"`` reuses the trace at the model's
    own ``final_output_dir`` (the previous publication, untouched until this run
    publishes); any other value is treated as a directory to read ``trace.nc``
    from. The saved DataTree is loaded whole; the later stages recompute the
    log-likelihood and posterior-predictive groups in place
    (``extend_inferencedata=True`` overwrites them), so every downstream artefact
    is regenerated from the saved draws without re-running NUTS. Returns True when
    a trace was loaded.
    """
    reuse = os.environ.get("DSE_LRP_REUSE_TRACE")
    if not reuse:
        return False
    source = context.final_output_dir if reuse == "1" else reuse
    trace_path = os.path.join(source, "trace.nc")
    if not os.path.exists(trace_path):
        raise FileNotFoundError(
            "reuse-trace mode requires the persisted primary trace at "
            f"{trace_path}; refusing to run fresh NUTS"
        )
    from language_reading_predictors.statistical_models.reporting import (
        require_reuse_compatibility,
    )

    require_reuse_compatibility(context, source)
    context.trace = az.from_netcdf(trace_path)
    rprint(f"[cyan]Reusing saved posterior (no sampling): {trace_path}[/cyan]")
    return True


def sample_posterior(context: StatisticalFitContext) -> None:
    if _reuse_existing_trace(context):
        return
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


#: Name of the child-aggregated log-likelihood that
#: :func:`compute_log_likelihood_and_loo` writes next to the row-level ``y_post``
#: so PSIS-LOO can leave one whole child out. It is a *re-expression* of ``y_post``
#: (each draw's row contributions summed within child), not a second likelihood:
#: anything that sums the ``log_likelihood`` group — power scaling above all —
#: must skip it, or the likelihood is counted twice (2026-08-22 adjusted-family
#: review, finding 1: every published likelihood sensitivity of the stacked
#: Byrne transition fit and the eight joint / joint-mechanism fits was doubled).
LOO_CHILD_AGGREGATE_NODE = "y_post_child"


def psense_likelihood_var_names(trace) -> list[str] | None:
    """The ``log_likelihood`` variables power scaling should sum over.

    Every variable in the group except :data:`LOO_CHILD_AGGREGATE_NODE`; ``None``
    (arviz-stats' "all of them") when the group is absent or carries no aggregate,
    so a trace without the child re-expression is handled exactly as before.
    """
    log_likelihood = getattr(trace, "log_likelihood", None)
    if log_likelihood is None:
        return None
    try:
        names = list(log_likelihood.data_vars)
    except AttributeError:  # pragma: no cover - defensive
        return None
    if LOO_CHILD_AGGREGATE_NODE not in names:
        return None
    kept = [name for name in names if name != LOO_CHILD_AGGREGATE_NODE]
    return kept or None


def _joint_log_likelihood_by_child(trace: xr.DataTree) -> xr.DataArray | None:
    """Aggregate a marked repeated-row likelihood to the child unit.

    The joint ITT likelihood stores one cell per observed child-outcome pair.
    Leaving cells out would condition prediction of one outcome on the same
    child's remaining outcomes and would not answer leave-one-child-out
    generalisation. The stacked Byrne transition model has the same issue across
    annual rows and explicitly persists ``loo_child_idx``. This helper recognises
    either map and sums each draw's likelihood contributions within child.
    """
    constant = getattr(trace, "constant_data", None)
    log_likelihood = getattr(trace, "log_likelihood", None)
    if constant is None or log_likelihood is None or "y_post" not in log_likelihood:
        return None
    if "loo_child_idx" in constant:
        map_name = "loo_child_idx"
        aggregation = "sum over repeated child rows"
    elif "y_post_cell_row" in constant:
        map_name = "y_post_cell_row"
        aggregation = "sum over observed outcomes"
    else:
        return None
    ll = log_likelihood["y_post"]
    unit_dims = [d for d in ll.dims if d not in {"chain", "draw"}]
    if len(unit_dims) != 1:
        raise ValueError(
            "child-aggregated y_post log likelihood must have one observation dimension"
        )
    unit_dim = unit_dims[0]
    rows = np.asarray(constant[map_name].values, dtype=int).ravel()
    if rows.size != ll.sizes[unit_dim]:
        raise ValueError("child-row map does not align with y_post log likelihood")
    if map_name == "loo_child_idx":
        # The explicit map's values ARE dense child indices, so the unit count
        # comes from the map itself. ``G`` is a per-observation constant, and in a
        # stacked repeated-row design its size is the number of rows, not the
        # number of children — reading it here would append silent zero-likelihood
        # phantom units to the LOO (2026-08-21 joint-mechanism review, finding 3).
        n_children = int(rows.max()) + 1 if rows.size else 0
    else:
        n_children = int(constant["G"].size) if "G" in constant else (
            int(rows.max()) + 1 if rows.size else 0
        )
    if rows.size and (rows.min() < 0 or rows.max() >= n_children):
        raise ValueError("child-row map contains an out-of-range child index")
    ordered = ll.transpose("chain", "draw", unit_dim)
    aggregated = np.zeros(
        (ordered.sizes["chain"], ordered.sizes["draw"], n_children), dtype=float
    )
    values = np.asarray(ordered.values, dtype=float)
    for child in range(n_children):
        aggregated[..., child] = values[..., rows == child].sum(axis=-1)
    return xr.DataArray(
        aggregated,
        dims=("chain", "draw", "loo_child"),
        coords={
            "chain": ordered.coords["chain"],
            "draw": ordered.coords["draw"],
            "loo_child": np.arange(n_children),
        },
        name="y_post_child",
        attrs={"loo_unit": "child", "aggregation": aggregation},
    )


def log_density_model(model: pm.Model) -> pm.Model:
    """Return a model whose value-variable names round-trip to their RV names (#453).

    ``pm.compute_log_prior`` / ``pm.compute_log_likelihood`` call
    ``remove_value_transforms`` and then subset the posterior by
    ``[rv.name for rv in model.free_RVs]``. The un-transforming step recovers each
    value variable's base name with ``pymc.util.get_untransformed_name``, which drops
    a **fixed three** trailing underscore-separated components — one for the transform
    name and two for the ``__`` marker. That is only correct when ``transform.name``
    itself contains no underscore. Two shipped transforms violate it:
    ``CholeskyCorrTransform`` (``cholesky_corr``, the default for ``pm.LKJCorr``) and
    ``LogExpM1`` (``log_exp_m1``). For those, ``corr_cholesky_corr__`` comes back as
    ``corr_cholesky`` while the posterior stores ``corr``, and ``xarray`` raises
    ``exact match required for all data variable names`` — for the *prior* group as
    well as the likelihood group. Present in PyMC 6.1.0 and 6.2.0 alike; see
    ``notes/assets/draft-pymc-issue-get-untransformed-name.md``.

    The repair is a rename, not a rescale: the posterior already stores the
    constrained draw, which is exactly what the un-transformed model's log-density
    expects, so only the label is wrong. We build the un-transformed model ourselves,
    restore each value variable's name from its RV, and hand *that* to PyMC — whose
    own second ``remove_value_transforms`` is then a no-op on the names.

    A model whose names already round-trip is returned unchanged, so the ordinary
    path is untouched. ``tests/statistical_models/test_diagnostics.py`` pins both
    halves, including the upstream round-trip itself, so a PyMC upgrade that fixes
    ``get_untransformed_name`` cannot silently reintroduce or mask the seam.
    """
    from pymc.model.transform.conditioning import remove_value_transforms
    from pymc.util import get_untransformed_name

    def _recovered_name(value_name: str) -> str:
        try:
            return get_untransformed_name(value_name)
        except ValueError:
            return value_name

    values = model.rvs_to_values
    if all(_recovered_name(values[rv].name) == rv.name for rv in model.free_RVs):
        return model

    untransformed = remove_value_transforms(model)
    for rv in untransformed.free_RVs:
        value = untransformed.rvs_to_values[rv]
        if value.name != rv.name:
            value.name = rv.name
    return untransformed


def compute_log_likelihood_and_prior(
    context: StatisticalFitContext, *, strict: bool = True
) -> None:
    """Add the ``log_likelihood`` and ``log_prior`` groups to the trace.

    Both are needed by PSIS-LOO and by power-scaling prior sensitivity
    (``arviz_stats.psense``). Split out of :func:`compute_log_likelihood_and_loo`
    (#381) so families that do not compute LOO (e.g. mediation, correlated-factor)
    can still reach psense by calling this directly, without the LOO step. Families
    with a bespoke log-likelihood (the longitudinal correlated-factor model) build
    their groups their own way and do not call this.

    Both calls go through :func:`log_density_model`, which repairs the ``pm.LKJCorr``
    value-variable naming seam (#453) that previously cost the RLM joint-growth model
    both groups — and therefore its psense — at every fit. Passing the model explicitly
    is what makes that possible, and it also removes this function's reliance on an
    ambient model context.

    ``strict`` (default True): re-raise a ``compute_log_likelihood`` failure — the
    contract the LOO path relies on. The psense-only callers pass ``strict=False`` so a
    model ``compute_log_likelihood`` refuses degrades to a warning and simply gets no
    psense, rather than crashing the fit over a secondary diagnostic. The RLM
    joint-growth family's ``compute_loo=False`` is a separate matter — one likelihood
    node per measure makes single-target pointwise PSIS-LOO undefined. ``log_prior`` is
    always guarded.
    """
    density_model = log_density_model(context.model)
    try:
        context.trace = compute_log_likelihood(
            context.trace, model=density_model, progressbar=False
        )
    except Exception as exc:
        if strict:
            raise
        rprint(f"[yellow]log_likelihood group skipped: {exc}[/yellow]")
    try:
        context.trace = compute_log_prior(
            context.trace, model=density_model, progressbar=False
        )
    except Exception as exc:  # pragma: no cover - psense is secondary
        rprint(f"[yellow]log_prior group skipped: {exc}[/yellow]")


def compute_log_likelihood_and_loo(context: StatisticalFitContext) -> None:
    """Add log-likelihood + log-prior groups and compute pointwise LOO.

    The LOO is computed ``pointwise=True`` so the per-observation Pareto-k
    diagnostics survive on ``context.loo`` for the report's Pareto-k bands
    (load-bearing at n ≈ 33–54, where one influential child can drive elpd). The
    ``log_prior`` group is added (guarded) so power-scaling prior sensitivity
    (``arviz_stats.psense``) is reachable from the persisted trace.
    """
    compute_log_likelihood_and_prior(context)
    child_ll = _joint_log_likelihood_by_child(context.trace)
    if child_ll is not None:
        context.trace.log_likelihood[LOO_CHILD_AGGREGATE_NODE] = child_ll
        context.loo = az.loo(
            context.trace, var_name=LOO_CHILD_AGGREGATE_NODE, pointwise=True
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
        save_table(context, "diagnostics", summary, index=True)

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
    with guard_optional(
        context,
        "deterministics summary",
        filename="diagnostics_deterministics.csv",
        kind="table",
    ):
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
        save_table(
            context, "diagnostics_deterministics", summary, index=True, required=False
        )


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
    gate_names = _gate_var_names(context, var_names)
    payload = _shared_write_diagnostics_summary(
        context.trace,
        context.output_dir,
        var_names=gate_names,
        tables=context.tables,
    )
    return _fail_on_unassessable(context, payload, gate_names)


def _fail_on_unassessable(
    context: StatisticalFitContext,
    payload: dict,
    gate_names: list[str] | None,
) -> dict:
    """Fail the gate when a gated variable's R-hat / ESS could not be assessed.

    The shared writer reduces R-hat / ESS with NaN-skipping operations
    (``np.nanmax``, a row-wise minimum, ``np.nanmin``), and a NaN also compares
    False against the thresholds, so it never reaches ``rhat_failing`` /
    ``ess_failing`` either. A trace holding one healthy parameter and one
    constant or unsampled one therefore returned ``passed=true`` with finite
    extrema and empty failing lists (2026-08-22 ITT audit, finding 1). Its BFMI
    check was already hardened against exactly this; R-hat and ESS were not.

    The scan runs over the same gated set as the writer — the model's free RVs
    plus the curated headline terms — so a *mathematically constant*
    deterministic (an ``LKJCorr`` matrix diagonal, the only non-finite rows in
    the stored bundles) cannot trip it: those are never free RVs and the curated
    lists do not name them. Recorded as its own ``diagnostics_assessable`` check
    rather than folded into ``rhat``, because "we measured this and it failed" and
    "we could not measure this" are different verdicts;
    ``reporting.convergence_gate_failures`` already fails closed on any non-``True``
    check it does not recognise, so the release gate picks it up unchanged.
    """
    try:
        signals = _sampling_quality(context.trace, var_names=gate_names)
    except Exception as exc:  # pragma: no cover - defensive
        rprint(f"[yellow]unassessable-parameter scan failed: {exc}[/yellow]")
        return payload
    unassessable = list(signals.unassessable)
    payload["checks"]["diagnostics_assessable"] = not unassessable
    payload["unassessable_parameters"] = unassessable
    if unassessable:
        payload["passed"] = False
        shown = ", ".join(unassessable[:6])
        if len(unassessable) > 6:
            shown += f", ... ({len(unassessable)} in total)"
        rprint(
            "[red]  Convergence gate: REVIEW — R-hat / ESS could not be assessed "
            f"for {shown}[/red]"
        )
    # Rewritten whether or not the scan found anything, so the artefact records
    # that the check *ran*. Writing only on failure left every clean fit's
    # ``diagnostics_summary.json`` without the key, which reads identically to a
    # fit from before the check existed — and left the file disagreeing with the
    # ``tables`` entry built from the same payload.
    with open(
        os.path.join(context.output_dir, "diagnostics_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(payload, handle, indent=2, default=str)
    if context.tables is not None:
        context.tables["diagnostics_summary"] = payload
    return payload


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
        "unassessable_parameters": "",
    }
    try:
        # Unrounded extraction lives in ``sampling_quality`` — see that module for the
        # ``round_to="none"`` and coercion traps it exists to stop recurring.
        signals = _sampling_quality(trace, var_names=var_names)
        max_rhat = signals.max_rhat
        min_ess = signals.min_ess
        min_bfmi = signals.min_bfmi
        if signals.n_divergences is None:
            # No ``diverging`` in sample_stats: the gate cannot be evaluated, which is
            # the "uncheckable" case (``converged=None``), not a failure. Previously the
            # missing key raised and landed in the except branch below.
            raise KeyError("sample_stats has no 'diverging' variable")
        n_div = signals.n_divergences
        result.update(
            max_rhat=max_rhat,
            min_ess=min_ess,
            min_bfmi=min_bfmi,
            n_divergences=n_div,
        )
        # An unassessable parameter fails the sub-fit gate for the same reason it
        # fails the primary one: the NaN-skipping extrema cannot see it, so
        # ``max_rhat`` / ``min_ess`` would report the healthy parameters alone
        # (2026-08-22 ITT audit, finding 1).
        # Comma-separated, not a list: this dict is spread into one-row provenance
        # frames (``influence.summarise_influence_refit``,
        # ``subfits.SubfitResult``) where an empty list cannot become a column.
        result["unassessable_parameters"] = ", ".join(signals.unassessable)
        result["converged"] = bool(
            not signals.unassessable
            and max_rhat <= RHAT_MAX
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
            f"divergences={result['n_divergences']}"
            + (
                f", unassessable={result['unassessable_parameters']}"
                if result.get("unassessable_parameters")
                else ""
            )
            + "); its published estimates are flagged not-converged.[/red]"
        )
    return result


def _ess_evolution_vars(
    context: StatisticalFitContext,
    causal_term: str | None,
    fallback_var_names: Sequence[str] | None,
) -> list[str] | None:
    """Variables for the ESS-evolution panel, bounded by ArviZ's subplot guard.

    A family with a causal term plots that term alone. A descriptive family has
    none, and ``None`` would enumerate every posterior variable — per-observation
    and per-subject deterministics included — so the plot exceeds
    ``rcParams["plot.max_subplots"]`` and is lost to ``_save_pc``'s guard
    (2026-08-21 historical-families review, finding 5). Fall back to the curated
    headline variables the caller already has, trimmed greedily to the subplot
    limit so a wide curated list still yields a figure rather than none.
    """
    if causal_term:
        return [causal_term]
    if not fallback_var_names:
        return None
    try:
        posterior = context.trace.posterior
    except Exception:  # pragma: no cover - defensive
        return list(fallback_var_names) or None
    limit = az.rcParams["plot.max_subplots"] or 40
    chosen: list[str] = []
    panels = 0
    for name in fallback_var_names:
        if name not in posterior:
            continue
        size = int(
            np.prod(
                [
                    length
                    for dim, length in posterior[name].sizes.items()
                    if dim not in {"chain", "draw"}
                ],
                dtype=int,
            )
        )
        if chosen and panels + size > limit:
            break
        chosen.append(name)
        panels += size
    return chosen or None


def run_extended_diagnostics(
    context: StatisticalFitContext,
    *,
    causal_term: str | None = None,
    include_loo_pit: bool = True,
    fallback_var_names: Sequence[str] | None = None,
) -> None:
    """Pareto-k, rank, ESS-evolution and LOO-PIT plots (issue #125 Area 3).

    Called after posterior-predictive sampling so all groups are present. Pareto-k
    reuses ``context.loo`` (computed ``pointwise=True``); rank focuses on the
    causal term; LOO-PIT needs the posterior-predictive group. All guarded.

    ``fallback_var_names`` is the family's curated diagnostic list, used for the
    ESS-evolution panel when no ``causal_term`` is declared — see
    :func:`_ess_evolution_vars`.
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
    #
    # A descriptive family declares no causal term, and ``var_names=None`` then
    # enumerates *every* posterior variable — including the per-observation and
    # per-subject deterministics — which trips ArviZ's max-subplots guard and
    # loses the figure entirely (493 requested panels against a limit of 40 for
    # a historical-growth fit; 2026-08-21 review, finding 5, and the same cause
    # recorded for four other termless families on 2026-08-21). Fall back to the
    # curated headline scalars instead of the whole posterior.
    ess_vars = _ess_evolution_vars(context, causal_term, fallback_var_names)
    _save_pc(
        out,
        lambda: azp.plot_ess_evolution(
            context.trace,
            var_names=ess_vars,
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


# Dimensions ``arviz_plots.plot_prior_posterior`` introduces on the combined
# tree: it concatenates the prior and posterior groups along a new ``group``
# dimension. A model *coordinate* named ``group`` — the cohort reading-group
# dimension in both RLM historical families — collides with it.
_PRIOR_POSTERIOR_RESERVED_DIMS = ("group",)


def _prior_posterior_plot_view(
    trace, var_names: list[str] | None
) -> tuple[object, list[str] | None]:
    """Return a trace view whose dimensions cannot collide with the plot's own.

    ``plot_prior_posterior`` concatenates the prior and posterior groups along a
    new dimension it calls ``group``, so a model that declares its own ``group``
    coordinate raises ``conflicting sizes for dimension 'group'`` before any
    figure exists. The concat runs on the whole dataset, so selecting different
    ``var_names`` cannot avoid the clash — the colliding *dimension* has to be
    renamed, exactly as :func:`_psense_plot_view` does for power scaling's own
    reserved dims (issue #340; 2026-08-21 historical-families review, finding 1,
    where this had silently suppressed the overlay in all eleven RLM fits).

    Renaming rather than dropping: ``group`` is a real reported dimension here
    (``sigma_subject`` / ``kappa`` / the growth deterministics are all indexed by
    it), so its panels must survive, labelled ``group (dimension)`` to separate
    them from the prior/posterior legend. Guarded — an unexpected structure
    degrades to the original pair, i.e. to the pre-existing behaviour.
    """
    try:
        renames = {
            dim: f"{dim} (dimension)"
            for dim in _PRIOR_POSTERIOR_RESERVED_DIMS
            if dim in trace.posterior.dims
        }
        if not renames:
            return trace, var_names
        groups = {}
        for group in trace.children:
            ds = trace[group].to_dataset()
            present = {k: v for k, v in renames.items() if k in ds.dims}
            groups[group] = ds.rename(present) if present else ds
        return type(trace).from_dict(groups), var_names
    except Exception as exc:  # pragma: no cover - plotting stays guarded below
        rprint(f"[yellow]prior/posterior plot view unchanged: {exc}[/yellow]")
        return trace, var_names


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
    # Rename any model dimension the plot reserves for its own use, before the
    # panel count is derived from it (the names are unchanged by the rename).
    tr, var_names = _prior_posterior_plot_view(tr, var_names)

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


# Dimensions ``arviz_plots.plot_psense_dist`` introduces on the resampled
# posterior: ``alpha`` (the power-scaling factor, from
# ``arviz_stats.power_scale_dataset``), ``component_group`` (prior vs
# likelihood) and ``sample`` (the stacked chain × draw dimension). A model
# parameter sharing one of these names collides with the new dimension.
_PSENSE_RESERVED_DIMS = ("alpha", "component_group", "sample")


def _psense_plot_view(trace, var_names: list[str]) -> tuple[object, list[str]]:
    """Return a trace view whose posterior cannot collide with psense's own dims.

    ``plot_psense_dist`` resamples the **whole** posterior group and then
    concatenates the results along a new ``alpha`` dimension, so a model with an
    ``alpha`` parameter — the intercept, in nearly every family here — raises
    ``alpha already exists as coordinate or variable name`` before ``var_names``
    is ever applied. Selecting different ``var_names`` therefore cannot avoid the
    clash (issue #340); the colliding *variable* has to go.

    Renaming rather than dropping, because ``alpha`` is itself a requested
    parameter for several families, and because ``arviz_stats.extract`` returns a
    bare ``DataArray`` when a posterior is cut down to one variable, which
    ``plot_psense_dist`` cannot consume. The renamed parameter keeps its panel in
    the figure, labelled ``alpha (parameter)`` to separate it from the
    power-scaling α in the legend. ``psense_summary.csv`` is computed from the
    untouched trace and keeps the original names.

    Returns the (possibly unchanged) trace and the correspondingly mapped
    ``var_names``. Guarded — an unexpected structure degrades to the original
    pair, i.e. to the pre-existing behaviour.
    """
    try:
        posterior = trace.posterior.to_dataset()
        clashing = [
            name
            for name in _PSENSE_RESERVED_DIMS
            if name in posterior.variables or name in posterior.dims
        ]
        if not clashing:
            return trace, var_names
        renames: dict[str, str] = {}
        for name in clashing:
            renamed = f"{name} (parameter)"
            while renamed in posterior.variables or renamed in renames.values():
                renamed += "_"
            renames[name] = renamed
        groups = {}
        for group in trace.children:
            ds = trace[group].to_dataset()
            groups[group] = ds.rename(renames) if group == "posterior" else ds
        return type(trace).from_dict(groups), [renames.get(n, n) for n in var_names]
    except Exception as exc:
        rprint(f"[yellow]psense plot view unchanged: {exc}[/yellow]")
        return trace, var_names


def _psense_layout(trace, var_names: list[str]) -> tuple[dict, dict]:
    """Explicit geometry for the two-column psense grid, and a matching rc patch.

    ``plot_psense_dist`` lays out one row per parameter — and per level of any
    non-sampling coordinate — against two fixed columns (prior, likelihood). Its
    auto-sized default barely grows with the row count: a five-row selection
    gets ~0.4 inches of plotting area per panel, which flattens every density
    into a line. Size the figure by the row count instead.

    Only ``figsize`` is set, deliberately. The house style
    (``set_matplotlib_default_style``, applied at the fit entry point) turns on
    matplotlib's constrained layout, which computes its own spacing and makes
    room for the ``suptitle`` ``_save_pc`` adds — but which also *overrides* any
    ``gridspec_kw``. Passing ``hspace``/``top`` alongside it does not tighten the
    grid, it collapses the panels to ~0.2 inches.

    A single row lays out correctly unaided, so leave that case on the defaults.
    Returns ``(plot_kwargs, rc)``, both empty when nothing needs overriding; the
    rc patch raises ArviZ's 40-panel guard for selections that exceed it.
    """
    rows = 0
    try:
        posterior = trace.posterior
        for name in var_names:
            if name not in posterior:
                continue
            levels = [
                size
                for dim, size in posterior[name].sizes.items()
                if dim not in {"chain", "draw"}
            ]
            rows += int(np.prod(levels, dtype=int))
    except Exception:
        return {}, {}
    if rows < 2:
        return {}, {}
    panels = 2 * rows
    limit = az.rcParams["plot.max_subplots"]
    rc = {"plot.max_subplots": panels} if limit is not None and panels > limit else {}
    # ~2 inches a row keeps the densities legible; the cap stops a 25-level
    # coefficient vector from producing an unmanageable figure file.
    height = min(2.0 * rows + 1.0, 36.0)
    return {"figure_kwargs": {"figsize": (11.0, height)}}, rc


def psense_artifacts(
    trace,
    out: str,
    var_names: list[str],
):
    """Write ``psense_summary.csv`` and ``psense.png`` for ``var_names``.

    Split out of :func:`run_psense` (#381) so the regeneration script can produce
    exactly the fit-time artefacts from a stored trace, with no second
    implementation to drift. Power-scaling is importance reweighting over the draws
    already in hand, **not** a refit, so any fit whose trace carries the
    ``log_prior`` and ``log_likelihood`` groups can be measured after the fact.

    Returns the summary frame, or ``None`` when psense could not be computed (a
    missing group, an API mismatch). The caller decides whether that is fatal: at
    fit time it is a warning, because psense is recommended-but-secondary at this
    n; a regeneration run reports it as a skip with its reason.
    """
    summary_path = os.path.join(out, "psense_summary.csv")
    df = None
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
        # (issue #270 item 2). ``likelihood_var_names`` keeps the child-level LOO
        # aggregate out of the likelihood sum (see LOO_CHILD_AGGREGATE_NODE).
        s = azs.psense_summary(
            trace,
            var_names=var_names,
            likelihood_var_names=psense_likelihood_var_names(trace),
        )
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
    except Exception as exc:  # pragma: no cover
        df = None
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
        rprint(f"[yellow]psense_summary skipped: {exc}[/yellow]")

    import arviz_plots as azp

    tr, plot_var_names = _psense_plot_view(thin_for_plots(trace), var_names)
    plot_kwargs, rc = _psense_layout(tr, plot_var_names)
    likelihood_var_names = psense_likelihood_var_names(tr)
    with az.rc_context(rc):
        _save_pc(
            out,
            lambda: azp.plot_psense_dist(
                tr,
                var_names=plot_var_names,
                likelihood_var_names=likelihood_var_names,
                **plot_kwargs,
            ),
            "psense.png",
            title="Prior/likelihood power-scaling sensitivity",
        )
    return df


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
    context.tables.pop("psense_summary", None)
    df = psense_artifacts(context.trace, context.output_dir, var_names)
    if df is not None:
        context.tables["psense_summary"] = df
        # The atomic temp-and-rename write stays with psense_artifacts (shared
        # with the post-hoc regeneration script); record it here so the fit's
        # manifest lists it as written rather than untracked.
        record_artifact(context, "psense_summary", df=df)


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

    An **empty** existing group is replaced rather than kept. Under
    ``--reuse-trace`` the saved DataTree is loaded whole, so a trace that was
    written with a restricted ``var_names`` carries a ``prior`` node with no
    variables in it — and a plain "already present" test then blocks the freshly
    drawn one from ever landing. That is how the three RLI measurement fits kept
    shipping without a prior-vs-posterior overlay even after the restriction that
    caused it was removed (#381): the re-emit drew the full prior and then
    declined to attach it. A populated group is still never overwritten, so a
    genuine reuse keeps the draws it was reusing.
    """
    if context.prior_samples is None or context.trace is None:
        return
    for group in ("prior", "prior_predictive"):
        try:
            if group not in context.prior_samples.children:
                continue
            existing = context.trace.children.get(group)
            if existing is None or not len(existing.data_vars):
                context.trace[group] = context.prior_samples[group]
        except Exception as exc:  # pragma: no cover
            rprint(f"[yellow]Could not attach {group} group to trace: {exc}[/yellow]")


def save_trace(context: StatisticalFitContext, filename: str = "trace.nc") -> str:
    _attach_prior_groups(context)
    path = os.path.join(context.output_dir, filename)
    context.trace.to_netcdf(path)
    record_artifact(
        context, os.path.splitext(filename)[0], filename=filename, kind="netcdf"
    )
    return path


def _joint_cell_outcome_index(
    context: StatisticalFitContext, outcome_symbol: str, node: str = "y_post"
) -> tuple[np.ndarray | None, int | None]:
    """Return the joint cell map and requested outcome position, or two ``None``s.

    ``index`` is the constant-data
    ``y_post_cell_outcome`` array (each flattened cell's outcome position in the
    outcome order); ``target`` is the requested outcome's position in that same
    resolved order. Two ``None`` values mark a non-joint family. If a joint map is
    present, failure to resolve the requested outcome raises rather than silently
    pooling counts with incompatible denominators.
    """
    samples = context.prior_samples if context.prior_samples is not None else context.trace
    cd = getattr(samples, "constant_data", None)
    if cd is None:
        return None, None
    # Per-node map first (``y_obs_cell_outcome`` for the stacked LCSM / growth
    # likelihoods), then the joint family's original name.
    key = next(
        (k for k in (f"{node}_cell_outcome", "y_post_cell_outcome") if k in cd), None
    )
    if key is None:
        return None, None
    idx = np.asarray(cd[key].values).ravel().astype(int)
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
    return idx, outcomes.index(outcome_symbol)


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
    cell_idx, target = _joint_cell_outcome_index(context, outcome_symbol, node=node)
    if cell_idx is None:
        return values, outcome_symbol
    assert target is not None
    if cell_idx.size != values.shape[-1]:
        raise ValueError("joint predictive cell map does not align with predictive draws")
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


def _observed_values_for_node(
    context: StatisticalFitContext,
    *,
    node: str,
    outcome_symbol: str,
    samples: Any = None,
) -> np.ndarray:
    """The likelihood's *own* observed vector for ``node``, selected like the replicates.

    Read from the trace's ``observed_data`` group — the exact array PyMC passed to
    ``observed=`` — and put through the same outcome selection as the predictive
    draws, so the two are aligned by construction.

    This must not be re-derived from the prepared container. Doing so silently
    compares different row sets whenever the likelihood's rows are not one-to-one
    with a symbol's panel cells, and the suite has two such shapes: the historical
    cohort fits core **plus** observed extension waves while ``LongitudinalPanel.counts``
    spans only the complete-case core window (300 modelled rows vs 228 core cells for
    ``lrp-rlm-hg-001``), and the LCSM / growth families flatten *all* measures into one
    ``y_obs`` vector (639 cells across W/L/E against 210 for W alone in
    ``lrp-rli-lcsm-067``). Both produced a plausible-looking overlay of one row set's
    replicates against another's observations. Same failure mode as the mechanism
    forest's reconstructed-index mismatch: derive the index once, in the factory, and
    read it back — never rebuild it in a consumer.
    """
    source = samples if samples is not None else context.prior_samples
    values, _ = _predictive_values_for_outcome(
        context,
        source,
        group="observed_data",
        node=node,
        outcome_symbol=outcome_symbol,
    )
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if node == "y_offfloor":
        values = (values > 0).astype(float)
    return values


def save_prior_predictive_rate_plot(
    context: StatisticalFitContext,
    outcome_symbol: str,
    *,
    node: str,
    filename_stem: str = "prior_predictive_check",
) -> None:
    """Prior-predictive check for a **binary / event** node, on the rate scale.

    A count histogram is the wrong instrument for a 0/1 node: every replicate is 0 or
    1, so the overlay carries no shape information. The posterior side already makes
    this distinction — :func:`reporting.ppc_offfloor_rate_coverage` checks the observed
    off-floor *rate* against the replicated-rate distribution "because per-observation
    interval coverage of a 0/1 indicator is degenerate" — and this is the prior-side
    counterpart of that same statistic, so the two ends of the check agree on what is
    being tested.

    The rate is taken over all modelled rows in one cell. The posterior check cells by
    arm (x wave where available), but a prior-predictive draw is arm-blind by
    construction — the prior does not know the assignment — so an overall rate is the
    meaningful summary here.
    """
    if context.prior_samples is None:
        rprint("[yellow]No prior samples to plot[/yellow]")
        return
    try:
        rep = np.asarray(
            getattr(context.prior_samples, "prior_predictive")[node].values, dtype=float
        )
        obs = np.asarray(
            getattr(context.prior_samples, "observed_data")[node].values, dtype=float
        ).ravel()
        obs = obs[np.isfinite(obs)]
        # Same 0/1 reduction as ``_offfloor_cell_rates`` so a raw-count node and a
        # Bernoulli node behave identically.
        obs_rate = float((obs > 0).mean())
        rep = (rep > 0).astype(float).reshape(-1, rep.shape[-1])
        rep_rate = rep.mean(axis=-1)  # one rate per prior draw
        import pandas as pd

        plt.figure(figsize=(6, 4))
        # Same palette as ``_overlay_count_histograms`` so the rate check reads as a
        # sibling of the count check rather than a different kind of figure.
        plt.hist(
            rep_rate, bins=30, density=True, alpha=0.55,
            label="prior predictive", color="#1f77b4",
        )
        plt.axvline(
            obs_rate, color="#d62728", linewidth=2,
            label=f"observed rate = {obs_rate:.2f}",
        )
        plt.xlabel(f"{outcome_symbol} event rate")
        plt.ylabel("density")
        plt.title(f"Prior-predictive event-rate check ({outcome_symbol})")
        plt.legend()
        summary = (
            pd.DataFrame(
                {
                    "prior_predictive_rate": pd.Series(rep_rate).describe(),
                    "observed_rate": pd.Series([obs_rate]).describe(),
                }
            )
            .reset_index()
            .rename(columns={"index": "statistic"})
        )
        save_styled_figure(context.output_dir, filename_stem, data=summary)
    except Exception as exc:  # pragma: no cover
        rprint(f"[yellow]Prior-predictive rate plot failed: {exc}[/yellow]")


def save_prior_predictive_dist_overlay(
    context: StatisticalFitContext,
    *,
    filename_stem: str = "prior_predictive_check",
) -> None:
    """Prior-predictive marginals for a **measurement / latent** node.

    The counterpart of the posterior side's generic overlay for models whose observed
    node is a multi-indicator matrix (the correlated-factor CFAs' standardised
    ``Z_obs`` / ``z_obs_*``). Those have no single count outcome, so
    ``ppc_artifacts.save_ppc`` deliberately gives them the distribution overlay and **no**
    coverage statistic; this keeps the prior end symmetric with that decision rather
    than inventing a coverage number for a latent measurement model. Pooling the
    indicators into one count histogram would be meaningless — they are standardised
    scores on different instruments.
    """
    if context.prior_samples is None:
        rprint("[yellow]No prior samples to plot[/yellow]")
        return
    try:
        import arviz_plots as azp

        pc = azp.plot_ppc_dist(context.prior_samples, group="prior_predictive")
        save_plotcollection(
            pc,
            context.output_dir,
            f"{filename_stem}.png",
            suptitle="Prior-predictive marginals",
        )
    except Exception as exc:  # pragma: no cover
        rprint(f"[yellow]Prior-predictive overlay failed: {exc}[/yellow]")


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
        obs = _observed_values_for_node(
            context, node=node, outcome_symbol=outcome_symbol
        )
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
        obs = _observed_values_for_node(
            context, node=node, outcome_symbol=outcome_symbol, samples=context.trace
        )
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


#: Posterior variables the joint LOO-PIT tree will carry for relative-ESS, in
#: preference order. ArviZ only needs *a* posterior group to compute the relative
#: effective sample size behind the PSIS weights; the ITT-shaped joint families
#: carry ``tau``, but a joint family whose reported coefficient is something else
#: (the ``joint_mechanism`` slopes, for instance) must not be silently skipped —
#: which is what happened before #427 review, when the hard ``tau`` requirement
#: raised a ``KeyError`` that :func:`save_joint_loo_pit_plot` swallowed.
_JOINT_LOO_PIT_POSTERIOR_VARS: tuple[str, ...] = ("tau", "beta_mech", "alpha")


def _joint_outcome_predictive_tree(
    context: StatisticalFitContext,
    outcome_symbol: str,
    *,
    samples: xr.DataTree | None = None,
    posterior_var: str | None = None,
) -> xr.DataTree:
    """Return one outcome's observed, replicated and log-likelihood cells.

    LOO-PIT must compare like with like. The fitted multi-outcome likelihood is
    flattened over a named ``cell`` axis, so this helper uses its outcome map to
    construct an outcome-specific ArviZ tree. Keeping the matching pointwise log
    likelihood means PSIS weights are recomputed for that outcome rather than
    borrowed from pooled tests with different denominators.

    ``posterior_var`` names the posterior variable carried into the tree for the
    relative-ESS calculation; ``None`` resolves the first of
    :data:`_JOINT_LOO_PIT_POSTERIOR_VARS` present, then any posterior variable.
    The choice does not change the LOO-PIT values — it only supplies ArviZ with a
    posterior group — so the fallback is safe, and it is what lets non-ITT joint
    families emit the plot at all.
    """
    samples = context.trace if samples is None else samples
    if samples is None:
        raise ValueError("joint LOO-PIT requires a posterior trace")
    cell_idx, target = _joint_cell_outcome_index(context, outcome_symbol)
    if cell_idx is None:
        raise ValueError("joint LOO-PIT requires y_post_cell_outcome constant data")
    assert target is not None
    keep = cell_idx == target
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
    if posterior_var is not None:
        if posterior_var not in posterior:
            raise KeyError(
                f"joint LOO-PIT requires posterior[{posterior_var!r}] for relative ESS"
            )
        ess_var = posterior_var
    else:
        ess_var = next(
            (v for v in _JOINT_LOO_PIT_POSTERIOR_VARS if v in posterior),
            next(iter(posterior.data_vars), None),
        )
        if ess_var is None:
            raise KeyError("joint LOO-PIT requires a non-empty posterior group")
    return xr.DataTree.from_dict(
        {
            "/posterior": posterior[[ess_var]],
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
    posterior_var: str | None = None,
) -> None:
    """Save an outcome-specific LOO-PIT calibration plot for a joint fit.

    ``posterior_var`` is forwarded to :func:`_joint_outcome_predictive_tree`; leave
    it ``None`` unless a family needs a specific relative-ESS variable.
    """
    try:
        samples = thin_for_plots(context.trace)
        outcome_tree = _joint_outcome_predictive_tree(
            context, outcome_symbol, samples=samples, posterior_var=posterior_var
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


def influence_diagnostics(ctx: StatisticalFitContext) -> tuple:
    """Persistable PSIS-LOO Pareto-k values for the likelihood's LOO units.

    Returns ``(dataframe, threshold, n_flagged)`` — the pointwise k values sorted
    descending (aligned to ``subject_ids``), the ``good_k`` threshold, and how
    many points exceed it. A point is one child in the single-period ITT/joint
    families, but one child-by-period row in repeated-measures families. Returns
    ``(None, None, None)`` if the LOO object exposes no aligned pointwise k.
    """
    if ctx.loo is None or getattr(ctx.loo, "pareto_k", None) is None:
        return None, None, None
    k = np.asarray(ctx.loo.pareto_k).ravel()
    ids = np.asarray(ctx.prepared.subject_ids)
    if len(k) != len(ids):
        # Historical-growth likelihood rows are the tidy ``panel.long`` rows,
        # while ``subject_ids`` is one value per child. Preserve the exact row
        # order the factory passed to PyMC so repeated children map correctly.
        long = getattr(ctx.prepared, "long", None)
        dataset = getattr(ctx.prepared, "dataset", None)
        subject_col = getattr(dataset, "subject_col", None)
        if (
            long is not None
            and subject_col is not None
            and subject_col in long
            and len(k) == len(long)
        ):
            ids = long[subject_col].to_numpy()
        else:
            child_idx = getattr(ctx.prepared, "child_idx", None)
            if child_idx is None:
                return None, None, None
            child_idx = np.asarray(child_idx, dtype=int)
            if child_idx.shape != ids.shape or len(k) != len(set(child_idx)):
                return None, None, None
            child_ids: list[object] = []
            for child in range(len(k)):
                matches = np.unique(ids[child_idx == child])
                if len(matches) != 1:
                    return None, None, None
                child_ids.append(matches[0])
            ids = np.asarray(child_ids)
    thr = float(getattr(ctx.loo, "good_k", 0.7) or 0.7)
    df = (
        pd.DataFrame(
            {
                "observation_index": np.arange(len(k), dtype=int),
                "subject_id": ids,
                "pareto_k": k,
            }
        )
        .sort_values("pareto_k", ascending=False)
        .reset_index(drop=True)
    )
    return df, thr, int((k > thr).sum())


def write_loo_influence(ctx: StatisticalFitContext) -> pd.DataFrame | None:
    """Persist pointwise Pareto-k values and explicit reliability flags.

    A sampler-convergence PASS does not guarantee that importance-sampled LOO is
    reliable.  Persisting the values makes the ``k > good_k`` gate available to
    report templates and downstream audits instead of leaving it visible only in
    a plot or the free-text ArviZ summary.
    """
    influence, threshold, _ = influence_diagnostics(ctx)
    if influence is None or threshold is None:
        return None
    out = influence.copy()
    out["good_k_threshold"] = threshold
    out["loo_reliable"] = out["pareto_k"] <= threshold
    save_table(ctx, "pareto_k", out)
    return out


# ``sample_subfit`` lived here until #394 design point 5. It sampled a sub-model and
# returned ``(trace, conv)``, leaving each caller to publish the verdict, persist the
# trace and record the provenance itself — and three families sampled their sub-fits
# with their own inline ``pm.sample`` call instead. Both paths are now
# :func:`subfits.run_subfit`, which returns a typed ``SubfitResult`` and writes
# ``subfit_provenance.csv``. :func:`subfit_convergence` above stays here: the
# prior-sensitivity sweep scripts and the influence/blending sensitivity modules check
# traces they sampled themselves.
