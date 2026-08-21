# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Latent change-score orchestration (``kind="lcsm"``, LRP67 and the #250 suite).

``fit_lcsm`` fits the coupled McArdle latent change-score model with process noise
and reports the per-target coupling tables. The settings select the shape: the
LRP67 default couples every other measure into the reading change, while the
lagged reverse-coupling models pass an explicit coupling map plus the
crossover-aware arm×window change intercepts — of which only the window-1
contrast is randomised — and a shared adjuster block. Couplings between measures
are adjusted associations; nothing but that window-1 contrast is causal.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Mapping

import pandas as pd

from language_reading_predictors.models._reporting import (
    print_table,
    ranked_dataframe_table,
    section_header,
)
from language_reading_predictors.statistical_models import (
    diagnostics as _diag,
    factories as _factories,
    lcsm as _lcsm,
    reporting as _report,
)
from language_reading_predictors.statistical_models.artifacts import save_table
from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
    make_context,
)
from language_reading_predictors.statistical_models.figure_artifacts import (
    write_panel_child_fit,
)
from language_reading_predictors.statistical_models.preprocessing import (
    load_wave_panel,
)
from language_reading_predictors.statistical_models.publication import (
    print_header,
    render_model_graph,
)
from language_reading_predictors.statistical_models.reporting import coef_row
from language_reading_predictors.statistical_models.runtime import (
    attach_built,
    finalize_report,
    require_spec,
    shared_stages,
    write_run_metadata,
)
from language_reading_predictors.statistical_models.stages import PrimaryFitPlan


def standardised_coupling_rows(
    post: Any, coupling_names: Mapping[tuple[str, str], str], ci_prob: float
) -> list[dict]:
    """SD-standardised level -> change couplings, and contrasts between sources.

    A raw coupling ``g_{src}`` is per unit of the *source's* latent logit, so two
    sources on different latent scales (letter sounds spread about three times as
    widely as expressive vocabulary on the logit scale) are not comparable in size.
    This standardises each coupling by the model's own latent scales, per draw,
    exactly as the reciprocal-dominance contrast does::

        g* = g * sd(prior source latent levels) / sd(target latent changes)

    so ``g*`` reads as "SDs of later change in the target per SD of prior level
    of the source". For every target with two or more sources it also returns the
    signed contrast ``g*_a - g*_b`` and the absolute-dominance contrast
    ``|g*_a| - |g*_b|`` for each pair. Contrasts are only formed between sources
    of the *same* target. ``post`` is any mapping of parameter name ->
    xarray DataArray with ``chain``/``draw`` dims carrying ``x_latent`` with
    ``(child, wave, outcome)`` dims — the posterior group of the fit's trace.
    Added 2026-08-19 (``notes/202608182200-findings-by-question.md``, question 8).
    """
    x = post["x_latent"]
    by_target: dict[str, list[str]] = {}
    for (src, tgt), _pname in coupling_names.items():
        by_target.setdefault(tgt, []).append(src)
    rows: list[dict] = []
    for tgt, sources in by_target.items():
        sd_target_change = (
            x.sel(outcome=tgt).diff("wave").std(dim=("child", "wave"))
        )
        std_g: dict[str, Any] = {}
        for src in sources:
            sd_source_level = (
                x.isel(wave=slice(0, -1))
                .sel(outcome=src)
                .std(dim=("child", "wave"))
            )
            g = post[coupling_names[(src, tgt)]]
            std_g[src] = g * sd_source_level / sd_target_change
            row = coef_row(
                f"std g ({src} -> {tgt} change)", std_g[src].values, ci_prob
            )
            row["kind"] = "standardised_coupling"
            row["source"] = src
            row["target"] = tgt
            row["sd_source_level_median"] = float(sd_source_level.median())
            row["sd_target_change_median"] = float(sd_target_change.median())
            rows.append(row)
        for a, b in combinations(sources, 2):
            signed = std_g[a] - std_g[b]
            row = coef_row(
                f"std g {a}->{tgt} - std g {b}->{tgt} (contrast)",
                signed.values,
                ci_prob,
            )
            row["kind"] = "contrast"
            row["source"] = f"{a} - {b}"
            row["target"] = tgt
            rows.append(row)
            dominance = abs(std_g[a]) - abs(std_g[b])
            row = coef_row(
                f"|std g {a}->{tgt}| - |std g {b}->{tgt}| (dominance)",
                dominance.values,
                ci_prob,
            )
            row["kind"] = "dominance"
            row["source"] = f"|{a}| - |{b}|"
            row["target"] = tgt
            rows.append(row)
    return rows


def write_standardised_couplings(
    ctx: Any, post: Any, coupling_names: Mapping[tuple[str, str], str]
) -> pd.DataFrame | None:
    """Write ``standardised_couplings.csv`` (see :func:`standardised_coupling_rows`).

    Returns the table, or ``None`` when the fit has no level couplings. Usable
    over a stored fit with a lightweight context (``output_dir`` and
    ``reporting.ci_prob``) — it reads the posterior only, so it needs no refit.
    """
    rows = standardised_coupling_rows(post, coupling_names, ctx.reporting.ci_prob)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    save_table(ctx, "standardised_couplings", df, required=False)
    return df


def fit_lcsm(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Latent change-score model (LRP67 + the lagged coupling suite, #250).

    Fits the coupled McArdle latent change-score model with process noise and
    reports the per-target coupling tables. The typed run plan selects the shape:
    the LRP67 default couples every other measure into the reading change; the
    lagged reverse-coupling models (LCSM-081/181/082) pass an explicit
    ``couplings`` map plus ``arm_window_intercepts`` (the crossover-aware
    arm x window change intercepts, with the window-1 randomised contrast
    written to ``itt_window1_contrast.csv``) and a shared adjuster
    ``covariate_block``. ``dominance_pair`` adds the SD-standardised
    reciprocal-dominance contrast (``dominance_summary.csv``).
    ``lagged_change_couplings`` (LCSM-091, #229 spec 2) adds prior-transition
    latent-change terms (``h_{src}``) to the named targets' change equations.
    """
    require_spec(spec, "lcsm")
    plan = _lcsm.resolve_lcsm_run_plan(spec)

    ctx = make_context(spec, config)
    ctx.resolved_plan = plan
    _report.write_model_recipe(ctx)

    section_header("Prepare data")
    outcomes = plan.outcomes
    reading_symbol = plan.reading_symbol
    couplings = plan.coupling_mapping()
    lagged_change_couplings = plan.lagged_coupling_mapping()
    arm_window = plan.arm_window_intercepts
    covariate_block = plan.covariate_block
    covariate_targets = plan.covariate_targets
    panel = load_wave_panel(**plan.prepare_kwargs())
    ctx.prepared = panel

    print_header(ctx)

    section_header("Build model")
    built = _factories.build_lcsm_model(
        panel,
        **plan.factory_kwargs(),
    )
    attach_built(ctx, built)

    render_model_graph(ctx)

    # Coupling parameter names mirror the factory's rule: single target keeps
    # LRP67's ``g_{src}``; multiple targets carry the target (``g_{src}_{tgt}``).
    single_target = len(couplings) == 1
    coupling_names = plan.coupling_names()
    # Lagged change-on-change names mirror the factory's rule on the lag map.
    lagged_names = plan.lagged_names()
    diag_vars = plan.diagnostic_vars()

    # One check per measure: ``y_obs`` flattens every measure into a single vector, so
    # a lone overlay would pool scales with different maxima. The headline reading
    # symbol keeps the unsuffixed filename the report partial expects.
    def _plot_prior_predictive(c: StatisticalFitContext) -> None:
        for symbol in outcomes:
            _diag.save_prior_predictive_plot(
                c,
                symbol,
                node=plan.observation_node,
                filename_stem=(
                    "prior_predictive_check"
                    if symbol == reading_symbol
                    else f"prior_predictive_check_{symbol.lower()}"
                ),
            )

    shared_stages().run_primary_fit(
        ctx,
        PrimaryFitPlan(
            diagnostic_vars=tuple(diag_vars),
            ppc_var_names=(plan.observation_node,),
            plot_prior_predictive=_plot_prior_predictive,
            extended_term=diag_vars[0] if diag_vars else None,
            compute_loo=plan.compute_loo,
        ),
    )
    _diag.save_prior_posterior_plot(ctx, var_names=diag_vars)

    # Per-target coupling table — the headline "what predicts whose change"
    # output. For LRP67 (single reading target) this reproduces the historical
    # reading-change table, labels included.
    section_header("Change-coupling summary")
    post = ctx.trace.posterior
    rows = [
        coef_row(
            f"{pname} (prior {src} -> {tgt} change)",
            post[pname].values,
            ctx.reporting.ci_prob,
        )
        for (src, tgt), pname in coupling_names.items()
    ]
    rows += [
        coef_row(
            f"{pname} (prior {src} change -> {tgt} change)",
            post[pname].values,
            ctx.reporting.ci_prob,
        )
        for (src, tgt), pname in lagged_names.items()
    ]
    for name in covariate_block:
        rows.append(
            coef_row(
                f"b_{name} ({name} -> {'/'.join(covariate_targets)} change)",
                post[f"b_{name}"].values,
                ctx.reporting.ci_prob,
            )
        )
    for tgt in couplings:
        # LRP67's historical row labels are kept verbatim ONLY for the actual
        # LRP67 shape (single word-reading target): keying on outcome_symbol
        # alone stamped "reading" on lcsm-181's taught-vocabulary rows
        # (2026-08-21 review, finding 10).
        legacy = single_target and tgt == reading_symbol and tgt == "W"
        rows.append(
            coef_row(
                f"b_self[{tgt}] (reading self-feedback)"
                if legacy
                else f"b_self[{tgt}] ({tgt} self-feedback)",
                post["b_self"].sel(outcome=tgt).values,
                ctx.reporting.ci_prob,
            )
        )
        if not arm_window:
            rows.append(
                coef_row(
                    f"a_change[{tgt}] (reading baseline change)"
                    if legacy
                    else f"a_change[{tgt}] ({tgt} baseline change)",
                    post["a_change"].sel(outcome=tgt).values,
                    ctx.reporting.ci_prob,
                )
            )
        rows.append(
            coef_row(
                f"d_age[{tgt}] (age -> reading change)"
                if legacy
                else f"d_age[{tgt}] (age -> {tgt} change)",
                post["d_age"].sel(outcome=tgt).values,
                ctx.reporting.ci_prob,
            )
        )
    coupling_df = pd.DataFrame(rows)
    save_table(ctx, "coupling_summary", coupling_df)
    print_table(
        ranked_dataframe_table(
            coupling_df,
            title=(
                f"Change couplings - {int(ctx.reporting.ci_prob * 100)}% CI "
                "(equal-tailed)"
            ),
            columns=["coefficient", "mean", "lo", "hi", "prob_pos"],
            rank_column=False,
            precision=3,
        )
    )

    # Window-1 randomised contrast on the latent change scale (immediate -
    # waitlist), the built-in consistency check against the available-case modified
    # ITT suite. Only the
    # arm x window shape carries it.
    itt_rows: list[dict] = []
    if arm_window:
        section_header(
            "Window-1 randomised contrast "
            "(available-case modified ITT consistency check)"
        )
        for s in outcomes:
            itt_rows.append(
                coef_row(
                    f"itt_w1[{s}] (immediate - waitlist, window-1 latent change)",
                    post["itt_w1_contrast"].sel(outcome=s).values,
                    ctx.reporting.ci_prob,
                )
            )
        itt_df = pd.DataFrame(itt_rows)
        save_table(ctx, "itt_window1_contrast", itt_df)
        print_table(
            ranked_dataframe_table(
                itt_df,
                title="Window-1 arm contrast (latent logit change)",
                columns=["coefficient", "mean", "lo", "hi", "prob_pos"],
                rank_column=False,
                precision=3,
            )
        )

    # Reciprocal-dominance contrast (LCSM-082): per draw, standardise each
    # direction's coupling by the model's own latent scales (g* = g *
    # sd(prior source levels) / sd(target changes)) and report |g*_AB| - |g*_BA|.
    dom_rows: list[dict] = []
    dominance_pair = plan.dominance_pair
    if dominance_pair:
        a, b = dominance_pair
        section_header(f"Reciprocal dominance: {a} <-> {b}")
        x = post["x_latent"]

        def _std_coupling(src: str, tgt: str):
            g = post[coupling_names[(src, tgt)]]
            sd_src = x.isel(wave=slice(0, -1)).sel(outcome=src).std(
                dim=("child", "wave")
            )
            sd_dt = x.sel(outcome=tgt).diff("wave").std(dim=("child", "wave"))
            return g * sd_src / sd_dt

        g_ab = _std_coupling(a, b)  # prior a -> b change
        g_ba = _std_coupling(b, a)  # prior b -> a change
        contrast = abs(g_ab) - abs(g_ba)
        dom_rows = [
            coef_row(f"std g ({a} -> {b} change)", g_ab.values, ctx.reporting.ci_prob),
            coef_row(f"std g ({b} -> {a} change)", g_ba.values, ctx.reporting.ci_prob),
            coef_row(
                f"|std g {a}->{b}| - |std g {b}->{a}| (dominance)",
                contrast.values,
                ctx.reporting.ci_prob,
            ),
        ]
        dom_df = pd.DataFrame(dom_rows)
        save_table(ctx, "dominance_summary", dom_df)
        print_table(
            ranked_dataframe_table(
                dom_df,
                title="SD-standardised reciprocal couplings",
                columns=["coefficient", "mean", "lo", "hi", "prob_pos"],
                rank_column=False,
                precision=3,
            )
        )

    # SD-standardised level -> change couplings for every target, with contrasts
    # between sources of the same target (2026-08-19): the raw couplings are per
    # unit of each source's latent logit and are not comparable in size across
    # sources. Reading-only; derivable from the stored posterior without a refit.
    section_header("Standardised level -> change couplings")
    std_df = write_standardised_couplings(ctx, post, coupling_names)
    if std_df is not None:
        print_table(
            ranked_dataframe_table(
                std_df,
                title="Standardised couplings (SD change per SD prior level)",
                columns=["coefficient", "mean", "lo", "hi", "prob_pos"],
                rank_column=False,
                precision=3,
            )
        )

    # Per-child fitted-vs-observed panels (#317 fig 2) for the focal reading target.
    write_panel_child_fit(ctx, latent_name="x_latent", focal_symbol=reading_symbol)

    write_run_metadata(
        ctx,
        extra={
            "loo_elpd": float(ctx.loo.elpd),
            "outcomes": list(outcomes),
            "reading_symbol": reading_symbol,
            "couplings": {tgt: list(srcs) for tgt, srcs in couplings.items()},
            "lagged_change_couplings": {
                tgt: list(srcs) for tgt, srcs in lagged_change_couplings.items()
            },
            "arm_window_intercepts": arm_window,
            "covariate_block": list(covariate_block),
            "covariate_targets": list(covariate_targets),
            "coupling_summary": rows,
            **({"itt_window1_contrast": itt_rows} if itt_rows else {}),
            **({"dominance_summary": dom_rows} if dom_rows else {}),
        },
    )

    return finalize_report(ctx)
