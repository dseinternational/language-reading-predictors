# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
End-to-end fit pipeline for the statistical models.

``fit_itt(spec, config)`` is the entry point for the ITT models.
``fit_joint(spec, config)`` is the entry point for the joint models (LRPITT12, LRPITT15/15b).
``fit_did(spec, config)`` is the entry point for the waitlist-crossover / DiD models.
``fit_mechanism(spec, config)`` is the entry point for LRP56/57/58 and companions.
``fit_dose_response(spec, config)`` is the entry point for LRP77 variants.
``fit_gain_factors`` / ``fit_level_factors`` / ``fit_aligned`` cover the factor
families, and ``fit_adjusted`` / ``fit_lcsm`` cover the LRP65/LRP67 companions.

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

import inspect
import os
import shutil
from collections.abc import Iterable, Sequence

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
    datasets as _datasets,
    diagnostics as _diag,
    factories as _factories,
    historical as _historical,
    priors as _priors,
    reporting as _report,
    survival as _survival,
)
from language_reading_predictors.statistical_models.plotting import (
    save_plotcollection,
    save_styled_figure,
)
from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
    make_context,
)
from language_reading_predictors.statistical_models.environment import DOCS_DIR
from language_reading_predictors.statistical_models.measures import (
    ITT_OUTCOMES,
    is_distal,
)
from language_reading_predictors.statistical_models.preprocessing import (
    load_and_prepare,
    load_and_prepare_aligned,
    load_and_prepare_lagged_outcome,
    load_longitudinal_panel,
    load_wave_panel,
    restrict_to_baseline_floored,
    restrict_to_off_floor,
    split_confounders_by_timing,
    split_covariates_by_wave,
)


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


def _default_of(fn, param: str) -> float:
    """The default value of keyword ``param`` in factory ``fn``'s signature.

    Makes the factory the single source of truth for a prior-scale default, so a
    ``spec.extra.get(param, ...)`` fallback in the pipeline cannot silently drift
    from the factory it feeds (the failure Copilot caught on #209: the adjusted
    fallback was re-hardcoded and lagged the reconciled factory default). Prefer
    this over re-typing the number: if ``param`` is ever renamed the lookup raises
    ``KeyError`` loudly at fit time rather than falling back to a stale literal.
    ``test_pipeline_fallback_defaults`` guards that this stays in step.
    """
    return inspect.signature(fn).parameters[param].default


def _raw_covariate_confounders(confounders: Iterable[str]) -> tuple[str, ...]:
    """The confounders that are raw covariates, needing ``covariates=`` loading.

    A mediation adjustment set mixes two kinds of confounder: bounded-count skill
    measures (E, R, ...) that arrive via ``prepared.pre_logit`` (they are in
    ``ITT_OUTCOMES`` or ``spec.extra['outcomes']``), and revised-DAG raw covariates
    (hearing ``hs``/``hs_missing``, speech ``deapp_c``, phonological memory
    ``erbto`` + missing indicators; #246) that must be requested as ``covariates``.
    A symbol is a raw covariate exactly when it is not a bounded-count measure.
    """
    from language_reading_predictors.statistical_models.measures import MEASURES

    return tuple(c for c in confounders if c not in MEASURES)


def _require_spec(
    spec: ModelSpec,
    kind: str,
    *,
    outcome: bool = False,
    mechanism: bool = False,
) -> None:
    """Validate model specs at runtime; unlike ``assert``, this is never optimised away."""
    if spec.kind != kind:
        msg = f"{spec.model_id}: expected kind {kind!r}, got {spec.kind!r}"
        raise ValueError(msg)
    if outcome and spec.outcome_symbol is None:
        msg = f"{spec.model_id}: outcome_symbol is required for {kind!r} models"
        raise ValueError(msg)
    if mechanism and spec.mechanism_symbol is None:
        msg = f"{spec.model_id}: mechanism_symbol is required for {kind!r} models"
        raise ValueError(msg)


def _effective_adjustment(
    spec: ModelSpec,
    prepared,
    *,
    measure_confounders: tuple[str, ...] = (),
    adjust_for: tuple[str, ...] = (),
    ability_covariate: str | None = None,
    baseline_symbol: str | None = None,
    skill_baselines: tuple[str, ...] = (),
) -> dict:
    """Describe the adjustment set the model **actually fitted**.

    ``spec.adjustment`` records what was *requested*; it is not what is fitted.
    ``ModelSpec.extra["adjust_for"]`` never reached ``config.json`` at all, so a
    model could report ``{G, A, W_pre}`` while conditioning on hearing, speech,
    sessions and their missingness indicators — a material misdescription that made
    exact auditing impossible (#258 review, P1). And a covariate that turns out
    constant on the fitted rows is dropped by the loader and gets no coefficient, so
    listing it would imply a term that was never estimated.

    The returned record therefore names, for every term that carries a coefficient,
    its source column, its measurement wave, and whether it is a missingness
    indicator — plus the requested-but-dropped terms, explicitly.

    ``skill_baselines`` records the gain-factor ``skill_symbols``, which — unlike the
    ``measure_confounders`` of the mechanism/mediation families — enter at the period
    **pre** (baseline) wave, not the post wave (#247). They are always fitted (the
    keep-mask requires their baselines), so they never appear in ``dropped_constant``.

    ``ability_covariate`` records the gain-/level-factor cognitive-ability adjuster
    (block design), a between-child t1 baseline broadcast across the panel and fitted
    as ``gamma_ability``. It was previously absent from the record even though the
    factory conditions on it, so the audited set understated the fitted set by one
    term across the whole factor family (this review's finding B2).
    """
    terms = []
    for s in skill_baselines:
        # Upstream-skill DAG-parent adjusters, entered as their period baseline
        # (pre-wave) logit — the ANCOVA lag that precedes that period's treatment.
        terms.append(
            {
                "term": f"{s}_pre",
                "kind": "measure_baseline",
                "source_column": prepared.column_map.get(s, s),
                "wave": "pre",
                "missing_indicator": False,
            }
        )
    for s in measure_confounders:
        if s == "G":
            # The randomised arm: time-invariant, not a wave-indexed measurement.
            terms.append(
                {
                    "term": "G",
                    "kind": "treatment",
                    "source_column": "group",
                    "wave": "time_invariant",
                    "missing_indicator": False,
                }
            )
        elif s == "A":
            # Age is read from the transition's pre row (age at the start of it).
            terms.append(
                {
                    "term": "A",
                    "kind": "covariate",
                    "source_column": "age",
                    "wave": "pre",
                    "missing_indicator": False,
                }
            )
        else:
            # Bounded-count measure confounders are taken at the POST wave,
            # contemporaneous with the exposure and the outcome.
            terms.append(
                {
                    "term": s,
                    "kind": "measure",
                    "source_column": prepared.column_map.get(s, s),
                    "wave": "post",
                    "missing_indicator": False,
                }
            )
    for c in adjust_for:
        terms.append(
            {
                "term": c,
                "kind": "covariate",
                "source_column": c,
                "wave": prepared.covariate_time.get(c, "unknown"),
                "missing_indicator": c.endswith("_missing"),
            }
        )
    if ability_covariate and ability_covariate in prepared.covariates:
        # Cognitive-ability (block-design) adjuster — a between-child t1 baseline
        # broadcast across the panel, fitted as ``gamma_ability``. Guarded on
        # presence so an ability covariate that went constant (and was dropped by
        # the loader) is reported under ``dropped_constant``, not as fitted.
        terms.append(
            {
                "term": ability_covariate,
                "kind": "ability_covariate",
                "source_column": ability_covariate,
                "wave": prepared.covariate_time.get(ability_covariate, "baseline"),
                "missing_indicator": False,
            }
        )
    if baseline_symbol:
        terms.append(
            {
                "term": f"{baseline_symbol}_pre",
                "kind": "autoregressive_baseline",
                "source_column": prepared.column_map.get(baseline_symbol, baseline_symbol),
                "wave": "pre",
                "missing_indicator": False,
            }
        )
    return {
        "requested": list(spec.adjustment)
        + list(skill_baselines)
        + ([ability_covariate] if ability_covariate else [])
        + list(spec.extra.get("adjust_for", ())),
        "fitted": terms,
        "dropped_constant": list(prepared.dropped_covariates),
    }


def _apply_spec_target_accept(ctx: StatisticalFitContext, spec: ModelSpec) -> None:
    """Apply a model-specific ``spec.extra['target_accept']`` with explicit precedence.

    Precedence is **CLI override > model-specific default > config preset**.

    Some models (the horseshoe's global-local funnel, the small-n correlated-factor
    CFA) need a higher ``target_accept`` than their tier preset gives, and declare
    it in ``spec.extra``. That must not silently outrank an explicit
    ``--target-accept`` from the command line: the previous
    ``max(preset_or_cli, spec_value)`` meant a deliberate ``--target-accept 0.95``
    was replaced by a spec's 0.999, so a diagnostic reproduction or an ablation
    silently did not run at the requested setting. ``scripts/fit_statistical_model``
    flags a CLI override on the sampling config; when that flag is set the spec
    value is ignored.
    """
    target_accept = spec.extra.get("target_accept")
    if target_accept is None:
        return
    target_accept = float(target_accept)
    if not 0.0 < target_accept < 1.0:
        raise ValueError(
            "spec.extra['target_accept'] must be in the open interval (0, 1); "
            f"got {target_accept!r}"
        )
    if getattr(ctx.sampling, "target_accept_overridden", False):
        rprint(
            "[yellow]Keeping the CLI --target-accept "
            f"({ctx.sampling.target_accept}) over {spec.model_id}'s "
            f"spec default ({target_accept}).[/yellow]"
        )
        return
    # No CLI override: the model-specific value takes precedence over the preset.
    ctx.sampling.target_accept = target_accept


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
        if os.environ.get("LRP_OFFLINE_QUARTO") == "1":
            _strip_quarto_code_links(dst)
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


def _strip_quarto_code_links(path: str) -> None:
    """Remove copied ``code-links: repo`` metadata for offline Quarto renders.

    Quarto resolves ``code-links: repo`` by probing the GitHub remote, including
    ``git ls-remote origin gh-pages``. In restricted reporting environments that
    optional link can make an otherwise valid report render fail after all cells
    execute. The source templates are left intact; only the copied output QMD is
    made renderable when ``LRP_OFFLINE_QUARTO=1`` is set.
    """
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    text = text.replace("    code-links:\n      - repo\n", "")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


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
    ctor_overrides, role_overrides, rationale_overrides = _prior_table_overrides(context)
    _priors.save_shared_prior_panel(
        context.output_dir,
        used=_priors.used_prior_keys(model, ctor_overrides=ctor_overrides),
    )
    table = _priors.priors_table(
        model,
        ctor_overrides=ctor_overrides,
        role_overrides=role_overrides,
        rationale_overrides=rationale_overrides,
    )
    table.to_csv(os.path.join(context.output_dir, "priors_table.csv"), index=False)
    context.tables["priors_table"] = table


def _prior_table_overrides(
    context: StatisticalFitContext,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Context-specific prior-table corrections for reused RV names.

    Some factories reuse a PyMC variable name with a different prior constructor
    or a different reporting role. Keep the model code stable and teach the
    artifact writer about those contextual meanings here.
    """
    ctor: dict[str, str] = {}
    role: dict[str, str] = {}
    rationale: dict[str, str] = {}
    spec = context.spec

    if spec.kind == "dose_response":
        ctor.update(
            {
                "beta_dose": "beta_mech",
                "beta_dose_phase": "beta_mech",
                "mu_dose": "beta_mech",
                "sigma_dose": "sigma_dose",
            }
        )
        role.update(
            {
                "beta_dose": "association",
                "beta_dose_phase": "association",
                "mu_dose": "association",
                "sigma_dose": "nuisance",
                "beta_G": "association",
            }
        )
    elif spec.kind == "did":
        # ``beta_period`` reuses the ``tau`` constructor (its Normal(0, 0.5) scale)
        # but it is the common time/maturation anchor (the immediate arm's P1→P2
        # trend), NOT the randomised within-person effect — that is ``delta``.
        # Report it as an adjusted association, never "causal".
        role["beta_period"] = "association"
        if spec.extra.get("dose", False):
            # Dose slopes now share build_dose_response_model's ``beta_mech`` prior
            # (Normal(0, 1)) so the shared summary compares like with like.
            if spec.extra.get("period_varying_dose", False):
                ctor.update(
                    {
                        "mu_dose": "beta_mech",
                        "beta_dose_phase": "beta_mech",
                        "sigma_dose": "sigma_dose",
                    }
                )
                role.update(
                    {
                        "mu_dose": "association",
                        "beta_dose_phase": "association",
                        "sigma_dose": "nuisance",
                    }
                )
            else:
                ctor["beta_dose"] = "beta_mech"
                role["beta_dose"] = "association"
    elif spec.kind in ("mediation", "mediation_multi"):
        # The mediation coefficients ``a_G`` (group→mediator) and ``b_G``
        # (group→outcome direct path) reuse the ``tau`` constructor's scale but
        # are structural building blocks of the g-formula, not the reported
        # estimand: the NDE/NIE come from the counterfactual simulation
        # (``mediation_summary.csv``), never a raw coefficient. Label them adjusted
        # associations so the prior table does not imply a bare coefficient is the
        # reported quantity. The simulated NDE/NIE are **not** causal either: they
        # are not identified natural effects (latent GA confounds the
        # mediator->outcome path, and dose ``IS`` is a treatment-induced
        # mediator-outcome confounder). See the :mod:`mediation` module docstring.
        role["a_G"] = "association"
        role["b_G"] = "association"
        # B3 (review 2026-07-13): in the SINGLE-mediator outcome leg every confounder
        # coefficient is ``b_{symbol}`` built from gamma_cross_prior (Normal(0, 0.3)).
        # But ``b_E``/``b_B`` are *also* globally mapped to the ``b_path`` constructor
        # (Normal(0, 1)) for the TWO-mediator models, where E/B are the mediators —
        # so in LRP59/62/78 (E a confounder) that global mapping mislabels the prior
        # table's rationale + panel (the distribution column, read off the RV, stays
        # correct). Route every confounder ``b_X`` (X not a structural term) to
        # gamma_cross. ``mediation_multi`` is left alone: there b_L/b_E or b_L/b_B ARE
        # the mediator b-paths, and its other confounders already route to gamma_cross
        # via the RV-distribution fallback.
        if spec.kind == "mediation" and context.model is not None:
            _structural_b = {"M", "G", "GM", "W", "A"}
            for rv in context.model.free_RVs:
                if rv.name.startswith("b_") and rv.name[2:] not in _structural_b:
                    ctor[rv.name] = "gamma_cross"
    elif spec.kind == "mechanism":
        # ``beta_G`` reuses the tau constructor (its Normal(0, 0.5) scale) but here
        # it is the group main effect entered as a DAG backdoor adjustment, not the
        # randomised ITT effect — an adjusted association, not a causal term.
        role["beta_G"] = "association"
    elif spec.kind == "aligned":
        ctor["beta_cohort"] = "tau"
        role["beta_cohort"] = "association"
    elif spec.kind == "adjusted" and context.model is not None:
        for rv in context.model.free_RVs:
            if rv.name.startswith("beta_"):
                ctor[rv.name] = "predictor_slope"
                role[rv.name] = "association"
    elif spec.kind == "growth":
        # Baseline non-verbal ability -> trajectory shape (gamma on the growth rate,
        # delta on the baseline level): adjusted, latent-GA-confounded associations,
        # never causal — routed to the predictor-slope panel / association role.
        for _rv in ("gamma", "delta"):
            ctor[_rv] = "predictor_slope"
            role[_rv] = "association"
    elif spec.kind == "level_factors" and spec.extra.get("group_by_time", True):
        # The prior table is one row per RV, while ``b_grp_time`` is a vector whose
        # elements have different interpretation: only b_grp_time[1] is the clean
        # randomised t2 contrast. Keep the vector row conservative and let
        # factor_summary.csv carry the element-level causal label.
        role["b_grp_time"] = "association"
        rationale["b_grp_time"] = (
            "Level-model group-by-time vector; only b_grp_time[1] is the "
            "randomised t2 contrast, while the vector row is documented "
            "conservatively because other elements are pre-randomisation or "
            "post-crossover associations."
        )

    # Distal outcomes take the tighter tau prior (issue #141): the factory built
    # the single-outcome causal treatment term at Normal(0, 0.3), so route it to
    # the ``tau_distal`` panel + distribution here so the report panel matches the
    # fitted scale. Only the randomised treatment terms are listed (never the
    # adjusted-association ``beta_G`` / ``beta_cohort``).
    if is_distal(getattr(spec, "outcome_symbol", None)):
        for _name in ("tau", "beta_trt", "b_grp_time", "beta_grp", "delta"):
            ctor.setdefault(_name, "tau_distal")
            role.setdefault(_name, "causal")
        # The ANCOVA intercept is likewise tiered for distal outcomes (Normal(0,
        # 1.0); prior-critical-review 2026-07-07, Finding 1). Route it to the
        # ``alpha_distal`` panel so the report rationale matches the fitted scale
        # (the distribution column already reads the true 1.0 off the built RV).
        ctor.setdefault("alpha", "alpha_distal")

    return ctor, role, rationale


def _render_model_graph(context: StatisticalFitContext) -> None:
    try:
        g = _graphviz(context.model)
        g.render(
            filename=os.path.join(context.output_dir, "model_graph"),
            format="png",
            cleanup=True,
        )
    except Exception as exc:  # pragma: no cover
        rprint(f"[yellow]Graphviz render failed: {exc}[/yellow]")


def _graphviz(model):
    import pymc as pm

    g = pm.model_to_graphviz(model)
    g.graph_attr["fontname"] = "Helvetica"
    # Raster PNG output (not SVG): the DAG's many nodes/edges make a large SVG
    # slow to browse, so render to PNG and bump DPI to keep the lightbox legible.
    g.graph_attr["dpi"] = "150"
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
        save_plotcollection(
            pc,
            context.output_dir,
            "posterior_predictive_check.png",
            suptitle="Posterior-predictive vs observed",
        )
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
        # Scalar PPC summary (rep excluded) is already written to CSV by the
        # graded/floor path, so no data= here — just the styled PNG + SVG.
        save_styled_figure(ctx.output_dir, "proportion_at_zero_ppc")
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
    moderators: Sequence[tuple[str, np.ndarray]] | None = None,
    items: np.ndarray | None = None,
    row_mask: np.ndarray | None = None,
) -> None:
    """ROPE-anchored figure for a randomised effect: the items-scale posterior with
    the region of practical equivalence, and ``P(effect > delta)`` as the
    minimally-important difference rises. Single-outcome version of the note figure
    (notes/202606261304-evidence-strength-and-rope-reporting.md).

    The ITT/gain path recomputes the items draws from ``_itt_ame_draws`` (``term`` /
    ``varying_term`` / ``moderators`` / ``G`` select the effect, including any
    treatment interactions); the level family passes its t2 contrast items draws
    directly via ``items`` (its AME nets out a group×ability interaction the generic
    core cannot reconstruct).
    """
    try:
        from scipy.stats import gaussian_kde

        if items is None:
            _, ame_prob = _report._itt_ame_draws(
                ctx.trace, G=G, term=term, varying_term=varying_term,
                moderators=moderators, row_mask=row_mask,
            )
            items = ame_prob * float(n_trials)
        med = float(np.median(items))
        risk_difference = n_trials == 1 and delta <= 1
        effect_label = (
            "treatment effect (risk difference)"
            if risk_difference
            else "treatment effect (extra items correct)"
        )
        delta_label = (
            "minimally-important difference, delta (probability)"
            if risk_difference
            else "minimally-important difference, delta (items)"
        )
        scale_title = "risk-difference scale" if risk_difference else "items scale"
        xmax = float(np.quantile(items, 0.995)) + 0.5
        xmin = min(-delta - 0.5, float(np.quantile(items, 0.005)))
        xs = np.linspace(xmin, xmax, 300)
        kde = gaussian_kde(items)

        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.2))
        ax_l.axvspan(
            -delta, delta, color="#bdbdbd", alpha=0.30,
            label=f"ROPE (within ±{delta:g})",
        )
        ax_l.axvline(0, color="#444444", lw=1.0, ls=":")
        ax_l.plot(xs, kde(xs), color="#1b7837", lw=2.2)
        ax_l.fill_between(xs, kde(xs), color="#1b7837", alpha=0.12)
        ax_l.axvline(med, color="#1b7837", lw=1.2, label=f"median {med:+.1f}")
        ax_l.set_xlabel(effect_label)
        ax_l.set_ylabel("posterior density")
        ax_l.set_title(f"{symbol}: effect on the {scale_title}, with ROPE")
        ax_l.legend(fontsize=8, frameon=False)

        dgrid = np.linspace(0.0, max(xmax, delta + 0.5), 200)
        pex = np.array([float((items > d).mean()) for d in dgrid])
        ax_r.plot(dgrid, pex, color="#2166ac", lw=2.2)
        ax_r.axvline(delta, color="#888888", lw=1.0, ls="--", label=f"delta = {delta:g}")
        ax_r.axhline(0.975, color="#cccccc", lw=1.0, ls=":")
        ax_r.set_ylim(0, 1.02)
        ax_r.set_xlabel(delta_label)
        ax_r.set_ylabel("P(effect > delta)")
        ax_r.set_title("Probability of a meaningful benefit")
        ax_r.legend(fontsize=8, frameon=False)

        for ax in (ax_l, ax_r):
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        fig.tight_layout()
        # rope_summary.csv is written by the ITT/factor result path, so no data= here.
        save_styled_figure(ctx.output_dir, "rope_summary", fig=fig)
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
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("P(row tau > column tau)", fontsize=8)
        fig.tight_layout()
        save_styled_figure(ctx.output_dir, "contrast_heatmap", fig=fig)
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
    moderators: Sequence[tuple[str, np.ndarray]] | None = None,
) -> None:
    """Area 1/4 extras for an ITT-style fit (issue #125).

    Writes ``prior_pushforward.csv`` (the estimand-scale prior check), the causal
    forest, the prior-vs-posterior overlay, and power-scaling sensitivity. Reads
    the persisted ``prior`` group (on ``ctx.prior_samples``) and the full trace,
    so call after ``save_trace``. ``n_trials=1`` gives the risk-difference scale
    for the binary off-floor model. ``moderators`` carries any treatment
    interactions so the prior is pushed through the same full-contribution AME.
    """
    try:
        pf = _report.prior_pushforward(
            ctx.prior_samples,
            G=built.prepared.G,
            n_trials=n_trials,
            term=term,
            varying_term=varying_term,
            moderators=moderators,
            ci_prob=ctx.reporting.ci_prob,
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
    title: str | None = None,
) -> None:
    """Forest plot of the causal term(s) with a reference line at 0 (#125 Area 4).

    For a single-outcome model ``var_names=["tau"]`` shows the one effect; for the
    joint model the vector ``tau`` forests every outcome's effect in one panel —
    the single most communicative artifact for the suite. Guarded.
    """
    try:
        import arviz_plots as azp

        tr = _diag.thin_for_plots(ctx.trace)
        # Equal-tailed nested bands (#177): inner central 50% + outer equal-tailed
        # 95% headline, matching the reported interval convention rather than the
        # arviz default (which can be an HDI, inconsistent with the prose).
        pc = azp.plot_forest(
            tr,
            var_names=var_names,
            combined=True,
            ci_kind=ctx.reporting.interval_kind,
            ci_probs=(0.5, 0.95),
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


# ---------------------------------------------------------------------------
# Shared pipeline phases (#82)
#
# Every fit_* pipeline runs the same scaffold: prepare -> build -> attach ->
# prior predictive -> sample -> LOO -> summary -> posterior predictive ->
# (model-specific summaries) -> metadata -> report. The phases that are
# byte-identical across pipelines live here so a fix to one (the LOO sequence,
# the PPC draw, the report tail) propagates to every model instead of drifting
# per-pipeline (the failure mode behind #78). The genuinely per-model phases
# (prepare, build, summary var_names, the headline summary tables) stay inline
# in each fit_* function.
# ---------------------------------------------------------------------------


def _attach_built(ctx: StatisticalFitContext, built) -> None:
    """Attach a freshly built model and emit its prior artifacts."""
    ctx.model = built.model
    ctx.model_vars = built.variables
    ctx.prepared = built.prepared
    _emit_priors(ctx)


def _run_sampling_and_loo(
    ctx: StatisticalFitContext, *, compute_loo: bool = True
) -> None:
    """Posterior sampling, then (optionally) LOO-PSIS + its summary and console row.

    ``compute_loo=False`` skips the LOO phase for the pipelines that do not
    report it (the mediation g-formula fits).
    """
    section_header("Sampling posterior (nutpie)")
    _diag.sample_posterior(ctx)

    if compute_loo:
        section_header("LOO-PSIS")
        _diag.compute_log_likelihood_and_loo(ctx)
        _report.write_loo_summary(ctx)
        _print_loo_row(ctx)


def _run_ppc(ctx: StatisticalFitContext, *, var_names: list[str] | None = None) -> None:
    """Posterior-predictive draw (defaults to ``y_post``) followed by the PPC plot."""
    section_header("Posterior predictive")
    _diag.sample_posterior_predictive(ctx, var_names=var_names or ["y_post"])
    _save_ppc(ctx)


def _finalize_report(ctx: StatisticalFitContext) -> StatisticalFitContext:
    """Copy the Quarto report template, print the footer, and return the context."""
    section_header("Report")
    _copy_report_template(ctx)
    _print_footer(ctx)
    return ctx


def _survival_summary(
    trace, *, ci_prob: float, hazard_link: str, use_treatment: bool
) -> pd.DataFrame:
    """Off-floor discrete-time hazard summary (log-hazard, hazard ratio, P>0).

    Reports the treatment hazard shift and baseline-covariate slopes on the
    log-hazard scale (with ``exp`` as a hazard ratio and ``P(effect > 0)``), plus
    the per-interval baseline off-floor probability for an untreated child at mean
    covariates, on the model's ``hazard_link`` scale. Equal-tailed intervals at
    ``ci_prob`` with the posterior median as the point estimate (the suite convention).
    """
    post = trace.posterior
    lo, hi = (1 - ci_prob) / 2, 1 - (1 - ci_prob) / 2

    def _row(term: str, draws: np.ndarray, *, as_ratio: bool) -> dict:
        d = np.asarray(draws).reshape(-1)
        return {
            "term": term,
            "median": float(np.median(d)),
            "ci_low": float(np.quantile(d, lo)),
            "ci_high": float(np.quantile(d, hi)),
            "hazard_ratio": float(np.exp(np.median(d))) if as_ratio else float("nan"),
            "P(>0)": float(np.mean(d > 0)) if as_ratio else float("nan"),
        }

    rows: list[dict] = []
    if use_treatment:
        rows.append(_row("tau (log hazard shift, treated)", post["tau"].values, as_ratio=True))
    for name in sorted(v for v in post.data_vars if str(v).startswith("beta_")):
        rows.append(_row(f"{name} (log hazard, per SD)", post[name].values, as_ratio=True))

    alpha = post["alpha"].stack(sample=("chain", "draw")).transpose("interval", "sample")
    labels = [str(v) for v in alpha.coords["interval"].values]
    for i, lab in enumerate(labels):
        a = alpha.values[i]
        base = 1.0 - np.exp(-np.exp(a)) if hazard_link == "cloglog" else 1.0 / (1.0 + np.exp(-a))
        rows.append(
            {
                "term": f"baseline off-floor prob [{lab}]",
                "median": float(np.median(base)),
                "ci_low": float(np.quantile(base, lo)),
                "ci_high": float(np.quantile(base, hi)),
                "hazard_ratio": float("nan"),
                "P(>0)": float("nan"),
            }
        )
    return pd.DataFrame(rows)


def fit_survival(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Discrete-time off-floor survival fit for a floored outcome P / N (#230 §5).

    Fits a person-period discrete-time hazard for the *time* to come off the floor,
    generalising the single-transition off-floor estimand of the LRPITT09/11 floor
    rule to all four waves. Treatment enters as an intervention-aligned hazard shift;
    the estimand is prognostic (both arms are treated by t4).
    """
    _require_spec(spec, "survival", outcome=True)
    ctx = make_context(spec, config)

    section_header("Prepare data")
    panel = _survival.prepare_survival(spec.outcome_symbol)
    ctx.prepared = panel
    _print_header(ctx)
    rprint(
        f"  Survival at-risk set: {panel.n_at_risk_children} children at the "
        f"{spec.outcome_symbol} floor at t1 contribute {panel.n_obs} person-period rows; "
        f"{panel.n_events} off-floor events."
    )
    if panel.dropped_rows:
        rprint(
            f"  [yellow]{panel.dropped_rows} at-risk child(ren) contributed no rows "
            "(t2 score unobserved, so no interval could be placed) and are excluded.[/yellow]"
        )
    for name, k in panel.imputed_covariate_rows.items():
        if k:
            rprint(
                f"  [yellow]{k} row(s) had a missing baseline {name}; mean-imputed (z=0).[/yellow]"
            )

    hazard_link = spec.extra.get("hazard_link", "cloglog")
    use_treatment = bool(spec.extra.get("use_treatment", True))

    section_header("Build model")
    built = _survival.build_survival_model(
        panel, hazard_link=hazard_link, use_treatment=use_treatment
    )
    _attach_built(ctx, built)
    _render_model_graph(ctx)

    diag_vars = (
        ["alpha"]
        + [f"beta_{n}" for n in panel.covariates]
        + (["tau"] if use_treatment else [])
    )

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)

    _run_sampling_and_loo(ctx)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=diag_vars)

    _run_ppc(ctx, var_names=["y_event"])

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=diag_vars)
    if use_treatment:
        _diag.run_extended_diagnostics(ctx, causal_term="tau")
    _diag.save_trace(ctx)

    section_header("Off-floor hazard summary")
    summary = _survival_summary(
        ctx.trace, ci_prob=ctx.reporting.ci_prob, hazard_link=hazard_link,
        use_treatment=use_treatment,
    )
    summary.to_csv(os.path.join(ctx.output_dir, "survival_summary.csv"), index=False)
    ctx.tables["survival_summary"] = summary
    print_table(
        ranked_dataframe_table(
            summary,
            title=(
                f"Off-floor discrete-time hazard ({spec.outcome_symbol}, {hazard_link}); "
                "positive = raises Pr(off-floor); prognostic, not a randomised effect"
            ),
            columns=list(summary.columns),
            rank_column=False,
            precision=3,
        )
    )

    _report.write_run_metadata(
        ctx,
        extra={
            "loo_elpd": float(ctx.loo.elpd) if ctx.loo is not None else None,
            "n_at_risk_children": panel.n_at_risk_children,
            "n_events": panel.n_events,
            "hazard_link": hazard_link,
        },
    )

    return _finalize_report(ctx)


def fit_itt(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    _require_spec(spec, "itt", outcome=True)

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
    # Re-filter the requested adjusters to those actually present after loading: a
    # covariate (or a ``{col}_missing`` indicator) that goes constant on the fitted
    # rows is dropped by the loader, and build_itt_model raises on an absent adjuster.
    # Mirror the gain-/level-factor and mechanism pipelines, which drop-and-continue so
    # a dropped-constant adjuster does not abort the whole ITT fit (Group-C cleanup).
    adjust_for = tuple(c for c in adjust_for if c in prepared.covariates)

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
    _attach_built(ctx, built)

    _render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)

    _run_sampling_and_loo(ctx)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=_itt_diag_vars(spec, adjust_for))

    _run_ppc(ctx)

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=_itt_diag_vars(spec, adjust_for))
    _diag.save_prior_predictive_plot(ctx, spec.outcome_symbol)
    _diag.run_extended_diagnostics(ctx, causal_term="tau")

    _diag.save_trace(ctx)

    # Area 1/4 extras that read the attached prior group or the full trace:
    # the prior pushforward to the items scale (estimand-scale prior check), the
    # tau forest, the prior-vs-posterior overlay, and power-scaling sensitivity.
    # Net out the full per-row treatment contribution: the age-varying ``tau_i``
    # is picked up automatically by the AME core; a linear tau moderator adds
    # ``gamma_tau_int·z_M`` (Part B). Latent today — no registered ITT spec sets
    # ``tau_moderator_symbol`` — but wired so a heterogeneity fit reports the
    # model-implied effect, not ``tau`` alone.
    tau_moderators = built.extras.get("tau_interaction_moderators", [])
    n_trials_own = int(built.prepared.n_trials[spec.outcome_symbol])
    _emit_itt_extras(
        ctx, built, n_trials=n_trials_own,
        overlay_vars=_itt_diag_vars(spec, adjust_for),
        moderators=tau_moderators,
    )

    # Treatment-effect summary on both scales.
    section_header("Treatment-effect summary")
    tau_s = _report.tau_summary_itt(
        ctx.trace,
        ci_prob=ctx.reporting.ci_prob,
        # built.prepared is the (possibly row-subset) frame the model was fit
        # on, so G aligns with eta's obs_id axis (finding #2 in issue #78).
        G=built.prepared.G,
        moderators=tau_moderators,
    )
    tau_df = pd.DataFrame([tau_s])
    tau_df.to_csv(os.path.join(ctx.output_dir, "tau_summary.csv"), index=False)
    ctx.tables["tau_summary"] = tau_df
    print_table(
        metrics_table(
            [{"metric": k, "value": v} for k, v in tau_s.items()],
            title=f"tau ({spec.outcome_symbol}) - {int(ctx.reporting.ci_prob * 100)}% CI (equal-tailed)",
            columns=["metric", "value"],
        )
    )

    # ROPE-anchored continuous summary on the items scale
    # (notes/202606261304-evidence-strength-and-rope-reporting.md). Emitted for
    # graded outcomes with an agreed minimally-important difference (delta);
    # floored outcomes (P/N) take the floor-rule path and a probability-scale delta.
    from language_reading_predictors.statistical_models.measures import (
        ROPE_DELTA,
        rope_delta_grid,
    )

    delta_items = ROPE_DELTA.get(spec.outcome_symbol)
    if delta_items is not None:
        rope_s = _report.rope_summary(
            ctx.trace,
            G=built.prepared.G,
            n_trials=int(built.prepared.n_trials[spec.outcome_symbol]),
            delta=delta_items,
            ci_prob=ctx.reporting.ci_prob,
            moderators=tau_moderators,
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
            moderators=tau_moderators,
        )

        # δ-sensitivity sweep (issue #144): P(benefit ≥ δ) at the adopted δ and a
        # stricter 2·δ (word reading at δ = 1 and 2), for every graded outcome.
        sens_df = _report.rope_sensitivity(
            ctx.trace,
            G=built.prepared.G,
            n_trials=int(built.prepared.n_trials[spec.outcome_symbol]),
            deltas=rope_delta_grid(spec.outcome_symbol),
            moderators=tau_moderators,
        )
        sens_df.to_csv(
            os.path.join(ctx.output_dir, "rope_sensitivity.csv"), index=False
        )
        ctx.tables["rope_sensitivity"] = sens_df

    # Tau-moderator (Part B / HTE) summary: the effect-modification coefficient
    # gamma_tau_int and the moderator main effect gamma_tau_mod, when a linear
    # tau moderator was fit. Returns {} (nothing written) for the standard
    # main-effect ITT models, so this is a no-op unless the moderator is present.
    tau_mod_s = _report.tau_moderation_summary(ctx.trace, ci_prob=ctx.reporting.ci_prob)
    if tau_mod_s:
        tau_mod_df = pd.DataFrame([tau_mod_s])
        tau_mod_df.to_csv(
            os.path.join(ctx.output_dir, "tau_moderation_summary.csv"), index=False
        )
        ctx.tables["tau_moderation_summary"] = tau_mod_df

    _report.write_run_metadata(
        ctx,
        extra={
            "loo_elpd": float(ctx.loo.elpd),
            "tau_summary": tau_s,
            "adjust_for": list(adjust_for),
        },
    )

    return _finalize_report(ctx)


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
    # Restrict the PRIMARY to the baseline-floored at-risk subset (pre == 0 at t1),
    # so the estimand is the pre-specified off-floor TRANSITION Pr(post > 0 | pre == 0)
    # rather than off-floor prevalence over everyone (issue #267 / #119). pre == 0 is
    # pre-randomisation, so this is a legitimate subgroup ITT.
    at_risk = restrict_to_baseline_floored(prepared, own)
    # Guard: the subgroup ITT is only identified if the at-risk subset keeps both
    # arms and enough rows. If a future floored outcome had (say) all baseline-floored
    # children in one arm, tau would be unidentified and the PRIMARY posterior
    # degenerate — fail loudly rather than publish it (issue #267 review).
    _n_arms = int(np.unique(at_risk.G).size)
    if at_risk.n_obs < 10 or _n_arms < 2:
        raise ValueError(
            f"floor rule for {own!r}: the baseline-floored at-risk subset is "
            f"degenerate (n={at_risk.n_obs}, arms present={_n_arms}) — the subgroup "
            "ITT Pr(post>0 | pre==0) is not identified. Re-check the floor rule / "
            "data or fit a different estimand."
        )
    rprint(
        f"  Floor rule: {own} is {p0:.0%} floored at t2 "
        f"(>= {_floor.FLOOR_THRESHOLD:.0%}); PRIMARY is the off-floor TRANSITION "
        f"Pr(off-floor at t2 | at floor at t1) on the {at_risk.n_obs} baseline-floored "
        f"children (of {prepared.n_obs}); a graded Beta-Binomial over all children "
        "and a graded contrast among off-floor children are flagged SECONDARIES."
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

    # ----- PRIMARY: binary off-floor TRANSITION (Bernoulli on post > 0 | pre == 0) -----
    section_header("Build model (PRIMARY: off-floor transition among baseline-floored)")
    built = _factories.build_itt_model(
        at_risk, likelihood="bernoulli_offfloor", **common
    )
    _attach_built(ctx, built)
    _render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    _diag.save_prior_predictive_plot(ctx, spec.outcome_symbol or "W")
    _run_sampling_and_loo(ctx)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(
        ctx,
        var_names=_itt_diag_vars(spec, adjust_for, likelihood="bernoulli_offfloor"),
    )

    _run_ppc(ctx, var_names=["y_offfloor"])

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(
        ctx,
        var_names=_itt_diag_vars(spec, adjust_for, likelihood="bernoulli_offfloor"),
    )
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
        ctx.trace, ci_prob=ctx.reporting.ci_prob, G=built.prepared.G
    )
    pd.DataFrame([off]).to_csv(
        os.path.join(ctx.output_dir, "tau_summary.csv"), index=False
    )
    ctx.tables["tau_summary"] = pd.DataFrame([off])
    print_table(
        metrics_table(
            [{"metric": k, "value": v} for k, v in off.items()],
            title=(
                f"off-floor transition tau ({own}, baseline-floored at-risk) - "
                f"{int(ctx.reporting.ci_prob * 100)}% CI (equal-tailed); positive = "
                "intervention raises Pr(off-floor at t2 | at floor at t1)"
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
    # 10 pp value is confirmed by the education lead (2026-07-01, issue #144).
    from language_reading_predictors.statistical_models.measures import (
        ROPE_DELTA_PROB,
        ROPE_DELTA_PROB_GRID,
    )

    delta_prob = ROPE_DELTA_PROB.get(own)
    if delta_prob is not None:
        rope_s = _report.rope_summary(
            ctx.trace,
            G=built.prepared.G,
            n_trials=1,
            delta=delta_prob,
            ci_prob=ctx.reporting.ci_prob,
            varying_term="",
        )
        rope_s["provisional_delta"] = False  # 10 pp signed off (#144, 2026-07-01)
        rope_s["delta_scale"] = "risk_difference"
        pd.DataFrame([rope_s]).to_csv(
            os.path.join(ctx.output_dir, "rope_summary.csv"), index=False
        )
        ctx.tables["rope_summary"] = pd.DataFrame([rope_s])
        _save_rope_plot(
            ctx, own, built.prepared.G, 1, delta_prob, varying_term=""
        )

        # δ-sensitivity sweep on the risk-difference scale (issue #144): 10/15/20 pp.
        sens_df = _report.rope_sensitivity(
            ctx.trace,
            G=built.prepared.G,
            n_trials=1,
            deltas=ROPE_DELTA_PROB_GRID,
            varying_term="",
        )
        sens_df.to_csv(
            os.path.join(ctx.output_dir, "rope_sensitivity.csv"), index=False
        )
        ctx.tables["rope_sensitivity"] = sens_df

    s = ctx.sampling

    def _fit_secondary(built_x, *, label: str):
        with built_x.model:
            tr = pm.sample(
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
            tr = pm.sample_posterior_predictive(
                tr,
                var_names=["y_post"],
                extend_inferencedata=True,
                random_seed=s.random_seed,
                progressbar=False,
            )
        conv = _diag.subfit_convergence(tr, label=label, var_names=["tau"])
        summ = _report.tau_summary_itt(tr, ci_prob=ctx.reporting.ci_prob, G=built_x.prepared.G)
        summ["converged"] = conv["converged"]
        return tr, summ

    # ----- SECONDARY (flagged cross-check): graded Beta-Binomial over ALL children.
    # Not the primary — it mixes already-off-floor children into a mover analysis and
    # is detection-limited; read only beside the mover table, never alone (#119).
    section_header("Build model (SECONDARY cross-check: graded Beta-Binomial, all children)")
    built_g = _factories.build_itt_model(prepared, likelihood="beta_binomial", **common)
    trace_g, graded = _fit_secondary(built_g, label=f"{spec.model_id} graded cross-check")
    pd.DataFrame([graded]).to_csv(
        os.path.join(ctx.output_dir, "tau_summary_graded.csv"), index=False
    )
    ctx.tables["tau_summary_graded"] = pd.DataFrame([graded])

    # ----- SECONDARY (flagged): graded contrast AMONG the off-floor children.
    # The #119 hurdle branch reads the graded score conditional on having come off
    # the floor. Two caveats keep this honest: (1) conditioning on post>0 is
    # POST-randomisation (selection on outcome), so the contrast is NOT a clean
    # randomised effect; and (2) this fits a *plain* Beta-Binomial to the post>0
    # subset — an untruncated proxy for the conditional-above-floor mean
    # E[post | post>0], because a zero-truncated Beta-Binomial is not cleanly
    # supported here (its vectorised logcdf is undefined); the untruncated fit
    # slightly overstates the conditional mean (issue #267 review). Reported flagged,
    # never as an ITT estimand.
    hurdle = None
    off_floor_data = restrict_to_off_floor(prepared, own)
    if off_floor_data.n_obs >= 8 and int(np.unique(off_floor_data.G).size) == 2:
        section_header(
            "Build model (SECONDARY: graded contrast among off-floor children | post>0)"
        )
        built_h = _factories.build_itt_model(
            off_floor_data, likelihood="beta_binomial", **common
        )
        _trace_h, hurdle = _fit_secondary(
            built_h, label=f"{spec.model_id} off-floor-subset graded contrast"
        )
        hurdle["n_off_floor"] = int(off_floor_data.n_obs)
        hurdle["untruncated_proxy"] = True
        pd.DataFrame([hurdle]).to_csv(
            os.path.join(ctx.output_dir, "tau_summary_hurdle.csv"), index=False
        )
        ctx.tables["tau_summary_hurdle"] = pd.DataFrame([hurdle])
    else:
        rprint(
            f"[yellow]hurdle conditional-above-floor secondary skipped for {own}: "
            f"only {off_floor_data.n_obs} off-floor rows (need >= 8, both arms).[/yellow]"
        )

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
                "primary_estimand": "Pr(off-floor at t2 | at floor at t1)",
                "at_risk_n": int(at_risk.n_obs),
                "total_n": int(prepared.n_obs),
            },
            "tau_offfloor_primary": off,
            "tau_graded_secondary": graded,
            "tau_hurdle_secondary": hurdle,
            "proportion_at_zero_ppc": {k: v for k, v in ppc0.items() if k != "rep"},
            "adjust_for": list(adjust_for),
        },
    )

    return _finalize_report(ctx)


# ---------------------------------------------------------------------------
# Joint pipeline (LRPITT12 joint; LRPITT15/15b contrasts)
# ---------------------------------------------------------------------------


def fit_joint(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    _require_spec(spec, "joint")

    ctx = make_context(spec, config)

    section_header("Prepare data")
    # A joint model may target an explicit outcome set (e.g. the taught-vs-not-
    # taught contrast in LRPITT15/15b); load exactly those so the complete-case mask is
    # not driven by the eight standardised outcomes. Defaults to ITT_OUTCOMES.
    joint_outcomes = tuple(spec.extra.get("outcomes") or ITT_OUTCOMES)
    if "outcomes" in spec.extra:
        prepared = load_and_prepare(phase_mode="itt", outcomes=joint_outcomes)
    else:
        prepared = load_and_prepare(phase_mode="itt")
    ctx.prepared = prepared

    _print_header(ctx)

    section_header("Build model")

    built = _factories.build_joint_model(
        prepared,
        outcomes=joint_outcomes,
        use_age_gp=spec.extra.get("use_age_gp", False),
        partial_pool_age_gp=spec.extra.get("partial_pool_age_gp", True),
        use_residual_correlation=spec.extra.get("use_residual_correlation", False),
        use_cross_baselines=spec.extra.get("use_cross_baselines", True),
        use_age_linear=spec.extra.get("use_age_linear", False),
    )
    _attach_built(ctx, built)

    _render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    _diag.save_prior_predictive_plot(ctx, spec.outcome_symbol or joint_outcomes[0])

    _run_sampling_and_loo(ctx)

    section_header("Summary diagnostics")
    _joint_vars = ["alpha", "tau", "gamma_own", "kappa"]
    if spec.extra.get("use_age_linear", False):
        _joint_vars.append("gamma_A")
    if spec.extra.get("use_residual_correlation", False):
        _joint_vars.append("sigma_outcome")
    _diag.summary_diagnostics(ctx, var_names=_joint_vars)

    _run_ppc(ctx)

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=_joint_vars)
    _diag.run_extended_diagnostics(ctx, causal_term="tau")
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=_joint_vars)
    # One forest of every outcome's tau — the single most communicative joint artifact.
    _save_forest_plot(ctx, ["tau"])

    section_header("Treatment-effect summary")
    outcomes = list(ctx.trace.posterior["outcome"].values)
    tau_df = _report.tau_summary_joint(ctx.trace, outcomes, ci_prob=ctx.reporting.ci_prob)
    tau_df.to_csv(os.path.join(ctx.output_dir, "tau_summary.csv"), index=False)
    ctx.tables["tau_summary"] = tau_df
    print_table(
        ranked_dataframe_table(
            tau_df,
            title=f"tau by outcome - {int(ctx.reporting.ci_prob * 100)}% CI (equal-tailed)",
            columns=["outcome", "tau_median", "tau_lo", "tau_hi", "prob_pos"],
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
            ctx.trace, outcomes, pair, ci_prob=ctx.reporting.ci_prob
        )
        diff_df = pd.DataFrame([diff_s])
        diff_df.to_csv(os.path.join(ctx.output_dir, "tau_difference.csv"), index=False)
        ctx.tables["tau_difference"] = diff_df
        print_table(
            metrics_table(
                [{"metric": k, "value": v} for k, v in diff_s.items()],
                title=(
                    f"tau[{pair[0]}] - tau[{pair[1]}] "
                    f"- {int(ctx.reporting.ci_prob * 100)}% CI (equal-tailed)"
                ),
                columns=["metric", "value"],
            )
        )
        meta_extra["tau_difference"] = diff_s

    _report.write_run_metadata(ctx, extra=meta_extra)

    return _finalize_report(ctx)


# ---------------------------------------------------------------------------
# Waitlist-crossover / difference-in-differences pipeline (kind="did")
# ---------------------------------------------------------------------------


def _did_diag_vars(spec: ModelSpec) -> list[str]:
    """Coefficients to summarise for a crossover/DiD fit, given the spec."""
    dose = bool(spec.extra.get("dose", False))
    period_varying = dose and bool(spec.extra.get("period_varying_dose", False))
    off_floor = spec.extra.get("likelihood") == "bernoulli_offfloor"
    if not dose:
        dose_vars = ["delta"]
    elif period_varying:
        dose_vars = ["mu_dose", "sigma_dose", "beta_dose_phase"]
    else:
        dose_vars = ["beta_dose"]
    v = ["alpha", "beta_period", *dose_vars]
    # Neither branch has an own-baseline term any more (A2, 2026-07-13): ``gamma_own``
    # was dropped from both, since the immediate arm's P2 period-start score is
    # treatment-affected. The graded Beta-Binomial keeps the dispersion ``kappa``; the
    # off-floor Bernoulli has none.
    if not off_floor:
        v += ["kappa"]
    if spec.extra.get("use_age", True):
        v.append("gamma_A")
    if spec.extra.get("use_child_re", True):
        v.append("sigma_child")
    if spec.extra.get("use_varying_delta", False):
        v.append("sigma_delta")
    return v


# Negligible-heterogeneity threshold on the logit scale for the "does the between-child
# treatment-effect SD concentrate near zero?" diagnostic (#230 §4a): an order of magnitude
# below the delta / tau prior scale (Normal(0, 0.5)).
_SIGMA_DELTA_ROPE = 0.1


def _did_heterogeneity_summary(trace, *, ci_prob: float) -> dict[str, float]:
    """Between-child SD of the on-intervention effect + its concentration near zero.

    Reports ``sigma_delta`` (median + equal-tailed CI on the logit scale), the ROPE-style
    ``P(sigma_delta < delta_het)`` "concentrates near zero" probability, and the prior mass
    below the same threshold under the HalfNormal(0.5) prior — so the reader can see the
    data moved it (#230 §2/§4a). A near-zero posterior is the clean "no reliable
    between-child variation" result that supports *not* gate-keeping on early response.
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


def fit_did(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    _require_spec(spec, "did", outcome=True)

    ctx = make_context(spec, config)

    section_header("Prepare data")
    sym = spec.outcome_symbol
    dose = bool(spec.extra.get("dose", False))
    period_varying = dose and bool(spec.extra.get("period_varying_dose", False))
    likelihood = spec.extra.get("likelihood", "beta_binomial")
    off_floor = likelihood == "bernoulli_offfloor"
    # Phase-stacked frame; load only this outcome so the complete-case mask does
    # not drop rows for measures the model never uses. The dose variant also needs
    # the per-period intervention-session count.
    outcomes = tuple(spec.extra.get("outcomes", (sym,)))
    covariates = ("attend",) if dose else ()
    # Neither DiD branch conditions on the own baseline any more (see build_did_model:
    # for the immediate arm's P2 the period-start score is post-treatment, so a
    # ``gamma_own`` term would bias the differenced ``delta``; A2, team decision
    # 2026-07-13). A missing period-start score is therefore no reason to drop a row —
    # the estimand needs only the period-end score / off-floor indicator — so neither
    # branch requires the pre-score. (Previously only the off-floor branch relaxed
    # this, which needlessly discarded four nonword P1 observations, #257 review.)
    pre_required = ()
    prepared = load_and_prepare(
        phase_mode="all",
        outcomes=outcomes,
        covariates=covariates,
        pre_required=pre_required,
    )
    ctx.prepared = prepared

    _print_header(ctx)

    section_header("Build model")
    built = _factories.build_did_model(
        prepared,
        outcome_symbol=sym,
        periods=tuple(spec.extra.get("periods", (0, 1))),
        use_child_re=spec.extra.get("use_child_re", True),
        use_age=spec.extra.get("use_age", True),
        dose=dose,
        period_varying_dose=period_varying,
        use_varying_delta=spec.extra.get("use_varying_delta", False),
        likelihood=likelihood,
    )
    _attach_built(ctx, built)

    _render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    _diag.save_prior_predictive_plot(ctx, spec.outcome_symbol or "W")

    _run_sampling_and_loo(ctx)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=_did_diag_vars(spec))

    if off_floor:
        _run_ppc(ctx, var_names=["y_offfloor"])
    else:
        _run_ppc(ctx)

    section_header("Extended diagnostics")
    _did_effect = "mu_dose" if period_varying else ("beta_dose" if dose else "delta")
    _diag.write_diagnostics_summary(ctx, var_names=_did_diag_vars(spec))
    _diag.run_extended_diagnostics(ctx, causal_term=_did_effect)
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=_did_diag_vars(spec))

    if period_varying:
        # Period-resolved dose readout (#135): partial-pooled per-period dose
        # slopes + a between-period SD, written by the shared dose-slope summary.
        # The headline question — does the L dose-gain slope vary by period? — is
        # answered by the nested PSIS-LOO vs the pooled comparator (lrp-rli-did-107)
        # in compare_statistical_models.py, not by this single-fit table.
        section_header("Period-resolved dose-slope summary")
        _write_dose_slope_summary(ctx, period_varying=True)
        _report.write_run_metadata(
            ctx,
            extra={
                "loo_elpd": float(ctx.loo.elpd),
                "dose": dose,
                "period_varying_dose": True,
                "dose_slope_summary": ctx.tables["dose_slope_summary"].to_dict(
                    "records"
                ),
            },
        )
    else:
        section_header("Crossover / DiD treatment-effect summary")
        from language_reading_predictors.statistical_models.measures import MEASURES

        did_s = _report.did_summary(
            ctx.trace,
            ci_prob=ctx.reporting.ci_prob,
            n_trials=1 if off_floor else MEASURES[sym].n_trials,
            dose=dose,
            off_floor=off_floor,
        )
        did_df = pd.DataFrame([did_s])
        did_df.to_csv(os.path.join(ctx.output_dir, "did_summary.csv"), index=False)
        ctx.tables["did_summary"] = did_df
        print_table(
            metrics_table(
                [{"metric": k, "value": v} for k, v in did_s.items()],
                title=(
                    f"crossover/DiD effect ({sym}"
                    f"{', off-floor risk difference' if off_floor else ''}) - "
                    f"{int(ctx.reporting.ci_prob * 100)}% CI "
                    "(equal-tailed); positive = intervention helps"
                ),
                columns=["metric", "value"],
            )
        )

        het = None
        if spec.extra.get("use_varying_delta", False):
            section_header("Treatment-effect heterogeneity (variance component)")
            het = _did_heterogeneity_summary(ctx.trace, ci_prob=ctx.reporting.ci_prob)
            pd.DataFrame([het]).to_csv(
                os.path.join(ctx.output_dir, "heterogeneity_summary.csv"), index=False
            )
            ctx.tables["heterogeneity_summary"] = pd.DataFrame([het])
            print_table(
                metrics_table(
                    [{"metric": k, "value": v} for k, v in het.items()],
                    title=(
                        f"treatment-effect heterogeneity ({sym}): between-child SD of the "
                        "on-intervention effect (logit); near-zero = homogeneous response"
                    ),
                    columns=["metric", "value"],
                )
            )

        _report.write_run_metadata(
            ctx,
            extra={
                "loo_elpd": float(ctx.loo.elpd),
                "did_summary": did_s,
                "dose": dose,
                **({"heterogeneity_summary": het} if het is not None else {}),
            },
        )

    return _finalize_report(ctx)


# ---------------------------------------------------------------------------
# Mechanism pipeline (LRP56 / LRP57 / LRP58)
# ---------------------------------------------------------------------------


def fit_mechanism(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    _require_spec(spec, "mechanism", mechanism=True)

    ctx = make_context(spec, config)
    # Some mechanism fits keep the HSGP curve and need a higher target_accept for
    # the residual boundary divergences (LRP58/71/158); honour it with the shared
    # CLI > model-specific > preset precedence.
    _apply_spec_target_accept(ctx, spec)

    section_header("Prepare data")
    # A model may restrict the prepared outcomes (e.g. LRP72 uses only L/B/N) so
    # ``drop_missing_pre`` does not discard rows for measures the model ignores.
    extra_outcomes = spec.extra.get("outcomes")
    # Raw-covariate adjusters (revised-DAG confounders that are not bounded-count
    # measures): hearing (hs/hs_missing), speech (deapp_c), phonological memory
    # (erbto), sessions (attend). Entered as standardised linear terms (#245).
    #
    # WAVE: split by semantics, not loaded wholesale from the pre row (#258 review,
    # P1). The DAG is contemporaneous, and the exposure, outcome and bounded-count
    # confounders are all read from the transition's POST row — so the *state*
    # covariates (hearing, speech, phonological memory) must come from the post row
    # too. Reading them from the pre row fits a hybrid pre/post adjustment set that
    # no graph licenses. Only ``attend`` is an *interval* variable (sessions
    # delivered during the pre -> post window; recorded t1-t3, absent at t4), so it
    # alone belongs on the pre row. See ``preprocessing.INTERVAL_COVARIATES``.
    adjust_for = tuple(spec.extra.get("adjust_for", ()))
    pre_adj, post_adj = split_covariates_by_wave(adjust_for)
    # Complete-case comparator: drop the mean-imputed rows so the confounders are
    # genuinely observed. Mean-imputation + a missingness indicator keeps every
    # child, but does not by itself guarantee adequate confounding control, so the
    # imputed fit needs this comparator beside it (#258 review).
    require_observed = tuple(spec.extra.get("require_observed", ()))
    _kw = {
        "covariates": pre_adj,
        "post_covariates": post_adj,
        "require_observed": require_observed,
    }
    if extra_outcomes is not None:
        prepared = load_and_prepare(
            phase_mode="all", outcomes=tuple(extra_outcomes), **_kw
        )
    else:
        prepared = load_and_prepare(phase_mode="all", **_kw)
    ctx.prepared = prepared
    # A constant covariate (e.g. a ``_missing`` indicator that is all-zero on the
    # fitted rows) is dropped by the loader and receives no coefficient, so it must
    # not be built into the model nor reported as adjusted-for.
    adjust_for = tuple(c for c in adjust_for if c in prepared.covariates)

    _print_header(ctx)

    section_header("Build model")
    moderator_symbol = spec.extra.get("moderator_symbol")
    # Drop the autoregressive baseline (any ``*_pre`` token, e.g. W_pre / N_pre)
    # from the confounder list — it enters via ``adjust_baseline_symbol``.
    from language_reading_predictors.statistical_models.measures import MEASURES

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
        confounder_symbols=tuple(
            s for s in confounders if s in ("G", "A") or s in MEASURES
        ),
        use_age_gp=spec.extra.get("use_age_gp", False),
        phase_specific_mechanism=spec.extra.get("phase_specific_mechanism", False),
        use_subject_random_intercept=spec.extra.get(
            "use_subject_random_intercept", True
        ),
        moderator_symbol=moderator_symbol,
        moderator_is_covariate=spec.extra.get("moderator_is_covariate", False),
        include_interaction=spec.extra.get("include_interaction", True),
        linear_mechanism=spec.extra.get("linear_mechanism", False),
        adjust_for=adjust_for,
    )
    _attach_built(ctx, built)

    _render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    _diag.save_prior_predictive_plot(ctx, spec.outcome_symbol or "W")

    _run_sampling_and_loo(ctx)

    section_header("Summary diagnostics")
    _mech_vars = ["alpha", "beta_G", "gamma_own", "kappa"]
    _mech_vars += [f"gamma_{s}" for s in confounders if s in MEASURES]
    _mech_vars += [f"gamma_{c}" for c in adjust_for]
    if "A" in confounders and not spec.extra.get("use_age_gp", False):
        _mech_vars.append("gamma_A")
    if spec.extra.get("use_subject_random_intercept", True):
        _mech_vars.append("sigma_child")
    if spec.extra.get("linear_mechanism", False):
        _mech_vars.append("beta_mech")
    if moderator_symbol is not None:
        _mech_vars.append("gamma_mod")
        if spec.extra.get("include_interaction", True):
            _mech_vars.append("gamma_int")
    _diag.summary_diagnostics(ctx, var_names=_mech_vars)

    _run_ppc(ctx)

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=_mech_vars)
    _diag.run_extended_diagnostics(ctx)

    # Mechanism curve: f_mech vs mech_post_logit grid (logit-contribution scale only).
    section_header("Mechanism curve")
    _write_mechanism_curve(ctx)
    _write_readiness_threshold(ctx)

    # Record the adjustment set that was actually FITTED — with each term's source
    # column, measurement wave and missing-indicator status — not just the requested
    # symbols. ``spec.adjustment`` alone materially misdescribed the model, because
    # the ``adjust_for`` covariates never reached config.json (#258 review, P1).
    meta_extra = {
        "loo_elpd": float(ctx.loo.elpd),
        "adjustment": spec.adjustment,
        "effective_adjustment": _effective_adjustment(
            spec,
            prepared,
            measure_confounders=tuple(
                s for s in confounders if s in ("G", "A") or s in MEASURES
            ),
            adjust_for=adjust_for,
            baseline_symbol=spec.extra.get("adjust_baseline_symbol", "W"),
        ),
    }

    # Linear-moderation summary (gamma_int / gamma_mod), when a moderator is set.
    if moderator_symbol is not None:
        section_header("Interaction summary")
        gi = _report.gamma_interaction_summary(ctx.trace, ci_prob=ctx.reporting.ci_prob)
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
                    f"- {int(ctx.reporting.ci_prob * 100)}% CI (equal-tailed)"
                ),
                columns=["metric", "value"],
            )
        )
        meta_extra["moderator_symbol"] = moderator_symbol
        meta_extra["interaction_summary"] = gi

    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=_mech_vars)
    _report.write_run_metadata(ctx, extra=meta_extra)

    return _finalize_report(ctx)


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
        # No f_mech / beta_mech in the posterior — e.g. a phase_specific_mechanism
        # fit, whose per-phase f_mech is not registered under either name, so the
        # curve would be silently skipped. Warn loudly rather than no-op (issue
        # #273); register the phase-specific curve as pm.Deterministic("f_mech",
        # ..., dims="obs_id") in the factory if such a model is ever shipped.
        rprint(
            "[yellow]_write_mechanism_curve: no 'f_mech'/'beta_mech' in the "
            f"posterior for {ctx.spec.model_id} (phase_specific_mechanism?); "
            "no mechanism_curve.csv/plot written.[/yellow]"
        )
        return

    order = np.argsort(mech_logit)
    x = mech_logit[order]
    f_ord = f[order]
    mean = f_ord.mean(axis=1)
    lo = np.quantile(f_ord, 0.025, axis=1)
    hi = np.quantile(f_ord, 0.975, axis=1)
    lo90 = np.quantile(f_ord, 0.05, axis=1)
    hi90 = np.quantile(f_ord, 0.95, axis=1)
    pd.DataFrame(
        {"mech_logit": x, "f_mean": mean, "f_lo": lo, "f_hi": hi,
         "f_lo90": lo90, "f_hi90": hi90}
    ).to_csv(os.path.join(ctx.output_dir, "mechanism_curve.csv"), index=False)
    outcome = ctx.spec.outcome_symbol or "W"
    plt.figure(figsize=(6, 4))
    plt.plot(x, mean, color="#1f77b4", lw=2)
    plt.fill_between(x, lo, hi, color="#1f77b4", alpha=0.2)
    plt.xlabel(f"logit({sym}_post)")
    plt.ylabel("predictor logit contribution")
    plt.title(f"Mechanism curve ({kind}): {sym} -> {outcome}")
    # mechanism_curve.csv (the plotted band) is written just above.
    save_styled_figure(ctx.output_dir, "mechanism_curve")


def _write_readiness_threshold(ctx: StatisticalFitContext) -> None:
    """Readiness-threshold summary for the mechanism curve (#230 §2/§5).

    Post-processes the fitted nonparametric mechanism curve (``f_mech``) into a
    posterior for the predictor count at which the outcome rises *fastest* — the
    "knee" (the steepest rise, not the onset), via
    :func:`reporting.readiness_threshold`. Only the GP mechanism has a curve to
    find a knee in; linear / phase-specific fits (no ``f_mech``) are skipped
    quietly. Writes ``readiness_threshold.csv`` and a plot. Guarded by the
    caller.
    """
    post = ctx.trace.posterior
    if "f_mech" not in post:
        return

    from language_reading_predictors.statistical_models.measures import MEASURES

    sym = ctx.spec.mechanism_symbol
    N = MEASURES[sym].n_trials

    try:
        summary = _report.readiness_threshold(ctx.trace, n_trials=N)
    except ValueError as exc:
        rprint(f"[yellow]_write_readiness_threshold: {exc}; skipped.[/yellow]")
        return
    pd.DataFrame([summary]).to_csv(
        os.path.join(ctx.output_dir, "readiness_threshold.csv"), index=False
    )

    # Mean curve on the raw count scale (inverse Haldane-corrected logit, as in
    # reporting._readiness_knee) with the knee posterior overlaid.
    ell = np.asarray(ctx.trace.constant_data["mech_post_logit"].values).reshape(-1)
    f = post["f_mech"].stack(sample=("chain", "draw")).values  # (n_obs, n_sample)
    order = np.argsort(ell)
    x_count = np.clip(
        (N + 1.0) / (1.0 + np.exp(-ell[order])) - 0.5, 0.0, float(N)
    )
    mean = f[order].mean(axis=1)
    outcome = ctx.spec.outcome_symbol or "W"
    plt.figure(figsize=(6, 4))
    plt.plot(x_count, mean, color="#1f77b4", lw=2)
    plt.axvspan(
        summary["knee_count_ci_low"],
        summary["knee_count_ci_high"],
        color="#d62728",
        alpha=0.15,
        label="knee 95% CI",
    )
    plt.axvline(
        summary["knee_count_median"], color="#d62728", lw=1.5, label="knee median"
    )
    plt.xlabel(f"{sym} (raw count, out of {N})")
    plt.ylabel(f"{outcome} logit contribution")
    plt.title(f"Readiness threshold (steepest rise): {sym} -> {outcome}")
    plt.legend(fontsize=8)
    save_styled_figure(ctx.output_dir, "readiness_threshold")


# ---------------------------------------------------------------------------
# Dose-response pipeline (LRP77, #104 Phase 2)
# ---------------------------------------------------------------------------


def fit_dose_response(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Period-resolved dose-response fit (#104 Phase 2).

    Reuses the mechanism-family backbone (Beta-Binomial conditional change,
    phase intercepts, subject random intercept) but the focal predictor is the
    per-period intervention **dose** (``attend``), entered with partial-pooled
    period-specific slopes. See :func:`factories.build_dose_response_model`.
    """
    _require_spec(spec, "dose_response")

    ctx = make_context(spec, config)

    section_header("Prepare data")
    dose_cov = spec.extra.get("dose_covariate", "attend")
    # Default OFF (issue #269): the cumulative-dose (attend_cumul) control conditions
    # on the IS collider; only the flagged sensitivity variant sets it.
    dose_stage_cov = spec.extra.get("dose_stage_covariate")
    ability = tuple(spec.extra.get("ability_adjust_symbols", ()))
    outcomes = tuple(spec.extra.get("outcomes", (spec.outcome_symbol or "W",)))
    cov_cols = tuple(c for c in (dose_cov, dose_stage_cov) if c)
    prepared = load_and_prepare(
        phase_mode="all", outcomes=outcomes, covariates=cov_cols
    )
    ctx.prepared = prepared

    _print_header(ctx)

    section_header("Build model")

    period_varying = spec.extra.get("period_varying_dose", True)
    adjust_group = spec.extra.get("adjust_group", True)
    adjust_age = spec.extra.get("adjust_age", True)
    built = _factories.build_dose_response_model(
        prepared,
        outcome_symbol=spec.outcome_symbol or "W",
        adjust_baseline_symbol=spec.extra.get("adjust_baseline_symbol", "W"),
        dose_covariate=dose_cov,
        dose_stage_covariate=dose_stage_cov,
        period_varying_dose=period_varying,
        use_subject_random_intercept=spec.extra.get(
            "use_subject_random_intercept", True
        ),
        adjust_group=adjust_group,
        adjust_age=adjust_age,
        ability_adjust_symbols=ability,
    )
    _attach_built(ctx, built)

    _render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    _diag.save_prior_predictive_plot(ctx, spec.outcome_symbol or "W")

    _run_sampling_and_loo(ctx)

    section_header("Summary diagnostics")
    dose_vars = ["alpha", "gamma_own", "kappa"]
    if spec.extra.get("use_subject_random_intercept", True):
        dose_vars.append("sigma_child")
    if adjust_group:
        dose_vars.append("beta_G")
    if adjust_age:
        dose_vars.append("gamma_A")
    if period_varying:
        dose_vars.extend(["mu_dose", "sigma_dose", "beta_dose_phase"])
    else:
        dose_vars.append("beta_dose")
    if dose_stage_cov is not None:
        dose_vars.append("gamma_dose_stage")
    dose_vars.extend(f"gamma_{s}_pre" for s in ability)
    _diag.summary_diagnostics(ctx, var_names=dose_vars)

    _run_ppc(ctx)

    section_header("Extended diagnostics")
    _dose_effect = "mu_dose" if period_varying else "beta_dose"
    _diag.write_diagnostics_summary(ctx, var_names=dose_vars)
    _diag.run_extended_diagnostics(ctx, causal_term=_dose_effect)

    section_header("Dose-slope summary")
    _write_dose_slope_summary(ctx, period_varying=period_varying)

    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=dose_vars)
    _report.write_run_metadata(
        ctx,
        extra={
            "loo_elpd": float(ctx.loo.elpd),
            "adjustment": spec.adjustment,
            "period_varying_dose": period_varying,
            "ability_adjust_symbols": list(ability),
        },
    )

    return _finalize_report(ctx)


def _summarise_draws(
    values: np.ndarray, ci_prob: float, *, include_p_pos: bool = True
) -> dict[str, float]:
    """Mean, equal-tailed CI and (optionally) P(>0) for a 1-D array of draws.

    ``ci_prob`` is the interval *coverage* probability (equal-tailed), read from
    ``ctx.reporting.ci_prob`` — see the naming note in ``context.make_context`` (#170).
    ``include_p_pos=False`` omits the directional ``P(>0)`` for a strictly-positive
    quantity (e.g. a between-period SD) where it is trivially 1 and meaningless.
    """
    lo_q = (1.0 - ci_prob) / 2.0
    out = {
        "mean": float(np.mean(values)),
        "lo": float(np.quantile(values, lo_q)),
        "hi": float(np.quantile(values, 1.0 - lo_q)),
        # 90% equal-tailed sensitivity band alongside the headline ci_prob interval.
        "lo90": float(np.quantile(values, 0.05)),
        "hi90": float(np.quantile(values, 0.95)),
    }
    if include_p_pos:
        out["p_pos"] = float(np.mean(values > 0.0))
    return out


def _write_dose_slope_summary(
    ctx: StatisticalFitContext, *, period_varying: bool
) -> None:
    """Posterior dose slope (overall + per-period) on the per-1-SD logit scale."""
    post = ctx.trace.posterior
    ci_prob = ctx.reporting.ci_prob
    rows: list[dict[str, object]] = []

    def _draws(name: str) -> np.ndarray:
        return post[name].stack(sample=("chain", "draw")).values

    if period_varying:
        rows.append({"term": "dose_overall", **_summarise_draws(_draws("mu_dose"), ci_prob)})
        bdp = _draws("beta_dose_phase")  # (phase, sample)
        for p in range(bdp.shape[0]):
            rows.append(
                {"term": f"dose_period{p + 1}", **_summarise_draws(bdp[p], ci_prob)}
            )
        rows.append(
            {
                "term": "sigma_dose_between_period",
                **_summarise_draws(_draws("sigma_dose"), ci_prob, include_p_pos=False),
            }
        )
    else:
        rows.append({"term": "dose_pooled", **_summarise_draws(_draws("beta_dose"), ci_prob)})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ctx.output_dir, "dose_slope_summary.csv"), index=False)
    ctx.tables["dose_slope_summary"] = df
    print_table(
        metrics_table(
            [
                {"metric": r["term"], "value": r["mean"], "lo": r["lo"], "hi": r["hi"]}
                for r in rows
            ],
            title=(
                f"Dose slope (logit / 1 SD dose) - {int(ci_prob * 100)}% CI (equal-tailed)"
            ),
            columns=["metric", "value", "lo", "hi"],
        )
    )


# ---------------------------------------------------------------------------
# Mediation pipeline (LRP59)
# ---------------------------------------------------------------------------

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
    # Match the primary fit's load set so a mediator/confounder outside
    # ITT_OUTCOMES (TE, N) is present in the lagged-outcome frame too.
    _extra_outcomes = spec.extra.get("outcomes")
    _lag_kwargs = (
        {"outcomes": tuple(_extra_outcomes)} if _extra_outcomes is not None else {}
    )
    prepared_t3 = load_and_prepare_lagged_outcome(
        outcome_symbol,
        outcome_time=_T3_SENSITIVITY_TIME,
        covariates=_raw_covariate_confounders(confounders),
        **_lag_kwargs,
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
    # Gate this temporal-ordering sensitivity sub-fit (bypasses the primary gate).
    conv = _diag.subfit_convergence(trace_t3, label=f"{spec.model_id} t3 sensitivity")
    df_t3 = _med.decompose(
        trace_t3,
        med_t3,
        ci_prob=ctx.reporting.ci_prob,
    )
    # Persist the verdict onto the published rows: this sub-fit bypasses the primary
    # gate, and the verdict was previously computed then discarded so the t3 table
    # shipped with no convergence flag (this review's finding B1). Flows through to
    # both mediation_summary_t3.csv and the mediation_t3_sensitivity metadata block.
    df_t3["converged"] = conv["converged"]
    return df_t3


def fit_mediation(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """ITT-phase mediation decomposition (LRP59): how much of G -> W flows via L."""
    _require_spec(spec, "mediation")
    from language_reading_predictors.statistical_models import mediation as _med

    ctx = make_context(spec, config)

    section_header("Prepare data")
    # Phase 0 only (t1 -> t2): the single randomised contrast. One row per child.
    mediator_symbol = spec.mechanism_symbol or "L"
    # Drop the structural markers and the mediator's own baseline ({mediator}_t1,
    # handled inside the factory) from the adjustment set; the rest are confounders.
    # The set mixes bounded-count skill measures (E, R — arriving via pre_logit) and
    # revised-DAG RAW covariates (hearing ``hs``/``hs_missing``, speech ``deapp_c``,
    # phonological memory ``erbto`` + indicators; #246), which must be requested as
    # covariates and are taken from the t1 pre-row (treatment-unaffected). Models
    # with no raw covariates get ``covariates=()`` — a no-op, so LRP59/62/64/66 and
    # the #263 mediation family are unchanged unless a spec adds raw confounders.
    confounders = tuple(
        s
        for s in spec.adjustment
        if s not in ("G", "A", "W_pre", f"{mediator_symbol}_t1")
    )
    _raw_cov = _raw_covariate_confounders(confounders)
    # A mediator or confounder outside ``ITT_OUTCOMES`` (e.g. taught-expressive TE,
    # nonword N) must be requested via ``extra["outcomes"]`` so it is loaded; this
    # also restricts the complete-case mask to the symbols the model uses (mirrors
    # fit_itt).
    _extra_outcomes = spec.extra.get("outcomes")
    _outcome_time = spec.extra.get("outcome_time")
    if _outcome_time is not None:
        # Longitudinal-ordering primary fit (LRP76): the mediator stays at t2 but
        # the outcome is taken from a later wave (t3/t4), so the mediator strictly
        # precedes the outcome — promoting the temporal-ordering check from a
        # sensitivity to the primary estimand. The t2 -> t{outcome_time} increment
        # is NOT randomised (both arms treated after t2), so this is a
        # triangulation design, read under stated assumptions, not a cleaner τ.
        _lag_outcomes = (
            tuple(_extra_outcomes) if _extra_outcomes is not None else ITT_OUTCOMES
        )
        prepared = load_and_prepare_lagged_outcome(
            spec.outcome_symbol or "W",
            outcome_time=int(_outcome_time),
            outcomes=_lag_outcomes,
            covariates=_raw_cov,
        )
    elif _extra_outcomes is not None:
        prepared = load_and_prepare(
            phase_mode="itt",
            outcomes=tuple(_extra_outcomes),
            covariates=_raw_cov,
            drop_missing_pre=bool(spec.extra.get("drop_missing_pre", True)),
        )
    else:
        prepared = load_and_prepare(phase_mode="itt", covariates=_raw_cov)
    ctx.prepared = prepared
    # A missing-indicator can be constant on the ITT-phase rows (SP/RW are near-
    # complete at t1) and be dropped by the loader; keep only confounders actually
    # present, so no vacuous coefficient is fitted for a dropped covariate.
    confounders = tuple(
        c for c in confounders if c in prepared.covariates or c in prepared.pre_logit
    )

    _print_header(ctx)

    section_header("Build model")

    mediator_kind = spec.extra.get("mediator_kind", "beta_binomial")
    route_symbols = tuple(spec.extra.get("route_symbols", ()))
    built, med_data = _factories.build_mediation_model(
        prepared,
        mediator_symbol=mediator_symbol,
        outcome_symbol=spec.outcome_symbol or "W",
        confounder_symbols=confounders,
        mediator_kind=mediator_kind,
        route_symbols=route_symbols,
    )
    _attach_built(ctx, built)

    # The mediator observed node differs by kind: Beta-Binomial "{mediator}_post"
    # vs the Gaussian composite "M_post".
    is_gaussian = mediator_kind == "gaussian_composite"
    mediator_node = "M_post" if is_gaussian else f"{mediator_symbol}_post"
    # Diagnose every scalar coefficient the model actually built (deterministics
    # and the observed mediator/outcome nodes are not free RVs), so the list
    # tracks the fitted confounder set instead of a hand-maintained constant.
    coef_vars = sorted(rv.name for rv in built.model.free_RVs if rv.ndim == 0)

    _render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    # The mediator likelihood is the FIRST observed RV, so name the outcome node
    # explicitly — else the plot overlays mediator draws on the outcome's counts.
    _diag.save_prior_predictive_plot(ctx, spec.outcome_symbol or "W", node="y_post")

    _run_sampling_and_loo(ctx, compute_loo=False)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=coef_vars)

    _run_ppc(ctx, var_names=[mediator_node, "y_post"])

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=coef_vars)
    _diag.run_extended_diagnostics(ctx)
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=coef_vars)

    section_header("Mediation decomposition (g-formula)")
    _interventional = spec.extra.get("estimand") == "interventional"
    med_df = _med.decompose(
        ctx.trace,
        med_data,
        ci_prob=ctx.reporting.ci_prob,
        interventional=_interventional,
    )
    med_df.to_csv(os.path.join(ctx.output_dir, "mediation_summary.csv"), index=False)
    ctx.tables["mediation_summary"] = med_df
    # Print the primary decomposition table before the (slow, ~21x-decompose) sensitivity
    # sweep, so the main NDE/NIE result shows under its own section header rather than
    # under the sensitivity header and only after the sweep finishes (#289 review).
    print_table(
        ranked_dataframe_table(
            med_df,
            title=f"Mediation (intervention-helps; words out of {med_data.n_trials_W})",
            columns=["quantity", "words_mean", "words_lo", "words_hi", "prob_pos"],
            rank_column=False,
            precision=3,
        )
    )

    # Unmeasured mediator-outcome confounding sensitivity for the NIE (#230): sweep a
    # bias off b_M and report the tipping point at which the indirect effect's CI
    # includes 0 (a Bayesian E-value analogue). Quantifies the no-unmeasured-
    # confounding assumption the decomposition otherwise only states.
    section_header("Mediation NIE sensitivity (unmeasured confounding)")
    sens_sweep, sens_summary = _med.sensitivity_sweep(
        ctx.trace,
        med_data,
        ci_prob=ctx.reporting.ci_prob,
        interventional=_interventional,
    )
    sens_sweep.to_csv(
        os.path.join(ctx.output_dir, "mediation_sensitivity.csv"), index=False
    )
    pd.DataFrame([sens_summary]).to_csv(
        os.path.join(ctx.output_dir, "mediation_sensitivity_summary.csv"), index=False
    )
    ctx.tables["mediation_sensitivity"] = sens_sweep
    if sens_summary["already_null_at_zero"]:
        rprint(
            "  NIE not credibly nonzero at delta=0 — sensitivity analysis N/A "
            "(no indirect effect to explain away)."
        )
    elif sens_summary["robust_over_full_sweep"]:
        rprint(
            f"  NIE robust across the full sweep (CI excludes 0 up to "
            f"delta={sens_sweep['delta'].max():.2f} logit)."
        )
    else:
        rprint(
            f"  NIE tipping point delta*={sens_summary['tipping_delta']:.3f} logit "
            f"({sens_summary['tipping_frac_of_bM']:.0%} of the fitted b_M+b_GM) — an "
            "unmeasured mediator-outcome confounder that strong would null the NIE."
        )

    # --- Temporal-ordering sensitivity: outcome at t3, mediator still at t2 ---
    # Triangulation for the contemporaneous-measurement caveat (issue #84): the
    # mediator now precedes the outcome in time. NB the t2 -> t3 increment is not
    # randomised (both arms treated after t2), so read this as triangulation only.
    # Skipped when the primary fit is ALREADY longitudinal (outcome_time set, LRP76)
    # — the sensitivity would double-lag and duplicate the primary estimand.
    med_df_t3 = None
    if _outcome_time is None and not _interventional:
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
    # Record the REQUESTED adjustment set and the confounders ACTUALLY fitted
    # separately (#246 review, P2). A raw covariate can be dropped by the loader
    # when its missing-indicator is constant on the ITT rows; recording only
    # ``spec.adjustment`` would then imply a coefficient that was never estimated.
    _requested_raw = _raw_covariate_confounders(
        s for s in spec.adjustment if s not in ("G", "A", "W_pre", f"{mediator_symbol}_t1")
    )
    _extra_meta = {
        "adjustment": spec.adjustment,
        "effective_confounders": list(confounders),
        "dropped_confounders": [c for c in _requested_raw if c not in confounders],
        "n_obs": prepared.n_obs,
        "mediation": _summary,
    }
    if med_df_t3 is not None:
        _extra_meta["mediation_t3_sensitivity"] = {
            r["quantity"]: r for r in med_df_t3.to_dict("records")
        }
    if _outcome_time is not None:
        _extra_meta["outcome_time"] = int(_outcome_time)
    _report.write_run_metadata(ctx, extra=_extra_meta)

    return _finalize_report(ctx)


# ---------------------------------------------------------------------------
# Gain-factors / level-factors pipelines (LRPGF / LRPLF, #127)
# ---------------------------------------------------------------------------


def _gf_coef_names(
    spec: ModelSpec, adjust_for: tuple[str, ...] | None = None
) -> list[str]:
    """Factor coefficients to report in the LRPGF factor table (interpretable
    terms only; nuisance alpha/alpha_phase/kappa/sigma_child are excluded).

    ``adjust_for`` overrides the requested ``spec.extra['adjust_for']`` with the
    **actually fitted** set (a constant ``_missing`` indicator is dropped by the
    loader and gets no ``gamma_{c}`` coefficient), so the pipeline passes the
    post-filter tuple; ``None`` falls back to the requested set (used off the fit
    path, e.g. in tests)."""
    extra = spec.extra
    treated_only = bool(extra.get("treated_only", False))
    adj = extra.get("adjust_for", ()) if adjust_for is None else adjust_for
    names: list[str] = []
    if not treated_only:
        names.append("beta_trt")
    # gamma_own drops on the off-floor (Bernoulli) path (A4) — see build_gain_factors_model.
    if extra.get("likelihood") != "bernoulli_offfloor":
        names.append("gamma_own")
    names.append("gamma_A")
    if extra.get("ability_covariate"):
        names.append("gamma_ability")
    names += [f"gamma_{s}" for s in extra.get("skill_symbols", ())]
    names += [f"gamma_{c}" for c in adj]
    for pair in extra.get("interactions", ()):
        a, b = tuple(pair)
        if treated_only and "trt" in (a, b):
            continue
        names.append(f"gamma_int_{a}_{b}")
    return names


def _gf_diag_vars(
    spec: ModelSpec, adjust_for: tuple[str, ...] | None = None
) -> list[str]:
    # No kappa under the off-floor Bernoulli likelihood.
    tail = (
        ["sigma_child"]
        if spec.extra.get("likelihood") == "bernoulli_offfloor"
        else ["kappa", "sigma_child"]
    )
    # Include the per-phase intercept vector, mirroring _lf_diag_vars' alpha_time
    # (issue #274 item 2); the gate already covers it via the free-RV scan, this
    # keeps the human-readable diagnostics.csv consistent across the two families.
    return ["alpha", "alpha_phase", *_gf_coef_names(spec, adjust_for), *tail]


def fit_gain_factors(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    _require_spec(spec, "gain_factors", outcome=True)
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
    # Revised-DAG raw-covariate confounders (hearing/speech/phonological memory; #247).
    # Timing (review finding A1; team decision 2026-07-13): the language-proximal SP/RW
    # confounders (deapp_c/erbto + their missing indicators) are read at the pre-
    # randomisation BASELINE (t1) — at the period-1 post wave (t2) they may already be
    # treatment-affected, so conditioning there would adjust a descendant of the exposure
    # and bias the randomised beta_trt. Hearing (hs) is exogenous to a language
    # intervention and stays contemporaneous (post); ``attend`` (if ever present) stays on
    # the interval pre row. Re-filter after loading — a constant ``_missing`` indicator is
    # dropped by the loader and must not be built or gated.
    adjust_for = tuple(extra.get("adjust_for", ()))
    pre_adj, post_adj = split_covariates_by_wave(adjust_for)
    baseline_adj, post_adj = split_confounders_by_timing(post_adj)
    prepared = load_and_prepare(
        phase_mode="all",
        outcomes=(spec.outcome_symbol, *skill_symbols),
        baseline_covariates=(*baseline_covariates, *baseline_adj),
        covariates=pre_adj,
        post_covariates=post_adj,
    )
    adjust_for = tuple(c for c in adjust_for if c in prepared.covariates)
    ctx.prepared = prepared
    _print_header(ctx)

    section_header("Build model")
    built = _factories.build_gain_factors_model(
        prepared,
        outcome_symbol=spec.outcome_symbol,
        skill_symbols=skill_symbols,
        ability_covariate=ability_covariate,
        adjust_for=adjust_for,
        interactions=interactions,
        treated_only=treated_only,
        likelihood=likelihood,
    )
    _attach_built(ctx, built)

    _render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    _diag.save_prior_predictive_plot(ctx, spec.outcome_symbol, node=obs_node)

    _run_sampling_and_loo(ctx)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=_gf_diag_vars(spec, adjust_for))

    _run_ppc(ctx, var_names=[obs_node])

    section_header("Extended diagnostics")
    _causal_gf = None if treated_only else "beta_trt"
    _diag.write_diagnostics_summary(ctx, var_names=_gf_diag_vars(spec, adjust_for))
    _diag.run_extended_diagnostics(ctx, causal_term=_causal_gf)
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=_gf_diag_vars(spec, adjust_for))
    if _causal_gf is not None:
        _save_forest_plot(ctx, [_causal_gf])
        _diag.run_psense(ctx, var_names=[_causal_gf])

    section_header("Factor summary")
    fs = _report.factor_summary(
        ctx.trace, _gf_coef_names(spec, adjust_for), ci_prob=ctx.reporting.ci_prob,
        causal_terms=("beta_trt",),
    )
    fs.to_csv(os.path.join(ctx.output_dir, "factor_summary.csv"), index=False)
    ctx.tables["factor_summary"] = fs
    _save_association_forest(ctx, _gf_coef_names(spec, adjust_for), ("beta_trt",))
    print_table(
        ranked_dataframe_table(
            fs,
            title=f"Factor summary ({spec.outcome_symbol}) - {int(ctx.reporting.ci_prob * 100)}% CrI",
            columns=["term", "role", "median", "lo", "hi", "prob_positive"],
            rank_column=False,
            precision=3,
        )
    )

    meta_extra = {
        "loo_elpd": float(ctx.loo.elpd),
        "treated_only": treated_only,
        # Requested vs actually-fitted adjustment set, incl. dropped-constant
        # covariates (#247 / #258 review P1). Skills enter at the pre baseline; the
        # raw-covariate confounders (hs/deapp_c/erbto) at the split wave.
        "effective_adjustment": _effective_adjustment(
            spec,
            built.prepared,
            adjust_for=adjust_for,
            ability_covariate=ability_covariate,
            baseline_symbol=spec.outcome_symbol,
            skill_baselines=skill_symbols,
        ),
    }
    # Items-scale marginal effect of the treatment term. Skipped when
    # treated_only (the on-intervention indicator is then constant and beta_trt
    # is absent).
    if not treated_only:
        trt = ((built.prepared.G == 1) | (built.prepared.phase >= 1)).astype(float)
        # The marginal treatment effect is averaged over the **period-1** rows only
        # (#247 P2): period 1 is the genuinely randomised, all-untreated-baseline
        # transition, so its switch-on-vs-off contrast is the causal ITT-anchor
        # estimand. The post-crossover transitions (phase >= 1) carry no untreated
        # observations and baselines that may already be treatment-affected, so
        # pooling them yields a model-based transported contrast, not the ITT effect.
        # The logit-scale beta_trt posterior itself is unchanged; only its
        # probability/items-scale marginalisation is restricted.
        p1_mask = built.prepared.phase == 0
        # Net out the *full* per-row treatment contribution — ``beta_trt`` plus every
        # fitted treatment interaction (``gamma_int_trt_*``) — so the marginal effect
        # reflects the modelled heterogeneity, not ``beta_trt`` alone. The factory
        # exposes the exact standardised moderator vectors it used.
        trt_moderators = built.extras.get("trt_interaction_moderators", [])
        # Off-floor models are Bernoulli on Pr(post > 0); the "items" scale then
        # collapses to the off-floor risk difference (n_trials = 1).
        n_marg = 1 if off_floor else built.prepared.n_trials[spec.outcome_symbol]
        tme = _report.treatment_marginal_effect(
            ctx.trace,
            trt=trt,
            n_trials=n_marg,
            moderators=trt_moderators,
            ci_prob=ctx.reporting.ci_prob,
            row_mask=p1_mask,
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
                term="beta_trt", varying_term="", moderators=trt_moderators,
                ci_prob=ctx.reporting.ci_prob, row_mask=p1_mask,
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
            ROPE_DELTA_PROB_GRID,
        )

        delta_items = ROPE_DELTA.get(spec.outcome_symbol)
        delta_prob = ROPE_DELTA_PROB.get(spec.outcome_symbol)
        if delta_items is not None and not off_floor:
            rope_s = _report.rope_summary(
                ctx.trace,
                G=trt,
                n_trials=n_marg,
                delta=delta_items,
                ci_prob=ctx.reporting.ci_prob,
                term="beta_trt",
                varying_term="",
                moderators=trt_moderators,
                row_mask=p1_mask,
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
                term="beta_trt", varying_term="", moderators=trt_moderators,
                row_mask=p1_mask,
            )
        elif off_floor and delta_prob is not None:
            # Off-floor risk-difference ROPE, matching the floored ITT path
            # (#125 Area 4). The 10 pp δ was signed off by the education lead
            # (2026-07-01, #144), so it is NOT provisional; the ITT floored path
            # sets provisional_delta=False and this mirrors it.
            rope_s = _report.rope_summary(
                ctx.trace, G=trt, n_trials=1, delta=delta_prob,
                ci_prob=ctx.reporting.ci_prob, term="beta_trt", varying_term="",
                moderators=trt_moderators, row_mask=p1_mask,
            )
            rope_s["provisional_delta"] = False  # 10 pp signed off (#144, 2026-07-01)
            rope_s["delta_scale"] = "risk_difference"
            pd.DataFrame([rope_s]).to_csv(
                os.path.join(ctx.output_dir, "rope_summary.csv"), index=False
            )
            ctx.tables["rope_summary"] = pd.DataFrame([rope_s])
            meta_extra["rope_summary"] = rope_s
            _save_rope_plot(
                ctx, spec.outcome_symbol, trt, 1, delta_prob,
                term="beta_trt", varying_term="", moderators=trt_moderators,
                row_mask=p1_mask,
            )
            # δ-sensitivity sweep on the risk-difference scale (#144): 10/15/20 pp,
            # the grid the sign-off mandates (mirrors the floored ITT path).
            sens_df = _report.rope_sensitivity(
                ctx.trace, G=trt, n_trials=1, deltas=ROPE_DELTA_PROB_GRID,
                term="beta_trt", varying_term="", moderators=trt_moderators,
                row_mask=p1_mask,
            )
            sens_df.to_csv(
                os.path.join(ctx.output_dir, "rope_sensitivity.csv"), index=False
            )
            ctx.tables["rope_sensitivity"] = sens_df

    _report.write_run_metadata(ctx, extra=meta_extra)
    return _finalize_report(ctx)


def _lf_coef_names(
    spec: ModelSpec, adjust_for: tuple[str, ...] | None = None
) -> list[str]:
    extra = spec.extra
    adj = extra.get("adjust_for", ()) if adjust_for is None else adjust_for
    names = ["b_grp_time" if extra.get("group_by_time", True) else "beta_grp", "gamma_A"]
    if extra.get("ability_covariate"):
        names.append(
            "gamma_ability_time" if extra.get("ability_by_time", True) else "gamma_ability"
        )
        if extra.get("group_ability", True):
            names.append("gamma_grp_ability")
    names += [f"gamma_{c}" for c in adj]
    return names


def _lf_diag_vars(
    spec: ModelSpec, adjust_for: tuple[str, ...] | None = None
) -> list[str]:
    tail = (
        ["sigma_child"]
        if spec.extra.get("likelihood") == "bernoulli_offfloor"
        else ["kappa", "sigma_child"]
    )
    return ["alpha", "alpha_time", *_lf_coef_names(spec, adjust_for), *tail]


def fit_level_factors(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    _require_spec(spec, "level_factors", outcome=True)
    ctx = make_context(spec, config)
    extra = spec.extra

    section_header("Prepare data")
    ability_covariate = extra.get("ability_covariate")
    likelihood = extra.get("likelihood", "beta_binomial")
    off_floor = likelihood == "bernoulli_offfloor"
    obs_node = "y_offfloor" if off_floor else "y_post"
    baseline_covariates = (ability_covariate,) if ability_covariate else ()
    # Revised-DAG raw-covariate confounders (hearing/speech/phonological memory; #247).
    # Timing (review finding A1; team decision 2026-07-13): the language-proximal SP/RW
    # confounders (deapp_c/erbto + missing indicators) are read at the pre-randomisation
    # BASELINE (t1) — the clean randomised contrast here is the t2 group term, and at t2
    # these language-proximal states may already be treatment-affected, so a
    # contemporaneous read would condition the causal contrast on a descendant of the
    # exposure. Hearing (hs) is exogenous and stays contemporaneous (post). Re-filter
    # after loading so a constant ``_missing`` indicator dropped by the loader is not
    # built or gated. The level model takes no measure-skill adjusters (post-treatment
    # mediators).
    adjust_for = tuple(extra.get("adjust_for", ()))
    pre_adj, post_adj = split_covariates_by_wave(adjust_for)
    baseline_adj, post_adj = split_confounders_by_timing(post_adj)
    prepared = load_and_prepare(
        phase_mode="levels",
        outcomes=(spec.outcome_symbol,),
        baseline_covariates=(*baseline_covariates, *baseline_adj),
        covariates=pre_adj,
        post_covariates=post_adj,
    )
    adjust_for = tuple(c for c in adjust_for if c in prepared.covariates)
    ctx.prepared = prepared
    _print_header(ctx)

    section_header("Build model")
    built = _factories.build_level_factors_model(
        prepared,
        outcome_symbol=spec.outcome_symbol,
        ability_covariate=ability_covariate,
        adjust_for=adjust_for,
        group_by_time=bool(extra.get("group_by_time", True)),
        ability_by_time=bool(extra.get("ability_by_time", True)),
        group_ability=bool(extra.get("group_ability", True)),
        likelihood=likelihood,
    )
    _attach_built(ctx, built)

    _render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    _diag.save_prior_predictive_plot(ctx, spec.outcome_symbol, node=obs_node)

    _run_sampling_and_loo(ctx)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=_lf_diag_vars(spec, adjust_for))

    _run_ppc(ctx, var_names=[obs_node])

    section_header("Extended diagnostics")
    _lf_group_by_time = extra.get("group_by_time", True)
    # For the shipped group-by-time LF models the flagged-causal term is the t2
    # element of the per-timepoint group vector, ``b_grp_time`` (``b_grp_time[1]``,
    # which reporting.level_t2_marginal_effect reads into the causal ROPE card), so
    # it must get the same prior-sensitivity + forest evidence as tau/beta_trt
    # rather than being skipped (issue #273).
    _causal_lf = "b_grp_time" if _lf_group_by_time else "beta_grp"
    _diag.write_diagnostics_summary(ctx, var_names=_lf_diag_vars(spec, adjust_for))
    _diag.run_extended_diagnostics(ctx, causal_term=_causal_lf)
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=_lf_diag_vars(spec, adjust_for))
    _save_forest_plot(ctx, [_causal_lf])
    _diag.run_psense(ctx, var_names=[_causal_lf])

    section_header("Factor summary")
    # Only the t2 group contrast (b_grp_time[1]) is the clean randomised effect;
    # the other timepoints are post-crossover (see the level-model caveat).
    causal = ("b_grp_time[1]",) if extra.get("group_by_time", True) else ()
    fs = _report.factor_summary(
        ctx.trace, _lf_coef_names(spec, adjust_for), ci_prob=ctx.reporting.ci_prob, causal_terms=causal
    )
    fs.to_csv(os.path.join(ctx.output_dir, "factor_summary.csv"), index=False)
    ctx.tables["factor_summary"] = fs
    _save_association_forest(ctx, _lf_coef_names(spec, adjust_for), causal)
    print_table(
        ranked_dataframe_table(
            fs,
            title=f"Factor summary ({spec.outcome_symbol}) - {int(ctx.reporting.ci_prob * 100)}% CrI",
            columns=["term", "role", "median", "lo", "hi", "prob_positive"],
            rank_column=False,
            precision=3,
        )
    )

    meta_extra = {
        "loo_elpd": float(ctx.loo.elpd),
        # Requested vs actually-fitted adjustment set, incl. dropped-constant
        # covariates (#247). The level model carries no skill baselines — only the
        # exogenous raw-covariate confounders (hs/deapp_c/erbto) at the split wave.
        "effective_adjustment": _effective_adjustment(
            spec, built.prepared, adjust_for=adjust_for, ability_covariate=ability_covariate
        ),
    }
    # ROPE-anchored continuous report for the one causal term — the t2 randomised
    # contrast b_grp_time[1] (notes/202606261304-...). The level model enters group
    # as a per-timepoint vector and also carries a group×ability interaction, so the
    # t2 items-scale AME nets both group terms out at the t2 rows
    # (reporting.level_t2_marginal_effect) rather than reusing the gain core. Emitted
    # when the t2 contrast exists (group_by_time): graded outcomes with an agreed items
    # delta (ROPE_DELTA -> W/R/E/L/B) report on the items scale; the floored outcomes P
    # and N report the off-floor risk difference (A4, 2026-07-13) — previously they got
    # no probability-scale card at all; F/T (no agreed delta) are still skipped.
    from language_reading_predictors.statistical_models.measures import (
        ROPE_DELTA,
        ROPE_DELTA_PROB,
        ROPE_DELTA_PROB_GRID,
    )

    delta_items = ROPE_DELTA.get(spec.outcome_symbol)
    delta_prob = ROPE_DELTA_PROB.get(spec.outcome_symbol)
    _gbt = extra.get("group_by_time", True)
    _graded_card = delta_items is not None and not off_floor and _gbt
    _offfloor_card = off_floor and delta_prob is not None and _gbt
    if _graded_card or _offfloor_card:
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
        if _graded_card:
            n_marg = int(built.prepared.n_trials[spec.outcome_symbol])
            delta = delta_items
            title = (
                f"ROPE summary (t2 contrast, {spec.outcome_symbol}, "
                f"delta={delta_items:g} items)"
            )
        else:
            # Off-floor (Bernoulli) t2 contrast: expit(eta) = Pr(off-floor), so the
            # probability-scale AME from level_t2_marginal_effect IS the off-floor risk
            # difference (n_trials = 1), matching the gain-factor off-floor path.
            n_marg = 1
            delta = delta_prob
            title = (
                f"ROPE summary (t2 off-floor risk difference, "
                f"{spec.outcome_symbol}, delta={delta_prob:g})"
            )
        items = ame_prob * n_marg
        rope_s = _report.rope_card(
            contrast_draws, items, delta=delta, ci_prob=ctx.reporting.ci_prob
        )
        if _offfloor_card:
            rope_s["provisional_delta"] = False  # 10 pp signed off (#144, 2026-07-01)
            rope_s["delta_scale"] = "risk_difference"
        rope_df = pd.DataFrame([rope_s])
        rope_df.to_csv(os.path.join(ctx.output_dir, "rope_summary.csv"), index=False)
        ctx.tables["rope_summary"] = rope_df
        meta_extra["rope_summary"] = rope_s
        print_table(
            metrics_table(
                [{"metric": k, "value": v} for k, v in rope_s.items()],
                title=title,
                columns=["metric", "value"],
            )
        )
        _save_rope_plot(ctx, spec.outcome_symbol, None, n_marg, delta, items=items)
        if _offfloor_card:
            # δ-sensitivity sweep on the risk-difference grid (10/15/20 pp), mirroring
            # the gain-factor off-floor path (#144). Built from the same ``items``
            # (risk-difference) draws so it cannot drift from the headline card.
            sens_rows = []
            for d in ROPE_DELTA_PROB_GRID:
                d = float(d)
                p_benefit = float(np.mean(items >= d))
                sens_rows.append(
                    {
                        "delta_items": d,
                        "prob_benefit_ge_delta": p_benefit,
                        "prob_in_rope": float(np.mean(np.abs(items) <= d)),
                        "prob_harm_ge_delta": float(np.mean(items <= -d)),
                        "benefit_label": _report.evidence_label(p_benefit),
                    }
                )
            sens_df = pd.DataFrame(sens_rows)
            sens_df.to_csv(
                os.path.join(ctx.output_dir, "rope_sensitivity.csv"), index=False
            )
            ctx.tables["rope_sensitivity"] = sens_df

    _report.write_run_metadata(ctx, extra=meta_extra)
    return _finalize_report(ctx)


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
    _require_spec(spec, "aligned", outcome=True)
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
    built = _factories.build_aligned_model(
        prepared,
        outcome_symbol=spec.outcome_symbol,
        ability_covariate=ability_covariate,
        use_cohort=use_cohort,
        use_dose=use_dose,
        likelihood=likelihood,
    )
    _attach_built(ctx, built)

    _render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    _diag.save_prior_predictive_plot(ctx, spec.outcome_symbol, node=obs_node)

    _run_sampling_and_loo(ctx)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=_al_diag_vars(spec))

    _run_ppc(ctx, var_names=[obs_node])

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=_al_diag_vars(spec))
    _diag.run_extended_diagnostics(ctx)
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=_al_diag_vars(spec))

    section_header("Factor summary")
    # Per-protocol design: NOTHING is a clean randomised effect, so no term is
    # flagged causal -- every coefficient (cohort included) is an association.
    fs = _report.factor_summary(
        ctx.trace, _al_coef_names(spec), ci_prob=ctx.reporting.ci_prob, causal_terms=()
    )
    fs.to_csv(os.path.join(ctx.output_dir, "factor_summary.csv"), index=False)
    ctx.tables["factor_summary"] = fs
    # Per-protocol: every term is an association, so the forest shows them all.
    _save_association_forest(ctx, _al_coef_names(spec), ())
    print_table(
        ranked_dataframe_table(
            fs,
            title=f"Factor summary ({spec.outcome_symbol}) - {int(ctx.reporting.ci_prob * 100)}% CrI",
            columns=["term", "role", "median", "lo", "hi", "prob_positive"],
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
            ci_prob=ctx.reporting.ci_prob,
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
    return _finalize_report(ctx)


# ---------------------------------------------------------------------------
# Two-mediator decomposition pipeline (LRP64)
# ---------------------------------------------------------------------------


def fit_mediation_multi(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """ITT-phase two-mediator decomposition (LRP64): G -> W via letter-sound and vocab.

    Mirrors :func:`fit_mediation` but builds the two-mediator joint model
    (:func:`factories.build_two_mediator_model`) and runs the two-mediator
    g-formula (:func:`mediation.decompose_two_mediator`), reporting the joint
    indirect effect as the headline plus the (ordering-dependent) path-specific
    indirect effects.
    """
    _require_spec(spec, "mediation_multi")
    from language_reading_predictors.statistical_models import mediation as _med

    ctx = make_context(spec, config)

    section_header("Prepare data")
    # Phase 0 only (t1 -> t2): the single randomised contrast. One row per child.
    mediators = tuple(spec.extra.get("mediators", ("L", "E")))
    # Drop the structural symbols and the two mediator baselines ({m}_t1) from the
    # adjustment set; whatever remains are the measured mediator-outcome confounders
    # C. Keyed off ``mediators`` so a non-(L, E) pair excludes its own baselines
    # (LRP64 -> L_t1/E_t1; LRP66 -> L_t1/B_t1). The set mixes bounded-count measures
    # (E, R — via pre_logit) and revised-DAG raw covariates (hs/deapp_c/erbto; #246 —
    # requested as covariates, taken from the t1 pre-row); ``covariates=()`` is a
    # no-op for models with no raw confounders.
    _mediator_baselines = tuple(f"{m}_t1" for m in mediators)
    confounders = tuple(
        s
        for s in spec.adjustment
        if s not in ("G", "A", "W_pre", *_mediator_baselines)
    )
    _raw_cov = _raw_covariate_confounders(confounders)
    prepared = load_and_prepare(phase_mode="itt", covariates=_raw_cov)
    # Drop any missing-indicator constant on the ITT-phase rows (see fit_mediation).
    confounders = tuple(
        c for c in confounders if c in prepared.covariates or c in prepared.pre_logit
    )
    ctx.prepared = prepared

    _print_header(ctx)

    section_header("Build model")

    built, med_data = _factories.build_two_mediator_model(
        prepared,
        outcome_symbol=spec.outcome_symbol or "W",
        mediator_symbols=mediators,
        confounder_symbols=confounders,
        chain=bool(spec.extra.get("chain", False)),
    )
    _attach_built(ctx, built)

    # Diagnose every scalar coefficient the model actually built, so the list
    # tracks the fitted confounder set instead of a hand-maintained constant
    # (mirrors fit_mediation).
    coef_vars = sorted(rv.name for rv in built.model.free_RVs if rv.ndim == 0)

    _render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    # The mediator likelihood is the FIRST observed RV, so name the outcome node
    # explicitly — else the plot overlays mediator draws on the outcome's counts.
    _diag.save_prior_predictive_plot(ctx, spec.outcome_symbol or "W", node="y_post")

    _run_sampling_and_loo(ctx, compute_loo=False)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=coef_vars)

    _run_ppc(ctx, var_names=[f"{mediators[0]}_post", f"{mediators[1]}_post", "y_post"])

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=coef_vars)
    _diag.run_extended_diagnostics(ctx)
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=coef_vars)

    section_header("Two-mediator decomposition (g-formula)")
    med_df = _med.decompose_two_mediator(
        ctx.trace,
        med_data,
        hdi_prob=ctx.reporting.ci_prob,
        order=tuple(spec.extra.get("order", ("L", "E"))),
    )
    med_df.to_csv(os.path.join(ctx.output_dir, "mediation_summary.csv"), index=False)
    ctx.tables["mediation_summary"] = med_df
    print_table(
        ranked_dataframe_table(
            med_df,
            title=(
                f"Two-mediator decomposition (intervention-helps; words out of "
                f"{med_data.n_trials_W})"
            ),
            columns=["quantity", "words_mean", "words_lo", "words_hi", "prob_pos"],
            rank_column=False,
            precision=3,
        )
    )

    _summary = {r["quantity"]: r for r in med_df.to_dict("records")}
    # Requested vs actually-fitted confounders, recorded separately (#246 review, P2).
    _requested_raw = _raw_covariate_confounders(
        s
        for s in spec.adjustment
        if s not in ("G", "A", "W_pre", *(f"{m}_t1" for m in mediators))
    )
    _report.write_run_metadata(
        ctx,
        extra={
            "adjustment": spec.adjustment,
            "effective_confounders": list(confounders),
            "dropped_confounders": [c for c in _requested_raw if c not in confounders],
            "n_obs": ctx.prepared.n_obs,
            "mediators": list(mediators),
            "n_trials_W": med_data.n_trials_W,
            "mediation": _summary,
        },
    )

    return _finalize_report(ctx)


# ---------------------------------------------------------------------------
# Adjusted pipeline (LRP65) — between-child baseline predictors of gain
# ---------------------------------------------------------------------------

# Human-readable labels for the LRP65 predictor keys (for tables / forest plot).
_ADJ_LABELS = {
    "L": "Letter sounds (T1)",
    "lang": "Language composite (T1)",
    "B": "Blending (T1)",
    "age": "Age (T1)",
    "blocks": "Non-verbal MA (T1)",
    "behav": "Behaviour (T1)",
    # Revised-DAG upstream traits, entered as tested covariates (#247).
    "hs": "Hearing status (T1)",
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


def _sample_model(model, sampling, *, label: str = "sub-fit"):
    """Sample a sub-model (bivariate / sensitivity / prior-sweep) with nutpie.

    Mirrors :func:`diagnostics.sample_posterior` but is standalone, so the sub-fit
    traces never overwrite the headline ``ctx.trace`` / ``trace.nc``. A convergence
    check runs on the result and warns loudly if the sub-fit failed the gate, since
    these traces bypass the primary ``diagnostics_summary.json`` gate.

    Returns ``(trace, conv)`` where ``conv`` is the
    :func:`diagnostics.subfit_convergence` verdict dict (``converged``/``max_rhat``/
    ``min_ess``/``n_divergences``). The caller persists ``conv["converged"]`` onto the
    sub-fit's published CSV: previously the verdict was computed and discarded, so the
    bivariate / prior-sweep / SES sensitivity tables were reported with no convergence
    flag despite bypassing the primary gate (this review's finding B1).
    """
    import pymc as pm

    with model:
        trace = pm.sample(
            draws=sampling.draws,
            tune=sampling.tune,
            chains=sampling.chains,
            cores=sampling.cores,
            target_accept=sampling.target_accept,
            nuts_sampler="nutpie",
            return_inferencedata=True,
            random_seed=sampling.random_seed,
            progressbar=False,
        )
    conv = _diag.subfit_convergence(trace, label=label)
    return trace, conv


def _beta_summary(trace, name: str, ci_prob: float) -> dict:
    """Posterior mean, equal-tailed ``ci_prob``-coverage interval, and P(>0) for ``name``.

    The interval is equal-tailed at ``ci_prob`` coverage, not an HDI — the parameter
    was previously named ``hdi``, which misdescribed it (the callers already pass
    ``ctx.reporting.ci_prob``).
    """
    draws = trace.posterior[name].stack(sample=("chain", "draw")).values
    lo_q, hi_q = (1 - ci_prob) / 2, 1 - (1 - ci_prob) / 2
    return {
        "mean": float(np.mean(draws)),
        "lo": float(np.quantile(draws, lo_q)),
        "hi": float(np.quantile(draws, hi_q)),
        "lo90": float(np.quantile(draws, 0.05)),
        "hi90": float(np.quantile(draws, 0.95)),
        "prob_pos": float(np.mean(draws > 0)),
    }


def _plot_associations(ctx: StatisticalFitContext, df: pd.DataFrame, hdi: float) -> None:
    y = np.arange(len(df))[::-1]
    plt.figure(figsize=(7.0, 0.6 * len(df) + 1.6))
    plt.errorbar(
        df["adj_mean"], y + 0.12,
        xerr=[df["adj_mean"] - df["adj_lo"], df["adj_hi"] - df["adj_mean"]],
        fmt="o", color="#1f77b4", capsize=3, label="adjusted (mutual)",
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
    plt.title("LRP65: baseline predictors of word-reading gain (between-child)")
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
                "delta_words_mean": float(np.mean(delta)),
                "delta_words_lo": float(np.quantile(delta, lo_q)),
                "delta_words_hi": float(np.quantile(delta, hi_q)),
                "delta_words_lo90": float(np.quantile(delta, 0.05)),
                "delta_words_hi90": float(np.quantile(delta, 0.95)),
                "prob_pos": float(np.mean(delta > 0)),
            }
        )
    return pd.DataFrame(rows)


def _influence_diagnostics(ctx: StatisticalFitContext) -> tuple:
    """Per-child PSIS-LOO Pareto-k, to flag whether a few children drive the fit.

    Returns ``(dataframe, threshold, n_flagged)`` — the per-child k sorted
    descending (aligned to ``subject_ids``), the ``good_k`` threshold, and how
    many children exceed it. At n ~ 51 this guards against a headline association
    resting on 2-3 influential children. Returns ``(None, None, None)`` if the
    LOO object exposes no per-observation k.
    """
    if ctx.loo is None or getattr(ctx.loo, "pareto_k", None) is None:
        return None, None, None
    k = np.asarray(ctx.loo.pareto_k).ravel()
    ids = np.asarray(ctx.prepared.subject_ids)
    if len(k) != len(ids):
        return None, None, None
    thr = float(getattr(ctx.loo, "good_k", 0.7) or 0.7)
    df = (
        pd.DataFrame({"subject_id": ids, "pareto_k": k})
        .sort_values("pareto_k", ascending=False)
        .reset_index(drop=True)
    )
    return df, thr, int((k > thr).sum())


def fit_horseshoe(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Regularized-horseshoe predictor-ranking fit (LRPHS, #116 Phase E).

    An independent Bayesian sensitivity cross-check on the gradient-boosting
    predictor ranking: one horseshoe regression (gain or level, per ``spec.extra``)
    over the full construct predictor set, ranked by posterior
    ``P(|beta| > delta)``. Writes ``predictor_ranking.csv`` alongside the standard
    trace / diagnostics / LOO / PPC artefacts. Not causal — a which-predictors
    -carry-signal read to compare against the GB cluster ranking.
    """
    _require_spec(spec, "horseshoe")
    e = spec.extra
    outcome = spec.outcome_symbol or "W"
    gain = bool(e.get("gain", True))
    predictors = list(e["predictors"])
    lang_symbols = tuple(e.get("language_composite_symbols", ["R", "E", "F"]))
    covariates = tuple(e.get("covariates", ()))
    delta = float(e.get("delta", 0.1))
    tau0 = float(e.get("tau0", 0.1))
    slab_scale = float(e.get("slab_scale", 2.0))
    slab_df = float(e.get("slab_df", 4.0))
    post_time = int(e.get("post_time", 4))
    phase_mode = e.get("phase_mode", "span" if gain else "levels")

    # 94% intervals, matching the LRP65 adjusted-model convention.
    ctx = make_context(spec, config, ci_prob=0.94)
    # The horseshoe has a funnel geometry (global-local scales); lift target_accept
    # above the tier default so the sampler takes smaller steps near the neck.
    _apply_spec_target_accept(ctx, spec)

    section_header("Prepare data")
    measure_syms = tuple(
        dict.fromkeys(
            [outcome]
            + [p for p in predictors if p not in ("age", "lang", *covariates)]
            + list(lang_symbols)
        )
    )
    prepared = load_and_prepare(
        phase_mode=phase_mode,
        post_time=post_time,
        outcomes=measure_syms,
        covariates=covariates,
    )
    ctx.prepared = prepared
    _print_header(ctx)

    section_header("Build model")
    built = _factories.build_horseshoe_model(
        prepared,
        outcome_symbol=outcome,
        predictors=predictors,
        gain=gain,
        tau0=tau0,
        slab_scale=slab_scale,
        slab_df=slab_df,
        language_composite_symbols=lang_symbols,
    )
    _attach_built(ctx, built)
    _render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    _diag.save_prior_predictive_plot(ctx, outcome)

    _run_sampling_and_loo(ctx)

    # Coupling term present in the model: gamma_own (gain) or the fixed age slope
    # gamma_A (level) — but the level model suppresses gamma_A when age is itself a
    # horseshoe-ranked predictor (build_horseshoe_model), so only list it then.
    if gain:
        coupling_vars = ["gamma_own"]
    elif "age" not in predictors:
        coupling_vars = ["gamma_A"]
    else:
        coupling_vars = []
    diag_vars = ["alpha", *coupling_vars, "kappa", "hs_tau", "hs_c2", "beta"]
    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=diag_vars)

    _run_ppc(ctx)

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=diag_vars)
    _diag.run_extended_diagnostics(ctx)
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=diag_vars)

    section_header("Predictor ranking")
    ranking = _report.horseshoe_ranking(ctx.trace, delta=delta)
    ranking.to_csv(os.path.join(ctx.output_dir, "predictor_ranking.csv"), index=False)
    ctx.tables["predictor_ranking"] = ranking
    print_table(ranked_dataframe_table(ranking.head(10), title="Horseshoe predictor ranking (top 10)"))

    meta_extra = {
        "framing": "gain" if gain else "level",
        "phase_mode": phase_mode,
        "predictors": predictors,
        "covariates": list(covariates),
        "delta": delta,
        "tau0": tau0,
        "slab_scale": slab_scale,
        "slab_df": slab_df,
        "gb_reference": e.get("gb_reference"),
        "ranking_top": ranking.head(3)[["predictor", "p_abs_gt_delta"]].to_dict(
            "records"
        ),
    }
    _report.write_run_metadata(ctx, extra=meta_extra)

    return _finalize_report(ctx)


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
    _require_spec(spec, "adjusted")
    e = spec.extra
    outcome = spec.outcome_symbol or "W"
    post_time = int(e.get("post_time", 4))
    predictor_symbols = list(e.get("predictor_symbols", ["L", "B"]))
    lang_symbols = tuple(e.get("language_composite_symbols", ["R", "E", "F"]))
    covariates = list(e.get("covariates", ["blocks", "behav"]))
    ses_covs = list(e.get("ses_covariates", ["mumedupost16"]))
    # The slope-prior default is sourced from the factory signature (single source
    # of truth) so this fallback cannot drift from the reconciled scale — prior-
    # critical-review 2026-07-07, recommendation 3; #209 review. The sweep default
    # brackets that scale from the looser side (no factory param mirrors it).
    sigma0 = float(
        e.get(
            "predictor_slope_sigma",
            _default_of(_factories.build_adjusted_model, "predictor_slope_sigma"),
        )
    )
    prior_sens = list(e.get("prior_sensitivity_sigmas", [0.5, 0.7]))
    use_age = bool(e.get("use_age_predictor", True))

    # 94% intervals (the brief's convention) rather than the project-wide 95%.
    ctx = make_context(spec, config, ci_prob=0.94)
    hdi = ctx.reporting.ci_prob

    section_header("Prepare data")
    measure_outcomes = tuple(
        dict.fromkeys([outcome, *predictor_symbols, *lang_symbols])
    )
    prepared = load_and_prepare(
        phase_mode="span",
        post_time=post_time,
        outcomes=measure_outcomes,
        covariates=tuple(covariates),
    )
    ctx.prepared = prepared
    # Drop any covariate the loader removed as constant on the fitted rows (e.g. a
    # `_missing` indicator that is all-zero once the complete cases are kept) so the
    # model never requests a coefficient for a term that was never estimated (#247).
    covariates = [c for c in covariates if c in prepared.covariates]
    # Headline predictor key order: skills, language composite, age, tested covariates.
    headline = (
        list(predictor_symbols)
        + ["lang"]
        + (["age"] if use_age else [])
        + covariates
    )
    _print_header(ctx)

    section_header("Build model")
    built = _factories.build_adjusted_model(
        prepared,
        outcome_symbol=outcome,
        predictors=headline,
        language_composite_symbols=lang_symbols,
        predictor_slope_sigma=sigma0,
    )
    _attach_built(ctx, built)

    _render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    _diag.save_prior_predictive_plot(ctx, outcome)

    _run_sampling_and_loo(ctx)

    section_header("Summary diagnostics")
    beta_names = [f"beta_{k}" for k in headline]
    _diag.summary_diagnostics(
        ctx, var_names=["alpha", "gamma_own", "kappa", *beta_names]
    )

    _run_ppc(ctx)
    _adjusted_diag_vars = ["alpha", "gamma_own", "kappa", *beta_names]

    section_header("Extended diagnostics")
    # Capture the primary gate verdict so the sub-fit tables can label their
    # primary-derived rows (the adjusted/mutual associations and the headline-sigma
    # prior-sweep rows come from ``ctx.trace``, which this gate covers) consistently
    # with the sub-fits' own ``subfit_convergence`` flags (this review's finding B1).
    _primary_gate = _diag.write_diagnostics_summary(ctx, var_names=_adjusted_diag_vars)
    _primary_converged = bool(_primary_gate.get("passed")) if _primary_gate else None
    _diag.run_extended_diagnostics(ctx)
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=_adjusted_diag_vars)

    # --- Adjusted vs bivariate associations --------------------------------
    section_header("Predictor associations (adjusted vs bivariate)")
    adjusted = {k: _beta_summary(ctx.trace, f"beta_{k}", hdi) for k in headline}
    bivariate: dict[str, dict] = {}
    biv_converged: dict[str, object] = {}
    for k in headline:
        b = _factories.build_adjusted_model(
            prepared,
            outcome_symbol=outcome,
            predictors=[k],
            language_composite_symbols=lang_symbols,
            predictor_slope_sigma=sigma0,
        )
        t, conv = _sample_model(b.model, ctx.sampling, label=f"{spec.model_id} bivariate {k}")
        bivariate[k] = _beta_summary(t, f"beta_{k}", hdi)
        biv_converged[k] = conv["converged"]

    rows = []
    for k in headline:
        a, bv = adjusted[k], bivariate[k]
        rows.append(
            {
                "predictor": k,
                "label": _adj_label(k),
                "adj_mean": a["mean"],
                "adj_lo": a["lo"],
                "adj_hi": a["hi"],
                "adj_lo90": a["lo90"],
                "adj_hi90": a["hi90"],
                "adj_prob_pos": a["prob_pos"],
                "biv_mean": bv["mean"],
                "biv_lo": bv["lo"],
                "biv_hi": bv["hi"],
                "biv_lo90": bv["lo90"],
                "biv_hi90": bv["hi90"],
                "biv_prob_pos": bv["prob_pos"],
                # Convergence flags: the adjusted column is the primary (gated) fit;
                # the bivariate column is a sub-fit that bypasses the primary gate (B1).
                "adj_converged": _primary_converged,
                "biv_converged": biv_converged[k],
            }
        )
    assoc_df = pd.DataFrame(rows)
    assoc_df.to_csv(
        os.path.join(ctx.output_dir, "predictor_associations.csv"), index=False
    )
    ctx.tables["predictor_associations"] = assoc_df
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
    section_header("Prior sensitivity")
    ps_rows = []
    for sig in [sigma0, *prior_sens]:
        if sig == sigma0:
            tr = ctx.trace
            sig_converged = _primary_converged  # headline sigma is the gated primary
        else:
            b = _factories.build_adjusted_model(
                prepared,
                outcome_symbol=outcome,
                predictors=headline,
                language_composite_symbols=lang_symbols,
                predictor_slope_sigma=sig,
            )
            tr, conv = _sample_model(
                b.model, ctx.sampling, label=f"{spec.model_id} prior-sweep sigma={sig}"
            )
            sig_converged = conv["converged"]
        for k in headline:
            ps_rows.append(
                {
                    "sigma": sig,
                    "predictor": k,
                    **_beta_summary(tr, f"beta_{k}", hdi),
                    "converged": sig_converged,
                }
            )
    ps_df = pd.DataFrame(ps_rows)
    ps_df.to_csv(os.path.join(ctx.output_dir, "prior_sensitivity.csv"), index=False)
    ctx.tables["prior_sensitivity"] = ps_df

    # --- SES complete-case sensitivity -------------------------------------
    section_header("SES sensitivity (complete cases)")
    ses_df = None
    ses_n = None
    ses_error = None
    try:
        prepared_ses = load_and_prepare(
            phase_mode="span",
            post_time=post_time,
            outcomes=measure_outcomes,
            covariates=tuple(covariates + ses_covs),
        )
        # Re-filter against the SES-complete subset: a `_missing` indicator can go
        # constant on this smaller subset even if it survived the headline fit, and the
        # loader then drops it — so rebuild the predictor list here too, or
        # ``build_adjusted_model`` would KeyError on the dropped term (#287 review). The
        # non-covariate predictors (skills / lang / age) are always kept.
        ses_headline = [
            k for k in headline if k not in covariates or k in prepared_ses.covariates
        ]
        ses_covs_fit = [c for c in ses_covs if c in prepared_ses.covariates]
        ses_predictors = ses_headline + ses_covs_fit
        b = _factories.build_adjusted_model(
            prepared_ses,
            outcome_symbol=outcome,
            predictors=ses_predictors,
            language_composite_symbols=lang_symbols,
            predictor_slope_sigma=sigma0,
        )
        t, conv = _sample_model(b.model, ctx.sampling, label=f"{spec.model_id} SES complete-case")
        ses_n = int(b.prepared.n_children)
        ses_rows = [
            {
                "predictor": k,
                "label": _adj_label(k),
                "n_children": ses_n,
                **_beta_summary(t, f"beta_{k}", hdi),
                "converged": conv["converged"],
            }
            for k in ses_predictors
        ]
        ses_df = pd.DataFrame(ses_rows)
        ses_df.to_csv(
            os.path.join(ctx.output_dir, "ses_sensitivity.csv"), index=False
        )
        ctx.tables["ses_sensitivity"] = ses_df
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
    words_df = _natural_scale_contrasts(ctx, ctx.prepared, headline, outcome, hdi)
    words_df.to_csv(
        os.path.join(ctx.output_dir, "predicted_gain_words.csv"), index=False
    )
    ctx.tables["predicted_gain_words"] = words_df
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
    section_header("Influence (PSIS-LOO Pareto-k)")
    infl_df, k_thr, n_flagged = _influence_diagnostics(ctx)
    if infl_df is not None:
        infl_df.to_csv(os.path.join(ctx.output_dir, "influence.csv"), index=False)
        ctx.tables["influence"] = infl_df
        rprint(
            f"  max Pareto-k = {infl_df['pareto_k'].max():.2f}; "
            f"{n_flagged} of {len(infl_df)} children exceed k = {k_thr:.2f}"
        )
    else:
        rprint("[yellow]Pareto-k unavailable from LOO; influence check skipped[/yellow]")

    _report.write_run_metadata(
        ctx,
        extra={
            "loo_elpd": float(ctx.loo.elpd) if ctx.loo is not None else None,
            "design": "between_child",
            "post_time": post_time,
            "predictors": headline,
            "predictor_slope_sigma": sigma0,
            "prior_sensitivity_sigmas": prior_sens,
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

    return _finalize_report(ctx)


# ---------------------------------------------------------------------------
# Longitudinal dynamic pipeline (LRP67 LCSM)
# ---------------------------------------------------------------------------


def _coef_row(label: str, draws, hdi_prob: float) -> dict:
    """Posterior mean, equal-tailed central interval and ``P(coef > 0)``.

    Equal-tailed quantiles at coverage ``hdi_prob`` — the same convention as
    :func:`reporting.tau_summary_itt` (not a highest-density interval).
    """
    d = np.asarray(draws).reshape(-1)
    lo_q = (1 - hdi_prob) / 2
    return {
        "coefficient": label,
        "mean": float(np.mean(d)),
        "lo": float(np.quantile(d, lo_q)),
        "hi": float(np.quantile(d, 1 - lo_q)),
        "lo90": float(np.quantile(d, 0.05)),
        "hi90": float(np.quantile(d, 0.95)),
        "prob_pos": float(np.mean(d > 0)),
    }


def fit_lcsm(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Latent change-score model (LRP67 + the lagged coupling suite, #250).

    Fits the coupled McArdle latent change-score model with process noise and
    reports the per-target coupling tables. ``spec.extra`` selects the shape:
    the LRP67 default couples every other measure into the reading change; the
    lagged reverse-coupling models (LCSM-081/181/082) pass an explicit
    ``couplings`` map plus ``arm_window_intercepts`` (the crossover-aware
    arm x window change intercepts, with the window-1 randomised contrast
    written to ``itt_window1_contrast.csv``) and a shared adjuster
    ``covariate_block``. ``dominance_pair`` adds the SD-standardised
    reciprocal-dominance contrast (``dominance_summary.csv``).
    """
    _require_spec(spec, "lcsm")

    ctx = make_context(spec, config)

    section_header("Prepare data")
    outcomes = tuple(spec.extra.get("outcomes", ("W", "L", "E")))
    reading_symbol = spec.outcome_symbol or "W"
    couplings_in = spec.extra.get("couplings")
    couplings: dict[str, tuple[str, ...]] = (
        {tgt: tuple(srcs) for tgt, srcs in couplings_in.items()}
        if couplings_in
        else {reading_symbol: tuple(s for s in outcomes if s != reading_symbol)}
    )
    arm_window = bool(spec.extra.get("arm_window_intercepts", False))
    covariate_block = tuple(spec.extra.get("covariate_block", ()))
    covariate_targets = tuple(spec.extra.get("covariate_targets", ()))
    # Loader needs from the covariate block: the hearing dummies come via
    # include_hearing; everything else names a per-wave source column (its
    # ``_missing`` companion is derived, not loaded).
    include_hearing = any(n in ("hs", "hs_missing") for n in covariate_block)
    wave_cov_cols = tuple(
        dict.fromkeys(
            n
            for n in covariate_block
            if n not in ("hs", "hs_missing") and not n.endswith("_missing")
        )
    )
    panel = load_wave_panel(
        outcomes=outcomes,
        wave_covariates=wave_cov_cols,
        include_hearing=include_hearing,
    )
    ctx.prepared = panel

    _print_header(ctx)

    section_header("Build model")
    built = _factories.build_lcsm_model(
        panel,
        reading_symbol=reading_symbol,
        couplings=couplings,
        arm_window_intercepts=arm_window,
        covariate_block=covariate_block,
        covariate_targets=covariate_targets,
        coupling_prior_sigma=spec.extra.get(
            "coupling_prior_sigma",
            _default_of(_factories.build_lcsm_model, "coupling_prior_sigma"),
        ),
        use_process_noise=spec.extra.get("use_process_noise", True),
        shared_process_noise=spec.extra.get("shared_process_noise", False),
    )
    _attach_built(ctx, built)

    _render_model_graph(ctx)

    # Coupling parameter names mirror the factory's rule: single target keeps
    # LRP67's ``g_{src}``; multiple targets carry the target (``g_{src}_{tgt}``).
    single_target = len(couplings) == 1
    coupling_names = {
        (src, tgt): (f"g_{src}" if single_target else f"g_{src}_{tgt}")
        for tgt, srcs in couplings.items()
        for src in srcs
    }
    diag_vars = list(coupling_names.values())
    diag_vars += [f"b_{name}" for name in covariate_block]
    diag_vars += ["a_change", "b_self", "d_age", "sigma1", "kappa"]
    if spec.extra.get("use_process_noise", True):
        diag_vars.append("sigma_proc")

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)

    _run_sampling_and_loo(ctx)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=diag_vars)

    _run_ppc(ctx, var_names=["y_obs"])

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=diag_vars)
    _diag.run_extended_diagnostics(
        ctx, causal_term=diag_vars[0] if diag_vars else None
    )
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=diag_vars)

    # Per-target coupling table — the headline "what predicts whose change"
    # output. For LRP67 (single reading target) this reproduces the historical
    # reading-change table, labels included.
    section_header("Change-coupling summary")
    post = ctx.trace.posterior
    rows = [
        _coef_row(
            f"{pname} (prior {src} -> {tgt} change)",
            post[pname].values,
            ctx.reporting.ci_prob,
        )
        for (src, tgt), pname in coupling_names.items()
    ]
    for name in covariate_block:
        rows.append(
            _coef_row(
                f"b_{name} ({name} -> {'/'.join(covariate_targets)} change)",
                post[f"b_{name}"].values,
                ctx.reporting.ci_prob,
            )
        )
    for tgt in couplings:
        # LRP67's historical row labels are kept verbatim for the single
        # reading-target shape.
        legacy = single_target and tgt == reading_symbol
        rows.append(
            _coef_row(
                f"b_self[{tgt}] (reading self-feedback)"
                if legacy
                else f"b_self[{tgt}] ({tgt} self-feedback)",
                post["b_self"].sel(outcome=tgt).values,
                ctx.reporting.ci_prob,
            )
        )
        if not arm_window:
            rows.append(
                _coef_row(
                    f"a_change[{tgt}] (reading baseline change)"
                    if legacy
                    else f"a_change[{tgt}] ({tgt} baseline change)",
                    post["a_change"].sel(outcome=tgt).values,
                    ctx.reporting.ci_prob,
                )
            )
        rows.append(
            _coef_row(
                f"d_age[{tgt}] (age -> reading change)"
                if legacy
                else f"d_age[{tgt}] (age -> {tgt} change)",
                post["d_age"].sel(outcome=tgt).values,
                ctx.reporting.ci_prob,
            )
        )
    coupling_df = pd.DataFrame(rows)
    coupling_df.to_csv(os.path.join(ctx.output_dir, "coupling_summary.csv"), index=False)
    ctx.tables["coupling_summary"] = coupling_df
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
    # waitlist), the built-in consistency check against the ITT suite. Only the
    # arm x window shape carries it.
    itt_rows: list[dict] = []
    if arm_window:
        section_header("Window-1 randomised contrast (ITT consistency check)")
        for s in outcomes:
            itt_rows.append(
                _coef_row(
                    f"itt_w1[{s}] (immediate - waitlist, window-1 latent change)",
                    post["itt_w1_contrast"].sel(outcome=s).values,
                    ctx.reporting.ci_prob,
                )
            )
        itt_df = pd.DataFrame(itt_rows)
        itt_df.to_csv(
            os.path.join(ctx.output_dir, "itt_window1_contrast.csv"), index=False
        )
        ctx.tables["itt_window1_contrast"] = itt_df
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
    dominance_pair = spec.extra.get("dominance_pair")
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
            _coef_row(f"std g ({a} -> {b} change)", g_ab.values, ctx.reporting.ci_prob),
            _coef_row(f"std g ({b} -> {a} change)", g_ba.values, ctx.reporting.ci_prob),
            _coef_row(
                f"|std g {a}->{b}| - |std g {b}->{a}| (dominance)",
                contrast.values,
                ctx.reporting.ci_prob,
            ),
        ]
        dom_df = pd.DataFrame(dom_rows)
        dom_df.to_csv(
            os.path.join(ctx.output_dir, "dominance_summary.csv"), index=False
        )
        ctx.tables["dominance_summary"] = dom_df
        print_table(
            ranked_dataframe_table(
                dom_df,
                title="SD-standardised reciprocal couplings",
                columns=["coefficient", "mean", "lo", "hi", "prob_pos"],
                rank_column=False,
                precision=3,
            )
        )

    _report.write_run_metadata(
        ctx,
        extra={
            "loo_elpd": float(ctx.loo.elpd),
            "outcomes": list(outcomes),
            "reading_symbol": reading_symbol,
            "couplings": {tgt: list(srcs) for tgt, srcs in couplings.items()},
            "arm_window_intercepts": arm_window,
            "covariate_block": list(covariate_block),
            "covariate_targets": list(covariate_targets),
            "coupling_summary": rows,
            **({"itt_window1_contrast": itt_rows} if itt_rows else {}),
            **({"dominance_summary": dom_rows} if dom_rows else {}),
        },
    )

    return _finalize_report(ctx)


def fit_growth(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Joint multivariate latent growth-curve model (LRP69 core / LRP70 factor).

    Characterises each verbal/reading measure's within-child trajectory across the
    four RLI waves and reports whether **baseline non-verbal ability** (``blocks``)
    predicts trajectory shape: ``gamma`` on the growth *rate* (the headline Q5
    estimand) and ``delta`` on the baseline *level*. With ``use_shared_factor`` a
    rank-1 shared growth-tempo factor couples the slopes and the block-design ->
    common-tempo association is read out post-hoc. Every non-randomised term is an
    **adjusted, latent-GA-confounded association**, never causal (locked DAG,
    ``notes/202606231600-dag-revision-consolidated.md``).
    """
    _require_spec(spec, "growth")

    ctx = make_context(spec, config)

    section_header("Prepare data")
    outcomes = tuple(spec.extra.get("outcomes", ("R", "E", "T", "W", "L")))
    baseline_cov = spec.extra.get("baseline_covariate", "blocks")
    use_factor = bool(spec.extra.get("use_shared_factor", False))
    age_ability = bool(spec.extra.get("age_ability_interaction", False))
    panel = load_wave_panel(outcomes=outcomes, baseline_covariates=(baseline_cov,))
    ctx.prepared = panel

    _print_header(ctx)

    section_header("Build model")
    built = _factories.build_growth_model(
        panel,
        baseline_covariate=baseline_cov,
        use_shared_factor=use_factor,
        age_ability_interaction=age_ability,
    )
    _attach_built(ctx, built)

    _render_model_graph(ctx)

    diag_vars = [
        "gamma", "delta", "beta", "alpha", "sigma_slope", "sigma_intercept", "kappa"
    ]
    if use_factor:
        diag_vars.append("loading")
    if age_ability:
        # LRP85 (#228 item 10): baseline-age main effect + the headline
        # age0 × ability interaction on the growth rate.
        diag_vars.extend(["gamma_age", "gamma_int"])

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)

    _run_sampling_and_loo(ctx)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=diag_vars)

    _run_ppc(ctx, var_names=["y_obs"])

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=diag_vars)
    _diag.run_extended_diagnostics(ctx, causal_term=None)
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=diag_vars)

    # Headline Q5 output: baseline non-verbal ability -> trajectory shape. The
    # gamma (growth-rate) rows are the answer; delta (level) and beta (mean slope)
    # round out the trajectory characterisation. All adjusted associations.
    section_header("Non-verbal ability -> trajectory shape (Q5)")
    gs = _report.growth_association_summary(ctx.trace, ci_prob=ctx.reporting.ci_prob)
    gs.to_csv(
        os.path.join(ctx.output_dir, "growth_association_summary.csv"), index=False
    )
    ctx.tables["growth_association_summary"] = gs
    _save_forest_plot(ctx, ["gamma"], name="gamma_forest.png")
    print_table(
        ranked_dataframe_table(
            gs[gs["coefficient"] == "gamma"],
            title="Baseline non-verbal ability -> growth rate (gamma, logit; 95% ETI)",
            columns=[
                "outcome", "median", "lo95", "hi95", "prob_positive",
                "favoured_direction_label",
            ],
            rank_column=False,
            precision=3,
        )
    )

    # Factor layer: is there *residual* coupling between baseline non-verbal
    # ability and the common growth tempo, beyond what the model already
    # attributes to block-design directly (the gamma/delta terms)? Block-design
    # enters the trajectory as a predictor, so G_tempo is the shared tempo net of
    # that modelled effect — this correlation is therefore a *residual*
    # association, not the total "does ability predict tempo". Read out post-hoc as
    # the per-draw correlation between each child's latent tempo G_i and their
    # standardised block-design score: independent a priori, but the posterior can
    # still correlate G and blocks through the likelihood. Descriptive only.
    tempo_corr: dict[str, float] | None = None
    if use_factor and "G_tempo" in ctx.trace.posterior:
        G = (
            ctx.trace.posterior["G_tempo"]
            .stack(sample=("chain", "draw"))
            .transpose("child", "sample")
            .values
        )  # (N, S)
        zb = np.asarray(panel.baseline[baseline_cov], dtype=float)  # (N,)
        Gc = G - G.mean(axis=0, keepdims=True)
        zc = (zb - zb.mean())[:, None]
        denom = np.sqrt((Gc**2).sum(0) * (zc**2).sum(0)) + 1e-12
        corr = (Gc * zc).sum(0) / denom  # (S,)
        lo_q = (1 - ctx.reporting.ci_prob) / 2
        tempo_corr = {
            "median": float(np.median(corr)),
            "lo": float(np.quantile(corr, lo_q)),
            "hi": float(np.quantile(corr, 1 - lo_q)),
            "lo90": float(np.quantile(corr, 0.05)),
            "hi90": float(np.quantile(corr, 0.95)),
            "prob_pos": float(np.mean(corr > 0)),
        }
        pd.DataFrame([tempo_corr]).to_csv(
            os.path.join(ctx.output_dir, "growth_tempo_corr.csv"), index=False
        )
        ctx.tables["growth_tempo_corr"] = pd.DataFrame([tempo_corr])
        rprint(
            f"[bold]blocks <-> growth-tempo residual corr:[/bold] {tempo_corr['median']:+.3f} "
            f"[{tempo_corr['lo']:+.3f}, {tempo_corr['hi']:+.3f}] "
            f"P(>0)={tempo_corr['prob_pos']:.3f}"
        )

    _report.write_run_metadata(
        ctx,
        extra={
            "loo_elpd": float(ctx.loo.elpd),
            "outcomes": list(outcomes),
            "baseline_covariate": baseline_cov,
            "use_shared_factor": use_factor,
            "growth_association_summary": gs.to_dict("records"),
            **({"blocks_tempo_corr": tempo_corr} if tempo_corr else {}),
        },
    )

    return _finalize_report(ctx)


# ---------------------------------------------------------------------------
# Historical group-by-wave growth (RLMHG, #165 - first non-RLI dataset)
# ---------------------------------------------------------------------------


def fit_historical_growth(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Historical group-by-wave growth model (RLMHG, #165).

    A descriptive natural-history growth model for a non-RLI historical cohort
    (the Byrne reading-language-memory study), run through the shared
    statistical-model pipeline so it uses the same sampler, convergence gate,
    output layout and report conventions as the intervention models. It is
    **not** an intervention-effect model - ``group`` carries no treatment
    semantics (see :func:`factories.build_historical_growth_model`).
    """
    _require_spec(spec, "historical_growth")

    ctx = make_context(spec, config)

    section_header("Prepare data")
    study_id = spec.extra.get("study_id", spec.study_id)
    measure = spec.extra.get("measure", spec.outcome_symbol or "basread")
    waves = tuple(spec.extra.get("waves", (1, 2, 3)))
    dataset, measures = _datasets.resolve_dataset(study_id)
    if measure not in measures:
        raise KeyError(f"measure {measure!r} not registered for study {study_id!r}")
    panel = load_longitudinal_panel(
        dataset, [measures[measure]], waves=waves, complete_case=True
    )
    ctx.prepared = panel

    _print_header(ctx)

    section_header("Build model")
    built = _factories.build_historical_growth_model(
        panel,
        measure=measure,
        eta_prior_sigma=spec.extra.get(
            "eta_prior_sigma",
            _default_of(_factories.build_historical_growth_model, "eta_prior_sigma"),
        ),
        sigma_subject_prior_sigma=spec.extra.get(
            "sigma_subject_prior_sigma",
            _default_of(
                _factories.build_historical_growth_model, "sigma_subject_prior_sigma"
            ),
        ),
        kappa_prior_sigma=spec.extra.get(
            "kappa_prior_sigma",
            _default_of(_factories.build_historical_growth_model, "kappa_prior_sigma"),
        ),
    )
    _attach_built(ctx, built)

    _render_model_graph(ctx)

    diag_vars = ["eta_group_wave", "sigma_subject", "kappa"]
    diag_vars += [
        v
        for v in (
            "growth_first_next_items",
            "growth_next_last_items",
            "growth_first_last_items",
        )
        if v in ctx.model.named_vars
    ]

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)

    _run_sampling_and_loo(ctx)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=diag_vars)

    _run_ppc(ctx, var_names=["score"])

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=diag_vars)
    _diag.run_extended_diagnostics(ctx)
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=diag_vars)

    # Descriptive summaries: observed complete-case baseline (the Table 2 audit
    # target), posterior group-by-wave fitted means, and within-group / between-
    # group growth in items.
    section_header("Growth summaries")
    measure_label = measures[measure].label
    baseline = _historical.observed_baseline(panel, measure, measure_label)
    baseline.to_csv(
        os.path.join(ctx.output_dir, "observed_complete_case_baseline.csv"),
        index=False,
    )
    ctx.tables["observed_complete_case_baseline"] = baseline
    cells = _historical.cell_summary(ctx.trace, panel, measure, measure_label, baseline)
    cells.to_csv(
        os.path.join(ctx.output_dir, "posterior_cell_summary.csv"), index=False
    )
    ctx.tables["posterior_cell_summary"] = cells
    growth = _historical.growth_summary(ctx.trace, panel, measure)
    growth.to_csv(
        os.path.join(ctx.output_dir, "posterior_growth_summary.csv"), index=False
    )
    ctx.tables["posterior_growth_summary"] = growth
    print_table(
        ranked_dataframe_table(
            growth,
            title=(
                f"{measure_label} growth (items) - "
                f"{int(ctx.reporting.ci_prob * 100)}% CI (equal-tailed)"
            ),
            columns=["label", "readgrp_label", "mean", "q2_5", "q97_5", "p_gt_0"],
            rank_column=False,
            precision=2,
        )
    )

    _report.write_run_metadata(
        ctx,
        extra={
            "loo_elpd": float(ctx.loo.elpd) if ctx.loo is not None else None,
            "study_id": study_id,
            "measure": measure,
            "measure_label": measure_label,
            "n_trials": panel.n_trials[measure],
            "waves": list(waves),
            "groups": dict(
                zip(panel.group_codes, panel.group_labels, strict=True)
            ),
            "n_subjects": panel.n_subjects,
        },
    )

    return _finalize_report(ctx)


# ---------------------------------------------------------------------------
# Correlated-domain-factor measurement model (LRPMM01, #134)
# ---------------------------------------------------------------------------


_DEFAULT_DOMAINS = {
    "vocabulary": ("R", "E"),
    "code": ("L", "B"),
    "grammar": ("F", "T"),
}


def fit_correlated_factor(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Correlated-domain-factor measurement model (LRPMM01, #134).

    Fits a reflective CFA with correlated vocabulary / code / grammar factors over
    the standardised T1 skill indicators, plus a structural Beta-Binomial leg for
    the reading-gain outcome, and reports the loadings / communalities, the factor
    correlation matrix, and the measurement-error-corrected factor->gain slopes.
    A triangulation / measurement model, not causal (every factor->gain slope is a
    latent-ability-confounded adjusted association; #115 ID-2).
    """
    _require_spec(spec, "corr_factor")

    ctx = make_context(spec, config)
    # The correlated-factor CFA is a small-n latent model; even with the factor
    # scores marginalised out of the measurement likelihood a few boundary
    # divergences survive at the tier-default target_accept, so lift it via the spec
    # (the strict gate requires zero), as the horseshoe fit does for its funnel.
    _apply_spec_target_accept(ctx, spec)

    section_header("Prepare data")
    domains = {
        k: tuple(v) for k, v in (spec.extra.get("domains") or _DEFAULT_DOMAINS).items()
    }
    outcome = spec.outcome_symbol or "W"
    structural_covs = tuple(spec.extra.get("structural_covariates", ("blocks",)))
    indicator_syms = tuple(dict.fromkeys(s for v in domains.values() for s in v))
    measure_outcomes = tuple(dict.fromkeys((outcome, *indicator_syms)))
    prepared = load_and_prepare(
        phase_mode="span",
        post_time=int(spec.extra.get("post_time", 4)),
        outcomes=measure_outcomes,
        covariates=structural_covs,
    )
    ctx.prepared = prepared
    _print_header(ctx)

    section_header("Build model")
    built = _factories.build_correlated_factor_model(
        prepared,
        outcome_symbol=outcome,
        domains=domains,
        structural_covariates=structural_covs,
        use_age=spec.extra.get("use_age", True),
        loading_mu=spec.extra.get(
            "loading_mu",
            _default_of(_factories.build_correlated_factor_model, "loading_mu"),
        ),
        loading_sigma=spec.extra.get(
            "loading_sigma",
            _default_of(_factories.build_correlated_factor_model, "loading_sigma"),
        ),
        residual_sigma=spec.extra.get(
            "residual_sigma",
            _default_of(_factories.build_correlated_factor_model, "residual_sigma"),
        ),
        predictor_slope_sigma=spec.extra.get(
            "predictor_slope_sigma",
            _default_of(
                _factories.build_correlated_factor_model, "predictor_slope_sigma"
            ),
        ),
    )
    _attach_built(ctx, built)
    _render_model_graph(ctx)

    summary_vars = [
        "alpha", "gamma_own", "kappa", "beta_factor", "lambda_load", "sigma_indicator",
        # The headline factor correlations MUST be in the gated set: they are what
        # the report releases, and the global checks (divergences, BFMI) are not a
        # substitute for parameter-specific R-hat / ESS on them. ``factor_corr``
        # itself is unusable for this — its constant unit diagonal has undefined
        # R-hat and zero variance — so the factory exposes the unique off-diagonals
        # as ``factor_corr_pairs``. ``factor_z`` is the latent-score offset the
        # structural leg consumes; gate it too.
        "factor_z",
    ]
    # Only present when there are >= 2 domains (a single factor has no off-diagonal).
    if len(domains) > 1:
        summary_vars.append("factor_corr_pairs")
    if spec.extra.get("use_age", True):
        summary_vars.append("beta_age")
    summary_vars += [f"beta_{c}" for c in structural_covs]

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000, var_names=["Z_obs", "y_post"])
    _diag.save_prior_predictive_plot(ctx, outcome, node="y_post")

    # Two observed nodes (the indicator matrix Z_obs + the structural y_post) make
    # a single-target PSIS-LOO ambiguous, so LOO is skipped here as in the
    # mediation family; this is a measurement / triangulation model, not a
    # predictive one, and #134 turns on the loadings / communalities, not on LOO.
    _run_sampling_and_loo(ctx, compute_loo=False)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=summary_vars)

    # Sample both observed nodes (the indicator matrix + the structural outcome).
    # These are two SEPARATE checks, not a joint predictive draw: the factor scores
    # condition on the observed indicator data (``Z_d``), not on the replicated
    # ``Z_obs``, so a replicated indicator is independent of the replicated factor
    # it loads on. ``Z_obs`` is a marginal check of the measurement covariance;
    # ``y_post`` is a check of the structural leg *conditional on the observed
    # indicators*. Together they do not certify the joint model. See the
    # predictive-simulation caveat in ``build_correlated_factor_model``.
    _run_ppc(ctx, var_names=["Z_obs", "y_post"])

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=summary_vars)
    _diag.run_extended_diagnostics(ctx)
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=summary_vars)

    post = ctx.trace.posterior
    hdi = ctx.reporting.ci_prob
    lo_q = (1.0 - hdi) / 2.0

    # --- Loadings + communalities (the measurement headline) ---
    section_header("Loadings + communalities")
    dom_of = {s: d for d, syms in domains.items() for s in syms}
    load_rows = []
    for j, name in enumerate(str(s) for s in post["indicator"].values):
        lam_d = post["lambda_load"].isel(indicator=j).values.reshape(-1)
        com_d = post["communality"].isel(indicator=j).values.reshape(-1)
        # The residual variance sigma is free, so the loading lambda is a
        # coefficient on the unit-variance factor, NOT in general a correlation.
        # The standardised loading / indicator-factor correlation is
        # lambda / sqrt(lambda**2 + sigma**2) = sqrt(communality).
        corr_d = np.sqrt(com_d)
        load_rows.append(
            {
                "indicator": name,
                "domain": dom_of.get(name, "?"),
                "loading_mean": float(np.mean(lam_d)),
                "loading_lo": float(np.quantile(lam_d, lo_q)),
                "loading_hi": float(np.quantile(lam_d, 1 - lo_q)),
                "loading_lo90": float(np.quantile(lam_d, 0.05)),
                "loading_hi90": float(np.quantile(lam_d, 0.95)),
                "correlation_mean": float(np.mean(corr_d)),
                "correlation_lo": float(np.quantile(corr_d, lo_q)),
                "correlation_hi": float(np.quantile(corr_d, 1 - lo_q)),
                "correlation_lo90": float(np.quantile(corr_d, 0.05)),
                "correlation_hi90": float(np.quantile(corr_d, 0.95)),
                "communality_mean": float(np.mean(com_d)),
                "communality_lo": float(np.quantile(com_d, lo_q)),
                "communality_hi": float(np.quantile(com_d, 1 - lo_q)),
                "communality_lo90": float(np.quantile(com_d, 0.05)),
                "communality_hi90": float(np.quantile(com_d, 0.95)),
            }
        )
    load_df = pd.DataFrame(load_rows)
    load_df.to_csv(os.path.join(ctx.output_dir, "loadings_summary.csv"), index=False)
    ctx.tables["loadings_summary"] = load_df
    print_table(
        ranked_dataframe_table(
            load_df,
            title=f"Loadings, correlations + communalities - {int(hdi * 100)}% CI (equal-tailed)",
            columns=[
                "indicator", "domain", "loading_mean", "correlation_mean",
                "communality_mean", "communality_lo", "communality_hi",
            ],
            rank_column=False,
            precision=3,
        )
    )

    # --- Factor correlation matrix ---
    section_header("Factor correlation")
    corr_draws = post["factor_corr"]  # (chain, draw, domain, domain2)
    corr = corr_draws.mean(dim=("chain", "draw")).values
    dnames = [str(d) for d in post["domain"].values]
    corr_df = pd.DataFrame(corr, index=dnames, columns=dnames)
    corr_df.to_csv(os.path.join(ctx.output_dir, "factor_correlation.csv"))
    ctx.tables["factor_correlation"] = corr_df
    # The bare mean matrix above is kept for the heatmap, but the house rule is
    # "never a bare point estimate": persist each unique off-diagonal pair with a
    # posterior mean, equal-tailed interval and tail probability alongside it.
    corr_stacked = corr_draws.stack(sample=("chain", "draw"))
    lo_q = (1 - hdi) / 2
    corr_rows = []
    for i, di in enumerate(dnames):
        for j, dj in enumerate(dnames):
            if j <= i:
                continue
            pair = np.asarray(corr_stacked.isel(domain=i, domain_b=j).values).reshape(-1)
            corr_rows.append(
                {
                    "domain_i": di,
                    "domain_j": dj,
                    "mean": float(np.mean(pair)),
                    "lo": float(np.quantile(pair, lo_q)),
                    "hi": float(np.quantile(pair, 1 - lo_q)),
                    "lo90": float(np.quantile(pair, 0.05)),
                    "hi90": float(np.quantile(pair, 0.95)),
                    "prob_pos": float(np.mean(pair > 0)),
                }
            )
    corr_summary_df = pd.DataFrame(corr_rows)
    corr_summary_df.to_csv(
        os.path.join(ctx.output_dir, "factor_correlation_summary.csv"), index=False
    )
    ctx.tables["factor_correlation_summary"] = corr_summary_df

    # --- Structural slopes: factor -> reading gain (adjusted associations) ---
    section_header("Structural slopes (factor -> gain)")
    struct_rows = [
        _coef_row(f"beta_{d}", post["beta_factor"].isel(domain=k).values, hdi)
        for k, d in enumerate(dnames)
    ]
    extra_terms = (["beta_age"] if spec.extra.get("use_age", True) else []) + [
        f"beta_{c}" for c in structural_covs
    ]
    struct_rows += [_coef_row(t, post[t].values, hdi) for t in extra_terms]
    struct_df = pd.DataFrame(struct_rows)
    struct_df.to_csv(os.path.join(ctx.output_dir, "structural_summary.csv"), index=False)
    ctx.tables["structural_summary"] = struct_df
    print_table(
        ranked_dataframe_table(
            struct_df,
            title=(
                f"Structural slopes (factor -> gain; adjusted associations) - "
                f"{int(hdi * 100)}% CI"
            ),
            columns=["coefficient", "mean", "lo", "hi", "prob_pos"],
            rank_column=False,
            precision=3,
        )
    )

    _report.write_run_metadata(
        ctx,
        extra={
            "domains": {k: list(v) for k, v in domains.items()},
            "loadings_summary": load_df.to_dict("records"),
            "factor_correlation": corr_df.to_dict(),
            "structural_summary": struct_df.to_dict("records"),
        },
    )

    return _finalize_report(ctx)
