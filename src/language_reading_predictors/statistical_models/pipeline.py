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

import hashlib
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
    _subset_prepared,
    load_and_prepare,
    load_and_prepare_aligned,
    load_and_prepare_lagged_outcome,
    load_longitudinal_panel,
    load_wave_panel,
    logit_safe,
    restrict_to_baseline_floored,
    restrict_to_off_floor,
    split_confounders_by_timing,
    split_covariates_by_wave,
    standardise,
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
            n_waves=getattr(prepared, "n_waves", None) if prepared else None,
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
        # Time offsets and every post-crossover term are associations.  Only the
        # saturated arm-by-wave model's t2 arm gap is licensed by randomisation.
        role["beta_period"] = "association"
        role["arm_gap_t1"] = "association"
        role["arm_gap_t3"] = "association"
        role["delta_crossover"] = "association"
        rationale["beta_period"] = (
            "Wave/period offset; an age, maturation and treatment-history association, "
            "not a randomised treatment effect."
        )
        rationale["arm_gap_t1"] = (
            "Pre-randomisation immediate-minus-waitlist balance quantity; regularised "
            "as an association, not interpreted as an effect."
        )
        rationale["tau_t2"] = (
            "Immediate-minus-waitlist t2 contrast identified by the original "
            "randomisation; the only causal coefficient in the binary crossover model."
        )
        rationale["arm_gap_t3"] = (
            "Post-crossover immediate-minus-waitlist t3 association comparing different "
            "treatment histories (approximately 40 versus 20 weeks)."
        )
        rationale["sigma_delta"] = (
            "Exploratory between-waitlist-child SD of unexplained t3 catch-up; may mix "
            "response, maturation, history, period shocks and measurement variation."
        )
        if not spec.extra.get("dose", False):
            role["tau_t2"] = "causal"
            role["alpha_offset"] = "nuisance"
            rationale["alpha_offset"] = (
                "Zero-centred offset around the pooled observed t1 logit anchor; "
                "the deterministic alpha is the anchored t1 level."
            )
        if spec.extra.get("dose", False):
            role["beta_group"] = "association"
            role["theta_treated"] = "association"
            role["gamma_t1"] = "precision"
            rationale["beta_group"] = (
                "Randomised-arm and prior-treatment-history adjustment in the transition "
                "dose model; not itself the t2 randomised arm contrast."
            )
            rationale["theta_treated"] = (
                "Modelled current-treatment presence at the mean treated dose; a "
                "crossover association, not a second randomised ITT effect."
            )
            rationale["gamma_t1"] = (
                "Shared pre-randomisation t1 outcome precision term broadcast to both "
                "period rows; never the treatment-affected t2 period-start score."
            )
            rationale["beta_dose"] = (
                "Observational intensive-margin association per treated-row SD of raw "
                "sessions, with untreated rows coded at zero intensity."
            )
            rationale["mu_dose"] = (
                "Average observational intensive-margin session association across P1/P2."
            )
            rationale["beta_dose_phase"] = (
                "Partial-pooled observational intensive-margin session associations by period."
            )
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
        for _name in (
            "tau",
            "beta_trt",
            "b_grp_time",
            "beta_grp",
            "delta",
            "tau_t2",
            "arm_gap_t3",
            "theta_treated",
        ):
            ctor.setdefault(_name, "tau_distal")
            role.setdefault(_name, "causal")
        # The ANCOVA intercept is likewise tiered for distal outcomes (Normal(0,
        # 1.0); prior-critical-review 2026-07-07, Finding 1). Route it to the
        # ``alpha_distal`` panel so the report rationale matches the fitted scale
        # (the distribution column already reads the true 1.0 off the built RV).
        ctor.setdefault("alpha", "alpha_distal")
        ctor.setdefault("alpha_offset", "alpha_distal")

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


def _save_did_cell_ppc_plot(ctx: StatisticalFitContext, cell_ppc: pd.DataFrame) -> None:
    """Plot observed and replicated summaries for every fitted DiD design cell."""
    try:
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        x = np.arange(len(cell_ppc))
        labels = cell_ppc["cell"].str.replace("_", "\n").tolist()

        for ax, stem, ylabel in (
            (axes[0], "mean", "cell mean"),
            (axes[1], "zero_rate", "proportion at zero"),
        ):
            centre = cell_ppc[f"replicated_{stem}_median"].to_numpy(float)
            lo = cell_ppc[f"replicated_{stem}_lo"].to_numpy(float)
            hi = cell_ppc[f"replicated_{stem}_hi"].to_numpy(float)
            observed = cell_ppc[f"observed_{stem}"].to_numpy(float)
            ax.errorbar(
                x,
                centre,
                yerr=np.vstack((centre - lo, hi - centre)),
                fmt="o",
                capsize=4,
                color="#1f77b4",
                label="posterior predictive median and 95% interval",
            )
            ax.scatter(
                x,
                observed,
                marker="x",
                s=55,
                linewidth=2,
                color="#d62728",
                label="observed",
            )
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.2)
        axes[0].legend(loc="best")
        axes[1].set_xticks(x, labels)
        axes[1].set_xlabel("fitted arm-by-time cell")
        fig.suptitle("Cell-stratified posterior-predictive checks")
        fig.tight_layout()
        save_styled_figure(ctx.output_dir, "did_cell_ppc", fig=fig)
    except Exception as exc:  # pragma: no cover
        rprint(f"[yellow]DiD cell PPC plot failed: {exc}[/yellow]")


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
            f"Proportion-at-zero PPC ({symbol}); two-sided tail = "
            f"{ppc0['ppc_two_sided_tail']:.2f}"
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
    """Heatmap of joint pairwise probability-scale AME ordering (#125 Area 4)."""
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
        ax.set_title("P(row AME > column AME)", fontsize=9)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("P(row AME > column AME)", fontsize=8)
        # ``save_styled_figure`` owns the layout engine. Switching engines after
        # a colorbar has been created raises on recent Matplotlib versions.
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


def _write_itt_analysis_audit(
    ctx: StatisticalFitContext,
    prepared,
    outcomes: Sequence[str],
) -> None:
    """Persist fitted-arm counts and full-randomised extreme-case bounds.

    The fitted frame may be outcome-, covariate- or subgroup-restricted. Bounds
    are therefore calculated from a fresh outcome-only frame that retains every
    archived child with a group code, without requiring its baseline. This keeps
    the 57-person randomised denominator separate from model-specific
    complete-case selection.
    """

    from language_reading_predictors.statistical_models.itt_audit import (
        analysis_set_table,
        randomised_postscore_bounds,
    )

    analysis_frames = []
    bound_frames = []
    multiple = len(outcomes) > 1
    for symbol in outcomes:
        analysis = analysis_set_table(prepared, outcome_symbol=symbol)
        if multiple:
            analysis.insert(0, "outcome", symbol)
        analysis_frames.append(analysis)

        attrition_source = load_and_prepare(
            path=getattr(prepared, "data_path", None) or None,
            phase_mode="itt",
            outcomes=(symbol,),
            pre_required=(),
        )
        bound_frames.append(randomised_postscore_bounds(attrition_source, symbol))

    analysis_df = pd.concat(analysis_frames, ignore_index=True)
    bounds_df = pd.concat(bound_frames, ignore_index=True)
    analysis_df.to_csv(os.path.join(ctx.output_dir, "analysis_set.csv"), index=False)
    bounds_df.to_csv(os.path.join(ctx.output_dir, "attrition_bounds.csv"), index=False)
    ctx.tables["analysis_set"] = analysis_df
    ctx.tables["attrition_bounds"] = bounds_df


def _write_itt_ppc_calibration(
    ctx: StatisticalFitContext,
    prepared,
    outcomes: Sequence[str],
    *,
    node: str = "y_post",
    filename: str = "posterior_predictive_calibration.csv",
) -> pd.DataFrame:
    """Save bounded-score PPC metrics by arm and baseline band.

    Joint fits additionally persist one outcome-wide median/Q3/IQR calibration
    row per outcome. That marginal shape check catches discrepancies that can be
    hidden by means and boundary rates calculated in small strata.
    """

    from language_reading_predictors.statistical_models.ppc_audit import (
        score_ppc_by_arm_and_baseline,
        score_ppc_distribution_shape,
    )

    predictive = np.asarray(ctx.trace.posterior_predictive[node].values, dtype=float)
    frames = []
    shape_frames = []
    if len(outcomes) == 1 and predictive.shape[-1] == prepared.n_obs:
        symbol = outcomes[0]
        observed = None
        denominator = None
        if node == "y_offfloor":
            observed = (prepared.post_counts[symbol] > 0).astype(float)
            denominator = 1
        frames.append(
            score_ppc_by_arm_and_baseline(
                prepared,
                symbol,
                predictive,
                observed_counts=observed,
                n_trials=denominator,
                ci_prob=ctx.reporting.ci_prob,
            )
        )
    else:
        constant = ctx.trace.constant_data
        cell_rows = np.asarray(constant["y_post_cell_row"].values, dtype=int).ravel()
        cell_outcomes = np.asarray(
            constant["y_post_cell_outcome"].values, dtype=int
        ).ravel()
        if predictive.shape[-1] != cell_rows.size:
            raise ValueError("joint posterior-predictive cells do not match cell map")
        for outcome_index, symbol in enumerate(outcomes):
            selected = cell_outcomes == outcome_index
            outcome_rows = cell_rows[selected]
            frames.append(
                score_ppc_by_arm_and_baseline(
                    prepared,
                    symbol,
                    predictive[..., selected],
                    row_indices=outcome_rows,
                    ci_prob=ctx.reporting.ci_prob,
                )
            )
            shape_frames.append(
                score_ppc_distribution_shape(
                    symbol,
                    predictive[..., selected],
                    np.asarray(prepared.post_counts[symbol], dtype=float)[outcome_rows],
                    n_trials=int(prepared.n_trials[symbol]),
                    ci_prob=ctx.reporting.ci_prob,
                )
            )

    calibration = pd.concat(frames, ignore_index=True)
    calibration.to_csv(os.path.join(ctx.output_dir, filename), index=False)
    ctx.tables[os.path.splitext(filename)[0]] = calibration
    if shape_frames:
        shape_calibration = pd.concat(shape_frames, ignore_index=True)
        shape_filename = "posterior_predictive_shape_calibration.csv"
        shape_calibration.to_csv(
            os.path.join(ctx.output_dir, shape_filename), index=False
        )
        ctx.tables[os.path.splitext(shape_filename)[0]] = shape_calibration
    return calibration


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
        _write_loo_influence(ctx)
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
    # Age-only outcomes (e.g. nonword N) can exempt baseline from the loader's
    # complete-case mask via ``pre_required``. The floor branch then reports
    # missing eligibility explicitly and restricts its transition estimand to
    # observed baseline zeros; the all-child graded secondary does not use pre.
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

    # Heavily-floored outcomes (P, N) take the post-hoc, data-adaptive
    # floor-rule branch in this reanalysis: a binary transition estimand as the
    # exploratory headline plus graded secondary checks (#119/#341).
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
        tau_sigma=spec.extra.get("tau_sigma"),
        alpha_sigma=spec.extra.get("alpha_sigma"),
        gamma_own_sigma=spec.extra.get("gamma_own_sigma"),
        kappa_sigma=spec.extra.get("kappa_sigma"),
    )
    _attach_built(ctx, built)
    _write_itt_analysis_audit(ctx, built.prepared, (spec.outcome_symbol,))

    _render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)

    _run_sampling_and_loo(ctx)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=_itt_diag_vars(spec, adjust_for))

    _run_ppc(ctx)
    _write_itt_ppc_calibration(ctx, built.prepared, (spec.outcome_symbol,))

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

    Fits two age-only models: the post-hoc exploratory binary off-floor
    transition (Bernoulli on ``post > 0`` among observed baseline zeros) and a
    flagged, detection-limited secondary graded Beta-Binomial. Writes
    ``tau_summary.csv`` (off-floor exploratory headline), the per-arm mover table,
    the proportion-at-zero PPC, and ``tau_summary_graded.csv``. The floor rule is
    post-hoc and data-adaptive in this reanalysis, although its mechanical gate is
    applied arm-blind.
    """
    import pymc as pm

    from language_reading_predictors.statistical_models import floor as _floor

    own = spec.outcome_symbol

    # Data-adaptive gate: the outcome must actually qualify (>= 40% at zero at
    # t2). Applying it arm-blind avoids using treatment labels in this mechanical
    # classification, but does not make the post-hoc choice pre-specified.
    p0 = _floor.proportion_at_zero(prepared, own)
    if not _floor.is_floored(prepared, own):
        raise ValueError(
            f"floor_rule set for {own!r}, but only {p0:.0%} of its post-scores "
            f"are at zero at t2 (threshold {_floor.FLOOR_THRESHOLD:.0%}); the "
            "post-hoc floor gate is arm-blind - remove floor_rule or "
            "check the data."
        )

    # Make eligibility and its missingness visible before restricting. The
    # registered loader retains missing pre-scores for P/N, so without this table
    # those children would disappear silently when np.isclose(NaN, floor) is false.
    eligibility = _floor.baseline_floor_eligibility_by_arm(prepared, own)
    eligibility.to_csv(
        os.path.join(ctx.output_dir, "baseline_floor_eligibility.csv"), index=False
    )
    ctx.tables["baseline_floor_eligibility"] = eligibility
    eligibility_sensitivity = _floor.baseline_floor_status_bounds(prepared, own)
    eligibility_sensitivity.to_csv(
        os.path.join(ctx.output_dir, "floor_eligibility_sensitivity.csv"), index=False
    )
    ctx.tables["floor_eligibility_sensitivity"] = eligibility_sensitivity
    transition_missingness = _floor.binary_transition_missingness_bounds(prepared, own)
    transition_missingness.to_csv(
        os.path.join(ctx.output_dir, "floor_transition_missingness_bounds.csv"),
        index=False,
    )
    ctx.tables["floor_transition_missingness_bounds"] = transition_missingness
    print_table(
        ranked_dataframe_table(
            eligibility,
            title=f"Observed baseline-floor eligibility by arm ({own})",
            columns=[
                "arm",
                "n_loaded",
                "n_post_observed",
                "n_pre_observed",
                "n_pre_missing",
                "n_pre_floor",
                "n_pre_above_floor",
                "n_exploratory_eligible",
            ],
            rank_column=False,
            precision=0,
        )
    )

    # Restrict the exploratory headline to children with an *observed* baseline
    # score of zero. This targets Pr(post > 0 | observed pre == 0), rather than
    # prevalence over everyone. Baseline status is pre-randomisation, so the arm
    # contrast remains causally valid for this observed subgroup, subject to the
    # missingness assumptions stated in the report.
    at_risk = restrict_to_baseline_floored(prepared, own)
    n_eligible = int(eligibility["n_exploratory_eligible"].sum())
    # This equality relies on the single-outcome loader requiring this outcome's
    # post-score; revisit it before applying the floor rule to a joint outcome load.
    if at_risk.n_obs != n_eligible:
        raise RuntimeError(
            f"floor-rule eligibility count drift for {own!r}: restriction kept "
            f"{at_risk.n_obs}, eligibility table reports {n_eligible}"
        )
    # Guard: the subgroup ITT is only identified if the at-risk subset keeps both
    # arms and enough rows. If a future floored outcome had (say) all baseline-floored
    # children in one arm, tau would be unidentified and the headline posterior
    # degenerate — fail loudly rather than publish it (issue #267 review).
    _n_arms = int(np.unique(at_risk.G).size)
    if at_risk.n_obs < 10 or _n_arms < 2:
        raise ValueError(
            f"floor rule for {own!r}: the baseline-floored at-risk subset is "
            f"degenerate (n={at_risk.n_obs}, arms present={_n_arms}) — the subgroup "
            "contrast Pr(post>0 | observed pre==0) is not identified. Re-check "
            "the floor rule / "
            "data or fit a different estimand."
        )
    _write_itt_analysis_audit(ctx, at_risk, (own,))
    missing_by_arm = ", ".join(
        f"{row.arm}: {int(row.n_pre_missing)}"
        for row in eligibility.itertuples(index=False)
    )
    rprint(
        f"  Floor rule: {own} is {p0:.0%} floored at t2 "
        f"(>= {_floor.FLOOR_THRESHOLD:.0%}); the post-hoc exploratory headline is "
        f"Pr(off-floor at t2 | observed at floor at t1) on {at_risk.n_obs} "
        f"eligible children (of {prepared.n_obs} with an available t2 outcome). "
        f"Missing baseline eligibility by arm — {missing_by_arm}. A graded "
        "Beta-Binomial over all children and a graded contrast among off-floor "
        "children are flagged secondaries."
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

    # ----- EXPLORATORY HEADLINE: binary transition among observed baseline zeros. -----
    section_header(
        "Build model (post-hoc headline: off-floor transition among observed "
        "baseline-floor children)"
    )
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
    _write_itt_ppc_calibration(
        ctx,
        built.prepared,
        (own,),
        node="y_offfloor",
    )

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

    section_header("Off-floor treatment-effect summary (post-hoc exploratory headline)")
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
                f"off-floor transition tau ({own}, observed baseline-floor subgroup) - "
                f"{int(ctx.reporting.ci_prob * 100)}% CI (equal-tailed); positive = "
                "intervention raises Pr(off-floor at t2 | observed at floor at t1)"
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

    def _fit_secondary(built_x, *, label: str, trace_filename: str):
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
        # Gate every free variable: a well-mixed tau cannot rescue a non-mixing
        # kappa/alpha/age term because those nuisance parameters determine the
        # fitted mean and posterior predictive distribution (#341).
        free_names = [rv.name for rv in built_x.model.free_RVs]
        conv = _diag.subfit_convergence(tr, label=label, var_names=free_names)
        summ = _report.tau_summary_itt(tr, ci_prob=ctx.reporting.ci_prob, G=built_x.prepared.G)
        summ.update(conv)
        # Secondary estimates are publication artefacts too. Persist the trace
        # so every convergence value and posterior can be audited independently
        # of the exploratory off-floor fit.
        tr.to_netcdf(os.path.join(ctx.output_dir, trace_filename))
        summ["trace_file"] = trace_filename
        return tr, summ

    # ----- SECONDARY (flagged cross-check): graded Beta-Binomial over ALL children.
    # Not the exploratory headline — it mixes already-off-floor children into a mover analysis and
    # is detection-limited; read only beside the mover table, never alone (#119).
    section_header("Build model (SECONDARY cross-check: graded Beta-Binomial, all children)")
    built_g = _factories.build_itt_model(prepared, likelihood="beta_binomial", **common)
    trace_g, graded = _fit_secondary(
        built_g,
        label=f"{spec.model_id} graded cross-check",
        trace_filename="trace_graded_secondary.nc",
    )
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
            built_h,
            label=f"{spec.model_id} off-floor-subset graded contrast",
            trace_filename="trace_hurdle_secondary.nc",
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

    # Proportion-at-zero PPC on the graded model: assess whether the graded
    # Beta-Binomial reproduces the observed floor.
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
                "status": "post_hoc_data_adaptive",
                "arm_blind_gate": True,
                "exploratory_estimand": (
                    "Pr(off-floor at t2 | observed at floor at t1)"
                ),
                "at_risk_n": int(at_risk.n_obs),
                "total_n": int(prepared.n_obs),
                "baseline_missing_n": int(eligibility["n_pre_missing"].sum()),
                "eligibility_by_arm": eligibility.to_dict(orient="records"),
                "eligibility_status_sensitivity": eligibility_sensitivity.to_dict(
                    orient="records"
                ),
                "transition_missingness_bounds": transition_missingness.to_dict(
                    orient="records"
                ),
            },
            "tau_offfloor_exploratory": off,
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
    _write_itt_analysis_audit(ctx, built.prepared, joint_outcomes)

    _render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    for index, symbol in enumerate(joint_outcomes):
        stem = (
            "prior_predictive_check"
            if index == 0
            else f"prior_predictive_check_{symbol.lower()}"
        )
        _diag.save_prior_predictive_plot(ctx, symbol, filename_stem=stem)

    _run_sampling_and_loo(ctx)

    section_header("Summary diagnostics")
    _joint_vars = ["alpha", "tau", "gamma_own", "kappa"]
    if spec.extra.get("use_age_linear", False):
        _joint_vars.append("gamma_A")
    if spec.extra.get("use_residual_correlation", False):
        _joint_vars.append("sigma_outcome")
    _diag.summary_diagnostics(ctx, var_names=_joint_vars)

    section_header("Posterior predictive")
    _diag.sample_posterior_predictive(ctx, var_names=["y_post"])
    for index, symbol in enumerate(joint_outcomes):
        stem = (
            "posterior_predictive_check"
            if index == 0
            else f"posterior_predictive_check_{symbol.lower()}"
        )
        _diag.save_joint_posterior_predictive_plot(
            ctx, symbol, filename_stem=stem
        )
    _write_itt_ppc_calibration(ctx, built.prepared, joint_outcomes)

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=_joint_vars)
    # The generic LOO-PIT would pool flattened cells from tests with different
    # denominators. Save one calibrated plot per outcome instead.
    _diag.run_extended_diagnostics(ctx, causal_term="tau", include_loo_pit=False)
    for index, symbol in enumerate(joint_outcomes):
        stem = "loo_pit" if index == 0 else f"loo_pit_{symbol.lower()}"
        _diag.save_joint_loo_pit_plot(ctx, symbol, filename_stem=stem)
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=_joint_vars)
    # The probability-scale AMEs in tau_summary.csv are the headline effects. This
    # forest is deliberately retained as an explicitly labelled secondary view of
    # the conditional-logit coefficients.
    _save_forest_plot(
        ctx,
        ["tau"],
        title="Secondary conditional-logit coefficients (forest, reference line at 0)",
    )

    section_header("Treatment-effect summary")
    outcomes = list(ctx.trace.posterior["outcome"].values)
    tau_df = _report.tau_summary_joint(
        ctx.trace,
        outcomes,
        ci_prob=ctx.reporting.ci_prob,
        G=built.prepared.G,
    )
    tau_df.to_csv(os.path.join(ctx.output_dir, "tau_summary.csv"), index=False)
    ctx.tables["tau_summary"] = tau_df
    print_table(
        ranked_dataframe_table(
            tau_df,
            title=(
                "Probability-scale AME by outcome - "
                f"{int(ctx.reporting.ci_prob * 100)}% CI (equal-tailed)"
            ),
            columns=[
                "outcome",
                "ame_prob_median",
                "ame_prob_lo",
                "ame_prob_hi",
                "prob_ame_pos",
            ],
            rank_column=False,
        )
    )

    contrast = _report.tau_contrast_matrix(
        ctx.trace, outcomes, G=built.prepared.G, scale="probability"
    )
    contrast.to_csv(os.path.join(ctx.output_dir, "tau_contrast_matrix.csv"))
    ctx.tables["tau_contrast_matrix"] = contrast
    _save_contrast_heatmap(ctx, contrast)

    logit_contrast = _report.tau_contrast_matrix(
        ctx.trace, outcomes, G=built.prepared.G, scale="logit"
    )
    logit_contrast.to_csv(
        os.path.join(ctx.output_dir, "tau_contrast_matrix_logit.csv")
    )
    ctx.tables["tau_contrast_matrix_logit"] = logit_contrast

    meta_extra: dict = {
        "loo_elpd": float(ctx.loo.elpd),
        "joint_structure": built.extras.get("joint_dependence"),
        "loo_unit": built.extras.get("loo_unit", "child"),
        "outcomes": list(joint_outcomes),
    }

    # Two-outcome contrast (LRPITT15/15b/16). ``difference = (a, b)`` reports the
    # headline probability-scale AME[a] - AME[b] and retains tau[a] - tau[b] as a
    # secondary conditional-logit contrast.
    difference = spec.extra.get("difference")
    if difference is not None:
        pair = tuple(difference)
        section_header("Treatment-effect difference")
        diff_s = _report.tau_difference_summary(
            ctx.trace,
            outcomes,
            pair,
            ci_prob=ctx.reporting.ci_prob,
            G=built.prepared.G,
            metadata=spec.extra.get("difference_metadata"),
        )
        diff_df = pd.DataFrame([diff_s])
        diff_df.to_csv(os.path.join(ctx.output_dir, "tau_difference.csv"), index=False)
        ctx.tables["tau_difference"] = diff_df
        print_table(
            metrics_table(
                [{"metric": k, "value": v} for k, v in diff_s.items()],
                title=(
                    f"AME[{pair[0]}] - AME[{pair[1]}] (probability-scale headline; "
                    f"logit secondary) - {int(ctx.reporting.ci_prob * 100)}% CI "
                    "(equal-tailed)"
                ),
                columns=["metric", "value"],
            )
        )
        meta_extra["tau_difference"] = diff_s
        meta_extra["difference_metadata"] = spec.extra.get("difference_metadata")

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
        v = [
            "alpha_offset",
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
    manifest.to_csv(os.path.join(ctx.output_dir, "analysis_rows.csv"), index=False)

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
                "alpha_anchor_logit": float(built.extras["alpha_anchor"]),
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
    _require_spec(spec, "did", outcome=True)

    ctx = make_context(spec, config)

    section_header("Prepare data")
    sym = spec.outcome_symbol
    dose = bool(spec.extra.get("dose", False))
    period_varying = dose and bool(spec.extra.get("period_varying_dose", False))
    likelihood = spec.extra.get("likelihood", "beta_binomial")
    off_floor = likelihood == "bernoulli_offfloor"
    # Binary models use t1--t3 levels so the randomised t2 arm gap and the
    # post-crossover t3 gap are estimated separately. Dose models retain the
    # transition frame because sessions are interval exposures.
    outcomes = tuple(spec.extra.get("outcomes", (sym,)))
    covariates = ("attend",) if dose else ()
    if dose:
        prepared = load_and_prepare(
            phase_mode="all",
            outcomes=outcomes,
            covariates=covariates,
            pre_required=(),
            require_any_post=False,
        )
    else:
        prepared = load_and_prepare(
            phase_mode="levels",
            outcomes=outcomes,
            require_any_post=False,
        )
    ctx.prepared = prepared

    _print_header(ctx)

    section_header("Build model")
    built = _factories.build_did_model(
        prepared,
        outcome_symbol=sym,
        waves=tuple(spec.extra.get("waves", (0, 1, 2))),
        periods=tuple(spec.extra.get("periods", (0, 1))),
        use_child_re=spec.extra.get("use_child_re", True),
        use_age=spec.extra.get("use_age", True),
        dose=dose,
        period_varying_dose=period_varying,
        use_varying_delta=spec.extra.get("use_varying_delta", False),
        likelihood=likelihood,
    )
    _attach_built(ctx, built)
    _print_header(ctx)
    did_contract = _did_analysis_contract(
        ctx,
        built,
        dose=dose,
        loaded_prepared=prepared,
    )

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
    did_cell_ppc = _report.did_cell_ppc(
        ctx.trace,
        phase=ctx.prepared.phase,
        G=ctx.prepared.G,
        dose=dose,
        node="y_offfloor" if off_floor else "y_post",
        ci_prob=ctx.reporting.ci_prob,
    )
    did_cell_ppc.to_csv(
        os.path.join(ctx.output_dir, "did_cell_ppc.csv"), index=False
    )
    ctx.tables["did_cell_ppc"] = did_cell_ppc
    _save_did_cell_ppc_plot(ctx, did_cell_ppc)

    section_header("Extended diagnostics")
    _did_effect = (
        "mu_dose" if period_varying else ("beta_dose" if dose else "tau_t2")
    )
    _diag.write_diagnostics_summary(ctx, var_names=_did_diag_vars(spec))
    _diag.run_extended_diagnostics(ctx, causal_term=_did_effect)
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=_did_diag_vars(spec))
    _diag.run_psense(ctx, var_names=[_did_effect])
    if not dose:
        try:
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
            prior_pushforward_df.to_csv(
                os.path.join(ctx.output_dir, "prior_pushforward.csv"), index=False
            )
            ctx.tables["prior_pushforward"] = prior_pushforward_df
        except Exception as exc:  # pragma: no cover
            rprint(f"[yellow]DiD prior pushforward skipped: {exc}[/yellow]")
        _save_forest_plot(
            ctx,
            ["tau_t2", "arm_gap_t3", "delta_crossover"],
            name="did_contrasts_forest.png",
            title="Randomised t2 and post-crossover contrasts",
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
    did_df.to_csv(os.path.join(ctx.output_dir, "did_summary.csv"), index=False)
    ctx.tables["did_summary"] = did_df
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

    if period_varying:
        # Period-resolved dose readout (#135): partial-pooled per-period dose
        # slopes + a between-period SD, written by the shared dose-slope summary.
        # The headline question — does the L dose-gain slope vary by period? — is
        # answered by the nested PSIS-LOO vs the pooled comparator (lrp-rli-did-107)
        # in compare_statistical_models.py, not by this single-fit table.
        section_header("Period-resolved dose-slope summary")
        _write_dose_slope_summary(ctx, period_varying=True)
    het = None
    if spec.extra.get("use_varying_delta", False):
        section_header("Exploratory waitlist catch-up heterogeneity")
        het = _did_heterogeneity_summary(ctx.trace, ci_prob=ctx.reporting.ci_prob)
        pd.DataFrame([het]).to_csv(
            os.path.join(ctx.output_dir, "heterogeneity_summary.csv"), index=False
        )
        ctx.tables["heterogeneity_summary"] = pd.DataFrame([het])
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

    _report.write_run_metadata(
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
    # Complete-case comparator: drop the mean-imputed rows so the confounders are
    # genuinely observed. Mean-imputation + a missingness indicator keeps every
    # child, but does not by itself guarantee adequate confounding control, so the
    # imputed fit needs this comparator beside it (#258 review).
    require_observed = tuple(spec.extra.get("require_observed", ()))
    # Covariate exposure (#311 route (b)): the mechanism variable is a standardised
    # continuous covariate (e.g. phonological memory ``erbto``, whose documented test
    # maximum is unrecorded, so it cannot honestly be a bounded-count Measure). Load
    # it — and, when the spec complete-cases on it, its ``_missing`` flag for the
    # loader's ``require_observed`` filter — alongside the adjusters. It must not
    # also be in ``adjust_for``: the factory gives it ``beta_mech``, so a ``gamma``
    # term too would enter it twice.
    mechanism_is_covariate = bool(spec.extra.get("mechanism_is_covariate", False))
    load_covariates = adjust_for
    if mechanism_is_covariate:
        if spec.mechanism_symbol in adjust_for:
            raise ValueError(
                f"{spec.model_id}: covariate exposure {spec.mechanism_symbol!r} "
                "must not also appear in adjust_for (it would enter the linear "
                "predictor twice)."
            )
        extra_load: tuple[str, ...] = (spec.mechanism_symbol,)
        if spec.mechanism_symbol in require_observed:
            extra_load += (f"{spec.mechanism_symbol}_missing",)
        load_covariates = tuple(dict.fromkeys((*adjust_for, *extra_load)))
    pre_adj, post_adj = split_covariates_by_wave(load_covariates)
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
    if mechanism_is_covariate and spec.mechanism_symbol not in prepared.covariates:
        # The drop-constant policy is fine for an adjuster but fatal for the
        # exposure itself — there is no model without it.
        raise ValueError(
            f"{spec.model_id}: covariate exposure {spec.mechanism_symbol!r} was "
            "dropped by the loader (constant on the fitted rows); cannot fit."
        )

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
        mechanism_is_covariate=mechanism_is_covariate,
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
    if mechanism_is_covariate:
        # Record the exposure's raw-units anchor so a report can translate the
        # per-SD ``beta_mech`` into raw score points: the factory re-standardises
        # the loader z on the kept rows, so +1 SD of the fitted exposure is
        # ``loader_sd * sd(z_kept)`` raw points.
        meta_extra["mechanism_is_covariate"] = True
        _sc = ctx.prepared.covariate_scalers.get(spec.mechanism_symbol)
        if _sc is not None:
            _z_kept = np.asarray(
                ctx.prepared.covariates[spec.mechanism_symbol], dtype=float
            )
            meta_extra["mechanism_exposure_sd_raw"] = float(
                _sc.sd * np.nanstd(_z_kept, ddof=1)
            )
            meta_extra["mechanism_exposure_mean_raw"] = float(
                _sc.mean + _sc.sd * np.nanmean(_z_kept)
            )

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
    branches hold the adjustment set fixed and write the same CSV/PNG schema, except
    for the x column: ``mech_logit`` for a bounded-count measure exposure,
    ``mech_x`` (the raw covariate score) for a covariate exposure
    (``mechanism_is_covariate``, always linear). Guarded by the caller.
    """
    post = ctx.trace.posterior

    from language_reading_predictors.statistical_models.measures import MEASURES
    from language_reading_predictors.statistical_models.preprocessing import (
        logit_safe,
        standardise,
    )

    sym = ctx.spec.mechanism_symbol
    is_covariate = bool(ctx.spec.extra.get("mechanism_is_covariate", False))
    if is_covariate:
        # Covariate exposure: x is the raw score (the loader scaler inverted); the
        # model's z is the loader z re-standardised on the kept rows, exactly as
        # the factory did it.
        z_loaded = np.asarray(ctx.prepared.covariates[sym], dtype=float)
        _scaler = ctx.prepared.covariate_scalers.get(sym)
        x_vals = _scaler.inverse(z_loaded) if _scaler is not None else z_loaded
        z_L, _ = standardise(z_loaded)
        x_col, x_label = "mech_x", f"{sym} (raw score)"
    else:
        N = MEASURES[sym].n_trials
        mech_logit = logit_safe(ctx.prepared.post_counts[sym], N)
        x_vals = mech_logit
        # z the same standardisation the factory applied to the logit input.
        z_L, _ = standardise(mech_logit)
        x_col, x_label = "mech_logit", f"logit({sym}_post)"

    if "f_mech" in post:
        f = post["f_mech"].stack(sample=("chain", "draw")).values  # (n_obs, n_sample)
        kind = "GP"
    elif "beta_mech" in post:
        # Linear mechanism: the predictor enters as beta_mech * z. Build the
        # per-observation contribution so the band mirrors the GP branch (an exact
        # straight line).
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

    order = np.argsort(x_vals)
    x = x_vals[order]
    f_ord = f[order]
    mean = f_ord.mean(axis=1)
    lo = np.quantile(f_ord, 0.025, axis=1)
    hi = np.quantile(f_ord, 0.975, axis=1)
    lo90 = np.quantile(f_ord, 0.05, axis=1)
    hi90 = np.quantile(f_ord, 0.95, axis=1)
    pd.DataFrame(
        {x_col: x, "f_mean": mean, "f_lo": lo, "f_hi": hi,
         "f_lo90": lo90, "f_hi90": hi90}
    ).to_csv(os.path.join(ctx.output_dir, "mechanism_curve.csv"), index=False)
    outcome = ctx.spec.outcome_symbol or "W"
    plt.figure(figsize=(6, 4))
    plt.plot(x, mean, color="#1f77b4", lw=2)
    plt.fill_between(x, lo, hi, color="#1f77b4", alpha=0.2)
    plt.xlabel(x_label)
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
    # Off-floor (Bernoulli) OUTCOME for a heavily-floored outcome such as nonword N
    # (#228 item 12): the outcome leg becomes a Bernoulli on the off-floor indicator
    # (node "y_offfloor") and the g-formula reports NIE/NDE on the off-floor
    # risk-difference scale. Default "beta_binomial" keeps every existing med model
    # byte-identical.
    outcome_kind = spec.extra.get("outcome_kind", "beta_binomial")
    off_floor = outcome_kind == "bernoulli_offfloor"
    outcome_node = "y_offfloor" if off_floor else "y_post"
    built, med_data = _factories.build_mediation_model(
        prepared,
        mediator_symbol=mediator_symbol,
        outcome_symbol=spec.outcome_symbol or "W",
        confounder_symbols=confounders,
        mediator_kind=mediator_kind,
        route_symbols=route_symbols,
        outcome_kind=outcome_kind,
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
    _diag.save_prior_predictive_plot(ctx, spec.outcome_symbol or "W", node=outcome_node)

    _run_sampling_and_loo(ctx, compute_loo=False)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=coef_vars)

    _run_ppc(ctx, var_names=[mediator_node, outcome_node])

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


def fit_mediation_period_stacked(
    spec: ModelSpec, config: str = "dev"
) -> StatisticalFitContext:
    """Period-stacked g-formula mediation on the gain-factor scaffold (MED-092, #229).

    The LRP59 mediator + outcome design refit over **all stacked period
    transitions** (``phase_mode="all"``), with the per-period on-intervention
    indicator as the exposure and the gain-factor machinery (phase intercepts,
    per-leg child random intercepts). Writes the all-period decomposition to
    ``mediation_summary.csv`` and the period-1 (ITT-anchored, LRP59-comparable)
    row restriction to ``mediation_summary_p1.csv``. No t3 temporal-ordering
    sensitivity is fitted — the stacked design already spans every window, and
    its mediator/outcome remain contemporaneous within each period by design.
    """
    _require_spec(spec, "mediation")
    from language_reading_predictors.statistical_models import mediation as _med

    ctx = make_context(spec, config)

    section_header("Prepare data")
    mediator_symbol = spec.mechanism_symbol or "L"
    outcome_symbol = spec.outcome_symbol or "W"
    # Structural markers aside, the adjustment list is the confounder set; the
    # raw covariates take the gain-factor timing split (hearing contemporaneous,
    # speech/phonological memory at the t1 baseline — the A1 timing decision).
    confounders = tuple(
        s
        for s in spec.adjustment
        if s not in ("T", "A", "W_pre", f"{mediator_symbol}_pre")
    )
    raw_cov = _raw_covariate_confounders(confounders)
    pre_adj, post_adj = split_covariates_by_wave(raw_cov)
    baseline_adj, post_adj = split_confounders_by_timing(post_adj)
    measure_confounders = tuple(c for c in confounders if c not in raw_cov)
    prepared = load_and_prepare(
        phase_mode="all",
        outcomes=(outcome_symbol, mediator_symbol, *measure_confounders),
        covariates=pre_adj,
        post_covariates=post_adj,
        baseline_covariates=baseline_adj,
    )
    ctx.prepared = prepared
    # Keep only confounders actually present (a constant ``_missing`` indicator
    # is dropped by the loader and gets no coefficient).
    confounders = tuple(
        c for c in confounders if c in prepared.covariates or c in prepared.pre_logit
    )

    _print_header(ctx)

    section_header("Build model")
    built, med_data = _factories.build_period_stacked_mediation_model(
        prepared,
        mediator_symbol=mediator_symbol,
        outcome_symbol=outcome_symbol,
        confounder_symbols=confounders,
    )
    _attach_built(ctx, built)

    mediator_node = f"{mediator_symbol}_post"
    # Scalar coefficients from the model itself, plus the per-phase intercept
    # vectors (the convergence gate scans every free RV regardless).
    coef_vars = sorted(rv.name for rv in built.model.free_RVs if rv.ndim == 0)
    diag_vars = [*coef_vars, "a_phase", "b_phase"]

    _render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    _diag.save_prior_predictive_plot(ctx, outcome_symbol, node="y_post")

    _run_sampling_and_loo(ctx, compute_loo=False)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=diag_vars)

    _run_ppc(ctx, var_names=[mediator_node, "y_post"])

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=diag_vars)
    _diag.run_extended_diagnostics(ctx)
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=diag_vars)

    section_header("Mediation decomposition (period-stacked g-formula)")
    med_df = _med.decompose_period_stacked(
        ctx.trace, med_data, ci_prob=ctx.reporting.ci_prob
    )
    med_df.to_csv(os.path.join(ctx.output_dir, "mediation_summary.csv"), index=False)
    ctx.tables["mediation_summary"] = med_df
    print_table(
        ranked_dataframe_table(
            med_df,
            title=(
                "Per-period mediation, all stacked periods "
                f"(on-intervention; words out of {med_data.n_trials_W})"
            ),
            columns=["quantity", "words_mean", "words_lo", "words_hi", "prob_pos"],
            rank_column=False,
            precision=3,
        )
    )

    # Period-1 restriction: the same posterior averaged over the randomised,
    # all-untreated-baseline transition only — the LRP59-comparable readout
    # (mirrors the gain-factor family's period-1 treatment marginal, #247 P2).
    med_df_p1 = _med.decompose_period_stacked(
        ctx.trace,
        med_data,
        ci_prob=ctx.reporting.ci_prob,
        row_mask=med_data.phase_idx == 0,
    )
    med_df_p1.to_csv(
        os.path.join(ctx.output_dir, "mediation_summary_p1.csv"), index=False
    )
    ctx.tables["mediation_summary_p1"] = med_df_p1
    print_table(
        ranked_dataframe_table(
            med_df_p1,
            title="Period-1 restriction (randomised window; LRP59-comparable)",
            columns=["quantity", "words_mean", "words_lo", "words_hi", "prob_pos"],
            rank_column=False,
            precision=3,
        )
    )

    section_header("Mediation NIE sensitivity (unmeasured confounding)")
    sens_sweep, sens_summary = _med.sensitivity_sweep(
        ctx.trace,
        med_data,
        ci_prob=ctx.reporting.ci_prob,
        decompose_fn=_med.decompose_period_stacked,
        interaction_name="b_trtM",
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
            f"({sens_summary['tipping_frac_of_bM']:.0%} of the fitted b_M+b_trtM) — an "
            "unmeasured mediator-outcome confounder that strong would null the NIE."
        )

    _requested_raw = _raw_covariate_confounders(
        s for s in spec.adjustment if s not in ("T", "A", "W_pre", f"{mediator_symbol}_pre")
    )
    _report.write_run_metadata(
        ctx,
        extra={
            "adjustment": spec.adjustment,
            "effective_confounders": list(confounders),
            "dropped_confounders": [c for c in _requested_raw if c not in confounders],
            "n_obs": prepared.n_obs,
            "exposure": "on_intervention (per-period; gain-factor ignorability)",
            "mediation": {r["quantity"]: r for r in med_df.to_dict("records")},
            "mediation_p1": {r["quantity"]: r for r in med_df_p1.to_dict("records")},
        },
    )

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


def _gf_association_terms(
    spec: ModelSpec,
    built: _factories.BuiltModel,
    *,
    adjust_for: tuple[str, ...],
    off_floor: bool,
) -> list[_report.AssociationTerm]:
    """Per-covariate ``AssociationTerm`` list for the gain items-scale marginals (#310).

    Reconstructs — from the *fitted* subset ``built.prepared`` — the exact standardised
    term vectors ``build_gain_factors_model`` used, so each covariate's ``+1 SD``
    perturbation is pushed through :func:`reporting.association_marginals` on the same
    scale the model was built on. The own baseline and skill baselines enter the linear
    predictor on the **raw logit** scale (their ``main_scale`` is that logit's SD) while
    their interactions use the standardised vector; age and cognitive ability are
    standardised throughout (``main_scale = 1``). Raw-covariate adjusters enter
    standardised with no interactions; their ``_missing`` companions are nuisance 0/1
    indicators (a ``+1 SD`` shift on them is not an interpretable association) and are
    skipped. The own baseline is dropped on the off-floor (Bernoulli) path, matching the
    factory (its ``gamma_own`` is not built there).
    """
    from scipy.special import expit as _expit
    from scipy.special import logit as _logit

    AT = _report.AssociationTerm
    bp = built.prepared
    own = spec.outcome_symbol
    extra = spec.extra
    skill_symbols = tuple(extra.get("skill_symbols", ()))
    ability_covariate = extra.get("ability_covariate")
    interactions = tuple(tuple(p) for p in extra.get("interactions", ()))
    treated_only = bool(extra.get("treated_only", False))

    # Standardised term vectors + main-effect scales, matching the factory on kept rows.
    term_vecs: dict[str, np.ndarray] = {"age": np.asarray(bp.A_std, dtype=float)}
    scales: dict[str, float] = {"age": 1.0}
    if ability_covariate is not None:
        z_ab, _ = standardise(bp.covariates[ability_covariate])
        term_vecs["ability"] = z_ab
        scales["ability"] = 1.0
    z_own, s_own = standardise(bp.pre_logit[own])
    term_vecs["own"] = z_own
    scales["own"] = s_own.sd
    for s in skill_symbols:
        z_s, sc = standardise(bp.pre_logit[s])
        term_vecs[s] = z_s
        scales[s] = sc.sd
    # The treatment indicator: a covariate marginal holds it fixed, but a ``trt ×
    # covariate`` interaction still moves with the covariate, so it must be available as
    # a partner. Omitted under treated_only (then constant, and the factory drops it).
    if not treated_only:
        term_vecs["trt"] = ((bp.G == 1) | (bp.phase >= 1)).astype(float)
    active_interactions = [
        pair for pair in interactions if not treated_only or "trt" not in pair
    ]

    def _ints_for(key: str) -> tuple[tuple[str, np.ndarray], ...]:
        out: list[tuple[str, np.ndarray]] = []
        for pair in active_interactions:
            if key not in pair:
                continue
            other = pair[0] if pair[1] == key else pair[1]
            if other not in term_vecs:  # partner unavailable (e.g. trt under treated_only)
                continue
            out.append(
                (f"gamma_int_{pair[0]}_{pair[1]}", np.asarray(term_vecs[other], dtype=float))
            )
        return tuple(out)

    def _sd_items(sd_logit: float, p: float, n: int) -> float:
        # Items equivalent of +1 SD of a bounded-count covariate, at its mean proportion.
        return float(n * (_expit(_logit(p) + sd_logit) - p))

    terms: list[_report.AssociationTerm] = []
    if not off_floor:
        p_own = float(np.mean(_expit(bp.pre_logit[own])))
        n_own = int(bp.n_trials[own])
        terms.append(
            AT("own", "gamma_own", scales["own"], _ints_for("own"),
               n_items=n_own, mean_prop=p_own, sd_items=_sd_items(scales["own"], p_own, n_own))
        )
    terms.append(AT("age", "gamma_A", 1.0, _ints_for("age")))
    if ability_covariate is not None:
        terms.append(AT("ability", "gamma_ability", 1.0, _ints_for("ability")))
    for s in skill_symbols:
        p_s = float(np.mean(_expit(bp.pre_logit[s])))
        n_s = int(bp.n_trials[s])
        terms.append(
            AT(s, f"gamma_{s}", scales[s], _ints_for(s),
               n_items=n_s, mean_prop=p_s, sd_items=_sd_items(scales[s], p_s, n_s))
        )
    for c in adjust_for:
        if c.endswith("_missing"):
            continue
        sd_c = float(np.std(np.asarray(bp.covariates[c], dtype=float), ddof=1))
        terms.append(AT(c, f"gamma_{c}", sd_c, ()))
    return terms


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

    # --- Per-covariate items-scale association marginals (#310) ---
    # The adjusted-association analogue of the treatment marginal: for each covariate
    # (own baseline, age, cognitive ability, skill baselines, raw-covariate adjusters)
    # push a +1 SD perturbation — and, for the bounded-count baselines, a +k-items one —
    # through the posterior onto the probability / items scales. Runs for the treated_only
    # (…b) companions too (they keep the covariate associations even without beta_trt).
    # Averaging population = ALL stacked rows (row_mask=None): these are descriptive
    # associations, not the randomised period-1 contrast, so every fitted observation
    # counts. That choice is recorded in config.json (meta_extra) as well as the note.
    assoc_terms = _gf_association_terms(
        spec, built, adjust_for=adjust_for, off_floor=off_floor
    )
    if assoc_terms:
        n_assoc = 1 if off_floor else built.prepared.n_trials[spec.outcome_symbol]
        am = _report.association_marginals(
            ctx.trace,
            terms=assoc_terms,
            n_trials=n_assoc,
            off_floor=off_floor,
            ci_prob=ctx.reporting.ci_prob,
            row_mask=None,
        )
        am.to_csv(
            os.path.join(ctx.output_dir, "association_marginals.csv"), index=False
        )
        ctx.tables["association_marginals"] = am
        meta_extra["association_marginals"] = {
            "averaging_population": "all_stacked_rows",
            "k_items": 5,
            "terms": [t.label for t in assoc_terms],
        }
        print_table(
            ranked_dataframe_table(
                am,
                title=f"Association marginals ({spec.outcome_symbol}) - items scale",
                columns=["term", "scale", "items_median", "items_lo", "items_hi", "prob_pos"],
                rank_column=False,
                precision=3,
            )
        )

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


def _bx_coef_names(
    spec: ModelSpec, adjust_for: tuple[str, ...] | None = None
) -> list[str]:
    """Interpretable block-exposure coefficients (alpha/alpha_time/kappa/sigma_child excluded)."""
    extra = spec.extra
    adj = extra.get("adjust_for", ()) if adjust_for is None else adjust_for
    names = ["delta", "gamma_A"]
    if extra.get("ability_covariate"):
        names.append("gamma_ability")
    names += [f"gamma_{c}" for c in adj]
    return names


def _bx_diag_vars(
    spec: ModelSpec, adjust_for: tuple[str, ...] | None = None
) -> list[str]:
    off_floor = spec.extra.get("likelihood") == "bernoulli_offfloor"
    tail: list[str] = [] if off_floor else ["kappa"]
    if spec.extra.get("use_child_re", True):
        tail.append("sigma_child")
    return ["alpha", "alpha_time", *_bx_coef_names(spec, adjust_for), *tail]


def fit_block_exposure(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Block-2 taught-vocabulary staggered block-active exposure fit (LRPBX, #228 item 5).

    Data-load like ``fit_level_factors`` (per-timepoint levels frame + the revised-DAG
    adjuster wiring); effect readout like ``fit_did`` (the focal ``delta`` + its
    items-scale AME). ``delta`` is an association (parallel-trends), so the factor
    summary flags no causal term.
    """
    _require_spec(spec, "block_exposure", outcome=True)
    ctx = make_context(spec, config)
    extra = spec.extra

    section_header("Prepare data")
    sym = spec.outcome_symbol
    ability_covariate = extra.get("ability_covariate")
    likelihood = extra.get("likelihood", "beta_binomial")
    off_floor = likelihood == "bernoulli_offfloor"
    obs_node = "y_offfloor" if off_floor else "y_post"
    baseline_covariates = (ability_covariate,) if ability_covariate else ()
    # Revised-DAG raw-covariate confounders, split by timing exactly as the
    # level-factor path (#247, A1 2026-07-13): language-proximal SP/RW (deapp_c/erbto)
    # read at the pre-randomisation baseline, hearing (hs) contemporaneous. Re-filter
    # after loading so a constant ``_missing`` indicator the loader drops is not gated.
    adjust_for = tuple(extra.get("adjust_for", ()))
    pre_adj, post_adj = split_covariates_by_wave(adjust_for)
    baseline_adj, post_adj = split_confounders_by_timing(post_adj)
    prepared = load_and_prepare(
        phase_mode="levels",
        outcomes=(sym,),
        baseline_covariates=(*baseline_covariates, *baseline_adj),
        covariates=pre_adj,
        post_covariates=post_adj,
        drop_ceiling_violations=tuple(extra.get("drop_ceiling_violations", ())),
    )
    adjust_for = tuple(c for c in adjust_for if c in prepared.covariates)
    ctx.prepared = prepared
    _print_header(ctx)

    section_header("Build model")
    built = _factories.build_block_exposure_model(
        prepared,
        outcome_symbol=sym,
        ability_covariate=ability_covariate,
        adjust_for=adjust_for,
        use_child_re=bool(extra.get("use_child_re", True)),
        likelihood=likelihood,
    )
    _attach_built(ctx, built)

    _render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    _diag.save_prior_predictive_plot(ctx, sym, node=obs_node)

    _run_sampling_and_loo(ctx)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=_bx_diag_vars(spec, adjust_for))

    _run_ppc(ctx, var_names=[obs_node])

    section_header("Extended diagnostics")
    # ``delta`` is the focal (association) effect — gets the prior-sensitivity +
    # forest evidence, exactly as the level-factor group term does.
    _diag.write_diagnostics_summary(ctx, var_names=_bx_diag_vars(spec, adjust_for))
    _diag.run_extended_diagnostics(ctx, causal_term="delta")
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=_bx_diag_vars(spec, adjust_for))
    _save_forest_plot(
        ctx, ["delta"], name="delta_forest.png",
        title="Block-active exposure effect (forest, reference line at 0)",
    )
    _diag.run_psense(ctx, var_names=["delta"])

    section_header("Factor summary")
    # No randomised contrast: block-active exposure is an association (parallel trends),
    # so no term is flagged causal.
    fs = _report.factor_summary(
        ctx.trace, _bx_coef_names(spec, adjust_for), ci_prob=ctx.reporting.ci_prob,
        causal_terms=(),
    )
    fs.to_csv(os.path.join(ctx.output_dir, "factor_summary.csv"), index=False)
    ctx.tables["factor_summary"] = fs
    _save_association_forest(ctx, _bx_coef_names(spec, adjust_for), ())
    print_table(
        ranked_dataframe_table(
            fs,
            title=f"Factor summary ({sym}) - {int(ctx.reporting.ci_prob * 100)}% CrI",
            columns=["term", "role", "median", "lo", "hi", "prob_positive"],
            rank_column=False,
            precision=3,
        )
    )

    section_header("Block-2 exposure effect summary")
    from language_reading_predictors.statistical_models.measures import MEASURES

    bx_s = _report.block_exposure_summary(
        ctx.trace,
        ci_prob=ctx.reporting.ci_prob,
        n_trials=1 if off_floor else MEASURES[sym].n_trials,
    )
    bx_df = pd.DataFrame([bx_s])
    bx_df.to_csv(os.path.join(ctx.output_dir, "block_exposure_summary.csv"), index=False)
    ctx.tables["block_exposure_summary"] = bx_df
    print_table(
        metrics_table(
            [{"metric": k, "value": v} for k, v in bx_s.items()],
            title=(
                f"block-2 exposure effect ({sym}) - "
                f"{int(ctx.reporting.ci_prob * 100)}% CI (equal-tailed); "
                "association (parallel-trends), positive = more taught-word learning"
            ),
            columns=["metric", "value"],
        )
    )

    _report.write_run_metadata(
        ctx,
        extra={"loo_elpd": float(ctx.loo.elpd), "block_exposure_summary": bx_s},
    )
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
    ``min_ess``/``min_bfmi``/``n_divergences``). The caller persists the verdict onto
    the sub-fit's published CSV: previously it was computed and discarded, so the
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
    conv = _diag.subfit_convergence(
        trace, label=label, var_names=[rv.name for rv in model.free_RVs]
    )
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
        return None, None, None
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


def _write_loo_influence(ctx: StatisticalFitContext) -> pd.DataFrame | None:
    """Persist pointwise Pareto-k values and explicit reliability flags.

    A sampler-convergence PASS does not guarantee that importance-sampled LOO is
    reliable.  Persisting the values makes the ``k > good_k`` gate available to
    report templates and downstream audits instead of leaving it visible only in
    a plot or the free-text ArviZ summary.
    """
    influence, threshold, _ = _influence_diagnostics(ctx)
    if influence is None or threshold is None:
        return None
    out = influence.copy()
    out["good_k_threshold"] = threshold
    out["loo_reliable"] = out["pareto_k"] <= threshold
    out.to_csv(os.path.join(ctx.output_dir, "pareto_k.csv"), index=False)
    ctx.tables["pareto_k"] = out
    return out


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


# ---------------------------------------------------------------------------
# Concurrent conditional-associations family (LRP-CA, #312, workstream #314)
# ---------------------------------------------------------------------------

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
        for stat in ("median", "lo", "hi", "lo90", "hi90")
    }


def _ca_sd_margin(df: pd.DataFrame, predictor: str) -> pd.Series:
    """Return the unique ``+1 SD`` marginal row for ``predictor``."""
    rows = df[(df["term"] == predictor) & (df["scale"] == "+1 SD")]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one +1 SD marginal for {predictor!r}; found {len(rows)}"
        )
    return rows.iloc[0]


_CA_MARGIN_STATS = ("median", "lo", "hi", "lo90", "hi90")
_CA_ASSOCIATION_REQUIRED = {
    "timepoint",
    "predictor",
    "label",
    "n",
    "predictor_n",
    "predictor_imputed_n",
    "ame_contrast",
    "adj_mean",
    "adj_lo",
    "adj_hi",
    "adj_lo90",
    "adj_hi90",
    "adj_prob_pos",
    "biv_mean",
    "biv_lo",
    "biv_hi",
    "biv_lo90",
    "biv_hi90",
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
    "prob_lo90",
    "prob_hi90",
    "items_median",
    "items_lo",
    "items_hi",
    "items_lo90",
    "items_hi90",
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
    wave-predictor association must have adjusted and bivariate ``+1 SD`` natural-scale
    rows and a matching fit-diagnostics row, while every wave has one adjusted-fit
    diagnostics row. This prevents a future refactor from silently publishing only one
    side of the requested adjusted/bivariate comparison.
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
        frame.to_csv(os.path.join(ctx.output_dir, f"{name}.csv"), index=False)
        ctx.tables[name] = frame

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
    bivariate (single-predictor, unadjusted) fit are sub-fits. Every published fit has
    R-hat, ESS, BFMI and divergence diagnostics recorded in
    ``concurrent_fit_diagnostics.csv``. ``concurrent_associations.csv`` carries the
    adjusted and bivariate logit coefficients plus matched +1-SD probability/items
    marginals (wave × predictor); ``concurrent_marginals.csv`` carries both fit kinds'
    detailed probability/items marginals (wave × predictor × {+1 SD, +k items}).
    """
    _require_spec(spec, "concurrent", outcome=True)
    e = spec.extra
    outcome = spec.outcome_symbol or "W"
    predictor_symbols = list(e.get("predictor_symbols", ["L", "B", "TR", "TE", "R", "E"]))
    include_age = bool(e.get("include_age", True))
    include_group = bool(e.get("include_group", True))
    sigma0 = float(
        e.get(
            "predictor_slope_sigma",
            _default_of(_factories.build_concurrent_model, "predictor_slope_sigma"),
        )
    )

    from language_reading_predictors.statistical_models.measures import MEASURES

    ctx = make_context(spec, config)
    hdi = ctx.reporting.ci_prob
    N_focal = MEASURES[outcome].n_trials

    section_header("Prepare data")
    measure_outcomes = tuple(dict.fromkeys([outcome, *predictor_symbols]))
    prepared_all = load_and_prepare(phase_mode="levels", outcomes=measure_outcomes)

    # Timepoints present; each wave's row count and its usable predictor set (a
    # predictor whose same-wave logit has positive variance on the wave's rows —
    # anything constant/all-missing at that wave is dropped, and a wave with no usable
    # predictor is skipped below).
    wave_indices = sorted({int(p) for p in np.unique(prepared_all.phase)})
    wave_subsets: dict[int, object] = {}
    wave_n: dict[int, int] = {}
    wave_preds: dict[int, list[str]] = {}
    dropped_by_wave: dict[int, list[str]] = {}
    for w in wave_indices:
        sub = _subset_prepared(prepared_all, prepared_all.phase == w)
        keep = ~np.isnan(sub.post_counts[outcome])
        sub = _subset_prepared(sub, keep)
        wave_subsets[w] = sub
        wave_n[w] = sub.n_obs
        wave_preds[w], dropped_by_wave[w] = _ca_wave_predictors(sub, predictor_symbols)
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
    _print_header(ctx)

    def _build(sub, preds, *, age, group):
        return _factories.build_concurrent_model(
            sub,
            outcome_symbol=outcome,
            predictor_symbols=preds,
            include_age=age,
            include_group=group,
            predictor_slope_sigma=sigma0,
        )

    # ---- Fit each wave's mutually-adjusted model --------------------------------
    wave_fits: dict[int, dict] = {}
    for w in wave_indices:
        sub = wave_subsets[w]
        preds = wave_preds[w]
        tp = w + 1  # 1-based timepoint for reports
        if not preds:
            rprint(f"[yellow]Concurrent: wave t{tp} has no usable predictors; skipped.[/yellow]")
            continue
        if w == primary_wave:
            section_header(f"Build model (primary wave t{tp})")
            built = _build(sub, preds, age=include_age, group=include_group)
            _attach_built(ctx, built)
            _render_model_graph(ctx)
            section_header("Prior predictive")
            _diag.run_prior_predictive(ctx, draws=1000)
            _diag.save_prior_predictive_plot(ctx, outcome)
            _run_sampling_and_loo(ctx)
            trace = ctx.trace
            convergence = None  # populated below after the full primary gate
        else:
            built = _build(sub, preds, age=include_age, group=include_group)
            trace, conv = _sample_model(
                built.model, ctx.sampling, label=f"{spec.model_id} wave t{tp}"
            )
            convergence = conv
        wave_fits[w] = {
            "trace": trace,
            "prepared": built.prepared,
            "preds": preds,
            "convergence": convergence,
        }

    # ---- Primary-wave diagnostics + standard artefacts --------------------------
    section_header("Summary diagnostics (primary wave)")
    prim = wave_fits[primary_wave]
    beta_names = [f"beta_{s}" for s in prim["preds"]]
    diag_vars = ["alpha", "kappa", *beta_names]
    if include_age:
        diag_vars.append("beta_age")
    if include_group:
        diag_vars.append("beta_group_nuisance")
    _diag.summary_diagnostics(ctx, var_names=diag_vars)
    _run_ppc(ctx)
    section_header("Extended diagnostics (primary wave)")
    _primary_gate = _diag.write_diagnostics_summary(ctx, var_names=diag_vars)
    _primary_conv = _diag.subfit_convergence(
        ctx.trace,
        label=f"{spec.model_id} primary wave t{primary_wave + 1}",
        var_names=[rv.name for rv in ctx.model.free_RVs],
    )
    if _primary_gate:
        _primary_conv["converged"] = bool(
            _primary_gate.get("passed") and _primary_conv.get("converged")
        )
    wave_fits[primary_wave]["convergence"] = _primary_conv
    _diag.run_extended_diagnostics(ctx)
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=diag_vars)

    # ---- Adjusted vs bivariate coefficients + natural-scale marginals -----------
    section_header("Concurrent associations (adjusted vs bivariate)")
    assoc_rows: list[dict] = []
    marg_frames: list[pd.DataFrame] = []
    fit_diagnostic_rows: list[dict] = []
    for w in wave_indices:
        if w not in wave_fits:
            continue
        tp = w + 1
        fit = wave_fits[w]
        sub, preds, trace = fit["prepared"], fit["preds"], fit["trace"]
        adj_conv = fit["convergence"]
        fit_diagnostic_rows.append(
            {
                "timepoint": tp,
                "fit_kind": "adjusted",
                "predictor": "all",
                "n": sub.n_obs,
                "n_predictors": len(preds),
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

        # Per-predictor: adjusted beta (this wave's full fit) + bivariate beta (refit).
        for sym in preds:
            adj = _beta_summary(trace, f"beta_{sym}", hdi)
            b = _build(sub, [sym], age=False, group=False)
            bt, bconv = _sample_model(
                b.model, ctx.sampling, label=f"{spec.model_id} t{tp} bivariate {sym}"
            )
            biv = _beta_summary(bt, f"beta_{sym}", hdi)
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
                    "adj_mean": adj["mean"],
                    "adj_lo": adj["lo"],
                    "adj_hi": adj["hi"],
                    "adj_lo90": adj["lo90"],
                    "adj_hi90": adj["hi90"],
                    "adj_prob_pos": adj["prob_pos"],
                    **_ca_margin_fields("adj", adj_sd),
                    "biv_mean": biv["mean"],
                    "biv_lo": biv["lo"],
                    "biv_hi": biv["hi"],
                    "biv_lo90": biv["lo90"],
                    "biv_hi90": biv["hi90"],
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
        "bivariate_adjustment": "predictor only; age, group and other skills omitted",
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
            "concurrent_associations.csv contains adjusted and bivariate logit, "
            "probability and items summaries for +1 SD; concurrent_marginals.csv "
            "contains both fit kinds for +1 SD and +k items"
        ),
    }
    _report.write_run_metadata(ctx, extra=meta_extra)

    return _finalize_report(ctx)


def _plot_concurrent(
    ctx: StatisticalFitContext, df: pd.DataFrame, hdi: float, *, primary_tp: int
) -> None:
    """Forest of adjusted vs bivariate coefficients for the primary wave (#312)."""
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
        fmt="o", color="#1f77b4", capsize=3, label="adjusted (mutual)",
    )
    plt.errorbar(
        d["biv_mean"], y - 0.12,
        xerr=[d["biv_mean"] - d["biv_lo"], d["biv_hi"] - d["biv_mean"]],
        fmt="s", color="#999999", capsize=3, label="bivariate (unadjusted)",
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
    ``lagged_change_couplings`` (LCSM-091, #229 spec 2) adds prior-transition
    latent-change terms (``h_{src}``) to the named targets' change equations.
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
    lagged_in = spec.extra.get("lagged_change_couplings")
    lagged_change_couplings: dict[str, tuple[str, ...]] = (
        {tgt: tuple(srcs) for tgt, srcs in lagged_in.items()} if lagged_in else {}
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
        lagged_change_couplings=lagged_change_couplings or None,
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
    # Lagged change-on-change names mirror the factory's rule on the lag map.
    single_lag_target = len(lagged_change_couplings) == 1
    lagged_names = {
        (src, tgt): (f"h_{src}" if single_lag_target else f"h_{src}_{tgt}")
        for tgt, srcs in lagged_change_couplings.items()
        for src in srcs
    }
    diag_vars = list(coupling_names.values())
    diag_vars += list(lagged_names.values())
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
    rows += [
        _coef_row(
            f"{pname} (prior {src} change -> {tgt} change)",
            post[pname].values,
            ctx.reporting.ci_prob,
        )
        for (src, tgt), pname in lagged_names.items()
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
    # #228 item 14 (errors-in-variables mechanism): optionally regress the outcome on a
    # SUBSET of the fitted factors (e.g. just "code") and/or add the randomised arm G as
    # an adjusted-association covariate. Defaults reproduce mm-001/101 exactly.
    _sf = spec.extra.get("structural_factors")
    structural_factors = tuple(_sf) if _sf is not None else None
    use_group = bool(spec.extra.get("use_group", False))
    indicator_syms = tuple(dict.fromkeys(s for v in domains.values() for s in v))
    measure_outcomes = tuple(dict.fromkeys((outcome, *indicator_syms)))
    prepared = load_and_prepare(
        phase_mode="span",
        post_time=int(spec.extra.get("post_time", 4)),
        outcomes=measure_outcomes,
        covariates=structural_covs,
    )
    ctx.prepared = prepared
    # A structural covariate can go constant on the fitted span rows — e.g. an
    # ``erbto_missing`` indicator that is all-zero because phonological memory is
    # observed for every fitted child at t1 — so the loader drops it. Re-filter to the
    # covariates actually present, mirroring the mechanism/mediation pipelines'
    # #247/#258 re-filter, so the factory is not asked for a coefficient on a dropped
    # covariate (it raises KeyError otherwise) and the effective set is honest.
    _dropped_structural = tuple(c for c in structural_covs if c not in prepared.covariates)
    if _dropped_structural:
        structural_covs = tuple(c for c in structural_covs if c in prepared.covariates)
        rprint(
            "[yellow]fit_correlated_factor: dropped constant structural covariate(s) "
            f"{list(_dropped_structural)} (not in prepared.covariates on the fitted "
            "rows).[/yellow]"
        )
    _print_header(ctx)

    section_header("Build model")
    built = _factories.build_correlated_factor_model(
        prepared,
        outcome_symbol=outcome,
        domains=domains,
        structural_covariates=structural_covs,
        structural_factors=structural_factors,
        use_group=use_group,
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
    if use_group:
        summary_vars.append("beta_G")

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
    # The structural leg regresses on all domain factors (beta_factor dims "domain")
    # unless structural_factors isolated a subset (dims "struct_domain", #228 item 14).
    struct_names = list(structural_factors) if structural_factors is not None else dnames
    _bf_dim = "struct_domain" if structural_factors is not None else "domain"
    struct_rows = [
        _coef_row(f"beta_{d}", post["beta_factor"].isel({_bf_dim: k}).values, hdi)
        for k, d in enumerate(struct_names)
    ]
    extra_terms = (
        (["beta_G"] if use_group else [])
        + (["beta_age"] if spec.extra.get("use_age", True) else [])
        + [f"beta_{c}" for c in structural_covs]
    )
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


# ---------------------------------------------------------------------------
# Longitudinal correlated-domain-factor model (LRP-RLI-LCF-001, #313)
# ---------------------------------------------------------------------------


_LCF_DEFAULT_DOMAINS = {
    "vocabulary": ("R", "E", "TR", "TE"),
    "code": ("L", "B"),
    "grammar": ("F", "T"),
}


def _lcf_child_log_likelihood(trace, built, *, chunk_size: int = 256):
    """Evaluate the LCF's exact per-child MvNormal log likelihood.

    PyMC 6.1 cannot reconstruct this model's log likelihood from its posterior:
    removing the ``LKJCorr`` transforms gives the compiled value variables an extra
    ``_cholesky`` suffix, while the posterior keeps the random-variable names. The
    generic helper therefore rejects the inputs before evaluating the density.

    The fitted trace already contains the exact marginal mean ``mean_z`` and
    covariance ``Sigma_z`` used by every observed-pattern ``MvNormal``. Evaluate
    that density directly, in bounded draw chunks, and return one contribution per
    child. This avoids PyMC's transformed-value plumbing. It can also recover LOO
    from a persisted trace when the exact matching model and data are rebuilt; the
    coordinate and observed-data checks below reject positional drift.
    """
    import xarray as xr

    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive (got {chunk_size})")

    posterior = trace.posterior
    mean_z = posterior["mean_z"].transpose("chain", "draw", "cell")
    sigma_z = posterior["Sigma_z"].transpose(
        "chain", "draw", "cell", "cell_b"
    )
    expected_cells = np.asarray(built.extras["cell_names"], dtype=str)
    for variable, dimension in ((mean_z, "cell"), (sigma_z, "cell"), (sigma_z, "cell_b")):
        actual_cells = np.asarray(variable.coords[dimension].values, dtype=str)
        if not np.array_equal(actual_cells, expected_cells):
            raise ValueError(
                f"LCF posterior {dimension!r} coordinates do not match the rebuilt "
                "model; refusing positional likelihood recovery"
            )
    n_chains = mean_z.sizes["chain"]
    n_draws = mean_z.sizes["draw"]

    groups = []
    all_children: list[int] = []
    for node in built.extras["z_nodes"]:
        children = np.asarray(built.extras["child_of_node"][node], dtype=int)
        cell_indices = np.asarray(
            built.extras["cell_indices_of_node"][node], dtype=int
        )
        observed = np.asarray(built.extras["observed_z_of_node"][node], dtype=float)
        expected_shape = (len(children), len(cell_indices))
        if observed.shape != expected_shape:
            raise ValueError(
                f"{node} observed data have shape {observed.shape}; expected "
                f"{expected_shape} from the child/cell indices"
            )
        if "observed_data" in trace.children:
            if node not in trace.observed_data:
                raise ValueError(f"LCF persisted observed_data is missing {node!r}")
            persisted = np.asarray(trace.observed_data[node].values, dtype=float)
            if persisted.shape != observed.shape or not np.allclose(
                persisted, observed, rtol=0.0, atol=0.0
            ):
                raise ValueError(
                    f"LCF persisted observations for {node!r} do not match the "
                    "rebuilt model data"
                )
        groups.append((children, cell_indices, observed))
        all_children.extend(children.tolist())

    used_children = np.asarray(sorted(all_children), dtype=int)
    if len(np.unique(used_children)) != len(used_children):
        raise ValueError("LCF observed-pattern groups assign a child more than once")
    expected_children = int(built.extras["n_used_children"])
    if len(used_children) != expected_children:
        raise ValueError(
            f"LCF observed-pattern groups cover {len(used_children)} children; "
            f"expected {expected_children}"
        )
    child_column = {child: i for i, child in enumerate(used_children)}

    log_likelihood = np.full(
        (n_chains, n_draws, expected_children), np.nan, dtype=float
    )
    log_2pi = np.log(2.0 * np.pi)
    for chain in range(n_chains):
        for draw_start in range(0, n_draws, chunk_size):
            draw_stop = min(draw_start + chunk_size, n_draws)
            means = np.asarray(
                mean_z.isel(chain=chain, draw=slice(draw_start, draw_stop))
            )
            covariances = np.asarray(
                sigma_z.isel(chain=chain, draw=slice(draw_start, draw_stop))
            )
            for children, cell_indices, observed in groups:
                covariance = covariances[:, cell_indices[:, None], cell_indices]
                mean = means[:, cell_indices]
                chol = np.linalg.cholesky(covariance)
                residual = observed[None, :, :] - mean[:, None, :]
                whitened = np.linalg.solve(chol, np.swapaxes(residual, 1, 2))
                quadratic = np.sum(whitened**2, axis=1)
                log_determinant = 2.0 * np.sum(
                    np.log(np.diagonal(chol, axis1=1, axis2=2)), axis=1
                )
                values = -0.5 * (
                    len(cell_indices) * log_2pi
                    + log_determinant[:, None]
                    + quadratic
                )
                columns = np.asarray(
                    [child_column[int(child)] for child in children], dtype=int
                )
                # Advanced indexing places the child dimension first on the target.
                log_likelihood[chain, draw_start:draw_stop, columns] = values.T

    if not np.isfinite(log_likelihood).all():
        raise ValueError("LCF per-child log likelihood contains non-finite values")

    return xr.DataArray(
        log_likelihood,
        dims=("chain", "draw", "child_lcf"),
        coords={
            "chain": mean_z.coords["chain"],
            "draw": mean_z.coords["draw"],
            "child_lcf": used_children,
        },
        name="lcf_child",
    )


def _lcf_log_prior(trace, model):
    """Compute exact constrained-scale LCF log-prior terms.

    PyMC 6.1's generic ``compute_log_prior`` removes transforms but then supplies
    the posterior's random-variable names to the resulting value variables. For
    ``LKJCorr`` those names differ (for example ``trait_corr_chol`` versus
    ``trait_corr_chol_cholesky``), so the generic helper fails before evaluation.

    Isolate the version-sensitive workaround here: build the same transform-free
    model PyMC uses, rename constrained posterior inputs to its value-variable names,
    and apply the compiled elementwise prior density over chain and draw. Keeping
    the component terms (rather than only their sum) preserves the standard
    ``log_prior`` contract used by power-scaling sensitivity diagnostics.
    """
    from pymc.backends.arviz import (
        apply_function_over_dataset,
        coords_and_dims_for_inferencedata,
    )
    from pymc.model.transform.conditioning import remove_value_transforms

    posterior_node = trace.posterior
    posterior = (
        posterior_node.to_dataset()
        if hasattr(posterior_node, "to_dataset")
        else posterior_node
    )
    original_names = [rv.name for rv in model.free_RVs]
    missing = sorted(set(original_names) - set(posterior.data_vars))
    if missing:
        raise ValueError(f"LCF posterior is missing free variables: {missing}")

    untransformed_model = remove_value_transforms(model)
    prior_rvs = list(untransformed_model.free_RVs)
    inputs = posterior[original_names]
    rename = {
        rv.name: untransformed_model.rvs_to_values[rv].name
        for rv in prior_rvs
        if rv.name != untransformed_model.rvs_to_values[rv].name
    }
    inputs = inputs.rename(rename)
    inputs = inputs.astype(
        {
            value.name: value.type.dtype
            for value in untransformed_model.value_vars
        },
        copy=False,
    )
    logp_fn = untransformed_model.compile_fn(
        inputs=untransformed_model.value_vars,
        outs=untransformed_model.logp(vars=prior_rvs, sum=False),
        on_unused_input="ignore",
    )
    coords, dims = coords_and_dims_for_inferencedata(untransformed_model)
    result = apply_function_over_dataset(
        logp_fn,
        inputs,
        output_var_names=[rv.name for rv in prior_rvs],
        sample_dims=("chain", "draw"),
        coords=coords,
        dims=dims,
        progressbar=False,
    )
    if any(not np.isfinite(variable).all() for variable in result.data_vars.values()):
        raise ValueError("LCF log-prior contains non-finite values")
    return result


def _lcf_stitch_loo(ctx: StatisticalFitContext, built) -> None:
    """Pointwise PSIS-LOO for the longitudinal CFA (custom, per-child stitch).

    The masked-cell likelihood is one ``MvNormal`` per observed-cell pattern, so
    there is no single observed node ``az.loo`` can key on. Compute each pattern
    node's per-child log-likelihood, stitch them into one ``(chain, draw, child)``
    array over all children, and run pointwise LOO on that — the honest per-child
    predictive score an invariance comparison (#313) would use. Attach the exact
    constrained-scale prior terms through the companion workaround above. As in the
    other LOO-enabled families, a failure is fatal: a reporting run must not silently
    complete without the likelihood, prior and predictive diagnostics its output
    contract requires.
    """
    import arviz as az
    import xarray as xr

    stitched = _lcf_child_log_likelihood(ctx.trace, built)
    if "log_likelihood" not in ctx.trace.children:
        ctx.trace["log_likelihood"] = xr.Dataset()
    ctx.trace.log_likelihood["lcf_child"] = stitched
    ctx.trace["log_prior"] = _lcf_log_prior(ctx.trace, ctx.model)
    ctx.loo = az.loo(ctx.trace, var_name="lcf_child", pointwise=True)
    _report.write_loo_summary(ctx)
    _print_loo_row(ctx)


def _lcf_observed_domain_corr(built) -> pd.DataFrame:
    """Observed same-wave cross-domain indicator correlations for triangulation.

    For each wave and each unique domain pair, compute the mean pairwise
    (pairwise-complete) Pearson correlation between the two domains' standardised
    logit indicators. This is a descriptive comparator, not an attenuation-bound:
    it is not the same estimand as the model's latent factor correlation.
    """
    panel = built.prepared
    domains = built.extras["domains"]
    standardisers = built.extras["standardisers"]
    waves = built.extras["waves"]
    dnames = list(domains)
    # Standardised logit per indicator (pooled, exactly as the factory), (N, T).
    z = {}
    for d, syms in domains.items():
        for s in syms:
            mean, sd = standardisers[s]
            z[s] = (np.asarray(panel.logit[s], dtype=float) - mean) / sd
    rows = []
    for w_i in range(len(waves)):
        for i in range(len(dnames)):
            for j in range(i + 1, len(dnames)):
                vals = []
                for si in domains[dnames[i]]:
                    for sj in domains[dnames[j]]:
                        a = z[si][:, w_i]
                        b = z[sj][:, w_i]
                        m = np.isfinite(a) & np.isfinite(b)
                        if m.sum() >= 3 and np.std(a[m]) > 0 and np.std(b[m]) > 0:
                            vals.append(float(np.corrcoef(a[m], b[m])[0, 1]))
                rows.append(
                    {
                        "wave": waves[w_i],
                        "domain_i": dnames[i],
                        "domain_j": dnames[j],
                        "observed_corr": float(np.mean(vals)) if vals else float("nan"),
                        "n_indicator_pairs": len(vals),
                    }
                )
    return pd.DataFrame(rows)


def _lcf_items_scale(ctx, built) -> pd.DataFrame:
    """Approximate items-scale translation of the headline cross-domain couplings.

    For one representative indicator pair per cross-domain combination (the first
    listed indicator of each domain), the delta-method slope of the target
    indicator's item count per +1 item of the predictor indicator, at the pooled-mean
    operating point, per wave. Derived from the latent correlation and the loadings:
    ``slope_z = lambda_m lambda_k rho / (lambda_k^2 + sigma_k^2)`` on the standardised
    logit scale, scaled to items by the two indicators' logit SDs, denominators, and
    operating-point ``p(1-p)``. **Approximate and descriptive** — a linearisation at
    the mean, comparable to the #312 concurrent items-scale marginals, not a causal
    effect.
    """
    from scipy.special import expit

    post = ctx.trace.posterior
    domains = built.extras["domains"]
    standardisers = built.extras["standardisers"]
    waves = built.extras["waves"]
    hdi = ctx.reporting.ci_prob
    lo_q = (1 - hdi) / 2
    dnames = list(domains)
    ind_names = [str(s) for s in post["indicator"].values]

    def _lam_sig(sym):
        k = ind_names.index(sym)
        lam = post["lambda_load"].isel(indicator=k).stack(sample=("chain", "draw")).values.ravel()
        sig = post["sigma_indicator"].isel(indicator=k).stack(sample=("chain", "draw")).values.ravel()
        return lam, sig

    fc = post["factor_corr"].stack(sample=("chain", "draw"))
    fc = fc.transpose("sample", "wave", "domain", "domain_b")
    corr = np.asarray(fc.values)  # (S, T, D, D)
    dom_idx = {d: i for i, d in enumerate(dnames)}

    rows = []
    for i in range(len(dnames)):
        for j in range(i + 1, len(dnames)):
            di, dj = dnames[i], dnames[j]
            k_sym = domains[dj][0]  # predictor indicator (domain j)
            m_sym = domains[di][0]  # target indicator (domain i)
            lam_k, sig_k = _lam_sig(k_sym)
            lam_m, _ = _lam_sig(m_sym)
            mean_k, sd_k = standardisers[k_sym]
            mean_m, sd_m = standardisers[m_sym]
            N_k = ctx.prepared.n_trials[k_sym]
            N_m = ctx.prepared.n_trials[m_sym]
            p_k = float(expit(mean_k))
            p_m = float(expit(mean_m))
            # ``logit_safe`` uses (y + 0.5)/(N + 1), so the inverse-count
            # derivative is (N + 1) p(1-p), not the binomial N p(1-p).
            info_k = (N_k + 1) * p_k * (1 - p_k)
            info_m = (N_m + 1) * p_m * (1 - p_m)
            for w_i, w in enumerate(waves):
                rho = corr[:, w_i, dom_idx[di], dom_idx[dj]]
                slope_z = lam_m * lam_k * rho / (lam_k**2 + sig_k**2)
                # Δitems_m per +1 item of k at the mean operating point.
                items_slope = slope_z * (sd_m / sd_k) * (info_m / info_k)
                rows.append(
                    {
                        "wave": w,
                        "predictor_indicator": k_sym,
                        "target_indicator": m_sym,
                        "predictor_domain": dj,
                        "target_domain": di,
                        "items_per_item_mean": float(np.mean(items_slope)),
                        "items_per_item_lo": float(np.quantile(items_slope, lo_q)),
                        "items_per_item_hi": float(np.quantile(items_slope, 1 - lo_q)),
                        "prob_pos": float(np.mean(items_slope > 0)),
                    }
                )
    return pd.DataFrame(rows)


_LCF_CA_COMPARISONS = {
    "L": ("R", "E", "TR", "TE"),
    "TR": ("L", "B"),
    "TE": ("L", "B"),
}
_LCF_CA_MODEL_IDS = {
    "L": "lrp-rli-ca-002",
    "TR": "lrp-rli-ca-003",
    "TE": "lrp-rli-ca-004",
}


def _lcf_observed_conditional_slope(
    corr: np.ndarray,
    loadings: np.ndarray,
    residual_sds: np.ndarray,
    *,
    target_domain_idx: int,
    predictor_domain_idx: int,
    target_indicator_idx: int,
    predictor_indicator_idx: int,
) -> np.ndarray:
    """Observed-indicator slope implied by an LCF conditional domain coupling.

    Conditions on every latent domain other than the target and predictor. If
    ``C`` denotes those domains, the relevant factor covariance and predictor
    variance are ``Cov(f_a, f_b | f_C)`` and ``Var(f_b | f_C)``. Mapping through
    the target loading and the noisy predictor indicator gives

    ``lambda_a lambda_b Cov_ab.C / (lambda_b^2 Var_b.C + sigma_b^2)``.

    Using the marginal reliability ``lambda_b / (lambda_b^2 + sigma_b^2)`` here
    would mix a C-conditional factor coefficient with a marginal measurement
    update and overstate or understate the resulting observed-score slope.
    """
    corr = np.asarray(corr, dtype=float)
    n_domains = corr.shape[-1]
    conditioned = [
        idx
        for idx in range(n_domains)
        if idx not in {target_domain_idx, predictor_domain_idx}
    ]
    if conditioned:
        corr_cc = corr[:, :, conditioned, :][:, :, :, conditioned]
        corr_cb = corr[:, :, conditioned, predictor_domain_idx]
        solve_cb = np.linalg.solve(corr_cc, corr_cb[..., None])[..., 0]
        predictor_variance = 1.0 - np.sum(
            corr[:, :, predictor_domain_idx, conditioned] * solve_cb, axis=-1
        )
        conditional_covariance = corr[
            :, :, target_domain_idx, predictor_domain_idx
        ] - np.sum(
            corr[:, :, target_domain_idx, conditioned] * solve_cb, axis=-1
        )
    else:
        predictor_variance = np.ones(corr.shape[:2], dtype=float)
        conditional_covariance = corr[:, :, target_domain_idx, predictor_domain_idx]

    lambda_target = loadings[:, target_indicator_idx, None]
    lambda_predictor = loadings[:, predictor_indicator_idx, None]
    sigma_predictor = residual_sds[:, predictor_indicator_idx, None]
    return (
        lambda_target * lambda_predictor * conditional_covariance
        / (
            lambda_predictor**2 * predictor_variance
            + sigma_predictor**2
        )
    )


def _lcf_concurrent_comparison(
    ctx,
    built,
    *,
    ca_tables: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Reproducible directed comparison with matching #312 associations.

    For each cross-domain target/predictor pair shared with CA002--004, translate
    the LCF's latent conditional slope to target items for a one same-wave-SD
    increase in the observed predictor. The translation conditions on the other
    latent domains and is evaluated at the target's observed wave-mean logit. Place
    it beside #312's adjusted average marginal effect for the same direction and
    raw predictor contrast.

    The columns deliberately keep the two estimates separate: #312 conditions on
    five observed tests plus age/group terms and averages a nonlinear marginal over
    children, whereas the LCF conditions on the remaining latent domains and uses a
    mean-operating-point translation. This is a directional triangulation table,
    not a claim that the numbers estimate the same parameter.
    """
    from scipy.special import expit

    post = ctx.trace.posterior
    ci_prob = ctx.reporting.ci_prob
    lo_q = (1.0 - ci_prob) / 2.0
    domains = [str(value) for value in post.coords["domain"].values]
    domain_index = {domain: i for i, domain in enumerate(domains)}
    indicator_names = [str(value) for value in post.coords["indicator"].values]
    indicator_index = {symbol: i for i, symbol in enumerate(indicator_names)}
    domain_of = built.extras["domain_of"]

    corr = (
        post["factor_corr"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "wave", "domain", "domain_b")
        .values
    )
    loadings = (
        post["lambda_load"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "indicator")
        .values
    )
    residual_sds = (
        post["sigma_indicator"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "indicator")
        .values
    )

    if ca_tables is None:
        ca_tables = {}
        models_dir = os.path.dirname(ctx.output_dir)
        config_name = ctx.reporting.config_name
        for target, model_id in _LCF_CA_MODEL_IDS.items():
            path = os.path.join(
                models_dir,
                f"{model_id}-{config_name}",
                "concurrent_marginals.csv",
            )
            if os.path.exists(path):
                ca_tables[target] = pd.read_csv(path)

    rows: list[dict] = []
    panel = built.prepared
    for target, predictors in _LCF_CA_COMPARISONS.items():
        if target not in indicator_index or target not in domain_of:
            continue
        target_domain = domain_of[target]
        target_domain_idx = domain_index[target_domain]
        target_indicator_idx = indicator_index[target]
        target_sd = float(built.extras["standardisers"][target][1])
        target_trials = int(panel.n_trials[target])
        ca_table = ca_tables.get(target)

        for predictor in predictors:
            if predictor not in indicator_index or predictor not in domain_of:
                continue
            predictor_domain = domain_of[predictor]
            if predictor_domain == target_domain:
                continue
            predictor_domain_idx = domain_index[predictor_domain]
            predictor_indicator_idx = indicator_index[predictor]
            predictor_pooled_sd = float(
                built.extras["standardisers"][predictor][1]
            )

            observed_slope = _lcf_observed_conditional_slope(
                corr,
                loadings,
                residual_sds,
                target_domain_idx=target_domain_idx,
                predictor_domain_idx=predictor_domain_idx,
                target_indicator_idx=target_indicator_idx,
                predictor_indicator_idx=predictor_indicator_idx,
            )

            for wave_idx, wave in enumerate(built.extras["waves"]):
                predictor_wave = np.asarray(panel.logit[predictor][:, wave_idx])
                target_wave = np.asarray(panel.logit[target][:, wave_idx])
                fitted_rows = np.isfinite(target_wave)
                predictor_wave_sd = float(
                    np.nanstd(predictor_wave[fitted_rows], ddof=1)
                )
                target_wave_mean = float(np.nanmean(target_wave[fitted_rows]))
                if not (
                    np.isfinite(predictor_wave_sd)
                    and predictor_wave_sd > 0
                    and np.isfinite(target_wave_mean)
                ):
                    continue
                predictor_delta_z = predictor_wave_sd / predictor_pooled_sd
                target_delta_logit = (
                    observed_slope[:, wave_idx] * predictor_delta_z * target_sd
                )
                # ``logit_safe`` uses the Haldane proportion (y + 0.5)/(N + 1),
                # whose inverse count difference is (N + 1) times the probability
                # difference (the -0.5 constants cancel).
                lcf_items = (target_trials + 1) * (
                    expit(target_wave_mean + target_delta_logit)
                    - expit(target_wave_mean)
                )

                ca_row = None
                if ca_table is not None:
                    matched = ca_table[
                        (ca_table["timepoint"] == int(wave))
                        & (ca_table["adjustment"] == "adjusted")
                        & (ca_table["term"] == predictor)
                        & (ca_table["scale"] == "+1 SD")
                    ]
                    if len(matched) > 1:
                        raise ValueError(
                            f"Expected at most one #312 row for {target} <- "
                            f"{predictor} at wave {wave}; found {len(matched)}"
                        )
                    if len(matched) == 1:
                        ca_row = matched.iloc[0]

                rows.append(
                    {
                        "wave": int(wave),
                        "target_indicator": target,
                        "predictor_indicator": predictor,
                        "target_domain": target_domain,
                        "predictor_domain": predictor_domain,
                        "predictor_contrast": "+1 same-wave SD",
                        "lcf_items_median": float(np.median(lcf_items)),
                        "lcf_items_lo": float(np.quantile(lcf_items, lo_q)),
                        "lcf_items_hi": float(np.quantile(lcf_items, 1.0 - lo_q)),
                        "lcf_prob_pos": float(np.mean(lcf_items > 0)),
                        "ca_items_median": (
                            float(ca_row["items_median"])
                            if ca_row is not None
                            else float("nan")
                        ),
                        "ca_items_lo": (
                            float(ca_row["items_lo"])
                            if ca_row is not None
                            else float("nan")
                        ),
                        "ca_items_hi": (
                            float(ca_row["items_hi"])
                            if ca_row is not None
                            else float("nan")
                        ),
                        "ca_prob_pos": (
                            float(ca_row["prob_pos"])
                            if ca_row is not None
                            else float("nan")
                        ),
                        "ca_available": ca_row is not None,
                        "ca_model_id": _LCF_CA_MODEL_IDS[target],
                    }
                )
    return pd.DataFrame(rows)


def fit_longitudinal_corr_factor(
    spec: ModelSpec, config: str = "dev"
) -> StatisticalFitContext:
    """Longitudinal correlated-domain-factor model (LRP-RLI-LCF-001, #313).

    Fits the four-wave extension of the ``corr_factor`` CFA over the child×wave
    panel: correlated vocabulary / code / grammar factors at every timepoint, with a
    trait/state across-wave structure and the factor scores marginalised out. Reports
    the per-wave latent skill correlation matrices, the conditional (partial) latent
    slopes, the loadings / communalities, and a descriptive comparison against the
    observed same-wave correlations (the #312 anchor). The quantities differ in their
    aggregation and conditioning, so no magnitude ordering is required. A measurement
    / triangulation model — every quantity is a descriptive association, never causal.
    """
    _require_spec(spec, "long_corr_factor")

    ctx = make_context(spec, config)
    # A small-n latent model; even fully marginalised a few boundary divergences can
    # survive at the tier default, so lift target_accept via the spec (as mm-001 does).
    _apply_spec_target_accept(ctx, spec)

    section_header("Prepare data")
    domains = {
        k: tuple(v)
        for k, v in (spec.extra.get("domains") or _LCF_DEFAULT_DOMAINS).items()
    }
    indicators = tuple(dict.fromkeys(s for v in domains.values() for s in v))
    panel = load_wave_panel(outcomes=indicators)
    ctx.prepared = panel
    _print_header(ctx)

    section_header("Build model")
    built = _factories.build_longitudinal_corr_factor_model(
        panel,
        domains=domains,
        loading_sigma=spec.extra.get(
            "loading_sigma",
            _default_of(_factories.build_longitudinal_corr_factor_model, "loading_sigma"),
        ),
        residual_sigma=spec.extra.get(
            "residual_sigma",
            _default_of(_factories.build_longitudinal_corr_factor_model, "residual_sigma"),
        ),
        lkj_eta=spec.extra.get(
            "lkj_eta",
            _default_of(_factories.build_longitudinal_corr_factor_model, "lkj_eta"),
        ),
        factor_mean_sigma=spec.extra.get(
            "factor_mean_sigma",
            _default_of(
                _factories.build_longitudinal_corr_factor_model, "factor_mean_sigma"
            ),
        ),
        trait_share_a=spec.extra.get(
            "trait_share_a",
            _default_of(
                _factories.build_longitudinal_corr_factor_model, "trait_share_a"
            ),
        ),
        trait_share_b=spec.extra.get(
            "trait_share_b",
            _default_of(
                _factories.build_longitudinal_corr_factor_model, "trait_share_b"
            ),
        ),
    )
    _attach_built(ctx, built)
    _render_model_graph(ctx)

    z_nodes = built.extras["z_nodes"]
    summary_vars = [
        "lambda_load",
        "sigma_indicator",
        "communality",
        "trait_share",
        # The headline: gate exactly the released per-wave off-diagonal correlations
        # (the full matrix's constant unit diagonal has undefined R-hat).
        "factor_corr_pairs",
    ]

    section_header("Prior predictive")
    prior_vars = [rv.name for rv in built.model.free_RVs]
    prior_vars += ["communality", "factor_corr_pairs", *z_nodes]
    _diag.run_prior_predictive(ctx, draws=1000, var_names=prior_vars)

    # Automatic single-target LOO is ambiguous with per-pattern observed nodes, so
    # sampling runs without it and the per-child stitch below computes LOO instead.
    _run_sampling_and_loo(ctx, compute_loo=False)

    section_header("LOO-PSIS (per-child stitch)")
    _lcf_stitch_loo(ctx, built)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=summary_vars)

    _run_ppc(ctx, var_names=z_nodes)

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=summary_vars)
    _diag.run_extended_diagnostics(ctx, causal_term=None)
    _diag.run_psense(ctx, var_names=["factor_corr_pairs", "trait_share"])
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=summary_vars)

    post = ctx.trace.posterior
    hdi = ctx.reporting.ci_prob
    lo_q = (1 - hdi) / 2

    # --- Loadings + communalities (the measurement layer) ---
    section_header("Loadings + communalities")
    dom_of = built.extras["domain_of"]
    load_rows = []
    for j, name in enumerate(str(s) for s in post["indicator"].values):
        lam_d = post["lambda_load"].isel(indicator=j).values.reshape(-1)
        com_d = post["communality"].isel(indicator=j).values.reshape(-1)
        corr_d = np.sqrt(com_d)
        load_rows.append(
            {
                "indicator": name,
                "domain": dom_of.get(name, "?"),
                "loading_mean": float(np.mean(lam_d)),
                "loading_lo": float(np.quantile(lam_d, lo_q)),
                "loading_hi": float(np.quantile(lam_d, 1 - lo_q)),
                "correlation_mean": float(np.mean(corr_d)),
                "communality_mean": float(np.mean(com_d)),
                "communality_lo": float(np.quantile(com_d, lo_q)),
                "communality_hi": float(np.quantile(com_d, 1 - lo_q)),
            }
        )
    load_df = pd.DataFrame(load_rows)
    load_df.to_csv(os.path.join(ctx.output_dir, "loadings_summary.csv"), index=False)
    ctx.tables["loadings_summary"] = load_df
    print_table(
        ranked_dataframe_table(
            load_df,
            title=f"Loadings + communalities - {int(hdi * 100)}% CI (equal-tailed)",
            columns=[
                "indicator", "domain", "loading_mean", "correlation_mean",
                "communality_mean", "communality_lo", "communality_hi",
            ],
            rank_column=False,
            precision=3,
        )
    )

    # --- Per-wave latent factor correlations (the headline) ---
    section_header("Per-wave latent factor correlations")
    corr_df = _report.longitudinal_factor_correlations(ctx.trace, ci_prob=hdi)
    corr_df.to_csv(
        os.path.join(ctx.output_dir, "factor_correlation_by_wave.csv"), index=False
    )
    ctx.tables["factor_correlation_by_wave"] = corr_df
    print_table(
        ranked_dataframe_table(
            corr_df,
            title=f"Per-wave latent factor correlations - {int(hdi * 100)}% ETI",
            columns=["wave", "domain_i", "domain_j", "mean", "lo", "hi", "prob_pos"],
            rank_column=False,
            precision=3,
        )
    )

    # --- Conditional (partial) latent slopes ---
    section_header("Conditional latent slopes")
    slope_df = _report.longitudinal_conditional_slopes(ctx.trace, ci_prob=hdi)
    slope_df.to_csv(
        os.path.join(ctx.output_dir, "latent_conditional_slopes.csv"), index=False
    )
    ctx.tables["latent_conditional_slopes"] = slope_df

    # --- Trait / state (across-wave) structure ---
    section_header("Trait / state structure")
    ts_rows = []
    for j, d in enumerate(str(s) for s in post["domain"].values):
        pi_d = post["trait_share"].isel(domain=j).values.reshape(-1)
        ts_rows.append(
            {
                "domain": d,
                "trait_share_mean": float(np.mean(pi_d)),
                "trait_share_lo": float(np.quantile(pi_d, lo_q)),
                "trait_share_hi": float(np.quantile(pi_d, 1 - lo_q)),
            }
        )
    ts_df = pd.DataFrame(ts_rows)
    ts_df.to_csv(os.path.join(ctx.output_dir, "trait_state_summary.csv"), index=False)
    ctx.tables["trait_state_summary"] = ts_df

    # --- Latent-versus-observed comparison (#312 triangulation anchor) --------
    section_header("Latent-versus-observed correlation comparison")
    obs_df = _lcf_observed_domain_corr(built)
    xcheck_df = _report.disattenuation_crosscheck(corr_df, obs_df)
    xcheck_df.to_csv(
        os.path.join(ctx.output_dir, "disattenuation_crosscheck.csv"), index=False
    )
    ctx.tables["disattenuation_crosscheck"] = xcheck_df
    n_latent_below = int((~xcheck_df["latent_ge_observed"]).sum())
    n_latent_at_or_above = len(xcheck_df) - n_latent_below
    rprint(
        "[cyan]Latent-versus-observed comparison: "
        f"{n_latent_below} wave/pair(s) are below and {n_latent_at_or_above} are at "
        "or above the mean observed indicator-pair magnitude. This is a descriptive "
        "gap direction between different estimands, not a pass/fail ordering.[/cyan]"
    )

    # --- Items-scale translation for the headline pairs ---
    section_header("Items-scale translation (selected pairs)")
    items_df = _lcf_items_scale(ctx, built)
    items_df.to_csv(
        os.path.join(ctx.output_dir, "latent_items_slopes.csv"), index=False
    )
    ctx.tables["latent_items_slopes"] = items_df

    # --- Directed comparison with matching concurrent associations (#312) ---
    section_header("Directed LCF-versus-concurrent comparison")
    concurrent_df = _lcf_concurrent_comparison(ctx, built)
    concurrent_df.to_csv(
        os.path.join(ctx.output_dir, "lcf_concurrent_comparison.csv"), index=False
    )
    ctx.tables["lcf_concurrent_comparison"] = concurrent_df
    n_ca_available = int(concurrent_df["ca_available"].sum())
    if n_ca_available < len(concurrent_df):
        rprint(
            "[yellow]Directed #312 comparison is incomplete: "
            f"{n_ca_available}/{len(concurrent_df)} matching concurrent rows were "
            "found under this output root/config. Fit CA002--004 at the same tier "
            "to populate the missing side.[/yellow]"
        )

    _report.write_run_metadata(
        ctx,
        extra={
            "loo_elpd": float(ctx.loo.elpd) if ctx.loo is not None else None,
            "domains": {k: list(v) for k, v in domains.items()},
            "invariance": built.extras["invariance"],
            "n_used_children": built.extras["n_used_children"],
            "loadings_summary": load_df.to_dict("records"),
            "factor_correlation_by_wave": corr_df.to_dict("records"),
            "trait_state_summary": ts_df.to_dict("records"),
            # Keep the legacy key for output compatibility; neither count is a gate.
            "disattenuation_reversals": n_latent_below,
            "latent_below_observed_count": n_latent_below,
            "latent_observed_comparison": "descriptive; no required ordering",
            "lcf_concurrent_comparison_rows": int(len(concurrent_df)),
            "lcf_concurrent_comparison_available": n_ca_available,
        },
    )

    return _finalize_report(ctx)
