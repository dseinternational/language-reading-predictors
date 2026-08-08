# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared execution stages for every statistical-model family.

Family pipelines decide how to prepare data, build a PyMC model, and summarise
their estimand.  This module owns the invariant execution order around those
decisions: attach the built model, sample, diagnose, draw posterior predictions,
record metadata, and finish the report.  The small hook boundary keeps plotting
and legacy artifact helpers replaceable while family modules are split out of the
historical monolithic pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from rich import print as rprint

from language_reading_predictors.models._reporting import section_header
from language_reading_predictors.statistical_models import (
    artifacts as _artifacts,
    diagnostics as _diag,
    release as _release,
    reporting as _report,
)
from language_reading_predictors.statistical_models.context import (
    StatisticalFitContext,
)


ContextHook = Callable[[StatisticalFitContext], Any]


@dataclass(frozen=True, slots=True)
class StageHooks:
    """Artifact hooks used by the shared stages during pipeline migration."""

    emit_priors: ContextHook
    save_ppc: Callable[..., Any]
    write_loo_influence: ContextHook
    print_loo_row: ContextHook
    copy_report_template: ContextHook
    publish_output: ContextHook
    print_footer: ContextHook


@dataclass(frozen=True, slots=True)
class PrimaryFitPlan:
    """The genuinely variable execution choices of a primary fit (#394 design 2).

    Everything else about the primary-fit sequence — its order, its section
    headers, which diagnostics run — is invariant and owned by
    :meth:`SharedFitStages.run_primary_fit`. A family declares only what varies:
    the curated diagnostic variables, the posterior-predictive nodes (the last
    one is the primary outcome node), its prior-predictive figure, which
    variables get power-scaling sensitivity, the term the extended diagnostics
    focus on, and the LOO / LOO-PIT / trace-persistence policy. Deliberately a
    flat value object rather than a base class with overridable methods: a
    reader should be able to see a family's whole execution profile in one
    declaration.
    """

    diagnostic_vars: tuple[str, ...]
    """Curated variables for the human-readable summary and the convergence gate
    (the gate itself widens to all free RVs via ``_gate_var_names``)."""

    ppc_var_names: tuple[str, ...] = ("y_post",)
    """Posterior-predictive nodes to draw; the last is the primary node."""

    prior_predictive_draws: int = 1000

    plot_prior_predictive: ContextHook | None = None
    """Family-specific prior-predictive figure (rate plot, count panel, …)."""

    psense_vars: tuple[str, ...] | None = None
    """Power-scaling sensitivity variables; ``None`` means ``diagnostic_vars``."""

    extended_term: str | None = None
    """Focus term for the extended diagnostics (rank / ESS-evolution plots);
    ``None`` skips the extended block (the gate still runs)."""

    include_loo_pit: bool = True
    compute_loo: bool = True
    save_trace: bool = True


@dataclass(frozen=True, slots=True)
class SharedFitStages:
    """Behaviour-preserving stages common to every family pipeline."""

    hooks: StageHooks

    def attach_built(self, ctx: StatisticalFitContext, built: Any) -> None:
        """Attach a freshly built model and emit its prior artifacts."""

        ctx.model = built.model
        ctx.prepared = built.prepared
        self.hooks.emit_priors(ctx)

    def sample_and_loo(
        self,
        ctx: StatisticalFitContext,
        *,
        compute_loo: bool = True,
    ) -> None:
        """Sample the posterior and optionally compute and report PSIS-LOO."""

        section_header("Sampling posterior (nutpie)")
        _diag.sample_posterior(ctx)

        if compute_loo:
            section_header("LOO-PSIS")
            _diag.compute_log_likelihood_and_loo(ctx)
            _report.write_loo_summary(ctx)
            self.hooks.write_loo_influence(ctx)
            self.hooks.print_loo_row(ctx)

    def posterior_predictive(
        self,
        ctx: StatisticalFitContext,
        *,
        var_names: list[str] | None = None,
    ) -> None:
        """Draw posterior predictions, then save coverage and figure artifacts."""

        section_header("Posterior predictive")
        names = list(var_names) if var_names else ["y_post"]
        _diag.sample_posterior_predictive(ctx, var_names=names)
        self.hooks.save_ppc(ctx, primary_node=names[-1])

    def run_primary_fit(
        self, ctx: StatisticalFitContext, plan: PrimaryFitPlan
    ) -> None:
        """Execute the invariant primary-fit sequence for a built, attached model.

        Prior prediction, posterior sampling with optional PSIS-LOO, the
        human-readable summary diagnostics, power-scaling sensitivity, posterior
        prediction, the all-free-variable convergence gate, the extended
        diagnostics, and trace persistence — in that order, once, for every
        family that adopts a :class:`PrimaryFitPlan`. Family scientific
        summaries and exceptional audits stay explicit in the family pipeline,
        before and after this call.
        """

        diag_vars = list(plan.diagnostic_vars)

        section_header("Prior predictive")
        _diag.run_prior_predictive(ctx, draws=plan.prior_predictive_draws)
        if plan.plot_prior_predictive is not None:
            plan.plot_prior_predictive(ctx)

        self.sample_and_loo(ctx, compute_loo=plan.compute_loo)

        section_header("Summary diagnostics")
        _diag.summary_diagnostics(ctx, var_names=diag_vars)
        psense_vars = (
            list(plan.psense_vars) if plan.psense_vars is not None else diag_vars
        )
        _diag.run_psense(ctx, var_names=psense_vars)

        self.posterior_predictive(ctx, var_names=list(plan.ppc_var_names))

        section_header("Extended diagnostics")
        _diag.write_diagnostics_summary(ctx, var_names=diag_vars)
        if plan.extended_term is not None:
            _diag.run_extended_diagnostics(
                ctx,
                causal_term=plan.extended_term,
                include_loo_pit=plan.include_loo_pit,
            )
        if plan.save_trace:
            _diag.save_trace(ctx)

    def write_metadata(
        self,
        ctx: StatisticalFitContext,
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Write the common run record plus optional family-specific metadata."""

        _report.write_run_metadata(ctx, extra=extra)

    def finalize_report(self, ctx: StatisticalFitContext) -> StatisticalFitContext:
        """Decide the release, generate key findings, copy the report, finish.

        The release decision comes first and explicitly (#394 design point 3):
        whether this fit may publish findings — and if not, at which stage and why
        — is settled and written to ``release_decision.json`` *before* the
        findings that follow from it are built. It was previously assembled
        inline inside ``generate_key_findings``, so finalisation never held it and
        nothing recorded it for the families the robustness gate does not cover.
        """

        section_header("Report")
        decision = _release.evaluate_publication(
            ctx.output_dir, artifacts=getattr(ctx, "artifacts", None)
        )
        _release.write_release_decision(ctx, decision)
        rprint(f"  Release decision: {decision.summary()}")
        findings = _report.generate_key_findings(ctx.output_dir, decision=decision)
        rprint(
            "  Key findings: "
            f"{findings['status']} ({len(findings['sentences'])} sentences)"
        )
        self.hooks.copy_report_template(ctx)
        # Manifest last-but-one: after the template copy so the report support
        # files are inventoried, before publication so it ships with the fit.
        manifest = _artifacts.write_manifest(ctx)
        rprint(
            f"  Artifact manifest: {manifest['n_written']} written, "
            f"{manifest['n_skipped']} skipped, {manifest['n_untracked']} untracked"
        )
        self.hooks.publish_output(ctx)
        self.hooks.print_footer(ctx)
        return ctx
