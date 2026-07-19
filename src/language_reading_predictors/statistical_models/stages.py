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
    diagnostics as _diag,
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
class SharedFitStages:
    """Behaviour-preserving stages common to every family pipeline."""

    hooks: StageHooks

    def attach_built(self, ctx: StatisticalFitContext, built: Any) -> None:
        """Attach a freshly built model and emit its prior artifacts."""

        ctx.model = built.model
        ctx.model_vars = built.variables
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

    def write_metadata(
        self,
        ctx: StatisticalFitContext,
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Write the common run record plus optional family-specific metadata."""

        _report.write_run_metadata(ctx, extra=extra)

    def finalize_report(self, ctx: StatisticalFitContext) -> StatisticalFitContext:
        """Generate key findings, copy the report, print the footer, and return."""

        section_header("Report")
        findings = _report.generate_key_findings(ctx.output_dir)
        rprint(
            "  Key findings: "
            f"{findings['status']} ({len(findings['sentences'])} sentences)"
        )
        self.hooks.copy_report_template(ctx)
        self.hooks.publish_output(ctx)
        self.hooks.print_footer(ctx)
        return ctx
