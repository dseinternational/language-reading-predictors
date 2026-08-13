# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Binds the shared fit stages to this package's artefact implementations.

:mod:`stages` defines the invariant lifecycle without knowing how any artefact is
produced; the concrete producers live in :mod:`prior_artifacts`,
:mod:`ppc_artifacts`, :mod:`publication` and :mod:`diagnostics`. This module is
where the two meet: :func:`shared_stages` supplies the hook bundle, and the thin
per-phase wrappers below are what every family entry point calls. It also holds
:func:`require_spec`, the runtime spec validation every entry point opens with.

Keeping the binding here (rather than in ``stages``) preserves the inversion:
the lifecycle stays testable without importing any artefact writer.
"""

from __future__ import annotations

from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
)
from language_reading_predictors.statistical_models.diagnostics import (
    write_loo_influence,
)
from language_reading_predictors.statistical_models.ppc_artifacts import save_ppc
from language_reading_predictors.statistical_models.prior_artifacts import emit_priors
from language_reading_predictors.statistical_models.publication import (
    copy_report_template,
    print_footer,
    print_loo_row,
)
from language_reading_predictors.statistical_models.stages import (
    SharedFitStages,
    StageHooks,
)


def require_spec(
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


# ---------------------------------------------------------------------------
# Shared pipeline phases (#82)
#
# Every fit_* pipeline runs the invariant primary lifecycle through
# ``shared_stages().run_primary_fit``. Family preparation, model construction,
# scientific summaries, metadata and report finalisation remain inline. The old
# one-phase sampling/PPC wrappers were removed once all primary paths adopted the
# full lifecycle, so a family cannot quietly reconstruct a partial sequence.
# ---------------------------------------------------------------------------


def shared_stages() -> SharedFitStages:
    """Bind shared execution stages to the current artifact implementations."""

    return SharedFitStages(
        StageHooks(
            emit_priors=emit_priors,
            save_ppc=save_ppc,
            write_loo_influence=write_loo_influence,
            print_loo_row=print_loo_row,
            copy_report_template=copy_report_template,
            publish_output=lambda ctx: ctx.publish_output_dir(),
            print_footer=print_footer,
        )
    )


def attach_built(ctx: StatisticalFitContext, built) -> None:
    """Attach a built model and its prepared data to the run context."""

    shared_stages().attach_built(ctx, built)


def write_run_metadata(
    ctx: StatisticalFitContext,
    *,
    extra: dict | None = None,
) -> None:
    """Write ``config.json`` — the resolved plan, sampling settings and family extras."""

    shared_stages().write_metadata(ctx, extra=extra)


def finalize_report(ctx: StatisticalFitContext) -> StatisticalFitContext:
    """Generate key findings, copy the report template, write the manifest, publish."""

    return shared_stages().finalize_report(ctx)
