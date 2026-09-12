# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Write declared priors and their implications for the reported quantities."""

from __future__ import annotations

from language_reading_predictors.statistical_models import predictive_checks as _predictive


from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from rich import print as rprint

from language_reading_predictors.models._reporting import (
    print_table,
    ranked_dataframe_table,
)
from language_reading_predictors.statistical_models import priors as _priors
from language_reading_predictors.statistical_models.artifacts import save_table
from language_reading_predictors.statistical_models.context import (
    StatisticalFitContext,
)


def emit_priors(context: StatisticalFitContext) -> None:
    """Write the distributions and explanations recorded during construction."""
    model = context.model
    # Remove density panels only; retain prior predictive and comparison figures.
    for stale in _priors.prior_density_panel_files(context.output_dir):
        stale.unlink(missing_ok=True)
    _priors.save_model_prior_panels(model, context.output_dir)
    table = _priors.priors_table(model)
    save_table(context, "priors_table", table)


def growth_contrast_pushforward_rows(
    ctx: StatisticalFitContext,
    panel,
    measure: str,
    *,
    fitted_var: str = "fitted_mean_items_obs",
    prefix: str = "",
) -> list[dict[str, object]]:
    """Prior pushforward for a historical-growth family's group contrasts (#381).

    The reported estimands are the pairwise ``total_growth_X_minus_Y`` rows over
    the common window — how much more a comparison group grows than the
    Down-syndrome group across the whole observed span. Running
    :func:`historical.growth_summary` on the ``prior`` group answers how much of
    that difference the priors alone permit; the cohorts are not randomised, so
    the rows are descriptive contrasts rather than effects.
    """
    from language_reading_predictors.statistical_models import historical as _hist

    source = getattr(ctx, "prior_samples", None) or ctx.trace
    try:
        require_prior_evidence(
            source,
            terms=(fitted_var,),
            what="the total-growth contrast prior check",
        )
    except PriorEvidenceUnavailable as exc:
        return [
            _predictive.unavailable_pushforward(
                estimand=f"{prefix}total_growth",
                estimand_label="the between-group total-growth contrasts",
                role="descriptive",
                reason=str(exc),
            )
        ]
    # Narrow by design (#637 stage 1): past the availability check a failure in
    # ``growth_summary`` is a defect in the summary, not absent prior evidence.
    prior_growth = _hist.growth_summary(
        source, panel, measure, fitted_var=fitted_var, group="prior"
    )
    contrasts = prior_growth[
        prior_growth["quantity"].astype(str).str.startswith("total_growth")
    ]
    rows: list[dict[str, object]] = []
    for _, r in contrasts.iterrows():
        rows.append(
            _predictive.labelled_pushforward(
                {
                    # The growth summary is already in items; there is no separate
                    # linear-predictor contrast to report, since the quantity is a
                    # difference of fitted means rather than a coefficient.
                    "prior_logit_median": float("nan"),
                    "prior_logit_lo": float("nan"),
                    "prior_logit_hi": float("nan"),
                    "prior_items_median": float(r["q50"]),
                    "prior_items_lo50": float(r["q25"]),
                    "prior_items_hi50": float(r["q75"]),
                    "prior_items_lo": float(r["q_lo"]),
                    "prior_items_hi": float(r["q_hi"]),
                    "n_trials": 0,
                },
                estimand=f"{prefix}{r['quantity']}",
                estimand_label=str(r["label"]),
                role="descriptive",
            )
        )
    return rows


def write_indicator_prior_check(
    ctx: StatisticalFitContext, nodes: Sequence[str]
) -> None:
    """Write ``indicator_prior_check.csv`` for a measurement family (#381).

    The CFA families have no outcome-scale estimand to push a prior through —
    they report loadings, communalities and factor correlations — so #381 asks
    them for this instead, on the scale they do observe: the standardised
    indicator matrix. Without it these families were exempt from the coverage
    guarantee by construction rather than by argument.
    """
    try:
        df = _predictive.indicator_prior_check(
            ctx.trace, nodes=list(nodes), ci_prob=ctx.reporting.ci_prob
        )
    except Exception as exc:  # noqa: BLE001 - a report extra must not fail a fit
        rprint(f"[yellow]indicator prior check skipped: {exc}[/yellow]")
        return
    if df.empty:
        rprint("[yellow]indicator prior check: no indicator nodes found[/yellow]")
        return
    save_table(ctx, "indicator_prior_check", df)
    print_table(
        ranked_dataframe_table(
            df,
            title="Indicator-scale prior check (SD ratio 1 = prior matches the data)",
            columns=["indicator", "observed_sd", "prior_sd", "sd_ratio", "coverage_90", "verdict"],
            rank_column=False,
            precision=3,
        )
    )


class PriorEvidenceUnavailable(LookupError):
    """The prior evidence a pushforward needs is not present in this fit.

    The **only** condition that may produce an ``unavailable`` row (#637 stage 1).
    Four families caught every ``Exception`` while pushing their prior through the
    reported estimand, so a ``KeyError``, a wrong dimension or a schema defect
    became a valid ``prior_pushforward.csv`` recording "check unavailable" — and
    the release stage checks the file's presence, not its row status, so a
    programming error read on the rendered page exactly like an honest absence.

    Raise this for the two things that genuinely are absences: a fit with no
    persisted ``prior`` group, and a prior group that does not carry the term the
    check is about. Everything else is a defect and must fail the run.
    """


def require_prior_evidence(
    source: Any, *, terms: Sequence[str] = (), what: str = "this prior check"
) -> Any:
    """Return the ``prior`` group, or raise :class:`PriorEvidenceUnavailable`.

    ``source`` is a fit's ``prior_samples`` or its trace. ``terms`` names the
    variables the caller is about to read; naming them here is what lets the call
    site catch only the narrow exception and let a genuine ``KeyError`` deeper in
    the transform fail the fit.
    """

    group = getattr(source, "prior", None) if source is not None else None
    if group is None:
        raise PriorEvidenceUnavailable(
            f"{what} needs this fit's prior group, which was not sampled or persisted"
        )
    missing = [name for name in terms if name not in group]
    if missing:
        raise PriorEvidenceUnavailable(
            f"{what} needs {', '.join(missing)} in the prior group, "
            "which this fit does not carry"
        )
    return group


def at_mean_pushforward_rows(
    ctx: StatisticalFitContext,
    terms: Sequence[tuple[str, str]],
    *,
    n_trials: int,
    own_pre_logit_mean: float,
    role: str = "association",
    intercept: str = "alpha",
    own_slope: str = "gamma_own",
) -> list[dict[str, object]]:
    """Prior pushforward at the span families' *at-the-mean* operating point (#381).

    The between-child adjusted span fits (RLI ``adj-065``, Byrne ``adj-001``–``005``)
    publish their items-scale contrast as ``N · [expit(α + γ_own·m̄ + β) −
    expit(α + γ_own·m̄)]`` — two children at the sample-mean own baseline ``m̄``,
    every other standardised predictor (and any group-nuisance dummy) at zero, who
    differ by one SD on one predictor. The prior check has to push the prior through
    the same functional, or the prior and posterior rows of the report describe two
    different estimands: :func:`marginal_pushforward_rows`' ``"forward"``
    convention is the *row-averaged* contrast the stacked transition design uses,
    not this one (2026-08-22 adjusted-family review, finding 6). Each entry of
    ``terms`` is ``(term, label)``; a term the prior group does not carry yields an
    ``unavailable`` row naming it, exactly like the marginal helper.
    """
    from scipy.special import expit

    source = getattr(ctx, "prior_samples", None) or ctx.trace
    rows: list[dict[str, object]] = []
    try:
        prior = require_prior_evidence(
            source,
            terms=(intercept, own_slope),
            what="the at-the-mean prior check",
        )
    except PriorEvidenceUnavailable as exc:
        return [
            _predictive.unavailable_pushforward(
                estimand=term,
                estimand_label=label,
                role=role,
                reason=str(exc),
            )
            for term, label in terms
        ]

    def draws(name: str) -> np.ndarray:
        return prior[name].stack(sample=("chain", "draw")).values.ravel()

    base_eta = draws(intercept) + draws(own_slope) * float(own_pre_logit_mean)
    base_items = float(n_trials) * expit(base_eta)
    for term, label in terms:
        try:
            require_prior_evidence(
                source, terms=(term,), what=f"the prior check on {term}"
            )
            beta = draws(term)
            items = float(n_trials) * expit(base_eta + beta) - base_items
            values = _predictive.pushforward_values(
                beta, items, n_trials=n_trials, ci_prob=ctx.reporting.ci_prob
            )
        except PriorEvidenceUnavailable as exc:
            rows.append(
                _predictive.unavailable_pushforward(
                    estimand=term, estimand_label=label, role=role, reason=str(exc)
                )
            )
        else:
            rows.append(
                _predictive.labelled_pushforward(
                    values, estimand=term, estimand_label=label, role=role
                )
            )
    return rows


def write_prior_pushforward(
    ctx: StatisticalFitContext, rows: Sequence[Mapping[str, object]]
) -> None:
    """Write ``prior_pushforward.csv`` — including when the check is unavailable (#381).

    The meta-finding behind #381 is that a *missing* artefact reads as a clean
    one: a family that never emitted the estimand-scale prior check looked, in the
    rendered report, exactly like one whose prior was checked and found harmless.
    So every family that reaches this point writes the file, and a row whose
    ``status`` is ``unavailable`` carries the reason instead of being dropped.
    """
    df = pd.DataFrame(list(rows))
    save_table(ctx, "prior_pushforward", df)


def horseshoe_pushforward_rows(
    ctx: StatisticalFitContext, predictors: Sequence[str], outcome: str
) -> list[dict[str, object]]:
    """Per-predictor prior pushforward for a horseshoe ranking fit (#381).

    The horseshoe deliverable is a ranking by ``P(|beta| > delta)``, which the
    prior-analysis review flagged as a direct function of ``tau0`` / ``slab_scale``
    — so "no signal" and "shrunk to nothing by the prior" are the two readings that
    have to be told apart. The ``prior_logit_*`` columns are the shrinkage prior's
    own implied spread for a single coefficient, against which the ranking's
    ``delta`` can be judged; the items columns put the same ``+1 SD`` shift on the
    outcome scale. Every predictor shares the global-local prior, so the rows
    differ only through each coefficient's own local scale draws.
    """
    n_trials = pushforward_n_trials(ctx, outcome)
    label = pushforward_outcome_label(ctx, outcome)
    return marginal_pushforward_rows(
        ctx,
        [
            (
                "beta",
                f"the shrunk association of +1 SD {p} with {label}",
                {"predictor": p},
            )
            for p in predictors
        ],
        n_trials=n_trials,
        convention="forward",
    )


def pushforward_outcome_label(ctx: StatisticalFitContext, outcome: str) -> str:
    """Reader-facing name for the pushforward's outcome, falling back to the symbol.

    The rows are read by a science reader, not by whoever picked the symbols, so
    ``W`` and ``basread`` should render as their measure labels. The study's own
    measure table is the source: RLI symbols resolve through ``measures.MEASURES``
    and the Byrne-cohort ones through their dataset's table, so neither study's
    labels are hard-coded here.
    """
    from language_reading_predictors.statistical_models import datasets as _datasets
    from language_reading_predictors.statistical_models.measures import MEASURES

    if outcome in MEASURES:
        return str(MEASURES[outcome].label)
    try:
        _, measures = _datasets.resolve_dataset(ctx.spec.study_id)
        return str(measures[outcome].label)
    except Exception:  # noqa: BLE001 - a label is cosmetic; the symbol still names it
        return outcome


def pushforward_n_trials(ctx: StatisticalFitContext, outcome: str) -> int:
    """The pushforward's item denominator, or 1 when the fit carries none (#381).

    Most families know their outcome's item ceiling and the check is most
    readable in items. Where the fit does not carry one, return 1 rather than
    inventing a denominator: the marginal is then a probability difference, and
    :func:`reporting.pushforward_scale_for` labels it in percentage points off
    that same 1 — one rule, so a denominator and its scale cannot disagree.
    """
    trials = getattr(ctx.prepared, "n_trials", None) or {}
    try:
        return int(trials[outcome])
    except (KeyError, TypeError, ValueError):
        return 1


def marginal_pushforward_rows(
    ctx: StatisticalFitContext,
    terms: Sequence[tuple],
    *,
    n_trials: int,
    role: str = "association",
    convention: str = "forward",
    eta_name: str = "eta",
    row_mask: np.ndarray | None = None,
    scale: str | None = None,
    score_mean_link: str = "logit",
) -> list[dict[str, object]]:
    """Build one labelled pushforward row per term (#381).

    Each entry is ``(term, label)``, or ``(term, label, index)`` to select one
    element of a vector-valued coefficient — ``("beta", "...", {"predictor":
    "age"})`` for the horseshoe families, whose ``beta`` carries a labelled
    predictor dimension.

    ``convention`` is passed straight through to
    :func:`reporting.marginal_prior_pushforward` and must match the convention the
    family's own posterior marginal uses. A term the prior group does not carry
    yields an ``unavailable`` row naming it, rather than a silently shorter table.
    """
    # ``prior_samples`` carries the prior group from ``run_prior_predictive``; the
    # trace only carries it after ``save_trace`` grafts it on, so prefer the former
    # and the call site stays free to sit either side of that step.
    source = getattr(ctx, "prior_samples", None) or ctx.trace
    rows: list[dict[str, object]] = []
    for entry in terms:
        term, label = entry[0], entry[1]
        index = entry[2] if len(entry) > 2 else None
        named = term if not index else f"{term}[{'/'.join(map(str, index.values()))}]"
        try:
            require_prior_evidence(
                source,
                terms=(term, eta_name),
                what=f"the prior check on {named}",
            )
            values = _predictive.marginal_prior_pushforward(
                source,
                term=term,
                n_trials=n_trials,
                eta_name=eta_name,
                ci_prob=ctx.reporting.ci_prob,
                convention=convention,
                row_mask=row_mask,
                term_index=index,
                score_mean_link=score_mean_link,
            )
        except PriorEvidenceUnavailable as exc:
            rows.append(
                _predictive.unavailable_pushforward(
                    estimand=named,
                    estimand_label=label,
                    role=role,
                    reason=str(exc),
                    scale=scale,
                )
            )
        else:
            rows.append(
                _predictive.labelled_pushforward(
                    values,
                    estimand=named,
                    estimand_label=label,
                    role=role,
                    scale=scale,
                )
            )
    return rows
