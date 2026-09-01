# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""New-child prediction target and its matching validation for the joint families (#626).

## The gap this closes

The joint families aggregate their pointwise likelihood to the **child** unit and call
the result leave-one-child-out PSIS-LOO. That aggregation is necessary — leaving out
one child-outcome *cell* would predict one measure from the same child's other measure
— but on its own it is not sufficient. Where the model carries a **child-level latent**
(the LKJ residual block in ``joint``, the bivariate child intercept in
``joint_mechanism``, the stable and within-child deviations in ``historical_joint``),
the importance weights reweight a posterior in which that child's *own* latent is still
informed by its own data. The quantity is then "predict this child's scores given a
posterior that has already seen them through their random effect" — a conditional
quantity, not the generalisation to a child the model has never met.

``historical_joint`` refused to publish any LOO at all for this reason, recording
``loo_unit="undeclared_prediction_target_not_implemented"``: the obstacle was never the
multiple likelihood nodes (they share an observation coordinate and sum per child-wave
row) but the absence of a *declared and implemented* prediction target.

## The declared target

**A new child in a replicate cohort**: same size, same group composition, same observed
covariates, a child the model has not seen. That is the target these families' estimands
are about — children are the sampling unit; the waves are a fixed balanced design rather
than a sample of occasions — and it is the target the child-aggregated LOO unit already
claims. The alternative (a new occasion for a known child) is a different question and
would need a different holdout; :data:`PREDICTION_TARGETS` names both so a family that
wants the other one has to say so.

Declaring the cohort as a *replicate of the realised sample* is what dissolves
``historical_joint``'s remaining obstacle. Its subject offsets are group-centred and its
within-child deviations double-centred, both over the realised sample, so "draw one new
child's latent from its population distribution" is not well defined in isolation. The
redraw here is performed **by the model itself** at the realised sample size, so the
sample-dependent centring is applied to the fresh draw exactly as it is applied to the
fitted one. No closed form for a marginal population distribution is needed, and none is
assumed.

## How it is computed

For each posterior draw :math:`\\theta^s` the child latents are re-drawn from their
population distribution given :math:`\\theta^s`, and the child's integrated (marginal)
likelihood is the Monte-Carlo average over those re-draws::

    log p(y_i | theta^s) = log ( 1/M sum_m p(y_i | theta^s, u^(m)) )

PSIS is then run on that **integrated** pointwise term, so both the importance weights
and the Pareto-:math:`k` diagnostics belong to the declared new-child target rather than
to a conditional one. The matching calibration diagnostic re-uses those same weights:
the PIT is computed from predictive draws generated under the same fresh latents, so its
holdout unit *is* the declared unit — which is what the conditional leave-one-cell-out
PIT plots the joint reports also carry could never be (see
``diagnostics.JOINT_LOO_PIT_UNIT_LABEL``).

The re-draw is forced by **removing** the latent from the posterior handed to
``pm.sample_posterior_predictive``. Naming a variable in ``var_names`` is not enough:
PyMC treats a variable it finds in the trace as given, so a request that leaves it in
place returns the fitted values unchanged — silently answering the conditional question
this module exists to stop answering.

## What is refused

A family must declare which of its free random variables are child-level.
:func:`verify_child_latents` then checks the declaration against the built model and
**refuses** when a free random variable is indexed by a declared child dimension and was
not declared: that omission is exactly the defect this module addresses, so it fails the
run rather than quietly producing a conditional number under a new-child label. Where
PSIS on the integrated term is unreliable (any Pareto-:math:`k` above ``good_k``) the
ELPD is recorded but flagged ``reliable=False`` and the report withholds it — #626's
"do not publish naive PSIS where Pareto-k is unacceptable", applied to the integrated
term as well as the conditional one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from rich import print as rprint

from language_reading_predictors.statistical_models.artifacts import (
    guard_optional,
    save_table,
)
from language_reading_predictors.statistical_models.context import (
    StatisticalFitContext,
)

__all__ = [
    "NewChildEvidenceUnavailable",
    "NewChildPlan",
    "NewChildValidation",
    "PREDICTION_TARGETS",
    "PREDICTION_TARGET_NEW_CHILD",
    "child_row_maps",
    "run_new_child_validation",
    "verify_child_latents",
    "write_new_child_validation",
]

#: The declared target: a child the model has not seen, in a replicate cohort of the
#: realised size and group composition, at the observed covariate values.
PREDICTION_TARGET_NEW_CHILD = "new_child"

#: Every prediction target a joint family may declare. ``new_occasion_known_child`` is
#: named but not implemented: it is a different holdout (one wave of an otherwise
#: observed child) and would need its own latent treatment, so a family declaring it
#: gets an explicit refusal rather than this module's new-child machinery under the
#: wrong label.
PREDICTION_TARGETS = (PREDICTION_TARGET_NEW_CHILD, "new_occasion_known_child")

#: ArviZ's ``good_k`` is sample-size dependent; this is the floor used when a LOO
#: object does not expose one.
DEFAULT_GOOD_K = 0.7

#: Draw budget that never thins, used where the integral needs only one pass.
_NO_THINNING = 1 << 62


class NewChildEvidenceUnavailable(LookupError):
    """Expected absence of the inputs new-child validation needs.

    Narrow on purpose, in the shape ``prior_artifacts.PriorEvidenceUnavailable``
    established (#637 stage 1): a missing child map or an absent posterior group is a
    fit that legitimately has nothing to validate, while a ``KeyError`` from a renamed
    coordinate or a shape mismatch is a defect that must fail the run rather than
    become a plausible-looking "unavailable" row.
    """


@dataclass(frozen=True, slots=True)
class NewChildPlan:
    """A family's declaration of its out-of-sample prediction target.

    ``child_dims`` are the model dimensions that index children, or rows nested within
    children. ``latent_vars`` are the free random variables re-drawn for a new child.
    Both are declared rather than inferred: inferring them from dimension sizes alone
    would silently accept a model whose child effect happens to share a length with
    something else, and the whole point of this module is that a silently-conditional
    answer is worse than no answer.

    ``observed_nodes`` may be left empty, in which case every observed random variable
    of the built model is used — the multi-node case (one likelihood node per measure)
    is the ordinary one for ``historical_joint``, and #626 records why several nodes
    were never the obstacle: they share an observation coordinate, so their
    contributions sum per child.
    """

    prediction_target: str = PREDICTION_TARGET_NEW_CHILD
    child_dims: tuple[str, ...] = ()
    latent_vars: tuple[str, ...] = ()
    observed_nodes: tuple[str, ...] = ()
    n_latent_draws: int = 64
    max_posterior_draws: int = 8000
    random_seed: int = 20260626

    def __post_init__(self) -> None:
        if self.prediction_target not in PREDICTION_TARGETS:
            raise ValueError(
                "prediction_target must be one of "
                f"{', '.join(PREDICTION_TARGETS)}; got {self.prediction_target!r}"
            )
        if self.prediction_target != PREDICTION_TARGET_NEW_CHILD:
            raise ValueError(
                f"prediction target {self.prediction_target!r} is declared but not "
                "implemented; only 'new_child' has a matching validation (#626)"
            )
        if not self.child_dims:
            raise ValueError("child_dims must name at least one child-indexed dimension")
        if self.n_latent_draws < 2:
            raise ValueError("n_latent_draws must be at least 2 to integrate anything")
        if self.max_posterior_draws < 100:
            raise ValueError("max_posterior_draws must be at least 100")

    def as_dict(self) -> dict[str, Any]:
        """The declaration as ``config.json`` records it."""
        return {
            "prediction_target": self.prediction_target,
            "child_dims": list(self.child_dims),
            "latent_vars": list(self.latent_vars),
            "observed_nodes": list(self.observed_nodes),
            "n_latent_draws": self.n_latent_draws,
            "max_posterior_draws": self.max_posterior_draws,
        }


@dataclass(frozen=True, slots=True)
class NewChildValidation:
    """Everything a fit publishes about its new-child predictive performance."""

    plan: NewChildPlan
    n_children: int
    posterior_draws_used: int
    elpd: float
    elpd_se: float
    p_loo: float
    pointwise_elpd: np.ndarray
    pareto_k: np.ndarray
    good_k: float
    latents_redrawn: tuple[str, ...]
    observed_nodes: tuple[str, ...]
    latent_mc_error: float = 0.0
    """Largest per-child half-split disagreement in the integrated log term."""
    pit: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def n_unreliable(self) -> int:
        return int((self.pareto_k > self.good_k).sum())

    @property
    def integration_reliable(self) -> bool:
        """Whether the latent integral is precise enough to carry the ELPD.

        Two things can be too rough here, and only one of them is Pareto-k. The
        integral itself is a Monte-Carlo average over a finite number of population
        draws, and its error scales with how many latent dimensions the child carries:
        the two-dimensional joint residual settles by a few dozen draws, while a
        multi-measure panel's stable-plus-within departures were still moving the ELPD
        by hundreds of nats between 64 and 256 draws (#626 probe, 2026-09-01). The
        test is relative rather than absolute — integration noise summed over children
        must stay inside the ELPD's own standard error, because an estimate whose
        numerical error rivals its sampling error is not measuring the model.
        """
        if not math.isfinite(self.latent_mc_error):
            return False
        if not math.isfinite(self.elpd_se) or self.elpd_se <= 0.0:
            return self.latent_mc_error == 0.0
        return self.n_children * self.latent_mc_error <= self.elpd_se

    @property
    def reliable(self) -> bool:
        """Whether this ELPD may be published.

        #626: naive PSIS is not published where Pareto-k is unacceptable. The same
        rule applies to the integrated term — re-drawing the latent changes the
        importance ratios, so a conditional fit's clean k values are no warrant — and
        it applies to the integral's own precision alongside it.
        """
        return self.n_unreliable == 0 and self.integration_reliable

    @property
    def max_pareto_k(self) -> float:
        return float(self.pareto_k.max()) if self.pareto_k.size else float("nan")

    def summary_row(self) -> dict[str, Any]:
        return {
            "prediction_target": self.plan.prediction_target,
            "holdout_unit": "child",
            "n_children": self.n_children,
            "elpd": self.elpd,
            "elpd_se": self.elpd_se,
            "p_loo": self.p_loo,
            "max_pareto_k": self.max_pareto_k,
            "good_k_threshold": self.good_k,
            "n_unreliable": self.n_unreliable,
            "reliable": self.reliable,
            "integration_reliable": self.integration_reliable,
            "latents_redrawn": " ".join(self.latents_redrawn) or "(none)",
            "observed_nodes": " ".join(self.observed_nodes),
            "latent_mc_half_split_error": self.latent_mc_error,
            "n_latent_draws": self.plan.n_latent_draws,
            "posterior_draws_used": self.posterior_draws_used,
        }


def _dataset(group: Any) -> xr.Dataset | None:
    """The variables behind an inference-data group, whatever wrapper it arrives in."""
    if group is None:
        return None
    inner = getattr(group, "dataset", group)
    return inner if hasattr(inner, "data_vars") else None


def _prepared_child_index(ctx: Any, n_rows: int) -> np.ndarray | None:
    """A row-to-child index derived from the fit's prepared data, or ``None``.

    The joint families mark their child map in ``constant_data``; the historical
    joint-growth family does not persist one, because its rows are the tidy panel
    rows and the mapping lives on the panel. Reading it off the panel keeps this
    module from having to change a fitted model's graph to add a map that the data
    already carry — a ``pm.Data`` addition would alter every stored design identity
    for a diagnostic.
    """
    prepared = getattr(ctx, "prepared", None)
    if prepared is None:
        return None
    child_idx = getattr(prepared, "child_idx", None)
    if child_idx is not None:
        candidate = np.asarray(child_idx, dtype=int).ravel()
        if candidate.size == n_rows:
            return candidate
    long = getattr(prepared, "long", None)
    subject_ids = getattr(prepared, "subject_ids", None)
    dataset = getattr(prepared, "dataset", None)
    subject_col = getattr(dataset, "subject_col", None)
    if (
        long is not None
        and subject_ids is not None
        and subject_col
        and subject_col in long
        and len(long) == n_rows
    ):
        position = {value: index for index, value in enumerate(list(subject_ids))}
        try:
            return np.asarray(
                [position[value] for value in long[subject_col]], dtype=int
            )
        except KeyError:  # pragma: no cover - a panel row outside its own id list
            return None
    return None


def child_row_maps(
    ctx: Any, observed_nodes: Sequence[str]
) -> tuple[dict[str, np.ndarray], int]:
    """``({node: row -> child index}, n_children)`` for the fit's likelihood nodes.

    Recognises the same two persisted maps ``diagnostics._joint_log_likelihood_by_child``
    does — the explicit ``loo_child_idx`` and the joint families' ``y_post_cell_row`` —
    so the new-child unit is *the same unit* the stored child-aggregated PSIS-LOO uses
    rather than a second, parallel definition of "child", and falls back to the panel
    for a family that persists neither.
    """
    observed = _dataset(getattr(ctx.trace, "observed_data", None))
    if observed is None:
        raise NewChildEvidenceUnavailable("trace carries no observed_data group")
    constant = _dataset(getattr(ctx.trace, "constant_data", None))
    candidates: list[np.ndarray] = []
    if constant is not None:
        for name in ("loo_child_idx", "y_post_cell_row"):
            if name in constant:
                candidates.append(np.asarray(constant[name].values, dtype=int).ravel())

    maps: dict[str, np.ndarray] = {}
    for node in observed_nodes:
        if node not in observed:
            raise NewChildEvidenceUnavailable(f"no observed data for node {node!r}")
        n_rows = int(np.asarray(observed[node].values).size)
        chosen = next((c for c in candidates if c.size == n_rows), None)
        if chosen is None:
            chosen = _prepared_child_index(ctx, n_rows)
        if chosen is None:
            raise NewChildEvidenceUnavailable(
                f"no child map aligns with the {n_rows} rows of node {node!r}"
            )
        if chosen.min() < 0:
            raise ValueError(f"child map for {node!r} contains a negative index")
        maps[node] = chosen
    if not maps:
        raise NewChildEvidenceUnavailable("the model has no observed likelihood node")
    n_children = max(int(rows.max()) + 1 for rows in maps.values())
    return maps, n_children


def verify_child_latents(model: pm.Model, plan: NewChildPlan) -> tuple[str, ...]:
    """Check the declaration against the built model and return what will be re-drawn.

    Fails closed in both directions. A declared variable that is not a free random
    variable is a stale declaration; an *undeclared* free random variable carrying one
    of the declared child dimensions is the defect this module exists to prevent —
    leaving it in place would keep that child's own data in the posterior and quietly
    turn the answer back into a conditional one.
    """
    free = {rv.name for rv in model.free_RVs}
    unknown = [name for name in plan.latent_vars if name not in free]
    if unknown:
        raise ValueError(
            f"declared child latent(s) {', '.join(sorted(unknown))} are not free "
            "random variables of this model"
        )
    declared = set(plan.latent_vars)
    child_dims = set(plan.child_dims)
    undeclared = sorted(
        rv.name
        for rv in model.free_RVs
        if rv.name not in declared
        and child_dims & set(model.named_vars_to_dims.get(rv.name) or ())
    )
    if undeclared:
        raise ValueError(
            f"free random variable(s) {', '.join(undeclared)} are indexed by a "
            f"declared child dimension ({', '.join(sorted(child_dims))}) but are not "
            "declared child latents; a new-child prediction that leaves them at their "
            "fitted values is a conditional prediction wearing a new-child label (#626)"
        )
    return tuple(plan.latent_vars)


def _thin_posterior(posterior: xr.Dataset, max_draws: int) -> xr.Dataset:
    """Thin the draw axis so ``chain x draw`` fits the diagnostic's budget.

    Thinning a converged chain leaves an unbiased posterior sample; what it costs is
    Monte-Carlo precision, and the whole computation is repeated ``n_latent_draws``
    times, so the budget is real. The realised count is published beside the ELPD, and
    it is load-bearing for the Pareto-k values rather than only for the ELPD: the
    shape estimate is a tail quantity, so a heavily thinned run reports a noisier k
    for the same fit.
    """
    total = int(posterior.sizes.get("chain", 1)) * int(posterior.sizes.get("draw", 1))
    if total <= max_draws:
        return posterior
    step = int(math.ceil(total / max_draws))
    return posterior.isel(draw=slice(None, None, step))


def _posterior_tree(posterior: xr.Dataset) -> xr.DataTree:
    tree = xr.DataTree()
    tree["posterior"] = xr.DataTree(posterior)
    return tree


def _aggregate_to_child(
    values: np.ndarray, rows: np.ndarray, n_children: int
) -> np.ndarray:
    """Sum a ``(chain, draw, row)`` array within child, giving ``(chain, draw, child)``."""
    out = np.zeros((*values.shape[:-1], n_children), dtype=float)
    for child in range(n_children):
        out[..., child] = values[..., rows == child].sum(axis=-1)
    return out


def _half_split_error(
    halves: Sequence[np.ndarray | None],
    counts: Sequence[int],
    latents: Sequence[str],
) -> float:
    """Monte-Carlo stability of the latent integral, as a half-split discrepancy.

    ``log (1/M) sum_m p(y_i | theta, u^(m))`` is estimated from a finite number of
    population draws, and how finite is enough depends on how many latent dimensions
    the child carries — two for the joint LKJ residual, far more for a correlated
    multi-measure panel. Splitting the re-draws into two independent halves and taking
    the largest per-child disagreement in the draw-averaged term gives the reader a
    number for that rather than an assurance. Zero when there is no latent to
    integrate, because the term is then exact.
    """
    if not latents:
        return 0.0
    first, second = halves[0], halves[1]
    if first is None or second is None or min(counts) < 1:
        return float("nan")
    # Each half is normalised by the number of re-draws it actually accumulated, not by
    # a formula derived from the total: the even and odd halves differ in size whenever
    # ``n_latent_draws`` is odd, and deriving it got the two the wrong way round.
    left = (first - math.log(counts[0])).mean(axis=(0, 1))
    right = (second - math.log(counts[1])).mean(axis=(0, 1))
    return float(np.max(np.abs(left - right)))


def _observed_node_names(model: pm.Model, plan: NewChildPlan) -> tuple[str, ...]:
    """The likelihood nodes to score, declared or read off the built model."""
    declared = tuple(plan.observed_nodes)
    if declared:
        return declared
    return tuple(rv.name for rv in model.observed_RVs)


def run_new_child_validation(
    ctx: StatisticalFitContext, plan: NewChildPlan
) -> NewChildValidation:
    """Integrate out the child latents, PSIS the result, and PIT the same draws.

    Raises :class:`NewChildEvidenceUnavailable` when the fit legitimately has nothing
    to validate (no child map, no posterior), and ordinary errors otherwise — #637
    stage 1's rule that a programming error must never be laundered into an
    available-looking "unavailable" row.
    """
    from arviz_stats import loo as _loo

    from language_reading_predictors.statistical_models.diagnostics import (
        log_density_model,
    )

    if ctx.model is None or ctx.trace is None:
        raise NewChildEvidenceUnavailable("no built model or trace on the context")
    posterior = _dataset(getattr(ctx.trace, "posterior", None))
    if posterior is None:
        raise NewChildEvidenceUnavailable("trace carries no posterior group")

    model = ctx.model
    nodes = _observed_node_names(model, plan)
    if not nodes:
        raise NewChildEvidenceUnavailable("the model has no observed likelihood node")
    maps, n_children = child_row_maps(ctx, nodes)
    latents = verify_child_latents(model, plan)

    # Thinning buys nothing when there is no latent to integrate: the loop below runs
    # once, so the full posterior costs a single log-likelihood pass. It also costs
    # something real. The Pareto shape estimate is a tail quantity, so a thinned run
    # reports a *different* k for what is, in that case, literally the same estimator
    # as the fit's stored conditional PSIS-LOO — 0.769 against 0.701 for
    # ``lrp-rli-itt-012``, two numbers for one quantity in one report. The identity
    # this module claims for a latent-free design has to hold in the artefacts.
    thinned = _thin_posterior(
        posterior,
        plan.max_posterior_draws if plan.latent_vars else _NO_THINNING,
    )
    draws_used = int(thinned.sizes["chain"]) * int(thinned.sizes["draw"])
    without_latents = thinned.drop_vars([n for n in latents if n in thinned])
    density_model = log_density_model(model)

    running: np.ndarray | None = None
    halves: list[np.ndarray | None] = [None, None]
    half_counts = [0, 0]
    predictive: dict[str, list[np.ndarray]] = {node: [] for node in nodes}
    # With no child latent the integral is over an empty set: one pass gives both the
    # exact log-likelihood and the predictive draws, and 63 more would repeat them.
    passes = plan.n_latent_draws if plan.latent_vars else 1
    for index in range(passes):
        with model:
            drawn = pm.sample_posterior_predictive(
                _posterior_tree(without_latents),
                var_names=[*latents, *nodes],
                extend_inferencedata=False,
                random_seed=plan.random_seed + index,
                progressbar=False,
            )
        redrawn = _dataset(getattr(drawn, "posterior_predictive", None))
        if redrawn is None:
            raise ValueError("posterior predictive re-draw returned no group")
        missing = [name for name in latents if name not in redrawn]
        if missing:
            # PyMC returns a variable it finds in the trace unchanged. Dropping the
            # latent from the posterior is what forces the re-draw; if one survived
            # that, every number below would be conditional on the fitted latent
            # under a new-child label.
            raise ValueError(
                f"latent(s) {', '.join(missing)} were not re-drawn; the new-child "
                "predictive would be conditional on their fitted values"
            )
        # Only the FIRST pass's predictive draws are kept. Each posterior draw
        # contributes one predictive draw to the PIT, and that draw already carries a
        # latent sampled fresh for it, so it is a draw from the new-child predictive
        # whether one batch was generated or sixty-four. Keeping every batch would
        # change nothing statistically and cost chain x draw x rows x n_latent_draws
        # floats — gigabytes for a multi-measure panel at reporting scale. The
        # re-draws still matter for the log-likelihood integral, which is what they
        # are generated for.
        if index == 0:
            for node in nodes:
                predictive[node].append(
                    np.asarray(redrawn[node].transpose("chain", "draw", ...).values)
                )
        if latents or running is None:
            # With no child latent the likelihood is a deterministic function of the
            # posterior draw, so every pass would recompute the same numbers; the
            # integral over an empty set is the conditional term itself, and that
            # identity is the finding for those fits rather than a shortcut around
            # one. The predictive re-draws above still vary, because the PIT needs
            # them to.
            with_fresh = thinned.copy()
            for name in latents:
                with_fresh[name] = redrawn[name]
            scored = pm.compute_log_likelihood(
                _posterior_tree(with_fresh),
                model=density_model,
                progressbar=False,
                extend_inferencedata=True,
            )
            log_likelihood = _dataset(getattr(scored, "log_likelihood", None))
            if log_likelihood is None:
                raise ValueError("no log likelihood was computed under the re-drawn latents")
            child_ll: np.ndarray = np.zeros(
                (
                    int(thinned.sizes["chain"]),
                    int(thinned.sizes["draw"]),
                    n_children,
                ),
                dtype=float,
            )
            for node in nodes:
                if node not in log_likelihood:
                    raise ValueError(
                        f"log likelihood for {node!r} was not computed under the "
                        "re-drawn latents"
                    )
                cell_ll = log_likelihood[node].transpose("chain", "draw", ...)
                child_ll += _aggregate_to_child(
                    np.asarray(cell_ll.values, dtype=float), maps[node], n_children
                )
            running = child_ll if running is None else np.logaddexp(running, child_ll)
            half = index % 2
            halves[half] = (
                child_ll if halves[half] is None else np.logaddexp(halves[half], child_ll)
            )
            half_counts[half] += 1

    if running is None:  # pragma: no cover - n_latent_draws >= 2 is validated on the plan
        raise ValueError("no latent re-draws were scored")
    integrated = running - math.log(plan.n_latent_draws if latents else 1)
    latent_mc_error = _half_split_error(halves, half_counts, latents)

    integrated_da = xr.DataArray(
        integrated,
        dims=("chain", "draw", "loo_child"),
        coords={
            "chain": thinned.coords["chain"],
            "draw": thinned.coords["draw"],
            "loo_child": np.arange(n_children),
        },
        name="y_child",
    )
    loo_tree = xr.DataTree()
    loo_tree["posterior"] = xr.DataTree(thinned)
    loo_tree["log_likelihood"] = xr.DataTree(xr.Dataset({"y_child": integrated_da}))
    elpd = _loo(loo_tree, pointwise=True, var_name="y_child")
    pareto_k = np.asarray(getattr(elpd, "pareto_k")).ravel().astype(float)
    pointwise = np.asarray(getattr(elpd, "elpd_i", np.array([]))).ravel().astype(float)
    good_k = float(getattr(elpd, "good_k", DEFAULT_GOOD_K) or DEFAULT_GOOD_K)

    pit = _new_child_pit(
        ctx,
        plan,
        thinned=thinned,
        integrated=integrated_da,
        predictive=predictive,
        maps=maps,
        n_children=n_children,
        nodes=nodes,
    )

    return NewChildValidation(
        plan=plan,
        n_children=n_children,
        posterior_draws_used=draws_used,
        elpd=float(getattr(elpd, "elpd", float("nan"))),
        elpd_se=float(getattr(elpd, "se", float("nan"))),
        p_loo=float(getattr(elpd, "p", float("nan"))),
        pointwise_elpd=pointwise,
        pareto_k=pareto_k,
        good_k=good_k,
        latents_redrawn=latents,
        observed_nodes=nodes,
        latent_mc_error=latent_mc_error,
        pit=pit,
    )


def _pit_groups(
    ctx: StatisticalFitContext, nodes: Sequence[str], maps: dict[str, np.ndarray]
) -> list[tuple[str, str, np.ndarray]]:
    """``(label, node, row mask)`` per calibration group.

    One group per likelihood node, split further by outcome where a node flattens
    several measures into one vector and records the cell map to say which is which.
    Never across nodes or outcomes: those have different item ceilings, and summing
    raw counts over them would compare two different scales — the rule the joint
    family's per-outcome predictive checks already follow.
    """
    from language_reading_predictors.statistical_models.ppc_artifacts import (
        cell_outcome_labels,
    )

    plan_obj = getattr(ctx, "resolved_plan", None)
    # Families name their outcome tuple differently — ``joint`` calls it ``outcomes``,
    # ``joint_mechanism`` ``outcome_symbols``. Reading only the first name silently
    # produced ONE calibration group per node for the joint-mechanism family, pooling a
    # 79-item word-reading count with a 6-item nonword one: exactly the incompatible-
    # denominator pooling every other predictive check in this repo refuses.
    outcomes = tuple(
        getattr(plan_obj, "outcomes", None)
        or getattr(plan_obj, "outcome_symbols", None)
        or ()
    )
    groups: list[tuple[str, str, np.ndarray]] = []
    for node in nodes:
        labels = cell_outcome_labels(ctx, node, outcomes) if outcomes else None
        if labels is not None and len(labels) == maps[node].size:
            for symbol in dict.fromkeys(labels):
                mask = np.array([label == symbol for label in labels])
                groups.append((str(symbol), node, mask))
        else:
            label = node[len("score_") :] if node.startswith("score_") else node
            groups.append((label, node, np.ones(maps[node].size, dtype=bool)))
    return groups


def _new_child_pit(
    ctx: StatisticalFitContext,
    plan: NewChildPlan,
    *,
    thinned: xr.Dataset,
    integrated: xr.DataArray,
    predictive: dict[str, list[np.ndarray]],
    maps: dict[str, np.ndarray],
    n_children: int,
    nodes: Sequence[str],
) -> pd.DataFrame:
    """Child-level PIT, one row per child and calibration group.

    The holdout unit is the **child**: the importance weights come from the integrated
    child term, and the predictive draws come from the same fresh latents, so a child's
    own data informs neither. The test quantity is the child's total on one measure,
    which is well defined because every row within a measure shares its denominator.
    """
    from arviz_stats import loo_pit as _loo_pit

    observed = _dataset(getattr(ctx.trace, "observed_data", None))
    if observed is None:  # pragma: no cover - checked by child_row_maps
        return pd.DataFrame()
    subject_ids = _subject_ids(ctx, n_children)
    frames: list[pd.DataFrame] = []
    for label, node, mask in _pit_groups(ctx, nodes, maps):
        rows = maps[node][mask]
        present = sorted(set(rows.tolist()))
        if not present:
            continue
        position = {child: index for index, child in enumerate(present)}
        obs_rows = np.asarray(observed[node].values).ravel()[mask]
        observed_totals: np.ndarray = np.zeros(len(present), dtype=float)
        for value, child in zip(obs_rows, rows, strict=True):
            observed_totals[position[child]] += float(value)
        stacked = np.stack([draw[..., mask] for draw in predictive[node]], axis=0)
        m, n_chain, n_draw, _ = stacked.shape
        chosen = stacked[_pit_draw_choice(m, n_chain, n_draw), np.arange(n_chain)[:, None], np.arange(n_draw)[None, :]]
        totals = np.zeros((n_chain, n_draw, len(present)), dtype=float)
        for column, child in enumerate(rows):
            totals[..., position[child]] += chosen[..., column]
        unit = np.arange(len(present))
        weights = integrated.isel(loo_child=present).rename({"loo_child": "pit_unit"})
        weights = weights.assign_coords(pit_unit=unit)
        tree = xr.DataTree()
        tree["posterior"] = xr.DataTree(thinned)
        tree["log_likelihood"] = xr.DataTree(xr.Dataset({"y_child": weights}))
        tree["posterior_predictive"] = xr.DataTree(
            xr.Dataset(
                {
                    "y_child": xr.DataArray(
                        totals,
                        dims=("chain", "draw", "pit_unit"),
                        coords={
                            "chain": weights.coords["chain"],
                            "draw": weights.coords["draw"],
                            "pit_unit": unit,
                        },
                    )
                }
            )
        )
        tree["observed_data"] = xr.DataTree(
            xr.Dataset(
                {
                    "y_child": xr.DataArray(
                        observed_totals, dims=("pit_unit",), coords={"pit_unit": unit}
                    )
                }
            )
        )
        values = _loo_pit(tree, var_names=["y_child"], random_state=plan.random_seed)
        frames.append(
            pd.DataFrame(
                {
                    "measure": label,
                    "likelihood_node": node,
                    "child_index": present,
                    "subject_id": [subject_ids[child] for child in present],
                    "observed_total": observed_totals,
                    "new_child_pit": np.asarray(values["y_child"].values).ravel(),
                }
            )
        )
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            columns=[
                "measure",
                "likelihood_node",
                "child_index",
                "subject_id",
                "observed_total",
                "new_child_pit",
            ]
        )
    )


def _pit_draw_choice(m: int, n_chain: int, n_draw: int) -> np.ndarray:
    """One latent re-draw per (chain, draw), chosen deterministically.

    Each posterior draw contributes one predictive draw, so the PIT sample has the
    shape ArviZ's weighted PIT expects while every latent re-draw still enters across
    the sample. A fixed cycling choice keeps the diagnostic reproducible.
    """
    return (np.arange(n_chain)[:, None] * n_draw + np.arange(n_draw)[None, :]) % m


def _subject_ids(ctx: StatisticalFitContext, n_children: int) -> list[Any]:
    """One publishable identifier per child index, falling back to the index itself."""
    prepared = getattr(ctx, "prepared", None)
    ids = getattr(prepared, "subject_ids", None)
    if ids is None:
        return list(range(n_children))
    ids = list(np.asarray(ids).ravel())
    if len(ids) == n_children:
        return ids
    return list(range(n_children))


def write_new_child_validation(
    ctx: StatisticalFitContext, plan: NewChildPlan
) -> NewChildValidation | None:
    """Run the validation and persist its tables plus a PIT figure per measure.

    Returns ``None`` only for the expected-absence case, which is recorded as a skip;
    a defect raises, because #637 stage 1 settled that a programming error must not be
    laundered into an available-looking "unavailable" row.
    """
    try:
        result = run_new_child_validation(ctx, plan)
    except NewChildEvidenceUnavailable as exc:
        rprint(f"[yellow]new-child validation unavailable: {exc}[/yellow]")
        save_table(
            ctx,
            "new_child_loo",
            pd.DataFrame(
                [
                    {
                        "prediction_target": plan.prediction_target,
                        "holdout_unit": "child",
                        "status": "unavailable",
                        "reason": str(exc),
                    }
                ]
            ),
            required=False,
        )
        return None

    save_table(ctx, "new_child_loo", pd.DataFrame([result.summary_row()]))
    save_table(
        ctx,
        "new_child_pareto_k",
        pd.DataFrame(
            {
                "child_index": np.arange(result.n_children, dtype=int),
                "subject_id": _subject_ids(ctx, result.n_children),
                "pareto_k": result.pareto_k,
                "good_k_threshold": result.good_k,
                "new_child_loo_reliable": result.pareto_k <= result.good_k,
            }
        ).sort_values("pareto_k", ascending=False),
    )
    if not result.pit.empty:
        save_table(ctx, "new_child_pit", result.pit)
        _plot_new_child_pit(ctx, result)
    if not result.reliable:
        reasons = []
        if result.n_unreliable:
            reasons.append(
                f"{result.n_unreliable} of {result.n_children} children exceed "
                f"good_k = {result.good_k:.2f}"
            )
        if not result.integration_reliable:
            reasons.append(
                f"the latent integral's half-split error ({result.latent_mc_error:.3g} "
                f"per child) is large beside the ELPD's own SE ({result.elpd_se:.3g})"
            )
        rprint(f"[yellow]new-child ELPD withheld: {'; '.join(reasons)}[/yellow]")
    return result


def _plot_new_child_pit(ctx: StatisticalFitContext, result: NewChildValidation) -> None:
    """One PIT ECDF-difference figure per measure, named for the declared unit.

    Plotted as the difference from uniform with a Kolmogorov-Smirnov envelope rather
    than as a density: at n = 33-96 children a kernel density of that many PIT values
    reads as structure it cannot support, while the ECDF difference shows the same
    departure against a band that is honest about the sample size. One figure per
    measure, each with its PNG, SVG and CSV siblings, so a report can use one without
    the others.
    """
    import matplotlib.pyplot as plt

    from dse_research_utils.plot.styles import COLOUR_BLUE, FIGSIZE_LG

    from language_reading_predictors.figure_io import save_styled_figure

    for measure, frame in result.pit.groupby("measure", sort=False):
        label = f"new_child_pit_{str(measure).lower()}"
        with guard_optional(ctx, f"{label}.png", filename=f"{label}.png"):
            values = np.sort(np.asarray(frame["new_child_pit"], dtype=float))
            n = values.size
            if n == 0:
                continue
            grid = np.linspace(0.0, 1.0, 201)
            ecdf = np.searchsorted(values, grid, side="right") / n
            band = 1.36 / math.sqrt(n)  # 95% Kolmogorov-Smirnov envelope
            fig, ax = plt.subplots(figsize=FIGSIZE_LG)
            ax.axhline(0.0, color="0.4", lw=1.0)
            ax.fill_between(grid, -band, band, color="0.88", label="95% uniform envelope")
            ax.plot(grid, ecdf - grid, lw=1.8, color=COLOUR_BLUE)
            ax.set_xlabel("New-child PIT value")
            ax.set_ylabel("ECDF minus uniform")
            ax.set_title(
                f"New-child PIT calibration ({measure}) - the whole child is held out",
                fontsize=10,
            )
            ax.legend(loc="upper right", fontsize="small")
            save_styled_figure(
                ctx.output_dir,
                label,
                fig=fig,
                data=pd.DataFrame({"pit_grid": grid, "ecdf_minus_uniform": ecdf - grid}),
            )
