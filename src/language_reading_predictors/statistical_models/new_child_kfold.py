# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Grouped child-level K-fold cross-validation for the new-child target (#626).

The companion to :mod:`new_child_predictive`, and the route a family takes when the
importance-sampling one is refused. Both answer the *same* declared question — how well
does this model predict a child it has never seen — so a family declares the target once
and picks the estimator its diagnostics allow.

## Why a second estimator is needed at all

PSIS approximates the leave-one-out posterior by reweighting the full-data posterior.
That works while the two are close. For a child contributing one row per measure it
usually is; for a child contributing a whole four-wave profile across several correlated
measures it is not, and the Pareto shape estimate says so loudly — the historical
joint-growth fits report values in the tens, not fractions. #626 is explicit that naive
PSIS must not be published where Pareto-k is unacceptable, and an unpublishable estimate
is not a validation. Refitting without the held-out children removes the approximation
entirely: nothing is reweighted, so there is no shape parameter to fail.

## What one fold does

The children are partitioned into ``n_folds`` folds, stratified by group so every fold
leaves out a comparable mix rather than, say, most of one cohort. For each fold the model
is **rebuilt and refitted on the training children only**, through the family's own
loader-and-factory callback — the alignment guard :mod:`loo_refit` established, for the
same reason: a held-out density is only interpretable if the refit is the same model.

The held-out children are then scored under that fold's posterior. Their own latent
effects were never in the training fit, so they are drawn from the population exactly as
:mod:`new_child_predictive` draws them, and the fold's global parameters are transplanted
into the full model to do it. A fold whose training subset changes the *shape* of any
global parameter — a group-by-wave cell that no training child supports — is refused
rather than aligned by hand, because a silently reshaped transplant would score the
held-out children against a different model.

## What it gives back

``elpd_kfold`` with its standard error, a per-child pointwise contribution, and — the
half of #626 that PSIS could not supply for these families — a calibration diagnostic
whose holdout unit is genuinely the child: the PIT of each held-out child's total on each
measure, computed from predictive draws of a fit that never saw them. No importance
weights are involved, so unlike the PSIS-weighted PIT it carries no reliability caveat of
its own.

The cost is ``n_folds`` refits at the fit's own sampling settings, which is why this is
opt-in per model rather than a default diagnostic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from rich import print as rprint

from language_reading_predictors.statistical_models.artifacts import save_table
from language_reading_predictors.statistical_models.context import (
    StatisticalFitContext,
)
from language_reading_predictors.statistical_models.new_child_predictive import (
    NewChildEvidenceUnavailable,
    NewChildPlan,
    _dataset,
    _observed_node_names,
    _pit_draw_choice,
    _subject_ids,
    child_row_maps,
    verify_child_latents,
)

__all__ = [
    "KFoldPlan",
    "KFoldValidation",
    "run_child_kfold",
    "subset_panel_children",
    "write_child_kfold",
]


def subset_panel_children(panel: Any, child_indices: Sequence[int]) -> Any:
    """A :class:`LongitudinalPanel` restricted to ``child_indices``, order preserved.

    A K-fold refit needs the family's own container, not a hand-built stand-in, so this
    narrows the panel rather than re-loading it: the source path, digest, denominators,
    group codes, waves and dropped-row bookkeeping all survive unchanged, and only the
    per-subject rows and arrays shrink. Keeping the original child order matters —
    ``child_indices`` are positions in the full model's ordering, and the caller maps
    the fold's held-out children back by that position.

    The per-subject ``counts`` and ``obs_mask`` arrays are narrowed alongside ``long``.
    The joint-growth factory reads neither, but leaving them at full length would put a
    panel in circulation whose arrays disagree with its own subject list, and the next
    reader of that panel would have no way to tell.
    """
    import dataclasses

    kept = [int(index) for index in child_indices]
    subject_ids = list(panel.subject_ids)
    if any(index < 0 or index >= len(subject_ids) for index in kept):
        raise ValueError("child index outside the panel's subject list")
    keep_ids = [subject_ids[index] for index in kept]
    subject_col = panel.dataset.subject_col
    long = panel.long[panel.long[subject_col].isin(set(keep_ids))].copy()
    positions = np.asarray(kept, dtype=int)
    return dataclasses.replace(
        panel,
        long=long,
        subject_ids=keep_ids,
        n_subjects=len(keep_ids),
        counts={
            symbol: np.asarray(values)[positions]
            for symbol, values in panel.counts.items()
        },
        obs_mask={
            symbol: np.asarray(values)[positions]
            for symbol, values in panel.obs_mask.items()
        },
    )


@dataclass(frozen=True, slots=True)
class KFoldPlan:
    """How a family partitions its children and how much it spends per fold."""

    n_folds: int = 5
    n_latent_draws: int = 64
    random_seed: int = 20260626
    stratify: bool = True
    """Balance each fold's group composition. Off only for a single-group cohort."""

    def __post_init__(self) -> None:
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least 2")
        if self.n_latent_draws < 2:
            raise ValueError("n_latent_draws must be at least 2 to integrate anything")

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_folds": self.n_folds,
            "n_latent_draws": self.n_latent_draws,
            "stratify": self.stratify,
        }


@dataclass(frozen=True, slots=True)
class KFoldValidation:
    """The held-out predictive performance of one fitted model."""

    plan: NewChildPlan
    kfold: KFoldPlan
    n_children: int
    n_scored: int
    elpd: float
    elpd_se: float
    pointwise_elpd: np.ndarray
    fold_of_child: np.ndarray
    fold_converged: dict[int, bool]
    latents_redrawn: tuple[str, ...]
    observed_nodes: tuple[str, ...]
    pit: pd.DataFrame = field(default_factory=pd.DataFrame)
    refused_folds: dict[int, str] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        """Whether every child was scored by a converged fold.

        A partial K-fold is reportable but is not the declared estimate: the ELPD then
        covers a subset of children chosen by which refits happened to work, which is
        a selection the reader has to be told about rather than a smaller sample.
        """
        return self.n_scored == self.n_children and all(self.fold_converged.values())

    def summary_row(self) -> dict[str, Any]:
        return {
            "prediction_target": self.plan.prediction_target,
            "holdout_unit": "child",
            "estimator": "grouped_child_kfold",
            "n_folds": self.kfold.n_folds,
            "n_children": self.n_children,
            "n_children_scored": self.n_scored,
            "elpd_kfold": self.elpd,
            "elpd_kfold_se": self.elpd_se,
            "complete": self.complete,
            "n_folds_converged": sum(1 for ok in self.fold_converged.values() if ok),
            "n_folds_refused": len(self.refused_folds),
            "latents_redrawn": " ".join(self.latents_redrawn) or "(none)",
            "observed_nodes": " ".join(self.observed_nodes),
            "n_latent_draws": self.kfold.n_latent_draws,
        }


def _fold_assignment(
    n_children: int, groups: np.ndarray | None, kfold: KFoldPlan
) -> np.ndarray:
    """Deterministic fold index per child, balanced within group where asked.

    Children are shuffled with a fixed seed and dealt round-robin within each group, so
    a fold never happens to hold out most of one cohort — which at 20 to 40 children per
    group would leave a group-indexed scale parameter estimated from a handful of
    training children and make the fold's refit incomparable to the others.
    """
    rng = np.random.default_rng(kfold.random_seed)
    folds: np.ndarray = np.empty(n_children, dtype=int)
    if groups is None or not kfold.stratify:
        order = rng.permutation(n_children)
        folds[order] = np.arange(n_children) % kfold.n_folds
        return folds
    for value in np.unique(groups):
        members = np.flatnonzero(groups == value)
        order = members[rng.permutation(members.size)]
        folds[order] = np.arange(members.size) % kfold.n_folds
    return folds


def _child_groups(ctx: StatisticalFitContext, n_children: int) -> np.ndarray | None:
    """One group code per child, from whatever container the family prepared."""
    prepared = getattr(ctx, "prepared", None)
    if prepared is None:
        return None
    for attribute in ("group_codes", "G"):
        values = getattr(prepared, attribute, None)
        if values is None:
            continue
        arr = np.asarray(values).ravel()
        if arr.size == n_children:
            return arr
    long = getattr(prepared, "long", None)
    dataset = getattr(prepared, "dataset", None)
    subject_col = getattr(dataset, "subject_col", None)
    group_col = getattr(dataset, "group_col", None)
    subject_ids = getattr(prepared, "subject_ids", None)
    if (
        long is not None
        and subject_col
        and group_col
        and subject_ids is not None
        and subject_col in long
        and group_col in long
    ):
        by_subject = long.drop_duplicates(subject_col).set_index(subject_col)[group_col]
        try:
            return np.asarray([by_subject.loc[value] for value in subject_ids])
        except KeyError:  # pragma: no cover - a panel id missing from its own frame
            return None
    return None


def _transplant(
    model: pm.Model,
    full_posterior: xr.Dataset,
    fold_posterior: xr.Dataset,
    latents: Sequence[str],
) -> xr.Dataset:
    """The fold's posterior, shaped for the full model, with the latents removed.

    Only the **free** random variables are carried over. Deterministics are functions
    of those, so the full model recomputes them for its own child set; transplanting
    them would import the training cohort's ``subject_offset`` — one value per training
    child — into a model that has a row for every child, which is a shape error dressed
    as data.

    Every global free variable must arrive with the same non-sample dimensions it has
    in the full fit. A mismatch means the training subset changed the model rather than
    only its rows — a lost group-by-wave cell, a dropped coordinate — and scoring the
    held-out children against it would be scoring them against a different model.
    """
    kept: dict[str, xr.DataArray] = {}
    for rv in model.free_RVs:
        name = rv.name
        if name in latents:
            continue
        if name not in full_posterior:  # pragma: no cover - defensive
            raise ValueError(f"full posterior is missing the free variable {name!r}")
        if name not in fold_posterior:
            raise ValueError(f"fold posterior is missing the free variable {name!r}")
        variable = full_posterior[name]
        candidate = fold_posterior[name]
        expected = {
            d: variable.sizes[d] for d in variable.dims if d not in ("chain", "draw")
        }
        actual = {
            d: candidate.sizes[d] for d in candidate.dims if d not in ("chain", "draw")
        }
        if expected != actual:
            raise ValueError(
                f"fold posterior reshaped {name!r} ({actual} vs {expected}); the "
                "training subset changed the model, not only its rows"
            )
        kept[name] = candidate
    return xr.Dataset(kept)


def run_child_kfold(
    ctx: StatisticalFitContext,
    plan: NewChildPlan,
    kfold: KFoldPlan,
    rebuild: Callable[[Sequence[int]], Any],
) -> KFoldValidation:
    """Refit once per fold and score each fold's held-out children.

    ``rebuild`` receives the **training** child indices, in the full model's own child
    ordering, and returns a ``BuiltModel`` for those children alone — the family's own
    loader and factory, so the refit is the same model by construction.
    """
    from language_reading_predictors.statistical_models.diagnostics import (
        log_density_model,
    )
    from language_reading_predictors.statistical_models.subfits import run_subfit

    if ctx.model is None or ctx.trace is None:
        raise NewChildEvidenceUnavailable("no built model or trace on the context")
    full_posterior = _dataset(getattr(ctx.trace, "posterior", None))
    if full_posterior is None:
        raise NewChildEvidenceUnavailable("trace carries no posterior group")

    model = ctx.model
    nodes = _observed_node_names(model, plan)
    maps, n_children = child_row_maps(ctx, nodes)
    latents = verify_child_latents(model, plan)
    density_model = log_density_model(model)
    observed = _dataset(getattr(ctx.trace, "observed_data", None))
    if observed is None:  # pragma: no cover - checked by child_row_maps
        raise NewChildEvidenceUnavailable("trace carries no observed_data group")

    folds = _fold_assignment(n_children, _child_groups(ctx, n_children), kfold)
    pointwise: np.ndarray = np.full(n_children, np.nan, dtype=float)
    converged: dict[int, bool] = {}
    refused: dict[int, str] = {}
    pit_frames: list[pd.DataFrame] = []

    for fold in range(kfold.n_folds):
        held_out = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        if held_out.size == 0:  # pragma: no cover - only with more folds than children
            continue
        try:
            built = rebuild(training.tolist())
        except Exception as exc:  # noqa: BLE001 - a refused fold is reported, not fatal
            refused[fold] = f"rebuild failed: {exc}"
            rprint(f"[yellow]K-fold {fold}: {refused[fold]}[/yellow]")
            continue
        result = run_subfit(
            ctx,
            built,
            label=f"new_child_kfold_{fold}",
            role="cross_validation",
        )
        converged[fold] = bool(getattr(result, "converged", False))
        fold_posterior = _dataset(getattr(result.trace, "posterior", None))
        if fold_posterior is None:  # pragma: no cover - run_subfit always returns one
            refused[fold] = "fold trace carries no posterior"
            continue
        try:
            transplanted = _transplant(
                model, full_posterior, fold_posterior, latents
            )
        except ValueError as exc:
            refused[fold] = str(exc)
            rprint(f"[yellow]K-fold {fold}: {exc}[/yellow]")
            continue
        scored, predictive = _score_held_out(
            model,
            plan,
            kfold,
            transplanted=transplanted,
            latents=latents,
            nodes=nodes,
            maps=maps,
            n_children=n_children,
            density_model=density_model,
        )
        pointwise[held_out] = scored[held_out]
        pit_frames.append(
            _fold_pit(
                ctx,
                plan,
                predictive=predictive,
                observed=observed,
                maps=maps,
                nodes=nodes,
                held_out=held_out,
                fold=fold,
                n_children=n_children,
            )
        )

    scored_mask = np.isfinite(pointwise)
    n_scored = int(scored_mask.sum())
    values = pointwise[scored_mask]
    elpd = float(values.sum()) if n_scored else float("nan")
    elpd_se = (
        float(math.sqrt(n_scored * float(np.var(values, ddof=1))))
        if n_scored > 1
        else float("nan")
    )
    pit = (
        pd.concat([f for f in pit_frames if not f.empty], ignore_index=True)
        if any(not f.empty for f in pit_frames)
        else pd.DataFrame()
    )
    return KFoldValidation(
        plan=plan,
        kfold=kfold,
        n_children=n_children,
        n_scored=n_scored,
        elpd=elpd,
        elpd_se=elpd_se,
        pointwise_elpd=pointwise,
        fold_of_child=folds,
        fold_converged=converged,
        latents_redrawn=latents,
        observed_nodes=nodes,
        pit=pit,
        refused_folds=refused,
    )


def _score_held_out(
    model: pm.Model,
    plan: NewChildPlan,
    kfold: KFoldPlan,
    *,
    transplanted: xr.Dataset,
    latents: Sequence[str],
    nodes: Sequence[str],
    maps: dict[str, np.ndarray],
    n_children: int,
    density_model: pm.Model,
) -> tuple[np.ndarray, dict[str, list[np.ndarray]]]:
    """Per-child held-out log predictive density under one fold's posterior.

    ``log (1/S) sum_s (1/M) sum_m p(y_i | theta_fold^s, u^(m))`` — the fold posterior
    integrated over both its own draws and fresh population draws of the child's latent,
    which is what "a child this fit has never seen" means. Returns the per-child values
    for every child (the caller keeps only the held-out ones) and the predictive draws
    the calibration diagnostic reuses.
    """
    running: np.ndarray | None = None
    predictive: dict[str, list[np.ndarray]] = {node: [] for node in nodes}
    n_chain = int(transplanted.sizes["chain"])
    n_draw = int(transplanted.sizes["draw"])
    for index in range(kfold.n_latent_draws):
        tree = xr.DataTree()
        tree["posterior"] = xr.DataTree(transplanted)
        with model:
            drawn = pm.sample_posterior_predictive(
                tree,
                var_names=[*latents, *nodes],
                extend_inferencedata=False,
                random_seed=plan.random_seed + 100_000 + index,
                progressbar=False,
            )
        redrawn = _dataset(getattr(drawn, "posterior_predictive", None))
        if redrawn is None:  # pragma: no cover - defensive
            raise ValueError("posterior predictive re-draw returned no group")
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
        with_fresh = transplanted.copy()
        for name in latents:
            with_fresh[name] = redrawn[name]
        fresh_tree = xr.DataTree()
        fresh_tree["posterior"] = xr.DataTree(with_fresh)
        computed = pm.compute_log_likelihood(
            fresh_tree, model=density_model, progressbar=False, extend_inferencedata=True
        )
        log_likelihood = _dataset(getattr(computed, "log_likelihood", None))
        if log_likelihood is None:  # pragma: no cover - defensive
            raise ValueError("no log likelihood was computed under the re-drawn latents")
        child_ll: np.ndarray = np.zeros((n_chain, n_draw, n_children), dtype=float)
        for node in nodes:
            cell = np.asarray(
                log_likelihood[node].transpose("chain", "draw", ...).values, dtype=float
            )
            rows = maps[node]
            for child in range(n_children):
                child_ll[..., child] += cell[..., rows == child].sum(axis=-1)
        running = child_ll if running is None else np.logaddexp(running, child_ll)
    if running is None:  # pragma: no cover - n_latent_draws >= 2 is validated
        raise ValueError("no latent re-draws were scored")
    # One log-mean-exp over the (draw, latent re-draw) product: both are Monte-Carlo
    # samples of the same predictive integral, so they collapse together.
    integrated = running - math.log(kfold.n_latent_draws)
    flat = integrated.reshape(-1, n_children)
    scored = np.log(np.mean(np.exp(flat - flat.max(axis=0)), axis=0)) + flat.max(axis=0)
    return scored, predictive


def _fold_pit(
    ctx: StatisticalFitContext,
    plan: NewChildPlan,
    *,
    predictive: dict[str, list[np.ndarray]],
    observed: xr.Dataset,
    maps: dict[str, np.ndarray],
    nodes: Sequence[str],
    held_out: np.ndarray,
    fold: int,
    n_children: int,
) -> pd.DataFrame:
    """Randomised PIT of each held-out child's per-measure total.

    Genuinely held out: the predictive draws come from a fit the child was not in, so
    no importance weights and no Pareto-k caveat. Randomised because the totals are
    discrete — without the tie-breaking term a count distribution's PIT is
    systematically non-uniform whatever the model does (Czado, Gneiting and Held 2009).
    """
    from language_reading_predictors.statistical_models.new_child_predictive import (
        _pit_groups,
    )

    rng = np.random.default_rng(plan.random_seed + fold)
    subject_ids = _subject_ids(ctx, n_children)
    held = set(held_out.tolist())
    frames: list[pd.DataFrame] = []
    for label, node, mask in _pit_groups(ctx, nodes, maps):
        rows = maps[node][mask]
        present = [child for child in sorted(set(rows.tolist())) if child in held]
        if not present:
            continue
        position = {child: index for index, child in enumerate(present)}
        obs_rows = np.asarray(observed[node].values).ravel()[mask]
        totals_obs: np.ndarray = np.zeros(len(present), dtype=float)
        for value, child in zip(obs_rows, rows, strict=True):
            if child in position:
                totals_obs[position[child]] += float(value)
        stacked = np.stack([draw[..., mask] for draw in predictive[node]], axis=0)
        m, n_chain, n_draw, _ = stacked.shape
        chosen = stacked[
            _pit_draw_choice(m, n_chain, n_draw),
            np.arange(n_chain)[:, None],
            np.arange(n_draw)[None, :],
        ]
        totals = np.zeros((n_chain, n_draw, len(present)), dtype=float)
        for column, child in enumerate(rows):
            if child in position:
                totals[..., position[child]] += chosen[..., column]
        flat = totals.reshape(-1, len(present))
        below = (flat < totals_obs[None, :]).mean(axis=0)
        equal = (flat == totals_obs[None, :]).mean(axis=0)
        pit = below + rng.uniform(size=len(present)) * equal
        frames.append(
            pd.DataFrame(
                {
                    "measure": label,
                    "likelihood_node": node,
                    "fold": fold,
                    "child_index": present,
                    "subject_id": [subject_ids[child] for child in present],
                    "observed_total": totals_obs,
                    "held_out_pit": pit,
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


#: Sampling tiers whose fits are diagnostic-only, where ``n_folds`` extra refits buy
#: nothing: a dev-config fold cannot converge, so its held-out density is noise. The
#: skip is recorded rather than silent, and the code path itself is still exercised by
#: the tests and by any rep-lite run.
_SKIP_CONFIGS = frozenset({"dev"})


def write_child_kfold(
    ctx: StatisticalFitContext,
    plan: NewChildPlan,
    kfold: KFoldPlan,
    rebuild: Callable[[Sequence[int]], Any],
) -> KFoldValidation | None:
    """Run grouped K-fold and persist its summary, per-child and PIT tables.

    Returns ``None`` at a diagnostic-only sampling tier, where the refits would cost
    as much as the fit and produce folds that cannot converge.
    """
    config_name = str(getattr(ctx.reporting, "config_name", "") or "")
    if config_name in _SKIP_CONFIGS:
        rprint(
            f"[yellow]new-child K-fold skipped at the {config_name!r} sampling tier: "
            f"{kfold.n_folds} refits that cannot converge would not validate "
            "anything[/yellow]"
        )
        save_table(
            ctx,
            "new_child_kfold",
            pd.DataFrame(
                [
                    {
                        "prediction_target": plan.prediction_target,
                        "holdout_unit": "child",
                        "estimator": "grouped_child_kfold",
                        "status": "skipped",
                        "reason": f"diagnostic-only sampling tier {config_name!r}",
                    }
                ]
            ),
            required=False,
        )
        return None
    result = run_child_kfold(ctx, plan, kfold, rebuild)
    save_table(ctx, "new_child_kfold", pd.DataFrame([result.summary_row()]))
    save_table(
        ctx,
        "new_child_kfold_pointwise",
        pd.DataFrame(
            {
                "child_index": np.arange(result.n_children, dtype=int),
                "subject_id": _subject_ids(ctx, result.n_children),
                "fold": result.fold_of_child,
                "held_out_elpd": result.pointwise_elpd,
                "fold_converged": [
                    result.fold_converged.get(int(f), False) for f in result.fold_of_child
                ],
            }
        ),
    )
    if not result.pit.empty:
        save_table(ctx, "new_child_kfold_pit", result.pit)
        _plot_kfold_pit(ctx, result)
    if not result.complete:
        rprint(
            f"[yellow]K-fold covered {result.n_scored} of {result.n_children} "
            "children; the ELPD is partial and is reported as such[/yellow]"
        )
    return result


def _plot_kfold_pit(ctx: StatisticalFitContext, result: KFoldValidation) -> None:
    """One held-out PIT figure per measure, in the shape the PSIS version uses."""
    import matplotlib.pyplot as plt

    from dse_research_utils.plot.styles import COLOUR_BLUE, FIGSIZE_LG

    from language_reading_predictors.figure_io import save_styled_figure
    from language_reading_predictors.statistical_models.artifacts import guard_optional

    for measure, frame in result.pit.groupby("measure", sort=False):
        label = f"new_child_kfold_pit_{str(measure).lower()}"
        with guard_optional(ctx, f"{label}.png", filename=f"{label}.png"):
            values = np.sort(np.asarray(frame["held_out_pit"], dtype=float))
            n = values.size
            if n == 0:
                continue
            grid = np.linspace(0.0, 1.0, 201)
            ecdf = np.searchsorted(values, grid, side="right") / n
            band = 1.36 / math.sqrt(n)
            fig, ax = plt.subplots(figsize=FIGSIZE_LG)
            ax.axhline(0.0, color="0.4", lw=1.0)
            ax.fill_between(grid, -band, band, color="0.88", label="95% uniform envelope")
            ax.plot(grid, ecdf - grid, lw=1.8, color=COLOUR_BLUE)
            ax.set_xlabel("Held-out PIT value")
            ax.set_ylabel("ECDF minus uniform")
            ax.set_title(
                f"K-fold held-out PIT calibration ({measure}) - refits, not weights",
                fontsize=10,
            )
            ax.legend(loc="upper right", fontsize="small")
            save_styled_figure(
                ctx.output_dir,
                label,
                fig=fig,
                data=pd.DataFrame({"pit_grid": grid, "ecdf_minus_uniform": ecdf - grid}),
            )
