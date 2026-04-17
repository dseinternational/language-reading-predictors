# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Declarative DAG specifications for Bayesian models.

A :class:`BayesianDAGSpec` is a list of :class:`NodeSpec` entries, each of
which names an outcome column (at t2) and the parent predictors (at t1)
that enter its linear predictor. Parents come in four kinds:

``gp``
    Parent enters as ``β·x + f(x)`` where ``f`` is an HSGP-approximated
    1-D Gaussian process. Captures variation from the central linear
    trend.
``linear``
    Parent enters as ``β·x`` only (no GP).
``binary``
    Parent enters as a single treatment contrast ``β·x`` for a 0/1 coded
    variable. Semantically equivalent to ``linear`` but makes intent
    explicit and triggers a different prior.
``offset``
    Parent is the baseline (t1) measurement of the same quantity as the
    outcome. Enters as a linear coefficient on a standardised baseline,
    providing a "gain conditional on baseline" parameterisation for
    beta-binomial outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


ParentKind = Literal["gp", "linear", "binary", "offset"]
LikelihoodKind = Literal["beta_binomial", "normal"]


@dataclass
class ParentSpec:
    """A single predictor of a node."""

    name: str
    """Short symbol (e.g. ``"A"`` for age), used for naming PyMC variables."""

    column: str
    """Column name in the input dataframe, read at t1 (baseline)."""

    kind: ParentKind = "gp"
    """How the parent enters the linear predictor."""

    description: str = ""
    """Free-text description, used in reports."""

    # GP-specific knobs (ignored when kind != "gp")
    ell_alpha: float = 3.0
    ell_beta: float = 3.0
    eta_sigma: float = 0.4
    n_basis: int = 20


@dataclass
class NodeSpec:
    """An outcome modelled as a function of its parents."""

    name: str
    """Short symbol (e.g. ``"W"`` for word reading)."""

    outcome_column: str
    """Column name in the input dataframe, read at t2 (outcome)."""

    likelihood: LikelihoodKind
    """Likelihood family."""

    n_trials: int | None = None
    """Number of trials for ``beta_binomial``. Must be set for that family."""

    parents: list[ParentSpec] = field(default_factory=list)
    """Ordered list of parents entering the linear predictor."""

    description: str = ""
    """Free-text description, used in reports."""

    intercept_sigma: float = 1.5
    """Normal prior sigma for the node's intercept on the link scale."""

    beta_sigma: float = 1.0
    """Normal prior sigma for each parent's linear coefficient."""

    sigma_sigma: float = 1.0
    """HalfNormal prior sigma for the residual SD (``normal`` only)."""

    def __post_init__(self) -> None:
        if self.likelihood == "beta_binomial" and self.n_trials is None:
            msg = (
                f"NodeSpec {self.name!r} uses beta_binomial likelihood but "
                f"has no n_trials set."
            )
            raise ValueError(msg)


@dataclass
class BayesianDAGSpec:
    """A full DAG specification."""

    nodes: list[NodeSpec]
    """Ordered list of outcome nodes."""

    subject_id_column: str = "subject_id"
    """Column identifying subjects — used for sorting and future random effects."""

    time_column: str = "time"
    """Column identifying timepoints in the long-format data."""

    time_filter: int | None = 1
    """Filter the input dataframe to rows where ``time_column`` equals this
    value. Set to ``None`` to disable filtering."""

    def node(self, name: str) -> NodeSpec:
        for n in self.nodes:
            if n.name == name:
                return n
        msg = f"No node named {name!r} in DAG spec."
        raise KeyError(msg)

    def required_columns(self) -> list[str]:
        """All columns this DAG needs loaded from the data frame."""
        cols: set[str] = {self.subject_id_column}
        if self.time_column:
            cols.add(self.time_column)
        for n in self.nodes:
            cols.add(n.outcome_column)
            for p in n.parents:
                cols.add(p.column)
        return sorted(cols)

    def topological_order(self) -> list[str]:
        """Return node names in topological order (parents before children).

        Offset parents (baseline of the same measure) are ignored for
        dependency purposes.
        """
        node_names = {n.name for n in self.nodes}
        deps: dict[str, set[str]] = {
            n.name: {
                p.name
                for p in n.parents
                if p.kind != "offset" and p.name in node_names
            }
            for n in self.nodes
        }
        order: list[str] = []
        remaining = {k: set(v) for k, v in deps.items()}
        while remaining:
            ready = sorted(n for n, d in remaining.items() if not d)
            if not ready:
                msg = f"Cyclic DAG dependency among nodes: {sorted(remaining)}"
                raise ValueError(msg)
            order.extend(ready)
            for n in ready:
                del remaining[n]
            for d in remaining.values():
                d -= set(ready)
        return order

    def to_json_dict(self) -> dict:
        """JSON-serialisable summary, for ``config.json``."""
        return {
            "nodes": [
                {
                    "name": n.name,
                    "outcome_column": n.outcome_column,
                    "likelihood": n.likelihood,
                    "n_trials": n.n_trials,
                    "description": n.description,
                    "parents": [
                        {
                            "name": p.name,
                            "column": p.column,
                            "kind": p.kind,
                            "description": p.description,
                        }
                        for p in n.parents
                    ],
                }
                for n in self.nodes
            ],
            "subject_id_column": self.subject_id_column,
            "time_column": self.time_column,
            "time_filter": self.time_filter,
        }
