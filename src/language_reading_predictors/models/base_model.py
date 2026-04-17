# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Declarative model definition base classes.

Each model family is a Python class. Selection variants (derivatives) are
subclasses that **chain** their parent's feature changes via
``selection_steps``. Concrete classes (those setting ``model_id``) are
auto-registered in the global ``MODELS`` dict at class-creation time.

Example
-------
::

    class LRP01(GainModel):
        model_id = "lrp01"
        target_var = V.EWRSWR_GAIN
        include = [V.EWRSWR]
        cv_splits = 53

    class LRP01Select01(LRP01):
        model_id = "lrp01_select01"
        variant_of = "lrp01"
        selection_steps = [SelectionStep(removed=[V.BEHAV], notes="...")]
"""

from __future__ import annotations

from typing import Any, ClassVar

from language_reading_predictors.data_variables import Predictors
from language_reading_predictors.models.common import (
    ModelConfig,
    SelectionStep,
    ShapScatterSpec,
)


# ── global registry ──────────────────────────────────────────────────────

MODELS: dict[str, ModelConfig] = {}
"""Global model registry, populated by ``ModelDefinition.__init_subclass__``."""


# ── helpers ──────────────────────────────────────────────────────────────


def _build_predictors(
    target_var: str,
    base: list[str],
    include: list[str],
    exclude: list[str],
    selection_steps: list[SelectionStep],
) -> list[str]:
    """Assemble the final predictor list.

    1. Start from *base* (e.g. ``Predictors.DEFAULT_GAIN``).
    2. Remove the target variable.
    3. Prepend *include* vars not already present.
    4. Remove *exclude* vars.
    5. Apply *selection_steps* in order (remove then add for each step).
    """
    exclude_set = {target_var, *exclude}
    predictors = [p for p in base if p not in exclude_set]

    extra = [v for v in include if v not in predictors and v != target_var]
    predictors = extra + predictors

    for step in selection_steps:
        remove_set = set(step.removed)
        predictors = [p for p in predictors if p not in remove_set]
        add = [v for v in step.added if v not in predictors]
        predictors = predictors + add

    return predictors


def _collect_selection_history(cls: type) -> list[SelectionStep]:
    """Walk the MRO to collect chained selection steps (oldest ancestor first)."""
    steps: list[SelectionStep] = []
    # Collect classes in MRO order (excluding object and ModelDefinition itself)
    chain = [
        c
        for c in reversed(cls.__mro__)
        if c is not cls
        and hasattr(c, "selection_steps")
        and c.__dict__.get("selection_steps") is not None
    ]
    for ancestor in chain:
        own = ancestor.__dict__.get("selection_steps", [])
        steps.extend(own)
    # Add this class's own steps last
    own_steps = cls.__dict__.get("selection_steps") or []
    steps.extend(own_steps)
    return steps


# ── base class ───────────────────────────────────────────────────────────


class ModelDefinition:
    """Abstract base for declarative model definitions.

    Subclasses set class-level attributes to configure the model. Any
    concrete class (one that sets ``model_id``) is auto-registered in
    ``MODELS`` via ``__init_subclass__``.
    """

    # ── required (must be set by concrete subclasses) ────────────────────

    model_id: ClassVar[str | None] = None
    """Short identifier, e.g. ``"lrp01"``. ``None`` for abstract bases."""

    target_var: ClassVar[str]
    """Column name of the target variable."""

    # ── optional overrides ───────────────────────────────────────────────

    description: ClassVar[str] = ""
    """Human-readable description for console output and reports."""

    include: ClassVar[list[str]] = []
    """Extra predictor vars to prepend to the base set."""

    exclude: ClassVar[list[str]] = []
    """Predictor vars to remove from the base set."""

    pipeline_cls: ClassVar[Any] = None
    """Pipeline class (e.g. ``LGBMPipeline``). Set lazily to avoid circular imports."""

    params: ClassVar[dict[str, Any]] = {}
    """Estimator hyperparameters (passed to the estimator constructor)."""

    cv_splits: ClassVar[int] = 51
    """Number of GroupKFold splits."""

    outlier_threshold: ClassVar[float | None] = None
    """If set, exclude rows where target >= this value."""

    perm_importance_repeats: ClassVar[int] = 50
    """Number of repeats for permutation importance."""

    pdp_features: ClassVar[list[str] | None] = None
    """Explicit PDP feature list. ``None`` → auto-select top-N."""

    pdp_top_n: ClassVar[int] = 15
    """Number of top features for auto-selected PDP."""

    shap_scatter_specs: ClassVar[list[ShapScatterSpec]] = []
    """Ordered list of SHAP scatter/dependence plot sets to generate for
    this model. Empty by default — set on concrete subclasses to declare
    the plots the model exploration needs."""

    random_seed: ClassVar[int] = 47

    variant_of: ClassVar[str | None] = None
    """If set, this model is a selection variant of another model."""

    notes: ClassVar[str] = ""
    """Free-text rationale persisted in ``config.json``."""

    selection_steps: ClassVar[list[SelectionStep] | None] = None
    """Feature-selection steps applied at *this* class level.
    Steps from ancestor classes are collected automatically."""

    # ── base predictor set (overridden by GainModel / LevelModel) ────────

    _base_predictors: ClassVar[list[str]] = []

    # ── auto-registration ────────────────────────────────────────────────

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.model_id is not None:
            MODELS[cls.model_id] = cls.to_config()

    # ── config generation ────────────────────────────────────────────────

    @classmethod
    def to_config(cls) -> ModelConfig:
        """Generate a ``ModelConfig`` for the pipeline."""
        all_history = _collect_selection_history(cls)
        predictors = _build_predictors(
            target_var=cls.target_var,
            base=cls._base_predictors,
            include=cls.include,
            exclude=cls.exclude,
            selection_steps=all_history,
        )

        # Resolve pipeline_cls: walk MRO to find the first explicitly set value
        pipeline_cls = cls.pipeline_cls
        if pipeline_cls is None:
            for ancestor in cls.__mro__:
                pcls = ancestor.__dict__.get("pipeline_cls")
                if pcls is not None:
                    pipeline_cls = pcls
                    break

        # Resolve params: walk MRO to find the first non-empty
        params = cls.params
        if params is None or (not params and "params" not in cls.__dict__):
            for ancestor in cls.__mro__:
                p = ancestor.__dict__.get("params")
                if p:
                    params = p
                    break

        return ModelConfig(
            model_id=cls.model_id,
            description=cls.description,
            target_var=cls.target_var,
            predictor_vars=predictors,
            model_params=dict(params),
            pipeline_cls=pipeline_cls,
            cv_splits=cls.cv_splits,
            outlier_threshold=cls.outlier_threshold,
            perm_importance_repeats=cls.perm_importance_repeats,
            pdp_features=cls.pdp_features,
            pdp_top_n=cls.pdp_top_n,
            shap_scatter_specs=list(cls.shap_scatter_specs),
            random_seed=cls.random_seed,
            variant_of=cls.variant_of,
            notes=cls.notes,
            selection_history=all_history,
        )


# ── convenience subclasses ───────────────────────────────────────────────


class GainModel(ModelDefinition):
    """Base for gain-prediction models.

    Automatically uses ``Predictors.DEFAULT_GAIN`` and prepends the base
    variable (target name with ``_gain`` suffix stripped).
    """

    _base_predictors: ClassVar[list[str]] = Predictors.DEFAULT_GAIN

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # Auto-include the base variable (e.g. ewrswr for ewrswr_gain)
        if hasattr(cls, "target_var") and cls.target_var:
            base_var = cls.target_var.removesuffix("_gain")
            own_include = cls.__dict__.get("include")
            include = list(own_include if own_include is not None else cls.include or [])
            if base_var not in include:
                include.insert(0, base_var)
                cls.include = include
        super().__init_subclass__(**kwargs)


class LevelModel(ModelDefinition):
    """Base for level-prediction models.

    Automatically uses ``Predictors.DEFAULT_LEVEL``.
    """

    _base_predictors: ClassVar[list[str]] = Predictors.DEFAULT_LEVEL


class BayesianModel(ModelDefinition):
    """Declarative base for Bayesian DAG models.

    Concrete subclasses set ``model_id``, ``dag_spec``, ``description``,
    and ``pipeline_cls``. The tree-model predictor-selection flow is
    bypassed — predictors are implicit in the DAG spec.
    """

    dag_spec: ClassVar[Any] = None
    """A :class:`BayesianDAGSpec` instance declaring the model's DAG."""

    # Tree-specific fields stay at their defaults; they are ignored.

    target_var: ClassVar[str] = ""
    """Populated from the DAG spec's primary outcome node for display."""

    primary_node: ClassVar[str | None] = None
    """Name of the primary outcome node (for surface display / summaries).
    Defaults to the last node in the DAG spec."""

    @classmethod
    def to_config(cls) -> ModelConfig:
        if cls.dag_spec is None:
            msg = f"BayesianModel {cls.__name__} has no dag_spec set."
            raise ValueError(msg)

        primary_name = cls.primary_node or cls.dag_spec.nodes[-1].name
        primary = cls.dag_spec.node(primary_name)
        target_var = cls.target_var or primary.outcome_column

        # Resolve pipeline_cls via MRO, as the parent class does.
        pipeline_cls = cls.__dict__.get("pipeline_cls")
        if pipeline_cls is None:
            for ancestor in cls.__mro__:
                pcls = ancestor.__dict__.get("pipeline_cls")
                if pcls is not None:
                    pipeline_cls = pcls
                    break

        return ModelConfig(
            model_id=cls.model_id,
            description=cls.description,
            target_var=target_var,
            predictor_vars=[],
            model_params={},
            pipeline_cls=pipeline_cls,
            cv_splits=cls.cv_splits,
            outlier_threshold=cls.outlier_threshold,
            random_seed=cls.random_seed,
            variant_of=cls.variant_of,
            notes=cls.notes,
            selection_history=[],
            dag_spec=cls.dag_spec,
        )
