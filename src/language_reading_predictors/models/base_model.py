# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Declarative model definition base classes.

Each model family is a Python class that configures its predictor set via
class-level attributes (``include`` / ``exclude`` on top of the family's
``DEFAULT_*`` base). Concrete classes (those setting ``model_id``) are
auto-registered in the global ``MODELS`` dict at class-creation time.

Example
-------
::

    class LRPGBG12(GainModel):
        model_id = "lrp-rli-gbg-012"
        target_var = V.EWRSWR_GAIN
        include = (V.EWRSWR,)
        cv_splits = 53
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, ClassVar

from language_reading_predictors.data_variables import Predictors
from language_reading_predictors.models.common import (
    ModelConfig,
    ShapScatterSpec,
)


# ── global registry ──────────────────────────────────────────────────────

MODELS: dict[str, ModelConfig] = {}
"""Global model registry, populated by ``ModelDefinition.__init_subclass__``."""


# ── helpers ──────────────────────────────────────────────────────────────


def _build_predictors(
    target_var: str,
    base: Sequence[str],
    include: Sequence[str],
    exclude: Sequence[str],
) -> list[str]:
    """Assemble the final predictor list.

    1. Start from *base* (e.g. ``Predictors.DEFAULT_GAIN``).
    2. Remove *target_var* and all *exclude* vars in a single pass.
    3. Prepend any *include* vars not already present (and not equal to
       *target_var*).  Because *include* is applied after *exclude*, a
       variable listed in both will end up in the predictor set — i.e.
       *include* overrides *exclude*.
    """
    exclude_set = {target_var, *exclude}
    predictors = [p for p in base if p not in exclude_set]

    extra = [v for v in include if v not in predictors and v != target_var]
    predictors = extra + predictors

    return predictors


# ── base class ───────────────────────────────────────────────────────────


class ModelDefinition:
    """Abstract base for declarative model definitions.

    Subclasses set class-level attributes to configure the model. Any
    concrete class (one that sets ``model_id``) is auto-registered in
    ``MODELS`` via ``__init_subclass__``.
    """

    # ── required (must be set by concrete subclasses) ────────────────────

    model_id: ClassVar[str | None] = None
    """Short identifier, e.g. ``"lrp-rli-gbg-012"``. ``None`` for abstract bases."""

    target_var: ClassVar[str]
    """Column name of the target variable."""

    # ── optional overrides ───────────────────────────────────────────────

    description: ClassVar[str] = ""
    """Human-readable description for console output and reports."""

    include: ClassVar[Sequence[str]] = ()
    """Extra predictor vars to prepend to the base set."""

    exclude: ClassVar[Sequence[str]] = ()
    """Predictor vars to remove from the base set."""

    pipeline_cls: ClassVar[Any] = None
    """Pipeline class (e.g. ``LGBMPipeline``). Set lazily to avoid circular imports."""

    params: ClassVar[Mapping[str, Any]] = MappingProxyType({})
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

    shap_scatter_specs: ClassVar[Sequence[ShapScatterSpec]] = ()
    """Ordered list of SHAP scatter/dependence plot sets to generate for
    this model. Empty by default — set on concrete subclasses to declare
    the plots the model exploration needs."""

    random_seed: ClassVar[int] = 47

    variant_of: ClassVar[str | None] = None
    """If set, this model is a selection variant of another model."""

    notes: ClassVar[str] = ""
    """Free-text rationale persisted in ``config.json``."""

    # ── base predictor set (overridden by GainModel / LevelModel) ────────

    _base_predictors: ClassVar[Sequence[str]] = ()

    # ── auto-registration ────────────────────────────────────────────────

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.model_id is not None:
            MODELS[cls.model_id] = cls.to_config()

    # ── config generation ────────────────────────────────────────────────

    @classmethod
    def to_config(cls) -> ModelConfig:
        """Generate a ``ModelConfig`` for the pipeline."""
        predictors = _build_predictors(
            target_var=cls.target_var,
            base=cls._base_predictors,
            include=cls.include,
            exclude=cls.exclude,
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
        )


# ── convenience subclasses ───────────────────────────────────────────────


class GainModel(ModelDefinition):
    """Base for gain-prediction models.

    Automatically uses ``Predictors.DEFAULT_GAIN`` and prepends the base
    variable (target name with ``_gain`` suffix stripped).
    """

    _base_predictors: ClassVar[Sequence[str]] = Predictors.DEFAULT_GAIN

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # Auto-include the base variable (e.g. ewrswr for ewrswr_gain)
        if hasattr(cls, "target_var") and cls.target_var:
            base_var = cls.target_var.removesuffix("_gain")
            own_include = cls.__dict__.get("include")
            include = tuple(own_include if own_include is not None else cls.include or ())
            if base_var not in include:
                cls.include = (base_var, *include)
        super().__init_subclass__(**kwargs)


class LevelModel(ModelDefinition):
    """Base for level-prediction models.

    Automatically uses ``Predictors.DEFAULT_LEVEL``.
    """

    _base_predictors: ClassVar[Sequence[str]] = Predictors.DEFAULT_LEVEL
