# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Lazy discovery of runnable statistical-model modules (#165, #361 Phase 3).

Replaces the hand-maintained import block + ``MODELS`` dict in
``scripts/fit_statistical_model.py``. Any submodule of
``language_reading_predictors.statistical_models`` that defines its **own**
top-level ``fit(config)`` function is a runnable model, registered under its
**canonical CLI id** — the module name (canonical underscore form since #168
Phase 2, e.g. ``lrp_rli_itt_001``) rewritten with hyphens (``lrp-rli-itt-001``).
Adding a new model is then just dropping in a new ``lrp_.../rlm...`` module - no
registry edit. Legacy ids (``lrpitt01``) still resolve: the fit CLIs build a
legacy-alias index over these keys via ``model_ids``.

Keying by module name (not ``SPEC.model_id``) is deliberate: most modules expose a
module-level ``SPEC``, but some (e.g. the adjusted model) build their spec lazily
via ``get_spec()`` so the DAG-only path imports without the Bayesian stack. Every
runnable model does, however, define a top-level ``fit`` - that is the invariant
we discover on. The ``fit.__module__ == module`` check ignores any ``fit`` symbol
merely imported into an infrastructure module.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from language_reading_predictors import statistical_models as _pkg
from language_reading_predictors.statistical_models.run_options import (
    StatisticalRunOptions,
    use_run_options,
)


_MODEL_MODULE = re.compile(r"^lrp_(?:rli|rlm)_[a-z0-9_]+_\d{3}[a-z]?$")


def _defines_fit(mod: ModuleType) -> bool:
    """True if ``mod`` defines its own top-level ``fit`` callable."""
    fn = getattr(mod, "fit", None)
    return callable(fn) and getattr(fn, "__module__", "") == mod.__name__


@dataclass(frozen=True, slots=True)
class LazyModel:
    """A lightweight manifest entry that imports its model only when used."""

    model_id: str
    module_name: str

    def load(self) -> ModuleType:
        """Import and validate the referenced runnable model module."""

        module = importlib.import_module(self.module_name)
        if not _defines_fit(module):
            raise TypeError(
                f"{self.module_name} does not define its own top-level fit(config)"
            )
        return module

    def fit(
        self,
        config: str = "dev",
        *,
        options: StatisticalRunOptions | None = None,
    ) -> Any:
        """Load and fit the model with options scoped to this invocation."""

        effective = options or StatisticalRunOptions()
        with use_run_options(effective):
            return self.load().fit(config)

    def __getattr__(self, name: str) -> Any:
        """Preserve module-like access for callers that inspect ``SPEC``."""

        return getattr(self.load(), name)


def discover_models() -> dict[str, LazyModel]:
    """Return a sorted lazy import map for every convention-named model.

    Discovery reads package filenames only.  Importing and validating a module is
    deferred until its ``fit`` method or an attribute such as ``SPEC`` is accessed.
    This keeps a CLI help/error path independent of the full PyMC model graph.
    """
    models: dict[str, LazyModel] = {}
    for info in pkgutil.iter_modules(_pkg.__path__):
        if info.ispkg or _MODEL_MODULE.fullmatch(info.name) is None:
            continue
        model_id = info.name.replace("_", "-")
        models[model_id] = LazyModel(
            model_id=model_id,
            module_name=f"{_pkg.__name__}.{info.name}",
        )
    return dict(sorted(models.items()))
