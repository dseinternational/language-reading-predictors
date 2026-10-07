# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Discover convention-named models without importing their Bayesian graphs.

Discovery selects filenames matching ``_MODEL_MODULE`` and converts underscores
to hyphens for CLI IDs. Loading then verifies that the module defines its own
``fit(config)`` callable. Imported ``fit`` symbols do not qualify.

Using filenames avoids importing ``SPEC`` during discovery and also supports
modules that build a specification lazily. Fit CLIs resolve legacy aliases
through ``model_ids``.
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
            raise TypeError(f"{self.module_name} does not define its own top-level fit(config)")
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
