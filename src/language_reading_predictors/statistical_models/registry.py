# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Auto-discovery of runnable statistical-model modules (#165).

Replaces the hand-maintained import block + ``MODELS`` dict in
``scripts/fit_statistical_model.py``. Any submodule of
``language_reading_predictors.statistical_models`` that defines its **own**
top-level ``fit(config)`` function is a runnable model, registered under its
module name (which is its CLI id / ``model_id``). Adding a new model is then just
dropping in a new ``lrp.../rlm...`` module - no registry edit.

Keying by module name (not ``SPEC.model_id``) is deliberate: most modules expose a
module-level ``SPEC``, but some (e.g. ``lrp65``) build their spec lazily via
``get_spec()`` so the DAG-only path imports without the Bayesian stack. Every
runnable model does, however, define a top-level ``fit`` - that is the invariant
we discover on. The ``fit.__module__ == module`` check ignores any ``fit`` symbol
merely imported into an infrastructure module.
"""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType

from language_reading_predictors import statistical_models as _pkg


def _defines_fit(mod: ModuleType) -> bool:
    """True if ``mod`` defines its own top-level ``fit`` callable."""
    fn = getattr(mod, "fit", None)
    return callable(fn) and getattr(fn, "__module__", "") == mod.__name__


def discover_models() -> dict[str, ModuleType]:
    """Return ``{model_id: module}`` for every discoverable model, sorted by id.

    A model module is any (non-package) submodule of the statistical-models
    package that defines its own top-level ``fit(config)`` - e.g. ``lrpitt01``,
    ``lrp65`` (lazy spec), ``rlmhg01`` (historical growth). Infrastructure modules
    (``context``, ``factories``, ``pipeline``, ``reporting`` ...) are digit-free and
    define no top-level ``fit``, so they are skipped without being imported.
    """
    models: dict[str, ModuleType] = {}
    for info in pkgutil.iter_modules(_pkg.__path__):
        # Model ids carry a number (``lrpitt01``, ``rlmhg01`` ...); skipping the
        # digit-free infrastructure modules (``context``, ``factories``,
        # ``pipeline`` ...) avoids importing them — and any heavy import side
        # effects — just to discover on ``fit``. The ``_defines_fit`` check below
        # still guards against a numbered module that is not a runnable model.
        if info.ispkg or not any(ch.isdigit() for ch in info.name):
            continue
        mod = importlib.import_module(f"{_pkg.__name__}.{info.name}")
        if _defines_fit(mod):
            models[info.name] = mod
    return dict(sorted(models.items()))
