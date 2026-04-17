# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model registry — shared defaults and the global ``MODELS`` dict.

``MODELS`` is populated automatically when model-definition classes
(subclasses of ``ModelDefinition``) are imported. This module re-exports
the dict from ``base_model`` so downstream code can still do::

    from language_reading_predictors.models.registry import MODELS
"""

from language_reading_predictors.models.base_model import MODELS

__all__ = ["MODELS"]
