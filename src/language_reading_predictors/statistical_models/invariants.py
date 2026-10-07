# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Runtime invariant checks that survive ``python -O``.

Unlike ``assert``, these checks remain active under optimisation. This module
has no package dependencies, so callers need not import the sampling stack.
"""

from __future__ import annotations

from typing import TypeVar

__all__ = ["require_value"]

T = TypeVar("T")


def require_value(value: T | None, what: str) -> T:
    """Return ``value``, or raise ``ValueError`` naming what was missing.

    The narrowing replacement for ``assert value is not None``. ``what`` should
    name the setting and, where it is not obvious, the design that requires it —
    ``"predictor_slope_sigma (the levels design's regularising slope prior)"``
    reads usefully in a traceback; ``"value"`` does not.
    """

    if value is None:
        raise ValueError(f"{what} is required here but was not resolved")
    return value
