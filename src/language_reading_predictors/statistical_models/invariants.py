# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Runtime invariant checks that survive ``python -O`` (#637 stage 4).

``assert`` is removed by the optimiser. Every use of it to narrow a resolved run
plan's optional field — ``assert plan.post_time is not None`` and its sixteen
siblings — therefore did nothing under ``-O`` except let a ``None`` travel one
statement further, into an index or an arithmetic expression that fails with a
message about the *symptom* rather than the missing setting.

``runtime.require_spec`` already made this point for ``ModelSpec``; the helpers
here make it available to the run plans, panels and payloads, and live in a module
with no package dependencies so a factory or a release check can use them without
importing the sampling stack.
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
