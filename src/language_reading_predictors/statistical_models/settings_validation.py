# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Validate declared Boolean dataclass fields without changing their types.

Strings and integers are rejected. None is allowed only for optional Boolean fields.
"""

from __future__ import annotations

import typing
from dataclasses import fields
from functools import lru_cache
from typing import Any

__all__ = ["boolean_fields", "require_declared_booleans"]


@lru_cache(maxsize=None)
def boolean_fields(cls: type) -> tuple[tuple[str, bool], ...]:
    """``(name, optional)`` for every field ``cls`` annotates as Boolean.

    ``optional`` is True for ``bool | None`` — the one declared type for which
    ``None`` is a legitimate value rather than an unset flag. A union that mixes
    ``bool`` with anything else is deliberately *not* treated as a Boolean field:
    its own class knows what the other members mean.

    Resolution failures are not swallowed. These are module-level dataclasses in
    this package; an annotation that cannot be resolved is a defect in the class,
    not a reason to stop checking it.
    """

    hints = typing.get_type_hints(cls)
    resolved: list[tuple[str, bool]] = []
    for field in fields(cls):
        hint = hints.get(field.name, field.type)
        if hint is bool:
            resolved.append((field.name, False))
            continue
        args = set(typing.get_args(hint))
        if bool in args and args <= {bool, type(None)}:
            resolved.append((field.name, True))
    return tuple(resolved)


def require_declared_booleans(settings: Any) -> None:
    """Raise ``TypeError`` unless every declared Boolean field holds a Boolean."""

    for name, optional in boolean_fields(type(settings)):
        value = getattr(settings, name)
        if optional and value is None:
            continue
        if not isinstance(value, bool):
            suffix = " or None" if optional else ""
            raise TypeError(f"{name} must be a boolean{suffix}, got {value!r}")
