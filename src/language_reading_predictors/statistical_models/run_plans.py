# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The small part of a family run plan that shared fit stages need."""

from typing import Any, Protocol


class ResolvedRunPlan(Protocol):
    """A validated design that can describe its settings and analysis recipe."""

    def as_dict(self) -> dict[str, Any]: ...

    def recipe_markdown(self, *, title: str) -> str: ...
