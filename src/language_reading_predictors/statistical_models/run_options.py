# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Immutable command-level options for one statistical-model run.

The fit CLI used to implement ``--target-accept`` by replacing
``dse_research_utils.statistics.models.sampling.get_sampling_configuration`` at
module scope.  That mutation leaked outside the requested fit and made the value
reaching a model depend on import order.  A context variable gives the unchanged
``fit(config)`` model-module API a scoped, concurrency-safe route to explicit run
options while the registry migrates model families incrementally.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True, slots=True)
class StatisticalRunOptions:
    """Command-level overrides that apply to exactly one model invocation."""

    target_accept: float | None = None

    def __post_init__(self) -> None:
        if self.target_accept is None:
            return
        value = float(self.target_accept)
        if not 0.0 < value < 1.0:
            raise ValueError(
                "target_accept must be in the open interval (0, 1); "
                f"got {self.target_accept!r}"
            )
        object.__setattr__(self, "target_accept", value)


_DEFAULT_OPTIONS = StatisticalRunOptions()
_CURRENT_OPTIONS: ContextVar[StatisticalRunOptions] = ContextVar(
    "statistical_model_run_options",
    default=_DEFAULT_OPTIONS,
)


def current_run_options() -> StatisticalRunOptions:
    """Return the immutable options scoped to the current model invocation."""

    return _CURRENT_OPTIONS.get()


@contextmanager
def use_run_options(
    options: StatisticalRunOptions,
) -> Iterator[StatisticalRunOptions]:
    """Scope ``options`` to one model call and restore the previous value."""

    token = _CURRENT_OPTIONS.set(options)
    try:
        yield options
    finally:
        _CURRENT_OPTIONS.reset(token)
