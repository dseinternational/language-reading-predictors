# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""One publication rule for live and persisted new-child PSIS evidence."""

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

VALIDATION_SCHEMA_VERSION = 3


def _true(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _number(row: Mapping[str, Any], name: str) -> float:
    value = row.get(name)
    if value is None or isinstance(value, bool):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return float("nan")


@dataclass(frozen=True)
class NewChildVerdict:
    diagnostics_valid: bool
    integration_reliable: bool
    reliable: bool
    reasons: tuple[str, ...]


def new_child_validation_verdict(row: Mapping[str, Any]) -> NewChildVerdict:
    """Recheck scalar evidence, including CSV values, without trusting verdict flags.

    The writer checks the shape and finiteness of the pointwise vectors. Its
    explicit pointwise flag is required alongside the scalar evidence here.
    Split-score disagreement is a stability check, not an error bound. The raw
    maximum log-likelihood disagreement is reported but has no ELPD threshold.
    """
    reasons: list[str] = []
    current = _number(row, "validation_schema_version") == VALIDATION_SCHEMA_VERSION
    if not current:
        reasons.append("the current validation schema has not been recorded")
    fields = ("elpd", "elpd_se", "p_loo", "max_pareto_k", "good_k_threshold",
              "n_children", "n_unreliable", "posterior_draws_used")
    values = {name: _number(row, name) for name in fields}
    finite = all(math.isfinite(value) for value in values.values())
    counts = ("n_children", "n_unreliable", "posterior_draws_used")
    diagnostics = bool(
        current and finite and _true(row.get("pointwise_diagnostics_valid"))
        and all(values[name].is_integer() for name in counts)
        and values["n_children"] > 0 and values["posterior_draws_used"] > 0
        and 0 <= values["n_unreliable"] <= values["n_children"]
        and values["elpd_se"] >= 0 and 0 < values["good_k_threshold"] <= 1
    )
    if not diagnostics:
        reasons.append("diagnostic evidence is missing, invalid or incomplete")
    pareto_ok = diagnostics and values["n_unreliable"] == 0 and values["max_pareto_k"] <= values["good_k_threshold"]
    if diagnostics and not pareto_ok:
        reasons.append("full-batch Pareto diagnostics exceed the reliability threshold")
    raw_error = _number(row, "latent_mc_half_split_error")
    latents = row.get("latents_redrawn")
    latent_declared = isinstance(latents, str) and bool(latents.strip())
    integration = diagnostics and latent_declared and math.isfinite(raw_error) and raw_error >= 0
    if latents != "(none)":
        score_error = _number(row, "latent_mc_elpd_error")
        draws = _number(row, "n_latent_draws")
        integration = bool(
            integration and math.isfinite(draws) and draws.is_integer() and draws >= 2
            and _true(row.get("integration_batches_reliable"))
            and math.isfinite(score_error) and 0 <= score_error <= values["elpd_se"]
        )
    if not integration:
        reasons.append("integration evidence is invalid or split-score stability has not passed")
    reliable = bool(pareto_ok and integration)
    for name, computed in (("diagnostics_valid", diagnostics), ("integration_reliable", integration), ("reliable", reliable)):
        if name in row and _true(row[name]) != computed:
            reasons.append("the stored verdict disagrees with its evidence")
            reliable = False
            break
    return NewChildVerdict(diagnostics, bool(integration), reliable, tuple(reasons))
