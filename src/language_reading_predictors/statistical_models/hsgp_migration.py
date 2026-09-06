# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Identify fits that require the corrected HSGP approximation (#660)."""

from collections.abc import Mapping
from typing import Any

HSGP_BASIS_VERSION = "midpoint-half-range-v1"


def uses_hsgp(config: Mapping[str, Any]) -> bool:
    """Read recorded settings, including the pre-typed legacy declarations."""
    kind = config.get("kind")
    if kind not in {"itt", "joint", "mechanism"}:
        return False
    settings = config.get("resolved_run_plan") or config.get("model_settings") or config.get("spec_extra") or {}
    if not isinstance(settings, Mapping):
        return True  # The stored declaration cannot establish that no GP was fitted.
    if kind == "mechanism" and settings.get("linear_mechanism") is not True:
        return True
    return any(settings.get(key) is True for key in ("use_age_gp", "use_own_baseline_gp", "use_varying_tau"))


def hsgp_refit_pending(config: Mapping[str, Any]) -> bool:
    """Older fits remain pending until a fresh fit records the corrected basis."""
    return uses_hsgp(config) and config.get("hsgp_basis_version") != HSGP_BASIS_VERSION
