# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Collision-free confounder names for single-mediator outcome models."""

from __future__ import annotations


def outcome_confounder_coefficient(confounder_symbol: str) -> str:
    """Name an outcome-leg confounder without colliding with legacy ``b_W``.

    Every graded single-mediator model historically calls its outcome
    own-baseline coefficient ``b_W``, irrespective of the outcome symbol. A
    baseline word-reading confounder therefore uses ``b_conf_W``; all other
    confounders retain their established ``b_<symbol>`` names.
    """
    if confounder_symbol == "W":
        return "b_conf_W"
    return f"b_{confounder_symbol}"
