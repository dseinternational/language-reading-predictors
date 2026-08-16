# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared, dependency-light contract for Byrne measurement sensitivities."""

UNRESOLVED_MEASURES = ("basspel", "woco", "basnum")
DENOMINATOR_FACTORS = (1, 2, 4)
MAX_MEDIAN_RANGE_FRACTION = 0.10
RLM_SENSITIVITY_WINDOWS = {
    "basspel": ((1, 2, 3), (4, 5)),
    "woco": ((1, 2, 3), (4, 5)),
    "basnum": ((1, 2, 3), (4,)),
}
