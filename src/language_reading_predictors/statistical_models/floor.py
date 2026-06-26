# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pre-specified floor rule for heavily-floored ITT outcomes (issue #119).

Two RCT-phase outcomes — phonetic spelling (``P``) and nonword reading (``N``) —
are heavily floored at t2: most children still score zero, so a graded
Beta-Binomial treatment effect is leveraged by a handful of dispersed tail values
rather than by the arm contrast. The suite handles them with a rule fixed
*before* fitting (so the choice reads as pre-specification, not estimand-shopping):
an outcome with at least :data:`FLOOR_THRESHOLD` of its post-scores at zero
(computed **arm-blind**)

1. drops the (degenerate) own-baseline precision term and uses an age-only
   predictor ``alpha + tau*G + gamma_A*A_std``; and
2. reports a binary "off-floor at t2" estimand — ``Pr(post > 0)`` via a
   Bernoulli/logistic ``tau`` — as the PRIMARY result, retaining the graded
   Beta-Binomial ``tau`` only as a flagged, detection-limited SECONDARY.

See ``notes/202606251124-lrpitt-floored-outcomes-nonword-spelling.md`` and the
"Floored outcomes" section of issue #119.
"""

from __future__ import annotations

import numpy as np

from language_reading_predictors.statistical_models.preprocessing import PreparedData

FLOOR_THRESHOLD: float = 0.40
"""An outcome is "floored" if at least this fraction of its post-scores are zero
at t2 (applied arm-blind, pre-specified before fitting)."""


def proportion_at_zero(prepared: PreparedData, symbol: str) -> float:
    """Fraction of (non-missing) post-scores equal to zero for ``symbol``.

    Computed **arm-blind** (pooling both groups) over ``prepared.post_counts``.
    For ``phase_mode="itt"`` these are the t2 post-scores. Returns ``nan`` if
    there are no non-missing post-scores.
    """
    if symbol not in prepared.post_counts:
        raise KeyError(f"{symbol!r} not in prepared.post_counts")
    post = np.asarray(prepared.post_counts[symbol], dtype=float)
    finite = post[np.isfinite(post)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite == 0.0))


def is_floored(
    prepared: PreparedData,
    symbol: str,
    threshold: float = FLOOR_THRESHOLD,
) -> bool:
    """Apply the pre-specified floor rule: ``proportion_at_zero >= threshold``."""
    p0 = proportion_at_zero(prepared, symbol)
    return bool(np.isfinite(p0) and p0 >= threshold)
