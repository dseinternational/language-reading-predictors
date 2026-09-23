# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Boosting-objective helpers shared by the tuner and the tests.

The gradient-boosting models use LightGBM's Huber objective (adopted
2026-09-22 after the objective-sensitivity check recorded in
``notes/202609221700-gb-objective-sensitivity.md``, superseding the #169 MAE
policy). Huber loss is quadratic for residuals within a threshold ``delta``
and linear beyond it. Its population target is a Huber location functional,
which need not equal the conditional mean for skewed or floored outcomes.
Clipping the residual gradient limits one source of influence; it does not
guarantee that extreme observations cannot affect the fitted trees.

The threshold is a per-model constant derived from the target's spread, not a
tuned hyperparameter. ``delta = 1.345 * sigma_hat`` where ``sigma_hat`` is the
normal-consistent MAD estimate ``1.4826 * MAD(y)``; the 1.345 factor is the
classical Huber tuning constant giving 95% asymptotic efficiency at the normal
(Huber, 1964, DOI 10.1214/aoms/1177703732). Heavily floored targets can have
``MAD(y) = 0`` (more than half the rows at the floor); the rule then falls back
to the mean absolute deviation from the median, which is positive whenever the
target varies at all.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt

HUBER_TUNING_CONSTANT = 1.345
"""Huber's ``k`` giving 95% efficiency at the normal distribution."""

MAD_TO_SIGMA = 1.4826
"""Scale factor making the MAD a consistent estimate of sigma at the normal."""

ROBUST_MAD_RULE = "robust-mad"
"""Name recorded in ``best_params.json`` / ``config.json`` for this rule."""


@dataclass(frozen=True)
class HuberDelta:
    """A derived Huber threshold and the statistics it came from."""

    delta: float
    rule: str
    scale_source: str
    """``"mad"`` when the MAD was positive, ``"mean_abs_dev"`` for the fallback."""
    median: float
    mad: float
    mean_abs_dev: float
    n_rows: int

    def as_dict(self) -> dict[str, float | int | str]:
        return dict(asdict(self))


def robust_huber_delta(y: npt.ArrayLike) -> HuberDelta:
    """Derive the Huber threshold ``1.345 * 1.4826 * MAD(y)`` with a floor fallback.

    Parameters
    ----------
    y
        The target values the model is tuned on (after the model's own row
        filters), with no missing values.

    Raises
    ------
    ValueError
        When ``y`` is empty, contains non-finite values or does not vary, so
        no positive threshold exists.
    """
    arr: npt.NDArray[np.float64] = np.asarray(y, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("Huber threshold needs at least one target value")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Huber threshold needs finite target values")
    median = float(np.median(arr))
    abs_dev = np.abs(arr - median)
    mad = float(np.median(abs_dev))
    mean_abs_dev = float(np.mean(abs_dev))
    if mad > 0:
        delta = HUBER_TUNING_CONSTANT * MAD_TO_SIGMA * mad
        source = "mad"
    elif mean_abs_dev > 0:
        delta = HUBER_TUNING_CONSTANT * mean_abs_dev
        source = "mean_abs_dev"
    else:
        raise ValueError("Huber threshold needs a target that varies")
    return HuberDelta(
        delta=float(delta),
        rule=ROBUST_MAD_RULE,
        scale_source=source,
        median=median,
        mad=mad,
        mean_abs_dev=mean_abs_dev,
        n_rows=int(arr.size),
    )
