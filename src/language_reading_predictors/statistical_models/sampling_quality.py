# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""One correct way to read sampling-quality signals off a trace.

Every gate and every ad-hoc script needs the same four numbers — max R-hat, min ESS,
min per-chain BFMI, total divergences — and each one that re-derives them can get them
subtly wrong. Two traps, both observed in this repository:

**Rounding.** ``az.summary()`` rounds to ``rcParams["stats.round_to"]`` (``"2g"`` — two
significant figures) unless passed ``round_to="none"``, *the string*. ``round_to=None``
and ``"auto"`` both fall through to the rounded default. ArviZ 1.2 returned a
string-dtype frame when the argument was omitted; ArviZ 1.3 returns numeric columns,
but still applies the rounded default. Rounding erases exactly the digits the gate turns
on: across the whole gate-relevant range every
R-hat from 1.011 to 1.049 rounds to ``1.0`` and clears an ``R-hat <= 1.01`` test it
should fail (dseinternational/research#65; recurred twice — in ``loo_refit`` and in a
one-off prototype script, issue #440).

**Coercion.** ``trace.sample_stats["diverging"]`` is an xarray ``DataArray``; reduce it
via ``np.asarray(....values).sum()`` rather than relying on ``DataArray.__int__``, and
take BFMI from :func:`dse_research_utils.statistics.diagnostics._bfmi_per_chain`, since
``az.bfmi`` returns a ``DataTree`` in ArviZ 1.x that cannot be coerced to an array.

This module extracts the numbers and nothing else. It deliberately does **not** decide
whether a fit passed: the call sites differ in which variables they gate over and in how
they treat a missing BFMI, and those policies stay where they are rather than being
silently homogenised here.
"""

from __future__ import annotations

from dataclasses import dataclass

import arviz as az
import numpy as np
from dse_research_utils.statistics.diagnostics import _bfmi_per_chain

__all__ = ["SamplingQuality", "sampling_quality"]


@dataclass(frozen=True)
class SamplingQuality:
    """Unrounded sampling-quality signals for one trace."""

    max_rhat: float
    """Largest R-hat over the summarised variables (NaNs skipped)."""
    min_ess: float
    """Smallest of bulk-ESS and tail-ESS over the summarised variables."""
    min_bfmi: float | None
    """Smallest per-chain BFMI, or ``None`` when it cannot be computed."""
    n_divergences: int | None
    """Total divergent transitions, or ``None`` when ``sample_stats`` lacks them."""

    def summary_line(self) -> str:
        """One-line human-readable rendering for logs and prototype scripts."""
        bfmi = "n/a" if self.min_bfmi is None else f"{self.min_bfmi:.2f}"
        div = "n/a" if self.n_divergences is None else str(self.n_divergences)
        return (
            f"max R-hat {self.max_rhat:.4f}, min ESS {self.min_ess:.0f}, "
            f"min BFMI {bfmi}, divergences {div}"
        )


def sampling_quality(trace, *, var_names: list[str] | None = None) -> SamplingQuality:
    """Read the four sampling-quality signals off ``trace``, unrounded.

    Parameters
    ----------
    trace
        An ArviZ ``InferenceData`` (or DataTree-backed equivalent) with a ``posterior``
        group; ``sample_stats`` is used for divergences and BFMI when present.
    var_names
        Restrict the R-hat / ESS summary to these variables. ``None`` summarises
        everything ArviZ reports for the trace, which includes deterministics — pass the
        caller's curated gate variables when that matters.

    Returns
    -------
    SamplingQuality
        The extracted signals. Exceptions from ArviZ propagate; callers that must
        tolerate a failed diagnostic calculation should catch them and decide what an
        uncheckable fit means for them.
    """
    # ``round_to="none"`` must be the string — see the module docstring.
    summ = az.summary(trace, var_names=var_names, round_to="none", kind="diagnostics")
    # pandas ``.max()`` / ``.min()`` skip NaN by default, so a constant or unsampled
    # variable does not poison the reduction.
    max_rhat = float(summ["r_hat"].max())
    min_ess = float(min(summ["ess_bulk"].min(), summ["ess_tail"].min()))

    n_div: int | None = None
    sample_stats = getattr(trace, "sample_stats", None)
    if sample_stats is not None and "diverging" in sample_stats:
        n_div = int(np.asarray(sample_stats["diverging"].values).sum())

    bfmi = _bfmi_per_chain(trace)
    min_bfmi = (
        float(np.min(bfmi)) if bfmi is not None and np.all(np.isfinite(bfmi)) else None
    )

    return SamplingQuality(
        max_rhat=max_rhat,
        min_ess=min_ess,
        min_bfmi=min_bfmi,
        n_divergences=n_div,
    )
