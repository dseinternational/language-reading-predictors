# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Block exposure calculations and summaries."""

from __future__ import annotations
import numpy as np
import xarray as xr
from dse_research_utils.statistics.evidence import (
    evidence_label,
)
from scipy.special import expit
from language_reading_predictors.statistical_models.posteriors import (
    band50,
)


def block_exposure_summary(
    trace: xr.DataTree,
    *,
    ci_prob: float,
    n_trials: int,
) -> dict[str, float | bool | str]:
    """Summarise the block-2 block-active exposure effect (kind="block_exposure").

    ``delta`` is the exposure effect on the logit scale; ``delta_items_*`` is the
    average marginal effect of toggling ``exposed`` 0 -> 1 across the fitted rows
    (per draw), times ``n_trials`` — the block-2 taught-word count attributable to
    block-2 being actively taught. This is an ASSOCIATION (parallel-trends), not a
    randomised effect (see :func:`factories.build_block_exposure_model`). Equal-tailed
    central intervals at coverage ``ci_prob``. Mirrors the ``delta`` block of
    :func:`did_summary` (the DiD sibling) but carries no ``beta_period`` — the
    per-timepoint ``alpha_time`` vector is the secular-trend anchor here.
    """
    posterior = trace.posterior
    lo_q = (1 - ci_prob) / 2
    hi_q = 1 - lo_q

    d = posterior["delta"].stack(sample=("chain", "draw")).values  # (S,)
    prob_pos = float(np.mean(d > 0))
    lo50, hi50 = band50(d)
    out: dict[str, float | bool | str] = {
        "delta_median": float(np.median(d)),
        "delta_mean": float(np.mean(d)),
        "delta_lo": float(np.quantile(d, lo_q)),
        "delta_hi": float(np.quantile(d, hi_q)),
        "delta_lo50": lo50,
        "delta_hi50": hi50,
        "prob_delta_pos": prob_pos,
        "delta_direction_label": evidence_label(prob_pos),
        "delta_favoured_direction": "positive" if prob_pos >= 0.5 else "negative",
        "delta_favoured_label": evidence_label(max(prob_pos, 1.0 - prob_pos)),
    }
    # Items-scale average marginal effect: toggle exposed 0 -> 1 per fitted row.
    eta_base = (
        posterior["eta_base"]
        .stack(sample=("chain", "draw"))
        .transpose("obs_id", "sample")
        .values
    )  # (n_obs, S)
    eff = (expit(eta_base + d[None, :]) - expit(eta_base)).mean(axis=0) * n_trials
    out["delta_items_median"] = float(np.median(eff))
    out["delta_items_mean"] = float(np.mean(eff))
    out["delta_items_lo"] = float(np.quantile(eff, lo_q))
    out["delta_items_hi"] = float(np.quantile(eff, hi_q))
    out["delta_items_lo50"], out["delta_items_hi50"] = band50(eff)
    return out
