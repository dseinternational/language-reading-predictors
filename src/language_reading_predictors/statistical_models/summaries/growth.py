# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Growth calculations and summaries."""

from __future__ import annotations


import numpy as np
import pandas as pd
import xarray as xr
from dse_research_utils.statistics.evidence import (
    evidence_label,
    favoured_direction,
)
from dse_research_utils.statistics.intervals import eti_bands


def growth_association_summary(
    trace: xr.DataTree,
    *,
    coefs: tuple[str, ...] = ("gamma", "delta", "beta", "loading"),
) -> pd.DataFrame:
    """Per-(coefficient, outcome) posterior summary for the growth models (LRP69/70).

    One row per element of each vector coefficient in ``coefs`` (each carries the
    ``outcome`` dim): the posterior **median**, the fixed 50 / 89 equal-tailed bands
    (:func:`eti_bands`, #177), ``prob_positive`` = ``P(coef > 0)`` and the
    evidence-language fields (:func:`favoured_direction`, #179).

    ``gamma`` (baseline non-verbal ability -> growth *rate*) is the headline Q5
    estimand; ``delta`` is the association with the level at the pooled-mean
    (mid-study) age — ``age_std`` is standardised over all child-wave cells, so
    the entry-level association is ``delta + gamma * E[age_std at t1]``, not
    ``delta``; ``beta`` is the mean slope (trajectory characterisation);
    ``loading`` is the shared growth-tempo loading present only in the factor
    model (LRP70) and skipped otherwise. The interaction model (LRP85) passes
    ``gamma_age``/``gamma_int`` in ``coefs`` so its registered headline reaches
    this summary (2026-08-21 review, finding 1). Every
    row is an **adjusted association** (``role`` fixed to ``"association"``): under
    the locked DAG these non-randomised, latent-GA-confounded terms are never read
    as "drives". The table reports fixed 50% and 89% equal-tailed intervals.
    """
    posterior = trace.posterior
    rows: list[dict[str, object]] = []
    for coef in coefs:
        if coef not in posterior:
            continue
        da = posterior[coef]
        outcome_dim = "outcome" if "outcome" in da.dims else None
        labels = list(da[outcome_dim].values) if outcome_dim else [coef]
        for lab in labels:
            sub = da.sel({outcome_dim: lab}) if outcome_dim else da
            group_dim = next(
                (name for name in ("reading_group", "group") if name in sub.dims),
                None,
            )
            groups: list[object | None] = (
                list(sub.coords[group_dim].values) if group_dim is not None else [None]
            )
            for group in groups:
                cell = (
                    sub.sel({group_dim: group})
                    if group is not None and group_dim is not None
                    else sub
                )
                d = cell.stack(sample=("chain", "draw")).values.ravel()
                prob_pos = float(np.mean(d > 0))
                rows.append(
                    {
                        "coefficient": coef,
                        "outcome": str(lab),
                        "group": group,
                        "role": "association",
                        "median": float(np.median(d)),
                        "prob_positive": prob_pos,
                        "direction_label": evidence_label(prob_pos),
                        **eti_bands(d, probs=(0.5, 0.89)),
                        **favoured_direction(prob_pos),
                    }
                )
    return pd.DataFrame(rows)
