# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Plain-language findings for the survival family."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
import numpy as np
import pandas as pd
from language_reading_predictors.statistical_models.findings.common import (
    _KeyFindingsUnavailable,
    _kf_association_direction,
    _kf_csv,
    _kf_float,
    _kf_most_resolved_row,
    _kf_pct,
    _kf_plain_label,
    _kf_sentence,
)


def _kf_build_survival(output_dir: str | Path, config: Mapping) -> list[dict[str, str]]:
    """Discrete-time off-floor hazard model.

    Window-aware (2026-08-21 survival review, finding 1): under the default
    ``treatment_window="randomised"`` the headline tau is the randomised
    first-interval arm contrast and the later intervals are both-arms-treated
    hazards; a stored fit whose plan predates the field was fitted with the
    legacy pooled shift, whose direction beyond interval 1 is prior-mediated —
    the box says so rather than presenting it as data evidence. The ratio word
    follows the link (hazard ratio under cloglog, odds ratio under the logistic
    sensitivity link — finding 4).
    """
    df = _kf_csv(output_dir, "survival_summary.csv")
    if df is None:
        raise _KeyFindingsUnavailable("survival_summary.csv is not present")
    plan = config.get("resolved_run_plan") or {}
    window = str(plan.get("treatment_window", "pooled"))
    link = str(
        plan.get(
            "hazard_link",
            (config.get("extra") or {}).get("hazard_link", "cloglog"),
        )
    )
    ratio_word = "hazard ratio" if link == "cloglog" else "odds ratio"
    effects = df[np.isfinite(pd.to_numeric(df["P(>0)"], errors="coerce"))]
    if effects.empty:
        raise _KeyFindingsUnavailable("survival summary has no directional effects")
    treatment = effects[effects["term"].astype(str).str.startswith("tau")]
    row = (treatment.iloc[0].to_dict() if not treatment.empty else _kf_most_resolved_row(
        effects, prob_col="P(>0)"
    ))
    ratio = np.exp(_kf_float(row["median"]))
    ratio_lo = np.exp(_kf_float(row["ci_low"]))
    ratio_hi = np.exp(_kf_float(row["ci_high"]))
    label = _kf_plain_label(row["term"])
    scope = (
        "in the randomised first interval"
        if window == "randomised" and not treatment.empty
        else "in an interval"
    )
    if treatment.empty:
        causal_text = (
            "The reported covariate term is an adjusted association with movement "
            "off the floor. This summary contains no assignment contrast."
        )
    elif window == "randomised":
        causal_text = (
            "The contrast is a model-based, available-case modified-ITT "
            "assignment contrast in the randomised first interval among children "
            "at the floor at wave 1; later intervals pool both treated arms and "
            "carry no arm contrast. The baseline-subgroup restriction, the "
            "observed-wave-2 requirement, mean-imputed covariates and the "
            "hazard-model form qualify it, and no causal headline is released "
            "(#631 finding 11)."
        )
    else:
        causal_text = (
            "This pooled coefficient is prognostic, not a randomised effect of "
            "record: only the first interval carries an arm contrast, so its "
            "direction beyond that interval is set by the baseline-hazard priors "
            "rather than by observed comparisons."
        )
    sentences = [
        _kf_sentence(
            f"The {label} corresponded to a {ratio_word} of **{ratio:.2f}** "
            f"(89% credible range {ratio_lo:.2f} to {ratio_hi:.2f}) for coming "
            f"off the floor {scope}.",
            "headline",
        ),
        _kf_sentence(
            _kf_association_direction(
                row["P(>0)"],
                positive_claim="the reported term accompanies earlier movement off the floor",
                negative_claim="the reported term accompanies later movement off the floor",
            ),
            "confidence",
        ),
        _kf_sentence(causal_text, "causal"),
    ]
    terms = df["term"].astype(str)
    untreated = df[terms.str.startswith("baseline off-floor prob") & terms.str.contains(r"\(untreated\)", regex=True)]
    treated_cells = df[terms.str.startswith("off-floor prob") & terms.str.contains("both arms treated")]
    legacy_baseline = df[terms.str.startswith("baseline off-floor prob")]
    if window == "randomised" and not untreated.empty:
        first = _kf_float(untreated.iloc[0]["median"])
        text = (
            f"For an untreated child at mean covariates, the fitted first-interval "
            f"off-floor probability was {_kf_pct(first)}%."
        )
        if not treated_cells.empty:
            values = [_kf_float(v) for v in treated_cells["median"]]
            text += (
                f" With both arms treated, the fitted later-interval probabilities "
                f"ranged from {_kf_pct(min(values))}% to {_kf_pct(max(values))}%."
            )
        sentences.append(_kf_sentence(text, "highlight"))
    elif not legacy_baseline.empty:
        values = [_kf_float(v) for v in legacy_baseline["median"]]
        sentences.append(
            _kf_sentence(
                f"The fitted untreated off-floor probability ranged from "
                f"{_kf_pct(min(values))}% to {_kf_pct(max(values))}% across "
                f"intervals; beyond the first interval those untreated values are "
                f"prior-mediated extrapolations (no untreated children were "
                f"observed there).",
                "highlight",
            )
        )
    return sentences
