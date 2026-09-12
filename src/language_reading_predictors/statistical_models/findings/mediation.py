# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Plain-language findings for the mediation family."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
from dse_research_utils.statistics.evidence import (
    favoured_direction,
)
from language_reading_predictors.statistical_models.findings.common import (
    _KeyFindingsUnavailable,
    _kf_csv,
    _kf_float,
    _kf_pct,
    _kf_sentence,
)


def _kf_build_mediation(output_dir: str | Path, config: Mapping) -> list[dict[str, str]]:
    """One- or two-mediator g-formula decomposition."""
    df = _kf_csv(output_dir, "mediation_summary.csv")
    if df is None or "quantity" not in df.columns:
        raise _KeyFindingsUnavailable("mediation_summary.csv is not present")
    indexed = df.set_index("quantity")
    if "total" not in indexed.index:
        raise _KeyFindingsUnavailable("mediation_summary.csv has no total-effect row")
    total = indexed.loc["total"].to_dict()
    off_floor = str(total.get("off_floor", "false")).lower() in {"true", "1"}
    scale = 100.0 if off_floor else 1.0
    unit = "percentage points" if off_floor else "items"
    med = _kf_float(total["words_median"]) * scale
    lo = _kf_float(total["words_lo"]) * scale
    hi = _kf_float(total["words_hi"]) * scale
    fav = favoured_direction(_kf_float(total["prob_pos"]))
    positive = fav["favoured_direction"] == "positive"
    direction = "positive" if positive else "negative"
    claim = (
        "the intervention improves the outcome under the fitted model"
        if positive
        else "the intervention worsens the outcome under the fitted model"
    )
    sentences = [
        _kf_sentence(
            f"The model-based total intervention contrast was **{med:+.1f} "
            f"{unit}** (89% credible range {lo:+.1f} to {hi:+.1f}).",
            "headline",
        ),
        _kf_sentence(
            f"The posterior probability that this model-based total contrast is "
            f"{direction} is {_kf_pct(fav['favoured_direction_prob'])}% — "
            f"{fav['favoured_direction_label']} evidence that {claim}.",
            "confidence",
        ),
    ]
    indirect_name = next(
        (name for name in ("NIE_joint", "NIE", "IIE") if name in indexed.index),
        None,
    )
    if indirect_name is not None:
        indirect = indexed.loc[indirect_name].to_dict()
        i_med = _kf_float(indirect["words_median"]) * scale
        i_lo = _kf_float(indirect["words_lo"]) * scale
        i_hi = _kf_float(indirect["words_hi"]) * scale
        sentences.append(
            _kf_sentence(
                f"The estimated indirect component ({indirect_name}) was "
                f"{i_med:+.1f} {unit} (89% credible range {i_lo:+.1f} to "
                f"{i_hi:+.1f}).",
                "highlight",
            )
        )
    # Period-stacked fits standardise over ONE window: the only one holding both
    # arms. Say so, so the headline is not read as an all-period average (#585).
    unsupported = (config.get("extra") or {}).get("unsupported_periods") or []
    if unsupported:
        listed = ", ".join(str(period) for period in unsupported)
        sentences.append(
            _kf_sentence(
                "This contrast is averaged over the randomised first period only. "
                f"Period(s) {listed} contain no untreated children after the "
                "wait-list crossover, so an all-period average would extrapolate "
                "an untreated counterfactual the data cannot support.",
                "scale",
            )
        )
    sentences.append(
        _kf_sentence(
            "The direct/indirect split is a model-based g-formula decomposition, "
            "not an identified causal mediation effect: unmeasured "
            "mediator-outcome confounding remains a binding assumption.",
            "causal",
        )
    )
    return sentences
