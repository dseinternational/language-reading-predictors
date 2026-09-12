# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Plain-language findings for the pooled levels family."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
from language_reading_predictors.statistical_models.findings.common import (
    _KeyFindingsUnavailable,
    _kf_csv,
    _kf_float,
    _kf_measure_label,
)


def _kf_build_pooled_levels(output_dir: str | Path, config: Mapping) -> list[dict[str, str]]:
    """Key findings for the wave-pooled level family.

    The headline is the *decomposition*, not a single slope: a between-child
    coefficient beside a within-child one, because the whole reason the family
    exists is that a random-intercept model with one exposure coefficient returns
    an uninterpretable blend of the two.
    """
    table = _kf_csv(output_dir, "pooled_levels_summary.csv")
    if table is None:
        raise _KeyFindingsUnavailable("pooled_levels_summary.csv is not present")
    plan = config.get("resolved_run_plan") or {}
    rows = {str(r["term"]): r for _, r in table.iterrows()}
    outcome = _kf_measure_label(plan.get("outcome_symbol"))
    exposure = _kf_measure_label(plan.get("mechanism_symbol"))
    # A raw-score covariate exposure (#553) is read in its own units: the fit
    # records how many raw points one SD of the fitted exposure is.
    extra = config.get("extra") or {}
    sd_raw = extra.get("mechanism_exposure_sd_raw")
    if bool(plan.get("mechanism_is_covariate", False)) and sd_raw is not None:
        try:
            unit = f"1 SD ≈ {float(sd_raw):.1f} raw points"
        except (TypeError, ValueError):
            unit = None
        if unit is not None:
            exposure = (
                f"{exposure[:-1]}; {unit})" if exposure.endswith(")") else f"{exposure} ({unit})"
            )

    sentences: list[dict[str, str]] = []
    between = rows.get("beta_between")
    within = rows.get("beta_within")
    if between is None:
        blended = rows.get("beta_mech")
        if blended is None:
            raise _KeyFindingsUnavailable("no exposure coefficient in the summary")
        sentences.append(
            {
                "text": (
                    f"Pooled across waves, a 1 SD higher {exposure} level goes with a "
                    f"**{_kf_float(blended['median']):+.2f}** logit difference in "
                    f"{outcome} (89% {_kf_float(blended['lo']):+.2f} to "
                    f"{_kf_float(blended['hi']):+.2f}). This fit does not separate the "
                    "between-child from the within-child association."
                ),
                "kind": "headline",
            }
        )
        return sentences

    sentences.append(
        {
            "text": (
                f"**Between children**, those sitting 1 SD higher on {exposure} across "
                f"the study sit **{_kf_float(between['median']):+.2f}** logit higher on "
                f"{outcome} (89% {_kf_float(between['lo']):+.2f} to "
                f"{_kf_float(between['hi']):+.2f}; "
                f"P(> 0) = {_kf_float(between['prob_positive']):.3f})."
            ),
            "kind": "headline",
        }
    )
    if within is not None:
        sentences.append(
            {
                "text": (
                    f"**Within a child**, at the waves where they are 1 SD above their "
                    f"own {exposure} average, {outcome} is "
                    f"**{_kf_float(within['median']):+.2f}** logit above their own "
                    f"average (89% {_kf_float(within['lo']):+.2f} to "
                    f"{_kf_float(within['hi']):+.2f}; "
                    f"P(> 0) = {_kf_float(within['prob_positive']):.3f})."
                ),
                "kind": "confidence",
            }
        )
        sentences.append(
            {
                "text": (
                    "The two are different questions. A large between-child coefficient "
                    "beside a small within-child one places the association in stable "
                    "differences between children rather than in a child's own "
                    "movement — the pattern a shared-cause account predicts."
                ),
                "kind": "highlight",
            }
        )
    skills = [str(sk) for sk in (plan.get("skill_symbols") or [])]
    sentences.append(
        {
            "text": (
                "Exposure and outcome are measured at the same wave, so nothing here "
                "orders them in time. Every term is an adjusted association, not a "
                "causal effect."
                + (
                    " The model also holds fixed the same-wave levels of "
                    + ", ".join(_kf_measure_label(sk) for sk in skills)
                    + " — contemporaneous skills that may themselves be affected by "
                    "the intervention, so their coefficients are associations too."
                    if skills
                    else ""
                )
            ),
            "kind": "causal",
        }
    )
    return sentences
