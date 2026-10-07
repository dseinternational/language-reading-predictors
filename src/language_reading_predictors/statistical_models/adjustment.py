# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The adjustment set a fit actually conditioned on.

``ModelSpec.adjustment`` records requested terms. Family plans add terms, and
loaders remove covariates that are constant on the fitted rows.
:func:`effective_adjustment` records each fitted term's source, wave and
missingness role in ``config.json``, alongside requested terms that were dropped.
"""

from __future__ import annotations

from language_reading_predictors.statistical_models.context import ModelSpec


def effective_adjustment(
    spec: ModelSpec,
    prepared,
    *,
    measure_confounders: tuple[str, ...] = (),
    adjust_for: tuple[str, ...] = (),
    requested_adjust_for: tuple[str, ...] | None = None,
    ability_covariate: str | None = None,
    baseline_symbol: str | None = None,
    baseline_symbols: tuple[str, ...] = (),
    skill_baselines: tuple[str, ...] = (),
    descriptive_skills: tuple[str, ...] = (),
    moderator_symbol: str | None = None,
    moderator_is_covariate: bool = False,
    moderator_interaction: bool = False,
    exposure_terms: tuple[dict, ...] = (),
) -> dict:
    """Describe fitted terms and requested covariates dropped by the loader.

    ``skill_baselines`` enter at each period's pre wave. Bounded
    ``measure_confounders`` enter at its post wave. ``ability_covariate`` is the
    factor families' t1 ability measure, fitted as ``gamma_ability`` across waves.

    ``requested_adjust_for`` preserves the plan before constant covariates are
    removed. It defaults to ``adjust_for`` when the two sets are identical.

    Moderators have their own term kinds. A moderator main effect can supply the
    age adjustment instead of a separate ``gamma_A``. Calling a term a moderator
    does not imply that it controls confounding, especially if the exposure can
    cause it.
    """
    requested_adjust_for = adjust_for if requested_adjust_for is None else requested_adjust_for
    terms = []
    for s in skill_baselines:
        # Downstream skills declared as descriptive associates must not be
        # labelled as upstream adjustment terms in the causal graph.
        terms.append(
            {
                "term": f"{s}_pre",
                "kind": ("descriptive_associate" if s in descriptive_skills else "measure_baseline"),
                "source_column": prepared.column_map.get(s, s),
                "wave": "pre",
                "missing_indicator": False,
            }
        )
    for s in measure_confounders:
        if s == "G":
            # The randomised arm: time-invariant, not a wave-indexed measurement.
            terms.append(
                {
                    "term": "G",
                    "kind": "treatment",
                    "source_column": "group",
                    "wave": "time_invariant",
                    "missing_indicator": False,
                }
            )
        elif s == "A":
            # Age is read from the transition's pre row (age at the start of it).
            terms.append(
                {
                    "term": "A",
                    "kind": "covariate",
                    "source_column": "age",
                    "wave": "pre",
                    "missing_indicator": False,
                }
            )
        else:
            # Bounded-count measure confounders are taken at the POST wave,
            # contemporaneous with the exposure and the outcome.
            terms.append(
                {
                    "term": s,
                    "kind": "measure",
                    "source_column": prepared.column_map.get(s, s),
                    "wave": "post",
                    "missing_indicator": False,
                }
            )
    for c in adjust_for:
        terms.append(
            {
                "term": c,
                "kind": "covariate",
                "source_column": c,
                "wave": prepared.covariate_time.get(c, "unknown"),
                "missing_indicator": c.endswith("_missing"),
            }
        )
    if ability_covariate and ability_covariate in prepared.covariates:
        # A constant ability covariate belongs in dropped_constant, not fitted.
        terms.append(
            {
                "term": ability_covariate,
                "kind": "ability_covariate",
                "source_column": ability_covariate,
                "wave": prepared.covariate_time.get(ability_covariate, "baseline"),
                "missing_indicator": False,
            }
        )
    if baseline_symbol:
        terms.append(
            {
                "term": f"{baseline_symbol}_pre",
                "kind": "autoregressive_baseline",
                "source_column": prepared.column_map.get(baseline_symbol, baseline_symbol),
                "wave": "pre",
                "missing_indicator": False,
            }
        )
    for s in baseline_symbols:
        terms.append(
            {
                "term": f"{s}_pre",
                "kind": "autoregressive_baseline",
                "source_column": prepared.column_map.get(s, s),
                "wave": "pre",
                "missing_indicator": False,
            }
        )
    terms.extend(dict(term) for term in exposure_terms)
    if moderator_symbol:
        if moderator_symbol == "A":
            _source, _wave, _scale = "age", "pre", "standardised age"
        elif moderator_is_covariate:
            _source = moderator_symbol
            _wave = prepared.covariate_time.get(moderator_symbol, "unknown")
            _scale = "standardised raw covariate"
        else:
            _source = prepared.column_map.get(moderator_symbol, moderator_symbol)
            _wave, _scale = "post", "standardised logit of the post count"
        terms.append(
            {
                "term": "gamma_mod",
                "kind": "moderator_main_effect",
                "moderator": moderator_symbol,
                "source_column": _source,
                "wave": _wave,
                "scale": _scale,
                "missing_indicator": False,
            }
        )
        if moderator_interaction:
            terms.append(
                {
                    "term": "gamma_int",
                    "kind": "moderator_interaction",
                    "moderator": moderator_symbol,
                    "source_column": _source,
                    "wave": _wave,
                    "scale": f"standardised exposure x {_scale}",
                    "missing_indicator": False,
                }
            )
    return {
        "requested": list(spec.adjustment)
        + list(skill_baselines)
        + ([ability_covariate] if ability_covariate else [])
        + list(requested_adjust_for),
        "fitted": terms,
        "dropped_constant": list(prepared.dropped_covariates),
    }
