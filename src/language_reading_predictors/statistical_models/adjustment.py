# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The adjustment set a fit actually conditioned on.

``ModelSpec.adjustment`` records what was *requested*. What is fitted can differ:
``spec.extra["adjust_for"]`` adds revised-DAG confounders, the factor families add
an ability covariate and upstream-skill baselines, and the loader drops any
covariate that turns out constant on the fitted rows. :func:`effective_adjustment`
builds the record that goes into ``config.json`` naming, for every term that
carries a coefficient, its source column, wave and missingness role — plus the
requested-but-dropped terms, explicitly (#258 review P1). Shared by the factor
families and the mechanism family, so it sits below both (#394 step 6).
"""

from __future__ import annotations

from language_reading_predictors.statistical_models.context import ModelSpec


def effective_adjustment(
    spec: ModelSpec,
    prepared,
    *,
    measure_confounders: tuple[str, ...] = (),
    adjust_for: tuple[str, ...] = (),
    ability_covariate: str | None = None,
    baseline_symbol: str | None = None,
    skill_baselines: tuple[str, ...] = (),
) -> dict:
    """Describe the adjustment set the model **actually fitted**.

    ``spec.adjustment`` records what was *requested*; it is not what is fitted.
    ``ModelSpec.extra["adjust_for"]`` never reached ``config.json`` at all, so a
    model could report ``{G, A, W_pre}`` while conditioning on hearing, speech,
    sessions and their missingness indicators — a material misdescription that made
    exact auditing impossible (#258 review, P1). And a covariate that turns out
    constant on the fitted rows is dropped by the loader and gets no coefficient, so
    listing it would imply a term that was never estimated.

    The returned record therefore names, for every term that carries a coefficient,
    its source column, its measurement wave, and whether it is a missingness
    indicator — plus the requested-but-dropped terms, explicitly.

    ``skill_baselines`` records the gain-factor ``skill_symbols``, which — unlike the
    ``measure_confounders`` of the mechanism/mediation families — enter at the period
    **pre** (baseline) wave, not the post wave (#247). They are always fitted (the
    keep-mask requires their baselines), so they never appear in ``dropped_constant``.

    ``ability_covariate`` records the gain-/level-factor cognitive-ability adjuster
    (block design), a between-child t1 baseline broadcast across the panel and fitted
    as ``gamma_ability``. It was previously absent from the record even though the
    factory conditions on it, so the audited set understated the fitted set by one
    term across the whole factor family (this review's finding B2).
    """
    terms = []
    for s in skill_baselines:
        # Upstream-skill DAG-parent adjusters, entered as their period baseline
        # (pre-wave) logit — the ANCOVA lag that precedes that period's treatment.
        terms.append(
            {
                "term": f"{s}_pre",
                "kind": "measure_baseline",
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
        # Cognitive-ability (block-design) adjuster — a between-child t1 baseline
        # broadcast across the panel, fitted as ``gamma_ability``. Guarded on
        # presence so an ability covariate that went constant (and was dropped by
        # the loader) is reported under ``dropped_constant``, not as fitted.
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
    return {
        "requested": list(spec.adjustment)
        + list(skill_baselines)
        + ([ability_covariate] if ability_covariate else [])
        + list(spec.extra.get("adjust_for", ())),
        "fitted": terms,
        "dropped_constant": list(prepared.dropped_covariates),
    }
