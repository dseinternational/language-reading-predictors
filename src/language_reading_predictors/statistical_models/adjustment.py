# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The adjustment set a fit actually conditioned on.

``ModelSpec.adjustment`` records what was *requested*. What is fitted can differ:
Family run plans add revised-DAG confounders, the factor families add an ability
covariate and upstream-skill baselines, and the loader drops any
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
    requested_adjust_for: tuple[str, ...] | None = None,
    ability_covariate: str | None = None,
    baseline_symbol: str | None = None,
    baseline_symbols: tuple[str, ...] = (),
    skill_baselines: tuple[str, ...] = (),
    moderator_symbol: str | None = None,
    moderator_is_covariate: bool = False,
    moderator_interaction: bool = False,
) -> dict:
    """Describe the adjustment set the model **actually fitted**.

    ``spec.adjustment`` records what was *requested*; it is not what is fitted.
    Family-declared ``adjust_for`` values once failed to reach ``config.json``, so
    a model could report ``{G, A, W_pre}`` while conditioning on hearing, speech,
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

    ``requested_adjust_for`` is the plan declaration before the loader removes a
    constant covariate. It defaults to ``adjust_for`` for callers whose fitted and
    requested sets are identical.

    ``moderator_symbol`` records the mechanism family's linear moderation terms. The
    factory always fits a moderator main effect (``gamma_mod``) and, when
    ``moderator_interaction``, an exposure-by-moderator product (``gamma_int``);
    neither reached the record, so a fit could name a term in ``requested`` and omit
    it from ``fitted`` while estimating a coefficient for it (#586 finding 9). The
    clearest case is age moderation (mech-073), where the factory deliberately drops
    the separate linear ``gamma_A`` because ``gamma_mod`` *is* the age adjustment —
    so age was listed as requested, absent from the fitted terms, and adjusted for
    all along. They are recorded under their own ``moderator*`` kinds, never
    relabelled as confounders: a moderator that descends from the exposure is not a
    backdoor adjuster, and the record must not imply that it is.
    """
    requested_adjust_for = (
        adjust_for if requested_adjust_for is None else requested_adjust_for
    )
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
    for s in baseline_symbols:
        # Multi-outcome ANCOVA (the joint-mechanism transition design): each
        # jointly fitted outcome keeps its own autoregressive baseline, so the
        # record carries one term per outcome rather than the singular
        # ``baseline_symbol``.
        terms.append(
            {
                "term": f"{s}_pre",
                "kind": "autoregressive_baseline",
                "source_column": prepared.column_map.get(s, s),
                "wave": "pre",
                "missing_indicator": False,
            }
        )
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
