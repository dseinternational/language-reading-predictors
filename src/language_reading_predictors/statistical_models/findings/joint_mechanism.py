# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Plain-language findings for the joint mechanism family."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
import pandas as pd
from language_reading_predictors.statistical_models.findings.common import (
    _KeyFindingsUnavailable,
    _kf_csv,
    _kf_float,
    _kf_measure_label,
    _kf_psense_diagnosis,
    _kf_sentence,
)


def _kf_jm_interval(row: pd.Series) -> str:
    """One coefficient as median + inner 50% + outer reporting interval.

    The house standard is median with an inner 50% and outer 89% equal-tailed
    interval (METHODS.md), which is also #421's acceptance criterion — the first cut
    of this family reported only the outer interval (#427 review).
    """
    return (
        f"**{_kf_float(row['median']):+.2f}** (50% "
        f"{_kf_float(row['lo50']):+.2f} to {_kf_float(row['hi50']):+.2f}; 89% "
        f"{_kf_float(row['lo']):+.2f} to {_kf_float(row['hi']):+.2f})"
    )


def _kf_jm_wave_series(rows: pd.DataFrame, *, decimals: int = 2) -> str:
    """Every wave's median in wave order, as ``t1 -0.47, t2 -0.17, ...``.

    The whole set, never a selection. The previous builder led with the wave whose
    ``P(> 0)`` sat furthest from 0.5 — a headline chosen after seeing which posterior
    was most extreme, which is exactly the selection a reader would have to discount
    (2026-08-23 follow-up review, finding 1).
    """
    return ", ".join(
        f"{str(r['wave'])} {_kf_float(r['median']):+.{decimals}f}"
        for _, r in rows.iterrows()
    )


def _kf_jm_psense_flags(output_dir: str | Path, rows: pd.DataFrame) -> list[str]:
    """Flagged power-scaling parameters per published wave, in wave order.

    Each wave is read against **its own** table where one exists, so a diagnosis is
    surfaced beside the result it belongs to rather than borrowed from the wave that
    happens to host the fit-level artefacts.
    """
    terms = ("beta_mech[W]", "beta_mech[N]", "delta_ls_decoding", "rho_outcome")
    flags: list[str] = []
    for _, row in rows.iterrows():
        wave = str(row.get("wave", "")).strip()
        filename = str(row.get("psense_file") or "").strip() or "psense_summary.csv"
        flagged = [
            term
            for term in terms
            if _kf_psense_diagnosis(output_dir, term, filename=filename) is not None
        ]
        if flagged:
            flags.append(f"{wave} ({', '.join(flagged)})" if wave else ", ".join(flagged))
    return flags


def _kf_build_joint_mechanism(output_dir: str | Path, config: Mapping) -> list[dict[str, str]]:
    """Decoding-specificity contrast — and, in the per-wave levels design, the
    conditional-to-marginal slope ratio — from the bivariate joint model (#421 Tier 3).

    Reads ``joint_mechanism_slopes.csv``, which carries one block of rows per fitted
    wave (``t1``…``t4`` for ``design="levels"``, a single ``stacked`` block for
    ``design="transition"``).

    Three things this deliberately does not do (2026-08-23 follow-up review). It does
    not pick a wave to headline: with several waves the whole set is reported, in wave
    order. It does not call the contrast a decoding-use *signature*: the difference is
    measurement-scale dependent, and unequal loadings on a common general-ability
    factor produce a non-zero contrast with no causal letter-sound route at all. And
    it does not read the conditional/marginal ratio as a mediated share: the ratio is
    unbounded and governed by ``conditional_slope_ratio.csv``.
    """
    df = _kf_csv(output_dir, "joint_mechanism_slopes.csv")
    if df is None:
        raise _KeyFindingsUnavailable("joint_mechanism_slopes.csv is missing")
    for column in (
        "wave", "term", "median", "lo50", "hi50", "lo", "hi", "prob_pos", "converged",
    ):
        if column not in df.columns:
            raise _KeyFindingsUnavailable(
                f"joint_mechanism_slopes.csv has no {column!r} column"
            )
    # A wave whose fit did not converge is published flagged in the CSV but must not
    # enter any number in the box. Since the 2026-08-23 review every published wave is
    # also release-gating, so this filter is a second line rather than the only one.
    converged_rows = df["converged"].astype(str).str.lower().isin({"true", "1"})
    excluded_waves = [str(w) for w in df.loc[~converged_rows, "wave"].drop_duplicates()]
    df = df[converged_rows]
    delta = df[df["term"] == "delta_ls_decoding"]
    if delta.empty:
        raise _KeyFindingsUnavailable(
            "joint_mechanism_slopes.csv has no converged delta_ls_decoding row"
        )
    waves = [str(w) for w in df["wave"].drop_duplicates()]
    per_wave = len(waves) > 1
    # The family's own keys live under ``extra``; the top-level ``design`` is the
    # human-readable study-design string, not the "levels"/"transition" switch.
    extra = config.get("extra") or {}
    design = str(extra.get("design", "levels"))
    contrast = extra.get("contrast") or ("N", "W")
    hi_sym, lo_sym = (str(contrast[0]), str(contrast[1]))
    diagnostics = _kf_csv(output_dir, "joint_mechanism_fit_diagnostics.csv")

    # No lead wave. The retired builder headlined the wave whose ``P(Δ > 0)`` sat
    # furthest from 0.5 — a selection made after seeing every posterior — and
    # labelled that choice exploratory (2026-08-23 joint audit, finding 3). Since
    # #591 every fitted wave receives the same full lifecycle and the whole set is
    # reported in wave order, so there is no selection left to label.
    sentences: list[dict[str, str]] = []
    if per_wave:
        headline = (
            f"Letter-sound knowledge tracks the two reading outcomes differently, "
            f"and by how much depends on the wave: Δ = β(LS→{hi_sym}) − "
            f"β(LS→{lo_sym}) is {_kf_jm_wave_series(delta)} logit per SD at the "
            f"{len(waves)} fitted timepoints. All fitted waves are reported; none is "
            "selected as a headline."
        )
    else:
        headline = (
            f"On this model's scale the identified contrast Δ = β(LS→{hi_sym}) − "
            f"β(LS→{lo_sym}) is {_kf_jm_interval(delta.iloc[0])} logit per SD."
        )
    headline += (
        " Both slopes come from one posterior with an explicit cross-outcome "
        "dependence block, so this is a within-model contrast — not the "
        "product-of-marginals sensitivity that separate fits can only bound."
    )
    # 2026-08-23 joint audit, finding 4: the numbers are right; the construct-level
    # reading is not licensed. The two tests differ in item count, score
    # distribution, discrimination, reliability and floor/ceiling behaviour, and the
    # model puts them on no common latent outcome scale — so one shared ability
    # loading differently on the two tests produces a non-zero slope contrast by
    # itself. This is an operational property of the two scores, not a measure of
    # decoding specificity.
    headline += (
        " Read it as an **operational contrast between two adjusted test-score "
        "associations**, not as construct-level decoding specificity: the two tests "
        "differ in item count, score distribution, discrimination, reliability and "
        "floor/ceiling behaviour, and this model calibrates them to no common latent "
        "outcome scale, so a single shared ability that loads differently on them "
        "would produce a non-zero contrast on its own."
    )
    if design == "levels":
        headline += (
            " This is a **levels** contrast (score at the wave); it is a different "
            "estimand from the transition/ANCOVA contrast the Tier-1 note reports, "
            "and the two need not agree in sign."
        )
    else:
        headline += (
            " It is an ANCOVA association — each outcome's post-level given its own "
            "baseline — not a within-child change effect."
        )
    sentences.append(_kf_sentence(headline, "headline"))

    # The direction, read off the fit, with the interpretation limit attached. A
    # positive contrast is consistent with a decoding route; it does not reject an
    # unobserved common factor, because unequal loadings on one general ability
    # already produce a non-zero difference between two differently scaled outcomes.
    probs = [_kf_float(v) for v in delta["prob_pos"]]
    where = " at every fitted wave" if per_wave else ""
    if all(p > 0.5 for p in probs):
        direction = (
            f"letter sounds track {_kf_measure_label(hi_sym)} more closely than "
            f"{_kf_measure_label(lo_sym)}{where}"
        )
    elif all(p < 0.5 for p in probs):
        direction = (
            f"letter sounds track {_kf_measure_label(lo_sym)} more closely than "
            f"{_kf_measure_label(hi_sym)}{where}"
        )
    else:
        direction = "the contrast does not keep one sign across the fitted waves"
    # A levels-scale reversal has a ready non-causal reading; the ANCOVA design's
    # must not borrow it (2026-08-21 review).
    if any(p < 0.5 for p in probs):
        direction += (
            " — which on the levels scale is what a shared reading-development / "
            "general-ability component would produce and what the 6-item nonword "
            "floor would exaggerate"
            if design == "levels"
            else " — a reversal of the Tier-1 contrast, to be read against the "
            "matched mech-096 / mech-101 pair"
        )
    sentences.append(
        _kf_sentence(
            f"Direction: {direction} (P(Δ > 0) = "
            f"{', '.join(f'{p:.2f}' for p in probs)}). Read it as an adjusted, "
            "measurement-scale-dependent association contrast, not a decoding-use "
            "signature: the two outcomes have different item counts, floors and link "
            "discrimination, and unequal loadings on one unobserved general-ability "
            "factor would produce a non-zero contrast with no causal letter-sound "
            "route at all.",
            "confidence",
        )
    )

    # The ratio: one stability rule, applied in the pipeline and reproduced in
    # ``conditional_slope_ratio.csv``. Reported only where it holds, never as a
    # median classified against 0.5, and never as a mediated share (2026-08-23 joint
    # audit, findings 4 and 10; #591 follow-up review, finding 5).
    share = df[df["term"] == "share_retained"]
    if not share.empty:
        stable_flags = (
            share["share_retained_stable"].astype(str).str.lower().isin({"true", "1"})
            if "share_retained_stable" in share.columns
            else pd.Series(True, index=share.index)
        )
        stable = share[stable_flags]
        unstable_waves = [str(w) for w in share.loc[~stable_flags, "wave"]]
        governance = _kf_csv(output_dir, "conditional_slope_ratio.csv")
        regions = ""
        if governance is not None and "prob_in_unit" in governance.columns:
            usable = governance[
                governance["wave"].astype(str).isin(set(stable["wave"].astype(str)))
            ]
            if not usable.empty:
                regions = (
                    " P(0 ≤ ratio ≤ 1) = "
                    + ", ".join(
                        f"{_kf_float(v):.2f}" for v in usable["prob_in_unit"]
                    )
                    + "."
                )
        if not stable.empty:
            sentences.append(
                _kf_sentence(
                    f"Holding latent {_kf_measure_label(hi_sym)} fixed, the ratio of "
                    f"the adjusted letter-sound → {_kf_measure_label(lo_sym)} "
                    "association to its unconditional value is "
                    f"{_kf_jm_wave_series(stable)}.{regions} It is a **ratio of two "
                    "adjusted associations** — unbounded, not a mediation "
                    "proportion, not a causal path fraction, and not evidence that "
                    "the association runs through a decoding channel. It partials "
                    "the *latent* held-fixed skill rather than an observed score.",
                    "detail",
                )
            )
        reduction = df[df["term"] == "abs_slope_reduction"]
        if not reduction.empty:
            sentences.append(
                _kf_sentence(
                    "On the denominator-free scale, holding latent "
                    f"{_kf_measure_label(hi_sym)} fixed reduces the absolute "
                    f"letter-sound → {_kf_measure_label(lo_sym)} slope by "
                    f"{_kf_jm_wave_series(reduction)} logit per SD. This companion "
                    "is reported whether or not the ratio is stable, because a "
                    "difference has no denominator to blow up.",
                    "detail",
                )
            )
        if unstable_waves:
            sentences.append(
                _kf_sentence(
                    f"The ratio is withheld as unstable at "
                    f"{', '.join(unstable_waves)}: the posterior does not put at "
                    "least 95% of its mass on the unconditional slope, or on the "
                    "held-fixed outcome's residual scale, being away from zero, so "
                    "the ratio is heavy-tailed and its summary would describe the "
                    "draws rather than the quantity. Read the two slopes and their "
                    "absolute difference instead.",
                    "note",
                )
            )

    # The transition design publishes one posterior and writes no per-wave diagnostic
    # table, so it is read against the fit-level power-scaling table.
    psense_rows = (
        diagnostics
        if diagnostics is not None
        else pd.DataFrame([{"wave": "", "psense_file": "psense_summary.csv"}])
    )
    flags = _kf_jm_psense_flags(output_dir, psense_rows)
    if flags:
        sentences.append(
            _kf_sentence(
                "Power-scaling sensitivity is flagged for "
                f"{'; '.join(flags)}. Those posteriors move materially when the "
                "prior or the likelihood is reweighted, so read the affected numbers "
                "as prior-dependent until direct alternative-prior fits resolve them.",
                "detail",
            )
        )

    if excluded_waves:
        sentences.append(
            _kf_sentence(
                f"Wave(s) {', '.join(excluded_waves)} did not meet the convergence "
                "gate; their rows are published flagged in "
                "joint_mechanism_slopes.csv but are excluded from every number "
                "above.",
                "detail",
            )
        )

    rho = df[df["term"] == "rho_outcome"]
    if not rho.empty:
        level = "within-wave residual" if design == "levels" else "between-child"
        sentences.append(
            _kf_sentence(
                f"The {level} correlation between the two outcomes is "
                + (
                    _kf_jm_wave_series(rho)
                    if per_wave
                    else _kf_jm_interval(rho.iloc[0])
                )
                + ". This is the dependence block doing the work: an interval "
                "sitting on zero would mean the joint fit buys little over two "
                "separate ones.",
                "detail",
            )
        )

    slopes = df[df["term"].isin([f"beta_mech[{lo_sym}]", f"beta_mech[{hi_sym}]"])]
    if not per_wave and len(slopes) == 2:
        at = {str(r["term"]): r for _, r in slopes.iterrows()}
        sentences.append(
            _kf_sentence(
                f"The two letter-sound slopes: {_kf_measure_label(hi_sym)} "
                f"{_kf_jm_interval(at[f'beta_mech[{hi_sym}]'])} versus "
                f"{_kf_measure_label(lo_sym)} "
                f"{_kf_jm_interval(at[f'beta_mech[{lo_sym}]'])}, on one commensurate "
                "logit-per-SD scale.",
                "detail",
            )
        )
    return sentences
