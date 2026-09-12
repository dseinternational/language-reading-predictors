# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Plain-language findings for the dose response family."""

from __future__ import annotations

from pathlib import Path
import os
from collections.abc import Mapping
import numpy as np
import pandas as pd
from language_reading_predictors.statistical_models.findings.common import (
    _KeyFindingsUnavailable,
    _kf_association_direction,
    _kf_csv,
    _kf_csv_row,
    _kf_float,
    _kf_outcome_label,
    _kf_sentence,
)


def _kf_dose_companion_location(output_dir: str | Path, config: Mapping) -> tuple[str, str | None]:
    """The registered opposite-link twin's id and the directory it would occupy.

    The companion sits beside the fit as ``<companion_id>-<config_name>`` -- the
    location :func:`blending_sensitivity._evaluate_stored_blending_link_pair`
    reads -- so "absent" here means exactly what it means to the release gate.
    The directory is ``None`` when the fit records no ``config_name``.
    """
    from language_reading_predictors.statistical_models.dose_response import (
        DOSE_BLENDING_COMPANION_MODEL_ID,
        DOSE_BLENDING_PRIMARY_MODEL_ID,
    )

    plan = config.get("resolved_run_plan") or {}
    this_id = str(config.get("model_id") or "")
    companion_id = str(plan.get("required_link_companion_model_id") or "") or (
        DOSE_BLENDING_COMPANION_MODEL_ID
        if this_id == DOSE_BLENDING_PRIMARY_MODEL_ID
        else DOSE_BLENDING_PRIMARY_MODEL_ID
    )
    config_name = str(config.get("config_name") or "")
    if not config_name:
        return companion_id, None
    parent = os.path.dirname(os.path.realpath(str(output_dir)))
    return companion_id, os.path.join(parent, f"{companion_id}-{config_name}")


def _kf_dose_blending_link_sentence(output_dir: str | Path, config: Mapping) -> dict[str, str] | None:
    """The dose family's paired-link sentence, or ``None`` when not a B dose fit.

    Mirrors :func:`_kf_level_blending_link_sentence` for ``dose_response`` (#619).
    ``lrp-rli-dose-084`` fits the ordinary logit score mean and ``lrp-rli-dose-384``
    the one-in-three guessing-floor mean over the same rows, and the family's
    headline -- the treated-row dose marginal in ``dose_marginal_summary.csv`` --
    is a natural-scale quantity, so it inherits the link whichever half a reader
    opens. The companion's marginal is read from its fit directory through the same
    stored-artefact pair check the release gate runs, so the two numbers quoted
    side by side are the ones that check verified.

    Fails closed like the level builder -- a pair the gate cannot verify raises,
    which withholds the box -- with one deliberate exception. When the companion
    directory is simply absent, the pair has not been fitted at this sampling
    configuration yet: that is a fact about the batch, not a defect in the stored
    evidence, so the fit carries an explicit "pair not yet fitted" caveat instead
    of a paired sentence it cannot write. The release gate still withholds it.
    """
    if str(config.get("kind")) != "dose_response" or str(config.get("outcome_symbol")) != "B":
        return None
    from language_reading_predictors.statistical_models.blending_sensitivity import (
        evaluate_dose_blending_link_pair,
    )
    from language_reading_predictors.statistical_models.dose_response import (
        DOSE_BLENDING_COMPANION_MODEL_ID,
        DOSE_BLENDING_PRIMARY_MODEL_ID,
    )

    status = evaluate_dose_blending_link_pair(output_dir, config=config)
    if not status.get("required"):
        return None
    this_id = str(config.get("model_id") or "")
    if not status.get("ready"):
        companion_id, companion_dir = _kf_dose_companion_location(output_dir, config)
        registered = this_id in {
            DOSE_BLENDING_PRIMARY_MODEL_ID,
            DOSE_BLENDING_COMPANION_MODEL_ID,
        }
        if registered and companion_dir is not None and not os.path.isfile(os.path.join(companion_dir, "config.json")):
            this_link = str((config.get("resolved_run_plan") or {}).get("score_mean_link", "logit"))
            link_clause = (
                "This model uses the ordinary link, which allows fitted means below that guessing level"
                if this_link == "logit"
                else "This model holds the fitted mean at or above that guessing level"
            )
            return _kf_sentence(
                "**Qualified result.** Phoneme blending is measured with ten "
                "three-choice items, so a child guessing at random still scores "
                f"about a third. {link_clause}, and its required opposite-link "
                f"companion ({companion_id}) has not been fitted at the "
                f"{config.get('config_name')} sampling configuration, so the two "
                "links cannot yet be read side by side. Read the blending "
                "association as provisional until the pair is fitted together.",
                "caveat",
            )
        raise _KeyFindingsUnavailable(str(status.get("reason") or "the B link pair is not ready"))
    cards = status["cards"]
    other_id = next(k for k in cards if k != this_id)
    ordinary, floored = (
        (cards[this_id], cards[other_id])
        if cards[this_id]["score_mean_link"] == "logit"
        else (cards[other_id], cards[this_id])
    )
    return _kf_sentence(
        "Phoneme blending is scored from ten three-choice items, so a child "
        "answering at random scores about 3 out of 10. Two models are reported "
        "together because that floor matters: the ordinary model, which does not "
        "know about it, puts the session-attendance association at "
        f"**{_kf_float(ordinary['items_median']):+.1f} items** (89% credible range "
        f"{_kf_float(ordinary['items_lo']):+.1f} to "
        f"{_kf_float(ordinary['items_hi']):+.1f}), and the model that holds the "
        "score at or above chance puts it at "
        f"**{_kf_float(floored['items_median']):+.1f} items** "
        f"({_kf_float(floored['items_lo']):+.1f} to "
        f"{_kf_float(floored['items_hi']):+.1f}). Neither number is the answer on "
        "its own, and both are observational associations, not effects of "
        "attending more sessions.",
        "sensitivity",
    )


def _kf_build_dose_response(output_dir: str | Path, config: Mapping) -> list[dict[str, str]]:
    """Observational session-dose association."""
    marginal = _kf_csv_row(output_dir, "dose_marginal_summary.csv")
    outcome_label = _kf_outcome_label(config)
    # Phoneme blending: resolve the response-link pair before anything is worded,
    # so a B fit whose pair the gate cannot verify withholds the box (#619).
    link_sentence = _kf_dose_blending_link_sentence(output_dir, config)
    sentences: list[dict[str, str]] = []
    if marginal is not None:
        # Say how big the step actually was and who it was averaged over (#587
        # finding 3). "A 1-SD increase in sessions" hid a 30.7-session step applied to
        # every row, including children with no intervention at all; the repaired
        # contrast is a within-period interquartile move over on-intervention rows,
        # and the sentence names the sessions rather than a standardised unit.
        # A fit written before the support-respecting contrast existed has neither
        # column; fall back to the old wording rather than refusing to render.
        raw_step = marginal.get("contrast_sessions_median")
        step = (
            float(raw_step)
            if raw_step is not None and np.isfinite(pd.to_numeric(raw_step, errors="coerce"))
            else float("nan")
        )
        rows = marginal.get("n_rows")
        if np.isfinite(step):
            opening = (
                f"Among children on the intervention, attending about "
                f"{step:.0f} more sessions in a period — an interquartile step "
                f"within that period's observed attendance — was associated with"
            )
        else:
            opening = "A 1-SD increase in sessions was associated with"
        sentences.append(
            _kf_sentence(
                f"{opening} "
                f"**{_kf_float(marginal['items_median']):+.1f} items** on "
                f"{outcome_label} "
                f"(89% credible range {_kf_float(marginal['items_lo']):+.1f} "
                f"to {_kf_float(marginal['items_hi']):+.1f}"
                + (f"; averaged over {int(rows)} rows)." if rows is not None else ")."),
                "headline",
            )
        )
        sentences.append(
            _kf_sentence(
                _kf_association_direction(
                    marginal["prob_pos"],
                    positive_claim=(
                        "attending more sessions accompanies a higher outcome among "
                        "children already on the intervention"
                    ),
                    negative_claim=(
                        "attending more sessions accompanies a lower outcome among children already on the intervention"
                    ),
                ),
                "confidence",
            )
        )
    else:
        slopes = _kf_csv(output_dir, "dose_slope_summary.csv")
        if slopes is None:
            raise _KeyFindingsUnavailable("neither dose_marginal_summary.csv nor dose_slope_summary.csv is present")
        row = slopes.iloc[0].to_dict()
        sentences.append(
            _kf_sentence(
                f"The headline dose slope was "
                f"**{_kf_float(row['median']):+.2f} logit units per 1 SD of "
                f"sessions** (89% credible range {_kf_float(row['lo']):+.2f} "
                f"to {_kf_float(row['hi']):+.2f}).",
                "headline",
            )
        )
        sentences.append(
            _kf_sentence(
                _kf_association_direction(
                    row["p_pos"],
                    positive_claim="higher session dose accompanies a higher outcome",
                    negative_claim="higher session dose accompanies a lower outcome",
                ),
                "confidence",
            )
        )
    slope_table = _kf_csv(output_dir, "dose_slope_summary.csv")
    if slope_table is not None and "on_intervention" in set(slope_table["term"]):
        presence = slope_table[slope_table["term"] == "on_intervention"].iloc[0]
        sentences.append(
            _kf_sentence(
                "Being on the intervention at all is reported separately from how "
                "much was attended: "
                f"**{_kf_float(presence['median']):+.2f} logit units** "
                f"(89% credible range {_kf_float(presence['lo']):+.2f} to "
                f"{_kf_float(presence['hi']):+.2f}). In period 1 that contrast is "
                "randomised — every immediate-arm child attended and every waitlist "
                "child attended none — so it, not the dose slope, is where this "
                "model's randomised evidence sits.",
                "robustness",
            )
        )
    sentences.append(
        _kf_sentence(
            "How many sessions a child attended was not randomised and may reflect "
            "ability, attendance or availability — the study DAG has age, latent "
            "general ability and assigned group all pointing into attendance — so "
            "every dose slope here is an observational association, not evidence "
            "that more sessions cause more progress.",
            "causal",
        )
    )
    if link_sentence is not None:
        # Phoneme blending: the paired-link sentence rides immediately behind the
        # headline, because the headline alone is a single-link number -- the same
        # placement as the level family. The #587-finding-6 caveat that used to sit
        # here said the guessing-floor companion "has not been built for this
        # family"; ``lrp-rli-dose-384`` has been registered, fitted and gated since
        # #619, so the slot now carries the two links side by side, or the
        # not-yet-fitted caveat when the companion directory is absent.
        sentences.insert(1, link_sentence)
    return sentences
