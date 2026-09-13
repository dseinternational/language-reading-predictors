# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Level factors calculations and summaries."""

from __future__ import annotations

from pathlib import Path
import json
import os
from collections.abc import Mapping
from typing import Any
import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import expit
from language_reading_predictors.statistical_models.likelihood import (
    ScoreMeanLink,
    apply_score_mean_link,
)


def level_t2_marginal_effect(
    trace: xr.DataTree,
    *,
    phase: np.ndarray,
    G: np.ndarray,
    t2_phase: int = 1,
    contrast_term: str = "b_grp_time",
    contrast_index: int | None = None,
    interaction_term: str = "gamma_grp_ability",
    balance_term: str | None = None,
    score_mean_link: ScoreMeanLink = "logit",
    ability: np.ndarray | None = None,
    eta_name: str = "eta",
    group: str = "posterior",
) -> tuple[np.ndarray, np.ndarray]:
    """The t2 randomised contrast and its **arm-free standardised** AME (LRPLF, #127).

    The level model enters group as a per-timepoint vector because the trial is a
    waitlist crossover; **only the t2 element is the randomised
    treated-versus-untreated contrast** (the later timepoints are randomised
    early-start-versus-delayed-start schedule contrasts, #631 finding 13). This
    isolates that one causal effect on the items scale.

    **The estimand** (#584 finding 1, decided 2026-08-23 —
    ``notes/202608231800-level-factors-584-decisions.md``): the average, over the
    fitted t2 rows **each evaluated at its own arm-free profile**, of the effect of
    the randomised t2 change in the adjusted arm gap. Per draw, the *whole* group
    contribution is netted out of every t2 row to recover an arm-free baseline

    ``eta0 = eta - (balance + contrast + gamma_grp_ability*ability) * G``

    and only the focal contrast is added back:
    ``mean_i [ expit(eta0_i + contrast) - expit(eta0_i) ]``. Each row keeps its own
    age, ability main effect, adjusters and fitted child intercept, so the
    standardisation population is the fitted t2 children and the random-effect
    convention is each child's own posterior intercept — not an average child.

    Until #584 the balance term was neither netted out nor added back, so the
    immediate arm's rows were evaluated around ``z + arm_gap_t1`` while the waiting
    arm's were evaluated around ``z``. That is a hybrid over *observed-arm*
    operating points rather than a named estimand. Netting it out costs nothing
    numerically (no stored fit moved by more than 0.04 items, and no direction
    probability moved at all, because ``expit`` is near-linear over a shift that
    small) and it makes the population one the report can state.

    ``contrast_term`` names the posterior vector carrying the randomised t2 element
    and ``contrast_index`` that element's position in it (default: ``t2_phase``,
    the position in a ``phase``-indexed vector). Under the t1-referenced arm-gap
    parameterisation (#552) the caller passes ``contrast_term="d_grp_time"``,
    ``contrast_index=0`` (the ``t2`` entry of the ``post_phase``-indexed change
    vector) and ``balance_term="arm_gap_t1"``. Under the free comparator the focal
    ``b_grp_time[1]`` *is* the whole t2 arm gap, so there is no separate balance
    term to remove: the caller passes ``balance_term=None`` and the default
    ``b_grp_time`` / ``t2_phase`` reproduces the raw t2 gap, unchanged by this
    decision.

    ``gamma_grp_ability`` is a single *time-invariant* coefficient (identified mostly
    from the non-randomised t1/t3/t4 rows), so the moderation increment is held at
    centred ability rather than folded into the causal card — the group×ability
    moderation is reported separately (issue #271 item 5). Because the same focal
    draw is added to every row, the card is a per-draw monotone transform of the
    contrast: ``P(card > 0)`` equals ``P(contrast > 0)``, so the items median, the
    direction probability and the ROPE cannot disagree with the coefficient the
    report flags causal. A marginal response-scale difference-in-differences would
    not have that property (#584 decision 1).

    Returns ``(contrast_draws, ame_prob)`` — the logit-scale focal-contrast draws
    ``(S,)`` (the term flagged causal in the report) and the probability-scale average
    marginal effect per draw ``(S,)``, ready for :func:`rope_card`. ``ability`` is the
    standardised ability covariate aligned with ``eta``'s ``obs_id`` axis (pass
    ``None`` when the model has no group×ability term).
    """
    # ``group`` selects the posterior (the estimate) or the prior (the estimand-scale
    # prior-predictive pushforward, #389 finding 3); both carry eta / contrast /
    # interaction, so the same net-out transform applies to either.
    posterior = getattr(trace, group)
    phase = np.asarray(phase)
    G = np.asarray(G, dtype=float)
    mask = phase == t2_phase
    if not mask.any():
        raise ValueError(f"No rows at t2_phase={t2_phase}; phases present: {np.unique(phase)}")

    eta = posterior[eta_name].stack(sample=("chain", "draw")).transpose("obs_id", "sample").values  # (n_obs, S)
    if eta.shape[0] != phase.shape[0]:
        raise ValueError(
            f"phase has {phase.shape[0]} rows but eta has {eta.shape[0]} observations; "
            "pass built.prepared.phase (aligned with the fitted subset)."
        )

    bgt = posterior[contrast_term]
    extra = [d for d in bgt.dims if d not in ("chain", "draw")]
    if not extra:
        raise ValueError(f"{contrast_term!r} is not a per-timepoint vector; t2 contrast undefined")
    idx = t2_phase if contrast_index is None else int(contrast_index)
    if not 0 <= idx < int(bgt.sizes[extra[0]]):
        raise ValueError(
            f"contrast_index {idx} is outside {contrast_term!r}'s "
            f"{extra[0]} dimension of size {int(bgt.sizes[extra[0]])}"
        )
    contrast_draws = bgt.isel({extra[0]: idx}).stack(sample=("chain", "draw")).values  # (S,)

    # δ_i per t2 row and draw: the WHOLE group contribution — the balance term (when
    # the parameterisation carries one), the focal t2 contrast, and the group×ability
    # slope times each row's ability if the interaction is in the model.
    delta_rows = contrast_draws[None, :]  # (1, S)
    if balance_term is not None:
        if balance_term not in posterior:
            raise ValueError(
                f"balance_term {balance_term!r} is not in the {group} group; pass "
                "the term the plan records (None under the free comparator)."
            )
        balance_draws = posterior[balance_term].stack(sample=("chain", "draw")).values.ravel()  # (S,)
        delta_rows = delta_rows + balance_draws[None, :]
    if interaction_term in posterior and ability is not None:
        g_ab = posterior[interaction_term].stack(sample=("chain", "draw")).values.ravel()  # (S,)
        ab_t2 = np.asarray(ability, dtype=float)[mask]  # (m,)
        delta_rows = delta_rows + np.outer(ab_t2, g_ab)  # (m, S)

    eta_t2 = eta[mask]  # (m, S)
    G_t2 = G[mask]  # (m,)
    # Arm-free baseline for every t2 row (#584 decision 1): remove the complete group
    # contribution, so a waiting-arm row and an immediate-arm row with the same
    # covariates are evaluated at the same operating point. Then add back ONLY the
    # focal contrast — the pre-randomisation balance term is a chance imbalance, not
    # part of the effect, and ``gamma_grp_ability`` is one time-invariant coefficient
    # identified mostly from the non-randomised waves, so the moderation increment is
    # held at centred ability (ability is standardised, so that simply drops it). The
    # interaction is reported separately, never folded into the causal card
    # (issue #271 item 5).
    eta0 = eta_t2 - delta_rows * G_t2[:, None]
    # The marginal is a difference of SCORE MEANS, so it goes through the fit's own
    # score-mean link (#584 decision 2). Under the blending guessing floor the mean
    # is 1/3 + 2/3 * expit(eta), which compresses the same logit contrast into a
    # smaller items difference — reading a floor-link fit through the ordinary expit
    # would publish an effect the model does not imply.
    treated = apply_score_mean_link(expit(eta0 + contrast_draws[None, :]), score_mean_link)
    untreated = apply_score_mean_link(expit(eta0), score_mean_link)
    ame_prob = (treated - untreated).mean(axis=0)  # (S,)
    return contrast_draws, ame_prob


def level_window_comparator_cards(output_dir: str | Path, config: Mapping) -> list[dict[str, Any]] | None:
    """The four-wave and t1/t2 cards side by side, when both fits are present.

    #584 decision 3 keeps the four-wave levels fit as the model of record and adds a
    randomised-window comparator, "reporting its difference". This finds the
    counterpart fit beside this one — ``lrp-rli-lf-0NN`` <-> ``lrp-rli-lf-2NN``,
    resolved through the id renumber table rather than by string arithmetic — and
    returns both cards, most-restricted last, or ``None`` when the counterpart has
    not been fitted.

    Deliberately **not** a gate. The comparator answers "how much did the
    longitudinal working model move the answer?", and a missing comparator leaves
    that question open rather than making the model of record unpublishable; the
    blending link pair is the case where absence *is* disqualifying, and it has its
    own check. Reads stored cards only.
    """
    from language_reading_predictors import model_ids

    plan = config.get("resolved_run_plan") or {}
    if str(config.get("kind")) != "level_factors" or not plan.get("waves"):
        return None
    model_id = str(config.get("model_id") or "")
    config_name = str(config.get("config_name") or "")
    if not model_id or not config_name:
        return None
    try:
        legacy = model_ids.to_legacy(model_id)
        counterpart_legacy = legacy[:-1] if legacy.endswith("a") else f"{legacy}a"
        counterpart = model_ids.to_canonical(counterpart_legacy, kind="level_factors")
    except Exception:  # noqa: BLE001 - an unmapped id simply has no counterpart
        return None

    def _card(directory: str | Path, expected_id: str) -> dict[str, Any] | None:
        rope_path = os.path.join(str(directory), "rope_summary.csv")
        config_path = os.path.join(str(directory), "config.json")
        if not (os.path.exists(rope_path) and os.path.exists(config_path)):
            return None
        try:
            with open(config_path, encoding="utf-8") as handle:
                stored = json.load(handle)
            row = pd.read_csv(rope_path).iloc[0]
        except OSError, ValueError, KeyError, IndexError:
            return None
        if str(stored.get("model_id")) != expected_id:
            return None
        waves = tuple((stored.get("resolved_run_plan") or {}).get("waves") or ())
        return {
            "model_id": expected_id,
            "waves": waves,
            "window": "t1-t2 only" if len(waves) == 2 else "all four waves",
            "items_median": float(row["items_median"]),
            "items_lo": float(row["items_lo"]),
            "items_hi": float(row["items_hi"]),
            "pd": float(row["pd"]),
        }

    here = Path(str(output_dir)).resolve()
    cards = [
        _card(here, model_id),
        _card(here.parent / f"{counterpart}-{config_name}", counterpart),
    ]
    if any(card is None for card in cards):
        return None
    return sorted((card for card in cards if card is not None), key=lambda card: -len(card["waves"]))
