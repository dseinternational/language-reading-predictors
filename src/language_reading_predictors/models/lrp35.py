# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP35: Predictors of DEAP average articulation gains (``deappav_gain``).

``deappav`` is the DEAP picture-naming average from the Diagnostic
Evaluation of Articulation and Phonology (Dodd et al., 2006) — the
proportion of sounds correctly produced in a picture-naming task.
It is a composite of ``deappin``, ``deappvo``, ``deappfi`` (in the
candidate pool), but gain targets are near-noise (regression-to-
the-mean dominates), so the level model carries the same-
instrument check.

The gain target spans min -13.9, max 12.9, median 0.20, mean 0.42,
std 4.67, skew -0.09, with ~45% negative and ~2% zero observations
(n = 152). Regression from the mean dominates gain targets across
the suite.

This is an exploratory gradient-boosting discovery model on the
same footing as LRP01–22: it asks how predictable average
articulation is and from what, to inform whether the shared DAG
needs a speech-sound accuracy node. It is not a causal or
intention-to-treat estimate.

Uniform feature selection (2026-06-23): reduced from the full
35-predictor DEFAULT_GAIN set to 13 predictors via a distance-
correlation redundancy filter (dcor >= 0.70) plus an importance
noise-floor cut, then re-tuned. See the SelectionStep below and
notes/202606230900-predictability-speech-memory-language.md.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps ────────────────────────────────────────────

_SELECTION_STEPS: list[SelectionStep] = [
    SelectionStep(
        removed=[
            V.GROUP, V.AREA, V.GENDER, V.APTGRAM, V.APTINFO, V.B1EXTO,
            V.B1RETO, V.EOWPVT, V.ERBWORD, V.BLENDING, V.SPPHON, V.TROG,
            V.YARCSI, V.DEAPPIN, V.DEAPPFI, V.BEHAV, V.VISION,
            V.HEARING, V.NUMCHIL, V.AGEBOOKS, V.MUMEDUPOST16,
            V.DADEDUPOST16
        ],
        notes=(
            "Uniform feature selection (2026-06-23): from the full 35-predictor DEFAULT_GAIN set, a distance-correlation redundancy filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative) plus an importance noise-floor cut (<= 0.005). The baseline measure was force-kept (regression-to-the-mean anchor). The standardised instrument was preferred over its bespoke taught sibling (``rowpvt`` <- ``b1reto``). Reduces to 13 predictors with no dcor >= 0.70 pairs remaining; re-tuned on the reduced set (Optuna 150-trial MAE, 10-fold GroupKFold, seed 47). Same method as the LRP01–22 suite; see scripts/rank_predictors.py (the full-set ranking that supersedes the retired hard-selection pass) and notes/202606230900-predictability-speech-memory-language.md."
        ),
        date="2026-06-23",
        metrics_before={"cv_mae_mean": 3.7360},
        metrics_after={"cv_mae_mean": 3.4775},
    ),
]


# ── hyperparameters (MAE-tuned on the reduced set) ───────────────────────

_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "learning_rate": 0.18291066454567442,
    "num_leaves": 20,
    "max_depth": 12,
    "min_child_samples": 5,
    "subsample": 0.8201087570197031,
    "colsample_bytree": 0.8675365863766314,
    "reg_alpha": 0.0070609566714952225,
    "reg_lambda": 0.1711341570967381,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 5,
}


class LRP35(GainModel):
    """DEAP average articulation gains predictors — baseline (MAE-tuned)."""

    model_id = "lrp35"
    target_var = V.DEAPPAV_GAIN
    description = (
        "LightGBM — DEAP average articulation gains predictors (13 predictors, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    cv_splits = 51
    outlier_threshold = None
    selection_steps = _SELECTION_STEPS
    shap_scatter_specs = [
        ShapScatterSpec(description="All predictors, SHAP auto-colouring"),
    ]
    notes = (
        "Exploratory model for deappav_gain (gain). Uniform feature selection (2026-06-23) from the full 35-predictor DEFAULT_GAIN set to 13 predictors (distance-correlation redundancy filter + importance noise-floor cut; baseline force-kept; no dcor >= 0.70 pairs remain), re-tuned on the reduced set (tuner-inner CV MAE 3.404). Gain models are near-noise (baseline-driven regression to the mean) — treat the reduced ranking as exploratory. See notes/202606230900-predictability-speech-memory-language.md."
    )
