# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP51: Bayesian DAG of word-reading gain from t1 to t2.

The first Bayesian model in the project. Models EWRSWR at t2 under an
explicit directed graph of influences from baseline age, receptive and
expressive vocabulary, letter-sound knowledge, intervention group, and
intervention attendance across the t1-t2 interval.

Each outcome (word reading; receptive, expressive, and letter-sound
intermediates) is modelled as a beta-binomial with a node-specific item
total. Each edge enters the child's logit-scale linear predictor as a
linear coefficient plus, where specified, an HSGP-approximated 1-D
Gaussian process capturing variation from the central linear trend.
``group`` enters as a single binary contrast on the word-reading node.

Scope for LRP51 is the single t1 → t2 transition; subject-level random
effects are deferred to models using multiple transitions.

See ``notes/202604171623-rct-word-reading-initial-lrp51.md`` for the
design rationale.
"""

from __future__ import annotations

from typing import ClassVar

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import BayesianModel
from language_reading_predictors.models.bayesian.definition import (
    BayesianDAGSpec,
    NodeSpec,
    ParentSpec,
)
from language_reading_predictors.models.bayesian.betabinom_pipeline import (
    BetaBinomialHSGPPipeline,
)


# ── node item totals ─────────────────────────────────────────────────────
#
# Provisional N values — confirm against test manuals before the reporting
# run. Observed maxima in the current dataset fall comfortably below each N.
N_ROWPVT = 190     # ROWPVT-4 item bank (to confirm edition)
N_EOWPVT = 190     # EOWPVT-4 item bank (to confirm edition)
N_YARCLET = 32     # YARC letter-sound subtest (matches observed max)
N_EWRSWR = 64      # EWRSWR item bank (observed max; to confirm)


# ── DAG specification ────────────────────────────────────────────────────

_DAG = BayesianDAGSpec(
    nodes=[
        NodeSpec(
            name="R",
            outcome_column=V.ROWPVT_NEXT,
            likelihood="beta_binomial",
            n_trials=N_ROWPVT,
            description="Receptive vocabulary at t2 (ROWPVT raw score)",
            parents=[
                ParentSpec(name="A", column=V.AGE, kind="gp",
                           description="Age in months at t1"),
                ParentSpec(name="R0", column=V.ROWPVT, kind="offset",
                           description="Baseline receptive vocabulary (t1)"),
            ],
        ),
        NodeSpec(
            name="E",
            outcome_column=V.EOWPVT_NEXT,
            likelihood="beta_binomial",
            n_trials=N_EOWPVT,
            description="Expressive vocabulary at t2 (EOWPVT raw score)",
            parents=[
                ParentSpec(name="A", column=V.AGE, kind="gp"),
                ParentSpec(name="R", column=V.ROWPVT, kind="gp",
                           description="Receptive vocabulary at t1"),
                ParentSpec(name="E0", column=V.EOWPVT, kind="offset",
                           description="Baseline expressive vocabulary (t1)"),
            ],
        ),
        NodeSpec(
            name="L",
            outcome_column=V.YARCLET_NEXT,
            likelihood="beta_binomial",
            n_trials=N_YARCLET,
            description="Letter-sound knowledge at t2 (YARC-LET, 32 items)",
            parents=[
                ParentSpec(name="A", column=V.AGE, kind="gp"),
                ParentSpec(name="L0", column=V.YARCLET, kind="offset",
                           description="Baseline letter-sound knowledge (t1)"),
            ],
        ),
        NodeSpec(
            name="W",
            outcome_column=V.EWRSWR_NEXT,
            likelihood="beta_binomial",
            n_trials=N_EWRSWR,
            description="Word reading at t2 (EWRSWR raw score)",
            parents=[
                ParentSpec(name="A", column=V.AGE, kind="gp"),
                ParentSpec(name="R", column=V.ROWPVT, kind="gp"),
                ParentSpec(name="E", column=V.EOWPVT, kind="gp"),
                ParentSpec(name="L", column=V.YARCLET, kind="gp"),
                ParentSpec(name="G", column=V.GROUP, kind="binary",
                           description="Intervention group (1=receiving, 2=waiting)"),
                ParentSpec(name="I", column=V.ATTEND, kind="gp",
                           description="Intervention sessions attended, t1 to t2"),
                ParentSpec(name="W0", column=V.EWRSWR, kind="offset",
                           description="Baseline word reading (t1)"),
            ],
        ),
    ],
)


# ── declaration ──────────────────────────────────────────────────────────


class LRP51(BayesianModel):
    model_id: ClassVar[str] = "lrp51"
    description: ClassVar[str] = (
        "Bayesian DAG of word-reading gain (t1 to t2): age, receptive / "
        "expressive vocabulary, letter-sound knowledge, group, and "
        "attendance as predictors of EWRSWR at t2. Beta-binomial "
        "likelihoods with HSGP edges."
    )
    dag_spec: ClassVar[BayesianDAGSpec] = _DAG
    primary_node: ClassVar[str] = "W"
    pipeline_cls: ClassVar = BetaBinomialHSGPPipeline
    notes: ClassVar[str] = (
        "Initial Bayesian model. Provisional N values: "
        f"ROWPVT={N_ROWPVT}, EOWPVT={N_EOWPVT}, YARCLET={N_YARCLET}, "
        f"EWRSWR={N_EWRSWR}. See notes/202604171623-rct-word-reading-initial-lrp51.md."
    )
