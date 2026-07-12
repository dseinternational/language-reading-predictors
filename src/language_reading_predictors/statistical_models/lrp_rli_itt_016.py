# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPITT16 - modality contrast: taught expressive (TE) vs taught receptive (TR).

The generalisation contrasts LRPITT15 (expressive: TE vs UE) and LRPITT15b/115
(receptive: TR vs UR) each compare taught vs not-taught *within* a modality. Neither
asks whether the intervention moved the directly-taught words more in the
**expressive** modality than in the **receptive** modality. LRPITT16 supplies that
missing contrast: the two Block 1 *taught* outcomes — expressive taught target words
(``TE`` = b1extau) and receptive taught target words (``TR`` = b1rectau) — are
modelled jointly over the randomised window, and the headline quantity is the
difference in intervention benefit between them.

Sign convention: ``tau`` is the coefficient on the intervention indicator
(positive => the intervention raised that outcome). The reported contrast is
``tau[TE] - tau[TR]`` (benefit on taught-expressive minus taught-receptive): a
positive value means the programme moved the taught words more in production than in
comprehension.

Both are causal (randomised-window ITT effects), so the contrast is a clean
comparison of two randomised effects — but the two outcomes are on different item
denominators, so it is read on the logit scale.

LKJ residual correlation is OFF, mirroring the documented convergence-safe fallback
of the LRPITT15 companions: with ``n ~ 53`` and two outcomes, the residual
correlation is weakly identified and the conservative choice widens the difference
slightly rather than risk poor mixing.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_joint

SPEC = ModelSpec(
    model_id="lrp-rli-itt-016",
    kind="joint",
    title="Modality contrast: taught expressive (TE) vs taught receptive (TR) vocabulary, block 1",
    extra={
        "outcomes": ("TE", "TR"),
        # DAG-faithful spec, mirroring the single-outcome suite (own baseline +
        # linear age, no cross-baselines).
        "use_cross_baselines": False,
        "use_age_linear": True,
        "use_residual_correlation": False,
        "difference": ("TE", "TR"),
    },
)


def fit(config: str = "dev"):
    return fit_joint(SPEC, config=config)
