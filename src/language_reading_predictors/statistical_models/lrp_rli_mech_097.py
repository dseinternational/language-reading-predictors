# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP97 - negative-control outcome: letter-sound knowledge (L) -> receptive vocabulary (R).

Tier-1 decoding-specificity mini-suite, design 1B (see
``notes/202607172330-tier1-decoding-specificity-spec.md``). A **negative-control
outcome** for the letter-sound slope (Lipsitch, Tchetgen Tchetgen & Cohen 2010).

Under the revised DAG (``dag/dag-language-reading.dagitty``) letter-sound knowledge
``LS`` has descendants only ``{NW, PA, PS, WR}`` - the written-code skills - so
receptive vocabulary ``RV`` is **not** a causal descendant of ``LS``: there is no
``LS -> ... -> RV`` path. Any adjusted ``L -> R`` association is therefore purely
backdoor (through the shared parents ``{A, GA, HS, IG, IS, SP}``), and those
backdoors run through the *same* parent set as the reading outcomes - exactly the
negative-control-outcome condition ("known not to be an effect of the primary
exposure", "subject to the same source of confounding").

**Prediction / read.** With the matched Tier-1 adjustment ``{G, A, HS, IS, SP}`` +
own baseline, the ``L`` slope should be ~0 here while it is clearly positive on the
written-code outcomes (W = LRP101, N = LRP96, and spelling). If ``L`` *also* predicts
receptive vocabulary, letter sounds are behaving like a general-ability / teaching-dose
marker and the letter -> reading association is suspect - the direct diagnostic of
"associated through other mechanisms". This is a falsification test, not a mechanism.

Same linear parameterisation and conditioning set as the rest of the Tier-1 panel so
the negative-control forest is like-for-like. GA is unblockable; the slope is an
**adjusted association**.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-097",
    kind="mechanism",
    title="Negative-control outcome: letter sounds (L) -> receptive vocabulary (R)",
    outcome_symbol="R",
    mechanism_symbol="L",
    adjustment=["G", "A", "R_pre"],
    extra={
        "adjust_baseline_symbol": "R",
        "outcomes": ("R", "L"),
        "adjust_for": ("hs", "hs_missing", "attend", "deapp_c", "deapp_c_missing"),
        "linear_mechanism": True,
        "use_age_gp": False,
        "phase_specific_mechanism": False,
        "use_subject_random_intercept": True,
    },
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
