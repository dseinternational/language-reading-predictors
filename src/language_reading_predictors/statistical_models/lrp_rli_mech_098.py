# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP98 - negative-control outcome: letter-sound knowledge (L) -> expressive vocabulary (E).

Tier-1 decoding-specificity mini-suite, design 1B (see
``notes/202607172330-tier1-decoding-specificity-spec.md``). A **negative-control
outcome** for the letter-sound slope (Lipsitch, Tchetgen Tchetgen & Cohen 2010).

Under the revised DAG (``dag/dag-language-reading.dagitty``) letter-sound knowledge
``LS`` has descendants only ``{NW, PA, PS, WR}``, so expressive vocabulary ``EV`` is
**not** a causal descendant of ``LS`` (no ``LS -> ... -> EV`` path). Any adjusted
``L -> E`` association is purely backdoor through the shared parents
``{A, GA, HS, IG, IS, SP}`` - the same source of confounding as the reading outcomes.

**Read.** The ``L`` slope should be ~0 here; a clearly-positive slope would flag that
letter sounds are proxying general ability / teaching dose rather than acting on the
written code. Same linear parameterisation and matched conditioning ``{G, A, HS, IS,
SP}`` + own baseline as the rest of the Tier-1 panel. GA unblockable; adjusted
association only.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-098",
    kind="mechanism",
    title="Negative-control outcome: letter sounds (L) -> expressive vocabulary (E)",
    outcome_symbol="E",
    mechanism_symbol="L",
    adjustment=["G", "A", "E_pre"],
    extra={
        "adjust_baseline_symbol": "E",
        "outcomes": ("E", "L"),
        "adjust_for": ("hs", "hs_missing", "attend", "deapp_c", "deapp_c_missing"),
        "linear_mechanism": True,
        "use_age_gp": False,
        "phase_specific_mechanism": False,
        "use_subject_random_intercept": True,
    },
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
