# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP198 - ability-adjusted Tier-1 panel: letter sounds (L) -> expressive vocabulary (E).

Ability-adjusted companion to LRP098. Identical in every respect except that the
matched conditioning set gains ``blocks`` (WPPSI Block Design), the study's measured
general-ability proxy, so the whole Tier-1 panel can be re-read with measured
ability partialled out.

**Why.** The panel's finding is that the ``L`` slope is clearly positive on the
oral-language negative controls as well as the written-code outcomes, i.e. part of
every slope is non-specific. The design note
(``notes/202607172330-tier1-decoding-specificity-spec.md``) names shared latent
general ability ``GA`` as the leading candidate and records that ``GA`` is
*structurally unblockable* - it is latent and a parent of every skill. That is a
statement about the latent node, not about the data: ``blocks`` is measured on every
child, complete, and constant within child across all four waves, so it is a clean
pre-treatment between-child proxy that can be adjusted for without conditioning on
anything post-treatment.

**Read.** If the non-specific component is largely measured ability, the negative
controls (E among them for the control rows) should shrink towards zero here
while the written-code slopes hold up. If every slope shrinks in proportion, the
adjustment is removing shared variance indiscriminately and the panel says less than
it appeared to. If nothing moves, the child random intercept was already absorbing
the between-child ability signal and the proxy adds nothing.

**Ceilings.** ``blocks`` is a single noisy subtest, not ``GA``; residual confounding
by the latent node survives any adjustment for it, so ``beta_mech`` remains an
**adjusted association** and never a causal skill-to-skill effect. Role in the panel:
negative control - an oral-language control.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.mechanism import (
    MechanismModelSettings,
)
from language_reading_predictors.statistical_models.pipelines.mechanism import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-198",
    kind="mechanism",
    title="Ability-adjusted Tier-1 panel: letter sounds (L) -> expressive vocabulary (E)",
    outcome_symbol="E",
    mechanism_symbol="L",
    adjustment=["G", "A", "E_pre"],
    model_settings=MechanismModelSettings(
        adjust_baseline_symbol="E",
        outcomes=("E", "L"),
        adjust_for=("hs", "hs_missing", "attend", "deapp_c", "deapp_c_missing"),
        ability_covariate="blocks",
        linear_mechanism=True,
        use_age_gp=False,
        phase_specific_mechanism=False,
        use_subject_random_intercept=True,
    ),
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
