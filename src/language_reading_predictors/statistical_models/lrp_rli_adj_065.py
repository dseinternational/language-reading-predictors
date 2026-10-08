# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP65 - adjusted baseline predictors of word-reading gain.

One row per child relates word reading at t4 to word reading at t1 and
standardised t1 predictors. The Beta-Binomial regression includes letter sounds,
blending, an equal-weight language composite (R, E and F), age and the declared
trait covariates. It has no child random intercept. A pooled transition model
would answer a different question and would combine within-child and
between-child information unless those components were explicitly separated.

Every slope is an adjusted association. Correlated skills and regularising
priors affect how the model allocates their shared information. A small adjusted
coefficient describes limited additional signal conditional on the other
measures; it does not establish that the skill is unrelated to progress. Latent
general ability is drawn in the explanatory DAG but is not estimated or
controlled by this regression.

Hearing, speech production and phonological memory enter with missingness
indicators; constant indicators are dropped. SES enters a separate sensitivity
fit on the SES-complete subset. Those handling choices and the small sample
qualify the fitted population and interpretation.

The report compares mutually adjusted slopes with baseline-conditioned
single-predictor slopes and refits under the declared wider slope priors. These
are sensitivity comparisons, not a decomposition of shared variance or a test
that identifies causal edges. Read the causal contrast from the available-case
modified ITT model ``lrp-rli-itt-010`` under its stated assumptions; its question
differs from this model's baseline-skill associations.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from language_reading_predictors.statistical_models.environment import DOCS_DIR

if TYPE_CHECKING:
    import graphviz

    from language_reading_predictors.statistical_models.context import (
        ModelSpec,
        StatisticalFitContext,
    )

# ---------------------------------------------------------------------------
# Step 1 - causal DAG (the review gate, rendered before fitting)
# ---------------------------------------------------------------------------

# Believed-present structural edges plus the two edges *under test* (non-verbal
# MA -> gain, behaviour -> gain), which are styled dashed in ``EDGE_PROPS``.
_EDGE_LIST: list[tuple[str, str]] = [
    # General ability drives the correlated T1 baselines.
    ("g", "ls"),
    ("g", "lang"),
    ("g", "blend"),
    ("g", "nvma"),
    ("g", "wpre"),
    # SES sits upstream of general ability / the home environment.
    ("ses", "g"),
    # Age acts *through* general ability (developmental level), not directly on
    # the individual skills — the apparent direct age->skill link was an
    # over-adjustment artefact (revised DAG, dag/dag-language-reading.dagitty). A
    # direct age->gain edge remains (younger children gain more, net of baseline).
    ("age", "g"),
    ("age", "wgain"),
    # The starting skills that carry the gain signal.
    ("ls", "wgain"),
    ("lang", "wgain"),
    ("blend", "wgain"),
    # Baseline coupling / regression to the mean (conditioned on).
    ("wpre", "wgain"),
    # Revised-DAG upstream traits (2026-07-10, dag/dag-language-reading.dagitty):
    # hearing (HS), speech production (SP) and phonological memory (RW) are causes
    # of the baseline-skill cluster. Their -> gain edges are under test (dashed),
    # exactly like non-verbal MA / behaviour (#247).
    ("hs", "ls"),
    ("hs", "lang"),
    ("hs", "blend"),
    ("sp", "ls"),
    ("sp", "lang"),
    ("sp", "blend"),
    ("rw", "lang"),
    ("rw", "blend"),
    # Dashed edges mark the proposed gain associations. The regression does
    # not test whether the causal edges exist. The four-wave design alone
    # does not establish equal elapsed assessment intervals.
    ("nvma", "wgain"),
    ("behav", "wgain"),
    ("ses", "wgain"),
    ("hs", "wgain"),
    ("sp", "wgain"),
    ("rw", "wgain"),
]

_NODE_PROPS: dict[str, dict[str, str]] = {
    "g": {
        "label": "General ability (g)\\n[latent, not fitted]",
        "shape": "circle",
        "style": "dashed",
        "color": "grey45",
        "fontcolor": "grey45",
    },
    "ses": {"label": "SES\\n(parental education)"},
    "age": {"label": "Age (T1)"},
    "ls": {"label": "Letter sounds (T1)\\nYARC-LSK"},
    "lang": {"label": "Language composite (T1)\\nROWPVT+EOWPVT+CELF"},
    "blend": {"label": "Blending (T1)"},
    "nvma": {"label": "Non-verbal MA (T1)\\nblock design"},
    "behav": {"label": "Behaviour (T1)"},
    "hs": {"label": "Hearing status (T1)\\nhs (from hearing_c)"},
    "sp": {"label": "Speech production (T1)\\ndeapp_c"},
    "rw": {"label": "Phon. memory (T1)\\nerbto"},
    "wpre": {"label": "Word reading (T1)\\nW_pre", "shape": "box"},
    "wgain": {
        "label": "Word-reading gain\\n(T1 -> last wave; W_last | W_T1)",
        "shape": "box",
        "style": "filled",
        "fillcolor": "#e8eef7",
    },
}

# Edges whose presence is the hypothesis under test (drawn dashed): the
# covariates the descriptives expect to carry no independent signal once
# language + letter sounds are adjusted for. SES is tested in a separate
# complete-case sensitivity fit.
_EDGE_PROPS: dict[tuple[str, str], dict[str, str]] = {
    ("nvma", "wgain"): {"style": "dashed", "color": "grey45"},
    ("behav", "wgain"): {"style": "dashed", "color": "grey45"},
    ("ses", "wgain"): {"style": "dashed", "color": "grey45"},
    ("hs", "wgain"): {"style": "dashed", "color": "grey45"},
    ("sp", "wgain"): {"style": "dashed", "color": "grey45"},
    ("rw", "wgain"): {"style": "dashed", "color": "grey45"},
}


def causal_dag() -> "graphviz.Digraph":
    """Return the LRP65 causal DAG as a ``graphviz.Digraph``.

    Latent general ability ``g`` (dashed circle) drives the correlated baselines;
    the dashed ``-> gain`` edges from non-verbal MA and behaviour are the
    associations the adjusted model is built to test.
    """
    # Lazy import: ``plot_utils`` pulls in the plotting stack (networkx etc.),
    # which the fit path does not need. Keeping it here lets the dispatcher
    # import ``lrp-rli-adj-065`` (and fit) with only the sampler dependencies present.
    from language_reading_predictors.plot_utils import draw_causal_graph

    return draw_causal_graph(
        _EDGE_LIST,
        node_props=_NODE_PROPS,
        edge_props=_EDGE_PROPS,
        graph_direction="TB",
    )


def render_dag(output_dir: str | None = None, *, fmt: str = "svg") -> str:
    """Render the causal DAG to ``{output_dir}/dag.{fmt}`` and return the path.

    Defaults to ``docs/models/lrp-rli-adj-065/`` so the DAG can be committed and reviewed
    before any fitting (the Step-1 gate).
    """
    if output_dir is None:
        output_dir = os.path.join(DOCS_DIR, "models", "lrp-rli-adj-065")
    os.makedirs(output_dir, exist_ok=True)
    g = causal_dag()
    # graphviz appends the format extension; ``cleanup`` removes the .gv source.
    g.render(filename=os.path.join(output_dir, "dag"), format=fmt, cleanup=True)
    return os.path.join(output_dir, f"dag.{fmt}")


# ---------------------------------------------------------------------------
# Step 2 - model specification (consumed by ``fit_adjusted``)
# ---------------------------------------------------------------------------


# ``ModelSpec`` lives in ``context``, which imports the Bayesian stack
# (arviz / pymc / dse_research_utils). The spec is therefore built lazily so the
# Step-1 DAG (``causal_dag`` / ``render_dag`` above) can be rendered with only
# graphviz available - no sampler dependencies needed for the review gate.
def get_spec() -> "ModelSpec":
    """Return the LRP65 model specification (built lazily; see note above)."""
    from language_reading_predictors.statistical_models.adjusted import (
        AdjustedModelSettings,
    )
    from language_reading_predictors.statistical_models.context import ModelSpec

    return ModelSpec(
        model_id="lrp-rli-adj-065",
        kind="adjusted",
        title="Adjusted model: independent baseline predictors of word-reading gain",
        outcome_symbol="W",
        adjustment=["L", "lang", "B", "A", "W_pre", "blocks", "behav", "hs", "deapp_c", "erbto"],
        model_settings=AdjustedModelSettings(
            # Headline = genuinely between-child: one row per child, T1 baselines,
            # full-study gain (W at last wave conditioned on W_T1). No phase
            # dimension and no child random intercept (one obs per child).
            design="between_child",
            post_time=4,
            # Standardised T1 predictors of interest (letter sounds, blending).
            predictor_symbols=("L", "B"),
            # Equal-weight language composite (receptive + expressive + concepts).
            language_composite_symbols=("R", "E", "F"),
            use_age_predictor=True,
            # Continuous covariates entered to test independent signal. The revised
            # 2026-07-10 DAG adds three upstream traits — hearing (HS = hs), speech
            # production (SP = deapp_c) and phonological memory (RW = erbto) — as
            # causes of the baseline-skill cluster; they are entered here (with the
            # missing-indicator method) to test whether any carries independent
            # word-reading-gain signal net of the language/letter-sound cluster (#247).
            # A constant _missing indicator on the fitted rows is dropped by the loader.
            covariates=(
                "blocks",
                "behav",
                "hs",
                "hs_missing",
                "deapp_c",
                "deapp_c_missing",
                "erbto",
                "erbto_missing",
            ),
            # SES sensitivity fit on the SES-complete subset (not the headline model).
            ses_covariates=("mumedupost16",),
            # Fixed weakly-informative slope prior + the sensitivity sweep that
            # checks the which-predictors-clear-zero conclusion is stable.
            # Reconciled 0.5 -> 0.3 to match the shared association scale
            # (gamma_cross) — prior-critical-review 2026-07-07, recommendation 3.
            # The sweep checks sensitivity to wider priors, including the old
            # default. Agreement must be assessed from the resulting fits.
            predictor_slope_sigma=0.3,
            prior_sensitivity_sigmas=(0.5, 0.7),
        ),
    )


def fit(config: str = "dev") -> StatisticalFitContext:
    # Lazy import: ``fit_adjusted`` is added in Step 2, after the DAG review.
    from language_reading_predictors.statistical_models.pipelines.adjusted import fit_adjusted

    return fit_adjusted(get_spec(), config=config)


if __name__ == "__main__":  # pragma: no cover
    print(render_dag())
