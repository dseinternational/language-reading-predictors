# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP65 - Adjusted model: independent baseline predictors of word-reading gain.

The descriptive layer (responder/non-responder comparison, decode-predictor test,
and the LightGBM discovery model LRP01) converged on the same carry signal for
word-reading gain: baseline letter-sound knowledge, language/vocabulary and
blending, with younger children gaining more. Non-verbal mental age (block
design), SES and behaviour did *not* add independent signal in those passes.

LRP65 is the Bayesian "interpretable estimand" follow-up: the candidate predictors
are a correlated general-ability cluster (vocab-blocks rho ~ 0.63; vocab-letter
sounds ~ 0.4-0.5), so "independent effect" is ambiguous and a naive joint
regression is multicollinear. We therefore (1) specify the DAG first, to fix the
estimand and the adjustment set, and (2) fit a mutually-adjusted regression with
regularising priors.

Estimand (between-child association)
------------------------------------
For each baseline skill, the **mutually-adjusted between-child association** with
subsequent word-reading gain - i.e. its partial association with the full-study
gain ``W_post | W_pre`` (pre = T1), holding the other measured T1 baselines
constant, *across children*.

This is an association, NOT a causal effect: n ~ 54, observational contrasts
between children. The randomised ITT models (LRP52 / LRP55) carry the causal claim
about the programme; LRP65 is about *which starting skills go with more gain*.

Estimand <-> design (the decision the DAG gate settles)
-------------------------------------------------------
A between-child estimand must be matched by a between-child design. Pooling all
phase transitions (t1->t2, t2->t3, t3->t4) and adding a child random intercept
does NOT estimate it: the random intercept absorbs stable between-child
differences and pulls the coefficients toward the *within-child* association
("when a child's level is higher at the start of a phase, do they gain more that
phase"). That is a different question and can disagree.

So the headline design is genuinely between-child: **one row per child**,
predictors are each child's **T1** baselines, outcome is the full-study gain
``W_t4 | W_t1`` (Beta-Binomial on the EWRSWR count), and there is **no** child
random intercept. The pooled cross-phase variant (all phases, child random
intercept) is not implemented here (the adjusted-model factory rejects pooled
data); it would answer the within+between question and need the estimand
reworded. Settle between-child T1 vs pooled cross-phase
(and the gain window) at the DAG review, with Frank.

DAG-derived adjustment set
--------------------------
Latent general ability ``g`` drives the correlated baselines (letter sounds,
language, blending, non-verbal MA, baseline word reading). Mutual adjustment among
indicators of ``g`` redistributes their shared variance: a near-zero adjusted
coefficient for non-verbal MA therefore means "no signal beyond the shared ability
already captured by language + letter sounds", NOT "non-verbal MA is unrelated to
gain". ``g`` is drawn to make this explicit; it is NOT estimated (decision: the
fitted model is a regression, not a latent-factor SEM).

- Predictors of interest (T1, standardised): letter sounds (L), a language
  composite (equal-weight ROWPVT + EOWPVT + CELF), blending (B), age.
- Baseline conditioning: ``W_pre`` (T1 word reading) enters linearly via
  ``gamma_own`` - the "gain" framing, shared with the mechanism models.
- Tested covariates (entered to *demonstrate* whether they carry independent
  signal, not assumed non-independent): non-verbal MA (block design, T1) and
  behaviour (T1). SES has notable missingness, so it enters a separate
  sensitivity fit on complete cases rather than the headline model.

Report each predictor's adjusted coefficient alongside its bivariate (total)
association, so the shared-variance shift is visible. Prediction to test (NOT the
assumed result): letter sounds + the language composite retain credible signal;
non-verbal MA / SES / behaviour shrink toward zero. Wide intervals at n ~ 54 are
the honest result.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from language_reading_predictors.statistical_models.environment import DOCS_DIR

if TYPE_CHECKING:
    from language_reading_predictors.statistical_models.context import ModelSpec

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
    # over-adjustment artefact (shared DAG v5). A direct age->gain edge remains
    # (younger children gain more, net of baseline).
    ("age", "g"),
    ("age", "wgain"),
    # The starting skills that carry the gain signal.
    ("ls", "wgain"),
    ("lang", "wgain"),
    ("blend", "wgain"),
    # Baseline coupling / regression to the mean (conditioned on).
    ("wpre", "wgain"),
    # Edges under test - the model decides whether these are non-zero once the
    # language + letter-sound signal is in. (Follow-up span is uniform - all
    # children have four waves - so time-between-waves is not a node.)
    ("nvma", "wgain"),
    ("behav", "wgain"),
    ("ses", "wgain"),
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
}


def causal_dag():
    """Return the LRP65 causal DAG as a ``graphviz.Digraph``.

    Latent general ability ``g`` (dashed circle) drives the correlated baselines;
    the dashed ``-> gain`` edges from non-verbal MA and behaviour are the
    associations the adjusted model is built to test.
    """
    # Lazy import: ``plot_utils`` pulls in the plotting stack (networkx etc.),
    # which the fit path does not need. Keeping it here lets the dispatcher
    # import ``lrp65`` (and fit) with only the sampler dependencies present.
    from language_reading_predictors.plot_utils import draw_causal_graph

    return draw_causal_graph(
        _EDGE_LIST,
        node_props=_NODE_PROPS,
        edge_props=_EDGE_PROPS,
        graph_direction="TB",
    )


def render_dag(output_dir: str | None = None, *, fmt: str = "svg") -> str:
    """Render the causal DAG to ``{output_dir}/dag.{fmt}`` and return the path.

    Defaults to ``docs/models/lrp65/`` so the DAG can be committed and reviewed
    before any fitting (the Step-1 gate).
    """
    if output_dir is None:
        output_dir = os.path.join(DOCS_DIR, "models", "lrp65")
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
    from language_reading_predictors.statistical_models.context import ModelSpec

    return ModelSpec(
        model_id="lrp65",
        kind="adjusted",
        title="Adjusted model: independent baseline predictors of word-reading gain",
        outcome_symbol="W",
        adjustment=["L", "lang", "B", "A", "W_pre", "blocks", "behav"],
        extra={
            # Headline = genuinely between-child: one row per child, T1 baselines,
            # full-study gain (W at last wave conditioned on W_T1). No phase
            # dimension and no child random intercept (one obs per child).
            "design": "between_child",
            "post_time": 4,
            # Standardised T1 predictors of interest (letter sounds, blending).
            "predictor_symbols": ["L", "B"],
            # Equal-weight language composite (receptive + expressive + concepts).
            "language_composite_symbols": ["R", "E", "F"],
            "use_age_predictor": True,
            # Continuous covariates entered to test independent signal.
            "covariates": ["blocks", "behav"],
            # SES sensitivity fit on the SES-complete subset (not the headline model).
            "ses_covariates": ["mumedupost16"],
            # Fixed weakly-informative slope prior + the sensitivity sweep that
            # checks the which-predictors-clear-zero conclusion is stable.
            "predictor_slope_sigma": 0.5,
            "prior_sensitivity_sigmas": [0.3, 0.7],
        },
    )


def fit(config: str = "dev"):
    # Lazy import: ``fit_adjusted`` is added in Step 2, after the DAG review.
    from language_reading_predictors.statistical_models.pipeline import fit_adjusted

    return fit_adjusted(get_spec(), config=config)


if __name__ == "__main__":  # pragma: no cover
    print(render_dag())
