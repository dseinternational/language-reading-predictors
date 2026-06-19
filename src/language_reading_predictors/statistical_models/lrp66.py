# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP66 - Latent general-ability model: general vs specific predictors of gain.

Tier 2 follow-up to LRP65. LRP65 used mutual adjustment to ask which baseline
skills go with word-reading gain, and found letter sounds + language retain
signal while non-verbal MA collapses. But mutual adjustment cannot cleanly
separate "this child has high *general ability*" from "this child is
specifically good at letter sounds" when the skills are noisy indicators of a
shared factor. LRP66 fits the latent general ability ``g`` the LRP65 DAG only
drew, and splits the question in two.

Estimand (between-child, same design as LRP65)
----------------------------------------------
One row per child (``phase_mode="span"``); outcome = word reading at the last
wave conditioned on its T1 baseline (the gain framing). Two quantities:

- **g -> gain** (``beta_g``): how much of "who gains" is general ability.
- **skill -> gain beyond g** (the ``observed_direct`` coefficients given ``g``,
  e.g. ``beta_L`` for letter sounds, ``beta_lang`` for the language composite):
  does a *specific* skill predict gain over and above general ability? This is
  the actionable "is this a direct teaching target, or just an ability marker?"
  question that mutual adjustment alone cannot answer.

Model
-----
- Measurement: a one-factor Gaussian CFA. ``g ~ Normal(0, 1)`` (scale fixed)
  loads on the standardised T1 skill indicators (letter sounds, ROWPVT, EOWPVT,
  CELF, blending, non-verbal MA) with **positive** loadings (all positive-manifold
  measures; this also fixes orientation). A loading is the indicator-``g``
  correlation; its square is the share of that skill explained by ``g``.
- Structural: a Beta-Binomial leg on ``W_post`` conditioned on ``W_pre`` via
  ``gamma_own``, regressed on ``g`` + the observed-direct skills + age.
- Headline = observed-direct beyond-g (identifiable at n ~ 51). Robustness arm
  (``use_language_specific``) adds an orthogonal language-specific latent factor
  on the three language measures and routes its beyond-g effect through
  ``beta_lang_specific`` - the gold-standard decomposition, but fragile at this n
  (priors do real work; expect some divergences). Reported as a sensitivity
  check, not the headline.

Honest framing
--------------
Between-child associations, n ~ 51 - **not** causal effects (LRP52/55 carry the
causal claim about the programme). A latent model at this n is fragile, so LRP66
is **triangulation against LRP65, not a definitive decomposition**: agreement
strengthens the story; divergence is itself the finding. A bonus of building
``g`` from several indicators is partial measurement-error correction, so the
specific-skill effects are less attenuated than LRP65's observed predictors.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from language_reading_predictors.plot_utils import draw_causal_graph
from language_reading_predictors.statistical_models.environment import DOCS_DIR

if TYPE_CHECKING:
    from language_reading_predictors.statistical_models.context import ModelSpec


# ---------------------------------------------------------------------------
# Causal DAG (g is now a *fitted* latent node, unlike LRP65 where it was drawn)
# ---------------------------------------------------------------------------

_EDGE_LIST: list[tuple[str, str]] = [
    # Measurement: general ability drives (loads on) the T1 skill indicators.
    ("g", "ls"),
    ("g", "lang"),
    ("g", "blend"),
    ("g", "nvma"),
    # Structural: the general-ability path to gain.
    ("g", "wgain"),
    # Beyond-g specific paths (the hypotheses under test).
    ("ls", "wgain"),
    ("lang", "wgain"),
    # Covariate and baseline conditioning.
    ("age", "wgain"),
    ("wpre", "wgain"),
]

_NODE_PROPS: dict[str, dict[str, str]] = {
    "g": {
        "label": "General ability (g)\\n[latent, FITTED]",
        "shape": "circle",
        "style": "filled",
        "fillcolor": "#eef3e8",
        "color": "grey45",
    },
    "ls": {"label": "Letter sounds (T1)"},
    "lang": {"label": "Language (T1)\\nROWPVT/EOWPVT/CELF"},
    "blend": {"label": "Blending (T1)"},
    "nvma": {"label": "Non-verbal MA (T1)\\nblock design"},
    "age": {"label": "Age (T1)"},
    "wpre": {"label": "Word reading (T1)\\nW_pre", "shape": "box"},
    "wgain": {
        "label": "Word-reading gain\\n(W_last | W_T1)",
        "shape": "box",
        "style": "filled",
        "fillcolor": "#e8eef7",
    },
}

# The beyond-g specific paths are what the model adjudicates: dashed.
_EDGE_PROPS: dict[tuple[str, str], dict[str, str]] = {
    ("ls", "wgain"): {"style": "dashed", "color": "grey45"},
    ("lang", "wgain"): {"style": "dashed", "color": "grey45"},
}


def causal_dag():
    """Return the LRP66 DAG as a ``graphviz.Digraph``.

    Latent general ability ``g`` (now *fitted*) loads on the T1 skills and has a
    general-ability path to gain; the dashed skill -> gain edges are the
    beyond-``g`` specific associations the model estimates.
    """
    return draw_causal_graph(
        _EDGE_LIST,
        node_props=_NODE_PROPS,
        edge_props=_EDGE_PROPS,
        graph_direction="TB",
    )


def render_dag(output_dir: str | None = None, *, fmt: str = "svg") -> str:
    """Render the causal DAG to ``{output_dir}/dag.{fmt}`` (default docs/models/lrp66)."""
    if output_dir is None:
        output_dir = os.path.join(DOCS_DIR, "models", "lrp66")
    os.makedirs(output_dir, exist_ok=True)
    g = causal_dag()
    g.render(filename=os.path.join(output_dir, "dag"), format=fmt, cleanup=True)
    return os.path.join(output_dir, f"dag.{fmt}")


# ---------------------------------------------------------------------------
# Model specification (consumed by ``fit_factor``)
# ---------------------------------------------------------------------------


def get_spec() -> "ModelSpec":
    """Return the LRP66 model specification (lazy import; see lrp65 for rationale)."""
    from language_reading_predictors.statistical_models.context import ModelSpec

    return ModelSpec(
        model_id="lrp66",
        kind="factor",
        title="Latent general-ability model: general vs specific predictors of gain",
        outcome_symbol="W",
        adjustment=["g", "L", "lang", "age", "W_pre"],
        extra={
            "design": "between_child",
            "post_time": 4,
            "indicator_symbols": ["L", "R", "E", "F", "B"],
            "indicator_covariates": ["blocks"],
            "language_composite_symbols": ["R", "E", "F"],
            "observed_direct": ["L", "lang", "age"],
            "predictor_slope_sigma": 0.5,
            # Robustness arm: orthogonal language-specific factor (fragile at this
            # n) — run at a higher target_accept and reported as a sensitivity check.
            "language_specific_symbols": ["R", "E", "F"],
            "latent_target_accept": 0.99,
        },
    )


def fit(config: str = "dev"):
    from language_reading_predictors.statistical_models.pipeline import fit_factor

    return fit_factor(get_spec(), config=config)


if __name__ == "__main__":  # pragma: no cover
    print(render_dag())
