# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Beta-binomial / Normal DAG pipeline with HSGP edges.

Builds a joint PyMC model from a :class:`BayesianDAGSpec`:

- Each node contributes one likelihood term.
  - ``beta_binomial``: logit-scale linear predictor → probability → BB(N, p).
  - ``normal``: identity-scale linear predictor → Normal(mu, sigma).
- Each parent contributes a linear coefficient.
  - ``gp`` parents also add a zero-mean HSGP function of the standardised
    input.
  - ``binary`` parents are treated as 0/1-coded contrasts (a Normal prior
    replaces the standard beta prior).
  - ``offset`` parents use the same standardised-linear term as ``linear``
    but are reported separately in the summary (they represent the
    baseline of the outcome itself).

One pipeline, many models: LRP51 and future Bayesian models only need to
declare their DAG.
"""

from __future__ import annotations

import numpy as np
import pymc as pm

from language_reading_predictors.models.bayesian.base_pipeline import (
    BayesianPipeline,
)
from language_reading_predictors.models.bayesian.hsgp import build_gp_edge


class BetaBinomialHSGPPipeline(BayesianPipeline):
    """Joint DAG pipeline with beta-binomial and normal likelihoods."""

    def build_model(self) -> None:
        ctx = self.context
        spec = ctx.dag_spec

        n_obs = len(ctx.df)
        coords = {
            "obs": np.arange(n_obs),
            **{
                f"node_{n.name}": [n.name] for n in spec.nodes
            },
        }

        with pm.Model(coords=coords) as model:
            for node_name in spec.topological_order():
                node = ctx.nodes[node_name]
                spec_node = node.spec

                # Intercept on the link scale.
                b0 = pm.Normal(
                    f"b0__{node_name}",
                    mu=0.0,
                    sigma=spec_node.intercept_sigma,
                )
                eta_linpred = b0 * pm.math.ones(n_obs)

                for parent_name, parent_arr in node.parents.items():
                    p_spec = parent_arr.spec
                    edge = f"{parent_name}_to_{node_name}"
                    x_std = parent_arr.std

                    if p_spec.kind == "binary":
                        # 0/1 parent: use the raw column (not the z-score).
                        x_lin = parent_arr.raw
                        beta = pm.Normal(
                            f"beta__{edge}",
                            mu=0.0,
                            sigma=spec_node.beta_sigma,
                        )
                        eta_linpred = eta_linpred + beta * x_lin
                        continue

                    if p_spec.kind == "gp":
                        # GP edges: no separate linear coefficient. The GP
                        # already captures linear trend as a limiting case
                        # (long length-scale); adding an explicit beta
                        # creates a funnel between beta and the GP's low-
                        # frequency basis coefficients.
                        gp_term = build_gp_edge(
                            name=edge, x_std=x_std, parent=p_spec
                        )
                        eta_linpred = eta_linpred + gp_term
                        continue

                    # "linear" and "offset": single linear coefficient on
                    # the standardised parent.
                    beta = pm.Normal(
                        f"beta__{edge}",
                        mu=0.0,
                        sigma=spec_node.beta_sigma,
                    )
                    eta_linpred = eta_linpred + beta * x_std

                if spec_node.likelihood == "beta_binomial":
                    p = pm.Deterministic(
                        f"p__{node_name}",
                        pm.math.sigmoid(eta_linpred),
                        dims="obs",
                    )
                    # Beta-binomial concentration (higher = closer to binomial)
                    kappa = pm.HalfNormal(
                        f"kappa__{node_name}", sigma=50.0
                    )
                    alpha = p * kappa
                    beta_ = (1.0 - p) * kappa
                    pm.BetaBinomial(
                        f"y__{node_name}",
                        n=int(spec_node.n_trials),
                        alpha=alpha,
                        beta=beta_,
                        observed=node.outcome,
                        dims="obs",
                    )
                elif spec_node.likelihood == "normal":
                    sigma = pm.HalfNormal(
                        f"sigma__{node_name}", sigma=spec_node.sigma_sigma
                    )
                    mu = pm.Deterministic(
                        f"mu__{node_name}", eta_linpred, dims="obs"
                    )
                    pm.Normal(
                        f"y__{node_name}",
                        mu=mu,
                        sigma=sigma,
                        observed=node.outcome,
                        dims="obs",
                    )
                else:
                    msg = (
                        f"Unknown likelihood {spec_node.likelihood!r} "
                        f"for node {node_name!r}."
                    )
                    raise ValueError(msg)

        ctx.pm_model = model
