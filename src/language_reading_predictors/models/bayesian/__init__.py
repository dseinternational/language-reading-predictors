# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Bayesian modelling subpackage.

Declarative DAG-specified models fit with PyMC using HSGP-approximated
Gaussian processes on edges. Shared pipeline steps live in
:mod:`base_pipeline`; concrete likelihood combinations (e.g. beta-binomial
intermediates + beta-binomial outcome) are implemented as subclasses of
:class:`BayesianPipeline` with their own ``build_model`` method.
"""

from language_reading_predictors.models.bayesian.definition import (
    BayesianDAGSpec,
    NodeSpec,
    ParentSpec,
)

__all__ = ["BayesianDAGSpec", "NodeSpec", "ParentSpec"]
