# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Trace reuse must identify executable expressions, not abbreviated displays."""

import numpy as np
import pymc as pm
import pytest
import pytensor.tensor as pt
from pytensor.compile.builders import OpFromGraph

from language_reading_predictors.statistical_models.model_identity import model_design_identity


def _model(transform=lambda x: x, *, x=(1.0, 2.0), prior_sd=1.0):
    with pm.Model(coords={"row": ["a", "b"]}) as model:
        data = pm.Data("x", np.asarray(x), dims="row")
        beta = pm.Normal("beta", sigma=prior_sd)
        eta = pm.Deterministic("eta", transform(beta * data), dims="row")
        pm.Binomial("y", n=10, logit_p=eta, observed=[3, 4], dims="row")
    return model


@pytest.mark.parametrize(
    "transform", [lambda x: -x, lambda x: x + 0.2, lambda x: x * 0.2, lambda x: x[::-1], lambda x: pt.clip(x, -2, 2)]
)
def test_graph_detects_operations_constants_and_indexing(transform):
    original, changed = _model(), _model(transform)
    a, b = model_design_identity(original), model_design_identity(changed)
    assert a.get("structure_sha256") and b.get("structure_sha256"), (a, b)
    assert a["structure_sha256"] != b["structure_sha256"]


def test_identical_rebuilds_and_repeated_hashes_agree():
    model = _model()
    assert model_design_identity(model) == model_design_identity(model)
    assert model_design_identity(model) == model_design_identity(_model())


def test_nested_graph_operators_are_not_abbreviated():
    def operation(sign):
        x = pt.vector("inner_x")
        return OpFromGraph([x], [sign * x])

    a = model_design_identity(_model(operation(1)))
    b = model_design_identity(_model(operation(-1)))
    assert a.get("structure_sha256") and b.get("structure_sha256"), (a, b)
    assert a["structure_sha256"] != b["structure_sha256"]


def test_lkj_nested_graph_identity_is_stable():
    def build():
        with pm.Model() as model:
            pm.LKJCorr("correlation_factor", n=3, eta=2)
        return model

    first, second = model_design_identity(build()), model_design_identity(build())
    assert first.get("structure_sha256"), first
    assert first == second
