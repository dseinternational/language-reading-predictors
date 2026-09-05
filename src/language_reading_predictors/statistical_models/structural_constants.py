# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Persist constraints established by a model's distribution, never by its draws."""

import json

import pymc as pm
import xarray as xr

from pymc.distributions.multivariate import LKJCorrRV

CORRELATION_CHOLESKY_ATTR = "model_correlation_cholesky_variables"


def record_structural_constraints(posterior: xr.Dataset | xr.DataTree, model: pm.Model) -> None:
    """Identify the actual correlation-Cholesky RVs and save their names.

    PyMC's LKJCorrRV returns a correlation Cholesky matrix. Its upper triangle
    is zero and its first diagonal entry is one. An ordinary matrix-valued RV
    has neither constraint, even when it happens to have the same sampled values.
    NetCDF retains this JSON attribute for later stored-trace diagnostics.
    """
    names = [rv.name for rv in model.free_RVs if rv.owner is not None and isinstance(rv.owner.op, LKJCorrRV)]
    posterior.attrs[CORRELATION_CHOLESKY_ATTR] = json.dumps(sorted(names))


def correlation_cholesky_variables(posterior: xr.Dataset | xr.DataTree) -> frozenset[str]:
    """Missing or malformed constraint evidence grants no exemption."""
    try:
        names = json.loads(posterior.attrs.get(CORRELATION_CHOLESKY_ATTR, "[]"))
    except TypeError, ValueError:
        return frozenset()
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        return frozenset()
    return frozenset(names)
