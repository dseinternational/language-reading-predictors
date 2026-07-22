# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Specialised inference algorithms for the longitudinal correlated-factor (LCF) model.

The LCF family needs two numerical routines that PyMC 6.1 cannot supply generically,
because removing the ``LKJCorr`` value transforms renames the compiled value
variables (for example ``trait_corr_chol`` versus ``trait_corr_chol_cholesky``)
while the posterior keeps the random-variable names, so the stock log-likelihood /
log-prior recovery rejects the inputs.

Both routines are pure functions of a fitted trace (plus the built model / its
``extras``) and return xarray objects — no output directory, Quarto template,
Matplotlib session or console dependency — so they can be tested independently of
report publication (issue #394, pillar 7). The family pipeline's LOO-stitching
orchestration composes them with the shared reporting helpers.
"""

from __future__ import annotations

import numpy as np


def child_log_likelihood(trace, built, *, chunk_size: int = 256):
    """Evaluate the LCF's exact per-child MvNormal log likelihood.

    PyMC 6.1 cannot reconstruct this model's log likelihood from its posterior:
    removing the ``LKJCorr`` transforms gives the compiled value variables an extra
    ``_cholesky`` suffix, while the posterior keeps the random-variable names. The
    generic helper therefore rejects the inputs before evaluating the density.

    The fitted trace already contains the exact marginal mean ``mean_z`` and
    covariance ``Sigma_z`` used by every observed-pattern ``MvNormal``. Evaluate
    that density directly, in bounded draw chunks, and return one contribution per
    child. This avoids PyMC's transformed-value plumbing. It can also recover LOO
    from a persisted trace when the exact matching model and data are rebuilt; the
    coordinate and observed-data checks below reject positional drift.
    """
    import xarray as xr

    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive (got {chunk_size})")

    posterior = trace.posterior
    mean_z = posterior["mean_z"].transpose("chain", "draw", "cell")
    sigma_z = posterior["Sigma_z"].transpose(
        "chain", "draw", "cell", "cell_b"
    )
    expected_cells = np.asarray(built.extras["cell_names"], dtype=str)
    for variable, dimension in ((mean_z, "cell"), (sigma_z, "cell"), (sigma_z, "cell_b")):
        actual_cells = np.asarray(variable.coords[dimension].values, dtype=str)
        if not np.array_equal(actual_cells, expected_cells):
            raise ValueError(
                f"LCF posterior {dimension!r} coordinates do not match the rebuilt "
                "model; refusing positional likelihood recovery"
            )
    n_chains = mean_z.sizes["chain"]
    n_draws = mean_z.sizes["draw"]

    groups = []
    all_children: list[int] = []
    for node in built.extras["z_nodes"]:
        children = np.asarray(built.extras["child_of_node"][node], dtype=int)
        cell_indices = np.asarray(
            built.extras["cell_indices_of_node"][node], dtype=int
        )
        observed = np.asarray(built.extras["observed_z_of_node"][node], dtype=float)
        expected_shape = (len(children), len(cell_indices))
        if observed.shape != expected_shape:
            raise ValueError(
                f"{node} observed data have shape {observed.shape}; expected "
                f"{expected_shape} from the child/cell indices"
            )
        if "observed_data" in trace.children:
            if node not in trace.observed_data:
                raise ValueError(f"LCF persisted observed_data is missing {node!r}")
            persisted = np.asarray(trace.observed_data[node].values, dtype=float)
            if persisted.shape != observed.shape or not np.allclose(
                persisted, observed, rtol=0.0, atol=0.0
            ):
                raise ValueError(
                    f"LCF persisted observations for {node!r} do not match the "
                    "rebuilt model data"
                )
        groups.append((children, cell_indices, observed))
        all_children.extend(children.tolist())

    used_children = np.asarray(sorted(all_children), dtype=int)
    if len(np.unique(used_children)) != len(used_children):
        raise ValueError("LCF observed-pattern groups assign a child more than once")
    expected_children = int(built.extras["n_used_children"])
    if len(used_children) != expected_children:
        raise ValueError(
            f"LCF observed-pattern groups cover {len(used_children)} children; "
            f"expected {expected_children}"
        )
    child_column = {child: i for i, child in enumerate(used_children)}

    log_likelihood = np.full(
        (n_chains, n_draws, expected_children), np.nan, dtype=float
    )
    log_2pi = np.log(2.0 * np.pi)
    for chain in range(n_chains):
        for draw_start in range(0, n_draws, chunk_size):
            draw_stop = min(draw_start + chunk_size, n_draws)
            means = np.asarray(
                mean_z.isel(chain=chain, draw=slice(draw_start, draw_stop))
            )
            covariances = np.asarray(
                sigma_z.isel(chain=chain, draw=slice(draw_start, draw_stop))
            )
            for children, cell_indices, observed in groups:
                covariance = covariances[:, cell_indices[:, None], cell_indices]
                mean = means[:, cell_indices]
                chol = np.linalg.cholesky(covariance)
                residual = observed[None, :, :] - mean[:, None, :]
                whitened = np.linalg.solve(chol, np.swapaxes(residual, 1, 2))
                quadratic = np.sum(whitened**2, axis=1)
                log_determinant = 2.0 * np.sum(
                    np.log(np.diagonal(chol, axis1=1, axis2=2)), axis=1
                )
                values = -0.5 * (
                    len(cell_indices) * log_2pi
                    + log_determinant[:, None]
                    + quadratic
                )
                columns = np.asarray(
                    [child_column[int(child)] for child in children], dtype=int
                )
                # Advanced indexing places the child dimension first on the target.
                log_likelihood[chain, draw_start:draw_stop, columns] = values.T

    if not np.isfinite(log_likelihood).all():
        raise ValueError("LCF per-child log likelihood contains non-finite values")

    return xr.DataArray(
        log_likelihood,
        dims=("chain", "draw", "child_lcf"),
        coords={
            "chain": mean_z.coords["chain"],
            "draw": mean_z.coords["draw"],
            "child_lcf": used_children,
        },
        name="lcf_child",
    )


def log_prior(trace, model):
    """Compute exact constrained-scale LCF log-prior terms.

    PyMC 6.1's generic ``compute_log_prior`` removes transforms but then supplies
    the posterior's random-variable names to the resulting value variables. For
    ``LKJCorr`` those names differ (for example ``trait_corr_chol`` versus
    ``trait_corr_chol_cholesky``), so the generic helper fails before evaluation.

    Isolate the version-sensitive workaround here: build the same transform-free
    model PyMC uses, rename constrained posterior inputs to its value-variable names,
    and apply the compiled elementwise prior density over chain and draw. Keeping
    the component terms (rather than only their sum) preserves the standard
    ``log_prior`` contract used by power-scaling sensitivity diagnostics.
    """
    from pymc.backends.arviz import (
        apply_function_over_dataset,
        coords_and_dims_for_inferencedata,
    )
    from pymc.model.transform.conditioning import remove_value_transforms

    posterior_node = trace.posterior
    posterior = (
        posterior_node.to_dataset()
        if hasattr(posterior_node, "to_dataset")
        else posterior_node
    )
    original_names = [rv.name for rv in model.free_RVs]
    missing = sorted(set(original_names) - set(posterior.data_vars))
    if missing:
        raise ValueError(f"LCF posterior is missing free variables: {missing}")

    untransformed_model = remove_value_transforms(model)
    prior_rvs = list(untransformed_model.free_RVs)
    inputs = posterior[original_names]
    rename = {
        rv.name: untransformed_model.rvs_to_values[rv].name
        for rv in prior_rvs
        if rv.name != untransformed_model.rvs_to_values[rv].name
    }
    inputs = inputs.rename(rename)
    inputs = inputs.astype(
        {
            value.name: value.type.dtype
            for value in untransformed_model.value_vars
        },
        copy=False,
    )
    logp_fn = untransformed_model.compile_fn(
        inputs=untransformed_model.value_vars,
        outs=untransformed_model.logp(vars=prior_rvs, sum=False),
        on_unused_input="ignore",
    )
    coords, dims = coords_and_dims_for_inferencedata(untransformed_model)
    result = apply_function_over_dataset(
        logp_fn,
        inputs,
        output_var_names=[rv.name for rv in prior_rvs],
        sample_dims=("chain", "draw"),
        coords=coords,
        dims=dims,
        progressbar=False,
    )
    if any(not np.isfinite(variable).all() for variable in result.data_vars.values()):
        raise ValueError("LCF log-prior contains non-finite values")
    return result
