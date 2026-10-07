# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Leave-one-out refits (``reloo``) for the mechanism family.

PSIS-LOO approximates leave-one-out cross-validation by importance sampling from the
full-data posterior. That approximation fails when an observation is influential
enough that the leave-one-out posterior is far from it, which ArviZ reports as a
Pareto shape estimate above ``good_k``.

``reloo`` refits the model without each flagged point and estimates its held-out
log predictive density, leaving PSIS to handle the rest. Refit estimates still
have posterior Monte Carlo error.

Refits use :func:`mechanism.build_mechanism_for_plan`, as primary fits do.
Alignment checks reject changed data, priors or model designs before replacing
pointwise scores.

The resulting comparison remains subject to the suite's ``|elpd_diff| < 4``
interpretability rule. Refitting flagged points does not settle the comparison.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pymc as pm
from arviz_stats.loo.wrapper import SamplingWrapper
from pymc.stats import compute_log_likelihood, compute_log_prior
from dse_research_utils.statistics.diagnostics import (
    BFMI_THRESHOLD,
    ESS_THRESHOLD,
    RHAT_MAX,
)

from language_reading_predictors.statistical_models import mechanism as _mechanism
from language_reading_predictors.statistical_models.preprocessing import _subset
from language_reading_predictors.statistical_models.fitted_payloads import (
    MechanismDesign,
    MechanismPayload,
)
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.preprocessing import PreparedData
from language_reading_predictors.statistical_models.sampling_quality import (
    sampling_quality,
)

__all__ = ["MechanismSamplingWrapper", "RefitPlan", "build_mechanism_wrapper"]

# Refits share the fit-time R-hat, ESS and available-BFMI thresholds.
GATE_MAX_DIVERGENCES = 0


def _as_dataset(group: Any) -> Any:
    """Return the xarray Dataset behind an inference-data group.

    ArviZ 1.x hands back a ``DataTree`` whose ``.items()`` yields *child nodes*, not
    variables, so iterating it directly and reading ``.values`` raises rather than
    comparing anything. ``.dataset`` is the accessor that reaches the variables; a
    plain Dataset (what ``compute_log_prior`` returns) passes through unchanged.
    """
    inner = getattr(group, "dataset", group)
    return inner if hasattr(inner, "data_vars") else group


def _observed_variable_name(model: pm.Model, idata_orig: Any, model_id: str) -> str:
    """Name of the model's single observed node, verified against the stored trace.

    Derive the name because Beta-Binomial models use ``y_post`` and floor models
    use ``y_offfloor``.

    Refuses anything the wrapper's single-variable pointwise indexing cannot represent:
    a multi-outcome model has one ``obs_id`` axis per outcome, and there is no
    well-defined single row for ``reloo`` to hold out.
    """
    observed = [rv.name for rv in model.observed_RVs]
    if len(observed) != 1:
        raise ValueError(
            f"{model_id}: rebuilt model has {len(observed)} observed variables "
            f"({sorted(observed) if observed else 'none'}); the exact-refit repair "
            "splices a single pointwise log-likelihood and cannot hold out a row of a "
            "multi-outcome model — refusing to refit"
        )
    name = observed[0]
    stored = set(_as_dataset(idata_orig.log_likelihood).data_vars)
    if len(stored) != 1:
        raise ValueError(
            f"{model_id}: stored log_likelihood group holds {sorted(stored)}; the "
            "exact-refit repair cannot tell which variable the Pareto-k indices refer "
            "to — refusing to refit"
        )
    if name not in stored:
        raise ValueError(
            f"{model_id}: rebuilt model observes {name!r} but the stored trace holds "
            f"{sorted(stored)}; the construction path has drifted from the one that "
            "produced this fit — refusing to refit"
        )
    return name


@dataclass(frozen=True)
class RefitPlan:
    """Sampler settings for a refit, taken from the original fit's ``config.json``.

    Preserve the recorded sampling effort. Each refit must still pass its own
    convergence checks; matching settings do not guarantee convergence.
    """

    draws: int
    tune: int
    chains: int
    target_accept: float
    random_seed: int | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> RefitPlan:
        sampling = dict(config.get("sampling") or {})
        missing = {"draws", "tune", "chains", "target_accept"} - set(sampling)
        if missing:
            raise ValueError(
                f"config.json sampling block is missing {sorted(missing)}; cannot "
                "reproduce the original sampler settings for a refit"
            )
        return cls(
            draws=int(sampling["draws"]),
            tune=int(sampling["tune"]),
            chains=int(sampling["chains"]),
            target_accept=float(sampling["target_accept"]),
            random_seed=sampling.get("random_seed"),
        )


class MechanismSamplingWrapper(SamplingWrapper):
    """``SamplingWrapper`` over one mechanism model, for :func:`arviz_stats.reloo`.

    ``fitted`` must be ``BuiltModel.prepared`` — the frame the factory actually built
    on *after* its own missing-data keep-mask — because that is the frame the stored
    ``log_likelihood`` / ``pareto_k`` indices refer to. Passing the pre-factory
    ``plan.prepared`` instead would misalign every index whenever the factory dropped
    a row, which it does for most specs in this family.
    """

    def __init__(
        self,
        plan: _mechanism.MechanismPlan,
        fitted: PreparedData,
        idata_orig: Any,
        refit: RefitPlan,
        full_model: pm.Model,
        design: MechanismDesign,
        obs_var: str,
        *,
        progressbar: bool = False,
    ) -> None:
        super().__init__(model=full_model, idata_orig=idata_orig)
        self.plan = plan
        self.fitted = fitted
        self.design = design
        self.refit = refit
        self.full_model = full_model
        # Derived from the built model rather than assumed, so a likelihood change
        # renames the node without silently breaking the splice (see
        # :func:`_observed_variable_name`).
        self.obs_var = obs_var
        self.progressbar = progressbar
        self.n_refits = 0

    # -- SamplingWrapper contract -------------------------------------------------

    def sel_observations(self, idx: int) -> tuple[PreparedData, int]:
        """Split off row ``idx``; returns (data without it, its integer index)."""
        idx = int(idx)
        safe, why = _mechanism.holdout_is_safe(self.fitted, idx)
        if not safe:
            raise ValueError(f"cannot hold out this observation: {why}")
        keep = _mechanism.holdout_mask(self.fitted, idx)
        return _subset(self.fitted, keep, reason="reloo_holdout"), idx

    def sample(self, modified_observed_data: PreparedData) -> Any:
        # Replay the *fit's* design (exposure/moderator standardisation and HSGP
        # boundary) rather than letting the n-1 rows re-derive their own. Without this
        # the refit's basis weights are defined against a slightly different design
        # than the full model that scores the held-out point, and the spliced density
        # is not exact LOO (#438 review).
        built = _mechanism.build_mechanism_for_plan(self.plan, modified_observed_data, frozen_design=self.design)

        # The factory runs its own missing-data keep-mask. On an already-fitted frame
        # it must be a no-op, so anything other than "exactly one row fewer" means the
        # refit is not the fit-minus-one-point and its density must not be spliced in.
        expected = self.fitted.n_obs - 1
        if built.prepared.n_obs != expected:
            raise ValueError(
                f"refit frame has {built.prepared.n_obs} rows, expected {expected}; "
                "the factory dropped rows beyond the held-out point, so the refit is "
                "not comparable to the original fit"
            )
        if built.prepared.n_children != self.fitted.n_children:
            raise ValueError(
                f"refit has {built.prepared.n_children} children vs "
                f"{self.fitted.n_children} in the fit; the child random-effect "
                "dimension changed and the held-out density cannot be evaluated"
            )

        self.n_refits += 1
        with built.model:
            idata = pm.sample(
                draws=self.refit.draws,
                tune=self.refit.tune,
                chains=self.refit.chains,
                target_accept=self.refit.target_accept,
                random_seed=self.refit.random_seed,
                nuts_sampler="nutpie",
                progressbar=self.progressbar,
                compute_convergence_checks=False,
            )
        # Every refit faces the same gate the original fit did. Reusing the original
        # sampler settings does *not* guarantee comparable convergence — the n-1
        # geometry can differ — and a divergent or badly-mixed refit would otherwise be
        # spliced in and the whole comparison marked valid on it. Raising here
        # propagates to ``_reloo_repair``, which abandons the repair and falls back to
        # the per-model table.
        self._assert_refit_converged(idata)
        return idata

    def _assert_refit_converged(self, idata: Any) -> None:
        """Fail the refit unless it clears the suite's sampling-quality thresholds.

        Read unrounded signals from :func:`sampling_quality`, including both bulk
        and tail ESS. This wrapper permits missing BFMI, unlike the subfit gate.
        """
        signals = sampling_quality(idata)
        max_rhat = signals.max_rhat
        min_ess = signals.min_ess
        divergences = signals.n_divergences
        # A missing BFMI is not treated as a failure here (unlike the fit-time sub-fit
        # check), preserving this gate's original policy.
        min_bfmi = np.inf if signals.min_bfmi is None else signals.min_bfmi

        failures = []
        if signals.unassessable:
            failures.append("unassessable parameter diagnostics")
        if divergences is None or divergences > GATE_MAX_DIVERGENCES:
            failures.append(f"{divergences} divergences")
        if not np.isfinite(max_rhat) or max_rhat > RHAT_MAX:
            failures.append(f"max R-hat {max_rhat:.4f}")
        if not np.isfinite(min_ess) or min_ess < ESS_THRESHOLD:
            failures.append(f"min ESS {min_ess:.0f}")
        if min_bfmi < BFMI_THRESHOLD:
            failures.append(f"min BFMI {min_bfmi:.3f}")
        if failures:
            raise ValueError(
                "leave-one-out refit failed the sampling-quality gate ("
                + ", ".join(failures)
                + "); the exact density it would contribute is not trustworthy"
            )

    def get_inference_data(self, fit: Any) -> Any:
        return fit

    def log_likelihood__i(self, excluded_observed_data: int, idata__i: Any) -> Any:
        """Held-out log density of row ``excluded_observed_data`` under the refit.

        Evaluated on the **full** model — the one carrying every fitted row — so the
        held-out point's covariates are present. Its child's random intercept is
        still identified because that child retains its other phases (guaranteed by
        :func:`mechanism.holdout_is_safe`).
        """
        with self.full_model:
            log_lik = compute_log_likelihood(idata__i, extend_inferencedata=False, progressbar=False)
        return log_lik[self.obs_var].isel(obs_id=int(excluded_observed_data))


def build_mechanism_wrapper(
    spec: ModelSpec,
    idata_orig: Any,
    config: dict[str, Any],
    *,
    progressbar: bool = False,
) -> MechanismSamplingWrapper:
    """Assemble a wrapper for ``spec`` against its stored trace.

    Verifies that rebuilding the model reproduces the stored fit before returning: if
    the reconstructed frame does not match the trace's observation count, the plan has
    drifted from what produced the trace and no refit derived from it can be trusted.
    """
    plan = _mechanism.resolve_mechanism_plan(spec)
    raw_design = (config.get("extra") or {}).get("mechanism_design")
    if not isinstance(raw_design, dict):
        raise ValueError(f"{spec.model_id}: missing saved mechanism design; a fresh full fit is required")
    design = MechanismDesign.from_dict(raw_design)
    if not plan.factory_kwargs.get("linear_mechanism", False):
        design.hsgp_kwargs()
    built = _mechanism.build_mechanism_for_plan(plan, frozen_design=design)
    obs_var = _observed_variable_name(built.model, idata_orig, spec.model_id)
    # Take the observation count from the log-likelihood group, which *is* the
    # authoritative observation index for LOO, rather than from a posterior dim that
    # only exists because some deterministic happens to carry ``obs_id``.
    stored_ll = _as_dataset(idata_orig.log_likelihood)[obs_var]
    n_trace = int(stored_ll.sizes["obs_id"])
    if built.prepared.n_obs != n_trace:
        raise ValueError(
            f"{spec.model_id}: rebuilt frame has {built.prepared.n_obs} rows but the "
            f"stored trace has {n_trace}; the construction path has drifted from the "
            "one that produced this fit — refusing to refit"
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        recomputed = compute_log_likelihood(
            idata_orig, model=built.model, extend_inferencedata=False, progressbar=False
        )
    delta = float(np.abs(stored_ll.values - recomputed[obs_var].values).max())
    if not np.isfinite(delta) or delta > 1e-6:
        raise ValueError(
            f"{spec.model_id}: rebuilt model does not reproduce the stored "
            f"log-likelihood (max |delta| = {delta:.3e}); refusing to refit"
        )
    # The likelihood check is blind to the *prior*: a changed prior constructor leaves
    # the conditional log-likelihood identical while making the refit target a
    # different posterior from the one that produced the stored trace. The fits persist
    # a ``log_prior`` group precisely so this is checkable, so compare it too and fail
    # closed when it is absent — an unverifiable prior is not a verified one.
    groups = {g.rstrip("/").split("/")[-1] for g in idata_orig.groups}
    if "log_prior" not in groups:
        raise ValueError(
            f"{spec.model_id}: stored trace has no log_prior group, so the rebuilt "
            "model's priors cannot be verified against the fit — refusing to refit"
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        recomputed_prior = compute_log_prior(
            idata_orig, model=built.model, extend_inferencedata=False, progressbar=False
        )
    stored_prior = _as_dataset(idata_orig.log_prior)
    fresh_prior = _as_dataset(recomputed_prior)
    for name, stored_var in stored_prior.data_vars.items():
        if name not in fresh_prior.data_vars:
            raise ValueError(
                f"{spec.model_id}: rebuilt model has no prior term {name!r} present in "
                "the stored trace; the model specification has drifted — refusing to refit"
            )
        prior_delta = float(np.abs(stored_var.values - fresh_prior[name].values).max())
        if not np.isfinite(prior_delta) or prior_delta > 1e-6:
            raise ValueError(
                f"{spec.model_id}: rebuilt model does not reproduce the stored log-prior "
                f"for {name!r} (max |delta| = {prior_delta:.3e}); the priors have changed "
                "since this fit — refusing to refit"
            )
    # The design the *fit* realised, replayed into every refit so the basis weights a
    # refit produces are defined against the same standardisation and HSGP boundary
    # the full model uses to score the held-out point.
    payload = built.require_payload(MechanismPayload, family="mechanism exact LOO")
    design = payload.design
    if design.hsgp_L is None and not plan.factory_kwargs.get("linear_mechanism", False):
        # No realised boundary on a non-linear model means the phase-specific path,
        # whose per-phase bases this design object does not capture.
        raise ValueError(
            f"{spec.model_id}: phase-specific mechanism curves are not supported by the "
            "exact-refit repair (their per-phase HSGP bases are not pinned)"
        )
    return MechanismSamplingWrapper(
        plan=plan,
        fitted=built.prepared,
        idata_orig=idata_orig,
        refit=RefitPlan.from_config(config),
        full_model=built.model,
        design=design,
        obs_var=obs_var,
        progressbar=progressbar,
    )
