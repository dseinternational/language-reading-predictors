# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""One runner for every secondary and sensitivity sub-fit (#394 design point 5).

A *sub-fit* is any posterior this suite samples besides the primary one: the
floor-rule graded and hurdle cross-checks, the mediation temporal-ordering
sensitivity, the adjusted family's bivariate refits, prior-slope sweep and SES
complete-case refit, and the concurrent / joint-mechanism non-anchor waves. They
publish numbers into the report exactly as the primary fit does, but they bypass
``diagnostics_summary.json`` — the convergence gate covers ``ctx.trace`` only.

Before this module the eleven sub-fit call sites split three ways. Eight went
through ``diagnostics.sample_subfit``, which sampled and returned a convergence
verdict the caller then had to remember to publish. Three — the two floor-rule
secondaries and the mediation t3 sensitivity — spelled out their own
``pm.sample`` call, so a sampling argument could drift between them unnoticed.
None of them recorded what data the sub-fit was actually fitted to, or at what
sampling settings; and a convergence check that could not be *computed* returned
``converged=None``, which lands in a published CSV as an empty cell —
indistinguishable from a column that was never populated at all.

:func:`run_subfit` is now the only way to sample a sub-fit. It returns a typed
:class:`SubfitResult` carrying the trace, the convergence verdict verbatim, the
fitted-data identity, the sampling settings actually used, the persisted trace
filename and a structured failure classification, and it appends a row to
``subfit_provenance.csv`` on every call. The families keep their scientific loops
explicit — which wave, which predictor, which prior width is being fitted stays
in the family module — and delegate only the sampling, convergence, persistence
and provenance mechanics. The log keeps trace-free copies, so recording a
sub-fit's provenance does not keep its posterior alive for the rest of the run.

The provenance table is rewritten by each call rather than assembled at
finalisation on purpose: a record of what a run did is worth least if it is the
first casualty of the failure it exists to document.

Deliberately absent: sub-fit PSIS-LOO. The typed result the issue sketches lists
it as optional, and no sub-fit in the suite computes one — a sensitivity refit is
read against the primary's estimate, not ranked against it by predictive
performance. Wiring it speculatively would add an untested branch; when a family
needs it, it belongs in :func:`run_subfit` beside the posterior-predictive step.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from language_reading_predictors.statistical_models.artifacts import (
    record_artifact,
    save_table,
)

SubfitRole = Literal[
    "secondary",
    "sensitivity",
    "bivariate",
    "prior_sweep",
    "wave",
]
"""What a sub-fit is for, so the provenance table groups by purpose.

``secondary`` a flagged cross-check published beside the headline estimand;
``sensitivity`` a refit that varies an analysis choice (rows, timing);
``bivariate`` an unadjusted single-predictor refit; ``prior_sweep`` the same
model at another prior width; ``wave`` a non-anchor timepoint of a multi-wave
family.
"""

ConvergenceScope = Literal["free_rvs", "all"]
"""Which parameters the sub-fit convergence check scans.

``free_rvs`` restricts the R-hat / ESS scan to the sub-model's free random
variables — a well-mixed focal coefficient cannot rescue a non-mixing nuisance
parameter, because the nuisances determine the fitted mean (#341). ``all`` leaves
``var_names`` unset, so ArviZ summarises everything it reports for the trace,
deterministics included; that is the stricter scan, and the mediation
temporal-ordering sensitivity has always used it.
"""

PROVENANCE_TABLE = "subfit_provenance"
"""Logical name (and CSV stem) of the per-fit sub-fit provenance record."""

NUTS_SAMPLER = "nutpie"
"""The one NUTS backend every fit in the suite uses, primary and sub-fit alike."""


@dataclass(frozen=True, slots=True)
class SubfitData:
    """What data a sub-fit was actually fitted to.

    Sub-fits differ from the primary fit in their *rows* as often as in their
    model: the SES refit drops children with missing SES, a bivariate refit keeps
    the rows its one predictor is observed on, and a wave sub-fit takes one
    timepoint. ``n_children`` / ``n_obs`` come from the prepared frame the factory
    returned — the post-drop frame, not what was requested.

    ``digest`` hashes the **row keys** (subject identifiers, and the phase or wave
    key where the frame carries one) alongside the observed arrays, their dtypes
    and their shapes. Hashing the observations alone would not identify the rows:
    on a floored outcome, two different subsets of children can share an identical
    ordered score vector — heavily-floored measures make that likely, not exotic —
    and the digests would then agree while the fitted children differed.
    ``identity_keys`` names what actually went into the hash, so the digest is
    self-describing rather than a number whose meaning has to be inferred.
    """

    n_children: int | None
    n_obs: int | None
    observed: tuple[tuple[str, tuple[int, ...]], ...]
    """``(observed node name, shape)`` for every observed RV, in model order."""
    identity_keys: tuple[str, ...]
    """The row-key arrays hashed into ``digest``, in hash order."""
    digest: str | None
    """Short SHA-256 over the row keys and observed arrays, or ``None``."""
    digest_error: str | None
    """Why the digest is absent, when it is."""


@dataclass(frozen=True, slots=True)
class SubfitResult:
    """The outcome of one sub-fit: trace, verdict, provenance.

    ``convergence`` is the :func:`diagnostics.subfit_convergence` dict verbatim,
    key order included, because families merge it straight into a published row
    and the CSV column order is part of the artefact schema.
    """

    label: str
    role: SubfitRole
    trace: Any
    convergence: dict[str, Any]
    sampling: dict[str, Any]
    data: SubfitData
    convergence_scope: ConvergenceScope
    posterior_predictive: tuple[str, ...] = ()
    trace_file: str | None = None
    failure_type: str | None = None
    failure: str | None = None

    @property
    def converged(self) -> bool | None:
        """``True`` passed, ``False`` failed, ``None`` could not be checked."""
        return self.convergence.get("converged")

    def provenance_row(self) -> dict[str, Any]:
        """One flat row for ``subfit_provenance.csv``."""
        return {
            "label": self.label,
            "role": self.role,
            "converged": self.convergence.get("converged"),
            "max_rhat": self.convergence.get("max_rhat"),
            "min_ess": self.convergence.get("min_ess"),
            "min_bfmi": self.convergence.get("min_bfmi"),
            "n_divergences": self.convergence.get("n_divergences"),
            "convergence_scope": self.convergence_scope,
            "n_children": self.data.n_children,
            "n_obs": self.data.n_obs,
            # Shapes travel with the names: an auditor comparing two rows has to be
            # able to see that differently shaped observations *are* different.
            "observed_nodes": ", ".join(
                f"{name}[{', '.join(str(n) for n in shape)}]"
                for name, shape in self.data.observed
            ),
            "identity_keys": ", ".join(self.data.identity_keys),
            "data_digest": self.data.digest,
            "data_digest_error": self.data.digest_error,
            "sampler": self.sampling.get("sampler"),
            "draws": self.sampling.get("draws"),
            "tune": self.sampling.get("tune"),
            "chains": self.sampling.get("chains"),
            "cores": self.sampling.get("cores"),
            "target_accept": self.sampling.get("target_accept"),
            "random_seed": self.sampling.get("random_seed"),
            "posterior_predictive": ", ".join(self.posterior_predictive),
            "trace_file": self.trace_file,
            "failure_type": self.failure_type,
            "failure": self.failure,
        }


@dataclass(slots=True)
class SubfitLog:
    """Per-fit provenance record, in the order the family ran the sub-fits.

    The log holds **trace-free** copies. A family runs its sub-fits in a loop and
    lets each trace go once the summary is computed — the concurrent family alone
    runs twenty-seven at reporting tier, several carrying a per-child random-effect
    vector over 36,000 draws — so a log that kept every ``InferenceData`` alive for
    the lifetime of the fit context would multiply peak memory for no gain. The
    caller still receives the full :class:`SubfitResult` from :func:`run_subfit`.
    """

    results: list[SubfitResult] = field(default_factory=list)

    def record(self, result: SubfitResult) -> None:
        self.results.append(replace(result, trace=None))

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([r.provenance_row() for r in self.results])


PROVENANCE_COLUMNS: tuple[str, ...] = tuple(
    SubfitResult(
        label="",
        role="secondary",
        trace=None,
        convergence={},
        sampling={},
        data=SubfitData(
            n_children=None,
            n_obs=None,
            observed=(),
            identity_keys=(),
            digest=None,
            digest_error=None,
        ),
        convergence_scope="free_rvs",
    ).provenance_row()
)
"""The provenance schema, derived from the row builder so the two cannot drift."""


def _log_of(ctx: Any) -> SubfitLog | None:
    """The context's sub-fit log, or ``None`` for a minimal test/sweep context."""
    log = getattr(ctx, "subfits", None)
    return log if isinstance(log, SubfitLog) else None


def _observed_arrays(model: Any) -> list[tuple[str, np.ndarray]]:
    """The observed array behind each observed RV, in model order.

    PyMC stores the observation as the RV's value variable: a ``TensorConstant``
    for a plain array, a shared variable when the model was built with
    ``pm.Data``. Anything else is skipped by the caller with a recorded reason
    rather than guessed at.
    """
    out: list[tuple[str, np.ndarray]] = []
    for rv in model.observed_RVs:
        value = model.rvs_to_values[rv]
        data = getattr(value, "data", None)
        if data is None and hasattr(value, "get_value"):
            data = value.get_value(borrow=True)
        if data is None:
            raise TypeError(
                f"observed node {rv.name!r} holds a "
                f"{type(value).__name__} with no readable array"
            )
        out.append((rv.name, np.asarray(data)))
    return out


#: Row-key attributes hashed into the digest when the prepared frame carries them.
#: ``subject_ids`` identifies *which children*, and is present on all three frame
#: types (``PreparedData``, ``WavePanel``, ``LongitudinalPanel``); ``phase`` and
#: ``waves`` identify *which transition or timepoint*, and only one of them exists
#: on any given frame. Absent attributes are skipped and simply do not appear in
#: ``identity_keys``.
_ROW_KEY_ATTRIBUTES: tuple[str, ...] = ("subject_ids", "phase", "waves")


def _row_key_arrays(prepared: Any) -> list[tuple[str, np.ndarray]]:
    """The identity arrays a prepared frame carries, in declared order."""
    keys: list[tuple[str, np.ndarray]] = []
    for name in _ROW_KEY_ATTRIBUTES:
        value = getattr(prepared, name, None)
        if value is None:
            continue
        array = np.asarray(value)
        if array.size:
            keys.append((name, array))
    return keys


def _hash_array(hasher: Any, name: str, array: np.ndarray) -> None:
    """Fold one named array into the digest, shape and dtype included.

    The shape matters on its own: without it, an array and a reshaping of it hash
    identically, so a ``(4,)`` observation and a ``(2, 2)`` one would claim to be
    the same data.
    """
    hasher.update(name.encode("utf-8"))
    hasher.update(str(array.dtype).encode("utf-8"))
    hasher.update(repr(array.shape).encode("utf-8"))
    # ``str`` / ``object`` subject identifiers have no meaningful buffer, so those
    # arrays are folded in by their text form instead.
    if array.dtype.kind in "OUS":
        hasher.update("|".join(str(v) for v in array.ravel()).encode("utf-8"))
    else:
        hasher.update(np.ascontiguousarray(array).tobytes())


def describe_fitted_data(built: Any) -> SubfitData:
    """Identify the rows and observations a built sub-model will be fitted to.

    The digest covers the prepared frame's row keys *and* the model's observed
    arrays. Observations alone are not an identity: a floored outcome makes it
    entirely possible for two different subsets of children to share one ordered
    score vector, and a digest over the scores would then declare two different
    analysis populations identical.
    """
    prepared = getattr(built, "prepared", None)
    n_children = getattr(prepared, "n_children", None)
    n_obs = getattr(prepared, "n_obs", None)
    row_keys = _row_key_arrays(prepared)
    try:
        arrays = _observed_arrays(built.model)
    except Exception as exc:  # noqa: BLE001 - provenance must not fail a fit
        # No partial digest: one computed over the row keys alone would sit in the
        # same column as full ones and invite a comparison that does not hold.
        # A blank digest with a reason beside it cannot be misread.
        return SubfitData(
            n_children=_as_int(n_children),
            n_obs=_as_int(n_obs),
            observed=(),
            identity_keys=(),
            digest=None,
            digest_error=f"{type(exc).__name__}: {exc}",
        )
    hasher = hashlib.sha256()
    for name, array in row_keys:
        _hash_array(hasher, f"key:{name}", array)
    observed: list[tuple[str, tuple[int, ...]]] = []
    for name, array in arrays:
        observed.append((name, tuple(int(n) for n in array.shape)))
        _hash_array(hasher, name, array)
    identified = bool(arrays) or bool(row_keys)
    return SubfitData(
        n_children=_as_int(n_children),
        n_obs=_as_int(n_obs),
        observed=tuple(observed),
        identity_keys=tuple(name for name, _ in row_keys),
        digest=hasher.hexdigest()[:16] if identified else None,
        digest_error=(
            None
            if identified
            else "the model has no observed nodes and the frame no row keys"
        ),
    )


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _classify_failure(convergence: dict[str, Any]) -> tuple[str | None, str | None]:
    """Name the reason a convergence verdict is missing, when it is.

    ``subfit_convergence`` absorbs its own failures: it warns and returns
    ``converged=None``. In a published table that is an empty cell, which reads
    the same as a column nobody filled in. Two distinguishable cases hide there,
    and the returned dict is enough to tell them apart — no change to the check
    itself is needed.
    """
    if convergence.get("converged") is not None:
        return None, None
    if convergence.get("max_rhat") is None:
        return (
            "convergence_unavailable",
            "the R-hat / ESS summary could not be computed for this sub-fit",
        )
    return (
        "divergences_unavailable",
        "sample_stats carries no 'diverging' variable, so the gate cannot be evaluated",
    )


def run_subfit(
    ctx: Any,
    built: Any,
    *,
    label: str,
    role: SubfitRole,
    posterior_predictive: Sequence[str] | None = None,
    trace_filename: str | None = None,
    convergence_scope: ConvergenceScope = "free_rvs",
) -> SubfitResult:
    """Sample one sub-fit, check it, record its provenance, return it typed.

    Mirrors :func:`diagnostics.sample_posterior` but stays standalone: the sub-fit
    trace never becomes ``ctx.trace`` and never overwrites ``trace.nc``, because
    the primary fit's diagnostics and report are keyed to those. ``ctx.sampling``
    supplies the settings, so a sub-fit is sampled exactly as its parent run was.

    ``posterior_predictive`` draws the named nodes inside the same model context
    (the floor-rule secondaries need ``y_post`` for their predictive summaries).
    ``trace_filename`` persists the sub-fit trace next to the primary's and
    records it on the artefact manifest, which is what makes a published
    secondary estimate independently auditable.

    Sampling failures propagate. A sub-fit that cannot be sampled is not a
    warn-and-continue artefact, and the two callers that tolerate one (the SES
    complete-case refit, whose covariates may be entirely missing) already guard
    the whole load-build-sample sequence themselves.
    """
    import pymc as pm

    # Local, and deliberately so: ``diagnostics`` imports ``context``, which
    # imports :class:`SubfitLog` from here. Importing the check at module level
    # would close that loop and put the whole shared layer back in one file.
    from language_reading_predictors.statistical_models.diagnostics import (
        subfit_convergence,
    )

    s = ctx.sampling
    sampling = {
        "sampler": NUTS_SAMPLER,
        "draws": int(s.draws),
        "tune": int(s.tune),
        "chains": int(s.chains),
        "cores": int(s.cores),
        "target_accept": float(s.target_accept),
        "random_seed": s.random_seed,
    }
    data = describe_fitted_data(built)
    pp = tuple(posterior_predictive or ())

    with built.model:
        trace = pm.sample(
            draws=s.draws,
            tune=s.tune,
            chains=s.chains,
            cores=s.cores,
            target_accept=s.target_accept,
            nuts_sampler=NUTS_SAMPLER,
            return_inferencedata=True,
            random_seed=s.random_seed,
            progressbar=False,
        )
        if pp:
            trace = pm.sample_posterior_predictive(
                trace,
                var_names=list(pp),
                extend_inferencedata=True,
                random_seed=s.random_seed,
                progressbar=False,
            )

    convergence = subfit_convergence(
        trace,
        label=label,
        var_names=(
            [rv.name for rv in built.model.free_RVs]
            if convergence_scope == "free_rvs"
            else None
        ),
    )
    failure_type, failure = _classify_failure(convergence)

    if trace_filename is not None:
        trace.to_netcdf(os.path.join(ctx.output_dir, trace_filename))
        record_artifact(
            ctx,
            os.path.splitext(trace_filename)[0],
            filename=trace_filename,
            kind="netcdf",
        )

    result = SubfitResult(
        label=label,
        role=role,
        trace=trace,
        convergence=convergence,
        sampling=sampling,
        data=data,
        convergence_scope=convergence_scope,
        posterior_predictive=pp,
        trace_file=trace_filename,
        failure_type=failure_type,
        failure=failure,
    )
    record_subfit(ctx, result)
    return result


def record_subfit(ctx: Any, result: SubfitResult) -> None:
    """Log a sub-fit and rewrite ``subfit_provenance.csv``.

    A no-op beyond the log when the context carries no output directory, so the
    unit tests can exercise the runner's bookkeeping without one.
    """
    log = _log_of(ctx)
    if log is None:
        return
    log.record(result)
    if getattr(ctx, "output_dir", None) is None:
        return
    save_table(
        ctx,
        PROVENANCE_TABLE,
        log.frame(),
        required_columns=PROVENANCE_COLUMNS,
    )
