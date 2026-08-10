# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the shared sub-fit runner and its typed result (#394 design point 5).

The runner's bookkeeping is exercised without sampling anything: a sub-fit costs
minutes, and none of the properties tested here depend on the posterior. The one
test that does sample uses a two-parameter toy model.
"""

from __future__ import annotations

import hashlib
import json
import os
from types import SimpleNamespace

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest

from language_reading_predictors.statistical_models.artifacts import ArtifactLog
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.reporting import (
    _reuse_compatibility_contract,
)
from language_reading_predictors.statistical_models.subfits import (
    PROVENANCE_COLUMNS,
    PROVENANCE_TABLE,
    SubfitData,
    SubfitLog,
    SubfitResult,
    _classify_failure,
    describe_fitted_data,
    record_subfit,
    refresh_subfit_trace_hash,
    run_subfit,
)


def _ctx(tmp_path=None, *, draws=800, tune=500, chains=2):
    # Enough draws that a one-parameter toy model clears the ESS >= 400 gate, so
    # the runner's verdict is a real pass rather than an artefact of short chains.
    return SimpleNamespace(
        output_dir=str(tmp_path) if tmp_path is not None else None,
        tables={},
        artifacts=ArtifactLog(),
        subfits=SubfitLog(),
        sampling=SimpleNamespace(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=1,
            target_accept=0.9,
            random_seed=47,
        ),
        spec=SimpleNamespace(model_id="lrp-rli-test-001"),
    )


def _built(counts=(3, 5, 2, 7), *, n_children=4, subject_ids=None, phase=None):
    """A tiny Binomial model plus the duck-typed ``prepared`` the runner reads."""
    observed = np.asarray(counts)
    with pm.Model() as model:
        p = pm.Beta("p", 2.0, 2.0)
        pm.Binomial("y", n=10, p=p, observed=observed)
    prepared = SimpleNamespace(
        n_children=n_children,
        n_obs=len(observed),
        n_phases=1,
        dropped_rows=0,
        dropped_by_reason={},
        data_sha256="a" * 64,
        subject_ids=np.asarray(
            subject_ids if subject_ids is not None else range(len(observed))
        ),
    )
    if phase is not None:
        prepared.phase = np.asarray(phase)
    return SimpleNamespace(model=model, prepared=prepared)


def _write_reuse_contract(
    ctx,
    built,
    source,
    *,
    label="reused toy sub-fit",
    role="sensitivity",
    trace_filename="trace_toy_subfit.nc",
):
    ctx.model = built.model
    ctx.prepared = built.prepared
    ctx.reporting = SimpleNamespace(config_name="reporting", ci_prob=0.89)
    ctx.resolved_plan = None
    ctx.spec = ModelSpec(
        model_id="lrp-rli-hg-999",
        kind="historical_growth",
        title="sub-fit reuse contract",
    )
    primary_trace = source / "trace.nc"
    primary_trace.write_bytes(b"primary trace")
    config = {
        **_reuse_compatibility_contract(ctx),
        "trace_sha256": hashlib.sha256(primary_trace.read_bytes()).hexdigest(),
    }
    (source / "config.json").write_text(json.dumps(config))

    subfit_trace = source / trace_filename
    data = describe_fitted_data(built)
    sampling = {
        "sampler": "nutpie",
        "draws": int(ctx.sampling.draws),
        "tune": int(ctx.sampling.tune),
        "chains": int(ctx.sampling.chains),
        "cores": int(ctx.sampling.cores),
        "target_accept": float(ctx.sampling.target_accept),
        "random_seed": ctx.sampling.random_seed,
    }
    row = SubfitResult(
        label=label,
        role=role,
        trace=None,
        convergence={},
        sampling=sampling,
        data=data,
        convergence_scope="free_rvs",
        trace_file=trace_filename,
        trace_sha256=hashlib.sha256(subfit_trace.read_bytes()).hexdigest(),
    ).provenance_row()
    pd.DataFrame([row]).to_csv(source / "subfit_provenance.csv", index=False)


def _persisted_toy_trace(path):
    rng = np.random.default_rng(47)
    trace = az.from_dict(
        {
            "posterior": {"p": rng.beta(3.0, 4.0, size=(2, 50))},
            "sample_stats": {
                "diverging": np.zeros((2, 50), dtype=bool),
                "energy": rng.normal(size=(2, 50)),
            },
        },
    )
    trace.to_netcdf(path)


def _verdict(**over):
    verdict = {
        "converged": True,
        "max_rhat": 1.001,
        "min_ess": 900.0,
        "min_bfmi": 0.8,
        "n_divergences": 0,
    }
    verdict.update(over)
    return verdict


def _result(**over):
    fields = {
        "label": "lrp-rli-test-001 bivariate L",
        "role": "bivariate",
        "trace": None,
        "convergence": _verdict(),
        "sampling": {
            "sampler": "nutpie",
            "draws": 40,
            "tune": 40,
            "chains": 2,
            "cores": 1,
            "target_accept": 0.9,
            "random_seed": 47,
        },
        "data": SubfitData(
            n_children=4,
            n_obs=4,
            observed=(("y", (4,)),),
            identity_keys=("subject_ids",),
            digest="abc",
            digest_error=None,
        ),
        "convergence_scope": "free_rvs",
    }
    fields.update(over)
    return SubfitResult(**fields)


# --- fitted-data identity ---------------------------------------------------


def test_the_data_identity_reads_the_rows_and_observations_actually_fitted():
    data = describe_fitted_data(_built((3, 5, 2, 7), n_children=4))
    assert (data.n_children, data.n_obs) == (4, 4)
    assert data.observed == (("y", (4,)),)
    assert data.identity_keys == ("subject_ids",)
    assert data.digest and data.digest_error is None


def test_the_digest_separates_sub_fits_that_differ_only_in_their_rows():
    """The point of the digest: same shape, same model, different children.

    A bivariate refit and an SES complete-case refit can produce identically
    shaped observations from different row sets. Counting rows cannot tell them
    apart; the digest can.
    """
    same = describe_fitted_data(_built((3, 5, 2, 7)))
    shuffled = describe_fitted_data(_built((5, 3, 2, 7)))
    identical = describe_fitted_data(_built((3, 5, 2, 7)))
    assert same.digest == identical.digest
    assert same.digest != shuffled.digest


def test_the_digest_separates_different_children_with_identical_scores():
    """Observations alone are not an identity, and floored outcomes prove it.

    On a heavily floored measure two different subsets of children routinely
    share one ordered score vector — all zeros, say. Hashing the scores alone
    would declare two different analysis populations the same data, so the row
    keys are hashed too.
    """
    floored = (0, 0, 0, 0)
    first = describe_fitted_data(_built(floored, subject_ids=[11, 12, 13, 14]))
    second = describe_fitted_data(_built(floored, subject_ids=[21, 22, 23, 24]))
    assert first.digest != second.digest
    # And the same children at a different phase are a different fit again.
    p1 = describe_fitted_data(_built(floored, subject_ids=[11, 12, 13, 14], phase=[1] * 4))
    p2 = describe_fitted_data(_built(floored, subject_ids=[11, 12, 13, 14], phase=[2] * 4))
    assert p1.digest != p2.digest
    assert p1.identity_keys == ("subject_ids", "phase")


def test_the_digest_covers_shape_so_a_reshape_is_not_the_same_data():
    built = _built((3, 5, 2, 7))
    flat = describe_fitted_data(built)
    with pm.Model() as model:
        p = pm.Beta("p", 2.0, 2.0)
        pm.Binomial("y", n=10, p=p, observed=np.asarray([[3, 5], [2, 7]]))
    reshaped = describe_fitted_data(
        SimpleNamespace(model=model, prepared=built.prepared)
    )
    assert flat.digest != reshaped.digest


def test_string_subject_identifiers_are_hashed_by_their_text():
    """Object/str id arrays have no meaningful buffer to hash."""
    a = describe_fitted_data(_built(subject_ids=["c01", "c02", "c03", "c04"]))
    b = describe_fitted_data(_built(subject_ids=["c01", "c02", "c03", "c99"]))
    assert a.digest and a.digest != b.digest


def test_a_missing_prepared_frame_degrades_to_nulls_rather_than_failing():
    """Provenance is never worth losing a fit over."""
    built = SimpleNamespace(model=_built().model)
    data = describe_fitted_data(built)
    assert (data.n_children, data.n_obs) == (None, None)
    assert data.identity_keys == ()
    assert data.digest is not None


def test_an_unreadable_observation_records_why_the_digest_is_absent():
    class Hostile:
        name = "y"

    built = SimpleNamespace(
        model=SimpleNamespace(observed_RVs=[Hostile()], rvs_to_values={}),
        prepared=SimpleNamespace(
            n_children=4, n_obs=4, subject_ids=np.arange(4)
        ),
    )
    data = describe_fitted_data(built)
    assert data.digest is None
    assert data.digest_error and "KeyError" in data.digest_error
    assert (data.n_children, data.n_obs) == (4, 4)
    # No partial digest, and no identity keys claimed for one: a blank cell with a
    # reason cannot be misread as comparable to a full digest.
    assert data.identity_keys == ()


# --- the typed result ------------------------------------------------------


def test_the_provenance_row_matches_the_declared_schema():
    assert tuple(_result().provenance_row()) == PROVENANCE_COLUMNS


@pytest.mark.parametrize("value", [True, False, None])
def test_converged_passes_the_verdict_through_including_uncheckable(value):
    assert _result(convergence=_verdict(converged=value)).converged is value


def test_an_uncomputable_convergence_check_is_named_not_left_blank():
    """``converged=None`` reaches a CSV as an empty cell, which reads as "not asked".

    The two ways the check can come back unusable are distinguishable from the
    verdict dict alone, so both get a name.
    """
    assert _classify_failure(_verdict()) == (None, None)
    kind, message = _classify_failure(
        {"converged": None, "max_rhat": None, "min_ess": None}
    )
    assert kind == "convergence_unavailable" and message
    kind, message = _classify_failure(
        {"converged": None, "max_rhat": 1.002, "min_ess": 900.0}
    )
    assert kind == "divergences_unavailable" and "diverging" in message


# --- the log and its table -------------------------------------------------


def test_the_log_keeps_sub_fits_in_the_order_the_family_ran_them():
    log = SubfitLog()
    for k in ("L", "B", "age"):
        log.record(_result(label=f"bivariate {k}"))
    assert [r.label for r in log.results] == ["bivariate L", "bivariate B", "bivariate age"]
    assert list(log.frame()["label"]) == ["bivariate L", "bivariate B", "bivariate age"]


def test_the_log_does_not_retain_the_traces():
    """Provenance must not pin every sub-fit posterior in memory.

    The concurrent family runs 27 sub-fits at reporting tier and drops each trace
    once its summary is computed. A log holding the ``InferenceData`` would undo
    that for the lifetime of the fit context; the caller still gets the full
    result back from the runner.
    """
    log = SubfitLog()
    full = _result(trace={"posterior": "a large object"})
    log.record(full)
    assert full.trace is not None
    assert log.results[0].trace is None
    assert log.results[0].label == full.label


def test_recording_writes_the_provenance_table_and_registers_it(tmp_path):
    ctx = _ctx(tmp_path)
    os.makedirs(ctx.output_dir, exist_ok=True)
    record_subfit(ctx, _result(role="prior_sweep", label="sigma=0.5"))
    record_subfit(ctx, _result(role="sensitivity", label="SES complete-case"))

    written = pd.read_csv(os.path.join(ctx.output_dir, f"{PROVENANCE_TABLE}.csv"))
    assert list(written["role"]) == ["prior_sweep", "sensitivity"]
    assert tuple(written.columns) == PROVENANCE_COLUMNS
    assert PROVENANCE_TABLE in ctx.tables
    assert f"{PROVENANCE_TABLE}.csv" in ctx.artifacts.records


def test_recording_without_an_output_directory_still_logs(tmp_path):
    """So the summaries can be tested without a fit directory (design point 6)."""
    ctx = _ctx(None)
    record_subfit(ctx, _result())
    assert len(ctx.subfits.results) == 1
    assert ctx.tables == {}


def test_a_context_with_no_subfit_log_is_tolerated():
    """Sweep and regeneration harnesses build minimal contexts."""
    ctx = SimpleNamespace(output_dir=None)
    record_subfit(ctx, _result())  # must not raise


# --- the runner -----------------------------------------------------------


def test_the_runner_samples_checks_persists_and_records(tmp_path):
    ctx = _ctx(tmp_path)
    os.makedirs(ctx.output_dir, exist_ok=True)
    result = run_subfit(
        ctx,
        _built(),
        label="lrp-rli-test-001 toy sub-fit",
        role="sensitivity",
        trace_filename="trace_toy_subfit.nc",
    )

    assert result.converged is True
    assert result.failure_type is None
    # Sampled at the run's settings, not at some sub-fit-specific default.
    s = ctx.sampling
    assert result.sampling == {
        "sampler": "nutpie",
        "draws": s.draws,
        "tune": s.tune,
        "chains": s.chains,
        "cores": s.cores,
        "target_accept": s.target_accept,
        "random_seed": s.random_seed,
    }
    # The sub-fit trace is persisted under its own name and recorded, so a
    # published secondary estimate stays auditable; ``trace.nc`` is untouched.
    assert result.trace_file == "trace_toy_subfit.nc"
    assert os.path.exists(os.path.join(ctx.output_dir, "trace_toy_subfit.nc"))
    assert not os.path.exists(os.path.join(ctx.output_dir, "trace.nc"))
    assert ctx.artifacts.records["trace_toy_subfit.nc"].kind == "netcdf"
    # And it is in the provenance table with the data it was fitted to.
    row = pd.read_csv(os.path.join(ctx.output_dir, f"{PROVENANCE_TABLE}.csv")).iloc[0]
    assert row["role"] == "sensitivity"
    assert row["n_obs"] == 4
    assert row["data_digest"] == describe_fitted_data(_built()).digest
    # Shapes are published with the node names, and the digest says what it covers.
    assert row["observed_nodes"] == "y[4]"
    assert row["identity_keys"] == "subject_ids"
    # The log holds provenance, not posteriors.
    assert ctx.subfits.results[0].trace is None


def test_the_convergence_scope_selects_the_parameters_scanned(monkeypatch):
    """``free_rvs`` scans the sub-model's free RVs; ``all`` leaves ArviZ unrestricted.

    The mediation temporal-ordering sensitivity has always used the unrestricted
    scan, so the runner must keep both available rather than impose one.
    """
    from language_reading_predictors.statistical_models import diagnostics as _diag

    seen: list[list[str] | None] = []
    real = _diag.subfit_convergence

    def spy(trace, *, label, var_names=None):
        seen.append(var_names)
        return real(trace, label=label, var_names=var_names)

    monkeypatch.setattr(_diag, "subfit_convergence", spy)
    ctx = _ctx(None)
    run_subfit(ctx, _built(), label="scoped", role="wave")
    run_subfit(ctx, _built(), label="unscoped", role="wave", convergence_scope="all")
    assert seen == [["p"], None]


def test_reuse_trace_loads_a_persisted_subfit_without_sampling(tmp_path, monkeypatch):
    source = tmp_path / "published"
    staging = tmp_path / "staging"
    source.mkdir()
    staging.mkdir()
    ctx = _ctx(staging, draws=50, tune=50, chains=2)
    ctx.final_output_dir = str(source)
    built = _built()

    _persisted_toy_trace(source / "trace_toy_subfit.nc")
    _write_reuse_contract(ctx, built, source)

    monkeypatch.setenv("DSE_LRP_REUSE_TRACE", "1")
    monkeypatch.setattr(
        pm,
        "sample",
        lambda **_kwargs: pytest.fail("reuse-trace must not call NUTS"),
    )
    result = run_subfit(
        ctx,
        built,
        label="reused toy sub-fit",
        role="sensitivity",
        trace_filename="trace_toy_subfit.nc",
    )

    assert result.trace.posterior.sizes["draw"] == 50
    assert (staging / "trace_toy_subfit.nc").is_file()


def test_reuse_trace_rejects_subfit_provenance_drift_before_sampling(
    tmp_path, monkeypatch
):
    source = tmp_path / "published"
    staging = tmp_path / "staging"
    source.mkdir()
    staging.mkdir()
    ctx = _ctx(staging, draws=50, tune=50, chains=2)
    ctx.final_output_dir = str(source)
    built = _built()

    _persisted_toy_trace(source / "trace_toy_subfit.nc")
    _write_reuse_contract(ctx, built, source)
    provenance_path = source / "subfit_provenance.csv"
    provenance = pd.read_csv(provenance_path)
    provenance.loc[0, "data_digest"] = "not-the-current-fitted-data"
    provenance.to_csv(provenance_path, index=False)

    monkeypatch.setenv("DSE_LRP_REUSE_TRACE", "1")
    monkeypatch.setattr(
        pm,
        "sample",
        lambda **_kwargs: pytest.fail("an incompatible reuse must not call NUTS"),
    )
    with pytest.raises(ValueError, match="data_digest"):
        run_subfit(
            ctx,
            built,
            label="reused toy sub-fit",
            role="sensitivity",
            trace_filename="trace_toy_subfit.nc",
        )


def test_refresh_subfit_trace_hash_rebinds_the_final_augmented_bytes(tmp_path):
    ctx = _ctx(tmp_path)
    os.makedirs(ctx.output_dir, exist_ok=True)
    trace_path = tmp_path / "trace_toy_subfit.nc"
    trace_path.write_bytes(b"initial bytes")
    record_subfit(
        ctx,
        _result(
            label="augmented sub-fit",
            role="sensitivity",
            trace_file=trace_path.name,
            trace_sha256=hashlib.sha256(b"initial bytes").hexdigest(),
        ),
    )

    trace_path.write_bytes(b"final bytes with attached prior groups")
    final_digest = refresh_subfit_trace_hash(
        ctx,
        label="augmented sub-fit",
        trace_filename=trace_path.name,
    )

    expected = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    assert final_digest == expected
    assert ctx.subfits.results[0].trace_sha256 == expected
    row = pd.read_csv(tmp_path / "subfit_provenance.csv").iloc[0]
    assert row["trace_sha256"] == expected


def test_reuse_trace_fails_closed_when_persisted_subfit_is_absent(
    tmp_path, monkeypatch
):
    source = tmp_path / "published"
    staging = tmp_path / "staging"
    source.mkdir()
    staging.mkdir()
    ctx = _ctx(staging)
    ctx.final_output_dir = str(source)
    monkeypatch.setenv("DSE_LRP_REUSE_TRACE", "1")
    monkeypatch.setattr(
        pm,
        "sample",
        lambda **_kwargs: pytest.fail("reuse-trace must not call NUTS"),
    )

    with pytest.raises(FileNotFoundError, match="persisted sub-fit trace"):
        run_subfit(
            ctx,
            _built(),
            label="missing reused toy sub-fit",
            role="sensitivity",
            trace_filename="trace_toy_subfit.nc",
        )


def test_reuse_trace_fails_closed_for_an_unnamed_subfit(monkeypatch):
    ctx = _ctx(None)
    monkeypatch.setenv("DSE_LRP_REUSE_TRACE", "1")
    monkeypatch.setattr(
        pm,
        "sample",
        lambda **_kwargs: pytest.fail("reuse-trace must not sample an unnamed sub-fit"),
    )

    with pytest.raises(FileNotFoundError, match="cannot reuse an unnamed sub-fit"):
        run_subfit(
            ctx,
            _built(),
            label="unnamed toy sub-fit",
            role="sensitivity",
        )
