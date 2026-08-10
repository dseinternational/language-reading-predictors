# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Privacy-preserving primary-fit subject identity metadata."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pymc as pm
import pytest

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.reporting import (
    _reuse_compatibility_contract,
    fitted_subject_identity,
    require_reuse_compatibility,
    write_run_metadata,
)


def _expected_digest(subject_ids: list[str]) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"dse-lrp-fitted-subject-identity-v1\0")
    for subject_id in subject_ids:
        encoded = subject_id.encode("utf-8")
        hasher.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        hasher.update(encoded)
    return hasher.hexdigest()


def _context(tmp_path, model_id: str, subject_ids: list[str]):
    output_dir = tmp_path / model_id
    return SimpleNamespace(
        spec=ModelSpec(model_id=model_id, kind="mediation", title="identity audit"),
        prepared=SimpleNamespace(
            subject_ids=np.asarray(subject_ids, dtype=object),
            n_obs=len(subject_ids),
            n_children=len(set(subject_ids)),
            n_phases=1,
            dropped_rows=0,
        ),
        reporting=SimpleNamespace(
            output_dir=str(output_dir), ci_prob=0.89, config_name="reporting"
        ),
        sampling=SimpleNamespace(
            draws=1, tune=1, chains=1, target_accept=0.9, random_seed=47
        ),
        output_dir=str(output_dir),
    )


def _reuse_context(tmp_path):
    output_dir = tmp_path / "current"
    output_dir.mkdir(parents=True)
    observed = np.asarray([2, 4, 3])
    with pm.Model() as model:
        p = pm.Beta("p", 2.0, 2.0)
        pm.Binomial("y", n=5, p=p, observed=observed)
    prepared = SimpleNamespace(
        subject_ids=np.asarray(["S1", "S2", "S3"], dtype=object),
        n_obs=3,
        n_children=3,
        n_phases=1,
        n_waves=None,
        dropped_rows=0,
        dropped_by_reason={},
        data_sha256="a" * 64,
    )
    return SimpleNamespace(
        spec=ModelSpec(
            model_id="lrp-rli-hg-999",
            kind="historical_growth",
            title="reuse contract",
        ),
        prepared=prepared,
        model=model,
        reporting=SimpleNamespace(config_name="reporting", ci_prob=0.89),
        sampling=SimpleNamespace(
            draws=6000,
            tune=6000,
            chains=6,
            cores=5,
            target_accept=0.95,
            random_seed=47,
        ),
        resolved_plan=None,
        output_dir=str(output_dir),
    )


def _compatible_publication(tmp_path, context):
    source = tmp_path / "published"
    source.mkdir()
    trace = source / "trace.nc"
    trace.write_bytes(b"persisted trace")
    config = {
        **_reuse_compatibility_contract(context),
        "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
    }
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return source


def test_fitted_subject_identity_has_pinned_ordered_encoding_without_raw_ids():
    subject_ids = ["child-02", "child-β", "child-02"]
    identity = fitted_subject_identity(
        SimpleNamespace(subject_ids=np.asarray(subject_ids, dtype=object))
    )

    assert identity == {
        "algorithm": "sha256",
        "domain_separator": "dse-lrp-fitted-subject-identity-v1",
        "encoding": "str(value) UTF-8 with uint64 big-endian byte-length prefix",
        "order": "prepared.subject_ids fitted-row order; unsorted; duplicates retained",
        "n_rows": 3,
        "n_unique_subjects": 2,
        "sha256": _expected_digest(subject_ids),
    }
    serialised = json.dumps(identity)
    assert all(subject_id not in serialised for subject_id in set(subject_ids))


def test_fitted_subject_identity_is_order_and_multiplicity_sensitive():
    forward = fitted_subject_identity(SimpleNamespace(subject_ids=["a", "b", "a"]))
    reordered = fitted_subject_identity(SimpleNamespace(subject_ids=["a", "a", "b"]))
    deduplicated = fitted_subject_identity(SimpleNamespace(subject_ids=["a", "b"]))

    assert forward["sha256"] != reordered["sha256"]
    assert forward["sha256"] != deduplicated["sha256"]


@pytest.mark.parametrize(
    "parent_id, companion_id",
    [
        ("lrp-rli-med-086", "lrp-rli-med-186"),
        ("lrp-rli-med-087", "lrp-rli-med-187"),
    ],
)
def test_config_can_audit_mediation_companion_primary_row_identity(
    tmp_path, parent_id, companion_id
):
    fitted_rows = ["S003", "S001", "S004"]
    configs = []
    for model_id in (parent_id, companion_id):
        context = _context(tmp_path, model_id, fitted_rows)
        write_run_metadata(context)
        config = json.loads(
            (tmp_path / model_id / "config.json").read_text(encoding="utf-8")
        )
        configs.append(config)

    parent, companion = configs
    assert parent["model_id"] == parent_id
    assert companion["model_id"] == companion_id
    assert parent["config_name"] == companion["config_name"] == "reporting"
    assert parent["fitted_subject_identity"] == companion["fitted_subject_identity"]
    assert parent["fitted_subject_identity"]["n_rows"] == 3
    assert all(subject_id not in json.dumps(parent) for subject_id in fitted_rows)


def test_fitted_subject_identity_is_absent_without_subject_ids():
    assert fitted_subject_identity(SimpleNamespace()) is None


def test_fitted_subject_identity_rejects_ambiguous_multidimensional_rows():
    with pytest.raises(ValueError, match="one-dimensional"):
        fitted_subject_identity(SimpleNamespace(subject_ids=[["a"], ["b"]]))


def test_reuse_contract_accepts_the_same_config_data_and_tier(tmp_path):
    context = _reuse_context(tmp_path)
    source = _compatible_publication(tmp_path, context)

    require_reuse_compatibility(context, source)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("config_name", "dev"),
        ("sampling", {"draws": 10}),
        ("data_sha256", "b" * 64),
    ],
)
def test_reuse_contract_rejects_prior_config_data_or_tier_drift(
    tmp_path, field, replacement
):
    context = _reuse_context(tmp_path)
    source = _compatible_publication(tmp_path, context)
    config_path = source / "config.json"
    config = json.loads(config_path.read_text())
    config[field] = replacement
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match=field):
        require_reuse_compatibility(context, source)


def test_reuse_contract_rejects_recipe_or_trace_mutation(tmp_path):
    context = _reuse_context(tmp_path)
    current_recipe = tmp_path / "current" / "model_recipe.md"
    current_recipe.write_text("registered recipe\n")
    source = _compatible_publication(tmp_path, context)
    prior_recipe = source / "model_recipe.md"
    prior_recipe.write_text("registered recipe\n")

    require_reuse_compatibility(context, source)

    prior_recipe.write_text("different recipe\n")
    with pytest.raises(ValueError, match="model_recipe_sha256"):
        require_reuse_compatibility(context, source)

    prior_recipe.write_text("registered recipe\n")
    (source / "trace.nc").write_bytes(b"mutated trace")
    with pytest.raises(ValueError, match="trace_sha256"):
        require_reuse_compatibility(context, source)
