# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Privacy-preserving primary-fit subject identity metadata."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.reporting import (
    fitted_subject_identity,
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
        reporting=SimpleNamespace(output_dir=str(output_dir), ci_prob=0.89),
        sampling=SimpleNamespace(
            draws=1, tune=1, chains=1, target_accept=0.9, random_seed=47
        ),
        output_dir=str(output_dir),
    )


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
    assert parent["fitted_subject_identity"] == companion["fitted_subject_identity"]
    assert parent["fitted_subject_identity"]["n_rows"] == 3
    assert all(subject_id not in json.dumps(parent) for subject_id in fitted_rows)


def test_fitted_subject_identity_is_absent_without_subject_ids():
    assert fitted_subject_identity(SimpleNamespace()) is None


def test_fitted_subject_identity_rejects_ambiguous_multidimensional_rows():
    with pytest.raises(ValueError, match="one-dimensional"):
        fitted_subject_identity(SimpleNamespace(subject_ids=[["a"], ["b"]]))
