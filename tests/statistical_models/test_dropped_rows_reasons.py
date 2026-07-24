# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Reason-attributed row exclusions (#390 P3).

``PreparedData.dropped_rows`` used to be one opaque count mixing load-time and
factory-stage exclusions. These tests pin the reason breakdown: every drop is
attributed to a labelled reason, the reasons are mutually exclusive, and their
counts always reconcile to ``dropped_rows``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from language_reading_predictors.statistical_models import factories as F
from language_reading_predictors.statistical_models.preprocessing import (
    load_and_prepare,
)

from .test_factories import _write_synthetic


def test_subset_records_reason_and_reconciles(tmp_path):
    prep = load_and_prepare(path=_write_synthetic(tmp_path, n_children=15), phase_mode="all")
    keep = np.ones(prep.n_obs, dtype=bool)
    keep[:3] = False
    sub = F._subset(prep, keep)
    assert sub.dropped_by_reason.get("factory_stage") == 3
    assert sub.dropped_rows == prep.dropped_rows + 3
    assert sum(sub.dropped_by_reason.values()) == sub.dropped_rows

    # A second exclusion for a different reason accumulates a distinct key; the
    # two are mutually exclusive and still reconcile.
    keep2 = np.ones(sub.n_obs, dtype=bool)
    keep2[:2] = False
    sub2 = F._subset(sub, keep2, reason="design_excluded")
    assert sub2.dropped_by_reason["factory_stage"] == 3
    assert sub2.dropped_by_reason["design_excluded"] == 2
    assert sum(sub2.dropped_by_reason.values()) == sub2.dropped_rows


def test_subset_all_keep_is_a_noop(tmp_path):
    prep = load_and_prepare(path=_write_synthetic(tmp_path, n_children=15), phase_mode="all")
    assert F._subset(prep, np.ones(prep.n_obs, dtype=bool)) is prep


def test_loader_attributes_its_own_drops(tmp_path):
    path = _write_synthetic(tmp_path, n_children=15)
    df = pd.read_csv(path)
    victim = df["subject_id"].iloc[0]
    # A child with no group assignment cannot enter the randomised analysis and is
    # dropped at load time — that drop must be attributed to the loader.
    df.loc[df["subject_id"] == victim, "group"] = np.nan
    df.to_csv(path, index=False)

    prep = load_and_prepare(path=path, phase_mode="all")
    assert prep.dropped_rows > 0
    assert prep.dropped_by_reason.get("loader") == prep.dropped_rows
    assert sum(prep.dropped_by_reason.values()) == prep.dropped_rows


def test_reasons_reconcile_after_a_factory_fit(tmp_path):
    prep = load_and_prepare(path=_write_synthetic(tmp_path, n_children=15), phase_mode="all")
    built = F.build_mechanism_model(
        prep, mechanism_symbol="R", outcome_symbol="W", confounder_symbols=()
    )
    fitted = built.prepared
    # The invariant holds through the whole prepare -> factory-subset pipeline.
    assert sum(fitted.dropped_by_reason.values()) == fitted.dropped_rows
