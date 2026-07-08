# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Guard against silent drift in borrowed GB hyperparameters.

Several exploratory GB models carry hyperparameters copied from a more
developed sibling "pending a target-specific tune". Those copies are easy to
desync: retuning the source (or a borrower) without updating the other leaves a
stale, unflagged duplicate. Each entry below records a *currently true*
source -> borrowers relationship; if it stops holding, this test fails and forces
the borrowing annotation to be reconciled (re-sync, or update the note).
"""

import pytest

from language_reading_predictors.models.registry import MODELS

# (source_model_id, [borrower_model_ids]) — borrowers must equal the source.
# Canonical registry keys since #168 Phase 2 (MODELS is keyed on the CLI id).
BORROWED_PARAM_GROUPS = [
    ("lrp-rli-gbg-002", ["lrp-rli-gbg-001", "lrp-rli-gbg-003", "lrp-rli-gbg-004"]),
    ("lrp-rli-gbg-009", ["lrp-rli-gbg-011"]),
    ("lrp-rli-gbl-002", ["lrp-rli-gbl-001", "lrp-rli-gbl-003", "lrp-rli-gbl-004"]),
    ("lrp-rli-gbl-009", ["lrp-rli-gbl-011"]),
]


@pytest.mark.parametrize("source,borrowers", BORROWED_PARAM_GROUPS)
def test_borrowed_params_match_source(source, borrowers):
    src = MODELS[source].model_params
    for borrower in borrowers:
        assert MODELS[borrower].model_params == src, (
            f"{borrower} is documented as borrowing hyperparameters from {source} "
            f"but they now differ. Re-sync {borrower} to {source}, or update the "
            f"borrowing annotation in {borrower}.py."
        )
