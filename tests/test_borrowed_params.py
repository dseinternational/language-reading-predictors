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
BORROWED_PARAM_GROUPS = [
    ("lrpgbg02", ["lrpgbg01", "lrpgbg03", "lrpgbg04"]),
    ("lrpgbg09", ["lrpgbg11"]),
    ("lrpgbl02", ["lrpgbl01", "lrpgbl03", "lrpgbl04"]),
    ("lrpgbl09", ["lrpgbl11"]),
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
