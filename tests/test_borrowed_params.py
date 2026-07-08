# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Guard that the retired borrowed-parameter groups stay retired (#169).

Several exploratory GB models used to carry hyperparameters copied from a more
developed sibling "pending a target-specific tune". Issue #169 retuned every GB
model on its own full predictor set, so each of these models now has
target-specific parameters and no longer borrows. This test records the former
source -> borrowers relationships and asserts they are genuinely *broken* — i.e.
each former borrower's params now differ from its old source — so a future edit
cannot silently reintroduce borrowing without this guard failing.
"""

import pytest

from language_reading_predictors.models.registry import MODELS

# Former (source, [borrowers]) borrowing relationships, retired by the #169
# target-specific retune. Borrowers must now DIFFER from the source. Canonical
# registry keys since #168 Phase 2 (MODELS is keyed on the CLI id).
FORMER_BORROWED_PARAM_GROUPS = [
    ("lrp-rli-gbg-002", ["lrp-rli-gbg-001", "lrp-rli-gbg-003", "lrp-rli-gbg-004"]),
    ("lrp-rli-gbg-009", ["lrp-rli-gbg-011"]),
    ("lrp-rli-gbl-002", ["lrp-rli-gbl-001", "lrp-rli-gbl-003", "lrp-rli-gbl-004"]),
    ("lrp-rli-gbl-009", ["lrp-rli-gbl-011"]),
]


@pytest.mark.parametrize("source,borrowers", FORMER_BORROWED_PARAM_GROUPS)
def test_former_borrowers_are_target_specific(source, borrowers):
    src = MODELS[source].model_params
    for borrower in borrowers:
        assert MODELS[borrower].model_params != src, (
            f"{borrower} was retuned target-specifically in #169 and should no "
            f"longer share {source}'s hyperparameters, but they are identical. "
            f"If borrowing was deliberately reintroduced, update this guard."
        )
