# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Compare the hyperparameters of former source and borrower models (#169).

These exploratory GB models were retuned after they had used parameters copied
from another model. The test requires each former borrower's parameters to
differ from its former source. Parameter differences alone do not verify the
tuning history.
"""

import pytest

from language_reading_predictors.models.registry import MODELS

# Former (source, [borrowers]) relationships, keyed by canonical CLI model ID.
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
