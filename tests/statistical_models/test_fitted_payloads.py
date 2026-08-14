# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Boundary tests for typed factory-to-consumer fitted payloads."""

from types import SimpleNamespace

import pytest

from language_reading_predictors.statistical_models.factories import BuiltModel
from language_reading_predictors.statistical_models.fitted_payloads import (
    EmptyPayload,
    IttPayload,
)


def test_built_model_has_no_untyped_extras_escape_hatch():
    built = BuiltModel(
        model=SimpleNamespace(),
        prepared=SimpleNamespace(),
        payload=EmptyPayload(),
    )

    assert not hasattr(built, "extras")


def test_require_payload_rejects_a_mismatched_family_payload():
    built = BuiltModel(
        model=SimpleNamespace(),
        prepared=SimpleNamespace(),
        payload=EmptyPayload(),
    )

    with pytest.raises(
        TypeError,
        match="itt requires IttPayload.*carries EmptyPayload",
    ):
        built.require_payload(IttPayload, family="itt")
